"""Opt-in, bounded latency-phase diagnostics for one conversational turn.

Built for Gate 10O (cold-start latency profiling). Measures wall-clock time
spent between natural, already-existing phase boundaries in the gateway
message-handling pipeline (see ``gateway/run.py:_handle_message_with_agent``,
``agent/stream_diag.py:stream_diag_init`` and
``agent/chat_completion_helpers.py``'s existing ``_diag["first_chunk_at"]``
hook) so a cold vs. warm turn can be told apart without guessing.

Design constraints (see Gate 10O spec):
  - Disabled by default. Every public function is a near-zero-cost no-op
    unless ``HERMES_LATENCY_DIAG`` (or ``diagnostics.latency_enabled`` in
    config.yaml) is truthy — normal operation must not be spammed.
  - Never logs message/prompt content, secrets, API keys, or raw
    Telegram/session identifiers — only counts, lengths, and monotonic
    timings, plus a one-way hash of the session identity.
  - A recorder lives on a ``contextvars.ContextVar`` so deeply nested call
    sites (a different module, a different file) can call ``mark_phase()``
    without any function signature change and without ever raising —
    a broken/missing mark degrades to "phase not recorded", never an
    exception that could affect the real request path.
  - Output is one bounded, rotating JSON-lines file — never unbounded
    growth, never the main application log (which is why this does not
    spam ``agent.log``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "PHASE_ORDER",
    "LatencyRecorder",
    "is_enabled",
    "start_recording",
    "mark_phase",
    "finish_and_emit",
    "hash_identity",
    "current_recorder",
]

# Ordered so a rendered record's phase deltas read top-to-bottom in the same
# order the turn actually happened. Field names match the Gate 10O spec.
PHASE_ORDER = (
    "receive",
    "agent_ready",
    "memory_start",
    "memory_end",
    "prompt_ready",
    "llm_start",
    "first_token",
    "generation_complete",
    "complete",
    "sent",
)

_DIAG_LOGGER_NAME = "hermes.latency_diagnostics"
_MAX_BYTES = 2 * 1024 * 1024  # 2 MiB per file
_BACKUP_COUNT = 3  # → 8 MiB worst case, bounded and small (JSON lines only)

_active: "ContextVar[Optional[LatencyRecorder]]" = ContextVar(
    "hermes_latency_recorder", default=None
)

_logger_lock = threading.Lock()
_diag_logger: Optional[logging.Logger] = None


def is_enabled() -> bool:
    """True iff latency diagnostics should record anything at all.

    Cheap (one env lookup) so callers on the hot path can early-out before
    doing any work. Config-file opt-in is also honored so this can be
    toggled without an environment variable, but the env var always wins
    for a quick one-off diagnostic session.
    """
    raw = os.environ.get("HERMES_LATENCY_DIAG")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        diagnostics = cfg.get("diagnostics") or {}
        return bool(diagnostics.get("latency_enabled", False))
    except Exception:
        return False


def hash_identity(raw: Any) -> str:
    """One-way, non-reversible identity tag safe to persist in a log file.

    Never store the raw session/user/chat id — this truncated sha256
    digest is enough to correlate rows for the same conversation across a
    diagnostics session without being reversible to a real Telegram id.
    """
    text = "" if raw is None else str(raw)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


@dataclass
class LatencyRecorder:
    """One turn's phase timestamps (monotonic seconds) plus metadata.

    ``marks`` uses ``time.monotonic()`` exclusively for duration math —
    wall-clock (``time.time()``) is never used for the deltas, only for the
    single ``timestamp`` field, so NTP adjustments/DST can never produce a
    negative duration.
    """

    session_hash: str
    platform: str = ""
    model_endpoint: str = ""
    started_wall: float = field(default_factory=time.time)
    marks: dict = field(default_factory=dict)
    memory_ran: Optional[bool] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cold_label: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def mark(self, phase: str) -> None:
        """Record ``phase`` at "now" — idempotent, first call wins.

        A retried stream attempt re-entering the same phase must not
        overwrite the original (earlier, more meaningful) timestamp.
        """
        if phase not in PHASE_ORDER:
            return
        with self._lock:
            if phase not in self.marks:
                self.marks[phase] = time.monotonic()

    def has(self, phase: str) -> bool:
        return phase in self.marks

    def _delta_ms(self, start_phase: str, end_phase: str) -> Optional[int]:
        start = self.marks.get(start_phase)
        end = self.marks.get(end_phase)
        if start is None or end is None:
            return None
        delta = end - start
        if delta < 0:
            # Monotonic clock guarantees this cannot happen for correctly
            # ordered marks; treat it as "not recorded" rather than lie.
            return None
        return int(round(delta * 1000))

    def to_record(self) -> dict:
        """Render the bounded, content-free structured timing record."""
        phase_pairs = [
            ("receive_to_agent_ready_ms", "receive", "agent_ready"),
            ("agent_ready_to_memory_start_ms", "agent_ready", "memory_start"),
            ("memory_start_to_memory_end_ms", "memory_start", "memory_end"),
            ("agent_ready_to_prompt_ready_ms", "agent_ready", "prompt_ready"),
            ("prompt_ready_to_llm_start_ms", "prompt_ready", "llm_start"),
            ("agent_ready_to_llm_start_ms", "agent_ready", "llm_start"),
            ("llm_start_to_first_token_ms", "llm_start", "first_token"),
            ("first_token_to_complete_ms", "first_token", "complete"),
            ("first_token_to_generation_complete_ms", "first_token", "generation_complete"),
            ("complete_to_sent_ms", "complete", "sent"),
        ]
        record: dict[str, Any] = {
            "timestamp": self.started_wall,
            "session_hash": self.session_hash,
            "platform": self.platform,
            "model_endpoint": self.model_endpoint,
            "memory_ran": self.memory_ran,
            "cold_label": self.cold_label,
        }
        if self.prompt_tokens is not None:
            record["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens is not None:
            record["completion_tokens"] = self.completion_tokens
        for out_name, start_phase, end_phase in phase_pairs:
            val = self._delta_ms(start_phase, end_phase)
            if val is not None:
                record[out_name] = val
        total = self._delta_ms("receive", "sent") or self._delta_ms("receive", "complete")
        if total is not None:
            record["total_ms"] = total
        record["phases_recorded"] = sorted(self.marks.keys(), key=PHASE_ORDER.index)
        return record


def current_recorder() -> Optional[LatencyRecorder]:
    return _active.get()


def start_recording(
    *, session_id: Any = None, platform: str = "", model_endpoint: str = ""
) -> Optional[LatencyRecorder]:
    """Begin recording for the current async context. No-op if disabled.

    Returns the recorder (already installed as the active one for this
    context) or ``None`` when diagnostics are disabled — callers never need
    to branch on the return value before calling ``mark_phase``/
    ``finish_and_emit``, which are themselves safe no-ops without an active
    recorder.
    """
    try:
        if not is_enabled():
            return None
        recorder = LatencyRecorder(
            session_hash=hash_identity(session_id),
            platform=str(platform or ""),
            model_endpoint=str(model_endpoint or ""),
        )
        recorder.mark("receive")
        _active.set(recorder)
        return recorder
    except Exception:
        return None


def mark_phase(phase: str) -> None:
    """Best-effort phase mark on the active recorder, if any. Never raises."""
    try:
        recorder = _active.get()
        if recorder is not None:
            recorder.mark(phase)
    except Exception:
        pass


def _get_diag_logger() -> logging.Logger:
    global _diag_logger
    if _diag_logger is not None:
        return _diag_logger
    with _logger_lock:
        if _diag_logger is not None:
            return _diag_logger
        logger = logging.getLogger(_DIAG_LOGGER_NAME)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # never bleed into agent.log / stderr
        # Use our own installed-marker rather than "if not logger.handlers"
        # — a shared-by-name stdlib logger can already carry handlers that
        # have nothing to do with us (e.g. a test harness's log-capture
        # handler attached to every named logger), and checking bare
        # truthiness would then wrongly skip installing our own handler.
        if not getattr(logger, "_hermes_latency_handler_installed", False):
            try:
                from hermes_constants import get_hermes_home

                log_dir = Path(get_hermes_home()) / "logs"
            except Exception:
                log_dir = Path.home() / ".hermes" / "logs"
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(
                    str(log_dir / "latency_diagnostics.jsonl"),
                    maxBytes=_MAX_BYTES,
                    backupCount=_BACKUP_COUNT,
                    encoding="utf-8",
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                logger.addHandler(handler)
            except Exception:
                # Diagnostics must never block or crash the real request
                # path — a NullHandler means finish_and_emit() silently
                # produces no output instead of raising.
                logger.addHandler(logging.NullHandler())
            logger._hermes_latency_handler_installed = True
        _diag_logger = logger
        return logger


def finish_and_emit(
    recorder: Optional[LatencyRecorder] = None,
    *,
    cold_label: Optional[str] = None,
    memory_ran: Optional[bool] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> Optional[dict]:
    """Finalize the active (or given) recorder and emit one bounded JSON line.

    Always clears the active context var so a leaked recorder from one turn
    can never bleed phase marks into the next. Returns the emitted record
    (useful for tests) or ``None`` if diagnostics are disabled/unavailable.
    """
    try:
        target = recorder if recorder is not None else _active.get()
        if target is None:
            return None
        if not target.has("complete"):
            target.mark("complete")
        if cold_label is not None:
            target.cold_label = cold_label
        if memory_ran is not None:
            target.memory_ran = memory_ran
        if prompt_tokens is not None:
            target.prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            target.completion_tokens = completion_tokens
        record = target.to_record()
        try:
            _get_diag_logger().info(json.dumps(record, sort_keys=True))
        except Exception:
            pass
        return record
    except Exception:
        return None
    finally:
        try:
            if _active.get() is not None:
                _active.set(None)
        except Exception:
            pass
