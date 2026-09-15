"""One-shot, best-effort local-backend KV-cache priming with the real Hermes
static system-prompt prefix (Gate 10P).

Gate 10O (``agent/llm_backend_warmup.py``) established that this deployment's
dominant first-response latency source is llama.cpp prompt prefill after a
KV-cache miss, and that the generic warmup there deliberately does NOT
reconstruct Hermes's real system prompt — building one safely required a
dedicated extraction, which is what :mod:`agent.static_system_prompt` now
provides. This module is the natural follow-up: prime the backend with that
REAL static prefix (not a placeholder prompt) so the first genuine user
message after gateway startup can reuse it from cache.

Safety boundary (mirrors ``llm_backend_warmup.py``; see tests in
``tests/agent/test_llm_static_prompt_priming.py``):
  - Only ever targets a loopback base_url (127.0.0.1 / localhost / ::1),
    and only a ``custom``/``ollama``/unset provider profile — same
    :func:`agent.llm_backend_warmup.resolve_local_backend` gate.
  - Fire-and-forget: runs on a background daemon thread, never blocks or
    fails gateway startup.
  - Exactly one priming attempt per process lifetime (module-level guard).
  - Waits for ``/health`` with a bounded retry budget before priming; never
    polls unboundedly.
  - Builds the prompt via ``agent.static_system_prompt.build_static_system_prompt``
    — pure config/file reads only: no AIAgent, no conversation, no session
    row, no memory write, no tool call, no Telegram/Discord/etc. send.
  - The completion output is discarded; only token counts and timings are
    logged (content-free) via the existing ``agent.latency_diagnostics``-style
    debug/info channel — never the prompt text, never an API key.
  - Bounded retry on the priming POST itself (default 3 attempts, short fixed
    backoff) — never an unbounded/tight loop.

Cache-preservation caveat: this backend runs with a single KV-cache slot
(``--parallel 1``). Any other request to the same endpoint between this
priming call and the first real user message can evict the primed prefix.
Priming improves the common case (nothing else fires in that narrow window)
but does not guarantee the first real request is a cache hit — see the
Gate 10P report for measured eviction risk and the recommended follow-up
(a separate, measured evaluation of ``--parallel 2`` / two KV slots).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from agent.llm_backend_warmup import resolve_local_backend

logger = logging.getLogger(__name__)

__all__ = [
    "PrimingResult",
    "prime_static_system_prompt",
    "schedule_startup_priming",
    "reset_priming_state_for_tests",
]

_PRIMING_USER_MESSAGE = "Reply with only the word READY."
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BACKOFF_SECONDS = (0.5, 1.0, 2.0)
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
_DEFAULT_HEALTH_TIMEOUT_SECONDS = 30.0
_DEFAULT_HEALTH_POLL_INTERVAL_SECONDS = 0.5

# Process-lifetime guard: priming must fire at most once per gateway run.
_priming_already_scheduled = False


@dataclass(frozen=True)
class PrimingResult:
    attempted: bool
    succeeded: bool
    attempts: int
    elapsed_ms: Optional[int]
    prompt_chars: Optional[int]
    prompt_tokens: Optional[int]
    cached_tokens: Optional[int]
    reason: str


def reset_priming_state_for_tests() -> None:
    """Test-only: clear the process-lifetime guard between test cases."""
    global _priming_already_scheduled
    _priming_already_scheduled = False


def _health_url(base_url: str) -> str:
    """Derive the llama.cpp ``/health`` URL from a ``.../v1``-style base_url."""
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return f"{trimmed}/health"


def _wait_for_health(
    base_url: str, *, timeout_seconds: float, poll_interval_seconds: float
) -> bool:
    """Bounded poll of ``/health``; True once it responds 200, False on timeout.

    Never raises. A backend that never becomes healthy within the budget is
    treated as "skip priming this run" — never an unbounded wait.
    """
    url = _health_url(base_url)
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        try:
            with urllib.request.urlopen(url, timeout=poll_interval_seconds) as response:
                if 200 <= response.status < 300:
                    return True
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_interval_seconds)


def _post_priming_completion(
    base_url: str, api_key: str, model: str, system_prompt: str, timeout: float
) -> dict:
    url = f"{base_url}/chat/completions"
    payload = json.dumps(
        {
            "model": model or "default",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _PRIMING_USER_MESSAGE},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'none'}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def prime_static_system_prompt(
    *,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: tuple = _DEFAULT_BACKOFF_SECONDS,
    request_timeout_seconds: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS,
    health_timeout_seconds: float = _DEFAULT_HEALTH_TIMEOUT_SECONDS,
    health_poll_interval_seconds: float = _DEFAULT_HEALTH_POLL_INTERVAL_SECONDS,
    platform: str = "telegram",
) -> PrimingResult:
    """Synchronous priming body — safe to call directly from tests.

    Same local-backend gate as :func:`agent.llm_backend_warmup.resolve_local_backend`
    (loopback + custom/ollama provider only). Waits for ``/health`` (bounded),
    builds the real static prompt, then POSTs it with a 1-token completion,
    discarding the output. Never raises — every failure mode (unreachable
    model, timeout, malformed response, prompt-build failure) is swallowed
    and reflected in the returned :class:`PrimingResult`.
    """
    backend = resolve_local_backend()
    if backend is None:
        return PrimingResult(
            attempted=False, succeeded=False, attempts=0, elapsed_ms=None,
            prompt_chars=None, prompt_tokens=None, cached_tokens=None,
            reason="no_local_backend_configured",
        )

    started = time.monotonic()

    if not _wait_for_health(
        backend["base_url"],
        timeout_seconds=health_timeout_seconds,
        poll_interval_seconds=health_poll_interval_seconds,
    ):
        elapsed_ms = int(round((time.monotonic() - started) * 1000))
        logger.info("static-prompt priming skipped: backend not healthy within %.1fs", health_timeout_seconds)
        return PrimingResult(
            attempted=True, succeeded=False, attempts=0, elapsed_ms=elapsed_ms,
            prompt_chars=None, prompt_tokens=None, cached_tokens=None,
            reason="backend_unhealthy",
        )

    try:
        from agent.static_system_prompt import build_static_system_prompt

        static_prompt = build_static_system_prompt(
            model=backend["model"] or None, platform=platform,
        )
    except Exception:
        elapsed_ms = int(round((time.monotonic() - started) * 1000))
        logger.debug("static-prompt priming: prompt build failed", exc_info=True)
        return PrimingResult(
            attempted=True, succeeded=False, attempts=0, elapsed_ms=elapsed_ms,
            prompt_chars=None, prompt_tokens=None, cached_tokens=None,
            reason="prompt_build_failed",
        )

    if not static_prompt:
        elapsed_ms = int(round((time.monotonic() - started) * 1000))
        return PrimingResult(
            attempted=True, succeeded=False, attempts=0, elapsed_ms=elapsed_ms,
            prompt_chars=0, prompt_tokens=None, cached_tokens=None,
            reason="empty_static_prompt",
        )

    last_error = "unknown"
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            body = _post_priming_completion(
                backend["base_url"], backend["api_key"], backend["model"],
                static_prompt, request_timeout_seconds,
            )
            elapsed_ms = int(round((time.monotonic() - started) * 1000))
            timings = body.get("timings") if isinstance(body, dict) else None
            usage = body.get("usage") if isinstance(body, dict) else None
            prompt_tokens = None
            cached_tokens = None
            if isinstance(timings, dict):
                prompt_tokens = timings.get("prompt_n")
                cached_tokens = timings.get("cache_n")
            elif isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
                details = usage.get("prompt_tokens_details")
                if isinstance(details, dict):
                    cached_tokens = details.get("cached_tokens")
            logger.info(
                "static-prompt priming succeeded: attempt=%d elapsed_ms=%d "
                "prompt_chars=%d prompt_tokens=%s cached_tokens=%s endpoint_host=%s",
                attempt, elapsed_ms, len(static_prompt), prompt_tokens, cached_tokens,
                urlparse(backend["base_url"]).hostname,
            )
            return PrimingResult(
                attempted=True, succeeded=True, attempts=attempt, elapsed_ms=elapsed_ms,
                prompt_chars=len(static_prompt), prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens, reason="ok",
            )
        except Exception as exc:  # noqa: BLE001 - priming must never propagate
            last_error = type(exc).__name__
            if attempt < max_attempts:
                delay = backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)]
                time.sleep(delay)

    elapsed_ms = int(round((time.monotonic() - started) * 1000))
    logger.info(
        "static-prompt priming gave up: attempts=%d elapsed_ms=%d last_error=%s",
        max_attempts, elapsed_ms, last_error,
    )
    return PrimingResult(
        attempted=True, succeeded=False, attempts=max_attempts, elapsed_ms=elapsed_ms,
        prompt_chars=len(static_prompt), prompt_tokens=None, cached_tokens=None,
        reason=last_error,
    )


def _run_priming_in_background() -> None:
    try:
        prime_static_system_prompt()
    except Exception:
        logger.debug("static-prompt priming: background task failed", exc_info=True)


def schedule_startup_priming() -> bool:
    """Fire the priming once, off the startup critical path.

    Same daemon-thread rationale as
    :func:`agent.llm_backend_warmup.schedule_startup_warmup` — a plain
    thread has no dependency on whatever asyncio loop is current when
    ``start_gateway()`` calls this, and ``prime_static_system_prompt`` is
    already a self-contained synchronous call.

    Idempotent per process. Returns True iff a priming thread was actually
    started this call, never raises.
    """
    global _priming_already_scheduled
    try:
        if _priming_already_scheduled:
            return False
        _priming_already_scheduled = True
        thread = threading.Thread(
            target=_run_priming_in_background,
            name="hermes-static-prompt-priming",
            daemon=True,
        )
        thread.start()
        return True
    except Exception:
        logger.debug("static-prompt priming: failed to schedule", exc_info=True)
        return False
