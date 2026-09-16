"""One-shot, best-effort local LLM backend warmup (Gate 10O).

Direct measurement against this deployment's local llama.cpp backend
(``model.provider: custom``, loopback ``base_url``) showed a large gap
between a "cold" request (no prompt-prefix KV cache hit) and a "warm" one
against the same prefix — and, separately, that the very first inference
call issued to the backend runs at a markedly slower per-token rate than
later calls in the same process lifetime. Neither effect is model-loading
(the backend process stays resident continuously via systemd
``Restart=always``) — see the Gate 10O report for the full measurement.

This module pays down the *generic* part of that cost (first-call runtime
ramp-up, and confirming the backend is actually reachable) once, right
after the gateway starts, off the critical path of any real message. It
deliberately does **not** attempt to reconstruct Hermes's real system
prompt (that would require constructing a full ``AIAgent`` — memory
provider, tool registry, skill config — purely to throw its output away,
which the Gate 10O spec explicitly discourages: "Do NOT generate a fake
conversation merely to warm Hermes unless unavoidable"). The dominant,
prompt-size-scaled portion of cold-start latency is a separate, larger
follow-up (see the report's recommended next step).

Gate 10S: this module is no longer scheduled automatically by
``gateway.run.start_gateway`` — measurement showed its generic warmup
request and Gate 10P/10R's real stable+skills priming (``agent.
llm_static_prompt_priming``) were two independent, unordered background
threads racing the SAME single llama.cpp KV slot (``--parallel 1``), and
that the real prime's own inference call already pays down everything this
generic warmup provides (backend reachability, chat-template exercise,
first-call kernel ramp-up) for a backend that stays resident across gateway
restarts. Removing the redundant actor made startup priming deterministic
by construction rather than by ordering. This module, its tests, and
``warm_local_llamacpp_backend``/``schedule_startup_warmup`` are otherwise
completely unchanged and remain available for manual/diagnostic use or a
future opt-in call site — see the Gate 10S report for the full comparison.

Safety boundary (all enforced below, see tests in
``tests/agent/test_llm_backend_warmup.py``):
  - Only ever targets a loopback base_url (127.0.0.1 / localhost / ::1).
    Never fires against a remote/paid provider, even if misconfigured.
  - Fire-and-forget: runs in a background asyncio task and never blocks or
    fails gateway startup.
  - Exactly one warmup per process lifetime (module-level guard).
  - Uses a bare HTTP POST (stdlib ``urllib``, no ``openai`` SDK, no
    ``AIAgent``) — cannot touch the agent cache, memory provider, tool
    registry, session store, Knowledge DB, review DB, or promotion ledger.
  - The completion output is discarded; only timing is logged, via the
    same opt-in ``agent.latency_diagnostics`` channel (content-free).
  - Bounded retry (default 3 attempts, short fixed backoff) — never an
    unbounded/tight loop.
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

logger = logging.getLogger(__name__)

__all__ = [
    "WarmupResult",
    "resolve_local_backend",
    "warm_local_llamacpp_backend",
    "schedule_startup_warmup",
    "reset_warmup_state_for_tests",
]

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
_WARMUP_PROMPT = "Reply with only the word READY."
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BACKOFF_SECONDS = (0.5, 1.0, 2.0)
_DEFAULT_TIMEOUT_SECONDS = 30.0

# Process-lifetime guard: warmup must fire at most once per gateway run,
# regardless of how many times startup code is (re-)entered.
_warmup_already_scheduled = False


@dataclass(frozen=True)
class WarmupResult:
    attempted: bool
    succeeded: bool
    attempts: int
    elapsed_ms: Optional[int]
    reason: str


def reset_warmup_state_for_tests() -> None:
    """Test-only: clear the process-lifetime guard between test cases."""
    global _warmup_already_scheduled
    _warmup_already_scheduled = False


def _is_loopback_base_url(base_url: str) -> bool:
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except Exception:
        return False
    return host in _LOOPBACK_HOSTS


def resolve_local_backend() -> Optional[dict]:
    """Read the configured primary model backend, read-only, no side effects.

    Returns ``None`` (skip warmup) unless the resolved provider is a local
    loopback endpoint — this warmup is only meaningful for a
    self-hosted backend (llama.cpp/Ollama/vLLM et al. under
    ``provider: custom``) that stays resident between requests; a remote
    hosted provider has no local process to "warm" and must never receive
    unsolicited traffic.
    """
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        model_cfg = cfg.get("model") or {}
        base_url = str(model_cfg.get("base_url") or "").strip()
        provider = str(model_cfg.get("provider") or "").strip().lower()
        model_name = str(model_cfg.get("default") or "").strip()
        if not base_url or not _is_loopback_base_url(base_url):
            return None
        if provider not in {"custom", "ollama", ""}:
            # Only known-local-style provider profiles. Anything else is
            # treated conservatively as "don't guess" even if the host
            # happens to resolve to loopback (e.g. an SSH tunnel to a
            # remote paid endpoint).
            return None
        return {
            "base_url": base_url.rstrip("/"),
            "model": model_name,
            "api_key": str(model_cfg.get("api_key") or "none"),
        }
    except Exception:
        logger.debug("latency warmup: backend resolution failed", exc_info=True)
        return None


def _post_completion(base_url: str, api_key: str, model: str, timeout: float) -> None:
    url = f"{base_url}/chat/completions"
    payload = json.dumps(
        {
            "model": model or "default",
            "messages": [{"role": "user", "content": _WARMUP_PROMPT}],
            "max_tokens": 4,
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
        # Output is intentionally discarded — this call exists purely to
        # pay down first-call/prefix-cache cost, never to produce content
        # a user or any downstream system observes.
        response.read()


def warm_local_llamacpp_backend(
    *,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: tuple = _DEFAULT_BACKOFF_SECONDS,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> WarmupResult:
    """Synchronous warmup body — safe to call directly from tests.

    Bounded retry only: at most ``max_attempts`` tries, each waiting a
    fixed short backoff before the next, then gives up harmlessly. Never
    raises — every failure mode (unreachable model, timeout, malformed
    response) is swallowed and reflected in the returned ``WarmupResult``.
    """
    backend = resolve_local_backend()
    if backend is None:
        return WarmupResult(
            attempted=False, succeeded=False, attempts=0, elapsed_ms=None,
            reason="no_local_backend_configured",
        )

    started = time.monotonic()
    last_error = "unknown"
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            _post_completion(
                backend["base_url"], backend["api_key"], backend["model"], timeout_seconds,
            )
            elapsed_ms = int(round((time.monotonic() - started) * 1000))
            logger.info(
                "llm backend warmup succeeded: attempt=%d elapsed_ms=%d endpoint_host=%s",
                attempt, elapsed_ms, urlparse(backend["base_url"]).hostname,
            )
            return WarmupResult(
                attempted=True, succeeded=True, attempts=attempt,
                elapsed_ms=elapsed_ms, reason="ok",
            )
        except Exception as exc:  # noqa: BLE001 - warmup must never propagate
            last_error = type(exc).__name__
            if attempt < max_attempts:
                delay = backoff_seconds[min(attempt - 1, len(backoff_seconds) - 1)]
                time.sleep(delay)

    elapsed_ms = int(round((time.monotonic() - started) * 1000))
    logger.info(
        "llm backend warmup gave up: attempts=%d elapsed_ms=%d last_error=%s",
        max_attempts, elapsed_ms, last_error,
    )
    return WarmupResult(
        attempted=True, succeeded=False, attempts=max_attempts,
        elapsed_ms=elapsed_ms, reason=last_error,
    )


def _run_warmup_in_background() -> None:
    try:
        warm_local_llamacpp_backend()
    except Exception:
        logger.debug("latency warmup: background task failed", exc_info=True)


def schedule_startup_warmup() -> bool:
    """Fire the warmup once, off the startup critical path.

    Uses a plain daemon thread rather than an asyncio task. ``start_gateway()``
    goes on to do a lot more event-loop-owning setup after this call (adapter
    startup, Telegram's own polling loop, etc.); a task scheduled on
    "whatever loop happens to be current right now" is at the mercy of all of
    that — if anything later replaces/closes/starts a fresh loop, or a
    startup-time task-cancellation sweep runs, the task can be silently
    dropped before it ever executes. A daemon thread has no such dependency:
    it runs independently of asyncio entirely, cannot block interpreter
    shutdown (``daemon=True``), and ``warm_local_llamacpp_backend`` is
    already a plain, fully synchronous, self-contained call.

    Idempotent per process: a second call (e.g. a defensive re-entry of
    gateway startup code) is a silent no-op. Returns True iff a warmup
    thread was actually started this call (useful for tests), never raises.
    """
    global _warmup_already_scheduled
    try:
        if _warmup_already_scheduled:
            return False
        _warmup_already_scheduled = True
        thread = threading.Thread(
            target=_run_warmup_in_background,
            name="hermes-llm-backend-warmup",
            daemon=True,
        )
        thread.start()
        return True
    except Exception:
        logger.debug("latency warmup: failed to schedule", exc_info=True)
        return False
