"""One-shot, best-effort local-backend KV-cache priming with the real Hermes
static system-prompt prefix (Gate 10P; extended to include the skills index
in Gate 10R — see ``agent.static_system_prompt.build_primed_prefix_with_skills``).

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
  - Builds the prompt via ``agent.static_system_prompt.build_primed_prefix_with_skills``
    (Gate 10P's stable tier plus the Gate 10R skills index) — pure config/file
    reads only: no AIAgent, no conversation, no session row, no memory write,
    no tool call, no Telegram/Discord/etc. send.
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

Gate CS9 (cold-start report, ``/tmp/hermes-cold-start-root-cause-report.txt``):
Gates 10O-10S all primed the system-prompt TEXT but never passed ``tools``
on the priming POST. A real turn always sends ``tools=agent.tools or None``
(``run_agent.py``'s ``self.tools`` — currently ~18 schemas, ~11k tokens as
JSON for the telegram platform), and llama.cpp's Qwen chat template renders
a whole extra ``<|im_start|>system\n# Tools...`` block *ahead of* the
caller-supplied system message whenever ``tools`` is non-empty. That means
the rendered token sequence priming produced NEVER shared a cache prefix
with any real tool-using turn, no matter how well the system-prompt text
itself matched — every real first turn silently paid full prefill for the
entire prompt (tools block included), live-measured at ~290s on this
hardware, while priming's own reported "success" (fast, high cache_n)
never reflected that miss. This module now also resolves and sends the
real tool schema list (:func:`agent.static_system_prompt.resolve_static_prompt_tools`)
so the primed KV-cache prefix actually matches what a real first turn
renders. Live-measured fix: the same real request dropped from ~290s
(cold, tools-only mismatch) to ~1.5s (cache_n covering the full tools +
system + skills prefix, only the trailing per-turn tokens novel).

Gate 10S: ``gateway.run.start_gateway`` used to ALSO schedule Gate 10O's
generic backend warmup (``agent.llm_backend_warmup.schedule_startup_
warmup``) immediately before this module's own ``schedule_startup_
priming``, as two independent daemon threads. Both hit the same single KV
slot with their own unrelated bounded health-wait/retry loops and no
ordering between them, so whichever POST actually landed last on the
backend nondeterministically decided what ended up cached — sometimes the
useless generic placeholder, sometimes this module's real prefix. Gate 10S
removed the generic warmup from the startup path entirely (its useful
behavior — reachability check, chat-template exercise, first-call kernel
ramp-up — is a strict subset of what THIS module's own real inference call
already does), so this priming call is now the ONLY startup action that can
touch the model, making its outcome deterministic by construction rather
than by scheduling order. See ``wait_for_priming_ready`` for an opt-in,
bounded way for a caller to avoid racing the remaining, much narrower
window (a genuine user message arriving before this one priming call
finishes) — not wired in automatically, since a missed prime only costs
one turn its cache speedup, never correctness.
"""

from __future__ import annotations

import dataclasses
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
    "wait_for_priming_ready",
    "get_last_priming_result",
]

_PRIMING_USER_MESSAGE = "Reply with only the word READY."
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BACKOFF_SECONDS = (0.5, 1.0, 2.0)
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 180.0
_DEFAULT_HEALTH_TIMEOUT_SECONDS = 30.0
_DEFAULT_HEALTH_POLL_INTERVAL_SECONDS = 0.5

# Process-lifetime guard: priming must fire at most once per gateway run.
_priming_already_scheduled = False

# Gate 10S: set (unconditionally, in a ``finally``) once the ONE scheduled
# background priming attempt concludes — success, failure, exception, or a
# same-process no-op (non-local backend). Never set on a fresh process/after
# ``reset_priming_state_for_tests()``, so a bounded wait on it can never
# return a stale "ready" from a previous, unrelated process lifetime.
_priming_done_event = threading.Event()

# Observability only (Gate 10S): the most recent background priming
# outcome, for a status endpoint or diagnostic to read without re-running
# anything. Content-free (see PrimingResult — no prompt text, no user data).
_last_priming_result: Optional["PrimingResult"] = None


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
    # Gate 10R: fingerprint of the on-disk skill registry at priming time
    # ("" when no skills are visible for the primed platform). Lets a
    # caller detect that skills changed since the prefix was primed and
    # decide whether a re-prime is worthwhile — never used to invalidate
    # anything automatically here.
    skills_fingerprint: str = ""
    # Gate 10S: wall-clock (time.time()) bounds of the background priming
    # attempt, for observability only — never used for cache logic. None
    # when the result was constructed directly (e.g. by a caller of
    # ``prime_static_system_prompt`` outside the scheduled background path).
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


def reset_priming_state_for_tests() -> None:
    """Test-only: clear the process-lifetime guard between test cases."""
    global _priming_already_scheduled, _last_priming_result
    _priming_already_scheduled = False
    _priming_done_event.clear()
    _last_priming_result = None


def wait_for_priming_ready(timeout: float = 5.0) -> bool:
    """Bounded, opt-in wait for the ONE scheduled background prime to finish.

    Returns True once priming has concluded (whatever the outcome — success
    or failure), False if *timeout* elapses first. Never raises, never
    blocks longer than *timeout*, and never blocks at all if priming was
    never scheduled in this process (the wait simply times out and the
    caller proceeds — see module docstring: "no user request may hang
    indefinitely").

    Nothing in this module calls this automatically today — it exists so a
    caller that wants to avoid racing the startup prime (e.g. the very
    first inbound message right after a gateway restart) CAN opt in with a
    short, bounded wait, without this module reaching into the request path
    itself. See the Gate 10S report for why the live gateway does not wire
    this in by default (the proven race — two independent warmup actors —
    is eliminated by removing the redundant one, not by blocking traffic;
    a missed prime just costs one turn its cache speedup, not correctness).
    """
    try:
        return _priming_done_event.wait(max(0.0, timeout))
    except Exception:
        return False


def get_last_priming_result() -> Optional["PrimingResult"]:
    """The most recent background priming outcome, or None if never run.

    Read-only observability accessor — never triggers a (re-)prime.
    """
    return _last_priming_result


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
    base_url: str, api_key: str, model: str, system_prompt: str, timeout: float,
    tools: Optional[list] = None,
) -> dict:
    url = f"{base_url}/chat/completions"
    body: dict = {
        "model": model or "default",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _PRIMING_USER_MESSAGE},
        ],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    # Gate CS9: a real turn sends ``tools=agent.tools or None`` (never an
    # empty list) alongside the system prompt. Chat templates that support
    # tool-calling (e.g. Qwen's) render a "# Tools" preamble as its OWN
    # leading system block whenever ``tools`` is non-empty/non-null —
    # ahead of the actual system message — so the rendered token sequence
    # for a tools-bearing request shares almost no prefix with one primed
    # without tools, regardless of how well the system-prompt TEXT matches.
    # Omitting this (as prior gates did) made priming's KV-cache warm-up
    # invisible to every real tool-using turn: measured live, an otherwise
    # cache-hit-ready request costs ~290s instead of ~1.5s when only the
    # ``tools`` field differs (see the Gate CS cold-start report). Matching
    # that shape here is what makes the primed prefix reusable at all.
    if tools:
        body["tools"] = tools
    payload = json.dumps(body).encode("utf-8")
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
        from agent.static_system_prompt import build_primed_prefix_with_skills

        static_prompt, skills_fingerprint = build_primed_prefix_with_skills(
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

    # Gate CS9: resolve the SAME tool schema list a real turn on *platform*
    # would send. Best-effort — a resolution failure must not sacrifice the
    # system-prompt priming that already worked; it only forfeits the
    # tools-preamble cache-hit benefit for this one attempt.
    try:
        from agent.static_system_prompt import resolve_static_prompt_tools

        priming_tools = resolve_static_prompt_tools(platform=platform)
    except Exception:
        logger.debug("static-prompt priming: tool resolution failed", exc_info=True)
        priming_tools = []

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
                tools=priming_tools,
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
                "prompt_chars=%d prompt_tokens=%s cached_tokens=%s tool_count=%d "
                "skills_fingerprint=%s endpoint_host=%s",
                attempt, elapsed_ms, len(static_prompt), prompt_tokens, cached_tokens,
                len(priming_tools), skills_fingerprint or "-",
                urlparse(backend["base_url"]).hostname,
            )
            return PrimingResult(
                attempted=True, succeeded=True, attempts=attempt, elapsed_ms=elapsed_ms,
                prompt_chars=len(static_prompt), prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens, reason="ok",
                skills_fingerprint=skills_fingerprint,
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
        reason=last_error, skills_fingerprint=skills_fingerprint,
    )


def _run_priming_in_background() -> None:
    """Gate 10S: the ONE background action that may touch the model at
    startup. Always records a result and always signals
    ``_priming_done_event`` in a ``finally`` — regardless of outcome — so
    :func:`wait_for_priming_ready` and :func:`get_last_priming_result` are
    reliable for every code path (success, failure, exception, or a
    same-process no-op for a non-local backend)."""
    global _last_priming_result
    started_at = time.time()
    try:
        result = prime_static_system_prompt()
    except Exception:
        logger.debug("static-prompt priming: background task failed", exc_info=True)
        result = PrimingResult(
            attempted=True, succeeded=False, attempts=0, elapsed_ms=None,
            prompt_chars=None, prompt_tokens=None, cached_tokens=None,
            reason="unexpected_exception",
        )
    finally:
        completed_at = time.time()
        try:
            _last_priming_result = dataclasses.replace(
                result, started_at=started_at, completed_at=completed_at,
            )
        except Exception:
            _last_priming_result = result
        _priming_done_event.set()


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
