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
  - Exactly one STARTUP priming attempt per process lifetime (module-level
    guard) — see Gate CS10 below for why later re-primes are now allowed.
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

Gate CS10 (cold-start re-priming, this report):
Gate CS9 fixed a content mismatch (missing ``tools``) but left the module's
core lifetime assumption broken: :func:`schedule_startup_priming` primed
the backend's KV cache EXACTLY ONCE per Hermes gateway process, and nothing
ever revisited that decision. In production, ``llama-server`` (the
``llama-vulkan.service`` systemd unit) can and does restart independently
of the Hermes gateway — a restart wipes the KV cache but leaves the
gateway's one-shot guard set, so the gateway never re-primed and the next
real user message paid full cold prefill (live-measured: 16,483 prompt
tokens, ~247.6s prompt eval, ~260.2s total). Separately, the resolved
tool-schema SET can drift within one gateway lifetime (a toolset add/
remove, a config change) without anything invalidating the earlier prime,
silently reproducing the exact CS9 mismatch it fixed.

This gate removes the "one successful prime is valid forever" assumption.
:func:`check_and_reprime_if_stale` computes a cheap :class:`BackendSignature`
(``/props`` config fields plus the highest ``id_task`` seen on ``/slots`` —
see that class's docstring for why: llama.cpp exposes no first-class
"process start time"/PID over its HTTP API, so a monotonic per-process task
counter that resets on restart is the least invasive reliable substitute
this deployment's llama.cpp build offers) and a deterministic SHA-256 of
the FULL resolved tool schema list, compares both against what the last
SUCCESSFUL prime actually used, and — if either has moved on — marks the
existing prime stale and starts exactly one background re-prime (the same
bounded/best-effort :func:`prime_static_system_prompt` call the startup
path already used; never blocks a caller). :func:`schedule_startup_priming`
additionally registers this check as a recurring callback on the shared
``agent.periodic_scheduler`` thread (the same one-thread-for-many-callbacks
mechanism ``agent/turn_liveness.py`` and ``agent/turn_facade_lease.py``
already use), so a restart LATER in the same gateway lifetime — not just
the very first one — gets caught automatically, without ever blocking a
real request on the check.

Limitation: the ``/slots`` task-counter signal requires llama.cpp's
``--slots``/``endpoint_slots`` to be enabled (confirmed true, live, on this
deployment) and can expose other callers' cached-prompt metadata to anyone
who can reach the endpoint, so some deployments disable it. Without it,
this module still catches a MODEL or BUILD change (``/props``'s
``model_path``/``model_ftype``/``build_info``) and any tool-schema drift,
but a bare restart of the SAME binary against the SAME model — the actual
scenario this gate targets — would go undetected until it is caught by a
future request anyway (safe, just slower). See the reliability-refinement
section below for why ``/slots`` is no longer this module's PRIMARY restart
signal on this deployment.

Reliability refinement (systemd instance identity): live testing exposed a
race in the ``/slots``-only design: if a real request reaches a freshly
restarted llama-server before the next watchdog check, that request gives
the new process's slot a normal-looking (even high) ``id_task`` — the
task-counter comparison can then no longer prove a restart happened. This
deployment runs llama-server under systemd (``llama-vulkan.service``), and
``systemctl show <unit> --property MainPID,ActiveEnterTimestampMonotonic``
is confirmed reachable without special privilege on this host and changes
deterministically on every bare restart of that unit, regardless of how
much traffic the new process has already served. :func:`_query_backend_
signature` now also captures this (via :func:`_query_systemd_instance_
signature`, a bounded, ``shell=False`` subprocess call — never raises, never
blocks longer than its timeout) and :func:`_is_stale_locked` prefers it over
the ``/slots`` heuristic whenever both the last-primed and current
signatures have it: an exact ``MainPID:ActiveEnterTimestampMonotonic``
match is treated as definitive proof of the same instance (skipping the
``/slots`` check entirely for that call, since it can no longer be trusted
once a request has already landed), and any difference is an immediate
restart signal. The ``/slots`` task-counter remains the fallback whenever
the systemd signal is unavailable on either side (systemd absent, `systemctl`
missing/erroring, or the unit name doesn't match — deliberately hardcoded
to ``llama-vulkan.service`` for this deployment; portability to a
differently-named or non-systemd backend was judged less important than
deterministic correctness here) — safe by construction, since an
unavailable signal never fabricates an identity, it just falls through to
the next one.

Reliability refinement (tool-drift debounce): tool availability ``check_fn``s
are dynamic and can flap (observed live: a Home Assistant integration
repeatedly connecting/disconnecting). A single differing tool-fingerprint
observation is no longer enough to trigger a re-prime by itself — see
:func:`_evaluate_tool_drift_locked` — a candidate fingerprint must be seen
on ``_TOOL_DRIFT_CONFIRM_THRESHOLD`` (2) CONSECUTIVE watchdog checks before
it is treated as confirmed drift, the candidate resets the moment an
in-between check observes the currently-primed fingerprint again, and a
confirmed drift additionally starts a ``_TOOL_DRIFT_COOLDOWN_SECONDS`` (180s)
cooldown before another tool-drift-triggered re-prime may fire — so a
flapping integration cannot cause continuous multi-minute priming work.
Backend-restart detection is NOT debounced — it remains immediate, exactly
as it was before this refinement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from agent.llm_backend_warmup import resolve_local_backend

try:
    from agent._prefixdiag import log_prefix_diag
except Exception:  # pragma: no cover - diagnostics must never block priming
    def log_prefix_diag(*_args, **_kwargs) -> None:
        return None

logger = logging.getLogger(__name__)

__all__ = [
    "PrimingResult",
    "BackendSignature",
    "prime_static_system_prompt",
    "schedule_startup_priming",
    "check_and_reprime_if_stale",
    "reset_priming_state_for_tests",
    "wait_for_priming_ready",
    "get_last_priming_result",
]

_PRIMING_USER_MESSAGE = "Reply with only the word READY."
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BACKOFF_SECONDS = (0.5, 1.0, 2.0)
# Cache-eviction diagnostic (/tmp/hermes-cache-eviction-diagnostic.txt):
# a true-cold prime of the real ~15k-token Hermes prefix measured
# 225-290s+ on this hardware — the previous 180s default guaranteed the
# first attempt would time out client-side on every true-cold prime
# (llama.cpp kept processing regardless, so the retry usually "succeeded"
# fast off the now-warm cache, but only after wasting ~180s first). 360s
# gives ~25% margin above the highest measured value without masking a
# genuinely hung backend for an unreasonable time.
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 360.0
_DEFAULT_HEALTH_TIMEOUT_SECONDS = 30.0
_DEFAULT_HEALTH_POLL_INTERVAL_SECONDS = 0.5

# Gate CS10: bounded timeout for the cheap /props + /slots identity probes —
# these are tiny local GETs (no inference), so a short timeout is enough
# and keeps a staleness check from ever hanging noticeably.
_DEFAULT_BACKEND_QUERY_TIMEOUT_SECONDS = 3.0
# How often the recurring staleness watchdog (registered by
# schedule_startup_priming) re-checks backend identity + tool fingerprint.
# Cheap (two small local GETs plus the same in-process tool resolution a
# real turn does), but not free (tool resolution runs each tool's own
# availability check_fn — see resolve_static_prompt_tools), so this is
# polled, not hammered.
_DEFAULT_STALENESS_POLL_INTERVAL_SECONDS = 30.0
# After a FAILED re-prime attempt, don't try again for at least this long —
# prevents a backend that is down (or a tool check_fn that is erroring) from
# turning every watchdog tick into another doomed attempt.
_MIN_REPRIME_RETRY_INTERVAL_SECONDS = 15.0

# Reliability refinement: the systemd unit backing this deployment's
# llama-server. Hardcoded (not read from config) — deliberately, per the
# instruction that deterministic correctness on THIS deployment matters
# more than portability to a hypothetically differently-named unit.
_LLAMA_SYSTEMD_UNIT = "llama-vulkan.service"
# `systemctl show` against a local unit is normally near-instant; bounded
# generously so a hung/overloaded system can never turn this into an
# unbounded wait.
_DEFAULT_SYSTEMCTL_TIMEOUT_SECONDS = 5.0

# Reliability refinement (tool-drift debounce): number of consecutive
# watchdog checks a candidate tool fingerprint must be observed before it
# is treated as confirmed drift (not just transient check_fn flapping).
_TOOL_DRIFT_CONFIRM_THRESHOLD = 2
# Once a CONFIRMED tool-drift re-prime FINISHES (regardless of whether it
# succeeded or failed), suppress further tool-drift-triggered re-primes for
# this long. Measured from completion, not from confirmation -- the re-prime
# itself can take ~200s+ (a true-cold prefill), and _prime_in_progress
# already blocks any new trigger for that whole stretch; this cooldown's
# job is purely to stop a flapping integration from immediately queuing
# ANOTHER one right after. Distinct from _MIN_REPRIME_RETRY_INTERVAL_SECONDS
# (which only guards a FAILED attempt, at 15s).
_TOOL_DRIFT_COOLDOWN_SECONDS = 180.0

# Process-lifetime guard: the STARTUP prime must fire at most once per
# gateway run. Gate CS10 does not change this — it only stops treating that
# one success as valid forever; see _prime_valid below.
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

# Gate CS10: guards everything below. Every section held under it is a
# handful of attribute reads/writes, never I/O, so a plain (non-reentrant)
# lock is enough.
_state_lock = threading.Lock()

# True only for the stretch between "a prime succeeded against the
# CURRENTLY recorded backend identity + tool fingerprint" and "we next
# observe either one to have moved on". Starts False (nothing primed yet)
# and is never treated as permanently True after one success — that
# lifetime assumption was Gate CS9's remaining blind spot (see module
# docstring).
_prime_valid = False

# Set for the duration of ANY prime attempt — the original startup one or a
# CS10-triggered re-prime — so at most one ever runs concurrently.
_prime_in_progress = False

# The backend/tool state the last SUCCESSFUL prime actually used. None
# until the first success.
_last_primed_backend_signature: Optional["BackendSignature"] = None
_last_primed_tool_fingerprint: Optional[str] = None

# time.monotonic() of the last FAILED re-prime attempt; used only to space
# out retries (see _MIN_REPRIME_RETRY_INTERVAL_SECONDS).
_last_failed_reprime_monotonic: Optional[float] = None

# Cancel handle for the recurring staleness watchdog registered on the
# shared agent.periodic_scheduler thread — None until
# schedule_startup_priming() registers it; cancelled and re-nulled by
# reset_priming_state_for_tests() so it can never fire against a later,
# unrelated test's monkeypatched config.
_watchdog_handle = None

# Reliability refinement (tool-drift debounce). All guarded by _state_lock,
# mutated only inside _evaluate_tool_drift_locked (see its docstring).
# The tool fingerprint currently being "watched" as a possible drift (not
# yet confirmed) -- None when no candidate is pending.
_tool_drift_candidate_fingerprint: Optional[str] = None
# How many CONSECUTIVE checks have observed _tool_drift_candidate_fingerprint.
_tool_drift_candidate_count: int = 0
# time.monotonic() deadline before which a newly-confirmed tool-drift must
# not trigger another re-prime -- None until the first confirmed drift.
_tool_drift_cooldown_until_monotonic: Optional[float] = None


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
    # Gate CS10: deterministic SHA-256 (full hex) of the resolved tool
    # schema list actually sent on this attempt — "" if never computed
    # (e.g. an early-return before tool resolution). Used by
    # check_and_reprime_if_stale to detect tool-schema drift.
    tool_fingerprint: str = ""
    # Gate CS10: the BackendSignature captured immediately after THIS
    # attempt's own successful completion — "" / None on failure or if the
    # /props probe itself failed. See BackendSignature's docstring.
    backend_config_fingerprint: str = ""
    backend_task_high_water_mark: Optional[int] = None
    # Gate CS10: whether /slots was actually reachable when the above was
    # captured -- needed to record an accurate BackendSignature (see that
    # class's docstring on why a None task_high_water_mark is informative
    # only when slots WAS reachable).
    backend_slots_reachable: bool = False
    # Reliability refinement: MainPID:ActiveEnterTimestampMonotonic from
    # `systemctl show` for the configured llama-server unit, captured
    # alongside the above -- None when systemd/the unit is unavailable.
    backend_systemd_instance: Optional[str] = None


@dataclass(frozen=True)
class BackendSignature:
    """A cheap fingerprint of "which llama-server process, with what static
    config, are we talking to" — Gate CS10.

    ``config_fingerprint`` is derived from ``/props`` fields that describe
    the loaded model/build (``model_path``, ``model_ftype``, ``build_info``,
    ``total_slots``) — stable across a bare restart of the SAME binary/
    model (confirmed by direct measurement against this deployment: none of
    those fields change when ``llama-vulkan.service`` is restarted with an
    unchanged command line), so this alone only catches a MODEL or BUILD
    change, never a bare restart.

    ``task_high_water_mark`` is the highest ``id_task`` observed across
    ``/slots`` — llama.cpp's own per-process, monotonically increasing
    request counter. It resets on every fresh process; direct observation
    against this deployment shows a slot that has not yet processed ANY
    request in its (new) process lifetime simply OMITS ``id_task`` from
    ``/slots`` entirely, rather than reporting it as ``0`` — so ``None``
    here is itself informative (not just "unknown") whenever
    ``slots_reachable`` is True: a previously-primed backend whose slot no
    longer reports the ``id_task`` we ourselves last left it with can ONLY
    be a fresh process. ``task_high_water_mark`` is ``None`` with
    ``slots_reachable=False`` when ``/slots`` itself is unreachable or the
    deployment reports ``endpoint_slots: false`` (some deployments disable
    it, since it can expose other callers' cached prompt metadata) — see
    the module docstring's Gate CS10 section for the resulting limitation
    in THAT case only.
    """

    config_fingerprint: str
    task_high_water_mark: Optional[int]
    # True iff /slots itself was successfully reached and parsed this probe
    # (regardless of whether any slot reported an id_task) — lets
    # _is_stale_locked tell "the counter is gone because the process is
    # new" apart from "I couldn't ask at all".
    slots_reachable: bool = False
    # Reliability refinement: ``f"{MainPID}:{ActiveEnterTimestampMonotonic}"``
    # from ``systemctl show`` for the configured llama-server systemd unit
    # (see :func:`_query_systemd_instance_signature`) — the AUTHORITATIVE
    # per-instance identity when available: unlike ``task_high_water_mark``,
    # it is unaffected by how many requests the (possibly new) process has
    # already served, closing the race where a real request reaches a
    # freshly restarted backend before this module's own next check.
    # ``None`` when systemd/the configured unit/``systemctl`` itself is
    # unavailable, in which case ``task_high_water_mark`` remains the
    # fallback restart signal (see module docstring's reliability-
    # refinement section).
    systemd_instance: Optional[str] = None


def reset_priming_state_for_tests() -> None:
    """Test-only: clear ALL priming/re-priming state between test cases,
    including cancelling any registered Gate CS10 staleness watchdog so it
    can never fire — against whatever config a LATER, unrelated test has
    monkeypatched — after this test ends."""
    global _priming_already_scheduled, _last_priming_result
    global _prime_valid, _prime_in_progress, _last_primed_backend_signature
    global _last_primed_tool_fingerprint, _last_failed_reprime_monotonic
    global _watchdog_handle
    global _tool_drift_candidate_fingerprint, _tool_drift_candidate_count
    global _tool_drift_cooldown_until_monotonic
    _priming_already_scheduled = False
    _priming_done_event.clear()
    _last_priming_result = None
    _prime_valid = False
    _prime_in_progress = False
    _last_primed_backend_signature = None
    _last_primed_tool_fingerprint = None
    _last_failed_reprime_monotonic = None
    if _watchdog_handle is not None:
        try:
            _watchdog_handle.cancel()
        except Exception:
            pass
    _watchdog_handle = None
    _tool_drift_candidate_fingerprint = None
    _tool_drift_candidate_count = 0
    _tool_drift_cooldown_until_monotonic = None


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


def _props_url(base_url: str) -> str:
    """Derive the llama.cpp ``/props`` URL from a ``.../v1``-style base_url."""
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return f"{trimmed}/props"


def _slots_url(base_url: str) -> str:
    """Derive the llama.cpp ``/slots`` URL from a ``.../v1``-style base_url."""
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[: -len("/v1")]
    return f"{trimmed}/slots"


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


def _get_json(url: str, *, timeout: float) -> Optional[dict]:
    """Bounded, best-effort GET-and-parse. Never raises; None on any failure
    (unreachable, non-2xx, malformed/non-object body) — Gate CS10's
    identity/staleness detection degrades to "unknown, don't change
    anything" on failure, never to an exception a caller has to handle."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if not (200 <= response.status < 300):
                return None
            body = json.loads(response.read())
            return body if isinstance(body, dict) else None
    except Exception:
        return None


def _get_json_list(url: str, *, timeout: float) -> Optional[list]:
    """Same contract as :func:`_get_json` but for an endpoint (``/slots``)
    whose top-level body is a JSON array."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if not (200 <= response.status < 300):
                return None
            body = json.loads(response.read())
            return body if isinstance(body, list) else None
    except Exception:
        return None


def _query_systemd_instance_signature(
    unit: str = _LLAMA_SYSTEMD_UNIT, *, timeout: float = _DEFAULT_SYSTEMCTL_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Reliability refinement: the AUTHORITATIVE backend-instance identity
    when available — ``MainPID`` + ``ActiveEnterTimestampMonotonic`` from
    ``systemctl show <unit>``, combined into one opaque string. Both are
    confirmed reachable without special privilege on this host and change
    deterministically on every bare restart of the unit, regardless of how
    many requests the new process has already served — unlike ``/slots``'
    ``id_task`` (see :class:`BackendSignature`'s docstring for the race this
    closes). Uses the MONOTONIC timestamp (not the wall-clock
    ``ActiveEnterTimestamp``) because it is a plain integer (microseconds
    since boot) — immune to clock adjustments/timezone/locale formatting
    differences that a string-compare on the human-readable timestamp would
    not be.

    A plain list of args is passed to :func:`subprocess.run` — never
    ``shell=True`` — with a bounded timeout. Returns ``None`` (never raises)
    if ``systemctl`` is missing, the call times out, exits non-zero, or the
    unit reports no ``MainPID``/timestamp (e.g. an unknown unit name, which
    systemd reports as an all-empty/zeroed property set with exit code 0
    rather than an error) — callers must treat ``None`` exactly like
    "identity unknown", never "restarted" (see :func:`_is_stale_locked`).
    """
    try:
        result = subprocess.run(
            [
                "systemctl", "show", unit, "--no-pager",
                "--property", "MainPID,ActiveEnterTimestampMonotonic",
            ],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if result.returncode != 0:
        return None

    props: dict = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        props[key.strip()] = value.strip()

    main_pid = props.get("MainPID", "")
    active_enter = props.get("ActiveEnterTimestampMonotonic", "")
    if not main_pid or main_pid == "0" or not active_enter or active_enter == "0":
        return None
    return f"{main_pid}:{active_enter}"


def _query_backend_signature(
    base_url: str, *, timeout: float = _DEFAULT_BACKEND_QUERY_TIMEOUT_SECONDS,
) -> Optional["BackendSignature"]:
    """Gate CS10: cheap (~1-2 GETs, no inference) snapshot of backend
    identity. Returns None only when ``/props`` itself is unreachable or
    unparseable — a fully-unreachable backend is otherwise handled by the
    existing bounded health-wait elsewhere in this module; here it just
    means "can't tell right now", which callers must treat as "no change",
    never as "definitely restarted" (see :func:`_is_stale_locked`)."""
    props = _get_json(_props_url(base_url), timeout=timeout)
    if props is None:
        return None

    config_view = {
        "model_path": props.get("model_path"),
        "model_ftype": props.get("model_ftype"),
        "build_info": props.get("build_info"),
        "total_slots": props.get("total_slots"),
    }
    try:
        config_fingerprint = hashlib.sha256(
            json.dumps(config_view, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
    except Exception:
        config_fingerprint = ""

    task_high_water_mark: Optional[int] = None
    slots_reachable = False
    # Absent on older llama.cpp builds that predate this flag; try /slots
    # anyway in that case rather than assuming it is disabled.
    if props.get("endpoint_slots", True):
        slots = _get_json_list(_slots_url(base_url), timeout=timeout)
        if slots is not None:
            slots_reachable = True
            task_ids = [
                slot.get("id_task")
                for slot in slots
                if isinstance(slot, dict) and isinstance(slot.get("id_task"), int)
            ]
            if task_ids:
                task_high_water_mark = max(task_ids)
            # else: reachable, but no slot has processed a task in THIS
            # process's lifetime yet -- task_high_water_mark stays None,
            # and slots_reachable=True lets _is_stale_locked tell that
            # apart from "couldn't ask" (see BackendSignature docstring).

    systemd_instance = _query_systemd_instance_signature()

    return BackendSignature(
        config_fingerprint=config_fingerprint, task_high_water_mark=task_high_water_mark,
        slots_reachable=slots_reachable, systemd_instance=systemd_instance,
    )


def _compute_tool_fingerprint(tools: Optional[list]) -> str:
    """Gate CS10: deterministic fingerprint of the FULL resolved tool schema
    list (never just names) — must catch anything that would change the
    rendered ``# Tools`` preamble bytes: an added/removed tool, a changed
    parameter schema, or a reordering. List order is preserved (not
    sorted) here because list order is itself part of what the chat
    template renders — see the Gate CS9 section of the module docstring."""
    try:
        blob = json.dumps(
            tools or [], sort_keys=False, ensure_ascii=False, default=str,
        ).encode("utf-8")
    except Exception:
        blob = repr(tools).encode("utf-8", "replace")
    return hashlib.sha256(blob).hexdigest()


def _is_stale_locked(current_signature: Optional["BackendSignature"]) -> "tuple[bool, str]":
    """Caller must hold ``_state_lock``. BACKEND-IDENTITY staleness only —
    always evaluated immediately, never debounced (see
    :func:`_evaluate_tool_drift_locked` for the separately-debounced
    tool-schema-drift check, which is deliberately NOT part of this
    function). Returns ``(is_stale, reason)``.
    """
    if not _prime_valid or _last_primed_backend_signature is None or _last_primed_tool_fingerprint is None:
        return True, "never_primed"

    if current_signature is None:
        # Backend unreachable right now — can't confirm either way. Do NOT
        # flip an otherwise-valid prime to stale on a transient probe
        # failure; a genuinely down backend is handled by the bounded
        # health-wait inside prime_static_system_prompt itself once a
        # re-prime is actually attempted for some OTHER reason.
        return False, "backend_unreachable"

    last = _last_primed_backend_signature
    if (
        current_signature.config_fingerprint
        and last.config_fingerprint
        and current_signature.config_fingerprint != last.config_fingerprint
    ):
        return True, "backend_config_changed"

    # Reliability refinement: prefer the deterministic systemd instance
    # signal over the /slots task-counter heuristic whenever BOTH the last-
    # primed and current signatures have it — it closes the exact race a
    # /slots-only design cannot: a real request already reaching the new
    # process before this check gives it a normal/high id_task that looks
    # like "no restart happened" even though the process changed. An exact
    # match is definitive proof of the same instance (the /slots check is
    # skipped entirely in that case, since it can no longer be trusted once
    # traffic has landed); any difference is an immediate restart signal.
    if last.systemd_instance and current_signature.systemd_instance:
        if current_signature.systemd_instance != last.systemd_instance:
            return True, "backend_restarted"
        return False, "fresh"

    # Fallback: the /slots task-counter heuristic, used only when the
    # authoritative systemd signal isn't available on both sides (systemd
    # absent, the configured unit doesn't match, systemctl unavailable, or
    # a transient miss on just this one check).
    if last.task_high_water_mark is not None and current_signature.slots_reachable:
        # A LOWER value is the direct restart signal; a now-ABSENT id_task
        # (None) is the same signal in disguise -- direct observation
        # against this deployment shows a freshly restarted slot omits
        # id_task entirely until it processes its first request, so "the
        # counter we left it with is simply gone" can only mean a new
        # process (see BackendSignature's docstring) -- PROVIDED no request
        # has landed on it yet; this is exactly the case the systemd signal
        # above exists to cover once a request DOES land first.
        if (
            current_signature.task_high_water_mark is None
            or current_signature.task_high_water_mark < last.task_high_water_mark
        ):
            return True, "backend_restarted"

    return False, "fresh"


def _evaluate_tool_drift_locked(current_tool_fingerprint: Optional[str]) -> "tuple[bool, str, dict]":
    """Caller must hold ``_state_lock``. Reliability refinement: DEBOUNCED
    tool-schema-drift check. Unlike backend-identity staleness
    (:func:`_is_stale_locked`), a single differing observation is not
    enough — the identical candidate fingerprint must be observed on
    ``_TOOL_DRIFT_CONFIRM_THRESHOLD`` CONSECUTIVE calls before it counts as
    confirmed drift, and a confirmed drift additionally respects
    ``_TOOL_DRIFT_COOLDOWN_SECONDS`` after the triggered re-prime completes.
    Only called when :func:`_is_stale_locked` already returned
    ``(False, "fresh")`` this same check (a backend restart takes priority
    and is never debounced).

    Mutates ``_tool_drift_candidate_fingerprint``/``_count`` as a side
    effect on every call (tracking is meaningless without doing this on
    every observation, not just confirming ones) — safe because the caller
    always holds ``_state_lock``.

    Returns ``(confirmed, reason, diag)``. ``diag`` carries
    ``tool_drift_candidate``/``tool_drift_confirmation_count`` (and
    ``tool_drift_confirmed`` / cooldown fields when relevant) for the
    caller's diagnostic log line — populated only when there is something
    notable to report (a candidate is being tracked, confirmed, cleared, or
    cooldown-suppressed), never for the boring "still matches baseline"
    case.
    """
    global _tool_drift_candidate_fingerprint, _tool_drift_candidate_count

    if current_tool_fingerprint is None:
        # Tool resolution itself failed this check — don't let a transient
        # resolution failure masquerade as "tools changed"; leave any
        # existing candidate tracking untouched.
        return False, "tool_resolution_unavailable", {}

    if current_tool_fingerprint == _last_primed_tool_fingerprint:
        had_candidate = _tool_drift_candidate_count > 0
        _tool_drift_candidate_fingerprint = None
        _tool_drift_candidate_count = 0
        if had_candidate:
            return False, "tool_schema_drift_candidate_cleared", {
                "tool_drift_candidate": "-", "tool_drift_confirmation_count": 0,
            }
        return False, "fresh", {}

    # Differs from the currently-primed baseline.
    if current_tool_fingerprint == _tool_drift_candidate_fingerprint:
        _tool_drift_candidate_count += 1
    else:
        # A NEW candidate (whether there was none before, or the previous
        # candidate was a different fingerprint) — confirmation restarts.
        _tool_drift_candidate_fingerprint = current_tool_fingerprint
        _tool_drift_candidate_count = 1

    diag = {
        "tool_drift_candidate": current_tool_fingerprint[:16],
        "tool_drift_confirmation_count": _tool_drift_candidate_count,
    }

    if _tool_drift_candidate_count < _TOOL_DRIFT_CONFIRM_THRESHOLD:
        return False, "tool_schema_drift_candidate", diag

    now = time.monotonic()
    if _tool_drift_cooldown_until_monotonic is not None and now < _tool_drift_cooldown_until_monotonic:
        diag["tool_drift_cooldown_suppressed"] = True
        diag["tool_drift_cooldown_remaining_s"] = round(_tool_drift_cooldown_until_monotonic - now, 1)
        return False, "tool_schema_drift_cooldown_suppressed", diag

    diag["tool_drift_confirmed"] = True
    return True, "tool_schema_drift_confirmed", diag


def _record_prime_outcome(result: "PrimingResult", *, trigger_reason: str = "") -> None:
    """Gate CS10: update re-priming validity state from a just-completed
    prime attempt. Called after BOTH the original startup prime and any
    re-prime (backend-restart- or tool-drift-triggered), so the watchdog's
    very first check right after a successful startup prime sees an
    already-valid baseline instead of immediately judging it stale. Always
    clears ``_prime_in_progress``, even on failure — a failed re-prime must
    never leave the "a prime is running" flag stuck set, which would
    permanently block future retries, but it also must never mark the
    cache valid (see module docstring).

    Reliability refinement: *trigger_reason* (``""`` for the plain startup
    prime, or whatever :func:`_is_stale_locked` /
    :func:`_evaluate_tool_drift_locked` returned) lets this function start
    the post-tool-drift cooldown (:data:`_TOOL_DRIFT_COOLDOWN_SECONDS`,
    measured from THIS completion — see that constant's comment for why)
    only for a tool-drift-triggered attempt, and clear any pending
    tool-drift candidate tracking on any SUCCESSFUL prime regardless of
    trigger (the baseline moved, so stale candidate tracking from before
    would otherwise compare against a now-outdated fingerprint).
    """
    global _prime_valid, _last_primed_backend_signature, _last_primed_tool_fingerprint
    global _last_failed_reprime_monotonic, _prime_in_progress
    global _tool_drift_candidate_fingerprint, _tool_drift_candidate_count
    global _tool_drift_cooldown_until_monotonic
    try:
        with _state_lock:
            _prime_in_progress = False
            if trigger_reason.startswith("tool_schema_drift"):
                _tool_drift_cooldown_until_monotonic = time.monotonic() + _TOOL_DRIFT_COOLDOWN_SECONDS
            if result.succeeded and result.backend_config_fingerprint:
                _last_primed_backend_signature = BackendSignature(
                    config_fingerprint=result.backend_config_fingerprint,
                    task_high_water_mark=result.backend_task_high_water_mark,
                    slots_reachable=result.backend_slots_reachable,
                    systemd_instance=result.backend_systemd_instance,
                )
                _last_primed_tool_fingerprint = result.tool_fingerprint
                _prime_valid = True
                _last_failed_reprime_monotonic = None
                # The baseline just moved -- any pending tool-drift
                # candidate was tracked against the OLD baseline and is no
                # longer meaningful (whether it matched the new tools sent
                # in THIS prime or not, the next check starts fresh).
                _tool_drift_candidate_fingerprint = None
                _tool_drift_candidate_count = 0
            else:
                # Either the attempt failed outright, or it succeeded but we
                # couldn't confirm backend identity afterward (empty
                # config_fingerprint) — in both cases we have no reliable
                # baseline to trust, so stay stale rather than risk serving
                # a silently-invalid cache as "fine".
                _prime_valid = False
                _last_failed_reprime_monotonic = time.monotonic()
    except Exception:
        logger.debug("static-prompt priming: failed to record CS10 prime outcome", exc_info=True)


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

    Gate CS10: on success, also snapshots a :class:`BackendSignature` and a
    tool-schema fingerprint into the returned result (``backend_config_
    fingerprint``, ``backend_task_high_water_mark``, ``tool_fingerprint``)
    so a caller (:func:`check_and_reprime_if_stale`, via
    :func:`_record_prime_outcome`) can later tell whether THIS prime is
    still the one the backend actually has cached. This function itself
    stays a pure "do the priming work once" call — it does not read or
    write the module's re-priming validity state.
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

    # Gate CS10: fingerprint whatever we actually resolved, whether or not
    # priming ultimately succeeds, so a caller can see the intended tool
    # set even for a failed attempt (used only by our own log line below;
    # only a SUCCESSFUL attempt's fingerprint is ever recorded as a
    # validity baseline — see _record_prime_outcome).
    tool_fingerprint = _compute_tool_fingerprint(priming_tools)

    if not static_prompt:
        elapsed_ms = int(round((time.monotonic() - started) * 1000))
        return PrimingResult(
            attempted=True, succeeded=False, attempts=0, elapsed_ms=elapsed_ms,
            prompt_chars=0, prompt_tokens=None, cached_tokens=None,
            reason="empty_static_prompt", tool_fingerprint=tool_fingerprint,
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

            # Gate CS10: snapshot backend identity right after our own
            # successful completion, so the recorded baseline reflects THIS
            # prime specifically (the /slots id_task read here will be at
            # least as high as the task our own POST above just consumed).
            # Best-effort: an empty config_fingerprint here just means the
            # next staleness check has no reliable baseline and will
            # re-prime again rather than silently trusting a cache that
            # might already be gone.
            signature = _query_backend_signature(backend["base_url"])
            backend_config_fingerprint = signature.config_fingerprint if signature else ""
            backend_task_high_water_mark = signature.task_high_water_mark if signature else None
            backend_slots_reachable = signature.slots_reachable if signature else False
            backend_systemd_instance = signature.systemd_instance if signature else None

            logger.info(
                "static-prompt priming succeeded: attempt=%d elapsed_ms=%d "
                "prompt_chars=%d prompt_tokens=%s cached_tokens=%s tool_count=%d "
                "skills_fingerprint=%s endpoint_host=%s",
                attempt, elapsed_ms, len(static_prompt), prompt_tokens, cached_tokens,
                len(priming_tools), skills_fingerprint or "-",
                urlparse(backend["base_url"]).hostname,
            )
            log_prefix_diag(
                "PRIME", static_prompt, tools=priming_tools,
                extra={
                    "success": True, "stale": False, "reason": "ok",
                    "backend_config_fp": backend_config_fingerprint or "-",
                    "task_hwm": backend_task_high_water_mark
                    if backend_task_high_water_mark is not None else "-",
                    "prompt_tokens": prompt_tokens if prompt_tokens is not None else "-",
                    "cached_tokens": cached_tokens if cached_tokens is not None else "-",
                    "elapsed_ms": elapsed_ms,
                },
            )
            return PrimingResult(
                attempted=True, succeeded=True, attempts=attempt, elapsed_ms=elapsed_ms,
                prompt_chars=len(static_prompt), prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens, reason="ok",
                skills_fingerprint=skills_fingerprint,
                tool_fingerprint=tool_fingerprint,
                backend_config_fingerprint=backend_config_fingerprint,
                backend_task_high_water_mark=backend_task_high_water_mark,
                backend_slots_reachable=backend_slots_reachable,
                backend_systemd_instance=backend_systemd_instance,
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
    log_prefix_diag(
        "PRIME", static_prompt, tools=priming_tools,
        extra={"success": False, "stale": True, "reason": last_error, "elapsed_ms": elapsed_ms},
    )
    return PrimingResult(
        attempted=True, succeeded=False, attempts=max_attempts, elapsed_ms=elapsed_ms,
        prompt_chars=len(static_prompt), prompt_tokens=None, cached_tokens=None,
        reason=last_error, skills_fingerprint=skills_fingerprint,
        tool_fingerprint=tool_fingerprint,
    )


def _run_priming_in_background() -> None:
    """Gate 10S: the ONE background action that may touch the model at
    startup. Always records a result and always signals
    ``_priming_done_event`` in a ``finally`` — regardless of outcome — so
    :func:`wait_for_priming_ready` and :func:`get_last_priming_result` are
    reliable for every code path (success, failure, exception, or a
    same-process no-op for a non-local backend).

    Gate CS10: also marks ``_prime_in_progress`` for the duration (so the
    staleness watchdog never races the very first startup prime) and
    records the outcome via :func:`_record_prime_outcome`, establishing the
    initial validity baseline the watchdog compares future checks against.
    """
    global _last_priming_result, _prime_in_progress
    with _state_lock:
        _prime_in_progress = True
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
        _record_prime_outcome(result)
        _priming_done_event.set()


def _run_reprime_in_background(platform: str, trigger_reason: str) -> None:
    """Gate CS10: background body for a re-prime triggered by
    :func:`check_and_reprime_if_stale`. Reuses the exact same synchronous
    :func:`prime_static_system_prompt` call as the startup path — only the
    trigger differs. ``_prime_in_progress`` is already True (set by the
    caller before this thread was started); this only ever needs to clear
    it, via :func:`_record_prime_outcome`, on every exit path.

    *trigger_reason* is forwarded to :func:`_record_prime_outcome` so it can
    start the post-tool-drift cooldown only for a tool-drift-triggered
    attempt (see that function's docstring)."""
    global _last_priming_result
    started_at = time.time()
    try:
        result = prime_static_system_prompt(platform=platform)
    except Exception:
        logger.debug("static-prompt re-priming: background task failed", exc_info=True)
        result = PrimingResult(
            attempted=True, succeeded=False, attempts=0, elapsed_ms=None,
            prompt_chars=None, prompt_tokens=None, cached_tokens=None,
            reason="unexpected_exception",
        )
    completed_at = time.time()
    try:
        _last_priming_result = dataclasses.replace(
            result, started_at=started_at, completed_at=completed_at,
        )
    except Exception:
        _last_priming_result = result
    _record_prime_outcome(result, trigger_reason=trigger_reason)


def check_and_reprime_if_stale(*, platform: str = "telegram") -> bool:
    """Gate CS10: cheap (no inference call) staleness probe.

    Safe to call as often as a caller likes — it returns False immediately,
    doing nothing further, whenever: no local backend is configured, a
    prime is already running, or a re-prime attempt failed too recently
    (see ``_MIN_REPRIME_RETRY_INTERVAL_SECONDS``). Otherwise it computes the
    current :class:`BackendSignature` (``/props`` + ``/slots`` + the
    systemd instance signal, no inference) and the current tool-schema
    fingerprint (via the SAME :func:`agent.static_system_prompt.
    resolve_static_prompt_tools` a real turn calls — this repeats each
    tool's own availability ``check_fn`` at the same cost a real turn
    already pays, which is why this is polled on an interval rather than on
    every caller).

    Two independent checks, evaluated in this order:
      1. Backend-IDENTITY staleness (:func:`_is_stale_locked`) — ALWAYS
         immediate; a detected restart triggers a re-prime on this very
         call, no debouncing, and takes priority over tool-drift.
      2. Tool-schema-drift staleness (:func:`_evaluate_tool_drift_locked`)
         — only reached when the backend is NOT stale AND reachable;
         reliability refinement: DEBOUNCED (a candidate fingerprint must be
         confirmed on ``_TOOL_DRIFT_CONFIRM_THRESHOLD`` consecutive checks,
         and a confirmed drift then respects ``_TOOL_DRIFT_COOLDOWN_
         SECONDS``) so a flapping tool-availability integration cannot
         trigger continuous re-primes.

    When either check confirms staleness, it marks the existing prime
    invalid, starts exactly one background re-prime (the same bounded/
    best-effort :func:`prime_static_system_prompt` call the startup path
    uses), and returns True. The caller's own request is never blocked on
    it — this function itself never waits on the re-prime thread.
    """
    global _prime_in_progress, _tool_drift_candidate_fingerprint, _tool_drift_candidate_count

    backend = resolve_local_backend()
    if backend is None:
        return False

    with _state_lock:
        if _prime_in_progress:
            return False
        if (
            _last_failed_reprime_monotonic is not None
            and (time.monotonic() - _last_failed_reprime_monotonic) < _MIN_REPRIME_RETRY_INTERVAL_SECONDS
        ):
            return False

    current_signature = _query_backend_signature(backend["base_url"])
    try:
        from agent.static_system_prompt import resolve_static_prompt_tools

        current_tools = resolve_static_prompt_tools(platform=platform)
        current_tool_fingerprint = _compute_tool_fingerprint(current_tools)
    except Exception:
        logger.debug(
            "static-prompt priming: tool resolution failed during staleness check", exc_info=True,
        )
        current_tools = []
        current_tool_fingerprint = None

    diag_extra: dict = {}
    with _state_lock:
        if _prime_in_progress:
            return False

        stale, reason = _is_stale_locked(current_signature)
        if stale:
            # A backend restart invalidates any pending tool-drift
            # candidate — the re-prime about to run re-establishes a fresh
            # tool-fingerprint baseline anyway (see _record_prime_outcome).
            _tool_drift_candidate_fingerprint = None
            _tool_drift_candidate_count = 0
        elif reason == "fresh":
            # Backend confirmed unchanged and reachable -- only NOW is it
            # meaningful to evaluate (debounced) tool-schema drift.
            stale, reason, diag_extra = _evaluate_tool_drift_locked(current_tool_fingerprint)
        # else reason == "backend_unreachable" or "never_primed" (the
        # latter already set stale=True above) -- skip tool-drift entirely;
        # a transient backend probe failure shouldn't also spend a check on
        # a tool resolution whose result we can't correlate with anything.

        if not stale:
            if reason not in ("fresh", "backend_unreachable"):
                # Notable non-triggering event (candidate tracked/cleared/
                # cooldown-suppressed) -- worth a content-free diagnostic
                # line even though no re-prime fires.
                log_prefix_diag(
                    "STALE_CHECK", "", tools=current_tools,
                    extra={"stale": False, "reason": reason, **diag_extra},
                )
            return False
        _prime_in_progress = True

    log_prefix_diag(
        "STALE_CHECK", "", tools=current_tools,
        extra={
            "stale": True, "reason": reason,
            "backend_config_fp": (current_signature.config_fingerprint if current_signature else "-"),
            "task_hwm": (
                current_signature.task_high_water_mark
                if current_signature and current_signature.task_high_water_mark is not None
                else "-"
            ),
            "systemd_instance": (
                current_signature.systemd_instance if current_signature and current_signature.systemd_instance
                else "-"
            ),
            **diag_extra,
        },
    )
    try:
        thread = threading.Thread(
            target=_run_reprime_in_background,
            args=(platform, reason),
            name="hermes-static-prompt-reprime",
            daemon=True,
        )
        thread.start()
        return True
    except Exception:
        logger.debug("static-prompt re-priming: failed to schedule", exc_info=True)
        with _state_lock:
            _prime_in_progress = False
        return False


def _staleness_watchdog_tick() -> None:
    """Gate CS10: ``agent.periodic_scheduler`` callback body — one cheap
    staleness probe per interval, for the life of the gateway process.
    Never returns False (never self-cancels); the only things that stop it
    are process exit or :func:`reset_priming_state_for_tests`."""
    try:
        check_and_reprime_if_stale()
    except Exception:
        logger.debug("static-prompt priming: staleness watchdog tick failed", exc_info=True)


def schedule_startup_priming() -> bool:
    """Fire the priming once, off the startup critical path.

    Same daemon-thread rationale as
    :func:`agent.llm_backend_warmup.schedule_startup_warmup` — a plain
    thread has no dependency on whatever asyncio loop is current when
    ``start_gateway()`` calls this, and ``prime_static_system_prompt`` is
    already a self-contained synchronous call.

    Idempotent per process for the STARTUP prime. Returns True iff a
    priming thread was actually started this call, never raises.

    Gate CS10: also registers :func:`check_and_reprime_if_stale` as a
    recurring callback on the shared ``agent.periodic_scheduler`` thread,
    so a llama-server restart LATER in this same gateway lifetime — not
    just the very first startup prime — is still caught automatically.
    This is purely additive: a failure to register the watchdog (e.g. the
    scheduler module is unavailable) must never make this function report
    failure, since the startup prime above may already have launched
    successfully.
    """
    global _priming_already_scheduled, _watchdog_handle
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
    except Exception:
        logger.debug("static-prompt priming: failed to schedule", exc_info=True)
        return False

    try:
        from agent.periodic_scheduler import schedule

        _watchdog_handle = schedule(_staleness_watchdog_tick, _DEFAULT_STALENESS_POLL_INTERVAL_SECONDS)
    except Exception:
        logger.debug("static-prompt priming: failed to start CS10 staleness watchdog", exc_info=True)

    return True
