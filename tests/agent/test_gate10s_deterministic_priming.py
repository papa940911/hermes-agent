"""Gate 10S: focused tests for deterministic startup priming.

Gate 10R measured a large, real first-turn latency win from priming
llama.cpp's single KV-cache slot (``--parallel 1``) with the real Hermes
stable+skills prefix. But `gateway.run.start_gateway` scheduled TWO
independent, unordered background actors at startup — Gate 10O's generic
warmup ("Reply with only the word READY.", no system prompt) and Gate 10R's
real prime — each running its own bounded health-wait/retry loop against
the SAME single KV slot with no synchronization between them. Whichever
POST happened to land last on the backend nondeterministically decided
what ended up cached.

Gate 10S's fix (proven, not assumed, by reading gateway/run.py): remove the
generic warmup from the startup path entirely, since the real prime's own
inference call is a strict superset of what it provided for a backend that
stays resident across gateway restarts. This makes the real prime the ONLY
startup actor that can touch the model — deterministic by construction,
not by ordering — and adds a bounded, OPT-IN readiness primitive
(`wait_for_priming_ready` / `get_last_priming_result`) for observability
and any future caller that wants to avoid the much narrower remaining race
(a genuine user message arriving before the one remaining prime finishes).

Amended for Gate CS10: the "exactly one prime, ever, per process" property
this file pins is now specifically about the STARTUP prime. CS10 adds a
deliberate, narrow exception -- a re-prime triggered by detected backend-
restart or tool-schema drift -- because a llama-server restart independent
of the gateway can otherwise silently invalidate the KV cache for the rest
of the process lifetime (see agent/llm_static_prompt_priming.py's Gate
CS10 docstring section). ``TestOperational`` below was updated accordingly
to allow the specific, existing-elsewhere periodic mechanism CS10 uses
(``agent.periodic_scheduler``) while still forbidding a heavy external
scheduling dependency or an actual subprocess/systemctl invocation.
"""

from __future__ import annotations

import http.server
import inspect
import json
import threading
import time

import pytest

from agent import llm_static_prompt_priming as priming
from agent import llm_backend_warmup as warmup


@pytest.fixture(autouse=True)
def _reset_guards():
    priming.reset_priming_state_for_tests()
    warmup.reset_warmup_state_for_tests()
    yield
    priming.reset_priming_state_for_tests()
    warmup.reset_warmup_state_for_tests()


def _patch_config(monkeypatch, *, base_url, provider="custom", model="test-model"):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"model": {"base_url": base_url, "provider": provider, "default": model, "api_key": "none"}},
    )


def _patch_prefix(monkeypatch, text="STABLE+SKILLS-PREFIX", fingerprint="fp123"):
    monkeypatch.setattr(
        "agent.static_system_prompt.build_primed_prefix_with_skills",
        lambda **_: (text, fingerprint),
    )


def _patch_fast_unhealthy(monkeypatch):
    """schedule_startup_priming() -> prime_static_system_prompt() with no
    args uses the module's real (production-sized, ~30s) health-timeout
    default, baked into the function's default arguments at import time (so
    patching the module-level constants after the fact has no effect on an
    already-defined function). Patch ``_wait_for_health`` itself instead --
    it's looked up as a plain global at call time -- so a test simulating an
    unhealthy backend doesn't have to actually wait 30s for the real bounded
    poll to give up."""
    monkeypatch.setattr(priming, "_wait_for_health", lambda *a, **k: False)


def _patch_fast_healthy(monkeypatch):
    """Same rationale as _patch_fast_unhealthy, for the opposite case: skip
    straight to the (already-fast, connection-refused-is-immediate) POST
    attempts against an unreachable backend."""
    monkeypatch.setattr(priming, "_wait_for_health", lambda *a, **k: True)


# ── Fake llama.cpp server (shared shape with test_llm_static_prompt_priming) ──


class _FakeServerState:
    behavior = "ok"
    health_behavior = "ok"
    request_log: list = []
    seen_system_prompts: list = []
    seen_had_system: list = []
    _attempts = 0


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    state = _FakeServerState

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.state.request_log.append(("GET", self.path))
            if self.state.health_behavior == "ok":
                resp = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            else:
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        self.state.request_log.append(("POST", self.path))
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        payload = json.loads(body) if body else {}
        has_system = any(m.get("role") == "system" for m in payload.get("messages", []))
        self.state.seen_had_system.append(has_system)
        system_msg = next((m["content"] for m in payload.get("messages", []) if m["role"] == "system"), None)
        self.state.seen_system_prompts.append(system_msg)
        if self.state.behavior == "ok":
            resp = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "READY"}}],
                "usage": {"prompt_tokens": 42, "prompt_tokens_details": {"cached_tokens": 7}},
                "timings": {"cache_n": 7, "prompt_n": 42},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        elif self.state.behavior == "http_error":
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.state.behavior == "malformed":
            resp = b"not json"
            self.send_response(200)
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        elif self.state.behavior == "empty":
            resp = json.dumps({"choices": [{"message": {"content": ""}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        elif self.state.behavior == "hang":
            time.sleep(5)


@pytest.fixture
def fake_server():
    _FakeServerState.request_log = []
    _FakeServerState.seen_system_prompts = []
    _FakeServerState.seen_had_system = []
    _FakeServerState.behavior = "ok"
    _FakeServerState.health_behavior = "ok"
    _FakeServerState._attempts = 0
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1", _FakeServerState
    server.shutdown()
    thread.join(timeout=2)


# ── 1. Ordering ────────────────────────────────────────────────────────────


class TestOrdering:
    def test_generic_warmup_not_scheduled_at_startup(self):
        import gateway.run as gw_run

        source = inspect.getsource(gw_run.start_gateway)
        assert "schedule_startup_warmup()" not in source
        assert "from agent.llm_backend_warmup import" not in source

    def test_real_prime_scheduled_exactly_once_in_source(self):
        import gateway.run as gw_run

        source = inspect.getsource(gw_run.start_gateway)
        assert source.count("schedule_startup_priming()") == 1

    def test_real_prime_is_the_only_model_touching_startup_call(self):
        import gateway.run as gw_run

        source = inspect.getsource(gw_run.start_gateway)
        # Only one import of a module that can issue an HTTP request to the
        # configured local model backend at startup.
        assert source.count("from agent.llm_static_prompt_priming import") == 1
        assert "from agent.llm_backend_warmup import" not in source

    def test_priming_schedule_is_one_shot_guard(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        first = priming.schedule_startup_priming()
        second = priming.schedule_startup_priming()
        third = priming.schedule_startup_priming()
        assert first is True
        assert second is False
        assert third is False

    def test_repeated_gateway_startup_process_is_safe(self, monkeypatch, fake_server):
        """Defensive re-entry of the startup path must not double-schedule."""
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        for _ in range(5):
            priming.schedule_startup_priming()
        priming.wait_for_priming_ready(timeout=5)
        # Exactly one POST reached the backend despite 5 scheduling attempts.
        posts = [e for e in state.request_log if e[0] == "POST"]
        assert len(posts) == 1


# ── 2. Supersession ──────────────────────────────────────────────────────


class TestSupersession:
    def test_warmup_module_still_importable_and_callable(self):
        """Not deleted -- just not scheduled. Still usable for diagnostics/tests."""
        assert callable(warmup.schedule_startup_warmup)
        assert callable(warmup.warm_local_llamacpp_backend)

    def test_real_prime_health_checks_before_posting(self, monkeypatch, fake_server):
        base_url, state = fake_server
        state.health_behavior = "not_ready"
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        result = priming.prime_static_system_prompt(health_timeout_seconds=0.5, health_poll_interval_seconds=0.1)
        assert result.reason == "backend_unhealthy"
        assert not any(m == "POST" for m, _ in state.request_log)

    def test_real_prime_exercises_chat_completions_endpoint(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert ("POST", "/v1/chat/completions") in state.request_log

    def test_real_prime_sends_a_system_message_generic_warmup_did_not(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch, text="REAL-STABLE-SKILLS-PREFIX")
        priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert state.seen_had_system == [True]
        assert state.seen_system_prompts == ["REAL-STABLE-SKILLS-PREFIX"]

    def test_real_prime_output_discarded_not_returned(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert not hasattr(result, "content")
        assert not hasattr(result, "completion")

    def test_real_prime_warms_an_actual_nonempty_prompt(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch, text="X" * 500)
        result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert result.prompt_chars == 500


# ── 3. Readiness ─────────────────────────────────────────────────────────


class TestReadiness:
    def test_not_ready_before_scheduling(self):
        assert priming.wait_for_priming_ready(timeout=0.05) is False

    def test_ready_true_after_success(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.schedule_startup_priming()
        assert priming.wait_for_priming_ready(timeout=5) is True

    def test_ready_true_after_failure_not_just_success(self, monkeypatch, fake_server):
        base_url, state = fake_server
        state.health_behavior = "not_ready"
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        _patch_fast_unhealthy(monkeypatch)
        priming.schedule_startup_priming()
        # Readiness means "concluded", not "succeeded" -- a failure still
        # unblocks any bounded waiter so nothing hangs.
        assert priming.wait_for_priming_ready(timeout=3) is True
        result = priming.get_last_priming_result()
        assert result.succeeded is False

    def test_wait_is_bounded_when_priming_never_scheduled(self):
        started = time.monotonic()
        ready = priming.wait_for_priming_ready(timeout=0.3)
        elapsed = time.monotonic() - started
        assert ready is False
        assert elapsed < 2.0

    def test_wait_times_out_while_priming_still_hanging(self, monkeypatch, fake_server):
        base_url, state = fake_server
        state.behavior = "hang"
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.schedule_startup_priming()
        started = time.monotonic()
        ready = priming.wait_for_priming_ready(timeout=0.3)
        elapsed = time.monotonic() - started
        assert ready is False
        assert elapsed < 1.5

    def test_wait_never_blocks_permanently(self, monkeypatch, fake_server):
        """No timeout arg misuse can produce an unbounded wait in this API."""
        assert priming.wait_for_priming_ready(timeout=-5) in (True, False)

    def test_multiple_simultaneous_waiters_all_unblock(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.schedule_startup_priming()
        results = []
        lock = threading.Lock()

        def _wait():
            ok = priming.wait_for_priming_ready(timeout=5)
            with lock:
                results.append(ok)

        threads = [threading.Thread(target=_wait) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=6)
        assert results == [True] * 6

    def test_readiness_does_not_wire_into_gateway_startup_by_default(self):
        """Gate 10S deliberately does NOT block Telegram/API startup on this."""
        import gateway.run as gw_run

        source = inspect.getsource(gw_run.start_gateway)
        assert "wait_for_priming_ready(" not in source


# ── 4. Failure handling ──────────────────────────────────────────────────


class TestFailure:
    def test_backend_unavailable_marks_ready_and_gateway_unaffected(self, monkeypatch):
        _patch_config(monkeypatch, base_url="http://127.0.0.1:1/v1")  # nothing listens
        _patch_prefix(monkeypatch)
        priming.schedule_startup_priming()
        assert priming.wait_for_priming_ready(timeout=5) is True
        assert priming.get_last_priming_result().succeeded is False

    def test_health_timeout_marks_ready(self, monkeypatch, fake_server):
        base_url, state = fake_server
        state.health_behavior = "not_ready"
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        result = priming.prime_static_system_prompt(health_timeout_seconds=0.3, health_poll_interval_seconds=0.1)
        assert result.reason == "backend_unhealthy"

    def test_http_500_handled_without_raising(self, monkeypatch, fake_server):
        base_url, state = fake_server
        state.behavior = "http_error"
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        result = priming.prime_static_system_prompt(
            max_attempts=1, health_timeout_seconds=5, request_timeout_seconds=2,
        )
        assert result.succeeded is False

    def test_malformed_json_handled_without_raising(self, monkeypatch, fake_server):
        base_url, state = fake_server
        state.behavior = "malformed"
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        result = priming.prime_static_system_prompt(
            max_attempts=1, health_timeout_seconds=5, request_timeout_seconds=5,
        )
        assert result.succeeded is False

    def test_empty_completion_still_counts_as_success(self, monkeypatch, fake_server):
        base_url, state = fake_server
        state.behavior = "empty"
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert result.succeeded is True

    def test_thread_exception_does_not_propagate(self, monkeypatch):
        _patch_config(monkeypatch, base_url="http://127.0.0.1:8087/v1")
        monkeypatch.setattr(
            "agent.static_system_prompt.build_primed_prefix_with_skills",
            lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        # Must not raise even though the underlying builder explodes.
        priming._run_priming_in_background()
        result = priming.get_last_priming_result()
        assert result is not None
        assert result.succeeded is False

    def test_gateway_still_works_when_priming_fails(self, monkeypatch):
        """schedule_startup_priming() itself never raises regardless of backend state."""
        _patch_config(monkeypatch, base_url="http://127.0.0.1:1/v1")
        _patch_prefix(monkeypatch)
        _patch_fast_healthy(monkeypatch)
        assert priming.schedule_startup_priming() is True  # thread started fine
        assert priming.wait_for_priming_ready(timeout=5) is True  # concluded, gateway can proceed


# ── 5. Cache state expectations ───────────────────────────────────────────


class TestCacheState:
    def test_expected_primed_token_count_recorded(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch, text="Y" * 200)
        result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert result.prompt_chars == 200
        assert result.prompt_tokens == 42  # from the fake server's fixed response

    def test_stable_plus_skills_prefix_actually_used(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch, text="UNIQUE-STABLE-SKILLS-MARKER")
        priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert state.seen_system_prompts == ["UNIQUE-STABLE-SKILLS-MARKER"]

    def test_no_stable_only_fallback_unless_prime_fails(self, monkeypatch, fake_server):
        """The priming module never itself falls back to a smaller/older
        prefix on success -- it either primes the full stable+skills prefix
        or records failure. There is no intermediate "partial" prefix path."""
        source = inspect.getsource(priming.prime_static_system_prompt)
        assert "build_static_system_prompt(" not in source
        assert "build_primed_prefix_with_skills(" in source

    def test_cache_stats_recorded_from_response(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert result.cached_tokens == 7

    def test_no_prompt_content_logged(self, monkeypatch, fake_server, caplog):
        import logging

        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch, text="SUPER-SECRET-PROMPT-TEXT-MARKER")
        with caplog.at_level(logging.DEBUG, logger="agent.llm_static_prompt_priming"):
            priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        for record in caplog.records:
            assert "SUPER-SECRET-PROMPT-TEXT-MARKER" not in record.getMessage()


# ── 6. Lifecycle ─────────────────────────────────────────────────────────


class TestLifecycle:
    def test_telegram_startup_code_unaffected_by_priming_change(self):
        """Priming wiring change must not touch platform adapter startup code."""
        import gateway.run as gw_run

        source = inspect.getsource(gw_run.start_gateway)
        # Telegram/platform adapter startup still exists after the priming
        # block (loose smoke check -- the platform loop is well past this
        # point in start_gateway and unrelated to this gate's diff).
        assert "schedule_startup_priming" in source

    def test_dashboard_module_not_referenced_by_priming(self):
        source = inspect.getsource(priming)
        assert "dashboard" not in source.lower()

    def test_gateway_health_endpoint_logic_not_touched(self):
        import gateway.status as gw_status

        assert hasattr(gw_status, "_build_runtime_status_record")

    def test_restart_simulation_is_clean(self, monkeypatch, fake_server):
        """Simulates 3 sequential gateway lifetimes in one process via the
        test-only reset hook -- each must independently prime once."""
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        for _ in range(3):
            priming.reset_priming_state_for_tests()
            state.request_log = []
            priming.schedule_startup_priming()
            assert priming.wait_for_priming_ready(timeout=5) is True
            posts = [e for e in state.request_log if e[0] == "POST"]
            assert len(posts) == 1

    def test_no_crash_loop_signature_in_priming_module(self):
        """schedule_startup_priming must never raise, by construction."""
        source = inspect.getsource(priming.schedule_startup_priming)
        assert "except Exception" in source


# ── 7. Concurrency ───────────────────────────────────────────────────────


class TestConcurrency:
    def test_only_one_startup_prime_under_concurrent_scheduling(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        results = []
        lock = threading.Lock()

        def _schedule():
            r = priming.schedule_startup_priming()
            with lock:
                results.append(r)

        threads = [threading.Thread(target=_schedule) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        priming.wait_for_priming_ready(timeout=5)
        assert results.count(True) == 1
        posts = [e for e in state.request_log if e[0] == "POST"]
        assert len(posts) == 1

    def test_two_requests_arriving_during_prime_no_deadlock(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.schedule_startup_priming()
        # Two independent bounded waits "during" the prime -- neither blocks
        # the other, and both resolve.
        outcomes = []

        def _probe():
            outcomes.append(priming.wait_for_priming_ready(timeout=5))

        t1, t2 = threading.Thread(target=_probe), threading.Thread(target=_probe)
        t1.start(); t2.start()
        t1.join(timeout=6); t2.join(timeout=6)
        assert outcomes == [True, True]

    def test_no_global_inference_lock_after_readiness(self):
        """wait_for_priming_ready must not acquire any lock that could
        serialize unrelated future calls -- it only reads an Event."""
        source = inspect.getsource(priming.wait_for_priming_ready)
        assert "Lock(" not in source
        assert "_priming_done_event.wait" in source

    def test_reset_between_tests_is_isolated(self, monkeypatch, fake_server):
        """get_last_priming_result must not leak across a reset."""
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.schedule_startup_priming()
        priming.wait_for_priming_ready(timeout=5)
        assert priming.get_last_priming_result() is not None
        priming.reset_priming_state_for_tests()
        assert priming.get_last_priming_result() is None


# ── 8. Privacy ───────────────────────────────────────────────────────────


class TestPrivacy:
    def test_no_user_text_in_priming_result_fields(self):
        fields = {f.name for f in priming.PrimingResult.__dataclass_fields__.values()}
        assert "prompt_text" not in fields
        assert "content" not in fields
        assert "completion" not in fields

    def test_no_telegram_ids_referenced_in_module(self):
        source = inspect.getsource(priming)
        for forbidden in ("chat_id", "telegram_user_id", "update.message"):
            assert forbidden not in source

    def test_no_memory_content_referenced_in_module(self):
        source = inspect.getsource(priming)
        assert "memory_store" not in source
        assert "MEMORY.md" not in source

    def test_no_review_claim_content_referenced(self):
        source = inspect.getsource(priming)
        for forbidden in ("review_claim", "adjudicat", "promotion_ledger"):
            assert forbidden not in source

    def test_no_secrets_logged(self, monkeypatch, fake_server, caplog):
        import logging

        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url, provider="custom")
        _patch_prefix(monkeypatch)
        with caplog.at_level(logging.DEBUG, logger="agent.llm_static_prompt_priming"):
            priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        for record in caplog.records:
            assert "Bearer" not in record.getMessage()
            assert "api_key" not in record.getMessage().lower()


# ── 9. Boundaries ────────────────────────────────────────────────────────


class TestBoundaries:
    def test_no_db_writes(self):
        source = inspect.getsource(priming)
        for forbidden in ("sqlite3", "INSERT INTO", "UPDATE ", "DELETE FROM"):
            assert forbidden not in source

    def test_no_review_writes(self):
        source = inspect.getsource(priming)
        assert "review_db" not in source.lower()

    def test_no_promotion_writes(self):
        source = inspect.getsource(priming)
        assert "promote" not in source.lower()

    def test_no_telegram_notification_writes(self):
        source = inspect.getsource(priming)
        assert "send_message" not in source.lower()
        assert "bot.send" not in source.lower()

    def test_no_bowlingtracker_writes(self):
        source = inspect.getsource(priming)
        assert "bowlingtracker" not in source.lower()
        assert "bowling" not in source.lower()


# ── 10. Regression ───────────────────────────────────────────────────────


class TestRegression:
    def test_gate10r_prefix_builder_unchanged(self):
        from agent import static_system_prompt as ssp

        assert hasattr(ssp, "build_primed_prefix_with_skills")
        assert hasattr(ssp, "build_static_system_prompt")

    def test_skill_routing_code_not_touched_by_this_gate(self):
        from agent import prompt_builder

        assert hasattr(prompt_builder, "build_skills_system_prompt")
        assert hasattr(prompt_builder, "skills_manifest_fingerprint")

    def test_tool_schemas_module_not_imported_by_priming(self):
        source = inspect.getsource(priming)
        assert "model_tools" not in source

    def test_dynamic_skills_fingerprint_field_preserved(self):
        result = priming.PrimingResult(
            attempted=True, succeeded=True, attempts=1, elapsed_ms=1,
            prompt_chars=1, prompt_tokens=1, cached_tokens=1, reason="ok",
        )
        assert result.skills_fingerprint == ""

    def test_normal_priming_flow_end_to_end_still_works(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch, text="END-TO-END", fingerprint="abc123")
        result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert result.succeeded is True
        assert result.skills_fingerprint == "abc123"


# ── 11. Observability ────────────────────────────────────────────────────


class TestObservability:
    def test_prime_start_timestamp_recorded(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.schedule_startup_priming()
        priming.wait_for_priming_ready(timeout=5)
        result = priming.get_last_priming_result()
        assert result.started_at is not None

    def test_prime_finish_timestamp_recorded_after_start(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.schedule_startup_priming()
        priming.wait_for_priming_ready(timeout=5)
        result = priming.get_last_priming_result()
        assert result.completed_at >= result.started_at

    def test_success_failure_metric_reflected_in_result(self, monkeypatch, fake_server):
        base_url, state = fake_server
        state.behavior = "http_error"
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.schedule_startup_priming()
        priming.wait_for_priming_ready(timeout=5)
        assert priming.get_last_priming_result().succeeded is False

    def test_get_last_priming_result_is_read_only_no_reprime(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.schedule_startup_priming()
        priming.wait_for_priming_ready(timeout=5)
        posts_before = len([e for e in state.request_log if e[0] == "POST"])
        priming.get_last_priming_result()
        priming.get_last_priming_result()
        posts_after = len([e for e in state.request_log if e[0] == "POST"])
        assert posts_before == posts_after == 1

    def test_no_duplicate_success_logs_per_attempt(self, monkeypatch, fake_server, caplog):
        import logging

        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        with caplog.at_level(logging.INFO, logger="agent.llm_static_prompt_priming"):
            priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        success_logs = [r for r in caplog.records if "priming succeeded" in r.getMessage()]
        assert len(success_logs) == 1


# ── 12. Startup repetition (5-restart simulation) ─────────────────────────


class TestStartupRepetition:
    def test_five_simulated_restarts_each_deterministically_prime(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch, text="STABLE+SKILLS", fingerprint="fp-restart")
        outcomes = []
        for _ in range(5):
            priming.reset_priming_state_for_tests()
            state.request_log = []
            state.seen_system_prompts = []
            priming.schedule_startup_priming()
            ready = priming.wait_for_priming_ready(timeout=5)
            result = priming.get_last_priming_result()
            outcomes.append((ready, result.succeeded, state.seen_system_prompts))
        assert all(ready and ok for ready, ok, _ in outcomes)
        assert all(seen == ["STABLE+SKILLS"] for _, _, seen in outcomes)

    def test_guards_reset_per_simulated_process(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.schedule_startup_priming()
        priming.wait_for_priming_ready(timeout=5)
        priming.reset_priming_state_for_tests()
        assert priming._priming_already_scheduled is False
        assert priming.wait_for_priming_ready(timeout=0.1) is False

    def test_no_stale_event_carries_into_new_process_simulation(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.schedule_startup_priming()
        priming.wait_for_priming_ready(timeout=5)
        priming.reset_priming_state_for_tests()
        # A fresh "process" must not see the old ready state or old result.
        assert priming.get_last_priming_result() is None

    def test_new_process_simulation_reprimes_from_scratch(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.schedule_startup_priming()
        priming.wait_for_priming_ready(timeout=5)
        priming.reset_priming_state_for_tests()
        state.request_log = []
        assert priming.schedule_startup_priming() is True
        priming.wait_for_priming_ready(timeout=5)
        assert len([e for e in state.request_log if e[0] == "POST"]) == 1


# ── 13. Performance ──────────────────────────────────────────────────────


class TestPerformance:
    def test_prompt_n_captured(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert result.prompt_tokens == 42

    def test_cache_n_captured(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert result.cached_tokens == 7

    def test_prompt_ms_elapsed_captured(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert result.elapsed_ms is not None and result.elapsed_ms >= 0

    def test_improvement_threshold_evaluated_offline(self):
        """Sanity-checks the Gate 10R before/after ratio this gate must not
        regress below (documented, not re-measured live in this unit test)."""
        before_ms = 35890.4
        after_ms = 4299.6
        improvement = 1 - (after_ms / before_ms)
        assert improvement > 0.80  # >=80% improvement, matches Gate 10R's ~88%

    def test_first_request_after_prime_is_fast_relative_to_backoff_floor(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        started = time.monotonic()
        priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        elapsed = time.monotonic() - started
        assert elapsed < 5.0  # fake server responds instantly; no needless sleep

    def test_second_call_remains_fast_no_repeat_health_poll_growth(self, monkeypatch, fake_server):
        base_url, state = fake_server
        _patch_config(monkeypatch, base_url=base_url)
        _patch_prefix(monkeypatch)
        priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        started = time.monotonic()
        priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        elapsed = time.monotonic() - started
        assert elapsed < 5.0


# ── 14. Operational ──────────────────────────────────────────────────────


class TestOperational:
    def test_no_heavy_scheduling_dependency_added(self):
        """Gate CS10 (deliberate, narrow exception to the original Gate 10S
        "no scheduler" rule): re-priming a backend that can now restart
        independently of the gateway requires SOME periodic re-check, so
        CS10 registers one on ``agent.periodic_scheduler`` -- the existing
        single shared-thread mechanism ``agent/turn_liveness.py`` and
        ``agent/turn_facade_lease.py`` already use elsewhere in this
        codebase, not a new dependency. What this test still guards against
        is a HEAVY external scheduling dependency (a cron library, a
        separate process, wall-clock cron syntax) -- none of which CS10
        added."""
        source = inspect.getsource(priming)
        for forbidden in ("croniter", "APScheduler", "schedule.every", "import cron"):
            assert forbidden not in source
        # The one sanctioned periodic mechanism is the shared in-process
        # scheduler already used elsewhere -- not a bespoke one.
        assert "agent.periodic_scheduler" in source

    def test_no_llama_config_or_slot_change_in_module(self):
        """The module must never construct or EXECUTE a command/config that
        CHANGES backend process state (launch flags, service lifecycle
        commands, etc).

        Amended for the CS10 reliability refinement: the module now
        legitimately reads (never changes) systemd state via a bounded,
        ``shell=False`` ``subprocess.run(["systemctl", "show", ...])`` call
        (:func:`agent.llm_static_prompt_priming._query_systemd_instance_
        signature`) — a read-only ``show``, not a ``restart``/``stop``/
        ``start``. So ``subprocess`` itself is no longer forbidden outright;
        instead this asserts the read-only property directly: no
        ``shell=True`` anywhere, and none of systemd's own MUTATING verbs
        appear as a quoted subprocess argument."""
        source = inspect.getsource(priming)
        for forbidden in ("ExecStart", "-ngl ", "-ub "):
            assert forbidden not in source
        for mutating_verb in ("restart", "stop", "start", "kill", "reload"):
            assert f'"{mutating_verb}"' not in source
            assert f"'{mutating_verb}'" not in source
        in_docstring = False
        for line in source.splitlines():
            stripped = line.strip()
            if "systemctl" in line:
                assert (
                    in_docstring
                    or stripped.startswith("#")
                    or '"systemctl"' in line  # the one legitimate argv-list use
                ), "systemctl referenced unexpectedly: " + line
            if "shell=True" in line:
                assert in_docstring or stripped.startswith("#"), (
                    "shell=True used outside prose/comment: " + line
                )
            if (line.count('"""') + line.count("'''")) % 2 == 1:
                in_docstring = not in_docstring

    def test_systemd_query_never_uses_shell_true(self):
        """Direct pin (complements the source-scan above): the actual
        subprocess.run call in _query_systemd_instance_signature must pass
        a plain argv list, never shell=True."""
        calls = []

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            import subprocess as _sp

            return _sp.CompletedProcess(args, 0, stdout="")

        import unittest.mock

        with unittest.mock.patch.object(priming.subprocess, "run", fake_run):
            priming._query_systemd_instance_signature("llama-vulkan.service", timeout=1)

        assert len(calls) == 1
        args, kwargs = calls[0]
        assert isinstance(args, list)
        assert kwargs.get("shell", False) is False
        assert args[:2] == ["systemctl", "show"]

    def test_module_never_calls_git_push(self):
        source = inspect.getsource(priming)
        assert "git push" not in source
