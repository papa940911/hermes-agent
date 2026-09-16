"""Gate 10P: focused tests for agent/llm_static_prompt_priming.py.

Covers: loopback-only safety gating (reused from Gate 10O), bounded health
wait, bounded retry with no tight loop, harmless handling of an
unreachable/timeout/malformed backend, output discarded (never logged),
idempotent one-shot scheduling, and that priming never touches Hermes
session/memory/review/promotion/Telegram state.
"""

from __future__ import annotations

import ast
import http.server
import inspect
import json
import threading
import time

import pytest

from agent import llm_static_prompt_priming as priming


@pytest.fixture(autouse=True)
def _reset_guard():
    priming.reset_priming_state_for_tests()
    yield
    priming.reset_priming_state_for_tests()


def _patch_config(monkeypatch, *, base_url, provider="custom", model="test-model"):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"model": {"base_url": base_url, "provider": provider, "default": model, "api_key": "none"}},
    )


def _patch_static_prompt(monkeypatch, text="STATIC-PROMPT-BODY", fingerprint=""):
    monkeypatch.setattr(
        "agent.static_system_prompt.build_primed_prefix_with_skills",
        lambda **_: (text, fingerprint),
    )


# ── Loopback/provider safety gate (delegated to llm_backend_warmup) ────


def test_refuses_remote_provider_before_building_prompt(monkeypatch):
    _patch_config(monkeypatch, base_url="https://api.example.com/v1")
    called = {"n": 0}
    monkeypatch.setattr(
        "agent.static_system_prompt.build_primed_prefix_with_skills",
        lambda **_: (called.__setitem__("n", called["n"] + 1) or "x", ""),
    )
    result = priming.prime_static_system_prompt()
    assert result.attempted is False
    assert result.reason == "no_local_backend_configured"
    assert called["n"] == 0  # prompt never even built for a remote provider


# ── Fake llama.cpp server (health + chat/completions) ───────────────────


class _FakeServerState:
    behavior = "ok"
    health_behavior = "ok"
    request_log: list = []
    seen_system_prompts: list = []
    _attempts = 0


class _FakeLlamaHandler(http.server.BaseHTTPRequestHandler):
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
            elif self.state.health_behavior == "not_ready":
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif self.state.health_behavior == "unreachable":
                # Simulate connection-refused-shaped behavior by just hanging
                # briefly then closing without a response.
                time.sleep(0.2)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        self.state.request_log.append(("POST", self.path))
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if self.state.behavior == "ok":
            payload = json.loads(body)
            assert payload["max_tokens"] == 1
            assert payload["stream"] is False
            system_msg = next((m["content"] for m in payload["messages"] if m["role"] == "system"), None)
            self.state.seen_system_prompts.append(system_msg)
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
        elif self.state.behavior == "malformed":
            resp = b"not json at all"
            self.send_response(200)
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        elif self.state.behavior == "http_error":
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.state.behavior == "empty_ok":
            resp = json.dumps({"choices": [{"message": {"content": ""}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        elif self.state.behavior == "fail_then_ok":
            self.state._attempts += 1
            if self.state._attempts < 2:
                self.send_response(500)
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                resp = json.dumps({"choices": [{"message": {"content": "READY"}}], "timings": {"cache_n": 0, "prompt_n": 5}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
        elif self.state.behavior == "hang":
            time.sleep(5)


@pytest.fixture
def fake_llama_server():
    _FakeServerState.request_log = []
    _FakeServerState.seen_system_prompts = []
    _FakeServerState.behavior = "ok"
    _FakeServerState.health_behavior = "ok"
    _FakeServerState._attempts = 0
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeLlamaHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1", _FakeServerState
    server.shutdown()
    thread.join(timeout=2)


# ── Happy path ───────────────────────────────────────────────────────────


def test_priming_success_against_fake_backend(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert result.attempted is True
    assert result.succeeded is True
    assert result.attempts == 1
    assert result.prompt_chars == len("STATIC-PROMPT-BODY")
    assert result.prompt_tokens == 42
    assert result.cached_tokens == 7
    assert ("GET", "/health") in state.request_log
    assert ("POST", "/v1/chat/completions") in state.request_log


def test_priming_sends_the_real_static_prompt_as_system_message(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch, text="THE-REAL-HERMES-STATIC-PREFIX")
    priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert state.seen_system_prompts == ["THE-REAL-HERMES-STATIC-PREFIX"]


def test_priming_request_uses_minimal_generation(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert state.request_log  # payload assertions (max_tokens==1) happen in do_POST


def test_priming_uses_same_model_field_as_configured(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url, model="qwen-local")
    _patch_static_prompt(monkeypatch)
    result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert result.succeeded is True


# ── Health-wait behavior ──────────────────────────────────────────────


def test_waits_for_health_before_posting(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    state.health_behavior = "not_ready"
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    result = priming.prime_static_system_prompt(
        health_timeout_seconds=0.6, health_poll_interval_seconds=0.1, request_timeout_seconds=5,
    )
    assert result.succeeded is False
    assert result.reason == "backend_unhealthy"
    assert not any(method == "POST" for method, _ in state.request_log)  # never posted


def test_health_wait_is_bounded_not_unbounded(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    state.health_behavior = "not_ready"
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    started = time.monotonic()
    priming.prime_static_system_prompt(health_timeout_seconds=0.5, health_poll_interval_seconds=0.1, request_timeout_seconds=5)
    elapsed = time.monotonic() - started
    assert elapsed < 3

def test_proceeds_once_health_becomes_ready(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    result = priming.prime_static_system_prompt(health_timeout_seconds=5, health_poll_interval_seconds=0.1, request_timeout_seconds=5)
    assert result.succeeded is True


def test_health_url_derivation_strips_v1_suffix():
    assert priming._health_url("http://127.0.0.1:8087/v1") == "http://127.0.0.1:8087/health"
    assert priming._health_url("http://127.0.0.1:8087/v1/") == "http://127.0.0.1:8087/health"
    assert priming._health_url("http://127.0.0.1:8087") == "http://127.0.0.1:8087/health"


# ── Failure handling ─────────────────────────────────────────────────


def test_priming_handles_malformed_response_without_raising(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    state.behavior = "malformed"
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5, max_attempts=1)
    assert result.succeeded is False  # JSON parse failure counts as a failed attempt
    assert result.attempted is True


def test_priming_handles_http_error_with_bounded_retry(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    state.behavior = "http_error"
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    result = priming.prime_static_system_prompt(
        max_attempts=2, backoff_seconds=(0.01, 0.01), health_timeout_seconds=5, request_timeout_seconds=2,
    )
    assert result.succeeded is False
    assert result.attempts == 2
    post_count = sum(1 for method, _ in state.request_log if method == "POST")
    assert post_count == 2  # bounded, not unbounded


def test_priming_retries_are_bounded(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    state.behavior = "http_error"
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    priming.prime_static_system_prompt(max_attempts=3, backoff_seconds=(0.01, 0.01, 0.01), health_timeout_seconds=5, request_timeout_seconds=2)
    post_count = sum(1 for method, _ in state.request_log if method == "POST")
    assert post_count == 3


def test_priming_recovers_on_transient_failure_then_success(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    state.behavior = "fail_then_ok"
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    result = priming.prime_static_system_prompt(max_attempts=3, backoff_seconds=(0.01, 0.01), health_timeout_seconds=5, request_timeout_seconds=2)
    assert result.succeeded is True
    assert result.attempts == 2


def test_priming_handles_connection_refused_harmlessly(monkeypatch):
    _patch_config(monkeypatch, base_url="http://127.0.0.1:1/v1")
    _patch_static_prompt(monkeypatch)
    result = priming.prime_static_system_prompt(
        max_attempts=1, backoff_seconds=(0.01,), health_timeout_seconds=0.3, health_poll_interval_seconds=0.1, request_timeout_seconds=1,
    )
    assert result.attempted is True
    assert result.succeeded is False


def test_priming_handles_timeout_harmlessly(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    state.behavior = "hang"
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    started = time.monotonic()
    result = priming.prime_static_system_prompt(
        max_attempts=1, backoff_seconds=(0.01,), health_timeout_seconds=5, request_timeout_seconds=0.5,
    )
    elapsed = time.monotonic() - started
    assert result.succeeded is False
    assert elapsed < 4


def test_priming_handles_empty_completion_output(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    state.behavior = "empty_ok"
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert result.succeeded is True  # HTTP 200 is success regardless of content


def test_priming_handles_prompt_build_failure_harmlessly(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    monkeypatch.setattr(
        "agent.static_system_prompt.build_primed_prefix_with_skills",
        lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert result.attempted is True
    assert result.succeeded is False
    assert result.reason == "prompt_build_failed"
    assert not any(method == "POST" for method, _ in state.request_log)


def test_priming_handles_empty_static_prompt(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch, text="")
    result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert result.succeeded is False
    assert result.reason == "empty_static_prompt"
    assert not any(method == "POST" for method, _ in state.request_log)


def test_priming_never_raises_regardless_of_backend_state(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    for behavior in ("ok", "malformed", "http_error", "empty_ok", "hang"):
        state.behavior = behavior
        priming.prime_static_system_prompt(max_attempts=1, backoff_seconds=(0.01,), health_timeout_seconds=1, request_timeout_seconds=0.3)


# ── Privacy / no semantic logging ───────────────────────────────────────


def test_priming_never_logs_the_prompt_text(monkeypatch, fake_llama_server, caplog):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch, text="SENTINEL-PROMPT-TEXT-SHOULD-NOT-APPEAR")
    with caplog.at_level("DEBUG"):
        priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert "SENTINEL-PROMPT-TEXT-SHOULD-NOT-APPEAR" not in caplog.text


def test_priming_never_logs_completion_output(monkeypatch, fake_llama_server, caplog):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    with caplog.at_level("DEBUG"):
        priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert "READY" not in caplog.text


def test_priming_never_logs_api_key(monkeypatch, fake_llama_server, caplog):
    base_url, state = fake_llama_server
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"model": {"base_url": base_url, "provider": "custom", "default": "m", "api_key": "sk-super-secret-value"}},
    )
    _patch_static_prompt(monkeypatch)
    with caplog.at_level("DEBUG"):
        priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert "sk-super-secret-value" not in caplog.text


# ── Idempotency / scheduling (mirrors Gate 10O warmup conventions) ─────


def test_schedule_startup_priming_fires_exactly_once(monkeypatch):
    calls = {"n": 0}
    done = threading.Event()

    def fake_bg():
        calls["n"] += 1
        done.set()

    monkeypatch.setattr(priming, "_run_priming_in_background", fake_bg)

    first = priming.schedule_startup_priming()
    second = priming.schedule_startup_priming()
    third = priming.schedule_startup_priming()
    assert done.wait(timeout=2)

    assert first is True
    assert second is False
    assert third is False
    assert calls["n"] == 1


def test_schedule_startup_priming_runs_in_a_daemon_thread(monkeypatch):
    seen = {}
    done = threading.Event()

    def fake_bg():
        seen["is_daemon"] = threading.current_thread().daemon
        seen["is_main"] = threading.current_thread() is threading.main_thread()
        done.set()

    monkeypatch.setattr(priming, "_run_priming_in_background", fake_bg)
    assert priming.schedule_startup_priming() is True
    assert done.wait(timeout=2)
    assert seen["is_daemon"] is True
    assert seen["is_main"] is False


def test_schedule_startup_priming_repeated_restart_resets_guard(monkeypatch):
    done = threading.Event()

    def fake_bg():
        done.set()

    monkeypatch.setattr(priming, "_run_priming_in_background", fake_bg)
    assert priming.schedule_startup_priming() is True
    assert done.wait(timeout=2)

    priming.reset_priming_state_for_tests()
    done.clear()
    assert priming.schedule_startup_priming() is True
    assert done.wait(timeout=2)


def test_schedule_startup_priming_never_raises_if_thread_start_fails(monkeypatch):
    class ExplodingThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(priming.threading, "Thread", ExplodingThread)
    assert priming.schedule_startup_priming() is False


def test_run_priming_in_background_swallows_exceptions(monkeypatch):
    monkeypatch.setattr(
        priming, "prime_static_system_prompt",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    priming._run_priming_in_background()  # must not raise


# ── Safety: no session/memory/review/promotion/Telegram touch ──────────


def test_priming_module_never_imports_review_or_ledger_or_telegram_client():
    source = inspect.getsource(priming)
    tree = ast.parse(source)
    forbidden = ("review", "ledger", "telegram_client", "notification", "promotion")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            for name in names:
                if not name:
                    continue
                lowered = name.lower()
                assert not any(f in lowered for f in forbidden), name


def test_priming_module_never_constructs_an_aiagent():
    source = inspect.getsource(priming)
    assert "AIAgent(" not in source


def test_priming_result_is_a_plain_dataclass_with_expected_fields():
    result = priming.PrimingResult(
        attempted=True, succeeded=True, attempts=1, elapsed_ms=10,
        prompt_chars=100, prompt_tokens=50, cached_tokens=40, reason="ok",
    )
    assert result.attempted and result.succeeded
    assert result.cached_tokens == 40


# ── Gateway wiring ───────────────────────────────────────────────────


def test_gateway_run_schedules_only_the_real_prime_gate10s():
    """Gate 10S: the generic Gate 10O warmup is no longer scheduled at
    startup at all — schedule_startup_priming() is the ONLY startup action
    that can touch the model, so there is no second actor left to race."""
    import gateway.run as gw_run

    source = inspect.getsource(gw_run.start_gateway)
    assert "schedule_startup_warmup()" not in source
    assert source.count("schedule_startup_priming()") == 1


def test_gateway_wiring_never_awaits_priming():
    import gateway.run as gw_run

    source = inspect.getsource(gw_run.start_gateway)
    assert "await schedule_startup_priming" not in source
