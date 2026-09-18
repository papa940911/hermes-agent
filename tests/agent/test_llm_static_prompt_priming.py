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

# Captured before any test can monkeypatch priming._query_systemd_instance_
# signature (the autouse _default_no_systemd fixture below replaces that
# module attribute for every test) -- the handful of tests that exercise
# the REAL implementation (mocking subprocess.run instead) call this saved
# reference directly rather than going through the patched attribute.
_REAL_QUERY_SYSTEMD_INSTANCE_SIGNATURE = priming._query_systemd_instance_signature


@pytest.fixture(autouse=True)
def _reset_guard():
    priming.reset_priming_state_for_tests()
    yield
    priming.reset_priming_state_for_tests()


@pytest.fixture(autouse=True)
def _default_no_systemd(monkeypatch):
    """Reliability refinement: tests must not depend on THIS test-running
    host's real llama-vulkan.service state (this repo's own dev host
    happens to run one). Default every test to "systemd unavailable" so
    the /slots-based fallback path is what existing/unrelated tests
    exercise; a test that specifically wants the systemd path overrides
    this itself with its own monkeypatch.setattr call."""
    monkeypatch.setattr(priming, "_query_systemd_instance_signature", lambda **_: None)
    yield


def _patch_systemd(monkeypatch, value):
    monkeypatch.setattr(priming, "_query_systemd_instance_signature", lambda **_: value)


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


def _patch_tools(monkeypatch, tools=None):
    monkeypatch.setattr(
        "agent.static_system_prompt.resolve_static_prompt_tools",
        lambda **_: tools if tools is not None else [],
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
    seen_tools: list = []
    _attempts = 0
    # Gate CS10: /props + /slots, for backend-identity signature tests.
    serve_props = True
    serve_slots = True
    props_response: dict = {
        "model_path": "/models/test.gguf",
        "model_ftype": "Q4_K - Medium",
        "build_info": "b1-fake",
        "total_slots": 1,
        "endpoint_slots": True,
    }
    slots_response: list = [{"id": 0, "id_task": 1}]


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
        if self.path == "/props":
            self.state.request_log.append(("GET", self.path))
            if not self.state.serve_props:
                self.send_response(404)
                self.end_headers()
                return
            resp = json.dumps(self.state.props_response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return
        if self.path == "/slots":
            self.state.request_log.append(("GET", self.path))
            if not self.state.serve_slots:
                self.send_response(501)
                self.end_headers()
                return
            resp = json.dumps(self.state.slots_response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
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
            self.state.seen_tools.append(payload.get("tools"))
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
    _FakeServerState.seen_tools = []
    _FakeServerState.behavior = "ok"
    _FakeServerState.health_behavior = "ok"
    _FakeServerState._attempts = 0
    _FakeServerState.serve_props = True
    _FakeServerState.serve_slots = True
    _FakeServerState.props_response = {
        "model_path": "/models/test.gguf",
        "model_ftype": "Q4_K - Medium",
        "build_info": "b1-fake",
        "total_slots": 1,
        "endpoint_slots": True,
    }
    _FakeServerState.slots_response = [{"id": 0, "id_task": 1}]
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


# ── Gate CS9: priming must send the same ``tools`` a real turn sends ─────
#
# Root cause (see /tmp/hermes-cold-start-root-cause-report.txt): a real turn
# always sends ``tools=agent.tools or None`` alongside the system prompt.
# llama.cpp's Qwen chat template renders an entire extra "# Tools" system
# block AHEAD of the caller-supplied system message whenever ``tools`` is
# non-empty, so a priming POST that omits ``tools`` primes a KV-cache prefix
# that no real tool-using turn can ever reuse -- live-measured at ~290s of
# unnecessary full-prompt prefill on the first real turn. These tests pin
# the fix: priming now resolves and sends the real tool schema list.


def test_priming_includes_resolved_tools_in_request(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    fake_tools = [{"type": "function", "function": {"name": "bowling_knowledge_answer", "parameters": {}}}]
    _patch_tools(monkeypatch, tools=fake_tools)
    priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert state.seen_tools == [fake_tools]


def test_priming_omits_tools_key_when_none_resolved(monkeypatch, fake_llama_server):
    """Matches the real request convention (``tools=agent.tools or None``):
    an empty resolution must not send a spurious empty ``tools: []``, which
    some chat templates treat differently from an absent/null field."""
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    _patch_tools(monkeypatch, tools=[])
    priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert state.seen_tools == [None]


def test_priming_tool_resolution_failure_does_not_block_priming(monkeypatch, fake_llama_server):
    """A tool-resolution failure must degrade to no-tools priming, never
    abort the whole (still-useful) system-prompt prime."""
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    monkeypatch.setattr(
        "agent.static_system_prompt.resolve_static_prompt_tools",
        lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert result.succeeded is True
    assert state.seen_tools == [None]


def test_priming_resolves_tools_for_the_primed_platform(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    seen_platforms = []

    def _fake_resolve(*, platform, **_):
        seen_platforms.append(platform)
        return []

    monkeypatch.setattr("agent.static_system_prompt.resolve_static_prompt_tools", _fake_resolve)
    priming.prime_static_system_prompt(
        health_timeout_seconds=5, request_timeout_seconds=5, platform="telegram",
    )
    assert seen_platforms == ["telegram"]


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


# ── Gate CS10: backend-restart + tool-drift re-priming ─────────────────
#
# Root cause (see the CS10 gate report): Gate CS9 fixed a content mismatch
# but priming still only ever ran ONCE per gateway process. A llama-server
# restart independent of the gateway wipes the KV cache but leaves the
# one-shot guard set, so the gateway never re-primed and the next real
# request paid full cold prefill. These tests pin Gate CS10's fix: a cheap
# BackendSignature (/props + /slots) and a tool-schema fingerprint are
# compared against the last successful prime on every staleness check, and
# either drifting triggers exactly one bounded re-prime.


def _wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _establish_baseline(monkeypatch, base_url, *, tools=None):
    """Run one successful prime and record it as the CS10 validity baseline,
    exactly as the real startup path (_run_priming_in_background) does."""
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    _patch_tools(monkeypatch, tools=tools if tools is not None else [])
    result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert result.succeeded is True
    priming._record_prime_outcome(result)
    assert priming._prime_valid is True
    return result


def test_cs10_initial_startup_prime_establishes_valid_baseline(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    result = _establish_baseline(monkeypatch, base_url)
    assert priming._last_primed_backend_signature is not None
    assert priming._last_primed_backend_signature.config_fingerprint == result.backend_config_fingerprint
    assert priming._last_primed_tool_fingerprint == result.tool_fingerprint


def test_cs10_same_backend_and_tools_does_not_reprime(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _establish_baseline(monkeypatch, base_url, tools=[])
    post_count_before = sum(1 for m, _ in state.request_log if m == "POST")

    scheduled = priming.check_and_reprime_if_stale(platform="telegram")

    assert scheduled is False
    post_count_after = sum(1 for m, _ in state.request_log if m == "POST")
    assert post_count_after == post_count_before  # no re-prime POST fired


def test_cs10_backend_identity_change_triggers_exactly_one_reprime(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _establish_baseline(monkeypatch, base_url, tools=[])
    post_count_before = sum(1 for m, _ in state.request_log if m == "POST")

    # Simulate an independent llama-server restart with the SAME model but a
    # different build (or: the /slots task counter having reset — either
    # alone is enough to flip config_fingerprint / task_high_water_mark).
    state.props_response = dict(state.props_response, build_info="b2-fake-after-restart")
    state.slots_response = [{"id": 0, "id_task": 1}]  # counter reset to a small value

    scheduled = priming.check_and_reprime_if_stale(platform="telegram")
    assert scheduled is True

    assert _wait_until(lambda: not priming._prime_in_progress)
    post_count_after = sum(1 for m, _ in state.request_log if m == "POST")
    assert post_count_after == post_count_before + 1  # exactly one re-prime
    assert priming._prime_valid is True  # the re-prime succeeded
    assert priming._last_primed_backend_signature.config_fingerprint != ""


def test_cs10_tool_schema_fingerprint_changes_on_parameter_drift(monkeypatch, fake_llama_server):
    tool_v1 = {"type": "function", "function": {"name": "search", "parameters": {"a": 1}}}
    tool_v2 = {"type": "function", "function": {"name": "search", "parameters": {"a": 1, "b": 2}}}
    fp1 = priming._compute_tool_fingerprint([tool_v1])
    fp2 = priming._compute_tool_fingerprint([tool_v2])
    fp1_again = priming._compute_tool_fingerprint([tool_v1])
    assert fp1 != fp2  # a parameter (not just the tool name) changed
    assert fp1 == fp1_again  # deterministic


def test_cs10_tool_schema_drift_triggers_exactly_one_reprime(monkeypatch, fake_llama_server):
    """Reliability refinement: tool drift is now DEBOUNCED -- the first
    observation of a changed fingerprint must NOT trigger a re-prime by
    itself (see the Issue-2 debounce test block below); the SAME candidate
    must be observed twice consecutively. This test pins that end-to-end:
    two checks, one re-prime."""
    base_url, state = fake_llama_server
    tool_v1 = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    tool_v2 = [{"type": "function", "function": {"name": "search", "parameters": {"extra": True}}}]
    _establish_baseline(monkeypatch, base_url, tools=tool_v1)
    post_count_before = sum(1 for m, _ in state.request_log if m == "POST")

    _patch_tools(monkeypatch, tools=tool_v2)  # same tool NAME, different schema

    first = priming.check_and_reprime_if_stale(platform="telegram")
    assert first is False  # candidate observation #1 -- not yet confirmed
    assert sum(1 for m, _ in state.request_log if m == "POST") == post_count_before

    second = priming.check_and_reprime_if_stale(platform="telegram")
    assert second is True  # candidate observation #2 -- confirmed

    assert _wait_until(lambda: not priming._prime_in_progress)
    post_count_after = sum(1 for m, _ in state.request_log if m == "POST")
    assert post_count_after == post_count_before + 1
    assert priming._last_primed_tool_fingerprint == priming._compute_tool_fingerprint(tool_v2)


def test_cs10_failed_reprime_leaves_state_stale(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _establish_baseline(monkeypatch, base_url, tools=[])

    # Trigger staleness via a BACKEND identity change (immediate, not
    # debounced) so this test isolates "failed re-prime" behavior from the
    # separate tool-drift confirmation mechanics.
    state.props_response = dict(state.props_response, build_info="b2-fake-after-restart")
    state.behavior = "http_error"

    scheduled = priming.check_and_reprime_if_stale(platform="telegram")
    assert scheduled is True
    # Default bounded retry backoff sums to ~1.5s (0.5s + 1.0s) across 3
    # attempts; give it real margin so this never flakes under load.
    assert _wait_until(lambda: not priming._prime_in_progress, timeout=8.0)

    assert priming._prime_valid is False  # must NOT be marked valid on failure
    assert priming._last_failed_reprime_monotonic is not None


def test_cs10_concurrent_stale_checks_do_not_launch_duplicate_primes(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _establish_baseline(monkeypatch, base_url, tools=[])
    post_count_before = sum(1 for m, _ in state.request_log if m == "POST")

    # Trigger a BACKEND identity change (immediate, not debounced -- see
    # the Issue-2 concurrency test below for the tool-drift-confirmed
    # equivalent) and fire the staleness check from two threads back-to-
    # back -- whichever loses the race must see _prime_in_progress already
    # set by the winner (under _state_lock) and refuse to launch a second
    # concurrent re-prime.
    state.props_response = dict(state.props_response, build_info="b2-fake-after-restart")

    results = []

    def _fire():
        results.append(priming.check_and_reprime_if_stale(platform="telegram"))

    t1 = threading.Thread(target=_fire)
    t2 = threading.Thread(target=_fire)
    t1.start()
    t2.start()
    t1.join(timeout=3)
    t2.join(timeout=3)

    assert _wait_until(lambda: not priming._prime_in_progress)
    assert sum(1 for r in results if r is True) == 1  # only one actually scheduled
    post_count_after = sum(1 for m, _ in state.request_log if m == "POST")
    assert post_count_after == post_count_before + 1  # exactly one re-prime POST


# ── Reliability refinement, Issue 2: tool-drift confirmation/debounce ──
#
# A single differing tool-fingerprint observation must NOT trigger a
# ~200s+ cold re-prime by itself -- tool availability check_fns are
# dynamic and can flap (observed live: a Home Assistant integration
# repeatedly connecting/disconnecting). These tests pin the state machine:
# 2 consecutive same-candidate observations to confirm, reset on reverting
# to baseline, restart confirmation on a different candidate, and a
# post-confirmation cooldown against continuous flapping.


def test_cs10_tool_drift_one_transient_change_does_not_reprime(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    tool_v1 = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    tool_v2 = [{"type": "function", "function": {"name": "search", "parameters": {"extra": True}}}]
    _establish_baseline(monkeypatch, base_url, tools=tool_v1)
    post_count_before = sum(1 for m, _ in state.request_log if m == "POST")

    _patch_tools(monkeypatch, tools=tool_v2)
    scheduled = priming.check_and_reprime_if_stale(platform="telegram")

    assert scheduled is False
    assert sum(1 for m, _ in state.request_log if m == "POST") == post_count_before
    assert priming._tool_drift_candidate_fingerprint == priming._compute_tool_fingerprint(tool_v2)
    assert priming._tool_drift_candidate_count == 1


def test_cs10_tool_drift_candidate_cleared_when_fingerprint_reverts(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    tool_v1 = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    tool_v2 = [{"type": "function", "function": {"name": "search", "parameters": {"extra": True}}}]
    _establish_baseline(monkeypatch, base_url, tools=tool_v1)

    _patch_tools(monkeypatch, tools=tool_v2)
    priming.check_and_reprime_if_stale(platform="telegram")
    assert priming._tool_drift_candidate_count == 1

    _patch_tools(monkeypatch, tools=tool_v1)  # back to the primed baseline
    scheduled = priming.check_and_reprime_if_stale(platform="telegram")

    assert scheduled is False
    assert priming._tool_drift_candidate_fingerprint is None
    assert priming._tool_drift_candidate_count == 0


def test_cs10_tool_drift_confirmed_on_two_consecutive_checks(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    tool_v1 = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    tool_v2 = [{"type": "function", "function": {"name": "search", "parameters": {"extra": True}}}]
    _establish_baseline(monkeypatch, base_url, tools=tool_v1)
    post_count_before = sum(1 for m, _ in state.request_log if m == "POST")

    _patch_tools(monkeypatch, tools=tool_v2)
    assert priming.check_and_reprime_if_stale(platform="telegram") is False
    assert priming.check_and_reprime_if_stale(platform="telegram") is True

    assert _wait_until(lambda: not priming._prime_in_progress)
    post_count_after = sum(1 for m, _ in state.request_log if m == "POST")
    assert post_count_after == post_count_before + 1  # exactly one re-prime


def test_cs10_tool_drift_candidate_switch_restarts_confirmation(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    tool_v1 = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    tool_a = [{"type": "function", "function": {"name": "search", "parameters": {"variant": "A"}}}]
    tool_b = [{"type": "function", "function": {"name": "search", "parameters": {"variant": "B"}}}]
    _establish_baseline(monkeypatch, base_url, tools=tool_v1)
    post_count_before = sum(1 for m, _ in state.request_log if m == "POST")

    _patch_tools(monkeypatch, tools=tool_a)
    assert priming.check_and_reprime_if_stale(platform="telegram") is False
    assert priming._tool_drift_candidate_count == 1

    _patch_tools(monkeypatch, tools=tool_b)  # a DIFFERENT candidate before A confirmed
    scheduled = priming.check_and_reprime_if_stale(platform="telegram")

    assert scheduled is False  # confirmation restarted for B, not yet confirmed
    assert priming._tool_drift_candidate_fingerprint == priming._compute_tool_fingerprint(tool_b)
    assert priming._tool_drift_candidate_count == 1
    assert sum(1 for m, _ in state.request_log if m == "POST") == post_count_before


def test_cs10_tool_drift_cooldown_prevents_prime_storm_from_flapping(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    tool_v1 = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    tool_v2 = [{"type": "function", "function": {"name": "search", "parameters": {"extra": True}}}]
    _establish_baseline(monkeypatch, base_url, tools=tool_v1)
    post_count_before = sum(1 for m, _ in state.request_log if m == "POST")

    _patch_tools(monkeypatch, tools=tool_v2)
    assert priming.check_and_reprime_if_stale(platform="telegram") is False
    assert priming.check_and_reprime_if_stale(platform="telegram") is True  # confirmed, re-prime #1
    assert _wait_until(lambda: not priming._prime_in_progress)
    assert sum(1 for m, _ in state.request_log if m == "POST") == post_count_before + 1

    # Flapping integration: bounce between baseline and the drifted set
    # repeatedly, re-confirming each time -- the cooldown must suppress
    # every one of these from becoming a second re-prime.
    for _ in range(4):
        _patch_tools(monkeypatch, tools=tool_v1)
        priming.check_and_reprime_if_stale(platform="telegram")
        _patch_tools(monkeypatch, tools=tool_v2)
        assert priming.check_and_reprime_if_stale(platform="telegram") is False
        assert priming.check_and_reprime_if_stale(platform="telegram") is False  # would confirm, but cooldown active

    assert sum(1 for m, _ in state.request_log if m == "POST") == post_count_before + 1  # still just one


def test_cs10_backend_restart_immediate_even_with_pending_tool_drift_candidate(monkeypatch, fake_llama_server):
    """Backend-restart detection must stay immediate regardless of tool-
    drift confirmation state -- a pending (unconfirmed) tool-drift
    candidate must not delay or interfere with an immediate backend-restart
    re-prime."""
    base_url, state = fake_llama_server
    tool_v1 = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    tool_v2 = [{"type": "function", "function": {"name": "search", "parameters": {"extra": True}}}]
    _establish_baseline(monkeypatch, base_url, tools=tool_v1)
    post_count_before = sum(1 for m, _ in state.request_log if m == "POST")

    # Start (but don't confirm) a tool-drift candidate.
    _patch_tools(monkeypatch, tools=tool_v2)
    assert priming.check_and_reprime_if_stale(platform="telegram") is False
    assert priming._tool_drift_candidate_count == 1

    # NOW a real backend restart happens -- must trigger immediately, on
    # this very call, without waiting for a second tool-drift observation.
    state.props_response = dict(state.props_response, build_info="b2-fake-after-restart")
    scheduled = priming.check_and_reprime_if_stale(platform="telegram")
    assert scheduled is True

    assert _wait_until(lambda: not priming._prime_in_progress)
    post_count_after = sum(1 for m, _ in state.request_log if m == "POST")
    assert post_count_after == post_count_before + 1  # exactly one re-prime, immediately
    assert priming._tool_drift_candidate_fingerprint is None  # cleared by the restart path


def test_cs10_diagnostic_logging_has_no_prompt_or_secret_content(monkeypatch, fake_llama_server, caplog):
    base_url, state = fake_llama_server
    sentinel_prompt = "SENTINEL-CS10-SYSTEM-PROMPT-SHOULD-NOT-APPEAR"
    sentinel_arg_value = "SENTINEL-CS10-TOOL-ARGUMENT-SHOULD-NOT-APPEAR"
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch, text=sentinel_prompt)
    _patch_tools(monkeypatch, tools=[
        {"type": "function", "function": {"name": "lookup", "parameters": {"q": sentinel_arg_value}}},
    ])
    with caplog.at_level("DEBUG"):
        result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    assert result.succeeded is True
    assert sentinel_prompt not in caplog.text
    assert sentinel_arg_value not in caplog.text
    assert "PREFIXDIAG" in caplog.text  # the facility did actually log


def test_cs10_prefixdiag_module_is_tracked_source_not_only_pyc():
    import agent._prefixdiag as prefixdiag

    assert inspect.getsourcefile(prefixdiag).endswith("_prefixdiag.py")


def test_cs10_backend_signature_query_reads_props_and_slots(fake_llama_server):
    base_url, state = fake_llama_server
    http_base = base_url[: -len("/v1")]
    signature = priming._query_backend_signature(http_base, timeout=3)
    assert signature is not None
    assert signature.config_fingerprint != ""
    assert signature.task_high_water_mark == 1
    assert signature.slots_reachable is True


def test_cs10_backend_signature_handles_disabled_slots_endpoint(fake_llama_server):
    base_url, state = fake_llama_server
    state.props_response = dict(state.props_response, endpoint_slots=False)
    http_base = base_url[: -len("/v1")]
    signature = priming._query_backend_signature(http_base, timeout=3)
    assert signature is not None
    assert signature.config_fingerprint != ""
    assert signature.task_high_water_mark is None  # documented limitation
    assert signature.slots_reachable is False


def test_cs10_backend_signature_returns_none_when_props_unreachable(fake_llama_server):
    base_url, state = fake_llama_server
    state.serve_props = False
    http_base = base_url[: -len("/v1")]
    signature = priming._query_backend_signature(http_base, timeout=3)
    assert signature is None


def test_cs10_backend_signature_fresh_process_omits_id_task(fake_llama_server):
    """Regression test for a real bug found during CS10 live validation:
    a freshly restarted llama-server's /slots response OMITS id_task
    entirely on a slot that has not yet processed any request in its new
    lifetime (confirmed live -- it does NOT report id_task=0). The query
    must still report slots_reachable=True so staleness detection can tell
    this apart from an unreachable /slots (see the next test)."""
    base_url, state = fake_llama_server
    state.slots_response = [{"id": 0, "n_ctx": 64000, "is_processing": False}]  # no id_task key
    http_base = base_url[: -len("/v1")]
    signature = priming._query_backend_signature(http_base, timeout=3)
    assert signature is not None
    assert signature.task_high_water_mark is None
    assert signature.slots_reachable is True


def test_cs10_restart_with_unchanged_config_detected_via_missing_id_task(monkeypatch, fake_llama_server):
    """Regression test for the exact live-validation finding: restarting
    llama-vulkan.service with an IDENTICAL model/build/config (so /props is
    byte-identical before and after) must still be detected as stale via
    the /slots id_task counter going from a known value to absent."""
    base_url, state = fake_llama_server
    _establish_baseline(monkeypatch, base_url, tools=[])
    assert priming._last_primed_backend_signature.task_high_water_mark == 1
    post_count_before = sum(1 for m, _ in state.request_log if m == "POST")

    # Simulate the restart: /props unchanged, /slots now has no id_task.
    state.slots_response = [{"id": 0, "n_ctx": 64000, "is_processing": False}]

    scheduled = priming.check_and_reprime_if_stale(platform="telegram")
    assert scheduled is True

    assert _wait_until(lambda: not priming._prime_in_progress)
    post_count_after = sum(1 for m, _ in state.request_log if m == "POST")
    assert post_count_after == post_count_before + 1
    assert priming._prime_valid is True


def test_cs10_is_stale_never_primed():
    priming.reset_priming_state_for_tests()
    stale, reason = priming._is_stale_locked(
        priming.BackendSignature(config_fingerprint="abc", task_high_water_mark=1),
    )
    assert stale is True
    assert reason == "never_primed"


def test_cs10_is_stale_treats_unreachable_backend_as_not_stale(monkeypatch):
    priming.reset_priming_state_for_tests()
    monkeypatch.setattr(priming, "_prime_valid", True)
    monkeypatch.setattr(
        priming, "_last_primed_backend_signature",
        priming.BackendSignature(config_fingerprint="abc", task_high_water_mark=5, slots_reachable=True),
    )
    monkeypatch.setattr(priming, "_last_primed_tool_fingerprint", "tool-fp")
    stale, reason = priming._is_stale_locked(None)
    assert stale is False
    assert reason == "backend_unreachable"


def test_cs10_is_stale_missing_id_task_on_reachable_slots_means_restarted(monkeypatch):
    """Direct unit-level pin of the live-validation regression fix: /slots
    reachable, config unchanged, but id_task went from a known value to
    absent -- must be treated as a restart, not "can't tell". (No systemd
    signal on either side here -- this is the /slots FALLBACK path.)"""
    priming.reset_priming_state_for_tests()
    monkeypatch.setattr(priming, "_prime_valid", True)
    monkeypatch.setattr(
        priming, "_last_primed_backend_signature",
        priming.BackendSignature(config_fingerprint="abc", task_high_water_mark=5, slots_reachable=True),
    )
    monkeypatch.setattr(priming, "_last_primed_tool_fingerprint", "tool-fp")
    current = priming.BackendSignature(config_fingerprint="abc", task_high_water_mark=None, slots_reachable=True)
    stale, reason = priming._is_stale_locked(current)
    assert stale is True
    assert reason == "backend_restarted"


def test_cs10_is_stale_slots_disabled_does_not_falsely_flag_restart(monkeypatch):
    """When /slots is disabled on both sides (slots_reachable=False) and
    there is no systemd signal either, the module has no restart signal to
    go on -- must not spuriously claim a restart just because
    task_high_water_mark is None on both sides."""
    priming.reset_priming_state_for_tests()
    monkeypatch.setattr(priming, "_prime_valid", True)
    monkeypatch.setattr(
        priming, "_last_primed_backend_signature",
        priming.BackendSignature(config_fingerprint="abc", task_high_water_mark=None, slots_reachable=False),
    )
    monkeypatch.setattr(priming, "_last_primed_tool_fingerprint", "tool-fp")
    current = priming.BackendSignature(config_fingerprint="abc", task_high_water_mark=None, slots_reachable=False)
    stale, reason = priming._is_stale_locked(current)
    assert stale is False
    assert reason == "fresh"


# ── Reliability refinement, Issue 1: systemd instance identity ─────────
#
# Live testing exposed a race in the /slots-only design: if a real request
# reaches a freshly restarted llama-server before the next watchdog check,
# /slots reports a normal/high id_task for the new process, and the
# task-counter comparison can no longer prove a restart happened. These
# tests pin the fix: MainPID + ActiveEnterTimestampMonotonic from
# `systemctl show llama-vulkan.service`, preferred over /slots whenever
# available on both sides.


def test_cs10_is_stale_same_systemd_instance_is_not_a_false_restart(monkeypatch):
    priming.reset_priming_state_for_tests()
    monkeypatch.setattr(priming, "_prime_valid", True)
    monkeypatch.setattr(
        priming, "_last_primed_backend_signature",
        priming.BackendSignature(
            config_fingerprint="abc", task_high_water_mark=5, slots_reachable=True,
            systemd_instance="12345:987654",
        ),
    )
    monkeypatch.setattr(priming, "_last_primed_tool_fingerprint", "tool-fp")
    current = priming.BackendSignature(
        config_fingerprint="abc", task_high_water_mark=9001, slots_reachable=True,
        systemd_instance="12345:987654",
    )
    stale, reason = priming._is_stale_locked(current)
    assert stale is False
    assert reason == "fresh"


def test_cs10_is_stale_changed_systemd_instance_detected_as_restart(monkeypatch):
    priming.reset_priming_state_for_tests()
    monkeypatch.setattr(priming, "_prime_valid", True)
    monkeypatch.setattr(
        priming, "_last_primed_backend_signature",
        priming.BackendSignature(
            config_fingerprint="abc", task_high_water_mark=5, slots_reachable=True,
            systemd_instance="12345:987654",
        ),
    )
    monkeypatch.setattr(priming, "_last_primed_tool_fingerprint", "tool-fp")
    current = priming.BackendSignature(
        config_fingerprint="abc", task_high_water_mark=6, slots_reachable=True,
        systemd_instance="67890:111222",
    )
    stale, reason = priming._is_stale_locked(current)
    assert stale is True
    assert reason == "backend_restarted"


def test_cs10_is_stale_systemd_wins_even_with_normal_high_id_task(monkeypatch):
    """The exact race this refinement closes: a real request already
    reached the NEW backend before this check, so /slots reports a
    normal/high id_task (higher than the old baseline, not reset) -- the
    /slots heuristic alone would say "not stale". The systemd instance
    signal must still detect the restart."""
    priming.reset_priming_state_for_tests()
    monkeypatch.setattr(priming, "_prime_valid", True)
    monkeypatch.setattr(
        priming, "_last_primed_backend_signature",
        priming.BackendSignature(
            config_fingerprint="abc", task_high_water_mark=5, slots_reachable=True,
            systemd_instance="12345:987654",
        ),
    )
    monkeypatch.setattr(priming, "_last_primed_tool_fingerprint", "tool-fp")
    current = priming.BackendSignature(
        config_fingerprint="abc", task_high_water_mark=42,  # normal/high, NOT reset
        slots_reachable=True, systemd_instance="99999:555555",  # different instance
    )
    stale, reason = priming._is_stale_locked(current)
    assert stale is True
    assert reason == "backend_restarted"


def test_cs10_is_stale_systemd_unavailable_falls_back_to_slots(monkeypatch):
    """When the systemd signal is unavailable on either side, the existing
    /slots-based fallback must remain fully functional (unchanged
    behavior)."""
    priming.reset_priming_state_for_tests()
    monkeypatch.setattr(priming, "_prime_valid", True)
    monkeypatch.setattr(
        priming, "_last_primed_backend_signature",
        priming.BackendSignature(
            config_fingerprint="abc", task_high_water_mark=5, slots_reachable=True,
            systemd_instance=None,  # systemd unavailable when this was primed
        ),
    )
    monkeypatch.setattr(priming, "_last_primed_tool_fingerprint", "tool-fp")
    current = priming.BackendSignature(
        config_fingerprint="abc", task_high_water_mark=2,  # lower -- /slots restart signal
        slots_reachable=True, systemd_instance=None,
    )
    stale, reason = priming._is_stale_locked(current)
    assert stale is True
    assert reason == "backend_restarted"


def test_cs10_query_systemd_instance_signature_uses_no_shell(monkeypatch):
    """Must invoke systemctl as an argv list, never shell=True."""
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        import subprocess as _sp

        return _sp.CompletedProcess(args, 0, stdout="MainPID=123\nActiveEnterTimestampMonotonic=456\n")

    monkeypatch.setattr(priming.subprocess, "run", fake_run)
    result = _REAL_QUERY_SYSTEMD_INSTANCE_SIGNATURE("llama-vulkan.service", timeout=1)
    assert result == "123:456"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert isinstance(args, list)
    assert kwargs.get("shell", False) is False
    assert "systemctl" in args


def test_cs10_query_systemd_instance_signature_none_on_missing_pid(monkeypatch):
    def fake_run(args, **kwargs):
        import subprocess as _sp

        return _sp.CompletedProcess(args, 0, stdout="MainPID=0\nActiveEnterTimestampMonotonic=0\n")

    monkeypatch.setattr(priming.subprocess, "run", fake_run)
    assert _REAL_QUERY_SYSTEMD_INSTANCE_SIGNATURE("llama-vulkan.service", timeout=1) is None


def test_cs10_query_systemd_instance_signature_none_when_systemctl_missing(monkeypatch):
    def fake_run(args, **kwargs):
        raise FileNotFoundError("systemctl not found")

    monkeypatch.setattr(priming.subprocess, "run", fake_run)
    assert _REAL_QUERY_SYSTEMD_INSTANCE_SIGNATURE("llama-vulkan.service", timeout=1) is None


def test_cs10_query_systemd_instance_signature_none_on_timeout(monkeypatch):
    def fake_run(args, **kwargs):
        raise priming.subprocess.TimeoutExpired(cmd=args, timeout=1)

    monkeypatch.setattr(priming.subprocess, "run", fake_run)
    assert _REAL_QUERY_SYSTEMD_INSTANCE_SIGNATURE("llama-vulkan.service", timeout=1) is None


def test_cs10_restart_detected_live_shaped_via_systemd_even_after_traffic(monkeypatch, fake_llama_server):
    """End-to-end (through check_and_reprime_if_stale): a restart with the
    new backend having ALREADY processed a request (/slots shows a normal,
    non-reset id_task) is still detected via the systemd instance signal."""
    base_url, state = fake_llama_server
    _patch_systemd(monkeypatch, "12345:1000000")
    _establish_baseline(monkeypatch, base_url, tools=[])
    assert priming._last_primed_backend_signature.systemd_instance == "12345:1000000"
    post_count_before = sum(1 for m, _ in state.request_log if m == "POST")

    # Simulate: backend restarted, already served another request --
    # /slots id_task is NORMAL/HIGH (not reset), config unchanged.
    state.slots_response = [{"id": 0, "id_task": 500}]
    _patch_systemd(monkeypatch, "67890:2000000")  # new MainPID/timestamp

    scheduled = priming.check_and_reprime_if_stale(platform="telegram")
    assert scheduled is True

    assert _wait_until(lambda: not priming._prime_in_progress)
    post_count_after = sum(1 for m, _ in state.request_log if m == "POST")
    assert post_count_after == post_count_before + 1
    assert priming._prime_valid is True


def test_cs10_schedule_startup_priming_registers_watchdog(monkeypatch):
    done = threading.Event()

    def fake_bg():
        done.set()

    monkeypatch.setattr(priming, "_run_priming_in_background", fake_bg)
    assert priming.schedule_startup_priming() is True
    assert done.wait(timeout=2)
    assert priming._watchdog_handle is not None


def test_cs10_schedule_startup_priming_survives_watchdog_registration_failure(monkeypatch):
    done = threading.Event()

    def fake_bg():
        done.set()

    monkeypatch.setattr(priming, "_run_priming_in_background", fake_bg)

    import agent.periodic_scheduler as periodic_scheduler

    def _boom(*a, **k):
        raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(periodic_scheduler, "schedule", _boom)
    assert priming.schedule_startup_priming() is True  # startup prime still "succeeds"
    assert done.wait(timeout=2)


def test_cs10_reset_cancels_watchdog_handle(monkeypatch):
    cancelled = {"n": 0}

    class _FakeHandle:
        def cancel(self):
            cancelled["n"] += 1

    priming._watchdog_handle = _FakeHandle()
    priming.reset_priming_state_for_tests()
    assert cancelled["n"] == 1
    assert priming._watchdog_handle is None


# ── Gate CS10.1: priming sends the EFFECTIVE (base + memory-provider) tools ─
#
# Root cause: a real AIAgent's tools array is base tools PLUS whatever the
# configured external memory provider contributes
# (agent_init.py -> agent.memory_manager.inject_memory_provider_tools).
# resolve_static_prompt_tools previously mirrored only the base half, so
# priming's tools array was silently shorter than a live turn's -- and since
# the Qwen chat template renders the tools block ahead of the system-prompt
# content, that length mismatch alone invalidated a large fraction of an
# otherwise byte-identical cached prefix (production f_sim_best=0.587; see
# /tmp/hermes_cs10_prefix_divergence_report.txt). These tests exercise the
# REAL agent.static_system_prompt.resolve_static_prompt_tools chain end to
# end through to the actual HTTP POST body -- only the base-tool resolver
# and the memory-provider's raw schema source are stubbed.


def _patch_base_tools(monkeypatch, tools):
    import agent.static_system_prompt as ssp

    monkeypatch.setattr(ssp, "_resolve_static_prompt_tool_defs", lambda **_: tools)


def _patch_memory_provider_schemas(monkeypatch, wrapped_schemas):
    """wrapped_schemas: list of {"type": "function", "function": {...}} --
    stubs the memory-provider SOURCE (not resolve_static_prompt_tools
    itself), so the real base+memory concatenation/dedup logic in
    agent.static_system_prompt runs for real."""
    def _fake(**kwargs):
        existing = kwargs.get("existing_tool_names") or set()
        return [w for w in wrapped_schemas if w["function"]["name"] not in existing]

    monkeypatch.setattr(
        "agent.memory_manager.resolve_configured_memory_provider_tool_schemas", _fake,
    )


def test_cs10_prime_request_includes_memory_provider_tools(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    _patch_base_tools(monkeypatch, [{"type": "function", "function": {"name": "terminal"}}])
    _patch_memory_provider_schemas(
        monkeypatch,
        [{"type": "function", "function": {"name": "fact_store", "description": "d", "parameters": {}}}],
    )

    result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)

    assert result.succeeded is True
    sent_tools = state.seen_tools[-1]
    sent_names = [t["function"]["name"] for t in sent_tools]
    assert sent_names == ["terminal", "fact_store"]  # base first, memory-provider appended


def test_cs10_prime_request_has_no_memory_provider_tools_when_none_configured(monkeypatch, fake_llama_server):
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    _patch_base_tools(monkeypatch, [{"type": "function", "function": {"name": "terminal"}}])
    _patch_memory_provider_schemas(monkeypatch, [])

    priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)

    sent_tools = state.seen_tools[-1]
    assert [t["function"]["name"] for t in sent_tools] == ["terminal"]


def test_cs10_tool_fingerprint_changes_when_memory_provider_schema_changes(monkeypatch, fake_llama_server):
    """Requirement E: a memory-provider schema change alters tools_sha256
    (via _compute_tool_fingerprint, the same hash check_and_reprime_if_stale
    watches) even though the base toolset is completely unchanged."""
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    _patch_base_tools(monkeypatch, [{"type": "function", "function": {"name": "terminal"}}])

    _patch_memory_provider_schemas(
        monkeypatch,
        [{"type": "function", "function": {"name": "fact_store", "parameters": {"v": 1}}}],
    )
    result_v1 = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)

    _patch_memory_provider_schemas(
        monkeypatch,
        [{"type": "function", "function": {"name": "fact_store", "parameters": {"v": 2}}}],
    )
    result_v2 = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)

    assert result_v1.succeeded is True and result_v2.succeeded is True
    assert result_v1.tool_fingerprint != result_v2.tool_fingerprint


def test_cs10_watchdog_fingerprint_reflects_memory_provider_tools_too(monkeypatch, fake_llama_server):
    """Requirement 5 (watchdog uses the SAME effective list): the tool
    fingerprint check_and_reprime_if_stale computes must be sensitive to a
    memory-provider-only tool change, exactly like a base-tool change."""
    base_url, state = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    _patch_static_prompt(monkeypatch)
    _patch_base_tools(monkeypatch, [{"type": "function", "function": {"name": "terminal"}}])

    _patch_memory_provider_schemas(
        monkeypatch,
        [{"type": "function", "function": {"name": "fact_store", "parameters": {}}}],
    )
    result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
    priming._record_prime_outcome(result)
    assert priming._prime_valid is True

    # Reliability refinement: tool drift is debounced -- one observation of
    # a NEW memory-provider-only fingerprint must not itself trigger a
    # re-prime; a second consecutive observation must (requirement F).
    _patch_memory_provider_schemas(
        monkeypatch,
        [{"type": "function", "function": {"name": "fact_store", "parameters": {"extra": True}}}],
    )
    first = priming.check_and_reprime_if_stale(platform="telegram")
    assert first is False
    second = priming.check_and_reprime_if_stale(platform="telegram")
    assert second is True
    assert _wait_until(lambda: not priming._prime_in_progress)
