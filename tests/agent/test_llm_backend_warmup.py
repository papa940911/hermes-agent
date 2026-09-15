"""Gate 10O: focused tests for agent/llm_backend_warmup.py.

Covers: loopback-only safety gating, bounded retry with no tight loop,
harmless handling of an unreachable/timeout/malformed backend, output
discarded (never logged), idempotent one-shot scheduling, and that a
warmup never touches Hermes session/memory/review/promotion state (by
construction — the warmup client never imports or references any of
those modules).
"""

from __future__ import annotations

import http.server
import json
import threading
import time

import pytest

from agent import llm_backend_warmup as warmup


@pytest.fixture(autouse=True)
def _reset_guard():
    warmup.reset_warmup_state_for_tests()
    yield
    warmup.reset_warmup_state_for_tests()


def _patch_config(monkeypatch, *, base_url, provider="custom", model="test-model"):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"model": {"base_url": base_url, "provider": provider, "default": model, "api_key": "none"}},
    )


# ── resolve_local_backend(): the safety gate ───────────────────────────────


def test_resolves_loopback_custom_provider(monkeypatch):
    _patch_config(monkeypatch, base_url="http://127.0.0.1:8087/v1")
    backend = warmup.resolve_local_backend()
    assert backend is not None
    assert backend["base_url"] == "http://127.0.0.1:8087/v1"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8087/v1",
        "http://127.0.0.1:8087/v1",
        "http://[::1]:8087/v1",
        "http://0.0.0.0:8087/v1",
    ],
)
def test_resolves_any_loopback_host_form(monkeypatch, base_url):
    _patch_config(monkeypatch, base_url=base_url)
    assert warmup.resolve_local_backend() is not None


def test_refuses_remote_base_url(monkeypatch):
    _patch_config(monkeypatch, base_url="https://api.openai.com/v1")
    assert warmup.resolve_local_backend() is None


def test_refuses_remote_lookalike_hostname(monkeypatch):
    # A hostname that merely contains "localhost" as a substring must not
    # be treated as loopback (host must equal a known loopback token).
    _patch_config(monkeypatch, base_url="https://localhost.evil.example.com/v1")
    assert warmup.resolve_local_backend() is None


def test_refuses_unknown_provider_even_on_loopback(monkeypatch):
    _patch_config(monkeypatch, base_url="http://127.0.0.1:8087/v1", provider="anthropic")
    assert warmup.resolve_local_backend() is None


def test_refuses_when_base_url_missing(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"model": {}})
    assert warmup.resolve_local_backend() is None


def test_resolve_never_raises_on_config_load_failure(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: (_ for _ in ()).throw(RuntimeError("disk error")),
    )
    assert warmup.resolve_local_backend() is None


def test_resolve_never_raises_on_malformed_base_url(monkeypatch):
    _patch_config(monkeypatch, base_url="not a url \x00 at all")
    assert warmup.resolve_local_backend() is None


# ── warm_local_llamacpp_backend(): a tiny fake llama.cpp server ────────────


class _FakeLlamaHandler(http.server.BaseHTTPRequestHandler):
    behavior = "ok"
    request_log: list = []

    def log_message(self, *a):
        pass  # keep test output quiet

    def do_POST(self):
        type(self).request_log.append(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if type(self).behavior == "ok":
            payload = json.loads(body)
            assert payload["max_tokens"] <= 8  # smallest possible inference
            assert payload["stream"] is False
            resp = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": "READY"}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        elif type(self).behavior == "malformed":
            resp = b"not json at all"
            self.send_response(200)
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
        elif type(self).behavior == "http_error":
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif type(self).behavior == "hang":
            time.sleep(5)
            self.send_response(200)
            self.end_headers()
        elif type(self).behavior == "fail_then_ok":
            attempts = getattr(type(self), "_attempts", 0) + 1
            type(self)._attempts = attempts
            if attempts < 2:
                self.send_response(500)
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                resp = json.dumps({"choices": [{"message": {"content": "READY"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)


@pytest.fixture
def fake_llama_server():
    _FakeLlamaHandler.request_log = []
    _FakeLlamaHandler.behavior = "ok"
    _FakeLlamaHandler._attempts = 0
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeLlamaHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1", _FakeLlamaHandler
    server.shutdown()
    thread.join(timeout=2)


def test_warmup_success_against_fake_backend(monkeypatch, fake_llama_server):
    base_url, handler = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    result = warmup.warm_local_llamacpp_backend(timeout_seconds=5)
    assert result.attempted is True
    assert result.succeeded is True
    assert result.attempts == 1
    assert result.elapsed_ms is not None and result.elapsed_ms >= 0
    assert handler.request_log == ["/v1/chat/completions"]


def test_warmup_request_payload_uses_minimal_inference(monkeypatch, fake_llama_server):
    base_url, handler = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url, model="qwen-test")
    warmup.warm_local_llamacpp_backend(timeout_seconds=5)
    assert handler.request_log  # assertions on payload happen in do_POST


def test_warmup_skips_entirely_for_remote_provider(monkeypatch, fake_llama_server):
    base_url, handler = fake_llama_server
    _patch_config(monkeypatch, base_url="https://api.example.com/v1")
    result = warmup.warm_local_llamacpp_backend(timeout_seconds=5)
    assert result.attempted is False
    assert result.reason == "no_local_backend_configured"
    assert handler.request_log == []  # never contacted


def test_warmup_handles_malformed_response_without_raising(monkeypatch, fake_llama_server):
    base_url, handler = fake_llama_server
    handler.behavior = "malformed"
    _patch_config(monkeypatch, base_url=base_url)
    # urlopen succeeds (200) even though body isn't valid JSON — warmup
    # discards the body outright, so this must still report success.
    result = warmup.warm_local_llamacpp_backend(timeout_seconds=5)
    assert result.succeeded is True


def test_warmup_handles_http_error_with_bounded_retry(monkeypatch, fake_llama_server):
    base_url, handler = fake_llama_server
    handler.behavior = "http_error"
    _patch_config(monkeypatch, base_url=base_url)
    result = warmup.warm_local_llamacpp_backend(
        max_attempts=2, backoff_seconds=(0.01, 0.01), timeout_seconds=2,
    )
    assert result.succeeded is False
    assert result.attempts == 2
    assert len(handler.request_log) == 2  # bounded, not a tight/infinite loop


def test_warmup_retries_are_bounded_not_unbounded(monkeypatch, fake_llama_server):
    base_url, handler = fake_llama_server
    handler.behavior = "http_error"
    _patch_config(monkeypatch, base_url=base_url)
    warmup.warm_local_llamacpp_backend(max_attempts=3, backoff_seconds=(0.01, 0.01, 0.01), timeout_seconds=2)
    assert len(handler.request_log) == 3


def test_warmup_recovers_on_transient_failure_then_success(monkeypatch, fake_llama_server):
    base_url, handler = fake_llama_server
    handler.behavior = "fail_then_ok"
    _patch_config(monkeypatch, base_url=base_url)
    result = warmup.warm_local_llamacpp_backend(
        max_attempts=3, backoff_seconds=(0.01, 0.01), timeout_seconds=2,
    )
    assert result.succeeded is True
    assert result.attempts == 2


def test_warmup_handles_connection_refused_harmlessly(monkeypatch):
    # No server listening on this port at all.
    _patch_config(monkeypatch, base_url="http://127.0.0.1:1/v1")
    result = warmup.warm_local_llamacpp_backend(
        max_attempts=1, backoff_seconds=(0.01,), timeout_seconds=1,
    )
    assert result.attempted is True
    assert result.succeeded is False
    assert result.reason  # some exception class name, never a raised exception


def test_warmup_handles_timeout_harmlessly(monkeypatch, fake_llama_server):
    base_url, handler = fake_llama_server
    handler.behavior = "hang"
    _patch_config(monkeypatch, base_url=base_url)
    started = time.monotonic()
    result = warmup.warm_local_llamacpp_backend(
        max_attempts=1, backoff_seconds=(0.01,), timeout_seconds=0.5,
    )
    elapsed = time.monotonic() - started
    assert result.succeeded is False
    assert elapsed < 3  # bounded — did not wait for the 5s hang


def test_warmup_never_raises_regardless_of_backend_state(monkeypatch, fake_llama_server):
    base_url, handler = fake_llama_server
    for behavior in ("ok", "malformed", "http_error", "hang"):
        handler.behavior = behavior
        _patch_config(monkeypatch, base_url=base_url)
        warmup.warm_local_llamacpp_backend(max_attempts=1, backoff_seconds=(0.01,), timeout_seconds=0.3)


def test_warmup_output_is_never_logged(monkeypatch, fake_llama_server, caplog):
    base_url, handler = fake_llama_server
    _patch_config(monkeypatch, base_url=base_url)
    with caplog.at_level("DEBUG"):
        warmup.warm_local_llamacpp_backend(timeout_seconds=5)
    assert "READY" not in caplog.text


def test_warmup_does_not_log_api_key(monkeypatch, fake_llama_server, caplog):
    base_url, handler = fake_llama_server
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"model": {"base_url": base_url, "provider": "custom", "default": "m", "api_key": "sk-super-secret-value"}},
    )
    with caplog.at_level("DEBUG"):
        warmup.warm_local_llamacpp_backend(timeout_seconds=5)
    assert "sk-super-secret-value" not in caplog.text


# ── schedule_startup_warmup(): idempotency, no runaway duplicates ─────────


def test_schedule_startup_warmup_fires_exactly_once(monkeypatch):
    calls = {"n": 0}
    done = threading.Event()

    def fake_bg():
        calls["n"] += 1
        done.set()

    monkeypatch.setattr(warmup, "_run_warmup_in_background", fake_bg)

    first = warmup.schedule_startup_warmup()
    second = warmup.schedule_startup_warmup()
    third = warmup.schedule_startup_warmup()
    assert done.wait(timeout=2)

    assert first is True
    assert second is False
    assert third is False
    assert calls["n"] == 1


def test_schedule_startup_warmup_runs_in_a_daemon_thread_not_asyncio(monkeypatch):
    # Must not depend on any running event loop at all — the whole point is
    # to be immune to asyncio task-lifecycle/loop-replacement fragility in
    # the surrounding gateway startup sequence.
    seen_thread = {}
    done = threading.Event()

    def fake_bg():
        seen_thread["is_daemon"] = threading.current_thread().daemon
        seen_thread["is_main"] = threading.current_thread() is threading.main_thread()
        done.set()

    monkeypatch.setattr(warmup, "_run_warmup_in_background", fake_bg)
    assert warmup.schedule_startup_warmup() is True
    assert done.wait(timeout=2)
    assert seen_thread["is_daemon"] is True
    assert seen_thread["is_main"] is False


def test_schedule_startup_warmup_repeated_service_restart_resets_guard(monkeypatch):
    done = threading.Event()

    def fake_bg():
        done.set()

    monkeypatch.setattr(warmup, "_run_warmup_in_background", fake_bg)

    assert warmup.schedule_startup_warmup() is True
    assert done.wait(timeout=2)

    warmup.reset_warmup_state_for_tests()  # simulate a fresh process (restart)
    done.clear()
    assert warmup.schedule_startup_warmup() is True
    assert done.wait(timeout=2)


def test_schedule_startup_warmup_never_raises_if_thread_start_fails(monkeypatch):
    class ExplodingThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(warmup.threading, "Thread", ExplodingThread)
    assert warmup.schedule_startup_warmup() is False


def test_run_warmup_in_background_swallows_exceptions(monkeypatch):
    monkeypatch.setattr(
        warmup, "warm_local_llamacpp_backend",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    warmup._run_warmup_in_background()  # must not raise


def test_warmup_module_never_imports_review_or_ledger_or_memory():
    import ast
    import inspect

    source = inspect.getsource(warmup)
    tree = ast.parse(source)
    forbidden_substrings = ("review", "ledger", "memory", "knowledge_db", "telegram")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            for name in names:
                if not name:
                    continue
                lowered = name.lower()
                assert not any(f in lowered for f in forbidden_substrings), name


def test_warmup_result_is_a_plain_dataclass_with_expected_fields():
    result = warmup.WarmupResult(
        attempted=True, succeeded=True, attempts=1, elapsed_ms=10, reason="ok"
    )
    assert result.attempted and result.succeeded
    assert result.attempts == 1
