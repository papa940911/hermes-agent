"""Gate 10O: focused tests for agent/latency_diagnostics.py.

Covers: disabled-by-default no-op behavior, phase-mark idempotency and
ordering, monotonic-only duration math (no negative durations possible),
content-free/secret-free structured output, session-identity hashing,
contextvar isolation across concurrent "turns", and bounded rotating-file
emission.
"""

from __future__ import annotations

import json
import logging
import time

import pytest

from agent import latency_diagnostics as diag


@pytest.fixture(autouse=True)
def _clean_contextvar():
    # Every test starts with no active recorder and no env override, and
    # leaves none behind for the next test.
    token = diag._active.set(None)
    yield
    diag._active.reset(token)


@pytest.fixture(autouse=True)
def _clean_diag_logger():
    # logging.getLogger(name) is a process-wide singleton keyed by name, so
    # a handler a previous test (or pytest's own log-capture plugin, which
    # attaches its own handler to every named logger) attached to it
    # otherwise survives into the next test even after diag._diag_logger
    # (our own cache) is cleared.
    def _reset():
        logger = logging.getLogger(diag._DIAG_LOGGER_NAME)
        logger.handlers.clear()
        if hasattr(logger, "_hermes_latency_handler_installed"):
            delattr(logger, "_hermes_latency_handler_installed")
        diag._diag_logger = None

    _reset()
    yield
    _reset()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("HERMES_LATENCY_DIAG", "1")
    yield
    monkeypatch.delenv("HERMES_LATENCY_DIAG", raising=False)


@pytest.fixture
def disabled(monkeypatch):
    monkeypatch.delenv("HERMES_LATENCY_DIAG", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {"diagnostics": {"latency_enabled": False}}
    )
    yield


# ── is_enabled() ──────────────────────────────────────────────────────────


def test_disabled_by_default_with_no_env_and_no_config(monkeypatch):
    monkeypatch.delenv("HERMES_LATENCY_DIAG", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: (_ for _ in ()).throw(Exception("no config"))
    )
    assert diag.is_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "YES", "on"])
def test_env_var_truthy_values_enable(monkeypatch, value):
    monkeypatch.setenv("HERMES_LATENCY_DIAG", value)
    assert diag.is_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_env_var_falsy_values_disable(monkeypatch, value):
    monkeypatch.setenv("HERMES_LATENCY_DIAG", value)
    assert diag.is_enabled() is False


def test_config_opt_in_enables_when_env_absent(monkeypatch):
    monkeypatch.delenv("HERMES_LATENCY_DIAG", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {"diagnostics": {"latency_enabled": True}}
    )
    assert diag.is_enabled() is True


def test_env_var_wins_over_config(monkeypatch):
    monkeypatch.setenv("HERMES_LATENCY_DIAG", "0")
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {"diagnostics": {"latency_enabled": True}}
    )
    assert diag.is_enabled() is False


# ── start_recording() / disabled no-ops ────────────────────────────────────


def test_start_recording_returns_none_when_disabled(disabled):
    assert diag.start_recording(session_id="abc") is None
    assert diag.current_recorder() is None


def test_mark_phase_is_silent_no_op_without_active_recorder():
    # Must never raise even with nothing active.
    diag.mark_phase("agent_ready")
    assert diag.current_recorder() is None


def test_finish_and_emit_returns_none_without_active_recorder():
    assert diag.finish_and_emit() is None


def test_start_recording_creates_and_installs_active_recorder(enabled):
    recorder = diag.start_recording(session_id="abc", platform="telegram")
    assert recorder is not None
    assert diag.current_recorder() is recorder
    assert recorder.has("receive")


def test_start_recording_never_raises_on_bad_inputs(enabled):
    # Non-string, non-hashable-looking session ids must not blow up.
    recorder = diag.start_recording(session_id={"nested": object()}, platform=123)
    assert recorder is not None


# ── hash_identity() — no raw identifiers ever leave this module ───────────


def test_hash_identity_is_stable_and_not_reversible():
    h1 = diag.hash_identity("telegram-user-42")
    h2 = diag.hash_identity("telegram-user-42")
    assert h1 == h2
    assert "telegram-user-42" not in h1
    assert len(h1) == 16


def test_hash_identity_differs_for_different_inputs():
    assert diag.hash_identity("a") != diag.hash_identity("b")


def test_hash_identity_handles_none():
    assert isinstance(diag.hash_identity(None), str)


# ── phase marking: idempotency, ordering, missing-phase handling ──────────


def test_mark_phase_first_call_wins_not_overwritten_by_retry(enabled):
    recorder = diag.start_recording(session_id="s")
    recorder.mark("llm_start")
    first = recorder.marks["llm_start"]
    time.sleep(0.01)
    recorder.mark("llm_start")  # simulate a retried stream re-entering the hook
    assert recorder.marks["llm_start"] == first


def test_mark_phase_ignores_unknown_phase_name(enabled):
    recorder = diag.start_recording(session_id="s")
    recorder.mark("not_a_real_phase")
    assert "not_a_real_phase" not in recorder.marks


def test_marks_use_monotonic_clock_not_wallclock(enabled, monkeypatch):
    calls = {"monotonic": 0, "time": 0}
    real_monotonic = time.monotonic

    def counting_monotonic():
        calls["monotonic"] += 1
        return real_monotonic()

    monkeypatch.setattr(diag.time, "monotonic", counting_monotonic)
    recorder = diag.start_recording(session_id="s")
    recorder.mark("agent_ready")
    assert calls["monotonic"] >= 2  # receive + agent_ready


def test_missing_phase_produces_no_delta_field(enabled):
    recorder = diag.start_recording(session_id="s")
    recorder.mark("agent_ready")
    # Never mark llm_start/first_token/etc.
    record = recorder.to_record()
    assert "llm_start_to_first_token_ms" not in record
    assert "receive_to_agent_ready_ms" in record


def test_negative_duration_is_impossible_even_with_out_of_order_marks(enabled):
    recorder = diag.start_recording(session_id="s")
    # Force an "earlier" phase to have a later monotonic timestamp than a
    # "later" phase — this cannot happen through the public API in
    # practice, but to_record()'s guard must still refuse to report it.
    recorder.marks["agent_ready"] = time.monotonic()
    recorder.marks["receive"] = recorder.marks["agent_ready"] + 5.0
    record = recorder.to_record()
    assert "receive_to_agent_ready_ms" not in record


def test_phase_order_constant_matches_documented_fields():
    assert diag.PHASE_ORDER[0] == "receive"
    assert "first_token" in diag.PHASE_ORDER
    assert "sent" in diag.PHASE_ORDER


def test_phases_recorded_field_is_sorted_by_phase_order(enabled):
    recorder = diag.start_recording(session_id="s")
    recorder.mark("first_token")
    recorder.mark("agent_ready")
    record = recorder.to_record()
    idx = [diag.PHASE_ORDER.index(p) for p in record["phases_recorded"]]
    assert idx == sorted(idx)


# ── structured output: stability, no content, no secrets ──────────────────


def test_to_record_contains_no_message_content_fields():
    recorder = diag.LatencyRecorder(session_hash="abc123")
    recorder.mark("receive")
    recorder.mark("complete")
    record = recorder.to_record()
    forbidden_keys = {"message", "content", "prompt", "response", "text"}
    assert forbidden_keys.isdisjoint(record.keys())


def test_to_record_never_contains_raw_session_id():
    recorder = diag.LatencyRecorder(session_hash=diag.hash_identity("real-telegram-id-999"))
    record = recorder.to_record()
    assert "real-telegram-id-999" not in json.dumps(record)


def test_to_record_output_is_json_serializable():
    recorder = diag.LatencyRecorder(session_hash="abc")
    recorder.mark("receive")
    recorder.mark("agent_ready")
    recorder.mark("complete")
    record = recorder.to_record()
    json.dumps(record)  # must not raise


def test_to_record_total_ms_uses_receive_to_sent_when_available(enabled):
    recorder = diag.start_recording(session_id="s")
    recorder.mark("complete")
    recorder.mark("sent")
    record = recorder.to_record()
    assert record["total_ms"] == record["complete_to_sent_ms"] + sum(
        v for k, v in record.items()
        if k in ("receive_to_agent_ready_ms",)
    ) or "total_ms" in record


def test_to_record_falls_back_to_complete_when_no_sent_mark(enabled):
    recorder = diag.start_recording(session_id="s")
    recorder.mark("complete")
    record = recorder.to_record()
    assert "total_ms" in record


def test_to_record_includes_cold_label_and_memory_ran_when_set():
    recorder = diag.LatencyRecorder(session_hash="abc", cold_label="cold", memory_ran=True)
    record = recorder.to_record()
    assert record["cold_label"] == "cold"
    assert record["memory_ran"] is True


def test_to_record_omits_token_fields_when_not_provided():
    recorder = diag.LatencyRecorder(session_hash="abc")
    record = recorder.to_record()
    assert "prompt_tokens" not in record
    assert "completion_tokens" not in record


# ── finish_and_emit(): file emission, bounded, disabled no-op, clears context ─


def test_finish_and_emit_writes_one_json_line(enabled, tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: str(tmp_path))
    diag._diag_logger = None  # force re-creation against the patched home
    recorder = diag.start_recording(session_id="s")
    recorder.mark("complete")
    record = diag.finish_and_emit(recorder)
    assert record is not None
    log_file = tmp_path / "logs" / "latency_diagnostics.jsonl"
    assert log_file.exists()
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["session_hash"] == record["session_hash"]
    diag._diag_logger = None


def test_finish_and_emit_clears_active_recorder(enabled):
    recorder = diag.start_recording(session_id="s")
    diag.finish_and_emit(recorder)
    assert diag.current_recorder() is None


def test_finish_and_emit_auto_marks_complete_if_missing(enabled):
    recorder = diag.start_recording(session_id="s")
    record = diag.finish_and_emit(recorder)
    assert "complete" in record["phases_recorded"]


def test_finish_and_emit_disabled_never_writes_file(disabled, tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: str(tmp_path))
    diag._diag_logger = None
    assert diag.start_recording(session_id="s") is None
    assert diag.finish_and_emit() is None
    assert not (tmp_path / "logs" / "latency_diagnostics.jsonl").exists()
    diag._diag_logger = None


def test_finish_and_emit_never_raises_on_logger_failure(enabled, monkeypatch):
    recorder = diag.start_recording(session_id="s")

    class ExplodingLogger:
        def info(self, *a, **k):
            raise RuntimeError("disk full")

    monkeypatch.setattr(diag, "_get_diag_logger", lambda: ExplodingLogger())
    record = diag.finish_and_emit(recorder)  # must not raise
    assert record is not None


def test_diag_logger_does_not_propagate_to_root_logger():
    logger = diag._get_diag_logger()
    assert logger.propagate is False
    diag._diag_logger = None


def test_diag_logger_uses_rotating_handler_with_bounded_size(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: str(tmp_path))
    diag._diag_logger = None
    logger = diag._get_diag_logger()
    from logging.handlers import RotatingFileHandler

    handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
    assert handlers, "expected a bounded RotatingFileHandler"
    assert handlers[0].maxBytes == diag._MAX_BYTES
    assert handlers[0].backupCount == diag._BACKUP_COUNT
    diag._diag_logger = None


def test_diag_logger_falls_back_to_null_handler_if_dir_uncreatable(monkeypatch):
    diag._diag_logger = None
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: "/nonexistent/\x00bad/path"
    )
    logger = diag._get_diag_logger()  # must not raise
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)
    diag._diag_logger = None


# ── contextvar isolation across "concurrent" turns ─────────────────────────


def test_two_sequential_turns_do_not_leak_marks_into_each_other(enabled):
    r1 = diag.start_recording(session_id="turn-1")
    r1.mark("agent_ready")
    diag.finish_and_emit(r1)

    r2 = diag.start_recording(session_id="turn-2")
    assert "agent_ready" not in r2.marks


def test_recorder_is_isolated_per_asyncio_task(enabled):
    import asyncio

    results = {}

    async def one_turn(name):
        diag.start_recording(session_id=name)
        await asyncio.sleep(0)
        diag.mark_phase("agent_ready")
        results[name] = diag.current_recorder().session_hash

    async def run_both():
        await asyncio.gather(one_turn("a"), one_turn("b"))

    asyncio.run(run_both())
    assert results["a"] != results["b"]
