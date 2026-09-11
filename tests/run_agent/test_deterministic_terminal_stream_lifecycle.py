"""Regression coverage for the deterministic-terminal stream callback lifecycle.

Bowling Knowledge / BowlingTracker / tool-guardrail terminal turns end the turn
immediately after a tool call, delivering their final answer directly instead
of through the normal ``_fire_stream_delta`` path. That direct delivery used
to call ONLY ``agent.stream_delta_callback`` -- the gateway/SSE/messaging-
platform display slot set once at agent construction. A CLI/TUI-gateway
session's live UI channel is a DIFFERENT slot, ``agent._stream_callback``,
wired fresh each turn via ``run_conversation(stream_callback=...)`` (see
``agent/turn_context.py``). That consumer never received the answer text and
never received a turn-completion signal on the channel it was actually
listening to, even though the turn itself completed correctly (correct
``final_response``/``completed``/``turn_exit_reason``) -- a live symptom of
"the answer displays somewhere, but the interactive session never returns to
a ready state" for exactly that consumer shape.

These tests exercise the real callback lifecycle through ``AIAgent.run_conversation``
(the same entry point every surface -- CLI, TUI-gateway, messaging gateway --
calls), wiring both consumer slots the way production code does:
``agent.stream_delta_callback`` directly, and ``agent._stream_callback`` via
the ``stream_callback=`` kwarg.
"""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name="bowling_knowledge_answer", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(*tool_names: str, max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("hermes_cli.config.load_config_readonly", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._disable_streaming = True
    return agent


class _Consumer:
    """Records every call a stream slot receives, preserving order."""

    def __init__(self):
        self.calls: list[object] = []

    def __call__(self, delta):
        self.calls.append(delta)


def _wire_both_stream_slots(agent: AIAgent) -> tuple[_Consumer, _Consumer]:
    """Wire ``stream_delta_callback`` (gateway/SSE slot) directly like a
    messaging-platform/dashboard caller does, returning both consumers so the
    caller can also pass the second one as ``run_conversation(stream_callback=...)``
    -- the CLI/TUI-gateway per-turn live UI slot (``agent._stream_callback``)."""
    gateway_consumer = _Consumer()
    ui_consumer = _Consumer()
    agent.stream_delta_callback = gateway_consumer
    return gateway_consumer, ui_consumer


_SUPPORTED_RESULT = json.dumps({
    "comparison_mode": False,
    "answer": "Mo Pinel says RG describes mass distance from axis; differential is the RG delta.",
    "evidence_status": "STRONG",
    "validation_status": "ok",
    "sources": ["Mo Pinel"],
    "citations": [{
        "claim_id": "c1", "source_name": "Mo Pinel",
        "document_title": "RG and Differential", "locator": "p.4", "url": None,
    }],
    "warnings": [],
})

_WEAK_RESULT = json.dumps({
    "comparison_mode": False,
    "answer": "The Bowling Knowledge corpus does not currently contain enough "
              "relevant sourced evidence to answer that confidently.",
    "evidence_status": "WEAK",
    "validation_status": "bypassed",
    "sources": [],
    "citations": [],
    "warnings": ["Only a single incidental search term appears to drive the top match."],
})

_BOWLINGTRACKER_RESULT = json.dumps({
    "contractVersion": "personal-bowling-evidence/v1",
    "status": "ok",
    "dataSource": "BowlingTracker",
    "evidenceKinds": ["personal_observation"],
    "answer": "Your last 5 sessions averaged 187.4, up from 179.2 the prior 5.",
})


def _assert_terminal_lifecycle(
    result: dict, gateway_consumer: _Consumer, ui_consumer: _Consumer,
    expected_final: str, expected_exit_reason: str,
):
    """Shared assertions for a deterministic-terminal turn's callback lifecycle."""
    # Finalizer fields (agent/turn_finalizer.py contract).
    assert result["turn_exit_reason"] == expected_exit_reason
    assert result["final_response"] == expected_final
    assert result["completed"] is True
    assert result.get("failed") is not True
    assert result.get("interrupted") is not True

    # The CLI/TUI-gateway live UI channel (_stream_callback) -- the channel
    # that was silently skipped before the fix -- must receive the final
    # answer exactly once, and must never receive the None end-of-speech
    # sentinel (TTS contract: None means "stop speaking now", not "segment
    # boundary").
    assert ui_consumer.calls == [expected_final], (
        "the CLI/TUI-gateway stream_callback slot must receive the terminal "
        f"answer exactly once; got {ui_consumer.calls!r}"
    )
    assert ui_consumer.calls.count(expected_final) == 1  # no duplicate delivery

    # The gateway/SSE display slot (stream_delta_callback) keeps its
    # pre-existing, already-correct lifecycle: a pre-tool-execution None
    # (segment/box-close boundary), then the terminal text, then a trailing
    # None (end-of-segment marker for that slot only).
    assert gateway_consumer.calls == [None, expected_final, None], (
        f"unexpected stream_delta_callback sequence: {gateway_consumer.calls!r}"
    )
    assert gateway_consumer.calls.count(expected_final) == 1  # no duplicate delivery


# --- Bowling Knowledge terminal turns -----------------------------------


def test_weak_bowling_knowledge_terminal_stream_lifecycle():
    """Weak/insufficient-evidence corpus answer: full callback lifecycle,
    including delivery on the CLI/TUI-gateway channel (the reported bug)."""
    agent = _make_agent("bowling_knowledge_answer", "web_search")
    gateway_consumer, ui_consumer = _wire_both_stream_slots(agent)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("bowling_knowledge_answer", json.dumps(
                {"question": "How should I use a balance hole to change my bowling ball reaction?"}
            ))],
        ),
    ]

    with (
        patch("run_agent.handle_function_call", return_value=_WEAK_RESULT),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "How should I use a balance hole to change my bowling ball reaction?",
            stream_callback=ui_consumer,
        )

    assert agent.client.chat.completions.create.call_count == 1
    _assert_terminal_lifecycle(
        result, gateway_consumer, ui_consumer,
        expected_final=(
            "The Bowling Knowledge corpus does not currently contain enough "
            "relevant sourced evidence to answer that confidently."
        ),
        expected_exit_reason="bowling_knowledge_terminal",
    )


def test_supported_bowling_knowledge_terminal_stream_lifecycle():
    """Supported/cited corpus answer: full callback lifecycle, citation intact
    on both consumer slots (it's part of the single final_response string)."""
    agent = _make_agent("bowling_knowledge_answer", "web_search")
    gateway_consumer, ui_consumer = _wire_both_stream_slots(agent)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("bowling_knowledge_answer", json.dumps(
                {"question": "What does Mo Pinel say about RG versus differential?"}
            ))],
        ),
    ]

    with (
        patch("run_agent.handle_function_call", return_value=_SUPPORTED_RESULT),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "What does Mo Pinel say about RG versus differential?",
            stream_callback=ui_consumer,
        )

    assert agent.client.chat.completions.create.call_count == 1
    assert "Sources:" in result["final_response"]
    assert "Mo Pinel" in result["final_response"]
    _assert_terminal_lifecycle(
        result, gateway_consumer, ui_consumer,
        expected_final=result["final_response"],
        expected_exit_reason="bowling_knowledge_terminal",
    )
    # Citation text reached the previously-broken UI channel too.
    assert "Sources:" in ui_consumer.calls[0]
    assert "Mo Pinel" in ui_consumer.calls[0]


# --- BowlingTracker terminal turns (structurally identical path) --------


def test_bowlingtracker_terminal_stream_lifecycle():
    """BowlingTracker's deterministic personal-data terminal has the exact
    same tool-boundary delivery code as Bowling Knowledge -- same bug, same
    fix, same lifecycle contract."""
    agent = _make_agent("bowlingtracker_personal_summary", "web_search")
    gateway_consumer, ui_consumer = _wire_both_stream_slots(agent)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("bowlingtracker_personal_summary", "{}")],
        ),
    ]

    with (
        patch("run_agent.handle_function_call", return_value=_BOWLINGTRACKER_RESULT),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "What's my recent average?",
            stream_callback=ui_consumer,
        )

    assert agent.client.chat.completions.create.call_count == 1
    _assert_terminal_lifecycle(
        result, gateway_consumer, ui_consumer,
        expected_final="Your last 5 sessions averaged 187.4, up from 179.2 the prior 5.",
        expected_exit_reason="bowlingtracker_terminal",
    )


# --- Unaffected paths (regression guard) ---------------------------------


def test_normal_text_response_stream_lifecycle_unchanged():
    """A plain text response (no tool calls) never touches the deterministic-
    terminal delivery code, so neither stream slot sees anything in this
    non-streaming mocked harness -- confirms the fix didn't leak into the
    ordinary path."""
    agent = _make_agent("web_search")
    gateway_consumer, ui_consumer = _wire_both_stream_slots(agent)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="Just a plain answer.", finish_reason="stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("Hello", stream_callback=ui_consumer)

    assert result["final_response"] == "Just a plain answer."
    assert result["turn_exit_reason"].startswith("text_response(")
    assert result["completed"] is True
    assert gateway_consumer.calls == []
    assert ui_consumer.calls == []


def test_ordinary_tool_call_stream_lifecycle_unchanged():
    """A non-bowling tool call keeps its pre-existing lifecycle: the
    pre-tool-execution None (box-close boundary) on stream_delta_callback,
    then a normal second round produces the final text. The deterministic-
    terminal delivery path must never fire for this turn."""
    agent = _make_agent("web_search")
    gateway_consumer, ui_consumer = _wire_both_stream_slots(agent)
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps({"query": "bowling news"}))],
        ),
        _mock_response(content="Here's what I found.", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"results": []})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_deliver_deterministic_terminal_text") as spy,
    ):
        result = agent.run_conversation("Any bowling news?", stream_callback=ui_consumer)

    assert agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == "Here's what I found."
    assert result["turn_exit_reason"].startswith("text_response(")
    assert getattr(agent, "_bowling_knowledge_terminal_response", None) is None
    assert getattr(agent, "_bowlingtracker_terminal_response", None) is None
    spy.assert_not_called()
    # Only the pre-tool-execution box-close boundary fired on the gateway
    # slot; the (mocked, non-streaming) second round never drives
    # _fire_stream_delta, so no premature/extra signals appear either.
    assert gateway_consumer.calls == [None]
    assert ui_consumer.calls == []


# --- Unit coverage for the shared helper ---------------------------------


def test_deliver_deterministic_terminal_text_reaches_both_slots():
    agent = _make_agent()
    gateway_consumer, ui_consumer = _wire_both_stream_slots(agent)
    agent._stream_callback = ui_consumer

    agent._deliver_deterministic_terminal_text("final text")

    assert gateway_consumer.calls == ["final text"]
    assert ui_consumer.calls == ["final text"]


def test_deliver_deterministic_terminal_text_skips_empty_text():
    agent = _make_agent()
    gateway_consumer, ui_consumer = _wire_both_stream_slots(agent)
    agent._stream_callback = ui_consumer

    agent._deliver_deterministic_terminal_text("")

    assert gateway_consumer.calls == []
    assert ui_consumer.calls == []


def test_deliver_deterministic_terminal_text_swallows_one_callback_error():
    agent = _make_agent()

    def _raises(_text):
        raise RuntimeError("boom")

    ui_consumer = _Consumer()
    agent.stream_delta_callback = _raises
    agent._stream_callback = ui_consumer

    agent._deliver_deterministic_terminal_text("final text")  # must not raise

    assert ui_consumer.calls == ["final text"]


def test_deliver_deterministic_terminal_text_handles_unset_slots():
    agent = _make_agent()
    agent.stream_delta_callback = None
    agent._stream_callback = None

    agent._deliver_deterministic_terminal_text("final text")  # must not raise
