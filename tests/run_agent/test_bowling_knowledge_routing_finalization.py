"""Runtime tests for Bowling Knowledge routing/finalization.

Bowling Knowledge (bowling_knowledge_answer) must be the authoritative first
path for bowling-domain knowledge questions, and its result -- supported or
unsupported -- must be terminal for the turn unless the user's own message
carried explicit current/public-web intent. See
agent/bowling_knowledge_finalization.py.
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

_UNSUPPORTED_RESULT = json.dumps({
    "comparison_mode": False,
    "answer": "The curated Bowling Knowledge corpus does not contain enough sourced "
              "evidence to answer that question.",
    "evidence_status": "NO_EVIDENCE",
    "validation_status": "bypassed",
    "sources": [],
    "citations": [],
    "warnings": [],
})

_COMPARISON_RESULT = json.dumps({
    "comparison_mode": True,
    "question": "Compare Mo Pinel and Joe Slowinski on layout.",
    "overlapping_topics": ["layout"],
    "warnings": [],
    "expert_sections": [
        {
            "source_name": "Mo Pinel", "answer": "Mo's answer.",
            "evidence_status": "STRONG", "validation_status": "ok",
            "sources": ["Mo Pinel"], "citations": [], "warnings": [],
        },
        {
            "source_name": "Joe Slowinski", "answer": "Joe's answer.",
            "evidence_status": "STRONG", "validation_status": "ok",
            "sources": ["Joe Slowinski"], "citations": [], "warnings": [],
        },
    ],
})


def test_supported_answer_terminates_turn_without_second_model_call():
    """Rule 2: a supported Bowling Knowledge result is used as-is and the
    turn ends -- the model is never asked whether to call web_search next."""
    agent = _make_agent("bowling_knowledge_answer", "web_search")
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
        result = agent.run_conversation("What does Mo Pinel say about RG versus differential?")

    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "bowling_knowledge_terminal"
    assert "Mo Pinel says RG describes mass distance" in result["final_response"]
    assert "Sources:" in result["final_response"]
    assert "Mo Pinel" in result["final_response"]


def test_unsupported_answer_terminates_turn_without_web_fallback():
    """Rule 3: an unsupported/insufficient-evidence result is terminal --
    the outer loop must not go on to call web_search or answer some other
    way. This is the exact bug this milestone closes."""
    agent = _make_agent("bowling_knowledge_answer", "web_search")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("bowling_knowledge_answer", json.dumps(
                {"question": "What's the WTBA world ranking points formula?"}
            ))],
        ),
    ]

    with (
        patch("run_agent.handle_function_call", return_value=_UNSUPPORTED_RESULT),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("What's the WTBA world ranking points formula?")

    # Only one completions call was ever made -- the model was never given
    # the chance to decide to call web_search after the unsupported result.
    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "bowling_knowledge_terminal"
    assert "does not contain enough sourced evidence" in result["final_response"]
    assert "Sources:" not in result["final_response"]


def test_comparison_mode_result_terminates_turn_and_keeps_experts_separate():
    agent = _make_agent("bowling_knowledge_answer", "web_search")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("bowling_knowledge_answer", json.dumps(
                {"question": "Compare Mo Pinel and Joe Slowinski on layout."}
            ))],
        ),
    ]

    with (
        patch("run_agent.handle_function_call", return_value=_COMPARISON_RESULT),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("Compare Mo Pinel and Joe Slowinski on layout.")

    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "bowling_knowledge_terminal"
    assert "Mo's answer." in result["final_response"]
    assert "Joe's answer." in result["final_response"]
    assert "Mo Pinel" in result["final_response"]
    assert "Joe Slowinski" in result["final_response"]


def test_explicit_web_intent_allows_web_search_after_unsupported_result():
    """Rule 4: explicit current/public-web intent in the user's own message
    is the one thing that allows a web call after Bowling Knowledge."""
    agent = _make_agent("bowling_knowledge_answer", "web_search")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("bowling_knowledge_answer", json.dumps(
                {"question": "What's the current WTBA world ranking formula?"}
            ))],
        ),
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps({"query": "WTBA ranking formula"}), "c-web")],
        ),
        _mock_response(content="Here is the current formula from the web.", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", side_effect=[_UNSUPPORTED_RESULT, json.dumps({"results": []})]),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("Search the web for the current WTBA world ranking formula.")

    assert agent.client.chat.completions.create.call_count == 3
    assert result["turn_exit_reason"].startswith("text_response")
    assert result["final_response"] == "Here is the current formula from the web."
    assert agent._bowling_knowledge_terminal_response is None


def test_non_bowling_tool_call_is_unaffected():
    """A turn that never calls bowling_knowledge_answer must behave exactly
    as before -- normal multi-tool-call loops are unaffected."""
    agent = _make_agent("web_search")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps({"query": "weather today"}), "c-w1")],
        ),
        _mock_response(content="It's sunny.", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"results": ["sunny"]})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("What's the weather today?")

    assert agent.client.chat.completions.create.call_count == 2
    assert result["turn_exit_reason"].startswith("text_response")
    assert result["final_response"] == "It's sunny."
    assert agent._bowling_knowledge_terminal_response is None


def test_precall_gate_blocks_web_search_and_redirects_to_bowling_knowledge():
    """Reproduces the live bug (observed via a real CLI run against the
    configured local model): for a bowling-domain question with no named
    expert, the model reached straight for web_search/web_extract without
    ever trying bowling_knowledge_answer, so the post-call finalization
    (Rule 2/3) never got a chance to fire. Rule 1's precall gate blocks the
    web call before it runs and redirects the model to bowling_knowledge_
    answer, whose result then terminates the turn as usual."""
    agent = _make_agent("bowling_knowledge_answer", "web_search")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(
                {"query": "WTBA world ranking points formula"}
            ), "c-web1")],
        ),
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("bowling_knowledge_answer", json.dumps(
                {"question": "What's the WTBA world ranking points formula?"}
            ), "c-bk1")],
        ),
    ]

    with (
        patch("run_agent.handle_function_call", return_value=_UNSUPPORTED_RESULT) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("What's the WTBA world ranking points formula?")

    # handle_function_call must only ever have run for the (allowed)
    # bowling_knowledge_answer call -- never for the blocked web_search
    # attempt, which never reached real dispatch at all.
    mock_hfc.assert_called_once()
    assert mock_hfc.call_args[0][0] == "bowling_knowledge_answer"
    assert agent.client.chat.completions.create.call_count == 2
    assert result["turn_exit_reason"] == "bowling_knowledge_terminal"
    blocked_tool_message = next(
        m for m in result["messages"]
        if m.get("role") == "tool" and m.get("tool_call_id") == "c-web1"
    )
    assert "bowling_knowledge_answer" in blocked_tool_message["content"]


def test_precall_gate_does_not_block_web_search_with_explicit_web_intent():
    """The precall gate must not interfere with Rule 4: explicit
    current/public-web intent may go straight to web_search."""
    agent = _make_agent("bowling_knowledge_answer", "web_search")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(
                {"query": "current WTBA world ranking formula"}
            ), "c-web1")],
        ),
        _mock_response(content="Here is the current formula.", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"results": []})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("Search the web for the current WTBA world ranking formula.")

    mock_hfc.assert_called_once()
    assert mock_hfc.call_args[0][0] == "web_search"
    assert agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == "Here is the current formula."


def test_concurrent_batch_race_never_leaks_web_content_into_final_answer():
    """Regression for a real live finding: a single assistant message can
    call bowling_knowledge_answer AND web_search TOGETHER in one batch
    (dispatched concurrently), which can race past the precall gate (Rule
    1) before bowling_knowledge_answer's "attempted" flag is set. Even in
    that race, the post-call finalization (Rule 2/3) must still be the
    authoritative backstop: the final answer must come only from
    bowling_knowledge_answer, and web_search's own result must never leak
    into what the user sees."""
    agent = _make_agent("bowling_knowledge_answer", "web_search")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="", finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call("bowling_knowledge_answer", json.dumps(
                    {"question": "What's the WTBA world ranking points formula?"}
                ), "c-bk1"),
                _mock_tool_call("web_search", json.dumps(
                    {"query": "WTBA world ranking points formula"}
                ), "c-web1"),
            ],
        ),
    ]

    def _hfc_side_effect(tool_name, *args, **kwargs):
        if tool_name == "bowling_knowledge_answer":
            return _UNSUPPORTED_RESULT
        return json.dumps({"results": ["some web page mentioning WTBA rankings"]})

    with (
        patch("run_agent.handle_function_call", side_effect=_hfc_side_effect),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("What's the WTBA world ranking points formula?")

    # Only one completions call: the batch resolved in a single round trip
    # and the turn ended deterministically without asking the model to
    # decide what to do with the web result.
    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "bowling_knowledge_terminal"
    assert "does not contain enough sourced evidence" in result["final_response"]
    assert "web page" not in result["final_response"]
    assert "some web page mentioning WTBA rankings" not in result["final_response"]
