"""Runtime wiring tests for terminal personal-data results."""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "bowlingtracker_personal_summary",
                "description": "personal data",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _tool_call():
    return SimpleNamespace(
        id=f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name="bowlingtracker_personal_summary", arguments="{}"),
    )


def _response(content="", finish_reason="tool_calls", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        model="test/model",
        usage=None,
    )


def _agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=4,
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


def _tool_result(answer, *, status="ok", error=None):
    payload = {
        "contractVersion": "personal-bowling-evidence/v1",
        "status": status,
        "dataSource": "BowlingTracker",
        "answer": answer,
    }
    if error:
        payload["error"] = error
    return json.dumps(payload)


def test_personal_answer_terminates_before_model_can_rewrite_it():
    agent = _agent()
    exact = "Your BowlingTracker data: 4 games existed; 3 scored; average 210.67 pins."
    agent.client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call()]),
    ]
    with (
        patch("run_agent.handle_function_call", return_value=_tool_result(exact)),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        output = agent.run_conversation("What did I average in my last 10 games?")
    assert agent.client.chat.completions.create.call_count == 1
    assert output["turn_exit_reason"] == "bowlingtracker_terminal"
    assert output["final_response"] == exact


def test_unavailable_result_is_terminal_and_cannot_fall_back():
    agent = _agent()
    exact = "Personal BowlingTracker data is unavailable because the local service timed out."
    agent.client.chat.completions.create.side_effect = [_response(tool_calls=[_tool_call()])]
    with (
        patch(
            "run_agent.handle_function_call",
            return_value=_tool_result(exact, status="unavailable", error={"code": "timeout"}),
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        output = agent.run_conversation("Show my recent BowlingTracker games")
    assert agent.client.chat.completions.create.call_count == 1
    assert output["turn_exit_reason"] == "bowlingtracker_terminal"
    assert output["final_response"] == exact


def test_explicit_fusion_request_can_continue_to_answer_layer():
    agent = _agent()
    agent.client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call()]),
        _response(
            content="Personal evidence and expert evidence remain separate.",
            finish_reason="stop",
            tool_calls=None,
        ),
    ]
    with (
        patch("run_agent.handle_function_call", return_value=_tool_result("Personal evidence.")),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        output = agent.run_conversation(
            "Given my recent BowlingTracker results, how would Mo Pinel's advice apply?"
        )
    assert agent.client.chat.completions.create.call_count == 2
    assert output["turn_exit_reason"].startswith("text_response")

