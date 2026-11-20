"""Agent-loop regressions: session-analysis results are terminal and grounded.

The model is faked to emit a ``bowlingtracker_session_analysis`` call; dispatch,
the real tool handler, the real BowlingTracker adapter (with an injected
session list and synthetic compact packages), and deterministic finalization all
run unmodified. These fixtures pin runtime behavior, not model tool selection.
"""

import copy
import importlib.util
import json
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tools import bowlingtracker_tool as btt

ADAPTER_PATH = btt._ADAPTER_PATH
FIXTURES = ADAPTER_PATH.parents[1] / "tests" / "fixtures"
pytestmark = pytest.mark.skipif(
    not (ADAPTER_PATH.exists() and FIXTURES.exists()),
    reason="BowlingTracker skill adapter or fixtures are not installed",
)

PRACTICE = "11111111-1111-4111-8111-111111111111"
LEAGUE = "22222222-2222-4222-8222-222222222222"
SESSIONS = [
    {"id": PRACTICE, "name": "Practice", "date": "2026-09-27T18:58:42Z",
     "sessionType": "Practice", "bowlingCenter": "Test Lanes"},
    {"id": LEAGUE, "name": "League Night", "date": "2026-09-19T18:20:06Z",
     "sessionType": "League", "bowlingCenter": "Test Lanes"},
]
BALLS = [{"id": "66666666-6666-4666-8666-666666666666", "name": "Venom Shock"}]
UNSAFE = (
    " because ", " caused ", " due to ", " therefore ", " explains ", " winner",
    " better session", " worse session", " you should ", " recommend",
)


def _load_adapter():
    spec = importlib.util.spec_from_file_location(
        "bowlingtracker_session_analysis_routing_adapter", ADAPTER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FixtureAdapter:
    """Real adapter operation with injected reads and synthetic compact packages."""

    def __init__(self):
        self.real = _load_adapter()
        self.single = json.loads((FIXTURES / "session-analysis-compact-single.json").read_text())
        self.pair = json.loads(
            (FIXTURES / "session-analysis-compact-comparison.json").read_text()
        )
        self.calls = []

    def session_analysis(self, **arguments):
        self.calls.append(arguments)
        return self.real.session_analysis(
            get=lambda path: {
                "items": copy.deepcopy(BALLS if "/balls" in path else SESSIONS),
                "limit": 100,
                "offset": 0,
            },
            analyze=lambda session_id: copy.deepcopy(self.single),
            compare=lambda ids: copy.deepcopy(self.pair),
            **arguments,
        )

    def failure_result(self, error):
        return self.real.failure_result(error)


def _tool_defs():
    return [{
        "type": "function",
        "function": {
            "name": "bowlingtracker_session_analysis",
            "description": "session analysis",
            "parameters": {"type": "object", "properties": {}},
        },
    }]


def _tool_call(arguments):
    return SimpleNamespace(
        id=f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(
            name="bowlingtracker_session_analysis", arguments=json.dumps(arguments)
        ),
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


@pytest.fixture
def adapter():
    fixture = FixtureAdapter()
    btt._adapter = fixture
    yield fixture
    btt._adapter = None


def _run(prompt, arguments, *, model_rewrite="The Practice session was worse because of fatigue."):
    agent = _agent()
    agent.client.chat.completions.create.side_effect = [
        _response(tool_calls=[_tool_call(arguments)]),
        _response(content=model_rewrite, finish_reason="stop", tool_calls=None),
    ]
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        output = agent.run_conversation(prompt)
    return agent, output


def _assert_grounded_terminal(agent, output):
    assert agent.client.chat.completions.create.call_count == 1
    assert output["turn_exit_reason"] == "bowlingtracker_terminal"
    text = output["final_response"]
    assert "Invalid session UUID" not in text
    assert "fatigue" not in text
    lowered = f" {text.lower()} "
    for phrase in UNSAFE:
        assert phrase not in lowered, phrase
    return text


def test_q1_why_lower_resolves_name_and_states_no_cause(adapter):
    agent, output = _run(
        "Why did I score lower in Practice?",
        {"session_name": "Practice", "question_focus": "cause"},
    )
    text = _assert_grounded_terminal(agent, output)
    assert text.startswith("The BowlingTracker and Witsee evidence cannot establish why")
    assert adapter.single["humanSummary"]["text"] in text


def test_q2_which_shot_states_no_shot_correspondence(adapter):
    agent, output = _run(
        "Which shot had the speed drop?",
        {"session_date": "2026-09-27", "question_focus": "shot_or_frame"},
    )
    text = _assert_grounded_terminal(agent, output)
    assert "cannot identify which shot or frame" in text


def test_q3_ball_reports_recorded_usage_without_judgment(adapter):
    agent, output = _run(
        "Was the Test Ball better?",
        {"session_name": "Practice", "question_focus": "ball", "ball": "Test Ball"},
    )
    text = _assert_grounded_terminal(agent, output)
    assert "BowlingTracker recorded Test Ball:" in text
    assert "no ball is judged better or worse" in text


def test_q4_game_three_states_no_game_correspondence(adapter):
    agent, output = _run(
        "Did my speed drop in game 3?",
        {"session_name": "Practice", "question_focus": "game", "game_number": 3},
    )
    text = _assert_grounded_terminal(agent, output)
    assert "cannot show what happened in game 3" in text


def test_q5_which_session_better_gives_differences_without_judgment(adapter):
    agent, output = _run(
        "Which session was better, League Night or Practice?",
        {
            "session_name": "League Night",
            "compare_session_name": "Practice",
            "question_focus": "better_or_worse",
        },
    )
    text = _assert_grounded_terminal(agent, output)
    assert text.startswith("No overall better or worse judgment is produced.")
    assert "Session B minus Session A" in text
    assert adapter.calls[0]["session_name"] == "League Night"


def test_ambiguous_or_unknown_session_is_terminal_without_fallback(adapter):
    agent, output = _run(
        "How did my Tuesday session go?", {"session_name": "Tuesday"}
    )
    assert output["turn_exit_reason"] == "bowlingtracker_terminal"
    assert output["final_response"].startswith("No BowlingTracker session matches")
    assert agent.client.chat.completions.create.call_count == 1


# The live 9B model was observed sending {"question_focus": "baseline"} or {} for these
# exact prompts. The runtime passes the user's own words to the adapter, so the verified
# answer must still be correct for each of the five challenge prompts.
@pytest.mark.parametrize(
    ("prompt", "observed_arguments", "expected_start", "expected_status"),
    [
        (
            "Why did I score lower in Practice?", {"question_focus": "baseline"},
            "The BowlingTracker and Witsee evidence cannot establish why", "ok",
        ),
        (
            "Which shot had the speed drop?", {},
            "Witsee throws are linked to the session only", "needs_session",
        ),
        (
            "Was the Venom better?", {},
            "Witsee throws are not matched to balls", "needs_session",
        ),
        (
            "Did my speed drop in game 3?", {"question_focus": "baseline"},
            "Witsee data is linked to the session, not to games.", "needs_session",
        ),
        (
            "Which session was better?", {},
            "No overall better or worse judgment of a session is produced.", "needs_session",
        ),
    ],
)
def test_exact_challenge_prompts_with_observed_model_arguments(
    adapter, prompt, observed_arguments, expected_start, expected_status
):
    agent, output = _run(prompt, observed_arguments)
    text = _assert_grounded_terminal(agent, output)
    assert text.startswith(expected_start)
    result = adapter.real.session_analysis(
        get=lambda path: {
            "items": copy.deepcopy(BALLS if "/balls" in path else SESSIONS),
            "limit": 100, "offset": 0,
        },
        analyze=lambda session_id: copy.deepcopy(adapter.single),
        compare=lambda ids: copy.deepcopy(adapter.pair),
        user_question=prompt,
        **observed_arguments,
    )
    assert result["status"] == expected_status
    assert result["answer"] == text
