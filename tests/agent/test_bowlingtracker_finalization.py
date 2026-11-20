"""Pure policy tests for authoritative BowlingTracker personal-data finalization."""

import json

from agent.bowlingtracker_finalization import (
    bowlingtracker_terminal_response,
    has_explicit_fusion_intent,
)


def result(**overrides):
    payload = {
        "contractVersion": "personal-bowling-evidence/v1",
        "status": "ok",
        "dataSource": "BowlingTracker",
        "answer": "Your BowlingTracker personal-data summary: 3 scored games, average 210.67.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def messages(text="What did I average in my last 10 games?"):
    return [{"role": "user", "content": text}]


def test_valid_personal_answer_is_terminal_exactly_as_authored():
    expected = "Your BowlingTracker personal-data summary: 3 scored games, average 210.67."
    assert (
        bowlingtracker_terminal_response(
            "bowlingtracker_personal_summary", result(), messages()
        )
        == expected
    )


def test_no_data_and_unavailable_answers_are_also_terminal():
    no_data = result(status="no_data", answer="No synchronized tournament sessions matched.")
    unavailable = result(
        status="unavailable",
        answer="Personal BowlingTracker data is unavailable because the local service timed out.",
        error={"code": "timeout"},
    )
    assert "No synchronized" in bowlingtracker_terminal_response(
        "bowlingtracker_recent_sessions", no_data, messages("Show my tournaments")
    )
    assert "timed out" in bowlingtracker_terminal_response(
        "bowlingtracker_recent_sessions", unavailable, messages("Show my recent sessions")
    )


def test_unknown_tool_malformed_contract_or_wrong_source_is_not_terminal():
    assert bowlingtracker_terminal_response("web_search", result(), messages()) is None
    assert bowlingtracker_terminal_response(
        "bowlingtracker_personal_summary", "not-json", messages()
    ) is None
    assert bowlingtracker_terminal_response(
        "bowlingtracker_personal_summary", result(dataSource="Other"), messages()
    ) is None
    assert bowlingtracker_terminal_response(
        "bowlingtracker_personal_summary", result(answer=""), messages()
    ) is None


def test_explicit_web_or_future_fusion_intent_stays_in_agent_loop():
    assert bowlingtracker_terminal_response(
        "bowlingtracker_personal_summary",
        result(),
        messages("Search the web and compare current specs with my BowlingTracker results."),
    ) is None
    fusion = "Given my recent BowlingTracker results, how would Mo Pinel's advice apply?"
    assert has_explicit_fusion_intent(fusion)
    assert bowlingtracker_terminal_response(
        "bowlingtracker_personal_summary", result(), messages(fusion)
    ) is None


def analysis_result(**overrides):
    payload = {
        "contractVersion": "personal-bowling-session-analysis/v1",
        "status": "ok",
        "dataSource": "BowlingTracker",
        "answer": "Session: Practice. Linkage is session-only.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_session_analysis_answer_is_terminal_exactly_as_authored():
    assert bowlingtracker_terminal_response(
        "bowlingtracker_session_analysis", analysis_result(), messages("How did Practice go?")
    ) == "Session: Practice. Linkage is session-only."


def test_session_analysis_errors_and_ambiguity_are_terminal():
    for status in ("error", "ambiguous", "not_found"):
        assert bowlingtracker_terminal_response(
            "bowlingtracker_session_analysis",
            analysis_result(status=status, answer=f"{status} answer"),
            messages("How did Practice go?"),
        ) == f"{status} answer"


def test_session_analysis_accepts_shared_failure_envelope():
    assert bowlingtracker_terminal_response(
        "bowlingtracker_session_analysis",
        result(status="unavailable", answer="Personal BowlingTracker data is unavailable."),
        messages("How did Practice go?"),
    ) == "Personal BowlingTracker data is unavailable."


def test_session_analysis_contract_is_not_accepted_for_other_tools():
    assert bowlingtracker_terminal_response(
        "bowlingtracker_personal_summary", analysis_result(), messages()
    ) is None


def test_session_analysis_still_yields_to_explicit_fusion_or_web_intent():
    assert bowlingtracker_terminal_response(
        "bowlingtracker_session_analysis",
        analysis_result(),
        messages("Given my BowlingTracker session, what would Mo Pinel say?"),
    ) is None
