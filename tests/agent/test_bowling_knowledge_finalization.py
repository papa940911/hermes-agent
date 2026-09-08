"""Unit tests for agent.bowling_knowledge_finalization (pure logic, no agent)."""

import inspect
import json

import agent.bowling_knowledge_finalization as bkf
from agent.bowling_knowledge_finalization import (
    BOWLING_KNOWLEDGE_TOOL_NAME,
    bowling_knowledge_precall_block_reason,
    bowling_knowledge_terminal_response,
    has_explicit_web_intent,
    is_bowling_domain_question,
    latest_user_text,
)


# --- provider/model independence ---------------------------------------------

def test_module_is_provider_and_model_agnostic():
    """This routing/finalization policy must never branch on a specific
    provider or model (e.g. Qwen) -- it only looks at the tool name, the
    tool's own result shape, and the user's own words. Guard against a
    future edit accidentally introducing provider-specific branching."""
    source = inspect.getsource(bkf)
    lowered = source.lower()
    for banned in ("qwen", "openai", "anthropic", "claude", "llama.cpp", "gpt-", "gemini"):
        assert banned not in lowered, f"provider-specific reference found: {banned}"
    # No model-client / network imports -- pure string/dict logic only.
    assert "import requests" not in source
    assert "httpx" not in source
    assert "openai" not in lowered


# --- has_explicit_web_intent: Rule 4 / Phase H -------------------------------

def test_explicit_web_intent_true_for_spec_markers():
    for text in [
        "Search the web for the current USBC rule.",
        "What are the current specs for Storm X?",
        "Look up the latest PBA standings.",
        "What's the current WTBA ranking formula?",
        "What does the manufacturer currently list for this ball?",
        "Look this up online.",
        "Can you look it up for me online",
        "Is there an up-to-date list somewhere?",
    ]:
        assert has_explicit_web_intent(text), text


def test_explicit_web_intent_false_without_markers():
    for text in [
        "What's the WTBA world ranking points formula?",
        "How often should a bowling center resurface its lane machines?",
        "What are the specs for the IQ Tour Fusion?",
        "What does Mo Pinel say about RG versus differential?",
    ]:
        assert not has_explicit_web_intent(text), text


def test_explicit_web_intent_false_positive_protection():
    """Phase G/H: narrow marker words must not fire on unrelated substrings
    that merely contain the marker as part of a longer word."""
    for text in [
        "What is my internet speed?",
        "How fast is my CPU?",
        "What is differential privacy?",
        "Who is Mark Baker the musician?",
        "This research paper is concurrent with another study.",
        "Check out my new website design.",
    ]:
        assert not has_explicit_web_intent(text), text


def test_explicit_web_intent_empty_text():
    assert not has_explicit_web_intent("")
    assert not has_explicit_web_intent(None)


# --- latest_user_text ---------------------------------------------------------

def test_latest_user_text_finds_most_recent_real_user_message():
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer"},
        {"role": "tool", "content": "tool result", "tool_call_id": "x"},
        {"role": "user", "content": "second question"},
    ]
    assert latest_user_text(messages) == "second question"


def test_latest_user_text_skips_synthetic_messages():
    messages = [
        {"role": "user", "content": "real question"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "user", "content": "synthetic nudge", "_kanban_stop_synthetic": True},
    ]
    assert latest_user_text(messages) == "real question"


def test_latest_user_text_handles_multimodal_content():
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "part one"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "text", "text": "part two"},
        ]},
    ]
    assert latest_user_text(messages) == "part one\npart two"


def test_latest_user_text_empty_for_no_messages():
    assert latest_user_text([]) == ""
    assert latest_user_text(None) == ""


# --- bowling_knowledge_terminal_response: Rule 2/3/4 -------------------------

def _supported_json(answer="Grounded answer.", citations=None):
    return json.dumps({
        "comparison_mode": False,
        "answer": answer,
        "evidence_status": "STRONG",
        "validation_status": "ok",
        "sources": ["Mo Pinel"],
        "citations": citations if citations is not None else [{
            "claim_id": "c1", "source_name": "Mo Pinel",
            "document_title": "Doc Title", "locator": "p.1", "url": "https://example.test/doc",
        }],
        "warnings": [],
    })


def _unsupported_json(answer="Insufficient evidence message."):
    return json.dumps({
        "comparison_mode": False, "answer": answer,
        "evidence_status": "NO_EVIDENCE", "validation_status": "bypassed",
        "sources": [], "citations": [], "warnings": [],
    })


def test_wrong_tool_name_returns_none():
    assert bowling_knowledge_terminal_response(
        "web_search", _supported_json(), [{"role": "user", "content": "q"}]
    ) is None


def test_tool_error_shape_returns_none():
    error_result = json.dumps({"error": "Bowling Knowledge could not answer that question right now."})
    assert bowling_knowledge_terminal_response(
        BOWLING_KNOWLEDGE_TOOL_NAME, error_result, [{"role": "user", "content": "q"}]
    ) is None


def test_malformed_json_returns_none():
    assert bowling_knowledge_terminal_response(
        BOWLING_KNOWLEDGE_TOOL_NAME, "not json at all", [{"role": "user", "content": "q"}]
    ) is None


def test_non_string_result_returns_none():
    assert bowling_knowledge_terminal_response(
        BOWLING_KNOWLEDGE_TOOL_NAME, None, [{"role": "user", "content": "q"}]
    ) is None


def test_supported_answer_includes_answer_and_citations():
    messages = [{"role": "user", "content": "What does Mo Pinel say about RG?"}]
    response = bowling_knowledge_terminal_response(
        BOWLING_KNOWLEDGE_TOOL_NAME, _supported_json(), messages
    )
    assert response is not None
    assert "Grounded answer." in response
    assert "Sources:" in response
    assert "Mo Pinel" in response
    assert "Doc Title" in response
    assert "p.1" in response
    assert "https://example.test/doc" in response


def test_unsupported_answer_has_no_sources_block():
    messages = [{"role": "user", "content": "What's the WTBA world ranking points formula?"}]
    response = bowling_knowledge_terminal_response(
        BOWLING_KNOWLEDGE_TOOL_NAME, _unsupported_json(), messages
    )
    assert response == "Insufficient evidence message."
    assert "Sources:" not in response


def test_explicit_web_intent_in_user_message_suppresses_finalization():
    messages = [{"role": "user", "content": "Look up the current specs for the IQ Tour Fusion on the web."}]
    response = bowling_knowledge_terminal_response(
        BOWLING_KNOWLEDGE_TOOL_NAME, _unsupported_json(), messages
    )
    assert response is None

    response_supported = bowling_knowledge_terminal_response(
        BOWLING_KNOWLEDGE_TOOL_NAME, _supported_json(), messages
    )
    assert response_supported is None


def test_no_raw_evidence_or_internal_fields_leak_into_response():
    """Never expose evidence_status/validation_status/warnings/claim_id
    labels or DB-internal terms in the final user-facing text."""
    messages = [{"role": "user", "content": "What does Mo Pinel say about RG?"}]
    response = bowling_knowledge_terminal_response(
        BOWLING_KNOWLEDGE_TOOL_NAME, _supported_json(), messages
    )
    for leaked_term in ("validation_status", "evidence_status", "claim_id", "sqlite", "knowledge.db"):
        assert leaked_term not in response


def test_comparison_mode_keeps_experts_separate_and_labeled():
    comparison = json.dumps({
        "comparison_mode": True,
        "expert_sections": [
            {"source_name": "Mo Pinel", "answer": "Mo's take.", "citations": []},
            {"source_name": "Joe Slowinski", "answer": "Joe's take.", "citations": []},
        ],
    })
    messages = [{"role": "user", "content": "Compare Mo Pinel and Joe Slowinski."}]
    response = bowling_knowledge_terminal_response(
        BOWLING_KNOWLEDGE_TOOL_NAME, comparison, messages
    )
    assert "Mo's take." in response
    assert "Joe's take." in response
    assert "Mo Pinel" in response
    assert "Joe Slowinski" in response


def test_comparison_mode_empty_sections_returns_none():
    comparison = json.dumps({"comparison_mode": True, "expert_sections": []})
    messages = [{"role": "user", "content": "Compare X and Y."}]
    assert bowling_knowledge_terminal_response(
        BOWLING_KNOWLEDGE_TOOL_NAME, comparison, messages
    ) is None


def test_empty_answer_string_returns_none():
    empty = json.dumps({
        "comparison_mode": False, "answer": "", "citations": [], "warnings": [],
    })
    messages = [{"role": "user", "content": "q"}]
    assert bowling_knowledge_terminal_response(
        BOWLING_KNOWLEDGE_TOOL_NAME, empty, messages
    ) is None


# --- is_bowling_domain_question: Rule 1 / Phase G ----------------------------

def test_bowling_domain_true_positives():
    for text in [
        "What does Mo Pinel say about RG versus differential?",
        "How do I determine PAP?",
        "What is Quiet Eye in bowling?",
        "What affects ball motion?",
        "What did Mark Baker recommend for senior bowlers?",
        "What's the WTBA world ranking points formula?",
        "How often should a bowling center resurface its lane machines?",
        "What does Kegel say about lane topography?",
        "What did Joe Slowinski teach about release mechanics?",
    ]:
        assert is_bowling_domain_question(text), text


def test_bowling_domain_false_positives_protected():
    for text in [
        "What is my internet speed?",
        "How fast is my CPU?",
        "What is differential privacy?",
        "Who is Mark Baker the musician?",
        "Who is Mo Pinel?",  # bare name, no attribution -- ambiguous, not routed
    ]:
        assert not is_bowling_domain_question(text), text


def test_bowling_domain_empty_text():
    assert not is_bowling_domain_question("")
    assert not is_bowling_domain_question(None)


# --- bowling_knowledge_precall_block_reason: Rule 1 --------------------------

def test_precall_gate_blocks_web_search_for_bowling_domain_question():
    messages = [{"role": "user", "content": "What's the WTBA world ranking points formula?"}]
    reason = bowling_knowledge_precall_block_reason("web_search", messages, False)
    assert reason is not None
    assert "bowling_knowledge_answer" in reason


def test_precall_gate_blocks_web_extract_too():
    messages = [{"role": "user", "content": "What does Kegel say about lane topography?"}]
    reason = bowling_knowledge_precall_block_reason("web_extract", messages, False)
    assert reason is not None


def test_precall_gate_allows_once_bowling_knowledge_already_attempted():
    messages = [{"role": "user", "content": "What's the WTBA world ranking points formula?"}]
    assert bowling_knowledge_precall_block_reason("web_search", messages, True) is None


def test_precall_gate_allows_non_bowling_question():
    messages = [{"role": "user", "content": "What's the weather like today?"}]
    assert bowling_knowledge_precall_block_reason("web_search", messages, False) is None


def test_precall_gate_allows_explicit_web_intent_even_for_bowling_question():
    messages = [{"role": "user", "content": "Search the web for the current WTBA world ranking formula."}]
    assert bowling_knowledge_precall_block_reason("web_search", messages, False) is None


def test_precall_gate_never_touches_other_tools():
    messages = [{"role": "user", "content": "What's the WTBA world ranking points formula?"}]
    for tool in ("terminal", "file", "memory", "clarify", "bowling_knowledge_answer"):
        assert bowling_knowledge_precall_block_reason(tool, messages, False) is None


def test_precall_gate_empty_messages_never_blocks():
    assert bowling_knowledge_precall_block_reason("web_search", [], False) is None
    assert bowling_knowledge_precall_block_reason("web_search", None, False) is None
