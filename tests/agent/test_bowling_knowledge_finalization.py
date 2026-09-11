"""Unit tests for agent.bowling_knowledge_finalization (pure logic, no agent)."""

import inspect
import json

from types import SimpleNamespace

import agent.bowling_knowledge_finalization as bkf
from agent.bowling_knowledge_finalization import (
    BOWLING_KNOWLEDGE_TOOL_NAME,
    bowling_knowledge_pending_in_tool_calls,
    bowling_knowledge_precall_block_reason,
    bowling_knowledge_terminal_response,
    has_explicit_web_intent,
    is_bowling_domain_question,
    latest_user_text,
)


def _tool_call(name, call_id="c1"):
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments="{}"))


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


def test_weak_evidence_relayed_as_terminal_bounded_message():
    """WEAK (like NO_EVIDENCE) must be terminal: the tool's own bounded
    unsupported message is relayed verbatim, no citations block, no
    supplementation."""
    weak = json.dumps({
        "comparison_mode": False,
        "answer": "The Bowling Knowledge corpus does not currently contain "
                  "enough relevant sourced evidence to answer that confidently.",
        "evidence_status": "WEAK", "validation_status": "bypassed",
        "sources": [], "citations": [],
        "warnings": ["Only a single incidental search term appears to drive the top match."],
    })
    messages = [{"role": "user", "content": "How should I use a balance hole to change my bowling ball reaction?"}]
    response = bowling_knowledge_terminal_response(BOWLING_KNOWLEDGE_TOOL_NAME, weak, messages)
    assert response == (
        "The Bowling Knowledge corpus does not currently contain enough "
        "relevant sourced evidence to answer that confidently."
    )
    assert "Sources:" not in response
    # No leaked drilling/mass-distribution/rev-rate/axis-tilt style advice --
    # the module never invents text beyond the tool's own answer field.
    for leaked in ("mass distribution", "rev rate", "axis tilt", "balance hole"):
        assert leaked not in response.lower()


def test_ambiguous_evidence_keeps_multiple_angles_explicit():
    """AMBIGUOUS: the underlying pipeline's multi-angle answer text is relayed
    as-is -- the finalization layer must not collapse or rewrite it into a
    single confident claim."""
    ambiguous = json.dumps({
        "comparison_mode": False,
        "answer": (
            "Evidence spans two distinct angles: some claims describe breakpoint "
            "shape as driven by layout, others by coverstock chemistry. The "
            "corpus does not settle which dominates for your setup."
        ),
        "evidence_status": "AMBIGUOUS", "validation_status": "ok",
        "sources": ["Mo Pinel"],
        "citations": [{
            "claim_id": "c9", "source_name": "Mo Pinel",
            "document_title": "Doc", "locator": "p.9", "url": None,
        }],
        "warnings": [],
    })
    messages = [{"role": "user", "content": "What determines breakpoint shape?"}]
    response = bowling_knowledge_terminal_response(BOWLING_KNOWLEDGE_TOOL_NAME, ambiguous, messages)
    assert "two distinct angles" in response
    assert "layout" in response and "coverstock chemistry" in response
    assert "does not settle which dominates" in response


def test_approve_with_context_mandatory_caveat_survives_relay():
    """A citation's mandatory_caveat (approve_with_context adjudication) is
    part of the tool's own evidence-backed answer text; relaying that text
    verbatim must not strip or alter the caveat substance."""
    caveat_text = (
        "Differential ratio can help describe the ball's post-transition shape, "
        "but it does not independently guarantee a sharper breakpoint because "
        "coverstock, layout, release, and lane conditions prevent it from being "
        "deterministic."
    )
    usable = json.dumps({
        "comparison_mode": False,
        "answer": caveat_text,
        "evidence_status": "USABLE", "validation_status": "ok",
        "sources": ["Mo Pinel"],
        "citations": [{
            "claim_id": 181, "source_name": "Mo Pinel",
            "document_title": "The Truth about Drilled Bowling Balls",
            "locator": "pdf_page P30", "url": None,
            "adjudication": "approve_with_context",
            "mandatory_caveat": (
                "State that cover, layout, release, and lane condition prevent "
                "differential ratio from being deterministic."
            ),
        }],
        "warnings": [],
    })
    messages = [{"role": "user", "content": "Does differential ratio determine breakpoint sharpness?"}]
    response = bowling_knowledge_terminal_response(BOWLING_KNOWLEDGE_TOOL_NAME, usable, messages)
    assert caveat_text in response
    assert "does not independently guarantee" in response
    assert "Sources:" in response and "Mo Pinel" in response


# --- bowling_knowledge_pending_in_tool_calls: Rule 5 -------------------------

def test_pending_true_when_bowling_knowledge_answer_is_among_tool_calls():
    assert bowling_knowledge_pending_in_tool_calls([_tool_call("bowling_knowledge_answer")])


def test_pending_true_in_mixed_batch_with_other_tools():
    assert bowling_knowledge_pending_in_tool_calls([
        _tool_call("web_search", "c1"),
        _tool_call("bowling_knowledge_answer", "c2"),
    ])


def test_pending_false_when_absent():
    assert not bowling_knowledge_pending_in_tool_calls([_tool_call("web_search")])


def test_pending_false_for_empty_or_none():
    assert not bowling_knowledge_pending_in_tool_calls([])
    assert not bowling_knowledge_pending_in_tool_calls(None)


def test_pending_true_with_dict_shaped_tool_calls():
    """Some call sites (tests, alternate transports) represent tool calls as
    plain dicts rather than SDK objects -- the predicate must handle both."""
    assert bowling_knowledge_pending_in_tool_calls([
        {"id": "c1", "function": {"name": "bowling_knowledge_answer", "arguments": "{}"}},
    ])


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
