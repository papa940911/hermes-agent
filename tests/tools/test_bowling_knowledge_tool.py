"""Tests for the native `bowling_knowledge_answer` tool registration and
dispatch (tools/bowling_knowledge_tool.py). The pipeline itself (evidence
planning, model call, validation, citations) is fully covered by
skills/bowling-knowledge/tests/ -- these tests only cover the Hermes-tool
surface: schema shape, registry registration, dispatch/error-handling
behavior, and that this tool cannot accept arbitrary path arguments.

Uses monkeypatching of the adapter, never a live model or the production
database.
"""

import json

import pytest

from tools import bowling_knowledge_tool as bkt
from tools.registry import registry


class _FakeAdapter:
    """Stand-in for skills/bowling-knowledge/scripts/hermes_tool.py,
    injected via monkeypatching `bkt._adapter` so these tests never touch
    the real adapter, a database, or a model."""

    class BowlingKnowledgeToolError(RuntimeError):
        pass

    def __init__(self, *, result=None, raise_tool_error=None, raise_other=None):
        self._result = result
        self._raise_tool_error = raise_tool_error
        self._raise_other = raise_other
        self.calls = []

    def answer_bowling_knowledge_question(self, question, *, top_k=None, corpus_only=True):
        self.calls.append({"question": question, "top_k": top_k, "corpus_only": corpus_only})
        if self._raise_tool_error is not None:
            raise self.BowlingKnowledgeToolError(self._raise_tool_error)
        if self._raise_other is not None:
            raise self._raise_other
        return self._result


@pytest.fixture(autouse=True)
def _reset_adapter():
    bkt._adapter = None
    yield
    bkt._adapter = None


# --- 1: schema contains the Bowling Knowledge tool ----------------------------

def test_tool_is_registered():
    entry = registry.get_entry("bowling_knowledge_answer")
    assert entry is not None
    assert entry.toolset == "bowling_knowledge"
    assert entry.schema["name"] == "bowling_knowledge_answer"
    assert entry.is_async is False


def test_tool_stays_directly_visible_in_model_schema():
    """Platform opt-in must expose the native tool itself, not defer it
    behind Tool Search as though it were an MCP/plugin capability."""
    from model_tools import get_tool_definitions

    definitions = get_tool_definitions(
        enabled_toolsets=["bowling_knowledge"], quiet_mode=True,
    )
    offered_names = [definition["function"]["name"] for definition in definitions]

    assert offered_names == ["bowling_knowledge_answer"]


# --- 2: question required -----------------------------------------------------

def test_question_is_required_in_schema():
    schema = registry.get_schema("bowling_knowledge_answer")
    assert schema["parameters"]["required"] == ["question"]


# --- 3: top_k bounded in schema -------------------------------------------------

def test_top_k_bounded_in_schema():
    schema = registry.get_schema("bowling_knowledge_answer")
    top_k_schema = schema["parameters"]["properties"]["top_k"]
    assert top_k_schema["type"] == "integer"
    assert top_k_schema["minimum"] == 1
    assert top_k_schema["maximum"] == 10


# --- 4: corpus_only defaults true ------------------------------------------------

def test_corpus_only_defaults_true_in_schema():
    schema = registry.get_schema("bowling_knowledge_answer")
    assert schema["parameters"]["properties"]["corpus_only"]["default"] is True


def test_corpus_only_defaults_true_in_handler(monkeypatch):
    fake = _FakeAdapter(result={"comparison_mode": False, "answer": "ok"})
    monkeypatch.setattr(bkt, "_adapter", fake)
    bkt.bowling_knowledge_answer_tool(question="anything")
    assert fake.calls[0]["corpus_only"] is True


# --- 13: no arbitrary path arguments accepted -----------------------------------

def test_schema_exposes_no_path_or_endpoint_arguments():
    schema = registry.get_schema("bowling_knowledge_answer")
    allowed = {"question", "top_k", "corpus_only"}
    assert set(schema["parameters"]["properties"].keys()) == allowed
    # explicitly guard against the exact controls Phase C forbids exposing
    forbidden_substrings = ("db", "path", "source", "endpoint", "sql", "url", "model")
    for prop_name in schema["parameters"]["properties"]:
        lowered = prop_name.lower()
        assert not any(bad in lowered for bad in forbidden_substrings), prop_name


# --- 6: supported answer returns validated response (via dispatch) --------------

def test_dispatch_supported_answer(monkeypatch):
    fake = _FakeAdapter(result={
        "comparison_mode": False,
        "answer": "Grounded answer.",
        "evidence_status": "STRONG",
        "validation_status": "ok",
        "sources": ["Mo Pinel"],
        "citations": [{"claim_id": 1, "source_name": "Mo Pinel", "document_title": "Doc", "locator": "P1", "url": None}],
        "warnings": [],
    })
    monkeypatch.setattr(bkt, "_adapter", fake)
    out = json.loads(registry.dispatch("bowling_knowledge_answer", {"question": "What does Mo Pinel say?"}))
    assert out["answer"] == "Grounded answer."
    assert out["validation_status"] == "ok"
    assert fake.calls == [{"question": "What does Mo Pinel say?", "top_k": None, "corpus_only": True}]


# --- 7: unsupported answer bypasses model (verified at adapter layer already;
#        here we only confirm the tool passes the adapter's bypass result through
#        untouched) ------------------------------------------------------------

def test_dispatch_unsupported_passthrough(monkeypatch):
    fake = _FakeAdapter(result={
        "comparison_mode": False,
        "answer": "The Bowling Knowledge corpus does not currently contain enough relevant sourced evidence to answer that confidently.",
        "evidence_status": "WEAK",
        "validation_status": "bypassed",
        "sources": [],
        "citations": [],
        "warnings": [],
    })
    monkeypatch.setattr(bkt, "_adapter", fake)
    out = json.loads(registry.dispatch("bowling_knowledge_answer", {"question": "unsupported question"}))
    assert out["validation_status"] == "bypassed"
    assert out["citations"] == []


# --- 8: model unavailable returns a safe operational error, not an exception ----

def test_dispatch_model_unavailable_result_passthrough(monkeypatch):
    fake = _FakeAdapter(result={
        "comparison_mode": False,
        "answer": "Bowling Knowledge evidence was found, but the configured local answer model is unavailable.",
        "evidence_status": "STRONG",
        "validation_status": "model_unavailable",
        "sources": [],
        "citations": [],
        "warnings": [],
    })
    monkeypatch.setattr(bkt, "_adapter", fake)
    out = json.loads(registry.dispatch("bowling_knowledge_answer", {"question": "anything"}))
    assert out["validation_status"] == "model_unavailable"
    assert "error" not in out


# --- 9: malformed/internal errors never leak ------------------------------------

def test_bounded_tool_error_never_leaks_internal_detail(monkeypatch):
    fake = _FakeAdapter(raise_other=RuntimeError("sqlite3.OperationalError: /secret/path/knowledge.db locked"))
    monkeypatch.setattr(bkt, "_adapter", fake)
    out = json.loads(registry.dispatch("bowling_knowledge_answer", {"question": "anything"}))
    assert "error" in out
    assert "/secret/path" not in out["error"]
    assert "sqlite3" not in out["error"]


def test_bounded_operational_error_message_passthrough(monkeypatch):
    fake = _FakeAdapter(raise_tool_error="A non-empty question is required.")
    monkeypatch.setattr(bkt, "_adapter", fake)
    out = json.loads(registry.dispatch("bowling_knowledge_answer", {"question": ""}))
    assert out["error"] == "A non-empty question is required."


# --- 14: no external-provider fallback ------------------------------------------

def test_handler_never_imports_a_paid_provider_client():
    import inspect
    source = inspect.getsource(bkt)
    for forbidden in ("openai.OpenAI", "anthropic.Anthropic", "AsyncOpenAI"):
        assert forbidden not in source


# --- 15: protected runtime behavior for other tools is unchanged ----------------

def test_other_tools_remain_registered_and_dispatchable():
    from tools import close_terminal_tool  # noqa: F401 - importing self-registers it

    assert registry.get_entry("close_terminal") is not None
    assert registry.get_tool_to_toolset_map()["close_terminal"] == "desktop_ui"
