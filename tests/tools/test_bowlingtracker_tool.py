"""Native-tool surface tests for read-only personal BowlingTracker data."""

import json

import pytest

from tools import bowling_knowledge_tool as _bowling_knowledge_tool  # noqa: F401
from tools import bowlingtracker_tool as btt
from tools.registry import registry


class FakeAdapter:
    class BowlingTrackerError(RuntimeError):
        pass

    class API:
        BowlingTrackerError = RuntimeError

    api = API()

    def __init__(self, result=None, error=None):
        self.result = result or {
            "contractVersion": "personal-bowling-evidence/v1",
            "status": "ok",
            "dataSource": "BowlingTracker",
            "evidenceKinds": ["personal_observation"],
        }
        self.error = error
        self.calls = []

    def _call(self, name, arguments):
        self.calls.append((name, arguments))
        if self.error:
            raise self.error
        return self.result

    def recent_sessions(self, **arguments):
        return self._call("recent_sessions", arguments)

    def session_detail(self, **arguments):
        return self._call("session_detail", arguments)

    def personal_summary(self, **arguments):
        return self._call("personal_summary", arguments)

    def session_analysis(self, **arguments):
        return self._call("session_analysis", arguments)

    def failure_result(self, error):
        return {
            "contractVersion": "personal-bowling-evidence/v1",
            "status": "unavailable",
            "dataSource": "BowlingTracker",
            "evidenceKinds": [],
            "error": {"code": "service_unavailable", "message": "Personal data unavailable."},
            "evidence": [],
        }


@pytest.fixture(autouse=True)
def reset_adapter():
    btt._adapter = None
    yield
    btt._adapter = None


def test_four_tools_registered_under_distinct_toolset():
    names = {
        "bowlingtracker_recent_sessions",
        "bowlingtracker_session_detail",
        "bowlingtracker_personal_summary",
        "bowlingtracker_session_analysis",
    }
    assert set(registry.get_tool_names_for_toolset("bowlingtracker")) == names
    for name in names:
        entry = registry.get_entry(name)
        assert entry is not None
        assert entry.toolset == "bowlingtracker"
        assert entry.is_async is False


def test_tools_visible_together_in_model_schema():
    from model_tools import get_tool_definitions

    definitions = get_tool_definitions(enabled_toolsets=["bowlingtracker"], quiet_mode=True)
    assert {item["function"]["name"] for item in definitions} == {
        "bowlingtracker_recent_sessions",
        "bowlingtracker_session_detail",
        "bowlingtracker_personal_summary",
        "bowlingtracker_session_analysis",
    }


def test_session_detail_requires_uuid_and_shots_default_false():
    schema = registry.get_schema("bowlingtracker_session_detail")["parameters"]
    assert schema["required"] == ["session_id"]
    assert schema["properties"]["session_id"]["format"] == "uuid"
    assert schema["properties"]["include_shots"]["default"] is False


def test_filter_and_limit_schema_bounds():
    recent = registry.get_schema("bowlingtracker_recent_sessions")["parameters"]["properties"]
    summary = registry.get_schema("bowlingtracker_personal_summary")["parameters"]["properties"]
    assert recent["limit"]["minimum"] == 1
    assert recent["limit"]["maximum"] == 20
    assert summary["game_limit"]["minimum"] == 1
    assert summary["game_limit"]["maximum"] == 50
    assert summary["session_limit"]["minimum"] == 1
    assert summary["session_limit"]["maximum"] == 20
    assert summary["session_limit"]["default"] == 20
    assert summary["session_type"]["enum"] == [
        "Tournament",
        "Sport Shot League",
        "League",
        "Practice",
    ]


def test_schemas_expose_no_write_sync_sql_path_endpoint_or_credentials():
    forbidden = (
        "write",
        "update",
        "delete",
        "sync",
        "watermark",
        "sql",
        "path",
        "endpoint",
        "token",
        "password",
        "credential",
        "database",
    )
    for name in registry.get_tool_names_for_toolset("bowlingtracker"):
        properties = registry.get_schema(name)["parameters"]["properties"]
        for property_name in properties:
            assert not any(term in property_name.casefold() for term in forbidden), property_name


def test_dispatch_passes_structured_filters_to_adapter(monkeypatch):
    fake = FakeAdapter()
    monkeypatch.setattr(btt, "_adapter", fake)
    output = json.loads(
        registry.dispatch(
            "bowlingtracker_recent_sessions",
            {"session_type": "Tournament", "limit": 3},
        )
    )
    assert output["dataSource"] == "BowlingTracker"
    assert fake.calls == [("recent_sessions", {"session_type": "Tournament", "limit": 3})]


def test_detail_handler_defaults_shots_false(monkeypatch):
    fake = FakeAdapter()
    monkeypatch.setattr(btt, "_adapter", fake)
    registry.dispatch(
        "bowlingtracker_session_detail",
        {"session_id": "11111111-1111-1111-1111-111111111111"},
    )
    assert fake.calls == [
        (
            "session_detail",
            {
                "session_id": "11111111-1111-1111-1111-111111111111",
                "include_shots": False,
            },
        )
    ]


def test_adapter_failure_returns_source_specific_envelope(monkeypatch):
    fake = FakeAdapter(error=RuntimeError("secret details must not escape"))
    monkeypatch.setattr(btt, "_adapter", fake)
    output = json.loads(registry.dispatch("bowlingtracker_personal_summary", {}))
    assert output["status"] == "unavailable"
    assert output["dataSource"] == "BowlingTracker"
    assert output["error"]["code"] == "service_unavailable"
    assert "secret details" not in json.dumps(output)


def test_descriptions_separate_personal_data_from_expert_knowledge():
    personal_descriptions = " ".join(
        registry.get_schema(name)["description"]
        for name in registry.get_tool_names_for_toolset("bowlingtracker")
    )
    knowledge_description = registry.get_schema("bowling_knowledge_answer")["description"]
    assert "personal" in personal_descriptions.casefold()
    assert "expert" in personal_descriptions.casefold()
    assert "personal records" in knowledge_description.casefold()
    assert "bowlingtracker" in knowledge_description.casefold()


def test_tool_module_is_provider_and_platform_independent():
    source = btt.__file__ and open(btt.__file__, encoding="utf-8").read()
    lowered = source.casefold()
    assert "qwen" not in lowered
    assert "anthropic" not in lowered
    assert "openai" not in lowered
    assert "telegram" not in lowered


def test_session_analysis_schema_is_preferred_and_needs_no_uuid():
    schema = registry.get_schema("bowlingtracker_session_analysis")
    description = schema["description"]
    assert description.startswith("PREFERRED tool")
    for phrase in ("comparing two sessions", "why did I score lower", "never invent"):
        assert phrase.casefold() in (description + json.dumps(schema)).casefold(), phrase
    for phrase in ("causes", "better/worse"):
        assert phrase in description
    parameters = schema["parameters"]
    assert "required" not in parameters
    assert parameters["properties"]["question_focus"]["enum"] == [
        "overview", "cause", "shot_or_frame", "game", "ball", "better_or_worse", "baseline",
    ]
    assert parameters["properties"]["game_number"]["minimum"] == 1
    detail = registry.get_schema("bowlingtracker_session_detail")["description"]
    assert "prefer bowlingtracker_session_analysis" in detail


def test_session_analysis_dispatch_passes_only_declared_arguments(monkeypatch):
    fake = FakeAdapter()
    monkeypatch.setattr(btt, "_adapter", fake)
    registry.dispatch(
        "bowlingtracker_session_analysis",
        {
            "session_name": "Practice",
            "compare_session_date": "2026-09-19",
            "question_focus": "cause",
            "get": "injected",
            "url": "http://elsewhere",
        },
    )
    assert fake.calls == [(
        "session_analysis",
        {
            "user_question": None,
            "session_name": "Practice",
            "compare_session_date": "2026-09-19",
            "question_focus": "cause",
        },
    )]


def test_session_analysis_adapter_failure_uses_safe_envelope(monkeypatch):
    fake = FakeAdapter(error=RuntimeError("secret details must not escape"))
    monkeypatch.setattr(btt, "_adapter", fake)
    result = json.loads(
        registry.dispatch("bowlingtracker_session_analysis", {"session_name": "Practice"})
    )
    assert result["status"] == "unavailable"
    assert "secret" not in json.dumps(result)


def test_lower_level_tools_still_dispatch_to_their_operations(monkeypatch):
    fake = FakeAdapter()
    monkeypatch.setattr(btt, "_adapter", fake)
    registry.dispatch("bowlingtracker_recent_sessions", {"limit": 2})
    registry.dispatch("bowlingtracker_session_detail", {"session_id": "x"})
    registry.dispatch("bowlingtracker_personal_summary", {"session_limit": 1})
    assert [name for name, _ in fake.calls] == [
        "recent_sessions", "session_detail", "personal_summary",
    ]


def test_session_analysis_receives_user_task_not_model_supplied_question(monkeypatch):
    fake = FakeAdapter()
    monkeypatch.setattr(btt, "_adapter", fake)
    registry.dispatch(
        "bowlingtracker_session_analysis",
        {"question_focus": "baseline", "user_question": "model-invented text"},
        user_task="Why did I score lower in Practice?",
    )
    assert fake.calls == [(
        "session_analysis",
        {"question_focus": "baseline", "user_question": "Why did I score lower in Practice?"},
    )]
