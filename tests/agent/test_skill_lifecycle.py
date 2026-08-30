import json
import shutil
from pathlib import Path

from agent.skill_finalization import (
    clear_skill_finalizers,
    finalize_skill_response,
    register_skill_finalizer,
)
from agent.skill_lifecycle import (
    clear_prepared_synthesis,
    get_prepared_synthesis,
    install_synthesis_handoff,
    register_prepared_synthesis,
    synthesis_handoff_text,
    tools_for_synthesis,
)
from tools import skills_tool


def _skill(tmp_path: Path, *, status="READY", handoff=True, prepared=None):
    root = tmp_path / f"skill-{status}-{handoff}"
    root.mkdir()
    validator = root / "validate.py"
    validator.write_text(
        "import json,sys; p=json.load(sys.stdin); c=json.loads(p['candidate']); "
        "json.dump({'status':'valid','content':json.dumps({'manifest':c," 
        "'evidence':p['prepared_context']},sort_keys=True),'violations':[]},sys.stdout)",
        encoding="utf-8",
    )
    return {
        "name": "synthetic",
        "_source_path": str(root / "SKILL.md"),
        "metadata": {"hermes": {
            "preparation": {
                "required": True,
                **({"handoff": "constrained_synthesis"} if handoff else {}),
            },
            "finalization": {"required": True, "script": "validate.py"},
        }},
        "preparation": {"status": status, "context": prepared or {"fact": 7}},
    }


def setup_function():
    clear_prepared_synthesis()
    clear_skill_finalizers()


def test_ready_declared_skill_arms_task_scoped_tool_free_handoff(tmp_path):
    skill = _skill(tmp_path)
    register_prepared_synthesis("task", skill, "request")
    state = get_prepared_synthesis("task", "request")
    assert state["phase"] == "constrained_synthesis"
    assert state["skill"] == "synthetic"
    assert state["prepared_context"] == {"fact": 7}
    ordinary = [
        {"function": {"name": name}}
        for name in ("terminal", "read_file", "search_files", "skill_view")
    ]
    assert tools_for_synthesis(state, ordinary) == []
    assert tools_for_synthesis(None, ordinary) is ordinary

    transcript = [
        {"role": "system", "content": "stable-system-prefix"},
        {"role": "user", "content": "request"},
        {"role": "assistant", "tool_calls": [{"id": "call"}]},
        {"role": "tool", "name": "skill_view", "content": '<persisted-output>{"name": "synthetic"} read_file me'},
    ]
    api_copy = [dict(message) for message in transcript]
    assert install_synthesis_handoff(api_copy, state)
    handoff = json.loads(api_copy[-1]["content"])
    assert handoff["prepared_context"] == {"fact": 7}
    assert "read_file" not in api_copy[-1]["content"]
    assert transcript[-1]["content"].startswith("<persisted-output>")
    assert api_copy[0] == transcript[0]
    assert [message["role"] for message in api_copy] == ["system", "user", "assistant", "tool"]


def test_limited_failed_and_undeclared_skills_do_not_arm(tmp_path):
    for index, skill in enumerate((
        _skill(tmp_path, status="LIMITED"),
        _skill(tmp_path, status="FAILED"),
        _skill(tmp_path, handoff=False),
    )):
        register_prepared_synthesis(f"task-{index}", skill, "request")
        assert get_prepared_synthesis(f"task-{index}") is None


def test_new_request_clears_handoff_and_normal_task_stays_unconstrained(tmp_path):
    register_prepared_synthesis("task", _skill(tmp_path), "first")
    assert get_prepared_synthesis("task", "second") is None
    assert get_prepared_synthesis("ordinary", "anything") is None


def test_valid_manifest_reaches_finalizer_with_prepared_evidence(tmp_path):
    skill = _skill(tmp_path, prepared={"authoritative": True})
    register_skill_finalizer("task", skill)
    register_prepared_synthesis("task", skill, "request")
    result = finalize_skill_response("task", '{"schema":"fixture_v1"}', [])
    rendered = json.loads(result["content"])
    assert rendered["manifest"]["schema"] == "fixture_v1"
    assert rendered["evidence"] == {"authoritative": True}


def test_malformed_or_missing_manifest_fails_closed_without_clearing_phase(tmp_path):
    skill = _skill(tmp_path)
    register_skill_finalizer("task", skill)
    register_prepared_synthesis("task", skill, "request")
    result = finalize_skill_response("task", "", [])
    assert result["status"] == "invalid"
    assert result["violations"][0]["code"] == "validator_error"


def test_real_pattern_play_ready_handoff_with_temporary_hermes_home(tmp_path, monkeypatch):
    """Exercise the installed declaration, preparation script, and resolvers."""
    source_home = Path(__file__).resolve().parents[3]
    temp_home = tmp_path / "hermes-home"
    (temp_home / "skills").mkdir(parents=True)
    for name in (
        "bowling-pattern-play", "bowling-patterns", "bowling-equipment",
        "bowling-principles",
    ):
        shutil.copytree(source_home / "skills" / name, temp_home / "skills" / name)
    shutil.copytree(source_home / "bowling", temp_home / "bowling")
    (temp_home / "hermes-agent").symlink_to(source_home / "hermes-agent")
    monkeypatch.setenv("HERMES_HOME", str(temp_home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", temp_home / "skills")
    skills_tool.reset_skill_view_dedup()

    task = "real-pattern-play"
    request = "Recommend equipment and targets for bowling on Boardwalk V2"
    result = json.loads(skills_tool._skill_view_with_bump(
        {"name": "bowling-pattern-play"}, task_id=task, user_task=request,
    ))
    state = get_prepared_synthesis(task, request)
    assert result["preparation"]["status"] == "READY", result["preparation"]["context"]["pattern"]
    assert state["phase"] == "constrained_synthesis"
    assert state["prepared_context"]["availability"] == {
        "pattern_ready": True, "principles_ready": True, "arsenal_ready": True,
    }
    forbidden = {"terminal", "read_file", "search_files", "skill_view"}
    ordinary = [{"function": {"name": name}} for name in forbidden]
    assert tools_for_synthesis(state, ordinary) == []
    handoff = json.loads(synthesis_handoff_text(state))
    assert handoff["prepared_context"]["schema"] == "bowling_pattern_play_context_v1"
