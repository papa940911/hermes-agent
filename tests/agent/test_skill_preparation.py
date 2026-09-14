import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.skill_finalization import (
    clear_skill_finalizers, finalize_skill_response, register_skill_finalizer,
)
from agent.skill_preparation import (
    MAX_SURFACE_CONTEXT_BYTES,
    detach_surface_context,
    run_skill_preparation,
    surface_context_from_event_metadata,
)
from tools import skills_tool


def _skill(tmp_path: Path, script_body: str, status="READY"):
    root = tmp_path / "synthetic"
    root.mkdir()
    (root / "prepare.py").write_text(script_body, encoding="utf-8")
    validator = root / "validate.py"
    validator.write_text(
        "import json,sys; p=json.load(sys.stdin); "
        "json.dump({'status':'valid','content':p['candidate'],'violations':[]},sys.stdout)",
        encoding="utf-8",
    )
    return {
        "name": "synthetic", "_source_path": str(root / "SKILL.md"),
        "metadata": {"hermes": {
            "preparation": {"required": True, "script": "prepare.py"},
            "finalization": {"required": True, "script": "validate.py"},
        }},
    }


def test_required_preparation_is_runtime_executed(tmp_path):
    skill = _skill(
        tmp_path,
        "import json,sys; p=json.load(sys.stdin); "
        "json.dump({'status':'READY','context':{'request':p['user_request']}},sys.stdout)",
    )
    result = run_skill_preparation(
        skill, user_task="synthetic request", task_id="task",
        capability_dependencies=[],
    )
    assert result["status"] == "READY"
    assert result["context"]["request"] == "synthetic request"


def test_context_absent_preserves_existing_preparation_payload(tmp_path):
    skill = _skill(
        tmp_path,
        "import json,sys; p=json.load(sys.stdin); "
        "json.dump({'status':'READY','context':{'keys':sorted(p)}},sys.stdout)",
    )
    result = run_skill_preparation(
        skill, user_task="request", task_id="task", capability_dependencies=[]
    )
    assert result["context"]["keys"] == [
        "capability_dependencies", "entity_resolvers", "protocol", "task_id",
        "user_request",
    ]


def test_explicit_allowlisted_context_reaches_preparation_detached(tmp_path):
    skill = _skill(
        tmp_path,
        "import json,sys; p=json.load(sys.stdin); "
        "json.dump({'status':'READY','context':p['surface_context']},sys.stdout)",
    )
    original = {"schema": "example/v1", "selection": {"id": "A"}}
    result = run_skill_preparation(
        skill, user_task="request", task_id="task", capability_dependencies=[],
        surface_context=original,
    )
    original["selection"]["id"] = "mutated"
    assert result["context"]["selection"]["id"] == "A"


def test_event_metadata_allowlist_drops_arbitrary_and_invalid_values():
    assert surface_context_from_event_metadata({"private": {"secret": True}}) is None
    assert surface_context_from_event_metadata({"skill_context": "not-an-object"}) is None
    assert surface_context_from_event_metadata({
        "skill_context": {"payload": "x" * MAX_SURFACE_CONTEXT_BYTES}
    }) is None
    assert surface_context_from_event_metadata({
        "skill_context": {"payload": object()}
    }) is None
    accepted = {"schema": "example/v1", "payload": {"id": "one"}}
    copied = surface_context_from_event_metadata({
        "skill_context": accepted,
        "gateway_session_id": "must-not-leak",
    })
    accepted["payload"]["id"] = "two"
    assert copied == {"schema": "example/v1", "payload": {"id": "one"}}


def test_invalid_direct_context_fails_preparation_without_running_hook(tmp_path):
    skill = _skill(tmp_path, "raise RuntimeError('must not execute')")
    result = run_skill_preparation(
        skill, user_task="request", task_id="task", capability_dependencies=[],
        surface_context={"payload": {"bad": object()}},
    )
    assert result["status"] == "FAILED"
    assert result["error"].startswith("SurfaceContextValidationError:")


def test_context_is_request_local_and_concurrent_preparations_are_isolated(tmp_path):
    def run(label):
        root = tmp_path / label
        root.mkdir()
        script = root / "prepare.py"
        script.write_text(
            "import json,sys; p=json.load(sys.stdin); "
            "json.dump({'status':'READY','context':p.get('surface_context')},sys.stdout)",
            encoding="utf-8",
        )
        skill = {
            "name": label,
            "_source_path": str(root / "SKILL.md"),
            "metadata": {"hermes": {"preparation": {
                "required": True, "script": "prepare.py"
            }}},
        }
        return run_skill_preparation(
            skill, user_task="request", task_id=label,
            capability_dependencies=[], surface_context={"request": label},
        )["context"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(run, ("request-a", "request-b"))
    assert first == {"request": "request-a"}
    assert second == {"request": "request-b"}

    no_context_root = tmp_path / "request-c"
    no_context_root.mkdir()
    no_context_skill = {
        "name": "request-c",
        "_source_path": str(no_context_root / "SKILL.md"),
        "metadata": {"hermes": {"preparation": {
            "required": True, "script": "prepare.py"
        }}},
    }
    (no_context_root / "prepare.py").write_text(
        "import json,sys; p=json.load(sys.stdin); "
        "json.dump({'status':'READY','context':p.get('surface_context')},sys.stdout)",
        encoding="utf-8",
    )
    assert run_skill_preparation(
        no_context_skill, user_task="request", task_id="request-c",
        capability_dependencies=[],
    )["context"] is None


def test_reserved_and_non_string_keys_are_rejected():
    for value in ({"task_id": "override"}, {1: "not-json-object-compatible"}, {"_x": 1}):
        with pytest.raises(ValueError):
            detach_surface_context(value)


def test_successful_skill_view_executes_declared_preparation(tmp_path, monkeypatch):
    root = tmp_path / "skills" / "synthetic"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        "---\nname: synthetic\ndescription: synthetic fixture\nmetadata:\n"
        "  hermes:\n    preparation:\n      required: true\n"
        "      protocol: json_stdio_v1\n      script: prepare.py\n---\nBody\n",
        encoding="utf-8",
    )
    (root / "prepare.py").write_text(
        "import json,sys; p=json.load(sys.stdin); "
        "json.dump({'status':'READY','context':{'request':p['user_request']}},sys.stdout)",
        encoding="utf-8",
    )
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", tmp_path / "skills")
    skills_tool.reset_skill_view_dedup()
    result = json.loads(skills_tool._skill_view_with_bump(
        {"name": "synthetic"}, task_id="runtime-task", user_task="prepare me"
    ))
    assert result["preparation_status"] == "PREPARATION STATUS: READY"
    assert result["preparation"]["context"]["request"] == "prepare me"


def test_required_preparation_cannot_be_skipped_at_finalization(tmp_path):
    clear_skill_finalizers()
    skill = _skill(tmp_path, "raise SystemExit(1)")
    register_skill_finalizer("task", skill)
    result = finalize_skill_response("task", json.dumps({"x": 1}), [])
    assert result["status"] == "invalid"
    assert result["violations"][0]["code"] == "preparation_not_completed"


def test_limited_completed_preparation_reaches_existing_finalizer(tmp_path):
    clear_skill_finalizers()
    skill = _skill(tmp_path, "raise SystemExit(1)")
    skill["preparation"] = {"status": "LIMITED"}
    register_skill_finalizer("task", skill)
    result = finalize_skill_response("task", "candidate", [])
    assert result["status"] == "valid"
    assert result["content"] == "candidate"
