import json
from pathlib import Path

from agent.skill_finalization import (
    clear_skill_finalizers, finalize_skill_response, register_skill_finalizer,
)
from agent.skill_preparation import run_skill_preparation
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
