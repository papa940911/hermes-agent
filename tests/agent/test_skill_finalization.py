import json
from pathlib import Path

from agent.skill_finalization import (
    clear_skill_finalizers, finalize_skill_response,
    has_required_finalizer, register_skill_finalizer,
)


def test_skill_declared_finalizer_is_required_and_runs_without_model_call(tmp_path):
    skill = tmp_path / "skill"; scripts = skill / "scripts"; scripts.mkdir(parents=True)
    script = scripts / "validator.py"
    script.write_text(
        "import json,sys\np=json.load(sys.stdin)\njson.dump({'status':'valid','violations':[],'content':'safe'},sys.stdout)\n",
        encoding="utf-8",
    )
    skill_md = skill / "SKILL.md"; skill_md.write_text("x", encoding="utf-8")
    task = "finalizer-test"
    register_skill_finalizer(task, {
        "name": "test", "_source_path": str(skill_md),
        "metadata": {"hermes": {"finalization": {"required": True, "script": "scripts/validator.py"}}},
    })
    assert has_required_finalizer(task)
    assert finalize_skill_response(task, json.dumps({"x": 1}), [])["content"] == "safe"
    clear_skill_finalizers(task)


def test_missing_required_validator_fails_closed(tmp_path):
    skill_md = tmp_path / "SKILL.md"; skill_md.write_text("x", encoding="utf-8")
    task = "missing-finalizer"
    register_skill_finalizer(task, {
        "name": "test", "_source_path": str(skill_md),
        "metadata": {"hermes": {"finalization": {"required": True, "script": "scripts/missing.py"}}},
    })
    result = finalize_skill_response(task, "unsafe", [])
    assert result["status"] == "invalid"
    assert "validator is unavailable" in result["content"]
    clear_skill_finalizers(task)
