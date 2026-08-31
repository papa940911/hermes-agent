import json
import hashlib
import os
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


def _register_validator(tmp_path, task, result):
    skill = tmp_path / task
    scripts = skill / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "validator.py"
    script.write_text(
        "import json,sys\n"
        "p=json.load(sys.stdin)\n"
        f"r={result!r}\n"
        "r['candidate_seen']=p['candidate']\n"
        "json.dump(r,sys.stdout,ensure_ascii=False)\n",
        encoding="utf-8",
    )
    skill_md = skill / "SKILL.md"
    skill_md.write_text("x", encoding="utf-8")
    register_skill_finalizer(task, {
        "name": "fixture", "_source_path": str(skill_md),
        "metadata": {"hermes": {"finalization": {
            "required": True, "script": "scripts/validator.py",
        }}},
    })


def _artifact_scope(root):
    scopes = [path for path in root.iterdir() if path.is_dir()]
    assert len(scopes) == 1
    return scopes[0]


def test_diagnostics_disabled_writes_no_candidate(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_INTERNAL_FINALIZATION_DIAGNOSTICS_DIR", raising=False)
    _register_validator(tmp_path, "disabled", {
        "status": "valid", "violations": [], "content": "rendered",
    })
    output = tmp_path / "diagnostics"
    assert finalize_skill_response("disabled", "raw candidate", [])["content"] == "rendered"
    assert not output.exists()
    clear_skill_finalizers("disabled")


def test_enabled_diagnostics_preserve_exact_candidate_result_and_metadata(tmp_path, monkeypatch):
    output = tmp_path / "diagnostics"
    monkeypatch.setenv("HERMES_INTERNAL_FINALIZATION_DIAGNOSTICS_DIR", str(output))
    candidate = ' {"quoted":"a\\\\b","unicode":"雪"}\n'
    expected = {"status": "valid", "violations": [], "content": "rendered"}
    _register_validator(tmp_path, "enabled/task", expected)

    result = finalize_skill_response(
        "enabled/task", candidate, [], finish_reason="stop",
        provider_transport_streaming=True,
    )
    scope = _artifact_scope(output)
    assert (scope / "pre_finalization_candidate.txt").read_bytes() == candidate.encode("utf-8")
    persisted = json.loads((scope / "finalization_result.json").read_text())
    assert persisted == result
    assert persisted["candidate_seen"] == candidate
    metadata = json.loads((scope / "finalization_metadata.json").read_text())
    assert metadata["candidate_character_length"] == len(candidate)
    assert metadata["candidate_utf8_byte_length"] == len(candidate.encode("utf-8"))
    assert metadata["candidate_sha256"] == hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    assert metadata["finish_reason"] == "stop"
    assert metadata["provider_transport_streaming"] is True
    assert metadata["required_finalization_active"] is True
    assert all((path.stat().st_mode & 0o077) == 0 for path in scope.iterdir())
    clear_skill_finalizers("enabled/task")


def test_invalid_violation_diagnostic_does_not_leak_candidate(tmp_path, monkeypatch):
    output = tmp_path / "diagnostics"
    monkeypatch.setenv("HERMES_INTERNAL_FINALIZATION_DIAGNOSTICS_DIR", str(output))
    refusal = "I can’t safely finalize this response."
    _register_validator(tmp_path, "invalid", {
        "status": "invalid",
        "violations": [{"code": "manifest_missing_or_invalid", "reason": "JSONDecodeError: line 1"}],
        "content": refusal,
        "metadata": {"parser": "json"},
    })
    result = finalize_skill_response("invalid", "SECRET RAW CANDIDATE", [])
    assert result["content"] == refusal
    assert "SECRET" not in result["content"]
    persisted = json.loads((_artifact_scope(output) / "finalization_result.json").read_text())
    assert persisted["violations"][0]["reason"] == "JSONDecodeError: line 1"
    assert persisted["metadata"] == {"parser": "json"}
    clear_skill_finalizers("invalid")


def test_diagnostic_task_scopes_do_not_collide(tmp_path, monkeypatch):
    output = tmp_path / "diagnostics"
    monkeypatch.setenv("HERMES_INTERNAL_FINALIZATION_DIAGNOSTICS_DIR", str(output))
    for task, candidate in (("same/path", "one"), ("same_path", "two")):
        _register_validator(tmp_path, task, {
            "status": "valid", "violations": [], "content": "safe",
        })
        finalize_skill_response(task, candidate, [])
        clear_skill_finalizers(task)
    scopes = sorted(path for path in output.iterdir() if path.is_dir())
    assert len(scopes) == 2
    assert {path.joinpath("pre_finalization_candidate.txt").read_text() for path in scopes} == {"one", "two"}
