"""Focused tests for deterministic, request-scoped skill dependencies."""

import json

import pytest

from tools.skills_tool import _skill_view_with_bump, reset_skill_view_dedup


@pytest.fixture
def capability_skills(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    principles = skills / "principles"
    references = principles / "references"
    references.mkdir(parents=True)
    (principles / "SKILL.md").write_text(
        "---\nname: principles\ndescription: Principles.\n---\n# Principles\n",
        encoding="utf-8",
    )
    (references / "pattern.md").write_text(
        "# Pattern\n\n"
        "## P-001 — Length interpretation\n\n"
        "- **Claim:** Interpret verified length using this framework.\n"
        "- **Scope:** Verified pattern lengths.\n"
        "- **Limitations:** This does not establish a complete strategy.\n\n"
        "## P-999 — Unrelated principle\n\nDo not inject this section.\n",
        encoding="utf-8",
    )
    consumer = skills / "pattern-play"
    consumer.mkdir(parents=True)
    (consumer / "SKILL.md").write_text(
        "---\n"
        "name: pattern-play\n"
        "description: Pattern play.\n"
        "metadata:\n"
        "  hermes:\n"
        "    capability_dependencies:\n"
        "      - capability: pattern_interpretation\n"
        "        when:\n"
        "          any_terms: [pattern, length]\n"
        "        requires:\n"
        "          skill: principles\n"
        "          reference: references/pattern.md\n"
        "          sections: [P-001]\n"
        "---\n"
        "# Pattern Play\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    reset_skill_view_dedup()
    return home


def _view(user_task, task_id):
    return json.loads(
        _skill_view_with_bump(
            {"name": "pattern-play"},
            task_id=task_id,
            user_task=user_task,
        )
    )


def test_matching_request_loads_only_declared_principle_section(capability_skills):
    result = _view("Interpret this verified pattern length.", "cap-match")

    assert result["pre_synthesis_gate"]["status"] == "ready"
    dependency = result["required_capability_dependencies"][0]
    assert dependency["status"] == "available"
    assert dependency["owning_skill"] == "principles"
    assert dependency["reference"] == "references/pattern.md"
    assert dependency["principles"][0]["identity"] == "P-001"
    assert "Scope" in dependency["principles"][0]["content"]
    assert "Limitations" in dependency["principles"][0]["content"]
    assert "P-999" not in json.dumps(dependency)
    assert dependency["reuse"] == {
        "status": "already_loaded",
        "authoritative_for_capability": True,
        "duplicate_read_required": False,
        "targeted_expansion_allowed": True,
    }
    assert "without manually loading" in result["pre_synthesis_gate"]["instruction"]


def test_undeclared_principle_remains_available_for_targeted_expansion(capability_skills):
    result = json.loads(
        _skill_view_with_bump(
            {"name": "principles", "file_path": "references/pattern.md"},
            task_id="cap-targeted-expansion",
            user_task="Need an undeclared principle.",
        )
    )
    assert result["success"] is True
    assert "P-999" in result["content"]


def test_unrelated_request_does_not_load_dependency_material(capability_skills):
    result = _view("List my active equipment.", "cap-unrelated")

    assert "required_capability_dependencies" not in result
    assert "pre_synthesis_gate" not in result


def test_changed_request_recomputes_dependencies_in_same_task(capability_skills):
    first = _view("List my active equipment.", "cap-changing-request")
    second = _view("Interpret this pattern length.", "cap-changing-request")

    assert "required_capability_dependencies" not in first
    assert second["pre_synthesis_gate"]["status"] == "ready"


def test_dependency_failure_is_explicitly_unavailable(capability_skills):
    (capability_skills / "skills" / "principles" / "references" / "pattern.md").unlink()
    result = _view("Interpret this pattern.", "cap-failure")

    assert result["pre_synthesis_gate"]["status"] == "limited"
    dependency = result["required_capability_dependencies"][0]
    assert dependency["status"] == "unavailable"
    assert "Do not substitute model memory" in dependency["synthesis_requirement"]
