"""Patch 27: deterministic ball-reaction dependencies for pattern play."""

import json
from pathlib import Path

import pytest

from tools.skills_tool import (
    _matching_capability_dependencies, _parse_frontmatter,
    _skill_view_with_bump, reset_skill_view_dedup,
)


HERMES_HOME = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def real_bowling_skills(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(HERMES_HOME))
    reset_skill_view_dedup()


def _prepared_dependencies(request: str) -> list[dict]:
    result = json.loads(
        _skill_view_with_bump(
            {"name": "bowling-pattern-play"},
            task_id=f"patch-27-{request}",
            user_task=request,
        )
    )
    return result["preparation"]["context"]["required_capability_dependencies"]


def _by_capability(request: str) -> dict[str, dict]:
    return {item["capability"]: item for item in _prepared_dependencies(request)}


def _ids(dependency: dict) -> list[str]:
    return [principle["identity"] for principle in dependency["principles"]]


def _matched_capabilities(request: str) -> list[str]:
    content = (HERMES_HOME / "skills/bowling-pattern-play/SKILL.md").read_text(encoding="utf-8")
    frontmatter, _ = _parse_frontmatter(content)
    return [item["capability"] for item in _matching_capability_dependencies(frontmatter, request)]


def test_starting_ball_loads_equipment_selection_reaction_principles():
    dependencies = _by_capability("Which ball from my arsenal should I start with on this pattern?")

    assert _ids(dependencies["equipment_selection"]) == [
        "BR-003", "BR-004", "BR-005", "BR-006",
        "BRM-001", "BRM-002", "BRM-004", "BRM-005",
    ]
    assert _ids(dependencies["pattern_equipment_context"]) == ["BR-007"]


def test_surface_context_adds_only_surface_comparison_principles():
    dependency = _by_capability(
        "Compare equipment for lane play using each ball's current prepared surface."
    )["equipment_selection"]

    assert _ids(dependency) == [
        "BR-001", "BR-002", "BR-003", "BR-004", "BR-005", "BR-006",
        "BRM-001", "BRM-002", "BRM-004", "BRM-005",
    ]


def test_observed_reaction_alternate_loads_transition_reaction_principles():
    dependencies = _by_capability(
        "Recommend an alternate ball after the observed ball reaction reads too early."
    )

    assert _ids(dependencies["reaction_transition_interpretation"]) == [
        "BR-001", "BR-002", "BR-006", "BR-008",
        "BRM-001", "BRM-002", "BRM-004", "BRM-005",
    ]


def test_pattern_only_does_not_load_equipment_reaction_dependencies():
    dependencies = _by_capability("Interpret this pattern length classification.")

    assert set(dependencies) == {"pattern_length_interpretation"}
    assert _ids(dependencies["pattern_length_interpretation"]) == [
        "PF-001", "PF-002", "PF-003", "PF-004", "PF-005", "PF-006",
    ]


def test_targeting_dependency_remains_unchanged():
    dependencies = _by_capability("What breakpoint target should I use?")

    assert _ids(dependencies["targeting_breakpoint_interpretation"]) == [
        "TB-001", "TB-002", "TB-003", "TB-004", "TB-005",
    ]


def test_ball_reaction_provenance_survives_into_prepared_evidence():
    request = "Recommend an alternate ball after the ball reaction reads too late."
    result = json.loads(
        _skill_view_with_bump(
            {"name": "bowling-pattern-play"},
            task_id="patch-27-prepared-provenance",
            user_task=request,
        )
    )
    context = result["preparation"]["context"]
    record = next(
        item for item in context["evidence_catalog"]["records"]
        if item["evidence_type"] == "capability_dependency"
        and item["payload"]["capability"] == "reaction_transition_interpretation"
    )
    dependency = record["payload"]

    assert record["evidence_ref"].startswith("dep_")
    assert [principle["identity"] for principle in dependency["principles"]] == [
        "BR-001", "BR-002", "BR-006", "BR-008",
        "BRM-001", "BRM-002", "BRM-004", "BRM-005",
    ]
    assert all(principle["content"] for principle in dependency["principles"])


@pytest.mark.parametrize("user_text", [
    "What ball should I switch to if this one reads too early?",
    "Give me reaction-based transition moves.",
    "When should I change balls?",
    "What should I do if the ball skids too far?",
    "Give me a starting ball and tell me when to go to the alternate.",
    "What ball change should I make when the reaction changes?",
    "Plan my ball transitions based on what I see.",
    "If I lose continuation, what should I switch to?",
    "Starting ball, alternate, and reaction triggers.",
    "deliverables: [starting_ball, alternate_ball, reaction_triggers]",
    "Recommend a starting ball, reaction signs to watch, and how to transition through my arsenal as lanes change.",
])
def test_reaction_transition_concept_matrix_positive(user_text):
    capabilities = _matched_capabilities(user_text)
    assert capabilities.count("reaction_transition_interpretation") == 1


@pytest.mark.parametrize("user_text", [
    "What is axis rotation?",
    "Tell me about this bowling ball.",
    "What is RG?",
    "What does reads early mean?",
    "List my arsenal.",
    "What ball should I start with?",
    "What surface is recorded for this ball?",
])
def test_reaction_transition_concept_matrix_negative(user_text):
    capabilities = _matched_capabilities(user_text)
    assert "reaction_transition_interpretation" not in capabilities


def test_reaction_transition_dependency_loads_br008_exactly_once():
    dependency = _by_capability(
        "Recommend a starting ball, alternate, and ball-reaction triggers as the lanes change."
    )["reaction_transition_interpretation"]
    assert _ids(dependency) == [
        "BR-001", "BR-002", "BR-006", "BR-008",
        "BRM-001", "BRM-002", "BRM-004", "BRM-005",
    ]
    assert _ids(dependency).count("BR-008") == 1
