import json
from pathlib import Path

from agent.skill_lifecycle import clear_prepared_synthesis, get_prepared_synthesis
from agent.skill_routing import (
    activate_routed_skill,
    install_routed_context,
    route_request,
)
from tools import skills_tool


def _owner(name, groups, priority=0):
    return {
        "skill": name,
        "priority": priority,
        "groups": [
            {"concept": concept, "phrases": [tuple(phrase.split()) for phrase in phrases]}
            for concept, phrases in groups
        ],
    }


OWNING = _owner("trip-planner", [
    ("destination", ["travel to", "city visit"]),
    ("lodging", ["recommend lodging", "choose hotel"]),
    ("schedule", ["daily itinerary", "day plan"]),
], priority=50)


def test_strong_unique_owner_routes_and_weak_or_ordinary_requests_do_not():
    selected = route_request(
        "Please recommend lodging and a daily itinerary for travel to Oslo.",
        [OWNING],
    )
    assert selected == {
        "route_status": "selected",
        "route_reason": "unique_strong_owner",
        "route_skill": "trip-planner",
        "route_priority": 50,
        "route_score": 6,
        "matched_groups": ["destination", "lodging", "schedule"],
    }
    assert route_request("Can you explain recursion?", [OWNING])["route_status"] == "no_match"
    assert route_request("I need a hotel.", [OWNING])["route_status"] == "no_match"


def test_tied_owners_fall_back_and_priority_resolves_deterministically():
    peer = {**OWNING, "skill": "alternate-planner"}
    request = "Recommend lodging and a daily itinerary for travel to Oslo."
    tied = route_request(request, [OWNING, peer])
    assert tied["route_status"] == "ambiguous"
    assert tied["route_candidates"] == ["alternate-planner", "trip-planner"]

    preferred = {**peer, "priority": 51}
    assert route_request(request, [OWNING, preferred])["route_skill"] == "alternate-planner"


def test_non_owner_dependency_cannot_steal_owner_request():
    dependency = {**OWNING, "skill": "hotel-database", "role": "data"}
    # Routing indexes contain owners only; a data declaration is deliberately absent.
    result = route_request(
        "Recommend lodging and a daily itinerary for travel to Oslo.",
        [OWNING] if dependency["role"] != "owner" else [dependency, OWNING],
    )
    assert result["route_skill"] == "trip-planner"


def test_activation_uses_skill_view_without_classifier_and_reports_status(monkeypatch):
    calls = []

    def fake_view(args, **kwargs):
        calls.append((args, kwargs))
        return json.dumps({
            "success": True,
            "name": "trip-planner",
            "preparation": {"status": "LIMITED"},
        })

    monkeypatch.setattr("agent.skill_routing.route_request", lambda request: {
        "route_status": "selected", "route_skill": "trip-planner",
    })
    monkeypatch.setattr(skills_tool, "_skill_view_with_bump", fake_view)
    route, result = activate_routed_skill("request one", "task-one")
    assert route["preparation_status"] == "LIMITED"
    assert result["preparation"]["status"] == "LIMITED"
    assert calls == [(
        {"name": "trip-planner"},
        {"task_id": "task-one", "user_task": "request one"},
    )]


def test_limited_and_failed_do_not_arm_constrained_synthesis(monkeypatch):
    clear_prepared_synthesis()
    monkeypatch.setattr("agent.skill_routing.route_request", lambda request: {
        "route_status": "selected", "route_skill": "trip-planner",
    })
    for status in ("LIMITED", "FAILED"):
        monkeypatch.setattr(skills_tool, "_skill_view_with_bump", lambda *args, _status=status, **kwargs: json.dumps({
            "success": True, "name": "trip-planner", "preparation": {"status": _status},
        }))
        route, _ = activate_routed_skill(f"request-{status}", f"task-{status}")
        assert route["preparation_status"] == status
        assert get_prepared_synthesis(f"task-{status}") is None


def test_request_copy_injection_preserves_system_and_provider_roles():
    messages = [
        {"role": "system", "content": "stable prefix"},
        {"role": "user", "content": "plan this"},
    ]
    assert install_routed_context(messages, '{"status":"READY"}')
    assert messages[0] == {"role": "system", "content": "stable prefix"}
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[1]["content"].endswith('{"status":"READY"}')


def test_different_requests_are_recomputed_without_route_state(monkeypatch):
    seen = []
    monkeypatch.setattr("agent.skill_routing.build_routing_index", lambda: [OWNING])
    first = route_request("Recommend lodging and a daily itinerary for travel to Oslo.")
    second = route_request("Explain recursion.")
    seen.extend((first["route_status"], second["route_status"]))
    assert seen == ["selected", "no_match"]


def test_core_router_contains_no_fixture_domain_literals():
    source = (Path(__file__).resolve().parents[2] / "agent" / "skill_routing.py").read_text(
        encoding="utf-8"
    ).casefold()
    for forbidden in ("boardwalk", "kegel", "bowling", "oil_pattern", "bowling_ball"):
        assert forbidden not in source
