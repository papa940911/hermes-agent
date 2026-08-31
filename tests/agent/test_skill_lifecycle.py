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
    provider_schema_from_contract,
    structured_output_extra_body,
)
from agent.skill_routing import activate_routed_skill, route_request
from tools import skills_tool


def _skill(tmp_path: Path, *, status="READY", handoff=True, prepared=None, provider=None,
           contract=False, structured=False):
    root = tmp_path / f"skill-{status}-{handoff}"
    root.mkdir()
    validator = root / "validate.py"
    validator.write_text(
        "import json,sys; p=json.load(sys.stdin); c=json.loads(p['candidate']); "
        "json.dump({'status':'valid','content':json.dumps({'manifest':c," 
        "'evidence':p['prepared_context']},sort_keys=True),'violations':[]},sys.stdout)",
        encoding="utf-8",
    )
    finalization = {"required": True, "script": "validate.py"}
    if contract:
        (root / "contract.json").write_text(json.dumps({
            "schema": "fixture_v1",
            "required_top_level": {"schema": "string"},
            "items": {},
        }), encoding="utf-8")
        finalization["synthesis_contract"] = "contract.json"
    if structured:
        finalization["structured_output"] = {"required": True, "mode": "json_schema"}
    return {
        "name": "synthetic",
        "_source_path": str(root / "SKILL.md"),
        "metadata": {"hermes": {
            "preparation": {
                "required": True,
                **({"handoff": "constrained_synthesis"} if handoff else {}),
            },
            "finalization": finalization,
        }},
        "preparation": {
            "status": status, "context": prepared or {"fact": 7},
            **({"provider_context": provider} if provider is not None else {}),
        },
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
    assert handoff["provider_context"] == {"fact": 7}
    assert "read_file" not in api_copy[-1]["content"]
    assert transcript[-1]["content"].startswith("<persisted-output>")
    assert api_copy[0] == transcript[0]
    assert [message["role"] for message in api_copy] == ["system", "user", "assistant", "tool"]


def test_declared_contract_is_model_visible_only_on_request_copy(tmp_path):
    skill = _skill(tmp_path, contract=True)
    register_prepared_synthesis("task", skill, "request")
    state = get_prepared_synthesis("task", "request")
    transcript = [
        {"role": "system", "content": "stable-system-prefix"},
        {"role": "user", "content": "request"},
        {"role": "assistant", "tool_calls": [{"id": "call"}]},
        {"role": "tool", "name": "skill_view", "content": '{"name":"synthetic"}'},
    ]
    persisted_snapshot = json.loads(json.dumps(transcript))
    api_copy = json.loads(json.dumps(transcript))
    assert install_synthesis_handoff(api_copy, state)
    visible = json.loads(api_copy[-1]["content"])
    assert visible["synthesis_contract"]["schema"] == "fixture_v1"
    assert list(visible).index("synthesis_instruction") + 1 == list(visible).index("synthesis_contract")
    assert "immediately adjacent synthesis_contract" in visible["synthesis_instruction"]
    assert transcript == persisted_snapshot
    assert api_copy[0] == transcript[0]
    assert get_prepared_synthesis("ordinary", "request") is None


def test_required_structured_output_is_derived_from_contract_and_request_local(tmp_path):
    skill = _skill(tmp_path, contract=True, structured=True)
    register_prepared_synthesis("structured", skill, "request")
    state = get_prepared_synthesis("structured", "request")
    addition = structured_output_extra_body(state)
    response_format = addition["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["name"] == "fixture_v1"
    schema = response_format["json_schema"]["schema"]
    assert schema == provider_schema_from_contract(state["synthesis_contract"])
    assert schema["properties"]["schema"]["enum"] == ["fixture_v1"]
    assert '"pattern"' not in json.dumps(schema)
    assert '"format"' not in json.dumps(schema)
    assert "structured_output" not in json.loads(synthesis_handoff_text(state))
    assert structured_output_extra_body(None) is None


def test_required_structured_output_with_unsuitable_contract_is_not_silently_unconstrained(tmp_path):
    skill = _skill(tmp_path, structured=True)
    register_prepared_synthesis("unavailable", skill, "request")
    state = get_prepared_synthesis("unavailable", "request")
    assert state["structured_output"]["required"] is True
    assert state["structured_output"]["unavailable"] is True
    assert structured_output_extra_body(state) is None


def test_missing_contract_preserves_existing_safe_constrained_lifecycle(tmp_path):
    register_prepared_synthesis("task", _skill(tmp_path), "request")
    handoff = json.loads(synthesis_handoff_text(get_prepared_synthesis("task", "request")))
    assert handoff["synthesis_contract"] is None
    assert "synthesize from only" in handoff["synthesis_instruction"].casefold()
    assert tools_for_synthesis(get_prepared_synthesis("task"), [{"name": "tool"}]) == []


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


def test_provider_projection_is_visible_but_authoritative_context_stays_private(tmp_path):
    skill = _skill(
        tmp_path, prepared={"private_catalog": {"secret": 7}},
        provider={"evidence_index": [{"evidence_ref": "ref_1", "display": "safe"}]},
    )
    register_skill_finalizer("task", skill)
    register_prepared_synthesis("task", skill, "request")
    state = get_prepared_synthesis("task", "request")
    visible = json.loads(synthesis_handoff_text(state))
    assert visible["provider_context"] == skill["preparation"]["provider_context"]
    assert "private_catalog" not in visible
    result = finalize_skill_response("task", '{"schema":"fixture_v1"}', [])
    assert json.loads(result["content"])["evidence"] == {"private_catalog": {"secret": 7}}


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
    route = route_request(request)
    assert route["route_skill"] == "bowling-pattern-play"
    route, result = activate_routed_skill(request, task)
    assert route["route_status"] == "selected"
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
    assert handoff["provider_context"]["schema"] == "bowling_pattern_play_context_v1"
    assert "evidence_index" in handoff["provider_context"]
    assert "evidence_catalog" not in handoff["provider_context"]
    assert "arsenal" not in handoff["provider_context"]
    assert handoff["synthesis_contract"]["schema"] == "bowling_pattern_play_recommendations_v1"
    schema = state["structured_output"]["schema"]
    transitions = schema["properties"]["transitions"]["items"]
    assert set(transitions["properties"]) == {"observation", "action"}
    assert "choice_ref" not in json.dumps(transitions)
    assert schema["properties"]["alternate_ball_choice_refs"]["maxItems"] == 1
    assert schema["properties"]["starting_ball_reaction_evidence_refs"]["type"] == "array"
    assert schema["properties"]["alternate_ball_reaction_evidence_refs"]["type"] == "array"
    start_item = schema["properties"]["starting_ball_reaction_evidence_refs"]["items"]
    alternate_item = schema["properties"]["alternate_ball_reaction_evidence_refs"]["items"]
    assert "owned only by starting_ball_choice_ref" in start_item["description"]
    assert "owned only by the sole alternate_ball_choice_refs choice" in alternate_item["description"]
    assert "starting_ball_support_refs" not in schema["properties"]
    assert all(key not in json.dumps(schema) for key in ("oneOf", "anyOf"))
    output = handoff["synthesis_contract"]["output"]
    assert "Return exactly one JSON object" in output
    assert "No Markdown. No prose outside JSON." in output
    assert "Use only supplied num_choice_, eq_choice_, and reaction_fact_ refs" in output
    assert "never author reaction_cmp_ refs" in output.casefold()
    assert "Do not return ball_fact_choice_ refs" in output
    assert "never mix start/alternate ownership" in output.casefold()
    # Routing and preparation are entirely local. Before synthesis there is no
    # provider-selected terminal/read/search/manual resolver/arsenal call.
    assert route["preparation_status"] == "READY"
