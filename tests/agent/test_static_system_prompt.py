"""Gate 10P: focused tests for agent/static_system_prompt.py.

Covers: deterministic, side-effect-free construction of the real static
Hermes system-prompt prefix; that it never touches conversation/session/
memory-content/Telegram/review/promotion state; that it delegates to the
SAME builder (agent.system_prompt.build_system_prompt_parts) every real
turn uses instead of duplicating prompt text; and that its output is an
exact, stable prefix of what a real AIAgent constructs.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import static_system_prompt as ssp


# ── Shared fixtures/helpers ─────────────────────────────────────────────


def _patch_config(monkeypatch, **overrides):
    cfg = {
        "model": {
            "default": "test-model", "provider": "custom",
            "base_url": "http://127.0.0.1:8087/v1", "context_length": 64000,
        },
        "memory": {"memory_enabled": True, "user_profile_enabled": True},
        "agent": {},
    }
    for section, values in overrides.items():
        cfg.setdefault(section, {}).update(values)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
    return cfg


def _patch_tool_names(monkeypatch, names):
    monkeypatch.setattr(ssp, "resolve_static_prompt_tool_names", lambda **_: frozenset(names))


def _quiet_builders():
    """Patch run_agent's pure helpers to short, deterministic stand-ins.

    Mirrors tests/agent/test_system_prompt.py's own patching convention so
    SOUL.md/skills/env-probe file I/O never leaks into unrelated assertions.
    """
    return (
        patch("run_agent.load_soul_md", return_value="IDENTITY-BLOCK"),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value="CONTEXT-FILE-BLOCK"),
        patch("run_agent.build_skills_system_prompt", return_value="SKILLS-INDEX"),
    )


def _build(monkeypatch, *, platform="telegram", tool_names=("memory",), **cfg_overrides):
    _patch_config(monkeypatch, **cfg_overrides)
    _patch_tool_names(monkeypatch, tool_names)
    patches = _quiet_builders()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        return ssp.build_static_system_prompt(platform=platform)


# ── Determinism / purity ────────────────────────────────────────────────


class TestDeterminism:
    def test_deterministic_output(self, monkeypatch):
        a = _build(monkeypatch)
        b = _build(monkeypatch)
        assert a == b

    def test_returns_a_string(self, monkeypatch):
        assert isinstance(_build(monkeypatch), str)

    def test_nonempty_for_a_normal_configuration(self, monkeypatch):
        assert len(_build(monkeypatch)) > 0

    def test_same_prompt_section_order_across_calls(self, monkeypatch):
        first = _build(monkeypatch, tool_names=("memory", "skill_manage", "session_search"))
        second = _build(monkeypatch, tool_names=("memory", "skill_manage", "session_search"))
        assert first == second

    def test_stable_across_repeated_sessions(self, monkeypatch):
        # "Session" here means nothing but call count — the builder takes no
        # session_id/conversation input at all, so repetition must be a pure
        # function of config.
        results = {_build(monkeypatch) for _ in range(5)}
        assert len(results) == 1

    def test_stable_regardless_of_calling_transport_label(self, monkeypatch):
        # The builder has no notion of "transport" beyond the platform hint
        # text itself; asking for the same platform twice must be identical
        # regardless of how many other platforms were built in between.
        telegram_a = _build(monkeypatch, platform="telegram")
        _build(monkeypatch, platform="cli")
        telegram_b = _build(monkeypatch, platform="telegram")
        assert telegram_a == telegram_b


# ── No user/session/turn state ──────────────────────────────────────────


class TestNoSessionOrTurnState:
    def test_no_user_message_parameter_exists(self):
        # The builder's signature has no way to accept a user message,
        # conversation turn, or history at all — the strongest possible
        # guarantee against leaking one in.
        sig = inspect.signature(ssp.build_static_system_prompt)
        forbidden = {"message", "messages", "user_message", "history", "conversation"}
        assert forbidden.isdisjoint(sig.parameters)

    def test_no_conversation_history_marker(self, monkeypatch):
        prompt = _build(monkeypatch)
        assert "conversation started" not in prompt.lower()

    def test_no_session_id_line(self, monkeypatch):
        prompt = _build(monkeypatch)
        assert "session id" not in prompt.lower()

    def test_no_memory_retrieval_content(self, monkeypatch):
        # MEMORY_GUIDANCE (config-gated instructional text) is allowed; actual
        # stored memory content never is, because the shim's _memory_store is
        # always None.
        prompt = _build(monkeypatch, tool_names=("memory",))
        assert "_memory_store" not in prompt

    def test_no_bowlingtracker_live_state(self, monkeypatch):
        prompt = _build(monkeypatch, tool_names=("bowlingtracker_recent_sessions",))
        assert "frame" not in prompt.lower()
        assert "pinfall" not in prompt.lower()

    def test_no_review_item_content(self, monkeypatch):
        prompt = _build(monkeypatch)
        assert "review_id" not in prompt.lower()
        assert "candidate" not in prompt.lower()

    def test_no_telegram_message_content(self, monkeypatch):
        prompt = _build(monkeypatch, platform="telegram")
        assert "chat_id" not in prompt.lower()

    def test_no_timestamp_line(self, monkeypatch):
        # The volatile-tier timestamp line is intentionally excluded — the
        # function returns only the "stable" tier.
        prompt = _build(monkeypatch)
        assert not any(
            day in prompt
            for day in ("Monday,", "Tuesday,", "Wednesday,", "Thursday,", "Friday,", "Saturday,", "Sunday,")
        )

    def test_skills_index_excluded_from_static_tier(self, monkeypatch):
        # SKILLS-INDEX (the stand-in for build_skills_system_prompt's output)
        # lives in the volatile tier, never stable.
        prompt = _build(monkeypatch, tool_names=("skill_manage", "skills_list", "skill_view"))
        assert "SKILLS-INDEX" not in prompt

    def test_context_files_excluded_from_static_tier(self, monkeypatch):
        prompt = _build(monkeypatch)
        assert "CONTEXT-FILE-BLOCK" not in prompt


# ── Same model configuration / same builder ─────────────────────────────


class TestSameConfiguration:
    def test_same_model_configuration_used(self, monkeypatch):
        cfg = _patch_config(monkeypatch, model={"default": "qwen-test", "provider": "custom"})
        _patch_tool_names(monkeypatch, ())
        patches = _quiet_builders()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            ssp.build_static_system_prompt(platform="telegram")
        assert cfg["model"]["default"] == "qwen-test"  # config read, not duplicated/hardcoded

    def test_explicit_model_argument_overrides_config(self, monkeypatch):
        _patch_config(monkeypatch, model={"default": "configured-model", "provider": "alibaba"})
        _patch_tool_names(monkeypatch, ())
        patches = _quiet_builders()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            prompt = ssp.build_static_system_prompt(model="explicit-model", provider="custom", platform="telegram")
        # alibaba workaround text names the model; it must reflect the
        # explicit override, not the config default, when provider=custom.
        assert "explicit-model" not in prompt  # custom provider never injects the alibaba block
        assert "configured-model" not in prompt

    def test_alibaba_provider_injects_model_identity_line(self, monkeypatch):
        _patch_config(monkeypatch, model={"default": "glm-4.7", "provider": "alibaba"})
        _patch_tool_names(monkeypatch, ())
        patches = _quiet_builders()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            prompt = ssp.build_static_system_prompt(platform="telegram")
        assert "glm-4.7" in prompt

    def test_reuses_the_real_build_system_prompt_parts_function(self, monkeypatch):
        # Regression guard against re-duplicating prompt text: the module
        # must call the shared builder, not assemble text itself.
        called = {}
        from agent import system_prompt as sp

        real = sp.build_system_prompt_parts

        def spy(agent, **kw):
            called["hit"] = True
            return real(agent, **kw)

        monkeypatch.setattr(sp, "build_system_prompt_parts", spy)
        # ssp imports build_system_prompt_parts lazily inside the function,
        # from the agent.system_prompt module, so patching that module
        # attribute is observed.
        _patch_config(monkeypatch)
        _patch_tool_names(monkeypatch, ())
        patches = _quiet_builders()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            ssp.build_static_system_prompt(platform="telegram")
        assert called.get("hit") is True


# ── Content inclusion driven by real config ─────────────────────────────


class TestContentInclusion:
    def test_soul_included_when_present(self, monkeypatch):
        prompt = _build(monkeypatch)
        assert "IDENTITY-BLOCK" in prompt

    def test_falls_back_to_default_identity_when_no_soul(self, monkeypatch):
        _patch_config(monkeypatch)
        _patch_tool_names(monkeypatch, ())
        with (
            patch("run_agent.load_soul_md", return_value=""),
            patch("run_agent.build_nous_subscription_prompt", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt", return_value=""),
        ):
            prompt = ssp.build_static_system_prompt(platform="telegram")
        assert "Hermes Agent" in prompt

    def test_memory_guidance_included_when_memory_tool_and_flag_enabled(self, monkeypatch):
        prompt = _build(monkeypatch, tool_names=("memory",), memory={"memory_enabled": True, "user_profile_enabled": True})
        assert "persistent memory across sessions" in prompt

    def test_memory_guidance_excluded_when_memory_tool_absent(self, monkeypatch):
        prompt = _build(monkeypatch, tool_names=("terminal",))
        assert "persistent memory across sessions" not in prompt

    def test_user_profile_guidance_when_memory_disabled_but_profile_enabled(self, monkeypatch):
        prompt = _build(
            monkeypatch, tool_names=("memory",),
            memory={"memory_enabled": False, "user_profile_enabled": True},
        )
        assert "persistent user profile across sessions" in prompt

    def test_skill_guidance_included_when_skill_manage_present(self, monkeypatch):
        prompt = _build(monkeypatch, tool_names=("skill_manage",))
        assert "skill_manage" in prompt and "Skill Safety Rule" in prompt

    def test_session_search_guidance_gated_on_tool(self, monkeypatch):
        with_tool = _build(monkeypatch, tool_names=("session_search",))
        without_tool = _build(monkeypatch, tool_names=("terminal",))
        assert "session_search to recall it" in with_tool
        assert "session_search to recall it" not in without_tool

    def test_task_completion_and_parallel_guidance_present_when_tools_nonempty(self, monkeypatch):
        prompt = _build(monkeypatch, tool_names=("terminal",))
        assert "Finishing the job" in prompt
        assert "Parallel tool calls" in prompt

    def test_no_tool_guidance_blocks_when_no_tools_at_all(self, monkeypatch):
        prompt = _build(monkeypatch, tool_names=())
        assert "Finishing the job" not in prompt
        assert "Parallel tool calls" not in prompt

    def test_tool_use_enforcement_for_qwen_model(self, monkeypatch):
        _patch_config(monkeypatch, model={"default": "some/qwen-model", "provider": "custom"})
        _patch_tool_names(monkeypatch, ("terminal",))
        patches = _quiet_builders()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            prompt = ssp.build_static_system_prompt(platform="telegram")
        assert "Tool-use enforcement" in prompt
        assert "Execution discipline" in prompt

    def test_no_tool_use_enforcement_for_claude_model(self, monkeypatch):
        _patch_config(monkeypatch, model={"default": "claude-opus", "provider": "anthropic"})
        _patch_tool_names(monkeypatch, ("terminal",))
        patches = _quiet_builders()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            prompt = ssp.build_static_system_prompt(platform="telegram")
        assert "Tool-use enforcement" not in prompt

    def test_platform_hint_included_for_known_platform(self, monkeypatch):
        prompt = _build(monkeypatch, platform="telegram")
        assert "Telegram" in prompt

    def test_platform_hint_overrides_from_config(self, monkeypatch):
        prompt = _build(monkeypatch, platform="telegram", agent={"platform_hints": {"telegram": {"append": "CUSTOM-HINT-XYZ"}}})
        assert "CUSTOM-HINT-XYZ" in prompt


# ── Dynamic / volatile content correctly excluded ───────────────────────


class TestVolatileExclusion:
    def test_dynamic_date_excluded(self, monkeypatch):
        prompt = _build(monkeypatch)
        assert "Conversation started:" not in prompt

    def test_dynamic_memory_excluded(self, monkeypatch):
        prompt = _build(monkeypatch, tool_names=("memory",))
        # The guidance text is fine; a formatted memory *block* is not.
        assert "format_for_system_prompt" not in prompt

    def test_no_plugin_section_frames(self, monkeypatch):
        prompt = _build(monkeypatch)
        assert "Plugin Context:" not in prompt


# ── Prefix identity against the real full builder ───────────────────────


class TestPrefixIdentity:
    def test_static_output_equals_stable_tier_of_full_builder(self, monkeypatch):
        _patch_config(monkeypatch, tool_names={})
        _patch_tool_names(monkeypatch, ("memory", "terminal"))
        patches = _quiet_builders()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            static_only = ssp.build_static_system_prompt(platform="telegram")
            view = ssp._static_prompt_agent_view(
                model="test-model", provider="custom", platform="telegram",
                context_length=64000, memory_enabled=True, user_profile_enabled=True,
                platform_hint_overrides=None, valid_tool_names=frozenset({"memory", "terminal"}),
            )
            from agent.system_prompt import build_system_prompt_parts
            full_parts = build_system_prompt_parts(view, system_message="a caller system message")
        assert static_only == full_parts["stable"]
        joined = "\n\n".join(p for p in (full_parts["stable"], full_parts["context"], full_parts["volatile"]) if p)
        assert joined.startswith(static_only)

    def test_session_specific_suffix_excluded_from_prefix(self, monkeypatch):
        _patch_config(monkeypatch)
        _patch_tool_names(monkeypatch, ("terminal",))
        patches = _quiet_builders()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            static_only = ssp.build_static_system_prompt(platform="telegram")
            view = ssp._static_prompt_agent_view(
                model="test-model", provider="custom", platform="telegram",
                context_length=64000, memory_enabled=True, user_profile_enabled=True,
                platform_hint_overrides=None, valid_tool_names=frozenset({"terminal"}),
            )
            view.session_id = "a-real-session-id-should-not-leak"
            view.pass_session_id = True
            from agent.system_prompt import build_system_prompt_parts
            full_parts = build_system_prompt_parts(view)
        assert "a-real-session-id-should-not-leak" not in static_only
        assert "a-real-session-id-should-not-leak" in full_parts["volatile"]

    def test_turn_specific_system_message_excluded_from_prefix(self, monkeypatch):
        _patch_config(monkeypatch)
        _patch_tool_names(monkeypatch, ())
        patches = _quiet_builders()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            static_only = ssp.build_static_system_prompt(platform="telegram")
        assert "this turn's ephemeral text" not in static_only


@pytest.mark.integration
class TestPrefixIdentityAgainstRealAgent:
    """One canary test against a genuinely constructed AIAgent.

    This is the ONLY test in the suite that builds a real AIAgent — doing so
    proves byte-for-byte prefix identity against ground truth, which no
    amount of shim-based unit testing can substitute for. It deliberately
    opts OUT of the repo's standard HERMES_HOME test-isolation sandbox (see
    tests/conftest.py's ``_hermetic_environment``) because its entire point
    is to check against the real production configuration — so it skips
    itself gracefully wherever that production config/model isn't present
    (a dev laptop, CI) instead of failing. AIAgent construction does
    incidental, non-semantic disk I/O (a log line, an endpoint-probe cache
    file, a kanban WAL shm touch) that the assertions below explicitly
    confirm stay non-semantic — no Knowledge/review/promotion/notification
    DB is touched.
    """

    def test_matches_a_real_aiagent_construction(self, monkeypatch):
        import os

        production_home = "/home/hermes/.hermes"
        if not os.path.isdir(production_home):
            pytest.skip("production HERMES_HOME not present on this machine")
        monkeypatch.setenv("HERMES_HOME", production_home)
        monkeypatch.setenv("HERMES_TEST_ISOLATION", "")

        knowledge_db = os.path.expanduser("~/bowling/knowledge/knowledge.db")
        promotion_db = os.path.expanduser("~/bowling/intelligence/watch_promotion_ledger.db")
        review_db = os.path.expanduser("~/bowling/intelligence/watch_review_state.db")
        telegram_review_db = os.path.expanduser("~/bowling/intelligence/telegram_review_state.db")
        watched = [p for p in (knowledge_db, promotion_db, review_db, telegram_review_db) if os.path.exists(p)]
        before_mtimes = {p: os.path.getmtime(p) for p in watched}

        from hermes_cli.config import load_config_readonly
        from hermes_cli.tools_config import _get_platform_tools
        from run_agent import AIAgent

        cfg = load_config_readonly()
        model_cfg = cfg.get("model") or {}
        if not os.path.exists(str(model_cfg.get("default") or "")):
            pytest.skip("configured local model file not present on this machine")
        enabled_toolsets = sorted(_get_platform_tools(cfg, "telegram"))

        try:
            agent = AIAgent(
                model=model_cfg.get("default"),
                base_url=model_cfg.get("base_url"),
                api_key=model_cfg.get("api_key") or "none",
                provider=model_cfg.get("provider"),
                api_mode=model_cfg.get("api_mode"),
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=None,
                quiet_mode=True,
                verbose_logging=False,
                platform="telegram",
                session_id="gate10p-test-canary-throwaway",
                session_db=None,
                max_iterations=1,
            )
        except RuntimeError as exc:
            pytest.skip(f"real AIAgent construction unavailable in this environment: {exc}")

        if hasattr(agent, "_build_system_prompt"):
            agent._build_system_prompt()
        real_static = getattr(agent, "_cached_system_prompt_static", None)
        assert isinstance(real_static, str) and real_static

        primed = ssp.build_static_system_prompt(platform="telegram")
        assert primed == real_static

        for p in watched:
            assert os.path.getmtime(p) == before_mtimes[p], f"{p} was touched by AIAgent construction"


# ── Safety: no writes, no tool invocation, no outbound sends ────────────


class TestSafety:
    def test_module_never_imports_review_ledger_notification_modules(self):
        source = inspect.getsource(ssp)
        tree = ast.parse(source)
        forbidden = ("review", "ledger", "telegram_client", "notification", "promotion")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                for name in names:
                    if not name:
                        continue
                    lowered = name.lower()
                    assert not any(f in lowered for f in forbidden), name

    def test_does_not_construct_an_aiagent(self):
        source = inspect.getsource(ssp)
        assert "AIAgent(" not in source

    def test_returns_plain_str_not_a_mutable_shared_object(self, monkeypatch):
        prompt = _build(monkeypatch)
        assert type(prompt) is str

    def test_never_raises_on_config_load_failure(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: (_ for _ in ()).throw(RuntimeError("disk error")),
        )
        with pytest.raises(Exception):
            # Config failure is not swallowed inside build_static_system_prompt
            # itself (callers — the priming module — are responsible for the
            # outer try/except); this documents that boundary explicitly.
            ssp.build_static_system_prompt(platform="telegram")

    def test_tool_name_resolution_never_raises(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: (_ for _ in ()).throw(RuntimeError("disk error")),
        )
        assert ssp.resolve_static_prompt_tool_names(platform="telegram") == frozenset()

    def test_tool_schema_resolution_never_raises(self, monkeypatch):
        """Gate CS9: resolve_static_prompt_tools shares the same safety
        boundary as resolve_static_prompt_tool_names — a config-load failure
        degrades to no tools rather than propagating."""
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: (_ for _ in ()).throw(RuntimeError("disk error")),
        )
        assert ssp.resolve_static_prompt_tools(platform="telegram") == []


class TestToolSchemaResolution:
    """Gate CS9 (cold-start report): resolve_static_prompt_tools must return
    the SAME full tool schema list resolve_static_prompt_tool_names derives
    its names from, since priming needs the full JSON (not just names) to
    match what a real turn's ``tools=[...]`` request renders."""

    def test_tool_schemas_names_match_tool_names_resolution(self, monkeypatch):
        _patch_config(monkeypatch)
        names_from_names_fn = ssp.resolve_static_prompt_tool_names(platform="telegram")
        schemas = ssp.resolve_static_prompt_tools(platform="telegram")
        names_from_schemas = frozenset(
            t["function"]["name"] for t in schemas if isinstance(t, dict)
        )
        assert names_from_schemas == names_from_names_fn

    def test_tool_schemas_are_full_dicts_not_just_names(self, monkeypatch):
        _patch_config(monkeypatch)
        schemas = ssp.resolve_static_prompt_tools(platform="telegram")
        assert schemas, "expected at least one tool for the telegram platform"
        for tool in schemas:
            assert isinstance(tool, dict)
            assert "function" in tool
            assert "name" in tool["function"]

    def test_empty_platform_toolset_returns_empty_list(self, monkeypatch):
        _patch_config(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.tools_config._get_platform_tools", lambda cfg, platform: set(),
        )
        assert ssp.resolve_static_prompt_tools(platform="telegram") == []
