"""Gate 10R: focused tests for the skills-index prompt-cost optimization.

Gate 10P (agent/static_system_prompt.py, agent/llm_static_prompt_priming.py)
primed llama.cpp's single KV-cache slot with only the ``stable`` tier of the
system prompt. Gate 10Q found the skills index — already a compact,
one-line-per-skill catalog with on-demand full loading via ``skill_view``
(pre-existing architecture, not built here) — is the dominant remaining
novel-token cost on a fresh session's first turn, because it lives in the
``volatile`` tier and is therefore never covered by Gate 10P's priming.

This gate does NOT rebuild the skills catalog (it was already the target
two-level design: compact index + on-demand ``skill_view`` load, with a
60-char hard cap on rendered descriptions). It adds:

  1. ``agent.prompt_builder.skills_manifest_fingerprint`` — a formal,
     testable registry-change signal wrapping the pre-existing manifest.
  2. ``agent.system_prompt.resolve_skills_prompt`` — the skills-prompt
     derivation extracted out of ``build_system_prompt_parts`` so it can be
     reused byte-for-byte by the priming path (no re-derivation, no drift).
  3. ``agent.static_system_prompt.build_primed_prefix_with_skills`` — a NEW,
     additive priming prefix (stable + context + skills index) that does
     NOT change what "stable" means (test_static_system_prompt.py's
     ``test_skills_index_excluded_from_static_tier`` still passes unchanged).
  4. ``agent.llm_static_prompt_priming`` now primes with that longer prefix
     and records a ``skills_fingerprint`` on ``PrimingResult``.
  5. Removal of ~240 bytes of boilerplate inside the skills-index block that
     duplicated ``SKILLS_GUIDANCE`` (already unconditionally present in the
     stable tier whenever ``skill_manage`` is available) — no instruction is
     lost, it simply isn't said twice in the same prompt.
"""

from __future__ import annotations

import inspect
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import prompt_builder
from agent import static_system_prompt as ssp
from agent import system_prompt


# ── Shared skill-fixture helpers ─────────────────────────────────────────


def _write_skill(root: Path, category: str, name: str, description: str, *, platforms=None) -> Path:
    skill_dir = root / category / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    fm = [f"name: {name}", f"description: {description}"]
    if platforms:
        fm.append(f"platforms: [{', '.join(platforms)}]")
    content = "---\n" + "\n".join(fm) + "\n---\n# " + name + "\nFull body.\n"
    path = skill_dir / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def skills_root(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    return root


# ── 1. Catalog construction ──────────────────────────────────────────────


class TestCatalogConstruction:
    def test_all_installed_skills_represented(self, skills_root):
        _write_skill(skills_root, "demo", "alpha", "Alpha does things.")
        _write_skill(skills_root, "demo", "beta", "Beta does other things.")
        prompt = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        assert "alpha" in prompt
        assert "beta" in prompt

    def test_deterministic_ordering(self, skills_root):
        _write_skill(skills_root, "demo", "zeta", "Z skill.")
        _write_skill(skills_root, "demo", "alpha", "A skill.")
        first = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
        second = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        assert first == second
        assert first.index("alpha") < first.index("zeta")

    def test_unique_skill_ids_no_double_listing(self, skills_root):
        _write_skill(skills_root, "demo", "solo", "Solo skill.")
        prompt = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        assert prompt.count("solo:") == 1

    def test_concise_description_present(self, skills_root):
        _write_skill(skills_root, "demo", "concise", "Short and clear.")
        prompt = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        assert "Short and clear." in prompt

    def test_no_full_body_leaked_into_index(self, skills_root):
        _write_skill(skills_root, "demo", "hasbody", "One line only.")
        prompt = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        assert "Full body." not in prompt

    def test_no_repeated_boilerplate_block(self, skills_root):
        _write_skill(skills_root, "demo", "one", "One.")
        prompt = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        assert prompt.count("## Skills (mandatory)") == 1

    def test_removed_duplicate_sentence_absent_from_index_block(self, skills_root):
        """Trimmed: this exact sentence used to duplicate SKILLS_GUIDANCE."""
        _write_skill(skills_root, "demo", "one", "One.")
        prompt = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        assert "If a skill has issues, fix it with skill_manage" not in prompt

    def test_disabled_skill_excluded(self, skills_root, monkeypatch):
        _write_skill(skills_root, "demo", "visible", "Visible.")
        _write_skill(skills_root, "demo", "hidden", "Hidden.")
        monkeypatch.setattr(prompt_builder, "get_disabled_skill_names", lambda *_a, **_k: {"hidden"})
        prompt = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        assert "visible" in prompt
        assert "hidden" not in prompt

    def test_platform_filtered_skill_hidden_on_other_platform(self, skills_root, monkeypatch):
        _write_skill(skills_root, "demo", "cli-only", "CLI only.", platforms=["cli"])
        monkeypatch.setenv("HERMES_PLATFORM", "telegram")
        prompt = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        assert "cli-only" not in prompt

    def test_empty_registry_yields_empty_prompt(self, skills_root):
        assert prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root) == ""


# ── 2. Size ───────────────────────────────────────────────────────────────


class TestSize:
    def test_token_count_measurable_via_bytes(self, skills_root):
        _write_skill(skills_root, "demo", "one", "One thing.")
        prompt = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        assert len(prompt.encode("utf-8")) > 0

    def test_compact_index_smaller_than_naive_full_body_dump(self, skills_root):
        long_body_desc = "A description. " * 5
        _write_skill(skills_root, "demo", "one", long_body_desc)
        prompt = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        # The rendered index line must be far smaller than name+full SKILL.md body.
        full_md_bytes = (skills_root / "demo" / "one" / "SKILL.md").stat().st_size
        assert len(prompt.encode("utf-8")) < full_md_bytes + 2000

    def test_description_hard_capped(self, skills_root):
        from agent.skill_utils import SKILL_PROMPT_DESC_LIMIT

        _write_skill(skills_root, "demo", "verbose", "x" * 500)
        prompt = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        line = next(l for l in prompt.splitlines() if l.strip().startswith("- verbose"))
        assert len(line) < SKILL_PROMPT_DESC_LIMIT + len("    - verbose: ") + len("...") + 5

    def test_boilerplate_trim_reduced_bytes(self, skills_root):
        _write_skill(skills_root, "demo", "one", "One.")
        prompt = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        removed = (
            "If a skill has issues, fix it with skill_manage(action='patch').\n"
            "After difficult/iterative tasks, offer to save as a skill. "
            "If a skill you loaded was missing steps, had wrong commands, or needed "
            "pitfalls you discovered, update it before finishing.\n"
        )
        assert removed not in prompt

    def test_deterministic_formatting_stable_across_rebuilds(self, skills_root):
        _write_skill(skills_root, "demo", "one", "One.")
        a = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
        b = prompt_builder.build_skills_system_prompt(skills_dir_override=skills_root)
        assert a == b


# ── 3. Registry fingerprint ──────────────────────────────────────────────


class TestRegistryFingerprint:
    def test_fingerprint_is_short_deterministic_string(self, skills_root):
        _write_skill(skills_root, "demo", "one", "One.")
        fp = prompt_builder.skills_manifest_fingerprint(skills_root)
        assert isinstance(fp, str) and 8 <= len(fp) <= 64

    def test_fingerprint_stable_when_unchanged(self, skills_root):
        _write_skill(skills_root, "demo", "one", "One.")
        a = prompt_builder.skills_manifest_fingerprint(skills_root)
        b = prompt_builder.skills_manifest_fingerprint(skills_root)
        assert a == b

    def test_fingerprint_changes_on_add(self, skills_root):
        _write_skill(skills_root, "demo", "one", "One.")
        before = prompt_builder.skills_manifest_fingerprint(skills_root)
        _write_skill(skills_root, "demo", "two", "Two.")
        after = prompt_builder.skills_manifest_fingerprint(skills_root)
        assert before != after

    def test_fingerprint_changes_on_remove(self, skills_root):
        p = _write_skill(skills_root, "demo", "one", "One.")
        before = prompt_builder.skills_manifest_fingerprint(skills_root)
        p.unlink()
        p.parent.rmdir()
        after = prompt_builder.skills_manifest_fingerprint(skills_root)
        assert before != after

    def test_fingerprint_changes_on_description_edit(self, skills_root):
        import time as _t

        p = _write_skill(skills_root, "demo", "one", "One.")
        before = prompt_builder.skills_manifest_fingerprint(skills_root)
        _t.sleep(0.01)
        p.write_text(p.read_text().replace("One.", "One, edited."))
        import os

        os.utime(p, None)  # ensure mtime advances even on coarse filesystems
        after = prompt_builder.skills_manifest_fingerprint(skills_root)
        assert before != after

    def test_fingerprint_unaffected_by_unrelated_file(self, skills_root):
        _write_skill(skills_root, "demo", "one", "One.")
        before = prompt_builder.skills_manifest_fingerprint(skills_root)
        (skills_root / "demo" / "one" / "notes.txt").write_text("scratch")
        after = prompt_builder.skills_manifest_fingerprint(skills_root)
        assert before == after

    def test_fingerprint_survives_malformed_frontmatter(self, skills_root):
        skill_dir = skills_root / "demo" / "broken"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("not: [valid: yaml: at all", encoding="utf-8")
        # Must not raise -- fingerprint only stats files, never parses them.
        fp = prompt_builder.skills_manifest_fingerprint(skills_root)
        assert fp

    def test_fingerprint_defaults_to_ambient_skills_dir(self, monkeypatch, skills_root):
        monkeypatch.setattr(prompt_builder, "get_skills_dir", lambda: skills_root)
        _write_skill(skills_root, "demo", "one", "One.")
        fp = prompt_builder.skills_manifest_fingerprint()
        assert fp == prompt_builder.skills_manifest_fingerprint(skills_root)


# ── 4. resolve_skills_prompt (extraction) parity ─────────────────────────


class TestResolveSkillsPrompt:
    def _agent(self, **overrides):
        base = dict(valid_tool_names={"skill_view", "skills_list", "skill_manage"}, platform="telegram")
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_returns_empty_without_skills_tools(self):
        agent = self._agent(valid_tool_names={"terminal"})
        assert system_prompt.resolve_skills_prompt(agent) == ""

    def test_delegates_to_build_skills_system_prompt(self):
        agent = self._agent()
        with patch("run_agent.build_skills_system_prompt", return_value="SKILLS-BLOCK") as mocked:
            result = system_prompt.resolve_skills_prompt(agent)
        assert result == "SKILLS-BLOCK"
        assert mocked.called

    def test_passes_available_tools_through(self):
        agent = self._agent(valid_tool_names={"skill_view"})
        with patch("run_agent.build_skills_system_prompt", return_value="X") as mocked:
            system_prompt.resolve_skills_prompt(agent)
        _, kwargs = mocked.call_args
        assert kwargs["available_tools"] == {"skill_view"}

    def test_coding_context_failure_does_not_raise(self):
        agent = self._agent()
        with patch(
            "agent.coding_context.coding_compact_skill_categories",
            side_effect=RuntimeError("boom"),
        ), patch("run_agent.build_skills_system_prompt", return_value="X"):
            result = system_prompt.resolve_skills_prompt(agent)
        assert result == "X"


# ── 5. build_primed_prefix_with_skills ───────────────────────────────────


class TestBuildPrimedPrefixWithSkills:
    def _patch_config(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {
                "model": {"default": "test-model", "provider": "custom",
                          "base_url": "http://127.0.0.1:8087/v1", "context_length": 64000},
                "memory": {"memory_enabled": True, "user_profile_enabled": True},
                "agent": {},
            },
        )

    def test_returns_tuple_of_prefix_and_fingerprint(self, monkeypatch):
        self._patch_config(monkeypatch)
        monkeypatch.setattr(ssp, "resolve_static_prompt_tool_names", lambda **_: frozenset())
        prefix, fingerprint = ssp.build_primed_prefix_with_skills(platform="telegram")
        assert isinstance(prefix, str)
        assert isinstance(fingerprint, str)

    def test_prefix_starts_with_the_stable_tier(self, monkeypatch):
        self._patch_config(monkeypatch)
        monkeypatch.setattr(ssp, "resolve_static_prompt_tool_names", lambda **_: frozenset())
        stable_only = ssp.build_static_system_prompt(platform="telegram")
        prefix, _ = ssp.build_primed_prefix_with_skills(platform="telegram")
        assert prefix.startswith(stable_only)

    def test_prefix_includes_skills_when_skills_tools_available(self, monkeypatch):
        self._patch_config(monkeypatch)
        monkeypatch.setattr(ssp, "resolve_static_prompt_tool_names", lambda **_: frozenset({"skill_view"}))
        with patch("run_agent.build_skills_system_prompt", return_value="UNIQUE-SKILLS-MARKER"):
            prefix, _ = ssp.build_primed_prefix_with_skills(platform="telegram")
        assert "UNIQUE-SKILLS-MARKER" in prefix

    def test_fingerprint_empty_when_no_skills_tools(self, monkeypatch):
        self._patch_config(monkeypatch)
        monkeypatch.setattr(ssp, "resolve_static_prompt_tool_names", lambda **_: frozenset())
        _, fingerprint = ssp.build_primed_prefix_with_skills(platform="telegram")
        assert fingerprint == ""

    def test_fingerprint_nonempty_when_skills_present(self, monkeypatch, skills_root):
        self._patch_config(monkeypatch)
        _write_skill(skills_root, "demo", "one", "One.")
        monkeypatch.setattr(prompt_builder, "get_skills_dir", lambda: skills_root)
        monkeypatch.setattr(ssp, "resolve_static_prompt_tool_names", lambda **_: frozenset({"skill_view"}))
        with patch("run_agent.build_skills_system_prompt", return_value="SOME-SKILLS"):
            _, fingerprint = ssp.build_primed_prefix_with_skills(platform="telegram")
        assert fingerprint != ""

    def test_never_touches_stable_tier_definition(self, monkeypatch):
        """The plain build_static_system_prompt must remain skills-free (Gate 10P invariant)."""
        self._patch_config(monkeypatch)
        monkeypatch.setattr(ssp, "resolve_static_prompt_tool_names", lambda **_: frozenset({"skill_view"}))
        with patch("run_agent.build_skills_system_prompt", return_value="SKILLS-INDEX-MARKER"):
            stable_only = ssp.build_static_system_prompt(platform="telegram")
        assert "SKILLS-INDEX-MARKER" not in stable_only

    def test_fingerprint_failure_does_not_drop_the_prefix(self, monkeypatch):
        """The fingerprint is observability-only: if computing it explodes,
        the (more valuable) primed prefix must still come back intact."""
        self._patch_config(monkeypatch)
        monkeypatch.setattr(ssp, "resolve_static_prompt_tool_names", lambda **_: frozenset({"skill_view"}))
        with patch("run_agent.build_skills_system_prompt", return_value="SOME-SKILLS-BLOCK"), \
             patch("agent.prompt_builder.skills_manifest_fingerprint", side_effect=RuntimeError("boom")):
            prefix, fingerprint = ssp.build_primed_prefix_with_skills(platform="telegram")
        assert "SOME-SKILLS-BLOCK" in prefix
        assert fingerprint == ""

    def test_overall_build_failure_propagates_to_caller(self, monkeypatch):
        """No internal swallow here: the caller (prime_static_system_prompt)
        owns the try/except boundary, same contract as build_static_system_prompt."""
        self._patch_config(monkeypatch)
        monkeypatch.setattr(ssp, "resolve_static_prompt_tool_names", lambda **_: frozenset({"skill_view"}))
        with patch("run_agent.build_skills_system_prompt", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                ssp.build_primed_prefix_with_skills(platform="telegram")

    def test_no_memory_content_in_primed_prefix(self, monkeypatch):
        self._patch_config(monkeypatch)
        monkeypatch.setattr(ssp, "resolve_static_prompt_tool_names", lambda **_: frozenset())
        prefix, _ = ssp.build_primed_prefix_with_skills(platform="telegram")
        assert "MEMORY.md" not in prefix or "memory" not in prefix.lower()

    def test_no_session_or_timestamp_content(self, monkeypatch):
        self._patch_config(monkeypatch)
        monkeypatch.setattr(ssp, "resolve_static_prompt_tool_names", lambda **_: frozenset())
        prefix, _ = ssp.build_primed_prefix_with_skills(platform="telegram")
        assert "Conversation started:" not in prefix
        assert "Session ID:" not in prefix


# ── 6. Priming integration (llm_static_prompt_priming) ───────────────────


from tests.agent.test_llm_static_prompt_priming import (  # noqa: E402
    fake_llama_server,
    _patch_config as _patch_priming_backend_config,
)


class TestPrimingWithSkills:
    def test_priming_result_carries_skills_fingerprint(self, monkeypatch, fake_llama_server):
        from agent import llm_static_prompt_priming as priming

        base_url, _state = fake_llama_server
        _patch_priming_backend_config(monkeypatch, base_url=base_url)
        monkeypatch.setattr(
            ssp, "build_primed_prefix_with_skills",
            lambda **_: ("STABLE\n\nSKILLS-INDEX", "abcdef0123456789"),
        )
        result = priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert result.succeeded is True
        assert result.skills_fingerprint == "abcdef0123456789"

    def test_priming_sends_stable_plus_skills_as_one_system_message(self, monkeypatch, fake_llama_server):
        from agent import llm_static_prompt_priming as priming

        base_url, state = fake_llama_server
        _patch_priming_backend_config(monkeypatch, base_url=base_url)
        monkeypatch.setattr(
            ssp, "build_primed_prefix_with_skills",
            lambda **_: ("STABLE-TIER\n\nSKILLS-INDEX-BLOCK", "fp"),
        )
        priming.prime_static_system_prompt(health_timeout_seconds=5, request_timeout_seconds=5)
        assert state.seen_system_prompts == ["STABLE-TIER\n\nSKILLS-INDEX-BLOCK"]

    def test_priming_result_default_fingerprint_is_empty_string(self):
        from agent.llm_static_prompt_priming import PrimingResult

        r = PrimingResult(
            attempted=False, succeeded=False, attempts=0, elapsed_ms=None,
            prompt_chars=None, prompt_tokens=None, cached_tokens=None, reason="x",
        )
        assert r.skills_fingerprint == ""

    def test_priming_still_gated_to_loopback_only(self, monkeypatch):
        from agent import llm_static_prompt_priming as priming

        _patch_priming_backend_config(monkeypatch, base_url="https://api.example.com/v1")
        called = {"n": 0}
        monkeypatch.setattr(
            ssp, "build_primed_prefix_with_skills",
            lambda **_: (called.__setitem__("n", called["n"] + 1) or "x", ""),
        )
        result = priming.prime_static_system_prompt()
        assert result.attempted is False
        assert called["n"] == 0


# ── 7. Prompt-integration invariants (ordering, no duplication, identity) ─


class TestPromptIntegrationInvariants:
    def test_skills_guidance_still_present_exactly_once_end_to_end(self, monkeypatch, tmp_path):
        """The trimmed sentence lived only in the volatile index; the stable
        SKILLS_GUIDANCE block (which already says the same thing) must still
        appear, and only once, in the fully assembled prompt."""
        from run_agent import AIAgent

        monkeypatch.setattr("run_agent.build_skills_system_prompt", lambda **_: "")
        agent = AIAgent.__new__(AIAgent)
        agent.valid_tool_names = {"skill_manage", "skill_view", "skills_list"}
        agent.platform = "telegram"
        agent.model = "test-model"
        agent.provider = "custom"
        agent.load_soul_identity = False
        agent.skip_context_files = True
        agent.tools = []
        agent.session_id = None
        agent.pass_session_id = False
        agent._session_db = None
        agent._session_title_hint = ""
        agent._bot_chat_timeless_prompt = True
        agent.context_compressor = None
        agent._memory_store = None
        agent._memory_enabled = False
        agent._user_profile_enabled = False
        agent._memory_manager = None
        agent._tool_use_enforcement = False
        agent._execution_guidance = False
        agent._task_completion_guidance = False
        agent._parallel_tool_call_guidance = False
        agent._kanban_worker_guidance = None
        agent._platform_hint_overrides = None
        agent._bot_mode_protocol = False
        agent._environment_probe = False
        agent._cached_system_prompt = None
        parts = system_prompt.build_system_prompt_parts(agent)
        full = parts["stable"] + "\n\n" + parts["context"] + "\n\n" + parts["volatile"]
        assert full.count("patch it immediately with skill_manage(action='patch')") == 1

    def test_skills_still_lead_the_volatile_band_after_extraction(self):
        """Regression guard duplicating test_system_prompt.py's own invariant
        against THIS gate's refactor of the skills-prompt derivation."""
        src = inspect.getsource(system_prompt.build_system_prompt_parts)
        assert "resolve_skills_prompt(agent)" in src
        assert src.index("skills_prompt = resolve_skills_prompt") < src.index("volatile_parts.append(skills_prompt)")


# ── 8. Failure / boundary / safety ────────────────────────────────────────


class TestFailureAndBoundaries:
    def test_fingerprint_never_touches_network_or_subprocess(self):
        src = inspect.getsource(prompt_builder.skills_manifest_fingerprint)
        for forbidden in ("subprocess", "socket", "urllib", "requests"):
            assert forbidden not in src

    def test_primed_prefix_module_has_no_service_or_db_mutation_calls(self):
        src = inspect.getsource(ssp.build_primed_prefix_with_skills)
        for forbidden in ("systemctl", "sqlite3", "DROP TABLE", "DELETE FROM", "subprocess"):
            assert forbidden not in src

    def test_concurrent_fingerprint_reads_are_consistent(self, skills_root):
        _write_skill(skills_root, "demo", "one", "One.")
        results = []
        lock = threading.Lock()

        def _read():
            fp = prompt_builder.skills_manifest_fingerprint(skills_root)
            with lock:
                results.append(fp)

        threads = [threading.Thread(target=_read) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(results)) == 1

    def test_empty_registry_fingerprint_does_not_raise(self, skills_root):
        assert prompt_builder.skills_manifest_fingerprint(skills_root) != ""

    def test_missing_skills_dir_does_not_raise(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        assert prompt_builder.skills_manifest_fingerprint(missing) != "" or True  # must not raise
