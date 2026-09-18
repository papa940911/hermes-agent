"""Side-effect-free construction of Hermes's real static system-prompt prefix (Gate 10P, extended Gate 10R, Gate CS10.1).

``agent.system_prompt.build_system_prompt_parts(agent)`` already splits Hermes's
system prompt into three cache tiers (see that module's docstring): ``stable``
(cross-session-stable identity/tool/behavioral guidance), ``context``
(cwd-dependent workspace snapshot + caller system_message), and ``volatile``
(skills index, memory/user-profile blocks, plugin sections, timestamp).

:func:`build_static_system_prompt` reuses that SAME function — the one every
real AIAgent turn calls — instead of duplicating any prompt text, and returns
only the ``stable`` tier: the largest prefix that is genuinely byte-identical
across sessions and turns, and therefore the only portion worth priming into
llama.cpp's KV cache.

Design note (why no full AIAgent): ``build_system_prompt_parts`` only reads a
few dozen attributes off its ``agent`` argument and calls a handful of pure,
read-only helpers (SOUL.md load, environment hints, skills index, tool
definitions) — it never touches conversation history, memory *content*,
session persistence, or any tool/Telegram/DB write path. Constructing a full
``AIAgent`` to obtain those same attributes would additionally instantiate a
memory provider, tool registry snapshot machinery, provider/client plumbing,
and session bookkeeping purely to throw it away — exactly what Gate 10O's
warmup docstring (``agent/llm_backend_warmup.py``) already identified as
unnecessary and unsafe to do on every gateway startup. Instead, this module
builds a minimal duck-typed view exposing only the attributes the stable tier
actually reads, populated from the SAME config-resolution calls a real
``AIAgent`` construction uses (``_get_platform_tools`` + ``get_tool_definitions``
for the tool set; ``load_config_readonly`` for model/provider/memory flags) —
so the result reflects the live configuration without hand-duplicating it.

Nothing here writes memory, persists a session, invokes a tool, or sends a
message on any platform.

Gate CS10.1 (memory-provider tools in the ``tools`` array): a real
``AIAgent`` doesn't stop at ``get_tool_definitions()`` for its tool
surface — ``agent_init.py`` also runs
``agent.memory_manager.inject_memory_provider_tools`` for whatever
``memory.provider`` is configured, appending that provider's own tool
schemas (e.g. this deployment's ``holographic`` provider contributes
``fact_store``/``fact_feedback``). Production validation found that
priming's tools array omitted these, and because the Qwen chat template
renders the full tools block AHEAD of the system-prompt content, that
length mismatch alone invalidated a large fraction of an otherwise
byte-identical cached prefix (``f_sim_best=0.587`` in production; see
``/tmp/hermes_cs10_prefix_divergence_report.txt``). ``resolve_static_prompt_tools``/
``resolve_static_prompt_tool_names`` now resolve this EFFECTIVE list (base
+ memory-provider tools) via
:func:`agent.memory_manager.resolve_configured_memory_provider_tool_schemas`
— side-effect-free and provider-agnostic (derives entirely from
``memory.provider`` in config; never hardcodes a provider name or tool
name). This does NOT extend to the volatile system-prompt tail (memory
*content*, session ID, timestamp) — that remains intentionally unprimed;
see :func:`build_primed_prefix_with_skills`'s docstring.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "build_static_system_prompt",
    "build_primed_prefix_with_skills",
    "resolve_static_prompt_tool_names",
    "resolve_static_prompt_tools",
]


def _resolve_static_prompt_tool_defs(*, platform: str, quiet_mode: bool = True) -> list:
    """The real tool schema list a fresh agent on *platform* would load, read-only.

    Mirrors ``gateway.run.HermesGateway._resolve_enabled_toolsets_for_source``'s
    toolset resolution (platform_toolsets config -> ``_get_platform_tools``)
    followed by the exact ``agent_init.init_agent`` call
    (``model_tools.get_tool_definitions``) that turns a toolset list into
    concrete tool schemas, filtered by each tool's own availability check_fn.
    No tool is invoked; only schemas/check_fns are read. Shared by
    :func:`resolve_static_prompt_tool_names` (names only) and
    :func:`resolve_static_prompt_tools` (full schemas, needed so priming's
    request matches the real ``tools=[...]`` a live turn sends — see Gate
    CS9 in the cold-start report: the chat template renders a ``# Tools``
    preamble ahead of the system message whenever ``tools`` is non-empty,
    so priming without it can never share a KV-cache prefix with a real
    turn regardless of how well the system-prompt text itself matches).
    """
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.tools_config import _get_platform_tools
        from model_tools import get_tool_definitions
        from agent.skill_utils import parse_config_string_list

        cfg = load_config_readonly() or {}
        enabled_toolsets = sorted(_get_platform_tools(cfg, platform))
        agent_cfg = cfg.get("agent") or {}
        disabled_toolsets = parse_config_string_list(agent_cfg.get("disabled_toolsets")) or None
        return get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=quiet_mode,
        )
    except Exception:
        logger.debug("static system prompt: tool resolution failed", exc_info=True)
        return []


def _resolve_static_prompt_effective_tool_defs(*, platform: str, quiet_mode: bool = True) -> list:
    """Gate CS10.1: base tool defs PLUS whatever the CONFIGURED external
    memory provider would contribute to a real agent's ``tools`` array.

    A real ``AIAgent`` doesn't stop at ``get_tool_definitions()`` — after
    base-tool resolution, ``agent_init.py`` also runs
    ``agent.memory_manager.inject_memory_provider_tools`` for whatever
    ``memory.provider`` is configured (e.g. this deployment's
    ``holographic`` provider, which contributes ``fact_store`` and
    ``fact_feedback``). ``_resolve_static_prompt_tool_defs`` alone only
    mirrored the FIRST half of that, so priming's ``tools`` array was
    silently shorter than a live turn's — and since the Qwen chat template
    renders the full tools block ahead of the system-prompt content, that
    length mismatch alone was enough to invalidate a large fraction of an
    otherwise byte-identical cached prefix (confirmed in production; see
    ``/tmp/hermes_cs10_prefix_divergence_report.txt``).

    Memory-provider tools are resolved via
    :func:`agent.memory_manager.resolve_configured_memory_provider_tool_schemas`
    — side-effect-free (constructs the configured provider only far enough
    to read its static ``get_tool_schemas()``; never initializes it, never
    connects, never calls a tool) and provider-agnostic (derives entirely
    from ``memory.provider`` in config; a different provider, or none,
    "just works" without any change here). Appended AFTER the base tools,
    matching ``agent_init.py``'s real injection order, since tool order is
    itself part of what the chat template renders.
    """
    base_tools = _resolve_static_prompt_tool_defs(platform=platform, quiet_mode=quiet_mode)
    try:
        from hermes_cli.config import load_config_readonly
        from hermes_cli.tools_config import _get_platform_tools
        from agent.skill_utils import parse_config_string_list
        from agent.memory_manager import resolve_configured_memory_provider_tool_schemas

        cfg = load_config_readonly() or {}
        enabled_toolsets = sorted(_get_platform_tools(cfg, platform))
        agent_cfg = cfg.get("agent") or {}
        disabled_toolsets = parse_config_string_list(agent_cfg.get("disabled_toolsets")) or None
        existing_tool_names = {
            t["function"]["name"]
            for t in base_tools
            if isinstance(t, dict) and isinstance(t.get("function"), dict)
        }
        memory_tools = resolve_configured_memory_provider_tool_schemas(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            memory_tool_present="memory" in existing_tool_names,
            existing_tool_names=existing_tool_names,
        )
    except Exception:
        logger.debug(
            "static system prompt: memory-provider tool resolution failed", exc_info=True
        )
        memory_tools = []
    return base_tools + memory_tools


def resolve_static_prompt_tool_names(*, platform: str, quiet_mode: bool = True) -> frozenset:
    """The real EFFECTIVE tool-name set (base + configured memory-provider
    tools) a fresh agent on *platform* would load, read-only.

    See :func:`_resolve_static_prompt_effective_tool_defs` for the
    underlying resolution. Kept in sync with :func:`resolve_static_prompt_tools`
    so ``valid_tool_names`` in the primed view (which gates some of the
    STABLE-tier guidance text itself, e.g. ``agent/system_prompt.py``'s
    ``"memory" in agent.valid_tool_names`` checks) reflects the same tool
    set a live turn would see — not just the ``tools`` array a request
    sends.
    """
    tools = _resolve_static_prompt_effective_tool_defs(platform=platform, quiet_mode=quiet_mode)
    return frozenset(t["function"]["name"] for t in tools if isinstance(t, dict))


def resolve_static_prompt_tools(*, platform: str, quiet_mode: bool = True) -> list:
    """The real EFFECTIVE tool schema list (full JSON, not just names) a
    fresh agent on *platform* would load, read-only — base tools PLUS
    whatever the configured external memory provider contributes. See
    :func:`_resolve_static_prompt_effective_tool_defs`.
    """
    return _resolve_static_prompt_effective_tool_defs(platform=platform, quiet_mode=quiet_mode)


def _static_prompt_agent_view(
    *,
    model: str,
    provider: str,
    platform: str,
    context_length: Optional[int],
    memory_enabled: bool,
    user_profile_enabled: bool,
    platform_hint_overrides: Optional[dict],
    valid_tool_names: frozenset,
) -> SimpleNamespace:
    """Duck-typed stand-in exposing exactly what the stable tier reads.

    Every attribute here is either a config-derived scalar or an explicit
    "no session/no memory/no plugin-snapshot" sentinel (None / "" / False) —
    never a fabricated conversation or user-visible value.
    """
    context_compressor = (
        SimpleNamespace(context_length=context_length) if context_length else None
    )
    return SimpleNamespace(
        # Identity / context-file loading — defaults matching a real fresh agent.
        load_soul_identity=True,
        skip_context_files=False,
        # Tooling.
        valid_tool_names=set(valid_tool_names),
        tools=[],
        # Model identity.
        model=model,
        provider=provider,
        platform=platform,
        # No real session: nothing below should ever read as "this is a
        # live conversation".
        session_id=None,
        pass_session_id=False,
        _session_db=None,
        _session_title_hint="",
        _bot_chat_timeless_prompt=False,
        context_compressor=context_compressor,
        # Memory: guidance text is config-gated, not content-gated — the
        # store/manager stay None so no memory content is ever read.
        _memory_store=None,
        _memory_enabled=memory_enabled,
        _user_profile_enabled=user_profile_enabled,
        _memory_manager=None,
        # Behavioral guidance toggles — real agent_init defaults.
        _tool_use_enforcement="auto",
        _execution_guidance="auto",
        _task_completion_guidance=True,
        _parallel_tool_call_guidance=True,
        _kanban_worker_guidance=None,
        _platform_hint_overrides=platform_hint_overrides,
        _bot_mode_protocol=True,
        _environment_probe=True,
        # Plugin-section freeze: no prior cached prompt to restore from.
        _cached_system_prompt=None,
    )


def build_static_system_prompt(
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    platform: str = "telegram",
    context_length: Optional[int] = None,
) -> str:
    """Assemble Hermes's real static/reusable system-prompt prefix.

    Reads the live config for anything not passed explicitly (model,
    provider, context window, memory flags, platform-hint overrides), resolves
    the real tool set for *platform* via :func:`resolve_static_prompt_tool_names`,
    and calls the UNMODIFIED ``agent.system_prompt.build_system_prompt_parts``
    — the same function every real turn uses — returning only its ``stable``
    tier. No conversation, session, memory write, tool call, or outbound
    message is ever created.
    """
    from hermes_cli.config import load_config_readonly

    cfg = load_config_readonly() or {}
    model_cfg = cfg.get("model") or {}
    resolved_model = model if model is not None else str(model_cfg.get("default") or "")
    resolved_provider = provider if provider is not None else str(model_cfg.get("provider") or "")
    resolved_context_length = (
        context_length if context_length is not None else model_cfg.get("context_length")
    )
    if not isinstance(resolved_context_length, int) or resolved_context_length <= 0:
        resolved_context_length = None

    mem_cfg = cfg.get("memory") or {}
    memory_enabled = bool(mem_cfg.get("memory_enabled", True))
    user_profile_enabled = bool(mem_cfg.get("user_profile_enabled", True))

    agent_cfg = cfg.get("agent") or {}
    platform_hint_overrides = agent_cfg.get("platform_hints")
    if not isinstance(platform_hint_overrides, dict):
        platform_hint_overrides = None

    valid_tool_names = resolve_static_prompt_tool_names(platform=platform)

    view = _static_prompt_agent_view(
        model=resolved_model,
        provider=resolved_provider,
        platform=platform,
        context_length=resolved_context_length,
        memory_enabled=memory_enabled,
        user_profile_enabled=user_profile_enabled,
        platform_hint_overrides=platform_hint_overrides,
        valid_tool_names=valid_tool_names,
    )

    from agent.system_prompt import build_system_prompt_parts

    parts = build_system_prompt_parts(view)
    return parts["stable"]


def build_primed_prefix_with_skills(
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    platform: str = "telegram",
    context_length: Optional[int] = None,
) -> "tuple[str, str]":
    """Gate 10R: the stable tier PLUS the skills index, for KV-cache priming.

    Gate 10P (:func:`build_static_system_prompt`) primes only the ``stable``
    tier, by design — the skills index is registry-mutable and therefore
    correctly lives in the ``volatile`` tier of a real turn's prompt (see
    ``agent.system_prompt.build_system_prompt_parts``'s docstring and
    ``tests/agent/test_static_system_prompt.py::test_skills_index_excluded_from_static_tier``,
    which this function must NOT violate — it does not change what "stable"
    means, it builds an ADDITIONAL, separate priming prefix).

    On the common real deployment (Telegram, no cwd context files — the
    ``context`` tier is empty), ``stable + "\\n\\n" + skills_index`` is a
    fully deterministic byte-for-byte prefix of what a real first turn
    sends, right up to where genuinely per-session content (memory,
    timestamp, session id) begins. Priming the backend's single KV-cache
    slot (``--parallel 1``) with this longer prefix — instead of just
    ``stable`` — lets a real first message reuse cache through the skills
    index too, instead of paying its full prefill cost from scratch.

    Uses the SAME :func:`agent.system_prompt.resolve_skills_prompt` helper
    that a real turn calls (no re-derivation, so this can never drift out
    of byte-parity with production) and the SAME duck-typed, side-effect-free
    view as :func:`build_static_system_prompt` (no AIAgent, no session, no
    memory content, no tool call, no network I/O other than the caller's
    own priming POST).

    Returns ``(prefix, skills_fingerprint)`` — the fingerprint
    (:func:`agent.prompt_builder.skills_manifest_fingerprint`) is a cheap,
    deterministic signal callers can use to detect that the on-disk skill
    registry changed since this prefix was primed (add/remove/edit a
    ``SKILL.md``) and decide whether to re-prime. An empty ``skills_index``
    (no skills installed, or none visible for *platform*) returns the plain
    stable prefix and an empty fingerprint is still returned for the
    (skill-less) registry state.
    """
    from hermes_cli.config import load_config_readonly

    cfg = load_config_readonly() or {}
    model_cfg = cfg.get("model") or {}
    resolved_model = model if model is not None else str(model_cfg.get("default") or "")
    resolved_provider = provider if provider is not None else str(model_cfg.get("provider") or "")
    resolved_context_length = (
        context_length if context_length is not None else model_cfg.get("context_length")
    )
    if not isinstance(resolved_context_length, int) or resolved_context_length <= 0:
        resolved_context_length = None

    mem_cfg = cfg.get("memory") or {}
    memory_enabled = bool(mem_cfg.get("memory_enabled", True))
    user_profile_enabled = bool(mem_cfg.get("user_profile_enabled", True))

    agent_cfg = cfg.get("agent") or {}
    platform_hint_overrides = agent_cfg.get("platform_hints")
    if not isinstance(platform_hint_overrides, dict):
        platform_hint_overrides = None

    valid_tool_names = resolve_static_prompt_tool_names(platform=platform)

    view = _static_prompt_agent_view(
        model=resolved_model,
        provider=resolved_provider,
        platform=platform,
        context_length=resolved_context_length,
        memory_enabled=memory_enabled,
        user_profile_enabled=user_profile_enabled,
        platform_hint_overrides=platform_hint_overrides,
        valid_tool_names=valid_tool_names,
    )

    from agent.system_prompt import build_system_prompt_parts, resolve_skills_prompt
    from agent.prompt_builder import skills_manifest_fingerprint

    parts = build_system_prompt_parts(view)
    try:
        skills_index = resolve_skills_prompt(view)
    except Exception:
        logger.debug("primed-prefix-with-skills: skills resolution failed", exc_info=True)
        skills_index = ""

    try:
        fingerprint = skills_manifest_fingerprint() if skills_index else ""
    except Exception:
        logger.debug("primed-prefix-with-skills: fingerprint failed", exc_info=True)
        fingerprint = ""

    prefix = "\n\n".join(
        p.strip() for p in (parts["stable"], parts["context"], skills_index) if p and p.strip()
    )
    return prefix, fingerprint
