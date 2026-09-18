"""Durable prefix-alignment diagnostic facility (Gate CS9-followup investigation;
made permanent and reused by Gate CS10's re-priming/staleness logic).

Originally written as a temporary investigation aid for the Gate CS9
prefix-drift investigation (see ``/tmp/hermes-cold-start-root-cause-report.txt``)
and imported from two call sites: ``agent/llm_static_prompt_priming.py`` and
``agent/conversation_loop.py``. Its source was later deleted while only the
compiled ``.pyc`` remained resident, which meant it could not be inspected,
modified, or version-controlled — Gate CS10 restores it to tracked source
(reconstructed from that ``.pyc``) and keeps it wired into the priming
module only, since that is the call site Gate CS10 actually needs for
verifying cache alignment; the ``conversation_loop.py`` call site is not
restored here (out of Gate CS10's scope).

Logs ONLY hashes, lengths, tool names, and small scalar runtime-state
flags — never prompt text, never conversation content, never memory/
profile content, never an API key or tool-call argument. All output lines
are prefixed with "PREFIXDIAG" so they can be grepped out of agent.log
independently of everything else.
"""

from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger("agent._prefixdiag")


def _tool_name(tool) -> str:
    try:
        return str(tool["function"]["name"])
    except Exception:
        return "<unnamed>"


def _stable_hash(obj) -> str:
    """Content hash: sort_keys=True so pure key-ORDER differences inside a
    single schema don't masquerade as content differences. Ordering of
    TOOLS in the list is reported separately (that's a real wire-format
    concern for llama.cpp's exact-prefix cache)."""
    try:
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    except Exception:
        blob = repr(obj).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()[:16]


def log_prefix_diag(tag: str, system_text: str, tools=None, extra: dict = None) -> None:
    """One PREFIXDIAG log line (INFO) summarizing a prefix-relevant snapshot.

    tag: e.g. "PRIME" (agent/llm_static_prompt_priming.py's own priming
    attempt) or "STALE_CHECK" (Gate CS10's staleness-check decision point).
    Originally also "REAL_TURN:<branch>" from the since-removed
    conversation_loop.py call site (branch = which code path produced that
    turn's system prompt: fresh_build / continuing_reuse / bot_stale_rebuild).
    """
    try:
        tools = tools or []
        names_in_order = [_tool_name(t) for t in tools]
        per_tool_hashes = [f"{n}:{_stable_hash(t)}" for n, t in zip(names_in_order, tools)]
        tools_sha256 = hashlib.sha256(
            json.dumps(tools, sort_keys=False, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        system_text = system_text or ""
        system_sha256 = hashlib.sha256(system_text.encode("utf-8")).hexdigest()
        cacheable_prefix_sha256 = hashlib.sha256(
            f"{tools_sha256}|{system_sha256}".encode("utf-8")
        ).hexdigest()
        extra_str = " ".join(f"{k}={v}" for k, v in sorted((extra or {}).items()))
        logger.info(
            "PREFIXDIAG tag=%s tool_count=%d tool_names=%s tool_hashes=%s tools_sha256=%s "
            "system_chars=%d system_sha256=%s cacheable_prefix_sha256=%s %s",
            tag, len(tools), ",".join(names_in_order), ",".join(per_tool_hashes),
            tools_sha256, len(system_text), system_sha256, cacheable_prefix_sha256, extra_str,
        )
    except Exception:
        logger.debug("PREFIXDIAG logging failed for tag=%s", tag, exc_info=True)
