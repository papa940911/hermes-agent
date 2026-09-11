"""Deterministic post-tool finalization policy for Bowling Knowledge.

Bowling Knowledge (the ``bowling_knowledge_answer`` native tool) is the
authoritative first path for bowling-domain knowledge questions. Its result
already carries the complete, safe, final-form answer text -- a grounded
answer with citations for a supported question, or a compact
insufficient-support message for an unsupported one (see
``skills/bowling-knowledge/scripts/hermes_tool.py``). A normal Hermes agent
loop has no built-in concept of "this tool result already IS the final
answer, stop here" -- without this module the outer model can (and, per the
reported bug, sometimes does) go on to call ``web_search`` or answer from its
own general knowledge after an unsupported result, exactly the silent
fallback this module exists to close off.

It also closes the other half of the same routing boundary (Rule 1): a
question can look bowling-domain to the *user* without the outer model ever
calling ``bowling_knowledge_answer`` at all -- observed live, the model
sometimes reasons ("this probably isn't in a niche corpus") and reaches
straight for ``web_search``/``web_extract`` instead of trying the corpus
first. :func:`bowling_knowledge_precall_block_reason` blocks exactly those
two web tools, and only those two, for a narrowly-detected bowling-domain
question until ``bowling_knowledge_answer`` has been tried this turn.

This module is intentionally narrow and provider/model agnostic: pure
string/dict logic, no model calls, no knowledge of any particular provider's
behavior. It only ever recognizes bowling_knowledge_answer and the two web
tools by name, and only overrides the loop when the user's own message did
not ask for current/public web information (Rule 4). It never inspects or
gates any other tool, and a non-bowling turn that never mentions this tool
or web search is completely unaffected.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

BOWLING_KNOWLEDGE_TOOL_NAME = "bowling_knowledge_answer"

# Rule 1 of the routing spec: web tools that must not run for an ordinary
# bowling-domain question until bowling_knowledge_answer has been tried this
# turn. Exactly the two tools registered in tools/web_tools.py -- this never
# gates any other tool (terminal, file, memory, etc. are all unaffected).
WEB_FALLBACK_TOOL_NAMES = frozenset({"web_search", "web_extract"})

# Rule 1 / Phase G: a narrow, deterministic bowling-domain marker vocabulary.
# Deliberately not a big taxonomy and no embeddings -- this is the corpus's
# own topic vocabulary (lane topography, ball motion, pin carry, drilling and
# layout theory, ...) plus the sport's own unambiguous governing-body/
# equipment terms. A bare generic word ("speed", "differential") is
# deliberately NOT a marker on its own -- see the false-positive tests.
_BOWLING_DOMAIN_PATTERN = re.compile(
    r"""\b(
        bowling|bowler|bowlers|tenpin|ten-pin|
        wtba|usbc|pba|
        pin\s+carry|lane\s+topography|lane\s+condition|oil\s+pattern|
        pap|positive\s+axis\s+point|
        rev\s+rate|axis\s+tilt|axis\s+rotation|entry\s+angle|
        reactive\s+resin|ball\s+motion|ball\s+layout|drilling\s+layout
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Corpus expert names (see the production `source` table) are ambiguous on
# their own -- "Mark Baker" is also a musician's name (Phase G false-positive
# example) -- so a bare name match additionally requires a nearby
# knowledge-attribution word typical of "what does <expert> say/recommend"
# phrasing, never on the name alone.
_BOWLING_EXPERT_NAMES = ("mo pinel", "kegel", "joe slowinski", "slowinski", "mark baker")
_EXPERT_ATTRIBUTION_PATTERN = re.compile(
    r"\b(say|says|said|recommend|recommends|recommended|advice|advise|"
    r"technique|coach|coaching|teach|teaches|taught)\b",
    re.IGNORECASE,
)


def is_bowling_domain_question(text: str) -> bool:
    """Conservative, deterministic bowling-domain check for Rule 1.

    True only for the narrow marker vocabulary above, or a known corpus
    expert's name combined with knowledge-attribution phrasing. Never fires
    on a bare generic word (speed, CPU, differential) or a bare ambiguous
    name alone.
    """
    if not text:
        return False
    if _BOWLING_DOMAIN_PATTERN.search(text):
        return True
    lowered = text.lower()
    if any(name in lowered for name in _BOWLING_EXPERT_NAMES):
        return bool(_EXPERT_ATTRIBUTION_PATTERN.search(lowered))
    return False


# Rule 4 / Phase H of the routing spec: an explicit, narrow allowlist of
# markers that signal the USER themselves asked for current/public/web
# information. Deliberately not a big taxonomy (Phase G) -- this is the
# exact small vocabulary the routing spec's own examples use ("current",
# "latest", "search", "web", "online", "look up ... online/on the web").
# An unsupported Bowling Knowledge result never implies this on its own.
_EXPLICIT_WEB_INTENT_PATTERN = re.compile(
    r"\b(current(?:ly)?|latest|up[\s-]?to[\s-]?date|search|\bweb\b|online|look\s+(?:this|it)?\s*up)\b",
    re.IGNORECASE,
)


def has_explicit_web_intent(text: str) -> bool:
    """Conservative, deterministic check for explicit current/public-web intent.

    Matches only the small marker vocabulary above. Being unsupported by the
    curated corpus does NOT itself imply permission to search the web --
    only the user's own words do.
    """
    if not text:
        return False
    return bool(_EXPLICIT_WEB_INTENT_PATTERN.search(text))


def latest_user_text(messages: Optional[list]) -> str:
    """The most recent real (non-synthetic) user message text in the transcript."""
    for message in reversed(messages or []):
        if not isinstance(message, Mapping):
            continue
        if message.get("role") != "user":
            continue
        if any(
            message.get(flag)
            for flag in (
                "_pre_verify_synthetic",
                "_kanban_stop_synthetic",
                "_active_turn_redirect",
            )
        ):
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, Mapping) and part.get("type") == "text"
            ]
            joined = "\n".join(p for p in parts if p)
            if joined:
                return joined
    return ""


def _format_citations(citations: Any) -> str:
    if not isinstance(citations, list) or not citations:
        return ""
    lines = []
    for citation in citations:
        if not isinstance(citation, Mapping):
            continue
        source_name = citation.get("source_name") or ""
        document_title = citation.get("document_title") or ""
        locator = citation.get("locator") or ""
        url = citation.get("url") or ""
        label = source_name
        if document_title and document_title != source_name:
            label = f"{label}, {document_title}" if label else document_title
        if locator:
            label = f"{label} ({locator})" if label else locator
        if url:
            label = f"{label} — {url}" if label else url
        if label:
            lines.append(f"- {label}")
    if not lines:
        return ""
    return "Sources:\n" + "\n".join(lines)


def _single_section_text(section: Mapping[str, Any]) -> Optional[str]:
    answer = section.get("answer")
    if not isinstance(answer, str) or not answer:
        return None
    citation_block = _format_citations(section.get("citations"))
    return answer if not citation_block else f"{answer}\n\n{citation_block}"


def bowling_knowledge_terminal_response(
    tool_name: str,
    function_result: Any,
    messages: Optional[list],
) -> Optional[str]:
    """Return the deterministic final-turn response text, or ``None`` to continue normally.

    Fires only for a successful (non-error) ``bowling_knowledge_answer``
    result, and only when the user's own message carries no explicit
    current/public-web intent. Returns ``None`` for anything else -- a
    ``tool_error()`` payload, an unrecognized shape, malformed JSON, or an
    explicit web/current request -- so the caller falls through to the
    normal agent loop unchanged.
    """
    if tool_name != BOWLING_KNOWLEDGE_TOOL_NAME:
        return None
    if not isinstance(function_result, str):
        return None
    try:
        data = json.loads(function_result)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if "answer" not in data and "expert_sections" not in data:
        # tool_error() shape (an "error" key) or something unrecognized --
        # an operational failure is not this module's concern; let the
        # normal error-handling / retry path run.
        return None
    if has_explicit_web_intent(latest_user_text(messages)):
        return None

    if data.get("comparison_mode"):
        sections = data.get("expert_sections")
        if not isinstance(sections, list) or not sections:
            return None
        parts = []
        for section in sections:
            if not isinstance(section, Mapping):
                continue
            text = _single_section_text(section)
            if text is None:
                continue
            name = section.get("source_name") or "Unknown source"
            parts.append(f"**{name}:**\n{text}")
        if not parts:
            return None
        return "\n\n".join(parts)

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer:
        return None
    citation_block = _format_citations(data.get("citations"))
    return answer if not citation_block else f"{answer}\n\n{citation_block}"


def bowling_knowledge_pending_in_tool_calls(tool_calls: Any) -> bool:
    """Rule 5: True when this tool round includes a call to
    ``bowling_knowledge_answer``.

    Rule 2/3 make that call's own result (via
    :func:`bowling_knowledge_terminal_response`) the sole authoritative,
    terminal answer for the turn -- but that check only runs AFTER the tool
    executes. The same assistant message that carries the tool call can
    also carry free-text ``content`` (a "let me explain how this works
    while I check..." preamble some providers emit alongside a function
    call). Left alone, that content is surfaced to the user as ordinary
    mid-turn commentary before the tool result -- and before the
    finalization check -- ever runs, which is exactly the "supplement the
    corpus answer with general bowling knowledge" failure Rule 2/3 closes
    off, just one step earlier. The caller uses this to withhold that
    specific message's commentary instead of streaming it unreviewed; the
    tool call itself, and everything that happens after it runs, is
    unaffected.
    """
    for tool_call in tool_calls or []:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", None)
        if name is None and isinstance(tool_call, Mapping):
            function = tool_call.get("function")
            if isinstance(function, Mapping):
                name = function.get("name")
        if name == BOWLING_KNOWLEDGE_TOOL_NAME:
            return True
    return False


def bowling_knowledge_precall_block_reason(
    tool_name: str,
    messages: Optional[list],
    bowling_knowledge_attempted: bool,
) -> Optional[str]:
    """Rule 1: an ordinary bowling-domain question must try
    ``bowling_knowledge_answer`` before a web fallback tool runs.

    Returns a short block-reason message (to redirect the model instead of
    letting the web call execute) or ``None`` to let the call through
    unchanged. Only ever considers :data:`WEB_FALLBACK_TOOL_NAMES` -- every
    other tool (terminal, file, memory, bowling_knowledge_answer itself,
    ...) is always allowed through this gate.
    """
    if tool_name not in WEB_FALLBACK_TOOL_NAMES:
        return None
    if bowling_knowledge_attempted:
        return None
    text = latest_user_text(messages)
    if not is_bowling_domain_question(text):
        return None
    if has_explicit_web_intent(text):
        return None
    return (
        "This looks like a bowling-domain knowledge question. Call "
        "bowling_knowledge_answer first -- it is the authoritative source "
        "for bowling technique, ball motion, and expert knowledge -- "
        "before searching the web."
    )
