"""Deterministic finalization for authoritative BowlingTracker personal data.

The native BowlingTracker tools already produce bounded final-form answers from
typed personal evidence. Returning that answer directly prevents a model from
changing sample counts, turning a 200 game into a "perfect game", inventing a
cause, or substituting other sources after a no-data/unavailable result.

Explicit multi-source requests are left in the ordinary agent loop so a future
fusion layer can consume both personal and expert/public evidence.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from agent.bowling_knowledge_finalization import has_explicit_web_intent, latest_user_text


BOWLINGTRACKER_TOOL_NAMES = frozenset(
    {
        "bowlingtracker_recent_sessions",
        "bowlingtracker_session_detail",
        "bowlingtracker_personal_summary",
        "bowlingtracker_session_analysis",
    }
)
PERSONAL_EVIDENCE_CONTRACT = "personal-bowling-evidence/v1"
SESSION_ANALYSIS_CONTRACT = "personal-bowling-session-analysis/v1"
# The session-analysis tool returns its own verified contract; adapter failures that
# escape it still use the shared personal-evidence failure envelope.
_ACCEPTED_CONTRACTS = {
    "bowlingtracker_session_analysis": frozenset(
        {SESSION_ANALYSIS_CONTRACT, PERSONAL_EVIDENCE_CONTRACT}
    ),
}

_EXPERT_MARKERS = (
    "mo pinel",
    "joe slowinski",
    "mark baker",
    "kegel says",
    "expert evidence",
    "expert advice",
    "bowling knowledge",
)
_PERSONAL_MARKERS = (
    "bowlingtracker",
    "my recent",
    "my last",
    "my session",
    "my games",
    "my shots",
    "my scores",
    "my misses",
    "my results",
)


def has_explicit_fusion_intent(text: str) -> bool:
    lowered = (text or "").casefold()
    return any(marker in lowered for marker in _EXPERT_MARKERS) and any(
        marker in lowered for marker in _PERSONAL_MARKERS
    )


def bowlingtracker_terminal_response(
    tool_name: str,
    function_result: Any,
    messages: Optional[list],
) -> Optional[str]:
    """Return a verified final-form personal-data answer, or continue normally."""
    if tool_name not in BOWLINGTRACKER_TOOL_NAMES or not isinstance(function_result, str):
        return None
    try:
        data = json.loads(function_result)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    accepted = _ACCEPTED_CONTRACTS.get(tool_name, frozenset({PERSONAL_EVIDENCE_CONTRACT}))
    if data.get("contractVersion") not in accepted:
        return None
    if data.get("dataSource") != "BowlingTracker":
        return None
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    user_text = latest_user_text(messages)
    if has_explicit_web_intent(user_text) or has_explicit_fusion_intent(user_text):
        return None
    return answer.strip()


__all__ = [
    "BOWLINGTRACKER_TOOL_NAMES",
    "bowlingtracker_terminal_response",
    "has_explicit_fusion_intent",
]

