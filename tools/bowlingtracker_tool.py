#!/usr/bin/env python3
"""Read-only native Hermes tools for personal BowlingTracker evidence.

The implementation lives in the outer repository's bowlingtracker skill.  This
module only supplies stable model-tool schemas, registration, and fail-closed
error handling.  Callers cannot supply an endpoint, file/database path, SQL,
credentials, or a write/sync operation.
"""

import importlib.util
import sys
from pathlib import Path

from tools.registry import registry, tool_result


_ADAPTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "bowlingtracker"
    / "scripts"
    / "personal_data.py"
)
_adapter = None


def _get_adapter():
    global _adapter
    if _adapter is None:
        spec = importlib.util.spec_from_file_location(
            "bowlingtracker_personal_data_adapter", _ADAPTER_PATH
        )
        if spec is None or spec.loader is None:  # pragma: no cover - installation failure
            raise RuntimeError("BowlingTracker personal-data adapter could not be loaded")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _adapter = module
    return _adapter


def _run(operation: str, **arguments) -> str:
    try:
        adapter = _get_adapter()
        result = getattr(adapter, operation)(**arguments)
    except Exception as exc:  # noqa: BLE001 - the envelope must never leak internals or secrets
        try:
            adapter = _get_adapter()
            result = adapter.failure_result(exc)
        except Exception:  # pragma: no cover - broken installation fallback
            result = {
                "contractVersion": "personal-bowling-evidence/v1",
                "status": "unavailable",
                "dataSource": "BowlingTracker",
                "evidenceKinds": [],
                "dataAsOf": None,
                "error": {
                    "code": "internal_error",
                    "message": "Personal BowlingTracker data is unavailable right now.",
                },
                "evidence": [],
            }
    return tool_result(result)


def bowlingtracker_recent_sessions_tool(**arguments) -> str:
    return _run("recent_sessions", **arguments)


def bowlingtracker_session_detail_tool(session_id: str, include_shots: bool = False) -> str:
    return _run(
        "session_detail", session_id=session_id, include_shots=include_shots
    )


def bowlingtracker_personal_summary_tool(**arguments) -> str:
    return _run("personal_summary", **arguments)


_SESSION_ANALYSIS_ARGUMENTS = (
    "session_id", "session_name", "session_date",
    "compare_session_id", "compare_session_name", "compare_session_date",
    "question_focus", "game_number", "ball",
)


def bowlingtracker_session_analysis_tool(user_question: str | None = None, **arguments) -> str:
    """Run session analysis. ``user_question`` is the runtime's copy of the user's own
    message (never a model argument); the adapter uses it only for exact session-name/date
    references and question focus."""
    return _run(
        "session_analysis",
        user_question=user_question if isinstance(user_question, str) else None,
        **{key: arguments[key] for key in _SESSION_ANALYSIS_ARGUMENTS if key in arguments},
    )


FILTER_PROPERTIES = {
    "date_from": {
        "type": "string",
        "description": "Inclusive ISO date (YYYY-MM-DD) or timestamp lower bound.",
    },
    "date_to": {
        "type": "string",
        "description": "Inclusive ISO date (YYYY-MM-DD) or timestamp upper bound.",
    },
    "bowling_center": {
        "type": "string",
        "description": "Case-insensitive bowling-center name substring.",
        "maxLength": 200,
    },
    "session_type": {
        "type": "string",
        "enum": ["Tournament", "Sport Shot League", "League", "Practice"],
        "description": "Exact normalized BowlingTracker session category.",
    },
    "pattern": {
        "type": "string",
        "description": "Case-insensitive pattern-name substring.",
        "maxLength": 200,
    },
}


BOWLINGTRACKER_RECENT_SESSIONS_SCHEMA = {
    "name": "bowlingtracker_recent_sessions",
    "description": (
        "Read the user's recent synchronized bowling sessions from BowlingTracker. "
        "Use for personal-history questions such as recent tournaments, Sport Shot "
        "sessions, centers, and patterns. Results are personal observations, not "
        "expert Bowling Knowledge. Read-only; no sync or write capability. Lists "
        "sessions only; for how a session went, which shot/game/ball, or which session "
        "was better, use bowlingtracker_session_analysis."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **FILTER_PROPERTIES,
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
                "description": "Maximum sessions to return, newest first.",
            },
        },
    },
}


BOWLINGTRACKER_SESSION_DETAIL_SCHEMA = {
    "name": "bowlingtracker_session_detail",
    "description": (
        "Read one BowlingTracker session with deterministic game/score, ball-use, "
        "execution, and frame summaries. Set include_shots only when exact recorded "
        "deliveries are needed. Personal measured/observed data only; never expert "
        "claims. Read-only; no sync or write capability. Requires a known UUID; for "
        "how a session went, comparisons, or Witsee delivery questions prefer "
        "bowlingtracker_session_analysis."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "format": "uuid",
                "description": "BowlingTracker session UUID returned by a session query.",
            },
            "include_shots": {
                "type": "boolean",
                "default": False,
                "description": "Include bounded, note-free shot records and typed evidence.",
            },
        },
        "required": ["session_id"],
    },
}


BOWLINGTRACKER_PERSONAL_SUMMARY_SCHEMA = {
    "name": "bowlingtracker_personal_summary",
    "description": (
        "Calculate a deterministic summary from the user's BowlingTracker records: "
        "recorded-score average/high/low, recent games, patterns, most recent ball "
        "use, and ball-use counts. Use for personal results; do not use Bowling "
        "Knowledge or model arithmetic as a substitute. For 'last session', set "
        "session_limit to 1. Read-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **FILTER_PROPERTIES,
            "ball": {
                "type": "string",
                "description": "Case-insensitive BowlingTracker ball-name substring.",
                "maxLength": 200,
            },
            "game_limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
                "description": "Most recent games included in deterministic calculations.",
            },
            "session_limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 20,
                "description": (
                    "Newest matching sessions eligible for the summary. Set to 1 "
                    "when the user asks about their last/latest matching session."
                ),
            },
        },
    },
}


_SESSION_REFERENCE_DESCRIPTION = (
    "Copy exactly what the user said. Use any of: UUID (only if already known; never "
    "invent one), exact session name, date YYYY-MM-DD. Supplied fields must all match."
)

BOWLINGTRACKER_SESSION_ANALYSIS_SCHEMA = {
    "name": "bowlingtracker_session_analysis",
    "description": (
        "PREFERRED tool for questions about one of the user's bowling sessions or for "
        "comparing two sessions: how a session went, what changed during it, Witsee "
        "delivery changes (speed, rev rate, axis, loft, skid/hook), personal-baseline "
        "comparisons, score plus delivery discussion, and 'why did I score lower?', "
        "'which shot', 'which game', 'was this ball better', or 'which session was "
        "better' questions. Identify sessions by name and/or date as the user said them "
        "(the tool resolves them); if the user named none, call it with no session "
        "fields and it will list recent sessions. For a comparison, put the first-mentioned session in "
        "session_* and the other in compare_session_* (differences are second minus "
        "first). Returns a verified read-only summary of personal BowlingTracker records "
        "and Witsee measurements, not expert knowledge. It never states causes, "
        "shot/frame/game/ball matching for Witsee throws, which change mattered most, or "
        "better/worse judgments."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "format": "uuid",
                "description": "First session UUID. " + _SESSION_REFERENCE_DESCRIPTION,
            },
            "session_name": {
                "type": "string",
                "maxLength": 200,
                "description": "First session's exact name, e.g. 'Practice'.",
            },
            "session_date": {
                "type": "string",
                "description": "First session's date as YYYY-MM-DD.",
            },
            "compare_session_id": {
                "type": "string",
                "format": "uuid",
                "description": "Second session UUID, only for comparisons.",
            },
            "compare_session_name": {
                "type": "string",
                "maxLength": 200,
                "description": "Second session's exact name, only for comparisons.",
            },
            "compare_session_date": {
                "type": "string",
                "description": "Second session's date as YYYY-MM-DD, only for comparisons.",
            },
            "question_focus": {
                "type": "string",
                "enum": [
                    "overview", "cause", "shot_or_frame", "game", "ball",
                    "better_or_worse", "baseline",
                ],
                "default": "overview",
                "description": (
                    "cause = why-questions; shot_or_frame = which shot/frame; game = a game "
                    "number (also set game_number); ball = a bowling ball (also set ball); "
                    "better_or_worse = which was better; baseline = baseline questions; "
                    "overview = anything else."
                ),
            },
            "game_number": {
                "type": "integer",
                "minimum": 1,
                "description": "Game number the user asked about, with question_focus=game.",
            },
            "ball": {
                "type": "string",
                "maxLength": 200,
                "description": "Ball name the user asked about, with question_focus=ball.",
            },
        },
    },
}


registry.register(
    name="bowlingtracker_recent_sessions",
    toolset="bowlingtracker",
    schema=BOWLINGTRACKER_RECENT_SESSIONS_SCHEMA,
    handler=lambda args, **kw: bowlingtracker_recent_sessions_tool(**args),
    emoji="🎳",
)
registry.register(
    name="bowlingtracker_session_detail",
    toolset="bowlingtracker",
    schema=BOWLINGTRACKER_SESSION_DETAIL_SCHEMA,
    handler=lambda args, **kw: bowlingtracker_session_detail_tool(
        session_id=args.get("session_id", ""),
        include_shots=args.get("include_shots", False),
    ),
    emoji="🎳",
)
registry.register(
    name="bowlingtracker_personal_summary",
    toolset="bowlingtracker",
    schema=BOWLINGTRACKER_PERSONAL_SUMMARY_SCHEMA,
    handler=lambda args, **kw: bowlingtracker_personal_summary_tool(**args),
    emoji="🎳",
)
registry.register(
    name="bowlingtracker_session_analysis",
    toolset="bowlingtracker",
    schema=BOWLINGTRACKER_SESSION_ANALYSIS_SCHEMA,
    handler=lambda args, **kw: bowlingtracker_session_analysis_tool(
        user_question=kw.get("user_task"),
        **{key: value for key, value in args.items() if key != "user_question"},
    ),
    emoji="🎳",
)
