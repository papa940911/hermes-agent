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
        "expert Bowling Knowledge. Read-only; no sync or write capability."
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
        "claims. Read-only; no sync or write capability."
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
