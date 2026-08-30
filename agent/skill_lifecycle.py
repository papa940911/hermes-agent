"""Task-scoped handoff from deterministic skill preparation to synthesis."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from typing import Any


_LOCK = threading.Lock()
_BY_TASK: dict[str, dict[str, Any]] = {}


def register_prepared_synthesis(task_id: str | None, skill_result: dict[str, Any], user_task: str) -> None:
    """Arm one constrained model turn when a skill explicitly declares it."""
    if not task_id or not isinstance(skill_result, dict):
        return
    metadata = skill_result.get("metadata", {})
    hermes = metadata.get("hermes", {}) if isinstance(metadata, dict) else {}
    preparation = hermes.get("preparation") if isinstance(hermes, dict) else None
    finalization = hermes.get("finalization") if isinstance(hermes, dict) else None
    result = skill_result.get("preparation")
    if not (
        isinstance(preparation, dict)
        and preparation.get("required")
        and preparation.get("handoff") == "constrained_synthesis"
        and isinstance(finalization, dict)
        and finalization.get("required")
        and isinstance(result, dict)
        and result.get("status") == "READY"
    ):
        return
    context = result.get("context")
    handoff = {
        "phase": "constrained_synthesis",
        "skill": skill_result.get("name"),
        "user_task": user_task if isinstance(user_task, str) else "",
        "prepared_context": deepcopy(context) if isinstance(context, dict) else {},
    }
    with _LOCK:
        _BY_TASK[str(task_id)] = handoff


def get_prepared_synthesis(task_id: str | None, user_task: str | None = None) -> dict[str, Any] | None:
    if not task_id:
        return None
    with _LOCK:
        state = _BY_TASK.get(str(task_id))
        if state is None:
            return None
        if user_task is not None and state.get("user_task") != user_task:
            _BY_TASK.pop(str(task_id), None)
            return None
        return deepcopy(state)


def clear_prepared_synthesis(task_id: str | None = None) -> None:
    with _LOCK:
        if task_id is None:
            _BY_TASK.clear()
        else:
            _BY_TASK.pop(str(task_id), None)


def synthesis_handoff_text(state: dict[str, Any]) -> str:
    """Compact authoritative evidence used in place of a spilled skill result."""
    return json.dumps(
        {
            "lifecycle": "PREPARATION READY — CONSTRAINED SYNTHESIS",
            "skill": state.get("skill"),
            "instruction": (
                "Use the prepared_context as authoritative runtime evidence and emit "
                "only the structured manifest required by this skill's finalizer. "
                "No further discovery is available in this phase."
            ),
            "prepared_context": state.get("prepared_context", {}),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def tools_for_synthesis(state: dict[str, Any] | None, ordinary_tools: list[Any] | None) -> list[Any] | None:
    """Return the request-local tool surface for the current lifecycle phase."""
    return [] if state else ordinary_tools


def install_synthesis_handoff(api_messages: list[dict[str, Any]], state: dict[str, Any]) -> bool:
    """Replace the owning skill_view result on the request copy, preserving roles."""
    skill = str(state.get("skill") or "")
    for message in reversed(api_messages):
        if message.get("role") != "tool" or message.get("name") != "skill_view":
            continue
        content = message.get("content")
        matches_owner = False
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                matches_owner = str(parsed.get("name") or "") == skill
            except (json.JSONDecodeError, AttributeError):
                # Oversized results retain a leading JSON preview, including
                # skill_view's canonical name field.
                matches_owner = (
                    f'"name": "{skill}"' in content
                    or f'"name":"{skill}"' in content
                )
        if matches_owner:
            message["content"] = synthesis_handoff_text(state)
            return True
    return False
