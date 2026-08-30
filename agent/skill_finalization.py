"""Deterministic, skill-declared final-response finalization.

The runtime owns only activation and a JSON/stdin subprocess protocol.  Domain
schemas and rules remain in the declaring skill.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()
_BY_TASK: dict[str, list[dict[str, Any]]] = {}


def register_skill_finalizer(task_id: str | None, skill_result: dict[str, Any]) -> None:
    if not task_id or not isinstance(skill_result, dict):
        return
    metadata = skill_result.get("metadata", {})
    hermes = metadata.get("hermes", {}) if isinstance(metadata, dict) else {}
    declaration = hermes.get("finalization") if isinstance(hermes, dict) else None
    source_path = skill_result.get("_source_path")
    if not isinstance(declaration, dict) or not declaration.get("required"):
        return
    script = declaration.get("script")
    if not isinstance(script, str) or not isinstance(source_path, str):
        return
    skill_root = Path(source_path).resolve().parent
    target = (skill_root / script).resolve()
    try:
        target.relative_to(skill_root)
    except ValueError:
        return
    item = {
        "skill": skill_result.get("name"),
        "script": str(target),
        "protocol": declaration.get("protocol", "json_stdio_v1"),
        "timeout_seconds": min(float(declaration.get("timeout_seconds", 2)), 10.0),
        "preparation_required": bool(
            isinstance(hermes.get("preparation"), dict)
            and hermes["preparation"].get("required")
        ),
        "preparation_status": (
            skill_result.get("preparation", {}).get("status")
            if isinstance(skill_result.get("preparation"), dict) else None
        ),
    }
    with _LOCK:
        bucket = _BY_TASK.setdefault(str(task_id), [])
        bucket[:] = [entry for entry in bucket if entry.get("skill") != item["skill"]]
        bucket.append(item)


def clear_skill_finalizers(task_id: str | None = None) -> None:
    with _LOCK:
        if task_id is None:
            _BY_TASK.clear()
        else:
            _BY_TASK.pop(str(task_id), None)


def has_required_finalizer(task_id: str | None) -> bool:
    with _LOCK:
        return bool(_BY_TASK.get(str(task_id))) if task_id else False


def finalize_skill_response(
    task_id: str | None, candidate: str, messages: list[dict[str, Any]]
) -> dict[str, Any]:
    with _LOCK:
        declarations = list(_BY_TASK.get(str(task_id), [])) if task_id else []
    try:
        from agent.skill_lifecycle import get_prepared_synthesis
        lifecycle = get_prepared_synthesis(task_id)
        prepared_context = lifecycle.get("prepared_context", {}) if lifecycle else {}
    except Exception:
        prepared_context = {}
    result = {"content": candidate, "status": "not_required", "violations": []}
    for declaration in declarations:
        if declaration.get("preparation_required") and declaration.get(
            "preparation_status"
        ) not in {"READY", "LIMITED"}:
            return {
                "content": "I can’t safely finalize this recommendation because its required deterministic preparation did not complete.",
                "status": "invalid",
                "violations": [{"code": "preparation_not_completed", "reason": str(declaration.get("preparation_status") or "not run")}],
            }
        script = Path(declaration["script"])
        if not script.is_file():
            return {
                "content": "I can’t safely finalize this pattern-play recommendation because its required local validator is unavailable.",
                "status": "invalid",
                "violations": [{"code": "validator_unavailable", "reason": str(script)}],
            }
        payload = json.dumps({
            "candidate": result["content"], "messages": messages,
            "prepared_context": prepared_context,
        })
        try:
            completed = subprocess.run(
                [sys.executable, str(script)], input=payload, text=True,
                capture_output=True, timeout=declaration["timeout_seconds"], check=False,
            )
            parsed = json.loads(completed.stdout)
            if completed.returncode or not isinstance(parsed, dict) or not isinstance(parsed.get("content"), str):
                raise ValueError(completed.stderr or "invalid validator response")
            result = parsed
        except Exception as exc:
            return {
                "content": "I can’t safely finalize this pattern-play recommendation because deterministic validation failed.",
                "status": "invalid",
                "violations": [{"code": "validator_error", "reason": f"{type(exc).__name__}: {exc}"}],
            }
    return result
