"""Runtime execution of skill-declared, pre-synthesis preparation hooks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_skill_preparation(
    skill_result: dict[str, Any], *, user_task: str, task_id: str | None,
    capability_dependencies: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Run a required local preparation hook and return its structured result.

    The runtime understands only declaration/path safety and JSON stdio. Domain
    extraction, routing, and evidence schemas remain owned by the skill.
    """
    metadata = skill_result.get("metadata", {})
    hermes = metadata.get("hermes", {}) if isinstance(metadata, dict) else {}
    declaration = hermes.get("preparation") if isinstance(hermes, dict) else None
    if not isinstance(declaration, dict):
        return None

    base = {"required": bool(declaration.get("required")), "status": "FAILED"}
    source_path = skill_result.get("_source_path")
    script = declaration.get("script")
    if not isinstance(source_path, str) or not isinstance(script, str):
        return {**base, "error": "Preparation declaration requires a script."}
    skill_root = Path(source_path).resolve().parent
    target = (skill_root / script).resolve()
    try:
        target.relative_to(skill_root)
    except ValueError:
        return {**base, "error": "Preparation script escapes the skill root."}
    if declaration.get("protocol", "json_stdio_v1") != "json_stdio_v1":
        return {**base, "error": "Unsupported preparation protocol."}
    if not target.is_file():
        return {**base, "error": f"Preparation script unavailable: {target}"}

    payload = {
        "protocol": "json_stdio_v1",
        "user_request": user_task if isinstance(user_task, str) else "",
        "task_id": str(task_id or ""),
        "capability_dependencies": capability_dependencies,
        "entity_resolvers": hermes.get("entity_resolvers", []),
    }
    timeout = min(max(float(declaration.get("timeout_seconds", 10)), 0.1), 30.0)
    try:
        completed = subprocess.run(
            [sys.executable, str(target)], input=json.dumps(payload), text=True,
            capture_output=True, timeout=timeout, check=False,
        )
        parsed = json.loads(completed.stdout)
        if completed.returncode or not isinstance(parsed, dict):
            raise ValueError(completed.stderr.strip() or "invalid preparation response")
        status = str(parsed.get("status", "")).upper()
        if status not in {"READY", "LIMITED", "FAILED"}:
            raise ValueError("preparation status must be READY, LIMITED, or FAILED")
        return {"required": base["required"], **parsed, "status": status}
    except Exception as exc:
        return {**base, "error": f"{type(exc).__name__}: {exc}"}
