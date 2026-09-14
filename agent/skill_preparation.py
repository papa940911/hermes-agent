"""Runtime execution of skill-declared, pre-synthesis preparation hooks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SKILL_CONTEXT_METADATA_KEY = "skill_context"
MAX_SURFACE_CONTEXT_BYTES = 64 * 1024
_RESERVED_SURFACE_CONTEXT_KEYS = frozenset({
    "protocol",
    "user_request",
    "task_id",
    "capability_dependencies",
    "entity_resolvers",
})


class SurfaceContextValidationError(ValueError):
    """An explicit request-scoped skill context is not safe to forward."""


def detach_surface_context(value: Any) -> dict[str, Any] | None:
    """Validate and detach one JSON-compatible request-scoped context object."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise SurfaceContextValidationError("surface context must be an object")

    def _check_keys(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise SurfaceContextValidationError(
                        "surface context object keys must be strings"
                    )
                if key.startswith("_"):
                    raise SurfaceContextValidationError(
                        "surface context keys beginning with '_' are reserved"
                    )
                _check_keys(child)
        elif isinstance(item, list):
            for child in item:
                _check_keys(child)

    _check_keys(value)
    if _RESERVED_SURFACE_CONTEXT_KEYS.intersection(value):
        raise SurfaceContextValidationError(
            "surface context contains a reserved preparation key"
        )
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SurfaceContextValidationError(
            "surface context must contain only JSON-compatible values"
        ) from exc
    if len(encoded) > MAX_SURFACE_CONTEXT_BYTES:
        raise SurfaceContextValidationError(
            f"surface context exceeds {MAX_SURFACE_CONTEXT_BYTES} bytes"
        )
    detached = json.loads(encoded.decode("utf-8"))
    if not isinstance(detached, dict):  # pragma: no cover - guarded above
        raise SurfaceContextValidationError("surface context must be an object")
    return detached


def surface_context_from_event_metadata(metadata: Any) -> dict[str, Any] | None:
    """Copy only the dedicated allowlisted event metadata key.

    Invalid values are dropped at the gateway boundary. They never become
    skill input and unrelated event metadata is never exposed.
    """

    if not isinstance(metadata, dict) or SKILL_CONTEXT_METADATA_KEY not in metadata:
        return None
    try:
        return detach_surface_context(metadata[SKILL_CONTEXT_METADATA_KEY])
    except SurfaceContextValidationError:
        return None


def run_skill_preparation(
    skill_result: dict[str, Any], *, user_task: str, task_id: str | None,
    capability_dependencies: list[dict[str, Any]],
    surface_context: Any = None,
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

    try:
        detached_surface_context = detach_surface_context(surface_context)
    except SurfaceContextValidationError as exc:
        return {**base, "error": f"SurfaceContextValidationError: {exc}"}

    payload = {
        "protocol": "json_stdio_v1",
        "user_request": user_task if isinstance(user_task, str) else "",
        "task_id": str(task_id or ""),
        "capability_dependencies": capability_dependencies,
        "entity_resolvers": hermes.get("entity_resolvers", []),
    }
    if detached_surface_context is not None:
        payload["surface_context"] = detached_surface_context
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
