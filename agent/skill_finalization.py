"""Deterministic, skill-declared final-response finalization.

The runtime owns only activation and a JSON/stdin subprocess protocol.  Domain
schemas and rules remain in the declaring skill.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()
_BY_TASK: dict[str, list[dict[str, Any]]] = {}
_DIAGNOSTICS_DIR_ENV = "HERMES_INTERNAL_FINALIZATION_DIAGNOSTICS_DIR"


def _diagnostic_directory(task_id: str) -> Path | None:
    """Return the opt-in, task-scoped diagnostic directory."""
    root = os.environ.get(_DIAGNOSTICS_DIR_ENV)
    if not root:
        return None
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._")[:64] or "task"
    suffix = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    # PID separates concurrent eval/CLI processes even if an external caller
    # accidentally reuses a task identifier.
    return Path(root) / f"{label}-{suffix}-{os.getpid()}"


def _write_private(path: Path, data: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
        handle.write(data)


def _persist_diagnostics(
    directory: Path | None,
    *,
    candidate: str | None = None,
    result: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Best-effort debug artifacts which can never affect finalization."""
    if directory is None:
        return
    try:
        if candidate is not None:
            _write_private(directory / "pre_finalization_candidate.txt", candidate)
        if result is not None:
            _write_private(
                directory / "finalization_result.json",
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            )
        if metadata is not None:
            _write_private(
                directory / "finalization_metadata.json",
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            )
    except Exception:
        # Diagnostics are strictly observational and must not change the safe
        # production or fail-closed finalization path.
        pass


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
    task_id: str | None,
    candidate: str,
    messages: list[dict[str, Any]],
    *,
    finish_reason: str | None = None,
    provider_transport_streaming: bool | None = None,
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
    diagnostic_dir = _diagnostic_directory(str(task_id)) if task_id and declarations else None
    if diagnostic_dir is not None:
        candidate_bytes = candidate.encode("utf-8")
        diagnostic_metadata = {
            "candidate_character_length": len(candidate),
            "candidate_utf8_byte_length": len(candidate_bytes),
            "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "finish_reason": finish_reason,
            "provider_transport_streaming": provider_transport_streaming,
            "required_finalization_active": bool(declarations),
            "task_scope": str(task_id),
            "finalizers": [
                {
                    "skill": declaration.get("skill"),
                    "script": declaration.get("script"),
                    "protocol": declaration.get("protocol"),
                }
                for declaration in declarations
            ],
        }
        if lifecycle and isinstance(lifecycle.get("structured_output"), dict):
            structured = lifecycle["structured_output"]
            diagnostic_metadata.update({
                "structured_output_required": bool(structured.get("required")),
                "structured_output_mode": structured.get("mode"),
                "structured_output_schema_name": structured.get("schema_name"),
                "structured_output_schema_sha256": structured.get("schema_sha256"),
                "structured_output_sent": bool(structured.get("sent")),
                "structured_output_provider_rejected": bool(structured.get("provider_rejected")),
                "structured_output_unavailable": bool(structured.get("unavailable")),
            })
        # This is the exact string which becomes the first validator's
        # candidate; do not normalize or reserialize it before this copy.
        _persist_diagnostics(
            diagnostic_dir, candidate=candidate, metadata=diagnostic_metadata
        )

    def finish(value: dict[str, Any]) -> dict[str, Any]:
        _persist_diagnostics(diagnostic_dir, result=value)
        return value

    for declaration in declarations:
        if declaration.get("preparation_required") and declaration.get(
            "preparation_status"
        ) not in {"READY", "LIMITED"}:
            return finish({
                "content": "I can’t safely finalize this recommendation because its required deterministic preparation did not complete.",
                "status": "invalid",
                "violations": [{"code": "preparation_not_completed", "reason": str(declaration.get("preparation_status") or "not run")}],
            })
        script = Path(declaration["script"])
        if not script.is_file():
            return finish({
                "content": "I can’t safely finalize this pattern-play recommendation because its required local validator is unavailable.",
                "status": "invalid",
                "violations": [{"code": "validator_unavailable", "reason": str(script)}],
            })
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
            return finish({
                "content": "I can’t safely finalize this pattern-play recommendation because deterministic validation failed.",
                "status": "invalid",
                "violations": [{"code": "validator_error", "reason": f"{type(exc).__name__}: {exc}"}],
            })
    return finish(result)
