"""Task-scoped handoff from deterministic skill preparation to synthesis."""

from __future__ import annotations

import json
import hashlib
import re
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()
_BY_TASK: dict[str, dict[str, Any]] = {}
_MAX_SYNTHESIS_CONTRACT_BYTES = 16_384


def _contract_type_schema(description: Any) -> dict[str, Any]:
    """Derive a structural JSON Schema leaf from the skill contract DSL."""
    if isinstance(description, dict):
        return _contract_object_schema(description)
    text = str(description or "").strip()
    lower = text.lower()
    if "boolean" in lower or lower == "true":
        return {"type": "boolean"}
    if lower.startswith("number"):
        return {"type": ["number", "string"]} if "range" in lower else {"type": "number"}
    if lower.startswith("array") or " array" in lower:
        return {"type": "array"}
    # Pipe-delimited leading tokens in contracts are closed vocabularies.
    enum_text = text.split(" ", 1)[0]
    enum_values = enum_text.split("|")
    if len(enum_values) > 1 and all(re.fullmatch(r"[A-Za-z0-9_.-]+", v) for v in enum_values):
        return {"type": "string", "enum": enum_values}
    return {"type": "string"}


def _contract_object_schema(item: dict[str, Any]) -> dict[str, Any]:
    shape = item.get("shape", item)
    if not isinstance(shape, dict):
        return {"type": "object"}
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {key: _contract_type_schema(value) for key, value in shape.items()},
        "additionalProperties": False,
    }
    required = item.get("required")
    if isinstance(required, list):
        schema["required"] = [value for value in required if isinstance(value, str)]
    return schema


def provider_schema_from_contract(contract: dict[str, Any]) -> dict[str, Any] | None:
    """Deterministically derive grammar-safe structural schema from one contract."""
    required_top = contract.get("provider_required_top_level", contract.get("required_top_level"))
    items = contract.get("provider_items", contract.get("items"))
    if not isinstance(required_top, dict) or not isinstance(items, dict):
        return None
    properties: dict[str, Any] = {}
    for key, description in required_top.items():
        if key == "schema" and isinstance(contract.get("schema"), str):
            properties[key] = {"type": "string", "enum": [contract["schema"]]}
        elif key in items:
            item_schema = (_contract_object_schema(items[key])
                           if isinstance(items[key], dict)
                           else _contract_type_schema(items[key]))
            if "array" in str(description).lower():
                properties[key] = {"type": "array", "items": item_schema}
                maximum = re.search(r"maxItems=(\d+)", str(description))
                if maximum:
                    properties[key]["maxItems"] = int(maximum.group(1))
            else:
                properties[key] = item_schema
        else:
            properties[key] = _contract_type_schema(description)
    return {
        "type": "object",
        "properties": properties,
        "required": list(required_top),
        "additionalProperties": False,
    }


def structured_output_extra_body(state: dict[str, Any] | None) -> dict[str, Any] | None:
    structured = state.get("structured_output") if isinstance(state, dict) else None
    if not isinstance(structured, dict) or structured.get("mode") != "json_schema":
        return None
    schema = structured.get("schema")
    if not isinstance(schema, dict):
        return None
    return {"response_format": {
        "type": "json_schema",
        "json_schema": {"name": structured["schema_name"], "strict": True, "schema": deepcopy(schema)},
    }}


def mark_structured_output_status(task_id: str | None, *, sent: bool | None = None,
                                  provider_rejected: bool | None = None) -> None:
    if not task_id:
        return
    with _LOCK:
        state = _BY_TASK.get(str(task_id))
        structured = state.get("structured_output") if isinstance(state, dict) else None
        if isinstance(structured, dict):
            if sent is not None:
                structured["sent"] = sent
            if provider_rejected is not None:
                structured["provider_rejected"] = provider_rejected


def _load_synthesis_contract(
    skill_result: dict[str, Any], finalization: dict[str, Any]
) -> dict[str, Any] | None:
    """Load a bounded, skill-owned contract without interpreting its schema."""
    reference = finalization.get("synthesis_contract")
    source_path = skill_result.get("_source_path")
    if not isinstance(reference, str) or not reference.strip() or not isinstance(source_path, str):
        return None
    skill_root = Path(source_path).resolve().parent
    target = (skill_root / reference).resolve()
    try:
        target.relative_to(skill_root)
        raw = target.read_bytes()
        if len(raw) > _MAX_SYNTHESIS_CONTRACT_BYTES:
            return None
        contract = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    return contract if isinstance(contract, dict) else None


def _provider_contract_view(contract: dict[str, Any] | None) -> dict[str, Any] | None:
    """Hide legacy/raw compatibility fields when a compact provider DSL exists."""
    if not isinstance(contract, dict) or not isinstance(contract.get("provider_required_top_level"), dict):
        return contract
    return {
        "schema": contract.get("schema"),
        "output": contract.get("output"),
        "instruction": "Select supplied choice refs only; do not construct evidence, coordinates, equipment identity, provenance, status, or prose.",
        "required_top_level": contract["provider_required_top_level"],
        "items": contract.get("provider_items", {}),
    }


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
    provider_context = result.get("provider_context")
    contract = _provider_contract_view(_load_synthesis_contract(skill_result, finalization))
    structured_declaration = finalization.get("structured_output")
    structured_output = None
    if isinstance(structured_declaration, dict) and structured_declaration.get("required"):
        mode = structured_declaration.get("mode")
        schema = provider_schema_from_contract(contract) if mode == "json_schema" and contract else None
        structured_output = {
            "required": True, "mode": mode, "schema": None,
            "schema_name": None, "schema_sha256": None, "schema_bytes": 0,
            "sent": False, "provider_rejected": False,
            "unavailable": schema is None,
        }
        if schema is not None:
            schema_bytes = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
            raw_name = str(contract.get("schema") or skill_result.get("name") or "skill_output")
            structured_output.update({
                "schema": schema,
                "schema_name": re.sub(r"[^A-Za-z0-9_-]+", "_", raw_name)[:64],
                "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
                "schema_bytes": len(schema_bytes), "unavailable": False,
            })
    handoff = {
        "phase": "constrained_synthesis",
        "skill": skill_result.get("name"),
        "user_task": user_task if isinstance(user_task, str) else "",
        "prepared_context": deepcopy(context) if isinstance(context, dict) else {},
        "provider_context": (
            deepcopy(provider_context)
            if isinstance(provider_context, dict)
            else deepcopy(context) if isinstance(context, dict) else {}
        ),
        "synthesis_contract": contract,
        "structured_output": structured_output,
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
            "synthesis_instruction": (
                "Synthesize from only the supplied provider_context. Follow the immediately "
                "adjacent synthesis_contract exactly. Omit unsupported recommendations or "
                "use its limitation/fallback mechanism. Final prose is rendered "
                "deterministically after validation; no further discovery is available."
            ),
            "synthesis_contract": state.get("synthesis_contract"),
            "provider_context": state.get("provider_context", {}),
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
