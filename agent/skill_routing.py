"""Deterministic, skill-declared routing before the first provider call."""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from agent.skill_utils import (
    get_all_skills_dirs,
    get_disabled_skill_names,
    get_project_skills_dirs,
    iter_project_skill_files,
    iter_skill_index_files,
    parse_frontmatter,
    skill_matches_environment,
    skill_matches_platform,
)
from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_CACHE_LOCK = threading.Lock()
_ROUTING_CACHE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
_SNAPSHOT_VERSION = 1


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_WORD_RE.findall(value.casefold()))


def _routing_declaration(frontmatter: dict[str, Any]) -> dict[str, Any] | None:
    metadata = frontmatter.get("metadata")
    hermes = metadata.get("hermes") if isinstance(metadata, dict) else None
    routing = hermes.get("routing") if isinstance(hermes, dict) else None
    if not isinstance(routing, dict) or routing.get("role") != "owner":
        return None
    match = routing.get("match")
    groups = match.get("all") if isinstance(match, dict) else None
    if not isinstance(groups, list) or len(groups) < 2:
        return None
    normalized_groups = []
    for group in groups:
        if not isinstance(group, dict):
            return None
        phrases = group.get("any_phrases")
        if not isinstance(phrases, list):
            return None
        normalized = [
            _tokens(phrase) for phrase in phrases
            if isinstance(phrase, str) and _tokens(phrase)
        ]
        if not normalized:
            return None
        normalized_groups.append({
            "concept": str(group.get("concept") or ""),
            "phrases": normalized,
        })
    priority = routing.get("priority", 0)
    if not isinstance(priority, int) or isinstance(priority, bool):
        return None
    return {"priority": priority, "groups": normalized_groups}


def _index_key(files: list[Path], disabled: set[str]) -> tuple[Any, ...]:
    fingerprints = []
    for path in files:
        try:
            stat = path.stat()
            fingerprints.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            continue
    return (tuple(fingerprints), tuple(sorted(disabled)))


def _snapshot_path() -> Path:
    return get_hermes_home() / ".skill_routing_snapshot.json"


def clear_routing_index_cache(*, clear_snapshot: bool = False) -> None:
    with _CACHE_LOCK:
        _ROUTING_CACHE.clear()
    if clear_snapshot:
        try:
            _snapshot_path().unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove skill routing snapshot", exc_info=True)


def build_routing_index() -> list[dict[str, Any]]:
    """Return compact owner declarations; skill bodies are not retained."""
    project_files = [
        path
        for root in get_project_skills_dirs()
        for path in iter_project_skill_files(root)
    ]
    installed_files = [
        path
        for root in get_all_skills_dirs()
        if root.exists()
        for path in iter_skill_index_files(root, "SKILL.md")
    ]
    files = project_files + installed_files
    disabled = get_disabled_skill_names()
    key = _index_key(files, disabled)
    with _CACHE_LOCK:
        cached = _ROUTING_CACHE.get(key)
        if cached is not None:
            return [dict(item) for item in cached]
    try:
        snapshot = json.loads(_snapshot_path().read_text(encoding="utf-8"))
        if (
            snapshot.get("version") == _SNAPSHOT_VERSION
            and snapshot.get("fingerprints") == [list(item) for item in key[0]]
            and snapshot.get("disabled") == list(key[1])
            and isinstance(snapshot.get("entries"), list)
        ):
            entries = snapshot["entries"]
            with _CACHE_LOCK:
                _ROUTING_CACHE[key] = entries
            return [dict(item) for item in entries]
    except Exception:
        pass

    entries: list[dict[str, Any]] = []
    claimed: set[str] = set()
    for path in files:
        try:
            frontmatter, _ = parse_frontmatter(path.read_text(encoding="utf-8-sig"))
            name = str(frontmatter.get("name") or path.parent.name)
            if name in claimed or name in disabled:
                continue
            if not skill_matches_platform(frontmatter) or not skill_matches_environment(frontmatter):
                continue
            declaration = _routing_declaration(frontmatter)
            claimed.add(name)
            if declaration is not None:
                entries.append({"skill": name, **declaration})
        except Exception:
            logger.debug("Could not index skill routing metadata from %s", path, exc_info=True)

    with _CACHE_LOCK:
        _ROUTING_CACHE.clear()
        _ROUTING_CACHE[key] = entries
    try:
        atomic_json_write(_snapshot_path(), {
            "version": _SNAPSHOT_VERSION,
            "fingerprints": [list(item) for item in key[0]],
            "disabled": list(key[1]),
            "entries": entries,
        })
    except Exception:
        logger.debug("Could not persist skill routing snapshot", exc_info=True)
    return [dict(item) for item in entries]


def _phrase_present(request_tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    phrase = tuple(phrase)
    width = len(phrase)
    return any(request_tokens[index:index + width] == phrase for index in range(len(request_tokens) - width + 1))


def route_request(user_request: Any, index: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Select one strong owner, or return a deterministic fallback result."""
    if not isinstance(user_request, str) or not user_request.strip():
        return {"route_status": "no_match", "route_reason": "request_not_text"}
    request_tokens = _tokens(user_request)
    matches = []
    for entry in index if index is not None else build_routing_index():
        matched_groups = []
        score = 0
        for group in entry.get("groups", []):
            hits = [phrase for phrase in group["phrases"] if _phrase_present(request_tokens, phrase)]
            if not hits:
                break
            best = max(hits, key=lambda phrase: (len(phrase), phrase))
            matched_groups.append(group.get("concept") or str(len(matched_groups) + 1))
            score += len(best)
        else:
            # Multiple required groups plus at least three lexical tokens keeps
            # a declaration from routing on one broad word.
            if len(matched_groups) >= 2 and score >= 3:
                matches.append({
                    "route_skill": entry["skill"],
                    "route_priority": entry.get("priority", 0),
                    "route_score": score,
                    "matched_groups": matched_groups,
                })
    if not matches:
        return {"route_status": "no_match", "route_reason": "no_strong_owner"}
    matches.sort(key=lambda item: (-item["route_priority"], -item["route_score"], item["route_skill"]))
    winner = matches[0]
    tied = [
        item for item in matches
        if (item["route_priority"], item["route_score"])
        == (winner["route_priority"], winner["route_score"])
    ]
    if len(tied) != 1:
        return {
            "route_status": "ambiguous",
            "route_reason": "owner_tie",
            "route_candidates": [item["route_skill"] for item in tied],
        }
    return {"route_status": "selected", "route_reason": "unique_strong_owner", **winner}


def activate_routed_skill(user_request: Any, task_id: str | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Route and execute the existing skill_view handler without a model call."""
    route = route_request(user_request)
    if route.get("route_status") != "selected":
        return route, None
    try:
        from tools.skills_tool import _skill_view_with_bump

        raw = _skill_view_with_bump(
            {"name": route["route_skill"]},
            task_id=task_id,
            user_task=user_request,
        )
        result = json.loads(raw)
    except Exception as exc:
        route.update(route_status="activation_failed", route_reason=f"{type(exc).__name__}: {exc}")
        return route, None
    if not isinstance(result, dict) or not result.get("success"):
        route.update(route_status="activation_failed", route_reason=str(result.get("error") if isinstance(result, dict) else "invalid_skill_view_result"))
        return route, result if isinstance(result, dict) else None
    preparation = result.get("preparation")
    route["preparation_status"] = preparation.get("status") if isinstance(preparation, dict) else None
    return route, result


def install_routed_context(api_messages: list[dict[str, Any]], payload: str) -> bool:
    """Append activation evidence to the current user row on the request copy."""
    for message in reversed(api_messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        notice = "[Deterministic skill activation]\n" + payload
        if isinstance(content, str):
            message["content"] = content + "\n\n" + notice
        elif isinstance(content, list):
            message["content"] = [*content, {"type": "text", "text": "\n\n" + notice}]
        else:
            return False
        return True
    return False
