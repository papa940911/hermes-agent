"""Gate 10N: narrow bridge from the Telegram adapter to the outer-repo
Bowling Knowledge review UX (Gate 10M).

This module holds NO review-domain logic, NO SQL, and NO promotion
calls of its own -- it only lazily loads
skills/bowling-knowledge/scripts/telegram_review_router.py (the exact
same sibling-module-loading convention already used by
tools/bowling_knowledge_tool.py for hermes_tool.py) and forwards calls to
it. See that module for the actual review-domain service, presentation,
session/notification-state, and command-routing logic.

Trust boundary: every public function here assumes the CALLER (the
Telegram adapter, in plugins/platforms/telegram/adapter.py) has already
authorized the caller using its own existing, tested authorization chain
(_is_user_authorized_from_message for plain text,
_is_callback_user_authorized for inline-button callbacks) BEFORE calling
into this module. The ReviewRouter instance built here is therefore
constructed with an always-true authorization check -- that is not a
bypass, it is where the real check already happened, one layer up, using
a more complete mechanism than anything this bridge could re-derive.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

# adapter.py lives at hermes-agent/plugins/platforms/telegram/adapter.py;
# the outer repo root (containing skills/) is four parents up from here.
_OUTER_REPO_ROOT = Path(__file__).resolve().parents[4]
_ROUTER_SCRIPTS_DIR = _OUTER_REPO_ROOT / "skills" / "bowling-knowledge" / "scripts"
_ROUTER_PATH = _ROUTER_SCRIPTS_DIR / "telegram_review_router.py"

_router_module = None
_router_instance = None
_router_signature = None


class BowlingReviewBridgeError(RuntimeError):
    """Raised only when the outer-repo module itself cannot be loaded at
    all (missing outer repo checkout, etc.) -- never for an ordinary
    review-domain refusal, which is always a normal Outbound reply."""


def _dependency_closure_signature() -> tuple[tuple[str, float], ...] | None:
    """Gate 10U.1: a cheap, deterministic staleness key covering every
    ``.py`` file telegram_review_router.py (transitively) loads.

    Root cause of the Gate 10U.1 live routing bug: this bridge previously
    cached ``_router_module``/``_router_instance`` for the entire gateway
    process's lifetime with NO staleness check at all. The outer repo's
    telegram_review_router.py -- and every module it (and, transitively,
    its own further sibling modules) lazily loads via the exact same
    ``_load_sibling`` file-based-import convention -- can be edited and
    committed at any time independently of a gateway restart. Once this
    bridge had loaded a module chain ONE time in a given process, it
    silently kept using that exact snapshot forever, even after an
    outer-repo command-grammar change (e.g. adding a new exact-match
    command) landed on disk. A live command that should have been
    recognized was then silently treated as "not a review command" and
    fell through to full conversational LLM routing -- exactly the
    "promote approved updates" incident this gate fixes.

    Rather than hand-maintain a fragile list of every file in that
    dependency graph (which grows over time -- see historical rule
    metadata support, added the same day as this incident), this hashes the
    (name, mtime) of every ``.py`` file directly in the scripts directory
    telegram_review_router.py lives in. Any future addition/edit to that
    directory is therefore automatically picked up on the next Telegram
    message with no gateway restart required, without this bridge needing
    to know the exact transitive import graph.
    """
    try:
        return tuple(sorted(
            (p.name, p.stat().st_mtime) for p in _ROUTER_SCRIPTS_DIR.glob("*.py")
        ))
    except OSError:
        return None


def _load_router_module():
    """Load skills/bowling-knowledge/scripts/telegram_review_router.py,
    transparently reloading it (and therefore every sibling module it
    lazily loads -- see _dependency_closure_signature's docstring) if
    anything in that directory has changed since the last load."""
    global _router_module, _router_instance, _router_signature
    current_signature = _dependency_closure_signature()
    if current_signature is None:
        if _router_module is None:
            raise BowlingReviewBridgeError(
                f"Bowling Knowledge review router not found at {_ROUTER_PATH}"
            )
        return _router_module
    if _router_module is not None and current_signature == _router_signature:
        return _router_module
    if not _ROUTER_PATH.is_file():
        raise BowlingReviewBridgeError(
            f"Bowling Knowledge review router not found at {_ROUTER_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        "bowling_knowledge_telegram_bridge_router", _ROUTER_PATH,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - environment failure
        raise BowlingReviewBridgeError("Bowling Knowledge review router could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _router_module = module
    _router_signature = current_signature
    # A stale ReviewRouter instance would still be built from the OLD
    # module's class objects (isinstance/class-identity would diverge from
    # the freshly reloaded module) -- force a rebuild against the new one.
    _router_instance = None
    return _router_module


def _always_authorized(user_id: str, chat_id: str) -> bool:
    return True


def _get_router():
    global _router_instance
    # Gate 10U.1: _load_router_module() must run FIRST on every call, not
    # only when _router_instance happens to already be None -- it is the
    # one place that detects an outer-repo change and resets
    # _router_instance to force a rebuild. Checking `is None` before
    # calling it (the previous, buggy order) meant an already-built stale
    # instance was returned forever without ever re-checking staleness.
    router_module = _load_router_module()
    if _router_instance is None:
        _router_instance = router_module.ReviewRouter(is_authorized=_always_authorized)
    return _router_instance


def is_review_command(text: str) -> bool:
    """Cheap, side-effect-free check: does this text deterministically
    parse as one of the recognized review commands? Callers use this to
    decide whether to intercept a message at all -- False means "leave
    this message completely alone."""
    router_module = _load_router_module()
    return router_module.is_review_command(text)


def handle_text(text: str, *, chat_id: str, user_id: str) -> Any:
    """Returns a telegram_review_router.Outbound."""
    router_module = _load_router_module()
    router = _get_router()
    inbound = router_module.Inbound(chat_id=chat_id, user_id=user_id, text=text)
    return router.handle(inbound)


def handle_callback(token: str, action: str, *, chat_id: str, user_id: str) -> Any:
    """Returns a telegram_review_router.Outbound."""
    router_module = _load_router_module()
    router = _get_router()
    inbound = router_module.Inbound(chat_id=chat_id, user_id=user_id, callback_token=token, callback_action=action)
    return router.handle(inbound)


def preview_new_pending_notification(chat_id: str) -> tuple[str | None, list[str]]:
    """Read-only: computes what a batch notification WOULD say for
    whatever pending items have never been notified yet, without marking
    anything as notified. Returns (text_or_None, [review_item_id, ...]).
    text is None when there is nothing new to notify about -- callers
    must never send a message for a None result (see the "do not spam"
    requirement). Marking only happens in confirm_notification_sent()
    below, and only after a real Telegram send has actually succeeded
    (see the Gate 10N "do not mark notified unless send succeeds"
    requirement) -- this function alone never mutates notification state."""
    router_module = _load_router_module()
    router = _get_router()
    items = router_module.service.list_pending(router.review_db_path)
    unnotified_ids = set(router.session_store.filter_unnotified([i.review_item_id for i in items]))
    fresh = [i for i in items if i.review_item_id in unnotified_ids]
    text = router_module.presentation.format_notification(fresh)
    return text, [i.review_item_id for i in fresh]


def confirm_notification_sent(review_item_ids: list[str], chat_id: str) -> None:
    """Call this ONLY after a Telegram send has actually succeeded. A
    failed send must leave every one of these review items eligible for
    retry on the next cycle -- simply never calling this function is
    sufficient for that; it performs no compensating action of its own,
    it only ever marks forward."""
    router_module = _load_router_module()
    router = _get_router()
    for review_item_id in review_item_ids:
        item = router_module.service.get_item(router.review_db_path, review_item_id)
        if item is not None:
            router.session_store.mark_notified(review_item_id, item.candidate_fingerprint, chat_id)
