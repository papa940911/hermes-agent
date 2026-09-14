"""Telegram's explicit, request-scoped bowling-session context producer."""

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.skill_preparation import (
    MAX_SURFACE_CONTEXT_BYTES,
    surface_context_from_event_metadata,
)
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    apply_explicit_bowling_session_context,
)


EXPECTED_CONTEXT = {
    "schema": "bowling_surface_context/v1",
    "contextType": "completed_session",
    "source": "surface_event",
    "provenance": "explicit_surface_selection",
    "payload": {"sessionID": "session-123"},
}


def _make_adapter() -> TelegramAdapter:
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra={})
    adapter._bot = SimpleNamespace(id=999, username="test_bot")
    adapter._message_handler = AsyncMock()
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = 0.01
    adapter._text_batch_split_delay_seconds = 0.05
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._forum_lock = asyncio.Lock()
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter.handle_message = AsyncMock()
    return adapter


def _make_message(text: str, *, message_id: int = 42):
    return SimpleNamespace(
        message_id=message_id,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(id=12345, type="private", title=None, is_forum=False),
        from_user=SimpleNamespace(id=111, full_name="Test User", first_name="Test"),
        reply_to_message=None,
        date=None,
        location=None,
        photo=None,
        video=None,
        audio=None,
        voice=None,
        document=None,
        sticker=None,
        media_group_id=None,
    )


def _make_update(text: str, *, update_id: int = 1):
    return SimpleNamespace(
        update_id=update_id,
        message=_make_message(text, message_id=update_id),
        effective_message=None,
    )


def test_explicit_completed_session_command_emits_exact_context():
    event = MessageEvent(
        text="/bowlingsession session-123 Review this session.",
        message_type=MessageType.COMMAND,
        metadata={"telegram_internal": "retained-only-as-event-metadata"},
    )

    assert apply_explicit_bowling_session_context(event) is True
    assert event.text == "Review this session."
    assert event.message_type == MessageType.TEXT
    assert event.metadata["skill_context"] == EXPECTED_CONTEXT
    assert "telegram_internal" not in event.metadata["skill_context"]
    assert surface_context_from_event_metadata(event.metadata) == EXPECTED_CONTEXT


@pytest.mark.parametrize(
    "question",
    ["Review this session.", "Why did I struggle?", "What should I work on?"],
)
@pytest.mark.asyncio
async def test_real_command_handler_dispatches_same_request_context(question):
    adapter = _make_adapter()

    await adapter._handle_command(
        _make_update(f"/bowlingsession@test_bot session-123 {question}"),
        SimpleNamespace(),
    )

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.call_args.args[0]
    assert event.text == question
    assert event.message_type == MessageType.TEXT
    assert event.metadata == {"skill_context": EXPECTED_CONTEXT}


def test_session_identifier_is_opaque_to_surface():
    event = MessageEvent(text="/bowlingsession definitely-not-real Review this session.")

    assert apply_explicit_bowling_session_context(event) is True
    assert event.metadata["skill_context"]["payload"] == {
        "sessionID": "definitely-not-real"
    }


@pytest.mark.parametrize(
    "text",
    [
        "/bowlingsession",
        "/bowlingsession session-123",
        "/bowlingsession " + "x" * 129 + " Review this session.",
        "/stop",
        "ordinary non-bowling conversation",
    ],
)
def test_malformed_or_unrelated_commands_preserve_legacy_event(text):
    event = MessageEvent(text=text, message_type=MessageType.COMMAND)

    assert apply_explicit_bowling_session_context(event) is False
    assert event.text == text
    assert event.message_type == MessageType.COMMAND
    assert event.metadata == {}


def test_existing_skill_context_is_never_overwritten():
    existing = {"schema": "another_skill_context/v1"}
    event = MessageEvent(
        text="/bowlingsession session-123 Review this session.",
        metadata={"skill_context": existing},
    )

    assert apply_explicit_bowling_session_context(event) is False
    assert event.metadata["skill_context"] is existing


def test_completed_session_cannot_become_live_context():
    event = MessageEvent(
        text="/bowlingsession session-123 My ball is reading early now."
    )

    assert apply_explicit_bowling_session_context(event) is True
    context = event.metadata["skill_context"]
    assert context["contextType"] == "completed_session"
    assert "liveState" not in context["payload"]
    assert "currentSessionID" not in context["payload"]


def test_context_is_request_scoped_and_does_not_leak():
    contextual = MessageEvent(
        text="/bowlingsession session-123 Review this session."
    )
    ordinary = MessageEvent(text="Review this session.")

    assert apply_explicit_bowling_session_context(contextual) is True
    assert apply_explicit_bowling_session_context(ordinary) is False
    assert "skill_context" in contextual.metadata
    assert ordinary.metadata == {}


def test_generic_runtime_size_guard_still_rejects_oversized_context():
    oversized = {"payload": "x" * MAX_SURFACE_CONTEXT_BYTES}
    assert surface_context_from_event_metadata({"skill_context": oversized}) is None


def test_surface_producer_has_no_storage_or_bowling_data_access():
    source = inspect.getsource(apply_explicit_bowling_session_context).lower()
    prohibited = (
        "bowlingtracker",
        "knowledge.db",
        "postgres",
        "sqlite",
        "sessionstore",
        "transcript",
        "state.db",
        "open(",
    )
    assert all(token not in source for token in prohibited)
