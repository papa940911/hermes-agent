"""Explicit Telegram producer for persistent bowling live-state references."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.skill_preparation import surface_context_from_event_metadata
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    apply_explicit_bowling_live_context,
)


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


def _message(text: str, message_id: int = 42):
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


def _update(text: str, update_id: int = 71):
    return SimpleNamespace(
        update_id=update_id,
        message=_message(text, message_id=update_id),
        effective_message=None,
    )


def test_explicit_live_state_reference_emits_exact_context():
    event = MessageEvent(
        text="/bowlinglive live-state-1 ball read early",
        message_type=MessageType.COMMAND,
        platform_update_id=71,
        metadata={"telegram_private": {"not": "forwarded"}},
    )

    assert apply_explicit_bowling_live_context(event) is True
    assert event.text == "ball read early"
    assert event.message_type == MessageType.TEXT
    assert event.metadata["skill_context"] == {
        "schema": "bowling_surface_context/v1",
        "contextType": "live_state_reference",
        "source": "surface_event",
        "provenance": "explicit_surface_selection",
        "payload": {
            "liveStateID": "live-state-1",
            "requestID": "telegram-update:71",
        },
    }
    bounded = surface_context_from_event_metadata(event.metadata)
    assert "telegram_private" not in bounded


@pytest.mark.asyncio
async def test_production_command_handler_dispatches_live_reference():
    adapter = _make_adapter()
    await adapter._handle_command(
        _update("/bowlinglive@test_bot live-state-1 outside didn't recover"),
        SimpleNamespace(),
    )
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.call_args.args[0]
    assert event.text == "outside didn't recover"
    assert event.metadata["skill_context"]["payload"] == {
        "liveStateID": "live-state-1",
        "requestID": "telegram-update:71",
    }


def test_create_command_forwards_structured_initialization_only():
    initialization = {
        "stateID": "live-state-1",
        "bowlingTrackerSessionID": "bt-1",
        "plan": {"schema": "bowling_pattern_play_plan/v1"},
        "transitionScenarios": [{"scenarioID": "synthetic"}],
        "planRef": "plan-1",
        "observationText": "ball read early",
    }
    event = MessageEvent(
        text="/bowlinglive create " + json.dumps(initialization),
        message_type=MessageType.COMMAND,
        platform_update_id=72,
    )
    assert apply_explicit_bowling_live_context(event) is True
    assert event.text == "ball read early"
    assert event.metadata["skill_context"] == {
        "schema": "bowling_surface_context/v1",
        "contextType": "live_state_create",
        "source": "surface_event",
        "provenance": "explicit_surface_selection",
        "payload": {
            "requestID": "telegram-update:72",
            "initialization": initialization,
        },
    }


@pytest.mark.parametrize(
    "text",
    [
        "/bowlinglive",
        "/bowlinglive live-state-1",
        "/bowlinglive  observation",
        "/bowlinglive " + "x" * 201 + " ball read early",
        "/bowlinglive create not-json",
        "/bowlinglive create []",
        "/stop",
        "ordinary text",
    ],
)
def test_malformed_or_unrelated_input_does_not_create_context(text):
    event = MessageEvent(
        text=text, message_type=MessageType.COMMAND, platform_update_id=73
    )
    assert apply_explicit_bowling_live_context(event) is False
    assert event.metadata == {}
    assert event.text == text


def test_missing_stable_delivery_identity_fails_closed():
    event = MessageEvent(text="/bowlinglive live-state-1 ball read early")
    assert apply_explicit_bowling_live_context(event) is False
    assert event.metadata == {}


def test_oversized_create_context_is_dropped_by_existing_runtime_guard():
    initialization = {
        "stateID": "live-state-1",
        "bowlingTrackerSessionID": "bt-1",
        "plan": {"schema": "bowling_pattern_play_plan/v1", "padding": "x" * 70_000},
        "transitionScenarios": [{"scenarioID": "synthetic"}],
        "planRef": "plan-1",
    }
    event = MessageEvent(
        text="/bowlinglive create " + json.dumps(initialization),
        message_type=MessageType.COMMAND,
        platform_update_id=74,
    )
    assert apply_explicit_bowling_live_context(event) is True
    assert surface_context_from_event_metadata(event.metadata) is None


def test_repeated_delivery_uses_same_request_identity():
    first = MessageEvent(
        text="/bowlinglive live-state-1 ball read early", platform_update_id=99
    )
    replay = MessageEvent(
        text="/bowlinglive live-state-1 ball read early", platform_update_id=99
    )
    assert apply_explicit_bowling_live_context(first) is True
    assert apply_explicit_bowling_live_context(replay) is True
    assert first.metadata["skill_context"] == replay.metadata["skill_context"]


def test_context_does_not_leak_to_next_request():
    first = MessageEvent(
        text="/bowlinglive live-state-1 ball read early", platform_update_id=1
    )
    second = MessageEvent(text="ball read early", platform_update_id=2)
    assert apply_explicit_bowling_live_context(first) is True
    assert apply_explicit_bowling_live_context(second) is False
    assert second.metadata == {}


@pytest.mark.asyncio
async def test_completed_session_and_legacy_commands_remain_compatible():
    adapter = _make_adapter()
    await adapter._handle_command(
        _update("/bowlingsession session-123 Review this session.", update_id=80),
        SimpleNamespace(),
    )
    first = adapter.handle_message.call_args.args[0]
    assert first.metadata["skill_context"]["contextType"] == "completed_session"

    adapter.handle_message.reset_mock()
    await adapter._handle_command(_update("/stop", update_id=81), SimpleNamespace())
    second = adapter.handle_message.call_args.args[0]
    assert second.text == "/stop"
    assert second.metadata == {}


def test_telegram_surface_has_no_bowling_data_or_persistence_access():
    import inspect
    import plugins.platforms.telegram.adapter as telegram_module

    source = inspect.getsource(telegram_module)
    helper = source[source.index("def apply_explicit_bowling_live_context"):]
    helper = helper[:helper.index("\ndef _coerce_duration_seconds")]
    for forbidden in (
        "sqlite3", "psycopg", "BowlingTracker", "knowledge.db",
        "SessionStore", "transcript", "live_session_state.db",
    ):
        assert forbidden not in helper
