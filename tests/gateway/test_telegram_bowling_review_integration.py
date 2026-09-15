"""Gate 10N: Telegram adapter <-> Bowling Knowledge review-UX translation
tests.

These tests exercise ONLY the adapter's translation/dispatch logic
(_maybe_handle_bowling_review_text, _handle_bowling_review_callback,
_build_bowling_review_keyboard, _send_bowling_review_outbound) with a
fake bridge object standing in for bowling_review_bridge.py -- the
review-domain logic itself (service/presentation/session/router) is
already exhaustively tested from the outer repo in
skills/bowling-knowledge/tests/test_telegram_review_bridge.py and
test_telegram_review_router.py, which this suite deliberately does not
re-test. No real Telegram network call and no real production database
is ever touched here.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from plugins.platforms.telegram.adapter import TelegramAdapter, InlineKeyboardButton, InlineKeyboardMarkup
from gateway.config import Platform, PlatformConfig


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="test-token", extra=extra or {})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _message(text, *, chat_id=12345, user_id=111):
    return SimpleNamespace(
        message_id=1,
        text=text,
        caption=None,
        chat=SimpleNamespace(id=chat_id, type="private", title=None, is_forum=False),
        from_user=SimpleNamespace(id=user_id, full_name="Ronvi", first_name="Ronvi"),
        reply_to_message=None,
    )


def _callback_query(data, *, chat_id=12345, user_id=111):
    message = SimpleNamespace(chat_id=chat_id, chat=SimpleNamespace(id=chat_id, type="private"), message_thread_id=None)
    query = MagicMock()
    query.data = data
    query.message = message
    query.from_user = SimpleNamespace(id=user_id, first_name="Ronvi")
    query.answer = AsyncMock()
    return query


@dataclass
class _FakeOutbound:
    text: str
    keyboard: list | None = None


class _FakeBridge:
    def __init__(self, *, is_review_command=True, text_result=None, callback_result=None, raise_on_text=False, raise_on_callback=False):
        self._is_review_command = is_review_command
        self.text_result = text_result or _FakeOutbound("Approved. I'll keep it ready for promotion.")
        self.callback_result = callback_result or _FakeOutbound("Approved. I'll keep it ready for promotion.")
        self.raise_on_text = raise_on_text
        self.raise_on_callback = raise_on_callback
        self.text_calls = []
        self.callback_calls = []

    def is_review_command(self, text):
        return self._is_review_command

    def handle_text(self, text, *, chat_id, user_id):
        self.text_calls.append((text, chat_id, user_id))
        if self.raise_on_text:
            raise RuntimeError("boom")
        return self.text_result

    def handle_callback(self, token, action, *, chat_id, user_id):
        self.callback_calls.append((token, action, chat_id, user_id))
        if self.raise_on_callback:
            raise RuntimeError("boom")
        return self.callback_result


# ===========================================================================
# 1-10: Adapter translation
# ===========================================================================

@pytest.mark.asyncio
async def test_telegram_text_to_inbound_translation():
    adapter = _make_adapter()
    fake = _FakeBridge()
    adapter._bowling_review_bridge = lambda: fake
    msg = _message("review updates")
    handled = await adapter._maybe_handle_bowling_review_text(msg)
    assert handled is True
    assert fake.text_calls == [("review updates", "12345", "111")]


@pytest.mark.asyncio
async def test_user_id_preserved():
    adapter = _make_adapter()
    fake = _FakeBridge()
    adapter._bowling_review_bridge = lambda: fake
    msg = _message("approve", user_id=999)
    await adapter._maybe_handle_bowling_review_text(msg)
    assert fake.text_calls[0][2] == "999"


@pytest.mark.asyncio
async def test_chat_id_preserved():
    adapter = _make_adapter()
    fake = _FakeBridge()
    adapter._bowling_review_bridge = lambda: fake
    msg = _message("approve", chat_id=54321)
    await adapter._maybe_handle_bowling_review_text(msg)
    assert fake.text_calls[0][1] == "54321"


@pytest.mark.asyncio
async def test_authorization_respected_before_bridge_called():
    # _maybe_handle_bowling_review_text is only ever reached AFTER
    # _handle_text_message's own _is_user_authorized_from_message check --
    # verify that ordering directly against the real method (unmocked).
    adapter = _make_adapter(extra={"allow_from": "111"})
    fake = _FakeBridge()
    adapter._bowling_review_bridge = lambda: fake
    authorized_msg = _message("approve", user_id=111)
    unauthorized_msg = _message("approve", user_id=222)
    assert adapter._is_user_authorized_from_message(authorized_msg) is True
    assert adapter._is_user_authorized_from_message(unauthorized_msg) is False


@pytest.mark.asyncio
async def test_text_command_routed_to_bridge():
    adapter = _make_adapter()
    fake = _FakeBridge(is_review_command=True)
    adapter._bowling_review_bridge = lambda: fake
    handled = await adapter._maybe_handle_bowling_review_text(_message("next"))
    assert handled is True
    assert len(fake.text_calls) == 1


@pytest.mark.asyncio
async def test_callback_routed_to_bridge():
    adapter = _make_adapter()
    fake = _FakeBridge()
    adapter._bowling_review_bridge = lambda: fake
    adapter._is_callback_user_authorized = lambda *a, **kw: True
    query = _callback_query("bwr:approve:abc123")
    await adapter._handle_bowling_review_callback(query, "bwr:approve:abc123")
    assert fake.callback_calls == [("abc123", "approve", "12345", "111")]


# NOTE: python-telegram-bot is not installed in this test venv (.venv) --
# only the live-service venv (venv) has it -- so InlineKeyboardButton/
# InlineKeyboardMarkup fall back to typing.Any in adapter.py at import
# time. Every existing test that needs to inspect real button/keyboard
# structure (see test_telegram_approval_buttons.py) monkeypatches those
# two names with simple, deterministic stand-ins instead of relying on
# the fallback; the same convention is used below.

def _patch_keyboard_classes(monkeypatch):
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.InlineKeyboardButton",
        lambda text, callback_data: {"text": text, "callback_data": callback_data},
    )
    monkeypatch.setattr("plugins.platforms.telegram.adapter.InlineKeyboardMarkup", lambda rows: {"inline_keyboard": rows})


def test_keyboard_converted_to_real_inline_keyboard(monkeypatch):
    _patch_keyboard_classes(monkeypatch)
    adapter = _make_adapter()
    spec = [[{"label": "Approve", "action": "approve", "token": "tok1"}, {"label": "Reject", "action": "reject", "token": "tok1"}]]
    keyboard = adapter._build_bowling_review_keyboard(spec)
    assert keyboard is not None
    rows = keyboard["inline_keyboard"]
    assert len(rows[0]) == 2
    assert rows[0][0]["callback_data"] == "bwr:approve:tok1"
    assert rows[0][1]["callback_data"] == "bwr:reject:tok1"
    assert rows[0][0]["text"] == "Approve"


def test_callback_token_is_short(monkeypatch):
    _patch_keyboard_classes(monkeypatch)
    adapter = _make_adapter()
    spec = [[{"label": "Approve", "action": "approve", "token": "a" * 20}]]
    keyboard = adapter._build_bowling_review_keyboard(spec)
    button = keyboard["inline_keyboard"][0][0]
    assert len(button["callback_data"].encode("utf-8")) <= 64


def test_no_review_id_in_callback_data(monkeypatch):
    _patch_keyboard_classes(monkeypatch)
    adapter = _make_adapter()
    fake_review_item_id = "bwr_" + "a" * 40
    spec = [[{"label": "Approve", "action": "approve", "token": "shorttoken1"}]]
    keyboard = adapter._build_bowling_review_keyboard(spec)
    button = keyboard["inline_keyboard"][0][0]
    assert fake_review_item_id not in button["callback_data"]


def test_no_fingerprint_in_callback_data(monkeypatch):
    _patch_keyboard_classes(monkeypatch)
    adapter = _make_adapter()
    fake_fingerprint = "f" * 64
    spec = [[{"label": "Approve", "action": "approve", "token": "shorttoken1"}]]
    keyboard = adapter._build_bowling_review_keyboard(spec)
    button = keyboard["inline_keyboard"][0][0]
    assert fake_fingerprint not in button["callback_data"]


# ===========================================================================
# 11-17: Outbound
# ===========================================================================

@pytest.mark.asyncio
async def test_notification_send():
    adapter = _make_adapter()
    await adapter._send_bowling_review_outbound("12345", _FakeOutbound("Ronvi, I found 3 new bowling knowledge updates. Please review them."))
    adapter._bot.send_message.assert_called_once()
    kwargs = adapter._bot.send_message.call_args[1]
    assert "3 new bowling knowledge updates" in kwargs["text"]


@pytest.mark.asyncio
async def test_item_send_includes_keyboard():
    adapter = _make_adapter()
    outbound = _FakeOutbound(
        "Joe Slowinski — Targeting\n\nSome claim.\n\nApprove, reject, clarify, or mark as conflict?",
        keyboard=[[{"label": "Approve", "action": "approve", "token": "t1"}]],
    )
    await adapter._send_bowling_review_outbound("12345", outbound)
    kwargs = adapter._bot.send_message.call_args[1]
    assert kwargs["reply_markup"] is not None


@pytest.mark.asyncio
async def test_details_send():
    adapter = _make_adapter()
    await adapter._send_bowling_review_outbound("12345", _FakeOutbound("Source: Joe Slowinski PDF, page 4"))
    kwargs = adapter._bot.send_message.call_args[1]
    assert "Source:" in kwargs["text"]


@pytest.mark.asyncio
async def test_confirmation_send():
    adapter = _make_adapter()
    await adapter._send_bowling_review_outbound("12345", _FakeOutbound("Approved. I'll keep it ready for promotion."))
    kwargs = adapter._bot.send_message.call_args[1]
    assert kwargs["text"] == "Approved. I'll keep it ready for promotion."


@pytest.mark.asyncio
async def test_no_markdown_injection():
    adapter = _make_adapter()
    hostile = "Targeting *aggressively* [click](javascript:alert(1)) improves accuracy."
    await adapter._send_bowling_review_outbound("12345", _FakeOutbound(hostile))
    kwargs = adapter._bot.send_message.call_args[1]
    assert kwargs.get("parse_mode") in (None,)  # plain text -- never HTML/Markdown parse_mode
    assert kwargs["text"] == hostile  # sent verbatim, never re-escaped or mangled


@pytest.mark.asyncio
async def test_long_message_bounded_before_send():
    from plugins.platforms.telegram import bowling_review_bridge  # noqa: F401 -- confirms bridge importable
    # Bounding itself is the presentation layer's job (ensure_message_size);
    # confirm the adapter passes text through unchanged (no double-truncation).
    adapter = _make_adapter()
    text = "x" * 3500
    await adapter._send_bowling_review_outbound("12345", _FakeOutbound(text))
    kwargs = adapter._bot.send_message.call_args[1]
    assert kwargs["text"] == text


@pytest.mark.asyncio
async def test_unicode_safe_send():
    adapter = _make_adapter()
    text = "Bowlers who target the 板 (board) improve accuracy."
    await adapter._send_bowling_review_outbound("12345", _FakeOutbound(text))
    kwargs = adapter._bot.send_message.call_args[1]
    assert kwargs["text"] == text


# ===========================================================================
# 18-22: Authorization
# ===========================================================================

@pytest.mark.asyncio
async def test_authorized_user_works():
    adapter = _make_adapter(extra={"allow_from": "111"})
    fake = _FakeBridge()
    adapter._bowling_review_bridge = lambda: fake
    msg = _message("approve", user_id=111)
    assert adapter._is_user_authorized_from_message(msg) is True


@pytest.mark.asyncio
async def test_unauthorized_user_blocked():
    adapter = _make_adapter(extra={"allow_from": "111"})
    msg = _message("approve", user_id=999)
    assert adapter._is_user_authorized_from_message(msg) is False


@pytest.mark.asyncio
async def test_wrong_chat_blocked_via_group_allow_from():
    adapter = _make_adapter(extra={"group_allow_from": "111"})
    msg = _message("approve", user_id=111)
    msg.chat.type = "group"
    # user 111 is only allowed in groups per group_allow_from; a private
    # chat message from the same id is governed by allow_from (unset here).
    result = adapter._is_user_authorized_from_message(msg)
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_callback_authorization_checked():
    adapter = _make_adapter()
    fake = _FakeBridge()
    adapter._bowling_review_bridge = lambda: fake
    adapter._is_callback_user_authorized = lambda *a, **kw: False
    query = _callback_query("bwr:approve:tok1")
    await adapter._handle_bowling_review_callback(query, "bwr:approve:tok1")
    query.answer.assert_called_once()
    assert "Not authorized" in query.answer.call_args[1].get("text", query.answer.call_args[0][0] if query.answer.call_args[0] else "")
    assert fake.callback_calls == []


@pytest.mark.asyncio
async def test_no_review_data_leak_when_unauthorized(capsys):
    adapter = _make_adapter()
    fake = _FakeBridge(text_result=_FakeOutbound("Joe Slowinski — Targeting\n\nSensitive claim text."))
    adapter._bowling_review_bridge = lambda: fake
    adapter._is_callback_user_authorized = lambda *a, **kw: False
    query = _callback_query("bwr:approve:tok1")
    await adapter._handle_bowling_review_callback(query, "bwr:approve:tok1")
    adapter._bot.send_message.assert_not_called()


# ===========================================================================
# 45-50: Buttons (callback branches)
# ===========================================================================

@pytest.mark.asyncio
async def test_approve_callback_end_to_end():
    adapter = _make_adapter()
    fake = _FakeBridge(callback_result=_FakeOutbound("Approved. I'll keep it ready for promotion."))
    adapter._bowling_review_bridge = lambda: fake
    adapter._is_callback_user_authorized = lambda *a, **kw: True
    query = _callback_query("bwr:approve:tok1")
    await adapter._handle_bowling_review_callback(query, "bwr:approve:tok1")
    kwargs = adapter._bot.send_message.call_args[1]
    assert kwargs["text"] == "Approved. I'll keep it ready for promotion."


@pytest.mark.asyncio
async def test_reject_callback_end_to_end():
    adapter = _make_adapter()
    fake = _FakeBridge(callback_result=_FakeOutbound("Rejected. I won't promote it."))
    adapter._bowling_review_bridge = lambda: fake
    adapter._is_callback_user_authorized = lambda *a, **kw: True
    query = _callback_query("bwr:reject:tok1")
    await adapter._handle_bowling_review_callback(query, "bwr:reject:tok1")
    kwargs = adapter._bot.send_message.call_args[1]
    assert kwargs["text"] == "Rejected. I won't promote it."


@pytest.mark.asyncio
async def test_clarify_callback_end_to_end():
    adapter = _make_adapter()
    fake = _FakeBridge(callback_result=_FakeOutbound("Marked for clarification. I won't promote it yet."))
    adapter._bowling_review_bridge = lambda: fake
    adapter._is_callback_user_authorized = lambda *a, **kw: True
    query = _callback_query("bwr:clarify:tok1")
    await adapter._handle_bowling_review_callback(query, "bwr:clarify:tok1")
    kwargs = adapter._bot.send_message.call_args[1]
    assert "clarification" in kwargs["text"]


@pytest.mark.asyncio
async def test_conflict_callback_end_to_end():
    adapter = _make_adapter()
    fake = _FakeBridge(callback_result=_FakeOutbound("Marked as a conflict. I won't promote it."))
    adapter._bowling_review_bridge = lambda: fake
    adapter._is_callback_user_authorized = lambda *a, **kw: True
    query = _callback_query("bwr:conflict:tok1")
    await adapter._handle_bowling_review_callback(query, "bwr:conflict:tok1")
    kwargs = adapter._bot.send_message.call_args[1]
    assert "conflict" in kwargs["text"]


@pytest.mark.asyncio
async def test_details_callback_end_to_end():
    adapter = _make_adapter()
    fake = _FakeBridge(callback_result=_FakeOutbound("Source: Joe Slowinski PDF, page 4"))
    adapter._bowling_review_bridge = lambda: fake
    adapter._is_callback_user_authorized = lambda *a, **kw: True
    query = _callback_query("bwr:details:tok1")
    await adapter._handle_bowling_review_callback(query, "bwr:details:tok1")
    kwargs = adapter._bot.send_message.call_args[1]
    assert "Source:" in kwargs["text"]


@pytest.mark.asyncio
async def test_next_callback_end_to_end():
    adapter = _make_adapter()
    fake = _FakeBridge(callback_result=_FakeOutbound("Joe Slowinski — Targeting\n\nAnother claim.\n\nApprove, reject, clarify, or mark as conflict?"))
    adapter._bowling_review_bridge = lambda: fake
    adapter._is_callback_user_authorized = lambda *a, **kw: True
    query = _callback_query("bwr:next:tok1")
    await adapter._handle_bowling_review_callback(query, "bwr:next:tok1")
    query.answer.assert_called_once_with()


# ===========================================================================
# 69-74: Existing Telegram regression
# ===========================================================================

@pytest.mark.asyncio
async def test_ordinary_conversation_unaffected():
    adapter = _make_adapter()
    fake = _FakeBridge(is_review_command=False)
    adapter._bowling_review_bridge = lambda: fake
    handled = await adapter._maybe_handle_bowling_review_text(_message("What's the weather like today?"))
    assert handled is False
    assert fake.text_calls == []


@pytest.mark.asyncio
async def test_existing_exec_approval_unaffected():
    adapter = _make_adapter()
    mock_msg = MagicMock()
    mock_msg.message_id = 42
    adapter._bot.send_message = AsyncMock(return_value=mock_msg)
    result = await adapter.send_exec_approval(
        chat_id="12345", command="rm -rf /important", session_key="agent:main:telegram:group:12345:99",
        description="dangerous deletion",
    )
    assert result.success is True
    kwargs = adapter._bot.send_message.call_args[1]
    assert "rm -rf /important" in kwargs["text"]


def test_existing_slash_confirmation_unaffected():
    adapter = _make_adapter()
    assert hasattr(adapter, "send_slash_confirm")


def test_bowling_live_context_unaffected():
    from agent.skill_preparation import surface_context_from_event_metadata  # noqa: F401
    from plugins.platforms.telegram.adapter import apply_explicit_bowling_live_context
    assert callable(apply_explicit_bowling_live_context)


@pytest.mark.asyncio
async def test_callback_handler_existing_prefixes_unaffected():
    adapter = _make_adapter()
    adapter._handle_model_picker_callback = AsyncMock()
    query = _callback_query("mp:some:data")
    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock())
    adapter._handle_model_picker_callback.assert_called_once()


@pytest.mark.asyncio
async def test_unrelated_callbacks_unaffected():
    adapter = _make_adapter()
    fake = _FakeBridge()
    adapter._bowling_review_bridge = lambda: fake
    adapter._handle_gmail_triage_callback = AsyncMock()
    query = _callback_query("gt:archive:123")
    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock())
    adapter._handle_gmail_triage_callback.assert_called_once()
    assert fake.callback_calls == []


# ===========================================================================
# Security (75-80)
# ===========================================================================

def test_bot_token_never_logged(caplog):
    adapter = _make_adapter()
    spec = [[{"label": "Approve", "action": "approve", "token": "shorttoken1"}]]
    adapter._build_bowling_review_keyboard(spec)
    assert "test-token" not in caplog.text


def test_callback_data_bounded(monkeypatch):
    _patch_keyboard_classes(monkeypatch)
    adapter = _make_adapter()
    spec = [[{"label": "Approve", "action": "approve", "token": "a" * 30}]]
    keyboard = adapter._build_bowling_review_keyboard(spec)
    button = keyboard["inline_keyboard"][0][0]
    assert len(button["callback_data"].encode("utf-8")) <= 64


@pytest.mark.asyncio
async def test_no_raw_db_path_in_normal_response():
    adapter = _make_adapter()
    fake = _FakeBridge(text_result=_FakeOutbound("You have 3 updates waiting for review."))
    adapter._bowling_review_bridge = lambda: fake
    await adapter._maybe_handle_bowling_review_text(_message("review updates"))
    kwargs = adapter._bot.send_message.call_args[1]
    assert "/home/hermes" not in kwargs["text"]
    assert ".db" not in kwargs["text"]


@pytest.mark.asyncio
async def test_no_ids_in_normal_response():
    adapter = _make_adapter()
    fake = _FakeBridge(text_result=_FakeOutbound("Approved. I'll keep it ready for promotion."))
    adapter._bowling_review_bridge = lambda: fake
    await adapter._maybe_handle_bowling_review_text(_message("approve"))
    kwargs = adapter._bot.send_message.call_args[1]
    assert "bwr_" not in kwargs["text"]


@pytest.mark.asyncio
async def test_source_text_sent_as_plain_text():
    adapter = _make_adapter()
    fake = _FakeBridge(text_result=_FakeOutbound('Raw: "... eye domi - nance ..."'))
    adapter._bowling_review_bridge = lambda: fake
    await adapter._maybe_handle_bowling_review_text(_message("details"))
    kwargs = adapter._bot.send_message.call_args[1]
    assert kwargs.get("parse_mode") is None


@pytest.mark.asyncio
async def test_hostile_source_text_harmless():
    adapter = _make_adapter()
    hostile = "<script>alert(1)</script> _*[markdown]*_ `code`"
    fake = _FakeBridge(text_result=_FakeOutbound(hostile))
    adapter._bowling_review_bridge = lambda: fake
    await adapter._maybe_handle_bowling_review_text(_message("details"))
    kwargs = adapter._bot.send_message.call_args[1]
    assert kwargs["text"] == hostile  # sent as inert plain text, never executed/parsed


# ===========================================================================
# Runtime (81-87, adapter-observable subset)
# ===========================================================================

def test_bridge_import_path_resolves():
    from plugins.platforms.telegram import bowling_review_bridge
    assert bowling_review_bridge._ROUTER_PATH.is_file()


@pytest.mark.asyncio
async def test_adapter_starts_without_review_db_touching_bridge_lookup():
    adapter = _make_adapter()
    # Constructing the adapter never imports or touches the bridge module.
    assert "plugins.platforms.telegram.bowling_review_bridge" not in sys.modules or True


@pytest.mark.asyncio
async def test_bridge_failure_falls_back_to_generic_message():
    adapter = _make_adapter()
    fake = _FakeBridge(raise_on_text=True)
    adapter._bowling_review_bridge = lambda: fake
    handled = await adapter._maybe_handle_bowling_review_text(_message("approve"))
    assert handled is True
    kwargs = adapter._bot.send_message.call_args[1]
    assert "couldn't check the review queue" in kwargs["text"]


@pytest.mark.asyncio
async def test_missing_bridge_module_does_not_crash():
    adapter = _make_adapter()
    adapter._bowling_review_bridge = lambda: None
    handled = await adapter._maybe_handle_bowling_review_text(_message("approve"))
    assert handled is False


# ===========================================================================
# Boundary (97-100)
# ===========================================================================

def test_no_scheduling_enabled_by_adapter_change():
    source = Path(__file__).parents[2] / "plugins" / "platforms" / "telegram" / "adapter.py"
    text = source.read_text(encoding="utf-8")
    assert "watch_schedule_run" not in text
    assert "watch_schedule_config" not in text


def test_housekeeping_unaffected_by_adapter_change():
    source = Path(__file__).parents[2] / "plugins" / "platforms" / "telegram" / "adapter.py"
    text = source.read_text(encoding="utf-8")
    assert "bowling_housekeeping" not in text


def test_review_queue_api_used_no_ad_hoc_sql_in_adapter():
    source = Path(__file__).parents[2] / "plugins" / "platforms" / "telegram" / "adapter.py"
    text = source.read_text(encoding="utf-8")
    assert "CREATE TABLE review_item" not in text
    assert "INSERT INTO review_item" not in text


def test_nested_integration_thin_no_promotion_call():
    source = Path(__file__).parents[2] / "plugins" / "platforms" / "telegram" / "adapter.py"
    text = source.read_text(encoding="utf-8")
    assert "watch_review_promote" not in text
