"""Request-scoped skill-context transport guards."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from gateway.run import GatewayRunner


def test_gateway_run_forwards_detached_context_as_a_turn_argument_only():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    seen = []

    async def fake_inner(*args, **kwargs):
        seen.append((args, kwargs))
        return {"final_response": "ok"}

    runner._run_agent_inner = fake_inner
    context = {"schema": "example/v1", "payload": {"id": "selected"}}
    result = asyncio.run(runner._run_agent(
        message="request",
        context_prompt="prompt",
        history=[],
        source=SimpleNamespace(),
        session_id="session",
        surface_context=context,
    ))
    assert result == {"final_response": "ok"}
    assert seen[0][1]["surface_context"] == context


def test_context_transport_has_no_session_transcript_or_routing_snapshot_sink():
    root = Path(__file__).resolve().parents[2]
    prohibited = (
        root / "gateway" / "session_store.py",
        root / "hermes_state.py",
        root / "agent" / "turn_context.py",
    )
    for path in prohibited:
        if path.is_file():
            assert "surface_context" not in path.read_text(encoding="utf-8")

    routing_source = (root / "agent" / "skill_routing.py").read_text(encoding="utf-8")
    assert 'route["surface_context"]' not in routing_source
    assert 'result["surface_context"]' not in routing_source


def test_runtime_uses_no_global_context_cache():
    root = Path(__file__).resolve().parents[2]
    preparation = (root / "agent" / "skill_preparation.py").read_text(encoding="utf-8")
    gateway = (root / "gateway" / "run.py").read_text(encoding="utf-8")
    for forbidden in (
        "_surface_context_cache",
        "surface_context.db",
        "surface_context.json",
        "SessionStore.surface_context",
    ):
        assert forbidden not in preparation
        assert forbidden not in gateway
