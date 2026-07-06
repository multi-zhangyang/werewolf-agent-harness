"""RoomManager 生命周期边界测试 —— 不调用真实 LLM。"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

import src.api.room_manager as room_manager_module
from src.api.room_manager import Room, RoomManager
from src.config import LLM_CONCURRENCY, LLM_MAX_RETRIES, LLM_TIMEOUT
from src.game.models import Phase
from src.game.roles import Team
from src.game.state import new_game
from src.llm.models import ModelConfig


class _CompleteOrchestrator:
    def __init__(self, room: Room) -> None:
        self.room = room

    async def run(self):
        self.room.state.phase = Phase.ENDED
        self.room.state.winner = Team.VILLAGE
        return self.room.state


class _NeverEndsOrchestrator:
    async def run(self):
        await asyncio.sleep(3600)


class _BoomOrchestrator:
    async def run(self):
        raise RuntimeError("boom")


def _room() -> Room:
    return Room(
        id="audit",
        state=new_game(["A", "B", "C", "D", "E", "F"]),
        status="running",
    )


def _recording_manager(*, room_timeout: float | None = 1.0) -> tuple[RoomManager, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    manager = RoomManager(room_timeout=room_timeout)

    async def record(_room: Room, payload: dict[str, Any], *, thinking: bool = False) -> None:
        events.append(payload)

    manager._broadcast = record  # type: ignore[method-assign]
    return manager, events


def test_room_manager_default_router_uses_llm_runtime_config():
    manager = RoomManager()

    assert manager.router.timeout == LLM_TIMEOUT
    assert manager.router.max_retries == LLM_MAX_RETRIES
    assert manager.router._sem._value == LLM_CONCURRENCY


def test_set_seat_model_config_rejects_endpoint_change_without_explicit_key():
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=ModelConfig(
            provider="openai",
            model="default-model",
            api_base="https://default.example/v1",
            api_key="unit-test-secret",
            temperature=0.7,
        ),
    )

    with pytest.raises(ValueError, match="api_key"):
        manager.set_seat_model_config(room, 1, {
            "model": "seat-model",
            "api_base": "https://attacker.invalid/v1",
        })


@pytest.mark.parametrize("field_name", ["extra_body", "thinking", "reasoning_effort", "top_k"])
def test_set_seat_model_config_rejects_non_standard_fields(field_name):
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=ModelConfig(
            provider="openai",
            model="default-model",
            api_base="https://default.example/v1",
            api_key="unit-test-secret",
        ),
    )

    with pytest.raises(ValueError, match=field_name):
        manager.set_seat_model_config(room, 1, {
            "model": "seat-model",
            field_name: {"enabled": True},
        })


@pytest.mark.parametrize(("field_name", "value"), [("thinking", ""), ("top_k", None), ("reasoning_effort", "")])
def test_set_seat_model_config_rejects_empty_non_standard_fields(field_name, value):
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=ModelConfig(
            provider="openai",
            model="default-model",
            api_base="https://default.example/v1",
            api_key="unit-test-secret",
        ),
    )

    with pytest.raises(ValueError, match=field_name):
        manager.set_seat_model_config(room, 1, {
            "model": "seat-model",
            field_name: value,
        })


@pytest.mark.parametrize("provider", ["openai_responses", "anthropic"])
def test_set_seat_model_config_rejects_provider_change_without_explicit_key(provider):
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=ModelConfig(
            provider="openai",
            model="default-model",
            api_base="https://default.example/v1",
            api_key="unit-test-secret",
            temperature=0.7,
        ),
    )

    with pytest.raises(ValueError, match="api_key"):
        manager.set_seat_model_config(room, 1, {
            "provider": provider,
            "model": "seat-model",
        })


@pytest.mark.parametrize("provider", ["openai_responses", "anthropic"])
def test_set_seat_model_config_accepts_provider_change_with_explicit_key(provider):
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=ModelConfig(
            provider="openai",
            model="default-model",
            api_base="https://default.example/v1",
            api_key="unit-test-secret",
            temperature=0.7,
        ),
    )

    manager.set_seat_model_config(room, 1, {
        "provider": provider,
        "model": "seat-model",
        "api_base": "https://seat.example/v1",
        "api_key": "seat-key",
    })

    cfg = room.seat_configs[1]
    assert cfg.provider == provider
    assert cfg.model == "seat-model"
    assert cfg.api_base == "https://seat.example/v1"
    assert cfg.api_key == "seat-key"


@pytest.mark.asyncio
async def test_start_game_merges_per_seat_config_with_room_default(monkeypatch):
    manager = RoomManager(room_timeout=1.0)

    async def no_run(_room: Room) -> None:
        return None

    monkeypatch.setattr(manager, "_run_room", no_run)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=ModelConfig(
            provider="openai",
            model="default-model",
            api_base="https://example.invalid/v1",
            api_key="unit-test-key",
            temperature=0.7,
        ),
    )
    manager.set_seat_model_config(room, 1, {"model": "seat-model"})

    await manager.start_game(room)
    try:
        actor = next(a for a in room.actors.values() if a.seat == 1)
        assert actor.model_config.model == "seat-model"
        assert actor.model_config.api_base == "https://example.invalid/v1"
        assert actor.model_config.api_key == "unit-test-key"
        assert actor.model_config.temperature == 0.7
    finally:
        assert room.task is not None
        await room.task


@pytest.mark.asyncio
async def test_start_game_passes_per_phase_deadlines_to_orchestrator(monkeypatch):
    manager = RoomManager(room_timeout=1.0)
    phase_deadlines = {
        "night": 0.0,
        "day": 0.25,
        "voting": 0.5,
        "pk": 0.0,
        "last_words": 0.0,
        "hunter": 0.0,
        "reflection": 0.0,
    }
    monkeypatch.setattr(room_manager_module, "AGENT_PHASE_DEADLINE", 0.0)
    monkeypatch.setattr(room_manager_module, "AGENT_PHASE_DEADLINE_BY_PHASE", phase_deadlines)

    async def no_run(_room: Room) -> None:
        return None

    monkeypatch.setattr(manager, "_run_room", no_run)
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])

    await manager.start_game(room)
    assert room.orchestrator is not None
    try:
        assert room.orchestrator.phase_deadline == 0.0
        assert room.orchestrator.phase_deadlines["day"] == 0.25
        assert room.orchestrator.phase_deadlines["voting"] == 0.5
    finally:
        assert room.task is not None
        await room.task


@pytest.mark.asyncio
async def test_start_game_broadcasts_running_status(monkeypatch):
    manager, events = _recording_manager(room_timeout=1.0)

    async def no_run(_room: Room) -> None:
        return None

    monkeypatch.setattr(manager, "_run_room", no_run)
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])

    await manager.start_game(room)
    try:
        assert room.status == "running"
        assert events[0] == {"type": "room_status", "status": "running", "reason": None}
    finally:
        assert room.task is not None
        await room.task


@pytest.mark.asyncio
async def test_run_room_marks_normal_completion_as_ended():
    manager, events = _recording_manager()
    room = _room()
    room.orchestrator = _CompleteOrchestrator(room)  # type: ignore[assignment]

    await manager._run_room(room)

    assert room.status == "ended"
    assert room.end_reason == "completed"
    assert room.error is None
    assert events[-1] == {"type": "room_status", "status": "ended", "reason": "completed"}


@pytest.mark.asyncio
async def test_run_room_timeout_is_not_reported_as_ended():
    manager, events = _recording_manager(room_timeout=0.05)
    room = _room()
    room.orchestrator = _NeverEndsOrchestrator()  # type: ignore[assignment]

    await manager._run_room(room)

    assert room.status == "timeout"
    assert room.end_reason == "timeout"
    assert room.error and "timeout" in room.error
    assert any(ev["type"] == "game_error" and ev.get("reason") == "timeout" for ev in events)
    assert events[-1]["type"] == "room_status"
    assert events[-1]["status"] == "timeout"


@pytest.mark.asyncio
async def test_run_room_exception_is_failed_not_ended():
    manager, events = _recording_manager()
    room = _room()
    room.orchestrator = _BoomOrchestrator()  # type: ignore[assignment]

    await manager._run_room(room)

    assert room.status == "failed"
    assert room.end_reason == "error"
    assert room.error == "RuntimeError during game loop"
    assert any(ev["type"] == "game_error" and ev.get("message") == "RuntimeError during game loop" for ev in events)
    assert not any("boom" in str(ev) for ev in events)
    assert events[-1]["status"] == "failed"


@pytest.mark.asyncio
async def test_run_room_cancelled_status_and_propagates_cancel():
    manager, events = _recording_manager(room_timeout=10.0)
    room = _room()
    room.orchestrator = _NeverEndsOrchestrator()  # type: ignore[assignment]

    task = asyncio.create_task(manager._run_room(room))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert room.status == "cancelled"
    assert room.end_reason == "cancelled"
    assert events[-1]["type"] == "room_status"
    assert events[-1]["status"] == "cancelled"
