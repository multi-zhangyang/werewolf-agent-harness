"""RoomManager 生命周期边界测试 —— 不调用真实 LLM。"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import pytest

import src.api.room_manager as room_manager_module
from src.api.room_manager import (
    DeliveryHistoryGapError,
    FutureDeliveryCursorError,
    Room,
    RoomCapacityError,
    RoomClientCapacityError,
    RoomEvidenceLimitError,
    RoomInUseError,
    RoomManager,
    RoomManagerCleanupError,
    RoomManagerUnavailableError,
)
from src.config import LLM_CONCURRENCY, LLM_MAX_RETRIES, LLM_TIMEOUT
from src.agent.schemas import AgentAction, Decision
from src.environments.werewolf.plugin import WerewolfEnvironmentPlugin
from src.game.models import Event, Phase
from src.game.roles import Team
from src.game.state import new_game
from src.harness.agent_protocol import ActionRequest, DecisionEnvelope
from src.harness.environment import EnvironmentOutcome
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


class _IncompleteOrchestrator:
    termination_status = "incomplete"
    termination_reason = "max_game_rounds"

    def __init__(self, room: Room) -> None:
        self.room = room

    async def run(self):
        self.room.state.phase = Phase.ENDED
        self.room.state.winner = None
        return self.room.state


class _BoomOrchestrator:
    async def run(self):
        raise RuntimeError("boom")


class _InternalTimeoutOrchestrator:
    async def run(self):
        raise TimeoutError("internal timeout detail")


class _IgnoresCancellationOrchestrator:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self):
        self.started.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                continue


class _FakeWebSocket:
    def __init__(self, *, block_on_send: int | None = None) -> None:
        self.block_on_send = block_on_send
        self.accepted = False
        self.send_count = 0
        self.messages: list[str] = []
        self.close_calls: list[tuple[int, str]] = []
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()

    async def accept(self, subprotocol: str | None = None) -> None:
        self.accepted = True

    async def send_text(self, message: str) -> None:
        self.send_count += 1
        if self.send_count == self.block_on_send:
            self.send_started.set()
            await self.release_send.wait()
        self.messages.append(message)

    async def close(self, *, code: int = 1000, reason: str = "") -> None:
        self.close_calls.append((code, reason))
        self.release_send.set()

    def json_messages(self) -> list[dict[str, Any]]:
        return [json.loads(message) for message in self.messages]


async def _wait_for_message_count(socket: _FakeWebSocket, count: int) -> None:
    async def wait() -> None:
        while len(socket.messages) < count:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1.0)


def _room() -> Room:
    return Room(
        id="trace-test",
        state=new_game(["A", "B", "C", "D", "E", "F"]),
        status="running",
    )


def _recording_manager(*, room_timeout: float | None = 1.0) -> tuple[RoomManager, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    manager = RoomManager(room_timeout=room_timeout)

    async def record(_room: Room, payload: dict[str, Any]) -> None:
        events.append(payload)

    manager._broadcast = record  # type: ignore[method-assign]
    return manager, events


def _model_config() -> ModelConfig:
    return ModelConfig(
        provider="openai",
        model="unit-test-model",
        api_base="https://example.invalid/v1",
        api_key="unit-test-key",
    )


def test_create_room_uses_room_id_as_game_id() -> None:
    manager = RoomManager()
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])

    assert room.state.id == room.id
    assert room.state.phase == Phase.SETUP
    assert all(player.role is None for player in room.state.players)
    assert room.run_spec is None
    assert room.transcript is None
    assert room.actors == {}
    assert room.orchestrator is None
    assert room.task is None


def test_create_room_deep_copies_default_model_config() -> None:
    manager = RoomManager()
    supplied = ModelConfig(
        model="original-model",
        api_key="original-key",
        reasoning={"effort": "high"},
    )

    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=supplied,
    )
    assert room.default_config is not None
    assert room.default_config is not supplied

    supplied.model = "mutated-model"
    assert supplied.reasoning is not None
    supplied.reasoning["effort"] = "low"
    assert room.default_config.model == "original-model"
    assert room.default_config.reasoning == {"effort": "high"}


def test_room_manager_default_router_uses_llm_runtime_config():
    manager = RoomManager()

    assert manager.router.timeout == LLM_TIMEOUT
    assert manager.router.max_retries == LLM_MAX_RETRIES
    assert manager.router._sem._value == LLM_CONCURRENCY


def test_room_evidence_limit_is_transactional_across_sources_and_transcript():
    manager = RoomManager(max_evidence_entries=2)
    room = _room()

    manager._store_room_event(room, {"type": "phase_started", "phase": "day"})
    record_trace = manager._make_trace_recorder(room)
    record_trace({"type": "decision_consumed", "seat": 1})

    assert len(room.event_history) == 1
    assert len(room.decision_trace) == 1
    assert room.transcript is not None
    assert len(room.transcript.entries) == 2

    with pytest.raises(RoomEvidenceLimitError) as exceeded:
        record_trace({"type": "decision_consumed", "seat": 2})

    assert exceeded.value.current == 3
    assert exceeded.value.limit == 2
    assert len(room.event_history) == 1
    assert len(room.decision_trace) == 1
    assert len(room.transcript.entries) == 2


def test_room_evidence_limit_includes_persisted_game_state_events():
    manager = RoomManager(max_evidence_entries=1)
    room = _room()
    room.state.events.append(Event(
        phase=Phase.DAY,
        day=1,
        type="speech",
        message="accepted environment event",
    ))

    # A rules event can already be present in state.events when its mirrored
    # manager event is admitted.  The prospective check must not double-count
    # that one logical row at the boundary.
    manager._ensure_evidence_capacity(room)
    manager._store_room_event(room, {"type": "speech", "seat": 1})
    assert len(room.event_history) == 1
    assert room.transcript is not None and len(room.transcript.entries) == 1

    with pytest.raises(RoomEvidenceLimitError) as exceeded:
        manager._store_room_event(room, {"type": "phase_started", "phase": "day"})

    assert exceeded.value.current == 2
    assert exceeded.value.limit == 1
    assert manager._room_record(room)["room_id"] == room.id

    oversized = _room()
    oversized.state.events.extend([
        Event(phase=Phase.DAY, day=1, type="speech", message="one"),
        Event(phase=Phase.DAY, day=1, type="speech", message="two"),
    ])
    with pytest.raises(RoomEvidenceLimitError) as oversized_error:
        manager._room_record(oversized)
    assert oversized_error.value.current == 2


@pytest.mark.asyncio
async def test_room_evidence_limit_fails_room_without_fabricating_or_dropping_rows():
    manager = RoomManager(max_evidence_entries=1)
    room = _room()
    delivered: list[dict[str, Any]] = []

    async def capture(_room: Room, payload: dict[str, Any]) -> None:
        delivered.append(dict(payload))

    manager._broadcast = capture  # type: ignore[method-assign]

    class OverflowingOrchestrator:
        async def run(self):
            manager._store_room_event(
                room,
                {"type": "phase_started", "phase": "day", "day": 1},
            )
            manager._store_room_event(
                room,
                {"type": "phase_started", "phase": "voting", "day": 1},
            )

    room.orchestrator = OverflowingOrchestrator()  # type: ignore[assignment]
    await manager._run_room(room)

    assert room.status == "failed"
    assert room.end_reason == "evidence_limit"
    assert room.error == "room evidence capacity reached"
    assert room.terminal_at is not None
    assert [event["type"] for event in room.event_history] == ["phase_started"]
    assert room.decision_trace == []
    assert room.transcript is not None
    assert len(room.transcript.entries) == 1
    assert [payload["type"] for payload in delivered] == ["game_error", "room_status"]
    assert delivered[-1]["status"] == "failed"
    budget = manager.provider_budget_ledger.snapshot(
        manager.provider_budget_ledger.room_scope(room.id)
    )
    assert budget is not None and budget.closed is True


@pytest.mark.asyncio
async def test_room_evidence_limit_survives_swallowed_event_sink_exception():
    """Game/Agent sinks may swallow observability errors; Room still aborts."""
    manager = RoomManager(max_evidence_entries=1)
    room = _room()
    delivered: list[dict[str, Any]] = []

    async def capture(_room: Room, payload: dict[str, Any]) -> None:
        delivered.append(dict(payload))

    manager._broadcast = capture  # type: ignore[method-assign]
    broadcaster = manager._make_event_broadcaster(room)

    class SwallowingOrchestrator:
        aborted = False

        async def run(self):
            await broadcaster({"type": "phase_started", "phase": "day", "day": 1})
            try:
                await broadcaster({"type": "phase_started", "phase": "voting", "day": 1})
            except RoomEvidenceLimitError:
                # Mirrors GameOrchestrator/AgentSession's defensive sink
                # boundary; RoomManager must not interpret this as success.
                pass

    room.orchestrator = SwallowingOrchestrator()  # type: ignore[assignment]
    await manager._run_room(room)

    assert room.status == "failed"
    assert room.end_reason == "evidence_limit"
    assert room.evidence_limit_error is not None
    assert room.orchestrator.aborted is True  # type: ignore[union-attr]
    assert [payload["type"] for payload in delivered] == [
        "phase_started",
        "game_error",
        "room_status",
    ]


def test_readiness_fails_when_router_contract_is_unavailable():
    manager = RoomManager()
    router = manager.router
    manager.router = object()  # type: ignore[assignment]
    try:
        assert manager.readiness() == (
            False,
            {"room_manager": "ready", "router": "unavailable"},
        )
    finally:
        manager.router = router


def test_payload_for_client_uses_harness_projection() -> None:
    manager = RoomManager()
    room = _room()
    private_payload = {
        "type": "seer_result",
        "visibility": "private",
        "recipients": [room.state.players[0].id],
        "seat": 1,
        "target_seat": 4,
        "result": "werewolves",
    }
    public_payload = {
        "type": "night_resolved",
        "deaths": [{"seat": 2, "name": "B", "reason": "wolf_kill"}],
    }
    validation_failure = {
        "type": "decision_validation_failed",
        "phase": "day",
        "seat": 2,
        "reason": "validator defect",
        "reasoning": "must not be public",
    }
    assert manager._payload_for_client(room, private_payload, 1, "play")["type"] == "seer_result"
    assert manager._payload_for_client(room, private_payload, None, "spectate") == {}
    public_visible = manager._payload_for_client(room, public_payload, None, "spectate")
    assert public_visible["deaths"] == [{"seat": 2, "name": "B"}]
    failure_visible = manager._payload_for_client(room, validation_failure, None, "spectate")
    assert failure_visible["type"] == "decision_validation_failed"
    assert "reasoning" not in failure_visible
    god_visible = manager._payload_for_client(room, public_payload, None, "god")
    assert god_visible["deaths"] == [{"seat": 2, "name": "B", "reason": "wolf_kill"}]


@pytest.mark.asyncio
async def test_delivery_cutover_orders_replay_before_concurrent_live_once() -> None:
    manager = RoomManager()
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    await manager._broadcast(
        room,
        {"type": "phase_started", "phase": "day", "day": 1},
        initial_replay=True,
    )

    socket = _FakeWebSocket(block_on_send=1)
    connect_task = asyncio.create_task(
        manager.connect(room, socket, seat=None, mode="spectate")  # type: ignore[arg-type]
    )
    await socket.send_started.wait()
    await manager._broadcast(
        room,
        {"type": "phase_started", "phase": "voting", "day": 1},
        initial_replay=True,
    )
    socket.release_send.set()
    cid = await connect_task
    await _wait_for_message_count(socket, 3)

    snapshot, replayed, live = socket.json_messages()
    assert snapshot["type"] == "snapshot"
    assert snapshot["cursor"] == 1
    assert [replayed["delivery_seq"], live["delivery_seq"]] == [1, 2]
    assert [replayed["phase"], live["phase"]] == ["day", "voting"]
    assert len({replayed["delivery_id"], live["delivery_id"]}) == 2

    manager.disconnect(room, cid)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_delivery_reconnect_since_replays_only_missing_records() -> None:
    manager = RoomManager()
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    for day in (1, 2):
        await manager._broadcast(
            room,
            {"type": "phase_started", "phase": "day", "day": day},
            initial_replay=True,
        )

    first = _FakeWebSocket()
    first_cid = await manager.connect(  # type: ignore[arg-type]
        room,
        first,
        seat=None,
        mode="spectate",
    )
    first_messages = first.json_messages()
    assert [item["delivery_seq"] for item in first_messages[1:]] == [1, 2]
    stream_id = first_messages[0]["stream_id"]
    manager.disconnect(room, first_cid)
    await asyncio.sleep(0)

    await manager._broadcast(
        room,
        {"type": "phase_started", "phase": "day", "day": 3},
        initial_replay=True,
    )
    resumed = _FakeWebSocket()
    resumed_cid = await manager.connect(  # type: ignore[arg-type]
        room,
        resumed,
        seat=None,
        mode="spectate",
        since=2,
    )
    resumed_messages = resumed.json_messages()
    assert resumed_messages[0]["stream_id"] == stream_id
    assert resumed_messages[0]["cursor"] == 3
    assert resumed_messages[0]["resumed_from"] == 2
    assert resumed_messages[0]["replay_from"] == 3
    assert resumed_messages[0]["history_gap"] is False
    assert len(resumed_messages) == 2
    assert resumed_messages[1]["delivery_seq"] == 3
    assert resumed_messages[1]["day"] == 3

    manager.disconnect(room, resumed_cid)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_delivery_cursor_rejects_future_and_retained_history_gap() -> None:
    manager = RoomManager(ws_delivery_history_size=2)
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    for day in range(1, 5):
        await manager._broadcast(
            room,
            {"type": "phase_started", "phase": "day", "day": day},
            initial_replay=True,
        )

    stale = _FakeWebSocket()
    with pytest.raises(DeliveryHistoryGapError) as gap:
        await manager.connect(  # type: ignore[arg-type]
            room,
            stale,
            seat=None,
            mode="spectate",
            since=1,
        )
    assert stale.accepted is False
    assert (gap.value.earliest, gap.value.current) == (3, 4)
    assert room.clients == {}

    future = _FakeWebSocket()
    with pytest.raises(FutureDeliveryCursorError, match="future delivery cursor"):
        await manager.connect(  # type: ignore[arg-type]
            room,
            future,
            seat=None,
            mode="spectate",
            since=5,
        )
    assert future.accepted is False
    assert room.clients == {}

    fresh = _FakeWebSocket()
    fresh_cid = await manager.connect(  # type: ignore[arg-type]
        room,
        fresh,
        seat=None,
        mode="spectate",
    )
    fresh_messages = fresh.json_messages()
    assert fresh_messages[0]["history_gap"] is True
    assert fresh_messages[0]["replay_from"] == 3
    assert [item["delivery_seq"] for item in fresh_messages[1:]] == [3, 4]
    manager.disconnect(room, fresh_cid)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_private_delivery_does_not_create_public_sequence_gap() -> None:
    manager = RoomManager()
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats={1},
    )
    recipient = room.state.players[0].id
    await manager._broadcast(
        room,
        {"type": "phase_started", "phase": "day", "day": 1},
        initial_replay=True,
    )
    await manager._broadcast(
        room,
        {
            "type": "seer_result",
            "seat": 1,
            "target_seat": 2,
            "result": "werewolves",
            "visibility": "private",
            "recipients": [recipient],
        },
        initial_replay=True,
    )
    await manager._broadcast(
        room,
        {"type": "phase_started", "phase": "voting", "day": 1},
        initial_replay=True,
    )

    public = list(room.delivery_streams["spectate"].history)
    player = list(room.delivery_streams["play:1"].history)
    god = list(room.delivery_streams["god"].history)
    assert [record.seq for record in public] == [1, 2]
    assert [record.payload["type"] for record in public] == ["phase_started", "phase_started"]
    assert [record.seq for record in player] == [1, 2, 3]
    assert player[1].payload["type"] == "seer_result"
    assert [record.seq for record in god] == [1, 2, 3]
    assert all("visibility" not in record.payload for record in public)
    assert all("recipients" not in record.payload for record in public)


@pytest.mark.asyncio
async def test_slow_client_is_disconnected_without_blocking_fast_client() -> None:
    manager = RoomManager(ws_client_queue_size=1)
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    slow = _FakeWebSocket(block_on_send=2)
    fast = _FakeWebSocket()
    slow_cid = await manager.connect(  # type: ignore[arg-type]
        room,
        slow,
        seat=None,
        mode="spectate",
    )
    fast_cid = await manager.connect(  # type: ignore[arg-type]
        room,
        fast,
        seat=None,
        mode="spectate",
    )

    await manager._broadcast(
        room,
        {"type": "phase_started", "phase": "day", "day": 1},
        initial_replay=True,
    )
    await slow.send_started.wait()
    await _wait_for_message_count(fast, 2)
    await manager._broadcast(
        room,
        {"type": "phase_started", "phase": "day", "day": 2},
        initial_replay=True,
    )
    await _wait_for_message_count(fast, 3)
    await manager._broadcast(
        room,
        {"type": "phase_started", "phase": "day", "day": 3},
        initial_replay=True,
    )
    await _wait_for_message_count(fast, 4)
    await asyncio.sleep(0)

    assert slow_cid not in room.clients
    assert slow.close_calls == [(4410, "client too slow")]
    assert [item["delivery_seq"] for item in fast.json_messages()[1:]] == [1, 2, 3]
    assert fast_cid in room.clients

    manager.disconnect(room, fast_cid)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_room_websocket_client_capacity_is_bounded_and_reclaimable() -> None:
    manager = RoomManager(max_ws_clients_per_room=1)
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    first = _FakeWebSocket()
    first_cid = await manager.connect(
        room,
        first,  # type: ignore[arg-type]
        seat=None,
        mode="spectate",
    )

    denied = _FakeWebSocket()
    with pytest.raises(RoomClientCapacityError, match="capacity reached"):
        await manager.connect(
            room,
            denied,  # type: ignore[arg-type]
            seat=None,
            mode="spectate",
        )
    assert denied.accepted is False
    assert list(room.clients) == [first_cid]

    manager.disconnect(room, first_cid)
    await asyncio.sleep(0)
    replacement = _FakeWebSocket()
    replacement_cid = await manager.connect(
        room,
        replacement,  # type: ignore[arg-type]
        seat=None,
        mode="spectate",
    )
    assert replacement.accepted is True
    manager.disconnect(room, replacement_cid)
    await asyncio.sleep(0)


def test_set_seat_model_config_allows_endpoint_change_without_explicit_key():
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
        "model": "seat-model",
        "api_base": "https://attacker.invalid/v1",
    })

    cfg = room.seat_configs[1]
    assert cfg.model == "seat-model"
    assert cfg.api_base == "https://attacker.invalid/v1"
    assert cfg.api_key == ""


def test_set_seat_model_config_rejects_unknown_seat():
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=_model_config(),
    )

    with pytest.raises(ValueError, match="座位不存在: 99"):
        manager.set_seat_model_config(room, 99, {"model": "seat-model"})


def test_set_seat_model_config_rejects_human_seat_override():
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats={1},
    )

    with pytest.raises(ValueError, match="人类席位"):
        manager.set_seat_model_config(room, 1, {"model": "seat-model"})

    assert 1 not in room.seat_configs
    assert room.status == "waiting"


def test_set_seat_model_config_deep_copies_input():
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    supplied = ModelConfig(
        model="seat-model",
        api_key="seat-key",
        reasoning={"effort": "high"},
    )

    manager.set_seat_model_config(room, 1, supplied)
    stored = room.seat_configs[1]
    assert stored is not supplied

    supplied.model = "mutated-model"
    assert supplied.reasoning is not None
    supplied.reasoning["effort"] = "low"
    assert stored.model == "seat-model"
    assert stored.reasoning == {"effort": "high"}


@pytest.mark.parametrize("field_name", ["extra_body", "reasoning_effort", "top_k"])
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


@pytest.mark.parametrize(("field_name", "value"), [("top_k", None), ("reasoning_effort", "")])
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


def test_set_seat_model_config_accepts_standard_reasoning_and_thinking_fields():
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=_model_config(),
    )

    manager.set_seat_model_config(room, 1, {
        "provider": "anthropic",
        "model": "seat-model",
        "reasoning": {"effort": "high", "summary": "auto"},
        "thinking": {"type": "enabled", "budget_tokens": 2048},
    })

    cfg = room.seat_configs[1]
    assert cfg.reasoning == {"effort": "high", "summary": "auto"}
    assert cfg.thinking == {"type": "enabled", "budget_tokens": 2048}


@pytest.mark.parametrize("provider", ["openai_responses", "anthropic"])
def test_set_seat_model_config_allows_provider_change_without_explicit_key(provider):
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
    })

    cfg = room.seat_configs[1]
    assert cfg.provider == provider
    assert cfg.model == "seat-model"
    assert cfg.api_key == ""


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
async def test_start_game_uses_detached_model_config_snapshot(monkeypatch):
    manager = RoomManager(room_timeout=1.0)

    async def no_run(_room: Room) -> None:
        return None

    monkeypatch.setattr(manager, "_run_room", no_run)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=ModelConfig(
            provider="openai",
            model="default-before-build",
            api_base="https://example.invalid/v1",
            api_key="unit-test-key",
        ),
    )
    manager.set_seat_model_config(room, 1, {"model": "seat-before-build"})

    original_build_actors = room_manager_module.build_actors

    def mutate_room_configs_then_build(*args, **kwargs):
        # Simulate a concurrent setup/configuration writer racing with the
        # actor construction boundary. The captured start snapshot must win.
        assert room.default_config is not None
        room.default_config.model = "late-default-mutation"
        room.seat_configs[1].model = "late-seat-mutation"
        return original_build_actors(*args, **kwargs)

    monkeypatch.setattr(room_manager_module, "build_actors", mutate_room_configs_then_build)

    await manager.start_game(room)
    try:
        assert room.run_spec is not None
        assert room.run_spec.default_model is not None
        assert room.run_spec.default_model.model == "default-before-build"
        assert room.run_spec.seat_models[1].model == "seat-before-build"
        assert room.actors[room.state.players[0].id].model_config.model == "seat-before-build"
        assert room.actors[room.state.players[1].id].model_config.model == "default-before-build"
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
    }
    monkeypatch.setattr(room_manager_module, "AGENT_PHASE_DEADLINE", 0.0)
    monkeypatch.setattr(room_manager_module, "AGENT_PHASE_DEADLINE_BY_PHASE", phase_deadlines)

    async def no_run(_room: Room) -> None:
        return None

    monkeypatch.setattr(manager, "_run_room", no_run)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=_model_config(),
    )

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
async def test_interactive_bindings_and_action_requests_use_one_run_id(monkeypatch):
    manager = RoomManager(room_timeout=1.0)

    async def no_run(_room: Room) -> None:
        return None

    monkeypatch.setattr(manager, "_run_room", no_run)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats=set(range(1, 7)),
    )

    await manager.start_game(room)
    try:
        assert room.state.id == room.id
        assert room.run_spec is not None
        assert room.run_spec.run_id == room.id
        assert room.core_run_spec is not None
        assert room.core_run_spec.run_id == room.id
        assert room.core_run_spec.actors.human_actor_ids == [
            f"seat:{seat}" for seat in range(1, 7)
        ]
        assert room.transcript is not None
        assert room.transcript.run_id == room.id
        assert room.transcript.metadata["run_spec_hash"] == room.core_run_spec.spec_hash
        assert room.transcript.metadata["legacy_run_spec_hash"] == room.run_spec.spec_hash
        assert len({id(actor) for actor in room.actors.values()}) == 6
        assert len({id(actor.memory) for actor in room.actors.values()}) == 6
        binding_rows = [
            entry.payload
            for entry in room.transcript.entries
            if entry.kind == "harness"
            and entry.payload.get("type") == "agent_bindings_finalized"
        ]
        assert binding_rows == [{
            "type": "agent_bindings_finalized",
            "actor_count": 6,
            "actor_ids": [f"seat:{seat}" for seat in range(1, 7)],
        }]

        requests: list[ActionRequest] = []
        for player in room.state.players:
            actor = room.actors[player.id]

            async def decide(request: ActionRequest) -> DecisionEnvelope:
                requests.append(request)
                return DecisionEnvelope(
                    request_id=request.request_id,
                    seat=request.seat,
                    decision=Decision(
                        action=AgentAction.SPEAK,
                        speech=f"seat {request.seat}",
                        bid=1,
                    ),
                    parse_status="not_applicable",
                )

            actor.decide = decide  # type: ignore[method-assign]
            await room.orchestrator._request_agent_decision(  # type: ignore[union-attr]
                actor,
                player.id,
                action_kind="speak",
                phase="night",
            )

        assert len(requests) == len(room.state.players)
        assert {request.run_id for request in requests} == {room.id}
        assert all(request.run_id == room.state.id for request in requests)
    finally:
        assert room.task is not None
        await room.task


@pytest.mark.asyncio
async def test_started_room_executes_through_one_core_owned_lifecycle(monkeypatch):
    manager = RoomManager(room_timeout=1.0)
    original_create_session = WerewolfEnvironmentPlugin.create_session

    async def create_completed_session(plugin, context):
        session = await original_create_session(plugin, context)

        async def complete_without_decisions():
            session.state.phase = Phase.ENDED
            session.state.winner = Team.VILLAGE
            session.orchestrator.termination_status = "completed"
            return EnvironmentOutcome(
                terminal=True,
                status="completed",
                outcome={"winner": "village", "days": session.state.day},
            )

        session.run = complete_without_decisions
        return session

    monkeypatch.setattr(
        WerewolfEnvironmentPlugin,
        "create_session",
        create_completed_session,
    )
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats=set(range(1, 7)),
    )

    await manager.start_game(room)
    assert room.task is not None
    await room.task

    assert room.status == "ended"
    assert room.end_reason == "completed"
    assert room.prepared_run is None
    assert room.core_result is not None
    assert room.core_run_spec is not None
    assert room.core_result.status == "completed"
    assert room.core_result.run_spec_hash == room.core_run_spec.spec_hash
    assert room.core_result.harness_metrics["resolved_actor_ids"] == [
        f"seat:{seat}" for seat in range(1, 7)
    ]
    assert room.transcript is not None
    harness_types = [
        entry.payload.get("type")
        for entry in room.transcript.entries
        if entry.kind == "harness"
    ]
    assert harness_types.count("run_started") == 1
    assert harness_types.count("agent_bindings_finalized") == 1
    assert harness_types.count("run_completed") == 1


@pytest.mark.asyncio
async def test_started_room_timeout_is_owned_and_normalized_by_core(monkeypatch):
    manager = RoomManager(
        room_timeout=0.02,
        cancellation_grace_seconds=0.02,
        cleanup_timeout_seconds=0.1,
    )
    original_create_session = WerewolfEnvironmentPlugin.create_session

    async def create_blocked_session(plugin, context):
        session = await original_create_session(plugin, context)

        async def block():
            await asyncio.sleep(3600)

        session.run = block
        return session

    monkeypatch.setattr(
        WerewolfEnvironmentPlugin,
        "create_session",
        create_blocked_session,
    )
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats=set(range(1, 7)),
    )

    await manager.start_game(room)
    assert room.task is not None
    await room.task

    assert room.status == "timeout"
    assert room.end_reason == "timeout"
    assert room.prepared_run is None
    assert room.core_result is not None
    assert room.core_result.status == "timed_out"
    assert room.core_result.error_type == "RunTimeout"
    assert room.core_result.harness_metrics["cleanup_failure_count"] == 0


@pytest.mark.asyncio
async def test_started_room_cancellation_closes_core_prepared_resources(monkeypatch):
    manager = RoomManager(room_timeout=10.0)
    original_create_session = WerewolfEnvironmentPlugin.create_session
    entered = asyncio.Event()

    async def create_blocked_session(plugin, context):
        session = await original_create_session(plugin, context)

        async def block():
            entered.set()
            await asyncio.sleep(3600)

        session.run = block
        return session

    monkeypatch.setattr(
        WerewolfEnvironmentPlugin,
        "create_session",
        create_blocked_session,
    )
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats=set(range(1, 7)),
    )

    await manager.start_game(room)
    assert room.task is not None
    await entered.wait()
    room.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await room.task

    assert room.status == "cancelled"
    assert room.end_reason == "cancelled"
    assert room.prepared_run is None
    assert room.core_result is None
    assert room.transcript is not None
    assert any(
        entry.kind == "harness" and entry.payload.get("type") == "run_cancelled"
        for entry in room.transcript.entries
    )


@pytest.mark.asyncio
async def test_start_game_rejects_live_actor_kind_mismatch(monkeypatch):
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats=set(range(1, 7)),
    )
    state_before_start = room.state.model_copy(deep=True)
    original_build_actors = room_manager_module.build_actors

    def build_with_wrong_kind(*args, **kwargs):
        actors = original_build_actors(*args, **kwargs)
        actors[room.state.players[0].id].is_human = False
        return actors

    monkeypatch.setattr(room_manager_module, "build_actors", build_with_wrong_kind)

    with pytest.raises(ValueError, match="kind does not match ActorSpec for seat:1"):
        await manager.start_game(room)

    assert room.status == "waiting"
    assert room.task is None
    assert room.state == state_before_start
    assert room.state.phase == Phase.SETUP
    assert all(player.role is None for player in room.state.players)
    assert room.run_spec is None
    assert room.transcript is None
    assert room.actors == {}
    assert room.orchestrator is None


@pytest.mark.asyncio
async def test_start_game_rejects_live_actor_manifest_mismatch(monkeypatch):
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=_model_config(),
    )
    state_before_start = room.state.model_copy(deep=True)
    original_build_actors = room_manager_module.build_actors

    def build_with_wrong_manifest(*args, **kwargs):
        actors = original_build_actors(*args, **kwargs)
        actors[room.state.players[0].id].model_config.model = "unattested-model"
        return actors

    monkeypatch.setattr(room_manager_module, "build_actors", build_with_wrong_manifest)

    with pytest.raises(ValueError, match="does not match ActorSpec for seat:1"):
        await manager.start_game(room)

    assert room.status == "waiting"
    assert room.task is None
    assert room.state == state_before_start
    assert room.state.phase == Phase.SETUP
    assert all(player.role is None for player in room.state.players)
    assert room.run_spec is None
    assert room.transcript is None
    assert room.actors == {}
    assert room.orchestrator is None


@pytest.mark.asyncio
async def test_start_game_staged_evidence_overflow_leaves_waiting_room_untouched():
    manager = RoomManager(room_timeout=1.0, max_evidence_entries=1)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats=set(range(1, 7)),
    )
    state_before_start = room.state.model_copy(deep=True)

    with pytest.raises(RoomEvidenceLimitError):
        await manager.start_game(room)

    assert room.status == "waiting"
    assert room.state == state_before_start
    assert room.state.phase == Phase.SETUP
    assert all(player.role is None for player in room.state.players)
    assert room.run_spec is None
    assert room.transcript is None
    assert room.actors == {}
    assert room.orchestrator is None
    assert room.task is None


@pytest.mark.asyncio
async def test_start_game_durable_commit_failure_rolls_back_before_publish(monkeypatch):
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats=set(range(1, 7)),
    )
    state_before_start = room.state.model_copy(deep=True)

    def fail_persist(_room: Room) -> None:
        raise RuntimeError("injected persistence failure")

    monkeypatch.setattr(manager, "_persist_room", fail_persist)
    with pytest.raises(RuntimeError, match="injected persistence failure"):
        await manager.start_game(room)

    assert room.status == "waiting"
    assert room.state == state_before_start
    assert room.run_spec is None
    assert room.transcript is None
    assert room.actors == {}
    assert room.orchestrator is None
    assert room.task is None


@pytest.mark.asyncio
async def test_start_game_publish_failure_becomes_terminal_without_starting_task(monkeypatch):
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats=set(range(1, 7)),
    )

    async def fail_broadcast(_room: Room) -> None:
        raise RuntimeError("injected running-status failure")

    monkeypatch.setattr(manager, "_broadcast_room_status", fail_broadcast)
    with pytest.raises(RuntimeError, match="injected running-status failure"):
        await manager.start_game(room)

    assert room.status == "failed"
    assert room.end_reason == "startup_failed"
    assert room.error == "RuntimeError during room startup"
    assert room.task is None
    assert room.run_spec is not None
    assert room.transcript is not None
    assert room.orchestrator is not None
    assert room.event_history[-1]["type"] == "room_status"
    assert room.event_history[-1]["status"] == "failed"
    assert room.event_history[-1]["reason"] == "startup_failed"
    assert room.transcript.entries[-1].payload["status"] == "failed"


@pytest.mark.asyncio
async def test_start_game_broadcasts_running_status(monkeypatch):
    manager, events = _recording_manager(room_timeout=1.0)

    async def no_run(_room: Room) -> None:
        return None

    monkeypatch.setattr(manager, "_run_room", no_run)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=_model_config(),
    )

    await manager.start_game(room)
    try:
        assert room.status == "running"
        assert events[0] == {"type": "room_status", "status": "running", "reason": None}
    finally:
        assert room.task is not None
        await room.task


@pytest.mark.asyncio
async def test_start_game_rejects_ai_room_without_real_model_config():
    manager = RoomManager(room_timeout=1.0)
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])

    with pytest.raises(ValueError, match="模型配置不完整"):
        await manager.start_game(room)

    assert room.status == "waiting"
    assert room.task is None


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
async def test_run_room_preserves_incomplete_guard_status_without_fake_error():
    manager, events = _recording_manager()
    room = _room()
    room.orchestrator = _IncompleteOrchestrator(room)  # type: ignore[assignment]

    await manager._run_room(room)

    assert room.status == "incomplete"
    assert room.end_reason == "max_game_rounds"
    assert room.error is None
    assert not any(event.get("type") == "game_error" for event in events)
    assert events[-1] == {
        "type": "room_status",
        "status": "incomplete",
        "reason": "max_game_rounds",
    }


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
    assert [item["type"] for item in room.event_history[-2:]] == [
        "game_error",
        "room_status",
    ]
    assert room.transcript is not None
    assert [entry.payload["type"] for entry in room.transcript.entries[-2:]] == [
        "game_error",
        "room_status",
    ]


@pytest.mark.asyncio
async def test_orchestrator_internal_timeout_error_is_not_room_deadline():
    manager, events = _recording_manager(room_timeout=1.0)
    room = _room()
    room.orchestrator = _InternalTimeoutOrchestrator()  # type: ignore[assignment]

    await manager._run_room(room)

    assert room.status == "failed"
    assert room.end_reason == "error"
    assert room.error == "TimeoutError during game loop"
    assert not any(event.get("reason") == "timeout" for event in events)
    assert room.event_history[-1]["type"] == "room_status"
    assert room.transcript is not None
    assert room.transcript.entries[-1].payload["status"] == "failed"


@pytest.mark.asyncio
async def test_room_deadline_quarantines_orchestrator_that_ignores_cancellation():
    manager, events = _recording_manager(room_timeout=0.01)
    manager.cancellation_grace_seconds = 0.02
    room = _room()
    orchestrator = _IgnoresCancellationOrchestrator()
    room.orchestrator = orchestrator  # type: ignore[assignment]

    started = time.monotonic()
    try:
        await manager._run_room(room)
        assert time.monotonic() - started < 0.5
        assert room.status == "failed"
        assert room.end_reason == "cleanup_failure"
        assert manager.unresolved_cleanup_task_count == 1
        cleanup = next(
            item for item in room.event_history if item["type"] == "room_cleanup_failed"
        )
        assert cleanup["error_type"] == "TaskIgnoredCancellation"
        assert cleanup["pending_task_count"] == 1
        assert cleanup["fatal"] is True
        assert room.transcript is not None
        assert any(
            entry.payload.get("type") == "room_cleanup_failed"
            for entry in room.transcript.entries
        )
        assert events[-1]["status"] == "failed"
    finally:
        orchestrator.release.set()
        for _ in range(100):
            if manager.unresolved_cleanup_task_count == 0:
                break
            await asyncio.sleep(0)
        assert manager.unresolved_cleanup_task_count == 0


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


def test_room_capacity_never_evicts_running_room():
    manager = RoomManager(max_rooms=1, terminal_room_ttl=1.0)
    running = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    running.status = "running"

    with pytest.raises(RoomCapacityError, match="capacity"):
        manager.create_room(player_names=["G", "H", "I", "J", "K", "L"])

    assert manager.rooms == {running.id: running}


def test_expired_terminal_room_is_reclaimed_before_capacity_check():
    manager = RoomManager(max_rooms=1, terminal_room_ttl=10.0)
    ended = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    ended.status = "ended"
    ended.terminal_at = 100.0

    assert manager.cleanup_expired_rooms(now=111.0) == [ended.id]
    replacement = manager.create_room(player_names=["G", "H", "I", "J", "K", "L"])

    assert replacement.id in manager.rooms
    assert ended.id not in manager.rooms


def test_terminal_ttl_does_not_remove_room_with_active_client():
    manager = RoomManager(max_rooms=1, terminal_room_ttl=10.0)
    ended = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    ended.status = "ended"
    ended.terminal_at = 100.0
    ended.clients["client"] = (object(), None, "spectate")  # type: ignore[assignment]

    assert manager.cleanup_expired_rooms(now=111.0) == []
    assert manager.rooms[ended.id] is ended


@pytest.mark.asyncio
async def test_terminal_ttl_does_not_remove_room_with_active_task():
    manager = RoomManager(max_rooms=1, terminal_room_ttl=10.0)
    ended = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    ended.status = "ended"
    ended.terminal_at = 100.0
    ended.task = asyncio.create_task(asyncio.sleep(3600))
    try:
        assert manager.cleanup_expired_rooms(now=111.0) == []
        assert manager.rooms[ended.id] is ended
    finally:
        ended.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ended.task


def test_explicit_cleanup_removes_waiting_room_and_rejects_running_room():
    manager = RoomManager(max_rooms=2)
    waiting = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    running = manager.create_room(player_names=["G", "H", "I", "J", "K", "L"])
    running.status = "running"

    assert manager.delete_room(waiting.id) is waiting
    assert waiting.id not in manager.rooms
    with pytest.raises(RoomInUseError, match="running"):
        manager.delete_room(running.id)
    assert manager.rooms[running.id] is running


@pytest.mark.asyncio
async def test_manager_readiness_tracks_router_close_lifecycle():
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class LifecycleRouter:
        async def complete_json(self, *args, **kwargs):  # pragma: no cover - contract only
            raise AssertionError("not called")

        async def aclose(self):
            close_started.set()
            await allow_close.wait()

    manager = RoomManager(router=LifecycleRouter())  # type: ignore[arg-type]
    assert manager.readiness() == (
        True,
        {"room_manager": "ready", "router": "ready"},
    )

    close_task = asyncio.create_task(manager.aclose())
    await close_started.wait()
    assert manager.closing is True
    assert manager.readiness() == (
        False,
        {"room_manager": "closing", "router": "ready"},
    )

    allow_close.set()
    await close_task
    assert manager.closed is True
    assert manager.readiness() == (
        False,
        {"room_manager": "closed", "router": "closed"},
    )
    with pytest.raises(RoomManagerUnavailableError):
        manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])


@pytest.mark.asyncio
async def test_failed_router_close_is_not_reported_ready():
    class BrokenCloseRouter:
        async def complete_json(self, *args, **kwargs):  # pragma: no cover - contract only
            raise AssertionError("not called")

        async def aclose(self):
            raise RuntimeError("close failed")

    manager = RoomManager(router=BrokenCloseRouter())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="close failed"):
        await manager.aclose()

    assert manager.readiness() == (
        False,
        {"room_manager": "closed", "router": "unavailable"},
    )


@pytest.mark.asyncio
async def test_shutdown_marks_not_ready_before_waiting_for_room_start(monkeypatch):
    manager = RoomManager(room_timeout=10.0)
    broadcast_started = asyncio.Event()
    allow_broadcast = asyncio.Event()

    async def blocked_broadcast(_room: Room, _payload: dict[str, Any]) -> None:
        broadcast_started.set()
        await allow_broadcast.wait()

    async def run_until_cancelled(_room: Room) -> None:
        await asyncio.sleep(3600)

    monkeypatch.setattr(manager, "_broadcast", blocked_broadcast)
    monkeypatch.setattr(manager, "_run_room", run_until_cancelled)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats={1, 2, 3, 4, 5, 6},
    )

    start_task = asyncio.create_task(manager.start_game(room))
    await broadcast_started.wait()
    prepared = room.prepared_run
    assert prepared is not None
    cleanup_calls: list[str] = []
    original_session_close = prepared.session.aclose
    original_runtime_close = prepared.decision_runtime.aclose

    async def close_session():
        cleanup_calls.append("session")
        await original_session_close()

    async def close_runtime():
        cleanup_calls.append("runtime")
        await original_runtime_close()

    prepared.session.aclose = close_session
    prepared.decision_runtime.aclose = close_runtime
    close_task = asyncio.create_task(manager.aclose())
    await asyncio.sleep(0)

    assert manager.closing is True
    assert manager.readiness()[0] is False
    with pytest.raises(RoomManagerUnavailableError):
        manager.create_room(player_names=["G", "H", "I", "J", "K", "L"])

    allow_broadcast.set()
    await start_task
    await close_task
    assert room.task is not None and room.task.cancelled()
    assert room.prepared_run is None
    assert cleanup_calls == ["session", "runtime"]
    assert manager.closed is True


@pytest.mark.asyncio
async def test_shutdown_quarantines_room_task_that_ignores_cancellation():
    manager = RoomManager(
        room_timeout=10.0,
        cleanup_timeout_seconds=0.02,
        cancellation_grace_seconds=0.02,
    )
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    room.status = "running"
    release = asyncio.Event()

    async def ignore_cancellation() -> None:
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue

    room.task = asyncio.create_task(ignore_cancellation())
    await asyncio.sleep(0)
    started = time.monotonic()
    try:
        with pytest.raises(RoomManagerCleanupError) as raised:
            await manager.aclose()
        assert time.monotonic() - started < 0.5
        assert manager.closed is True
        assert raised.value.pending_task_count >= 1
        assert room.status == "failed"
        assert room.end_reason == "cleanup_failure"
        assert any(
            event["type"] == "room_cleanup_failed"
            and event["error_type"] == "TaskIgnoredCancellation"
            for event in room.event_history
        )
        assert room.transcript is not None
        assert room.transcript.entries[-1].payload["status"] == "failed"
    finally:
        release.set()
        for _ in range(100):
            if manager.unresolved_cleanup_task_count == 0:
                break
            await asyncio.sleep(0)
        assert manager.unresolved_cleanup_task_count == 0


@pytest.mark.asyncio
async def test_shutdown_surfaces_orchestrator_child_quarantined_during_room_cancel():
    manager = RoomManager(
        room_timeout=10.0,
        cleanup_timeout_seconds=0.02,
        cancellation_grace_seconds=0.02,
    )
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    room.status = "running"
    orchestrator = _IgnoresCancellationOrchestrator()
    room.orchestrator = orchestrator  # type: ignore[assignment]
    room.task = asyncio.create_task(manager._run_room(room))
    await asyncio.sleep(0)

    try:
        with pytest.raises(RoomManagerCleanupError) as raised:
            await manager.aclose()
        assert raised.value.pending_task_count >= 1
        assert room.status == "failed"
        assert room.end_reason == "cleanup_failure"
        assert manager.unresolved_cleanup_task_count == 1
        assert any(
            failure.get("stage") == "orchestrator_run"
            and failure.get("pending_task_count") == 1
            for failure in raised.value.failures
        )
    finally:
        orchestrator.release.set()
        for _ in range(100):
            if manager.unresolved_cleanup_task_count == 0:
                break
            await asyncio.sleep(0)
        assert manager.unresolved_cleanup_task_count == 0


@pytest.mark.asyncio
async def test_provider_scope_cleanup_failure_replaces_success_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = RoomManager()
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    room.status = "ended"
    room.end_reason = "completed"

    def fail_scope_close(_room: Room) -> None:
        raise RuntimeError("opaque scope close failure")

    monkeypatch.setattr(manager, "_close_provider_scope", fail_scope_close)
    await manager._broadcast_room_status(room)

    assert room.status == "failed"
    assert room.end_reason == "cleanup_failure"
    assert room.error == "provider budget scope cleanup failed"
    assert [event["type"] for event in room.event_history[-4:]] == [
        "room_status",
        "room_cleanup_failed",
        "game_error",
        "room_status",
    ]
    assert room.event_history[-1]["status"] == "failed"
    assert room.transcript is not None
    assert room.transcript.entries[-1].payload["status"] == "failed"


@pytest.mark.asyncio
async def test_shutdown_bounds_router_that_ignores_cancellation():
    class StubbornRouter:
        def __init__(self) -> None:
            self.release = asyncio.Event()

        async def complete_json(self, *args, **kwargs):  # pragma: no cover - contract only
            raise AssertionError("not called")

        async def aclose(self) -> None:
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    continue

    router = StubbornRouter()
    manager = RoomManager(
        router=router,  # type: ignore[arg-type]
        cleanup_timeout_seconds=0.02,
        cancellation_grace_seconds=0.02,
    )
    started = time.monotonic()
    try:
        with pytest.raises(RoomManagerCleanupError) as raised:
            await manager.aclose()
        assert time.monotonic() - started < 0.5
        assert manager.closed is True
        assert manager.readiness()[1]["router"] == "unavailable"
        assert raised.value.failures[-1]["stage"] == "router_close"
        assert raised.value.failures[-1]["error_type"] == "TaskIgnoredCancellation"
        assert manager.unresolved_cleanup_task_count == 1
    finally:
        router.release.set()
        for _ in range(100):
            if manager.unresolved_cleanup_task_count == 0:
                break
            await asyncio.sleep(0)
        assert manager.unresolved_cleanup_task_count == 0


@pytest.mark.asyncio
async def test_shutdown_bounds_synchronous_persistence_close():
    release = threading.Event()

    class BlockingPersistence:
        def save_record(self, _record) -> None:
            return None

        def load_records(self) -> list[dict[str, Any]]:
            return []

        def delete_room(self, _room_id: str) -> None:
            return None

        def close(self) -> None:
            release.wait()

    manager = RoomManager(
        persistence=BlockingPersistence(),  # type: ignore[arg-type]
        cleanup_timeout_seconds=0.02,
        cancellation_grace_seconds=0.02,
    )
    started = time.monotonic()
    try:
        with pytest.raises(RoomManagerCleanupError) as raised:
            await manager.aclose()
        assert time.monotonic() - started < 0.5
        assert manager.closed is True
        assert raised.value.failures[-1] == {
            "stage": "persistence_close",
            "error_type": "CleanupWorkerTimeout",
            "timeout": True,
            "pending_task_count": 1,
            "fatal": True,
        }
        assert manager.unresolved_cleanup_task_count == 1
    finally:
        release.set()
        for _ in range(100):
            if manager.unresolved_cleanup_task_count == 0:
                break
            await asyncio.sleep(0.001)
        assert manager.unresolved_cleanup_task_count == 0


@pytest.mark.asyncio
async def test_shutdown_cancels_writer_and_closes_websocket():
    manager = RoomManager(
        cleanup_timeout_seconds=0.05,
        cancellation_grace_seconds=0.02,
    )
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    socket = _FakeWebSocket()
    cid = await manager.connect(room, socket, seat=None, mode="spectate")
    connection = room.clients[cid]
    assert connection.writer_task is not None

    await manager.aclose()

    assert manager.closed is True
    assert room.clients == {}
    assert connection.writer_task.cancelled()
    assert socket.close_calls[-1] == (1001, "server shutdown")


@pytest.mark.asyncio
async def test_shutdown_cancels_blocked_websocket_handshake():
    manager = RoomManager(
        cleanup_timeout_seconds=0.05,
        cancellation_grace_seconds=0.02,
    )
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    socket = _FakeWebSocket(block_on_send=1)
    connect_task = asyncio.create_task(
        manager.connect(room, socket, seat=None, mode="spectate")
    )
    await socket.send_started.wait()

    await manager.aclose()

    assert connect_task.cancelled()
    assert room.clients == {}
    assert socket.close_calls[-1] == (1001, "server shutdown")
