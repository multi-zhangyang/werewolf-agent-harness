"""Focused durable-evidence transaction tests for the interactive room host.

These tests intentionally stay below the HTTP/browser and provider layers.  A
room-owned sink is the commit boundary for each source, so failures must leave
the in-memory source, transcript, delivery cursor, and client queue coherent.
"""
from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace
from typing import Any

import pytest

from src.api.persistence import PersistenceError
from src.api.room_manager import Room, RoomClient, RoomManager
from src.game.models import Phase
from src.game.roles import Team
from src.game.state import new_game
from src.harness.transcript import Transcript


_NAMES = ["A", "B", "C", "D", "E", "F"]


def _room(manager: RoomManager, room_id: str = "durable-transaction") -> Room:
    room = Room(
        id=room_id,
        state=new_game(_NAMES, game_id=room_id),
        status="running",
    )
    room.transcript = Transcript(run_id=room.id)
    manager._initialize_delivery_streams(room)
    return room


def _delivery_snapshot(room: Room) -> dict[str, Any]:
    with room.delivery_lock:
        return {
            "source_seq": room.delivery_source_seq,
            "source_history": copy.deepcopy(list(room.delivery_source_history)),
            "streams": {
                key: {
                    "stream_id": stream.stream_id,
                    "cursor": stream.cursor,
                    "history": copy.deepcopy(list(stream.history)),
                    "history_gap": stream.history_gap,
                }
                for key, stream in sorted(room.delivery_streams.items())
            },
        }


def _spectator_client(room: Room) -> asyncio.Queue[str]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str] = asyncio.Queue()
    room.clients["spectator"] = RoomClient(
        websocket=object(),  # type: ignore[arg-type]
        seat=None,
        mode="spectate",
        stream_key="spectate",
        queue=queue,
        loop=loop,
    )
    return queue


@pytest.mark.asyncio
async def test_event_full_sink_failure_rolls_back_all_sources_and_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = RoomManager()
    room = _room(manager)
    queue = _spectator_client(room)
    emit_event = manager._make_event_broadcaster(room)

    # Establish a committed prefix so the failed append must restore an
    # existing transcript/source identity, rather than merely deleting a new
    # object.
    await emit_event({"type": "phase_started", "phase": "day", "day": 1})
    assert queue.qsize() == 1
    queue.get_nowait()
    before_events = copy.deepcopy(room.event_history)
    before_transcript = room.transcript
    assert before_transcript is not None
    before_transcript_export = before_transcript.export()
    before_trace_seq = room.trace_seq
    before_delivery = _delivery_snapshot(room)

    def fail_persist(_room: Room) -> None:
        raise PersistenceError("injected event persistence failure")

    monkeypatch.setattr(manager, "_persist_room", fail_persist)
    with pytest.raises(PersistenceError, match="injected event persistence failure"):
        await emit_event({"type": "speech", "seat": 1, "message": "second"})

    assert room.event_history == before_events
    assert room.transcript is before_transcript
    assert room.transcript.export() == before_transcript_export
    assert room.trace_seq == before_trace_seq
    assert _delivery_snapshot(room) == before_delivery
    assert queue.empty(), "a failed durable append must not expose a live message"
    assert room.evidence_sink_error_type == "PersistenceError"


def test_trace_sink_failure_rolls_back_decision_transcript_and_trace_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = RoomManager()
    room = _room(manager, "trace-transaction")
    record_trace = manager._make_trace_recorder(room)
    record_trace({"kind": "baseline_decision", "seat": 1})

    before_trace = copy.deepcopy(room.decision_trace)
    before_transcript = room.transcript
    assert before_transcript is not None
    before_transcript_export = before_transcript.export()
    before_trace_seq = room.trace_seq

    def fail_persist(_room: Room) -> None:
        raise PersistenceError("injected trace persistence failure")

    monkeypatch.setattr(manager, "_persist_room", fail_persist)
    with pytest.raises(PersistenceError, match="injected trace persistence failure"):
        record_trace({"kind": "second_decision", "seat": 2})

    assert room.decision_trace == before_trace
    assert room.transcript is before_transcript
    assert room.transcript.export() == before_transcript_export
    assert room.trace_seq == before_trace_seq
    assert room.event_history == []
    assert room.evidence_sink_error_type == "PersistenceError"


def test_harness_ordinary_sink_failure_rolls_back_transcript_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = RoomManager()
    room = _room(manager, "harness-transaction")
    record_harness = manager._make_harness_recorder(room)
    record_harness({
        "type": "run_started",
        "environment_id": "test-env",
        "environment_version": "1",
    })
    before_transcript = room.transcript
    assert before_transcript is not None
    before_transcript_export = before_transcript.export()
    before_status = (room.status, room.end_reason, room.error)
    before_delivery = _delivery_snapshot(room)

    def fail_persist(_room: Room) -> None:
        raise PersistenceError("injected harness persistence failure")

    monkeypatch.setattr(manager, "_persist_room", fail_persist)
    with pytest.raises(PersistenceError, match="injected harness persistence failure"):
        record_harness({"type": "agent_bindings_finalized", "actor_count": 6})

    assert room.transcript is before_transcript
    assert room.transcript.export() == before_transcript_export
    assert (room.status, room.end_reason, room.error) == before_status
    assert _delivery_snapshot(room) == before_delivery
    assert room.evidence_sink_error_type == "PersistenceError"


def test_terminal_run_completed_commits_one_status_and_restores_without_duplication() -> None:
    manager = RoomManager()
    room = _room(manager, "terminal-transaction")
    room.state.phase = Phase.ENDED
    room.state.winner = Team.VILLAGE
    record_harness = manager._make_harness_recorder(room)

    # This is the durable boundary reached immediately before Core returns.
    record_harness({
        "type": "run_completed",
        "status": "completed",
        "termination_reason": None,
        "outcome": {"winner": "village"},
        "metrics": {},
    })

    assert room.status == "ended"
    assert room.end_reason == "completed"
    assert [row["type"] for row in room.event_history] == ["room_status"]
    assert room.transcript is not None
    assert [entry.payload["type"] for entry in room.transcript.entries] == [
        "run_completed",
        "room_status",
    ]
    assert room.delivery_source_seq == 1

    restored = manager._room_from_record(manager._room_record(room))

    assert restored.status == "ended"
    assert restored.end_reason == "completed"
    assert [row["type"] for row in restored.event_history] == ["room_status"]
    assert restored.transcript is not None
    assert [entry.payload["type"] for entry in restored.transcript.entries] == [
        "run_completed",
        "room_status",
    ]
    assert restored.delivery_source_seq == 1
    assert sum(
        entry.payload.get("type") == "room_status"
        for entry in restored.transcript.entries
    ) == 1


@pytest.mark.parametrize(
    ("phase", "winner", "termination_reason", "expected_status", "expected_reason"),
    [
        (Phase.ENDED, None, "max_rounds", "incomplete", "max_rounds"),
        (Phase.DAY, None, "max_rounds", "failed", "invalid_core_outcome"),
        (Phase.ENDED, Team.VILLAGE, "max_rounds", "failed", "invalid_core_outcome"),
    ],
)
def test_terminal_run_incomplete_projection_accepts_only_valid_terminal_state(
    phase: Phase,
    winner: Team | None,
    termination_reason: str,
    expected_status: str,
    expected_reason: str,
) -> None:
    manager = RoomManager()
    room = _room(manager, f"incomplete-{expected_status}-{expected_reason}-{phase.value}")
    room.state.phase = phase
    room.state.winner = winner
    record_harness = manager._make_harness_recorder(room)

    record_harness({
        "type": "run_incomplete",
        "status": "incomplete",
        "termination_reason": termination_reason,
        "outcome": {},
        "metrics": {},
    })

    assert room.status == expected_status
    assert room.end_reason == expected_reason
    if expected_status == "incomplete":
        assert room.error is None
        assert [row["type"] for row in room.event_history] == ["room_status"]
    else:
        assert room.error == "Core incomplete outcome has an invalid terminal state or reason"
        assert [row["type"] for row in room.event_history] == [
            "game_error",
            "room_status",
        ]
    assert room.transcript is not None
    assert sum(
        entry.payload.get("type") == "run_incomplete"
        for entry in room.transcript.entries
    ) == 1
    assert sum(
        entry.payload.get("type") == "room_status"
        for entry in room.transcript.entries
    ) == 1


class _CloseCounter:
    def __init__(self) -> None:
        self.calls = 0

    async def aclose(self) -> None:
        self.calls += 1


@pytest.mark.asyncio
async def test_missing_core_spec_closes_prepared_session_and_runtime_once() -> None:
    manager = RoomManager()
    room = _room(manager, "missing-core-spec")
    session = _CloseCounter()
    runtime = _CloseCounter()
    room.prepared_run = SimpleNamespace(
        session=session,
        decision_runtime=runtime,
        _claimed=False,
    )
    room.core_run_spec = None

    await manager._run_core_room(room)

    assert session.calls == 1
    assert runtime.calls == 1
    assert room.prepared_run is None
    assert room.status == "failed"
    assert room.end_reason == "missing_core_runtime"


@pytest.mark.asyncio
async def test_core_quarantine_sink_transfers_pending_task_to_room_manager() -> None:
    manager = RoomManager()
    room = _room(manager, "quarantine-transfer")
    release = asyncio.Event()

    async def stubborn_task() -> None:
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue

    task = asyncio.create_task(stubborn_task())
    await asyncio.sleep(0)
    manager._make_core_task_quarantine_sink(room)(task, "session_run")

    assert manager.unresolved_cleanup_task_count == 1
    assert manager._quarantined_tasks[task]["stage"] == "core:session_run"

    release.set()
    await asyncio.wait_for(task, timeout=1.0)
    await asyncio.sleep(0)
    assert manager.unresolved_cleanup_task_count == 0
