"""Fail-closed provenance checks for persisted interactive room evidence."""
from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator

import pytest

from src.api.persistence import PersistenceError
from src.api.room_manager import RoomManager
from src.game.models import Phase
from src.game.roles import Team, default_role_deck
from src.harness.core_spec import CoreRunSpec
from src.harness.spec import RunSpec
from src.harness.transcript import Transcript, payload_digest
from src.llm.models import ModelConfig


@dataclass(frozen=True)
class _RestoreFixture:
    manager: RoomManager
    executed_record: dict[str, Any]
    waiting_record: dict[str, Any]


def _export_transcript(record: dict[str, Any]) -> None:
    """Recompute every transcript aggregate after an intentional mutation."""
    raw = record["transcript"]
    transcript = Transcript.model_validate(
        {
            "schema_version": raw["schema_version"],
            "run_id": raw["run_id"],
            "metadata": raw.get("metadata") or {},
            "entries": raw["entries"],
        }
    )
    record["transcript"] = transcript.export()


def _source_entry(
    record: dict[str, Any],
    *,
    kind: str,
    source_idx: int,
) -> dict[str, Any]:
    matches = [
        entry
        for entry in record["transcript"]["entries"]
        if entry["kind"] == kind and entry["source_idx"] == source_idx
    ]
    assert len(matches) == 1
    return matches[0]


def _mirror_source_payload(
    record: dict[str, Any],
    *,
    kind: str,
    source_idx: int,
) -> None:
    history_key = "event_history" if kind == "event" else "decision_trace"
    entry = _source_entry(record, kind=kind, source_idx=source_idx)
    entry["payload"] = deepcopy(record[history_key][source_idx])
    entry["payload_hash"] = payload_digest(entry["payload"])


def _renumber_and_export_transcript(record: dict[str, Any]) -> None:
    for seq, entry in enumerate(record["transcript"]["entries"], start=1):
        entry["seq"] = seq
    _export_transcript(record)


@pytest.fixture(scope="module")
def restore_fixture() -> Iterator[_RestoreFixture]:
    manager = RoomManager(max_rooms=4)
    names = ["A", "B", "C", "D", "E", "F"]

    room = manager.create_room(
        player_names=names,
        human_seats=set(range(1, 7)),
        experiment_seed=1701,
    )
    run_spec = manager._build_run_spec(
        room,
        ModelConfig(),
        {},
        default_role_deck(len(names)),
    )
    core_run_spec = manager._build_core_run_spec(run_spec)
    room.run_spec = run_spec
    room.core_run_spec = core_run_spec
    room.transcript = Transcript(
        run_id=room.id,
        metadata={
            "room_id": room.id,
            "core_run_spec_version": core_run_spec.schema_version,
            "run_spec_hash": core_run_spec.spec_hash,
            "legacy_run_spec_hash": run_spec.spec_hash,
            "environment_id": core_run_spec.environment.id,
            "environment_version": core_run_spec.environment.version,
            "role_seed": room.role_seed,
            "actor_seed": room.actor_seed,
            "orchestrator_seed": room.orchestrator_seed,
        },
    )
    room.status = "ended"
    room.state.phase = Phase.ENDED
    room.state.winner = Team.VILLAGE

    trace = manager._make_trace_recorder(room)
    manager._store_room_event(room, {"type": "fixture_event", "ordinal": 1})
    trace({"type": "fixture_decision", "ordinal": 1, "seat": 1})
    manager._store_room_event(room, {"type": "fixture_event", "ordinal": 2})
    trace({"type": "fixture_decision", "ordinal": 2, "seat": 2})
    assert room.trace_seq == 4
    room.terminal_at = time.monotonic()
    executed_record = manager._room_record(room)

    waiting = manager.create_room(
        player_names=names,
        human_seats=set(range(1, 7)),
        experiment_seed=1702,
    )
    waiting_record = manager._room_record(waiting)

    yield _RestoreFixture(
        manager=manager,
        executed_record=executed_record,
        waiting_record=waiting_record,
    )
    asyncio.run(manager.aclose())


def test_canonical_executed_record_is_restorable(
    restore_fixture: _RestoreFixture,
) -> None:
    restored = restore_fixture.manager._room_from_record(
        deepcopy(restore_fixture.executed_record)
    )

    assert restored.run_spec is not None
    assert restored.core_run_spec is not None
    assert restored.transcript is not None
    assert restored.trace_seq == 4
    assert [entry.source_idx for entry in restored.transcript.entries] == [0, 0, 1, 1]
    assert [
        entry.payload["_trace_seq"] for entry in restored.transcript.entries
    ] == [1, 2, 3, 4]


@pytest.mark.parametrize("mutation", ["missing", "malformed", "wrong"])
def test_core_record_requires_matching_canonical_run_spec_hash(
    restore_fixture: _RestoreFixture,
    mutation: str,
) -> None:
    record = deepcopy(restore_fixture.executed_record)
    metadata = record["transcript"]["metadata"]
    if mutation == "missing":
        metadata.pop("run_spec_hash")
    elif mutation == "malformed":
        metadata["run_spec_hash"] = "not-a-canonical-sha256"
    else:
        metadata["run_spec_hash"] = "0" * 64
    _export_transcript(record)

    with pytest.raises(PersistenceError):
        restore_fixture.manager._room_from_record(record)


def test_explicit_legacy_record_without_core_or_canonical_hash_is_compatible(
    restore_fixture: _RestoreFixture,
) -> None:
    record = deepcopy(restore_fixture.executed_record)
    record["core_run_spec"] = None
    record["transcript"]["metadata"].pop("run_spec_hash")
    record["transcript"]["metadata"].pop("core_run_spec_version", None)
    _export_transcript(record)

    restored = restore_fixture.manager._room_from_record(record)

    assert restored.run_spec is not None
    assert restored.core_run_spec is None
    assert restored.transcript is not None
    assert "run_spec_hash" not in restored.transcript.metadata
    assert (
        restored.transcript.metadata["legacy_run_spec_hash"]
        == restored.run_spec.spec_hash
    )


def test_legacy_spec_tamper_cannot_be_authorized_by_rehashing_legacy_metadata(
    restore_fixture: _RestoreFixture,
) -> None:
    record = deepcopy(restore_fixture.executed_record)
    record["run_spec"]["max_speak_rounds"] += 1
    tampered_legacy = RunSpec.model_validate(record["run_spec"])
    record["run_spec"] = tampered_legacy.model_dump(mode="json")
    record["transcript"]["metadata"][
        "legacy_run_spec_hash"
    ] = tampered_legacy.spec_hash
    _export_transcript(record)

    with pytest.raises(PersistenceError):
        restore_fixture.manager._room_from_record(record)


def test_core_spec_tamper_cannot_be_authorized_by_rehashing_canonical_metadata(
    restore_fixture: _RestoreFixture,
) -> None:
    record = deepcopy(restore_fixture.executed_record)
    record["core_run_spec"]["environment_config"]["max_speak_rounds"] += 1
    tampered_core = CoreRunSpec.model_validate(record["core_run_spec"])
    record["core_run_spec"] = tampered_core.model_dump(mode="json")
    record["transcript"]["metadata"]["run_spec_hash"] = tampered_core.spec_hash
    _export_transcript(record)

    with pytest.raises(PersistenceError):
        restore_fixture.manager._room_from_record(record)


@pytest.mark.parametrize("kind", ["event", "decision"])
def test_transcript_rejects_same_kind_source_index_reordering(
    restore_fixture: _RestoreFixture,
    kind: str,
) -> None:
    record = deepcopy(restore_fixture.executed_record)
    entries = record["transcript"]["entries"]
    indexes = [index for index, entry in enumerate(entries) if entry["kind"] == kind]
    assert len(indexes) == 2
    first, second = indexes
    entries[first], entries[second] = entries[second], entries[first]
    _renumber_and_export_transcript(record)

    with pytest.raises(PersistenceError):
        restore_fixture.manager._room_from_record(record)


def test_transcript_rejects_cross_kind_trace_sequence_reordering(
    restore_fixture: _RestoreFixture,
) -> None:
    record = deepcopy(restore_fixture.executed_record)
    entries = record["transcript"]["entries"]
    event_index = next(
        index
        for index, entry in enumerate(entries)
        if entry["kind"] == "event" and entry["source_idx"] == 1
    )
    decision_index = next(
        index
        for index, entry in enumerate(entries)
        if entry["kind"] == "decision" and entry["source_idx"] == 0
    )
    entries[event_index], entries[decision_index] = (
        entries[decision_index],
        entries[event_index],
    )
    _renumber_and_export_transcript(record)

    with pytest.raises(PersistenceError):
        restore_fixture.manager._room_from_record(record)


@pytest.mark.parametrize("trace_seq", [True, "4", -1])
def test_restore_rejects_noncanonical_trace_cursor_types(
    restore_fixture: _RestoreFixture,
    trace_seq: Any,
) -> None:
    record = deepcopy(restore_fixture.executed_record)
    record["trace_seq"] = trace_seq

    with pytest.raises(PersistenceError):
        restore_fixture.manager._room_from_record(record)


@pytest.mark.parametrize("mutation", ["missing", "smaller"])
def test_restore_rejects_trace_cursor_that_does_not_commit_source_history(
    restore_fixture: _RestoreFixture,
    mutation: str,
) -> None:
    record = deepcopy(restore_fixture.executed_record)
    if mutation == "missing":
        record.pop("trace_seq")
    else:
        record["trace_seq"] = 3

    with pytest.raises(PersistenceError):
        restore_fixture.manager._room_from_record(record)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reversed"])
def test_restore_rejects_invalid_source_trace_sequences(
    restore_fixture: _RestoreFixture,
    mutation: str,
) -> None:
    record = deepcopy(restore_fixture.executed_record)
    if mutation == "missing":
        record["event_history"][0].pop("_trace_seq")
        _mirror_source_payload(record, kind="event", source_idx=0)
    elif mutation == "duplicate":
        record["event_history"][1]["_trace_seq"] = 2
        _mirror_source_payload(record, kind="event", source_idx=1)
    else:
        record["event_history"][0]["_trace_seq"] = 3
        record["event_history"][1]["_trace_seq"] = 1
        _mirror_source_payload(record, kind="event", source_idx=0)
        _mirror_source_payload(record, kind="event", source_idx=1)
    _export_transcript(record)

    with pytest.raises(PersistenceError):
        restore_fixture.manager._room_from_record(record)


@pytest.mark.parametrize("spec_field", ["run_spec", "core_run_spec"])
def test_waiting_room_rejects_foreign_execution_spec_identity(
    restore_fixture: _RestoreFixture,
    spec_field: str,
) -> None:
    record = deepcopy(restore_fixture.waiting_record)
    foreign_spec = deepcopy(restore_fixture.executed_record[spec_field])
    foreign_spec["run_id"] = "foreign-run"
    record[spec_field] = foreign_spec

    with pytest.raises(PersistenceError):
        restore_fixture.manager._room_from_record(record)


def test_legacy_waiting_state_id_migration_remains_supported(
    restore_fixture: _RestoreFixture,
) -> None:
    record = deepcopy(restore_fixture.waiting_record)
    record["state"]["id"] = "legacy-generated-state-id"

    restored = restore_fixture.manager._room_from_record(record)

    assert restored.status == "waiting"
    assert restored.state.id == restored.id == record["room_id"]
    assert restored.run_spec is None
    assert restored.core_run_spec is None
    assert restored.transcript is None
    assert restored.event_history == []
    assert restored.decision_trace == []
    assert restored.trace_seq == 0
