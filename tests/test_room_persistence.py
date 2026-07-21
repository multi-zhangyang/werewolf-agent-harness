"""Opt-in room persistence, restart, and capability lifecycle tests."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from src.api.persistence import (
    PERSISTENCE_SCHEMA_VERSION,
    PersistenceCredentialError,
    PersistenceError,
    PersistenceIntegrityError,
    SQLiteRoomPersistence,
    hash_capability,
)
from src.api.room_manager import (
    CapabilityAuthorizationError,
    RoomClient,
    RoomManager,
)
from src.game.models import Phase
from src.game.roles import Team
from src.llm.models import ModelConfig


def _config() -> ModelConfig:
    return ModelConfig(
        provider="openai",
        model="offline-model",
        api_base="https://example.invalid/v1",
        api_key="unit-test-api-key-do-not-persist",
    )


def _manager(path: Path) -> RoomManager:
    return RoomManager(persistence_path=path, terminal_room_ttl=3600)


class _CaptureWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.accepted_subprotocol: str | None = None
        self.messages: list[str] = []
        self.close_calls: list[tuple[int, str]] = []

    async def accept(self, subprotocol: str | None = None) -> None:
        self.accepted = True
        self.accepted_subprotocol = subprotocol

    async def send_text(self, message: str) -> None:
        self.messages.append(message)

    async def close(self, *, code: int = 1000, reason: str = "") -> None:
        self.close_calls.append((code, reason))

    def json_messages(self) -> list[dict]:
        return [json.loads(message) for message in self.messages]


def test_sqlite_store_rejects_credentials_and_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "rooms.sqlite"
    store = SQLiteRoomPersistence(path)
    with pytest.raises(PersistenceCredentialError):
        store.save_record({
            "schema_version": PERSISTENCE_SCHEMA_VERSION,
            "room_id": "credential-test",
            "api_key": "sk-obviously-secret-value",
        })
    with pytest.raises(PersistenceCredentialError):
        store.save_record({
            "schema_version": PERSISTENCE_SCHEMA_VERSION,
            "room_id": "fake-hash-test",
            "admin_token_hash": "plain-capability-value-without-prefix-123456",
        })
    for field in ("api_key_hash", "authorization_hash", "secret_hash"):
        with pytest.raises(PersistenceCredentialError):
            store.save_record({
                "schema_version": PERSISTENCE_SCHEMA_VERSION,
                "room_id": f"fake-{field}",
                field: "plain-credential-value-123456",
            })

    store.save_record({
        "schema_version": PERSISTENCE_SCHEMA_VERSION,
        "room_id": "safe-test",
        "capability_hash": hash_capability("capability-test"),
        "api_key_configured": True,
    })
    store.close()

    # Alter the JSON without updating the row digest.  Reads fail closed.
    conn = sqlite3.connect(path)
    body = conn.execute("SELECT payload_json FROM rooms WHERE room_id='safe-test'").fetchone()[0]
    conn.execute(
        "UPDATE rooms SET payload_json=? WHERE room_id='safe-test'",
        (body.replace("safe-test", "forged-test"),),
    )
    conn.commit()
    conn.close()
    with pytest.raises(PersistenceIntegrityError):
        SQLiteRoomPersistence(path).load_records()


def test_sqlite_store_preserves_json_schema_credential_shaped_property_names(
    tmp_path: Path,
) -> None:
    response_format = ModelConfig(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "public_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string"},
                        "token_count": {"type": "integer", "minimum": 0},
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "access_token": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["api_key", "token_count"],
                },
            },
        },
    ).response_format
    assert response_format is not None

    with SQLiteRoomPersistence(tmp_path / "schema-properties.sqlite") as store:
        store.save_record({
            "schema_version": PERSISTENCE_SCHEMA_VERSION,
            "room_id": "schema-property-names",
            "default_model_config": {"response_format": response_format},
        })
        restored = store.load_records()[0]

    restored_properties = restored["default_model_config"]["response_format"][
        "json_schema"
    ]["schema"]["properties"]
    assert restored_properties["api_key"] == {"type": "string"}
    assert restored_properties["token_count"] == {"minimum": 0, "type": "integer"}
    assert restored_properties["rows"]["items"]["properties"]["access_token"] == {
        "type": "string"
    }


@pytest.mark.parametrize(
    "secret_text",
    [
        "sk-schema-secret-value-123456789",
        "Bearer schema-secret-value-123456789",
        "https://example.invalid/schema?api_key=real-secret-value",
    ],
)
def test_sqlite_store_rejects_credentials_inside_json_schema_values(
    tmp_path: Path,
    secret_text: str,
) -> None:
    response_format = ModelConfig(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "unsafe_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "description": secret_text,
                        },
                    },
                },
            },
        },
    ).response_format
    assert response_format is not None

    with SQLiteRoomPersistence(tmp_path / "unsafe-schema.sqlite") as store:
        with pytest.raises(PersistenceCredentialError):
            store.save_record({
                "schema_version": PERSISTENCE_SCHEMA_VERSION,
                "room_id": "unsafe-schema-value",
                "default_model_config": {"response_format": response_format},
            })


def test_waiting_room_round_trips_without_api_key_or_plaintext_tokens(tmp_path: Path) -> None:
    path = tmp_path / "rooms.sqlite"
    first = _manager(path)
    room = first.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats={1},
        default_model_config=_config(),
    )
    room_id = room.id
    assert room.state.id == room_id
    old_admin = room.admin_token
    old_seat = room.seat_tokens[1]
    first.persistence.close()  # emulate an orderly process stop without router work
    first._persistence_closed = True

    raw = path.read_bytes()
    assert b"unit-test-api-key-do-not-persist" not in raw
    assert old_admin.encode() not in raw
    assert old_seat.encode() not in raw

    second = _manager(path)
    restored = second.get_room(room_id)
    assert restored is not None
    assert restored.status == "waiting"
    assert restored.state.id == restored.id == room_id
    assert restored.run_spec is None
    assert restored.transcript is None
    assert restored.default_config is not None
    assert restored.default_config.api_key == ""
    # Plaintext is never reconstructed, but the original caller-held
    # capability remains verifiable through its persisted salted hash.
    assert restored.admin_token == ""
    assert restored.seat_tokens[1] == ""
    assert second.valid_admin_token(restored, old_admin) is True
    assert second.valid_seat_token(restored, 1, old_seat) is True
    assert second.valid_admin_token(restored, "wrong-admin") is False
    assert second.valid_seat_token(restored, 1, "wrong-seat") is False
    assert restored.admin_token_version == 1
    assert restored.seat_token_versions[1] == 1


def test_waiting_room_round_trips_json_schema_credential_shaped_properties(
    tmp_path: Path,
) -> None:
    path = tmp_path / "schema-room.sqlite"
    config = _config().model_copy(update={
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "usage_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string"},
                        "token_count": {"type": "integer", "minimum": 0},
                    },
                    "required": ["api_key", "token_count"],
                },
            },
        },
    })
    first = _manager(path)
    room = first.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        default_model_config=config,
    )
    assert first.persistence is not None
    first.persistence.close()
    first._persistence_closed = True

    restored = _manager(path).get_room(room.id)

    assert restored is not None and restored.default_config is not None
    schema = restored.default_config.response_format["json_schema"]["schema"]
    assert schema["properties"]["api_key"] == {"type": "string"}
    assert schema["properties"]["token_count"] == {
        "minimum": 0,
        "type": "integer",
    }


def test_legacy_waiting_room_state_id_is_normalized_before_first_run(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rooms.sqlite"
    first = _manager(path)
    room = first.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    room.state.id = "legacy-generated-game-id"
    first._persist_room(room)
    assert first.persistence is not None
    first.persistence.close()
    first._persistence_closed = True

    second = _manager(path)
    restored = second.get_room(room.id)

    assert restored is not None
    assert restored.status == "waiting"
    assert restored.state.id == restored.id == room.id
    assert restored.run_spec is None
    assert restored.transcript is None


def test_restart_rejects_room_evidence_above_current_capacity(tmp_path: Path) -> None:
    path = tmp_path / "rooms.sqlite"
    first = RoomManager(
        persistence_path=path,
        terminal_room_ttl=3600,
        max_evidence_entries=4,
    )
    room = first.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    first._store_room_event(room, {"type": "phase_started", "phase": "day", "day": 1})
    first._store_room_event(room, {"type": "phase_started", "phase": "voting", "day": 1})
    first._persist_room(room)
    assert first.persistence is not None
    first.persistence.close()
    first._persistence_closed = True

    second = RoomManager(
        persistence_path=path,
        restore_persisted_rooms=False,
        max_evidence_entries=1,
    )
    assert second.persistence is not None
    record = second.persistence.load_records()[0]
    try:
        with pytest.raises(PersistenceError, match="evidence exceeds capacity"):
            second._room_from_record(record)
    finally:
        second.persistence.close()
        second._persistence_closed = True


def test_running_room_restores_as_interrupted_with_audit_evidence(tmp_path: Path) -> None:
    path = tmp_path / "rooms.sqlite"

    async def scenario() -> None:
        first = _manager(path)
        room = first.create_room(
            player_names=["A", "B", "C", "D", "E", "F"],
            human_seats=set(range(1, 7)),
        )

        async def blocked(_room):
            await asyncio.sleep(3600)

        first._run_room = blocked  # type: ignore[method-assign]
        await first.start_game(room)
        assert room.status == "running"
        assert room.run_spec is not None
        assert room.core_run_spec is not None
        assert room.transcript is not None
        first.persistence.close()  # crash-like stop: persisted row still says running
        first._persistence_closed = True
        if room.task is not None:
            room.task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await room.task

    awaitable = scenario()
    asyncio.run(awaitable)

    second = _manager(path)
    restored = next(iter(second.rooms.values()))
    assert restored.status == "interrupted"
    assert restored.end_reason == "process_restart"
    assert restored.error == "room interrupted during process restart"
    assert restored.task is None
    assert restored.orchestrator is None
    assert restored.run_spec is not None
    assert restored.core_run_spec is not None
    assert restored.core_run_spec.run_id == restored.id
    assert restored.transcript is not None
    assert restored.transcript.entries[-1].payload["status"] == "interrupted"
    assert restored.transcript.entries[-1].payload["reason"] == "process_restart"


@pytest.mark.asyncio
async def test_non_waiting_restore_cross_checks_run_identity_and_spec_hash() -> None:
    manager = RoomManager(room_timeout=1.0)

    async def no_run(_room) -> None:
        return None

    manager._run_room = no_run  # type: ignore[method-assign]
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats=set(range(1, 7)),
    )
    await manager.start_game(room)
    assert room.task is not None
    await room.task
    assert room.run_spec is not None
    assert room.core_run_spec is not None
    assert room.transcript is not None

    valid_record = manager._room_record(room)
    identity_cases = {
        "state": "persisted game state id does not match room",
        "run_spec": "persisted RunSpec run id does not match room",
        "core_run_spec": "persisted CoreRunSpec run id does not match room",
        "transcript": "persisted transcript run id does not match room",
    }
    for status in ("running", "ended"):
        for field, expected_error in identity_cases.items():
            tampered = json.loads(json.dumps(valid_record))
            tampered["status"] = status
            if field == "state":
                tampered["state"]["id"] = "other-state"
            elif field == "run_spec":
                tampered["run_spec"]["run_id"] = "other-run"
            elif field == "core_run_spec":
                tampered["core_run_spec"]["run_id"] = "other-core-run"
            else:
                tampered["transcript"]["run_id"] = "other-run"
            with pytest.raises(PersistenceError, match=expected_error):
                manager._room_from_record(tampered)

    room.transcript.metadata["run_spec_hash"] = "0" * 64
    wrong_hash_record = manager._room_record(room)
    with pytest.raises(PersistenceError, match="run_spec_hash does not match"):
        manager._room_from_record(wrong_hash_record)

    missing_spec_record = json.loads(json.dumps(wrong_hash_record))
    missing_spec_record["run_spec"] = None
    missing_spec_record["core_run_spec"] = None
    with pytest.raises(PersistenceError, match="run_spec_hash has no RunSpec"):
        manager._room_from_record(missing_spec_record)


def test_terminal_state_event_history_and_transcript_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "rooms.sqlite"
    first = _manager(path)
    room = first.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats=set(range(1, 7)),
    )
    room.status = "ended"
    room.state.phase = Phase.ENDED
    room.state.winner = Team.VILLAGE
    room.event_history.append({"type": "game_ended", "winner": "village", "_ts": 1.0, "_trace_seq": 1})
    room.trace_seq = 1
    room.transcript = None
    first._append_transcript(room, "event", room.event_history[-1], source_idx=0)
    room.terminal_at = time.monotonic()
    first._persist_room(room)
    first.persistence.close()
    first._persistence_closed = True

    second = _manager(path)
    restored = second.get_room(room.id)
    assert restored is not None
    assert restored.status == "ended"
    assert restored.state.winner == Team.VILLAGE
    assert restored.event_history[0]["type"] == "game_ended"
    assert restored.transcript is not None
    assert restored.transcript.entries[0].kind == "event"
    assert restored.transcript.entries[0].payload["type"] == "game_ended"


@pytest.mark.asyncio
async def test_delivery_stream_cursor_and_identity_survive_bounded_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rooms.sqlite"
    first = RoomManager(
        persistence_path=path,
        terminal_room_ttl=3600,
        ws_delivery_history_size=2,
    )
    room = first.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    for day in range(1, 5):
        await first._broadcast(
            room,
            {"type": "phase_started", "phase": "day", "day": day},
            initial_replay=True,
        )
    original_stream = room.delivery_streams["spectate"]
    original_stream_id = original_stream.stream_id
    original_last_id = original_stream.history[-1].delivery_id
    assert original_stream.cursor == 4
    assert [record.seq for record in original_stream.history] == [3, 4]
    await first.aclose()

    second = RoomManager(
        persistence_path=path,
        terminal_room_ttl=3600,
        ws_delivery_history_size=2,
    )
    restored = second.get_room(room.id)
    assert restored is not None
    restored_stream = restored.delivery_streams["spectate"]
    assert restored_stream.stream_id == original_stream_id
    assert restored_stream.cursor == 4
    assert restored_stream.history_gap is True
    assert [record.seq for record in restored_stream.history] == [3, 4]
    assert restored_stream.history[-1].delivery_id == original_last_id

    socket = _CaptureWebSocket()
    cid = await second.connect(
        restored,
        socket,  # type: ignore[arg-type]
        seat=None,
        mode="spectate",
        since=3,
    )
    resumed = socket.json_messages()
    assert resumed[0]["stream_id"] == original_stream_id
    assert resumed[0]["cursor"] == 4
    assert resumed[0]["replay_from"] == 4
    assert [message["delivery_seq"] for message in resumed[1:]] == [4]
    assert resumed[1]["delivery_id"] == original_last_id

    await second._broadcast(
        restored,
        {"type": "phase_started", "phase": "day", "day": 5},
        initial_replay=True,
    )
    for _ in range(20):
        if len(socket.messages) >= 3:
            break
        await asyncio.sleep(0)
    live = socket.json_messages()[-1]
    assert live["delivery_seq"] == 5
    assert live["delivery_id"] == f"{original_stream_id}.5"

    second.disconnect(restored, cid)
    await asyncio.sleep(0)
    await second.aclose()


@pytest.mark.parametrize(
    "tamper",
    ["missing_digest", "payload_without_hash", "noncontiguous_seq", "source_mismatch"],
)
def test_restart_rejects_semantically_tampered_transcript(
    tmp_path: Path,
    tamper: str,
) -> None:
    path = tmp_path / f"rooms-{tamper}.sqlite"
    manager = _manager(path)
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    room.status = "ended"
    room.state.phase = Phase.ENDED
    room.state.winner = Team.VILLAGE
    room.event_history.append(
        {"type": "game_ended", "winner": "village", "_ts": 1.0, "_trace_seq": 1}
    )
    room.trace_seq = 1
    manager._append_transcript(room, "event", room.event_history[0], source_idx=0)
    room.terminal_at = time.monotonic()
    manager._persist_room(room)
    manager.persistence.close()
    manager._persistence_closed = True

    conn = sqlite3.connect(path)
    raw = conn.execute(
        "SELECT payload_json FROM rooms WHERE room_id = ?", (room.id,)
    ).fetchone()[0]
    payload = json.loads(raw)
    if tamper == "missing_digest":
        payload["transcript"].pop("stable_digest")
    elif tamper == "payload_without_hash":
        payload["transcript"]["entries"][0]["payload"]["winner"] = "werewolves"
    elif tamper == "noncontiguous_seq":
        payload["transcript"]["entries"][0]["seq"] = 2
    else:
        payload["event_history"][0]["winner"] = "werewolves"
    rewritten = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(rewritten.encode("utf-8")).hexdigest()
    conn.execute(
        "UPDATE rooms SET payload_json = ?, payload_sha256 = ? WHERE room_id = ?",
        (rewritten, digest, room.id),
    )
    conn.commit()
    conn.close()

    with pytest.raises(PersistenceError):
        _manager(path)


def test_rotation_and_revocation_invalidate_previous_capabilities(tmp_path: Path) -> None:
    path = tmp_path / "rooms.sqlite"
    manager = _manager(path)
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats={1},
    )
    old_admin = room.admin_token
    old_seat = room.seat_tokens[1]
    new_admin = manager.rotate_admin_token(room)
    new_seat = manager.rotate_seat_token(room, 1)
    assert old_admin != new_admin and old_seat != new_seat
    assert manager.valid_admin_token(room, old_admin) is False
    assert manager.valid_seat_token(room, 1, old_seat) is False
    assert manager.valid_admin_token(room, new_admin) is True
    assert manager.valid_seat_token(room, 1, new_seat) is True

    manager.revoke_seat_token(room, 1)
    assert manager.valid_seat_token(room, 1, new_seat) is False
    assert room.seat_tokens[1] == ""
    manager.persistence.close()
    manager._persistence_closed = True

    restored_manager = _manager(path)
    restored = restored_manager.get_room(room.id)
    assert restored is not None
    assert restored.seat_tokens[1] == ""
    assert restored_manager.valid_seat_token(restored, 1, new_seat) is False


def test_concurrent_persistence_cannot_reactivate_rotated_admin_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rooms.sqlite"
    manager = _manager(path)
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    stale_admin = room.admin_token
    assert manager.persistence is not None
    original_save = manager.persistence.save_record
    blocked = threading.Event()
    release = threading.Event()
    first_save_lock = threading.Lock()
    block_next = True

    def blocking_save(record) -> None:
        nonlocal block_next
        with first_save_lock:
            should_block = block_next
            block_next = False
        if should_block:
            blocked.set()
            assert release.wait(timeout=2)
        original_save(record)

    manager.persistence.save_record = blocking_save  # type: ignore[method-assign]
    persist_errors: list[BaseException] = []
    rotation_errors: list[BaseException] = []
    rotated_tokens: list[str] = []

    def persist_old_snapshot() -> None:
        try:
            manager._persist_room(room)
        except BaseException as err:  # pragma: no cover - asserted below
            persist_errors.append(err)

    def rotate() -> None:
        try:
            rotated_tokens.append(manager.rotate_admin_token(room))
        except BaseException as err:  # pragma: no cover - asserted below
            rotation_errors.append(err)

    persistence_thread = threading.Thread(target=persist_old_snapshot)
    rotation_thread = threading.Thread(target=rotate)
    persistence_thread.start()
    assert blocked.wait(timeout=2)
    rotation_thread.start()
    # Rotation must wait for the older durable snapshot to commit; otherwise
    # that snapshot can race behind the new hash and become the final row.
    time.sleep(0.02)
    assert rotation_thread.is_alive()
    release.set()
    persistence_thread.join(timeout=2)
    rotation_thread.join(timeout=2)
    assert not persistence_thread.is_alive()
    assert not rotation_thread.is_alive()
    assert persist_errors == []
    assert rotation_errors == []
    assert len(rotated_tokens) == 1

    current_admin = rotated_tokens[0]
    manager.persistence.close()
    manager._persistence_closed = True
    restored_manager = _manager(path)
    restored = restored_manager.get_room(room.id)
    assert restored is not None
    assert restored_manager.valid_admin_token(restored, stale_admin) is False
    assert restored_manager.valid_admin_token(restored, current_admin) is True


@pytest.mark.parametrize(
    "mutation",
    ["rotate_admin", "revoke_admin", "rotate_seat", "revoke_seat"],
)
def test_capability_mutation_rolls_back_when_persistence_fails(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    manager = RoomManager()
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats={1},
    )
    old_admin = room.admin_token
    old_seat = room.seat_tokens[1]
    before = (
        room.admin_token,
        room.admin_token_hash,
        room.admin_token_version,
        room.admin_token_revoked,
        list(room.revoked_admin_token_hashes),
        dict(room.seat_tokens),
        dict(room.seat_token_hashes),
        dict(room.seat_token_versions),
        {key: list(value) for key, value in room.revoked_seat_token_hashes.items()},
    )

    def fail(_room) -> None:
        raise PersistenceError("opaque injected persistence failure")

    monkeypatch.setattr(manager, "_persist_room", fail)
    with pytest.raises(PersistenceError):
        if mutation == "rotate_admin":
            manager.rotate_admin_token(room)
        elif mutation == "revoke_admin":
            manager.revoke_admin_token(room)
        elif mutation == "rotate_seat":
            manager.rotate_seat_token(room, 1)
        else:
            manager.revoke_seat_token(room, 1)

    after = (
        room.admin_token,
        room.admin_token_hash,
        room.admin_token_version,
        room.admin_token_revoked,
        list(room.revoked_admin_token_hashes),
        dict(room.seat_tokens),
        dict(room.seat_token_hashes),
        dict(room.seat_token_versions),
        {key: list(value) for key, value in room.revoked_seat_token_hashes.items()},
    )
    assert after == before
    assert manager.valid_admin_token(room, old_admin)
    assert manager.valid_seat_token(room, 1, old_seat)


@pytest.mark.parametrize("had_previous", [False, True])
def test_seat_model_config_rolls_back_when_persistence_fails(
    monkeypatch: pytest.MonkeyPatch,
    had_previous: bool,
) -> None:
    manager = RoomManager()
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    previous = ModelConfig(model="previous-model", api_key="previous-key")
    if had_previous:
        room.seat_configs[1] = previous

    def fail(_room) -> None:
        raise PersistenceError("opaque injected persistence failure")

    monkeypatch.setattr(manager, "_persist_room", fail)
    with pytest.raises(PersistenceError, match="opaque injected persistence failure"):
        manager.set_seat_model_config(
            room,
            1,
            ModelConfig(model="replacement-model", api_key="replacement-key"),
        )

    if had_previous:
        assert room.seat_configs[1] is previous
    else:
        assert 1 not in room.seat_configs


@pytest.mark.asyncio
async def test_rotation_disconnects_clients_bound_to_the_old_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = RoomManager()
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats={1},
    )
    loop = asyncio.get_running_loop()

    def client(mode: str, seat: int | None) -> RoomClient:
        return RoomClient(
            websocket=object(),  # type: ignore[arg-type]
            seat=seat,
            mode=mode,
            stream_key=mode,
            queue=asyncio.Queue(),
            loop=loop,
        )

    room.clients = {
        "god": client("god", None),
        "seat": client("play", 1),
        "spectator": client("spectate", None),
    }
    disconnected: list[str] = []

    def terminate(_room, connection, *, code: int, reason: str) -> None:
        assert code == 4403
        assert reason == "capability rotated or revoked"
        disconnected.append(connection.mode)

    monkeypatch.setattr(manager, "_terminate_client_on_owner_loop", terminate)
    manager.rotate_admin_token(room)
    assert disconnected == ["god"]
    manager.rotate_seat_token(room, 1)
    assert disconnected == ["god", "play"]
    assert list(room.clients) == ["spectator"]


@pytest.mark.asyncio
async def test_connection_registration_rechecks_capability_after_rotation() -> None:
    manager = RoomManager()
    room = manager.create_room(
        player_names=["A", "B", "C", "D", "E", "F"],
        human_seats={1},
    )
    stale_admin = room.admin_token
    stale_seat = room.seat_tokens[1]
    current_admin = manager.rotate_admin_token(room)
    current_seat = manager.rotate_seat_token(room, 1)

    for mode, seat, token in (
        ("god", None, stale_admin),
        ("play", 1, stale_seat),
    ):
        socket = _CaptureWebSocket()
        with pytest.raises(CapabilityAuthorizationError, match="changed or was revoked"):
            await manager.connect(
                room,
                socket,  # type: ignore[arg-type]
                seat=seat,
                mode=mode,
                capability_token=token,
            )
        assert socket.accepted is False
        assert room.clients == {}

    for mode, seat, token in (
        ("god", None, current_admin),
        ("play", 1, current_seat),
    ):
        socket = _CaptureWebSocket()
        cid = await manager.connect(
            room,
            socket,  # type: ignore[arg-type]
            seat=seat,
            mode=mode,
            capability_token=token,
        )
        assert socket.accepted is True
        manager.disconnect(room, cid)
        await asyncio.sleep(0)

    await manager.aclose()


def test_delete_removes_persisted_room(tmp_path: Path) -> None:
    path = tmp_path / "rooms.sqlite"
    manager = _manager(path)
    room = manager.create_room(player_names=["A", "B", "C", "D", "E", "F"])
    assert manager.delete_room(room.id) is room
    manager.persistence.close()
    manager._persistence_closed = True
    restored = _manager(path)
    assert restored.rooms == {}
