"""Durable, credential-free persistence for interactive rooms.

The live :class:`~src.api.room_manager.Room` object intentionally contains
credentials and asyncio/runtime objects that must never be serialized.  This
module provides the small storage boundary used by ``RoomManager``:

* SQLite is opt-in (the manager remains in-memory by default).
* Rows contain one canonical JSON payload and a SHA-256 integrity digest.
* Capability credentials are represented by salted PBKDF2 hashes only.
* Obvious credentials in a caller-supplied payload fail closed instead of
  being silently persisted.

The adapter is synchronous on purpose.  Room lifecycle mutations already run
on one event-loop thread and a short SQLite transaction gives a stronger
durability guarantee than scheduling an unawaited background write.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, SchemaError


PERSISTENCE_SCHEMA_VERSION = "werewolf.room.persistence.v1"
_HASH_ALGORITHM = "pbkdf2_sha256"
_HASH_ITERATIONS = 120_000
_SALT_BYTES = 16
_DIGEST_BYTES = 32
_TOKEN_RE = re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._~+/=-]+)\b")
_SENSITIVE_KEY_RE = re.compile(
    r"(?i)(?:^|_)(?:api[_-]?key|authorization|bearer|password|secret|token|room[_-]?token|admin[_-]?token|seat[_-]?tokens?|access[_-]?token)(?:$|_)"
)


class PersistenceError(RuntimeError):
    """Base class for durable-room storage failures."""


class PersistenceIntegrityError(PersistenceError):
    """A row was altered or is not a supported persistence schema."""


class PersistenceCredentialError(PersistenceError):
    """A caller attempted to write a credential-bearing value."""


class RoomPersistence(Protocol):
    """Minimal adapter contract consumed by ``RoomManager``."""

    def save_record(self, record: Mapping[str, Any]) -> None:
        ...

    def load_records(self) -> list[dict[str, Any]]:
        ...

    def delete_room(self, room_id: str) -> None:
        ...

    def close(self) -> None:
        ...


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_capability(token: str) -> str:
    """Return a salted, non-reversible capability hash.

    The encoded format is deliberately self-contained so verification still
    works after process restart without storing a server-wide secret.
    """

    if not isinstance(token, str) or not token:
        raise ValueError("capability token must be a non-empty string")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", token.encode("utf-8"), salt, _HASH_ITERATIONS, dklen=_DIGEST_BYTES
    )
    return f"{_HASH_ALGORITHM}${_HASH_ITERATIONS}${_b64(salt)}${_b64(digest)}"


def verify_capability(token: str | None, encoded: str | None) -> bool:
    """Constant-time verification for a hash produced by ``hash_capability``."""

    if not isinstance(token, str) or not token or not isinstance(encoded, str):
        return False
    try:
        algorithm, iterations_raw, salt_raw, digest_raw = encoded.split("$", 3)
        if algorithm != _HASH_ALGORITHM:
            return False
        iterations = int(iterations_raw)
        if iterations < 10_000 or iterations > 2_000_000:
            return False
        salt = _unb64(salt_raw)
        expected = _unb64(digest_raw)
        if len(salt) < 8 or len(expected) != _DIGEST_BYTES:
            return False
    except (TypeError, ValueError, binascii.Error):  # type: ignore[name-defined]
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256", token.encode("utf-8"), salt, iterations, dklen=len(expected)
    )
    return hmac.compare_digest(actual, expected)


def _valid_capability_hash(encoded: object) -> bool:
    """Validate the self-contained hash encoding without a plaintext token."""
    if not isinstance(encoded, str):
        return False
    try:
        algorithm, iterations_raw, salt_raw, digest_raw = encoded.split("$", 3)
        iterations = int(iterations_raw)
        salt = _unb64(salt_raw)
        digest = _unb64(digest_raw)
    except (TypeError, ValueError, binascii.Error):
        return False
    return bool(
        algorithm == _HASH_ALGORITHM
        and 10_000 <= iterations <= 2_000_000
        and len(salt) >= 8
        and len(digest) == _DIGEST_BYTES
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as err:
        raise PersistenceError("room persistence payload is not canonical JSON") from err


def _credential_key(key: str) -> bool:
    normalized = str(key).lower().replace("-", "_")
    # Only the explicit capability-hash paths handled by
    # ``_is_capability_hash_field`` are exempt. A generic ``*_hash`` suffix
    # must not turn credential fields such as ``api_key_hash`` into an
    # arbitrary-string persistence channel.
    if normalized.endswith("_configured") or normalized.endswith("_version"):
        return False
    return bool(_SENSITIVE_KEY_RE.search(normalized))


def _is_capability_hash_field(key: str, *, path: str) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return bool(
        "capability" in normalized
        or "token_hash" in normalized
        or "token_hashes" in normalized
        or (
            normalized in {"hash", "revoked_hashes"}
            and ".capabilities" in path
        )
    )


def _assert_capability_hash_value(value: Any, *, path: str) -> None:
    if value in (None, ""):
        return
    values = value if isinstance(value, (list, tuple)) else [value]
    if not all(_valid_capability_hash(item) for item in values):
        raise PersistenceCredentialError(
            f"capability hash field has an invalid encoding at {path}"
        )


def _is_valid_json_schema_response_format(value: Any) -> bool:
    """Return whether ``value`` is a structurally valid JSON-schema format.

    Persistence is a trust boundary, so the exemption for schema property
    names must not be enabled merely because an arbitrary mapping happens to
    contain a ``properties`` key.  ``ModelConfig`` validates this shape before
    normal room writes; validating it here also keeps direct persistence users
    fail-closed when they provide malformed data.
    """

    if not isinstance(value, Mapping) or value.get("type") != "json_schema":
        return False
    descriptor = value.get("json_schema")
    if not isinstance(descriptor, Mapping):
        return False
    schema = descriptor.get("schema")
    if not isinstance(schema, Mapping):
        return False
    try:
        Draft202012Validator.check_schema(dict(schema))
    except (SchemaError, TypeError, ValueError):
        return False
    return True


def _assert_safe_payload(
    value: Any,
    *,
    path: str = "payload",
    _response_format_context: bool = False,
    _response_format_descriptor: bool = False,
    _json_schema_context: bool = False,
) -> None:
    """Reject credentials rather than relying on best-effort redaction."""

    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"

            # A validated response_format contains one JSON Schema subtree.
            # Enter that subtree explicitly so only JSON Schema ``properties``
            # member names are treated as identifiers rather than credential
            # fields.  Values inside each property schema still go through the
            # normal recursive credential scanner.
            if (
                _response_format_context
                and key == "json_schema"
                and isinstance(raw_value, Mapping)
            ):
                _assert_safe_payload(
                    raw_value,
                    path=child_path,
                    _response_format_descriptor=True,
                )
                continue
            if (
                _response_format_descriptor
                and key == "schema"
                and isinstance(raw_value, Mapping)
            ):
                _assert_safe_payload(
                    raw_value,
                    path=child_path,
                    _json_schema_context=True,
                )
                continue
            if (
                _json_schema_context
                and key == "properties"
                and isinstance(raw_value, Mapping)
            ):
                # Property names (for example ``api_key`` or ``token_count``)
                # are part of the user's public schema, not stored secrets.
                # Scan each property's schema value, but deliberately do not
                # apply ``_credential_key`` to the member name itself.
                for property_name, property_schema in raw_value.items():
                    _assert_safe_payload(
                        property_schema,
                        path=f"{child_path}.{property_name}",
                        _json_schema_context=True,
                    )
                continue

            if key == "response_format" and _is_valid_json_schema_response_format(raw_value):
                _assert_safe_payload(
                    raw_value,
                    path=child_path,
                    _response_format_context=True,
                )
                continue

            if _is_capability_hash_field(key, path=path):
                _assert_capability_hash_value(raw_value, path=child_path)
            elif _credential_key(key):
                # Empty values and explicit redaction markers are harmless;
                # non-empty values under credential keys are never accepted.
                if raw_value not in (None, "", False, "[redacted]"):
                    raise PersistenceCredentialError(
                        f"credential-bearing field is not persistable at {child_path}"
                    )
            _assert_safe_payload(
                raw_value,
                path=child_path,
                _json_schema_context=_json_schema_context,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_safe_payload(
                item,
                path=f"{path}[{index}]",
                _response_format_context=_response_format_context,
                _response_format_descriptor=_response_format_descriptor,
                _json_schema_context=_json_schema_context,
            )
        return
    if isinstance(value, str):
        if _TOKEN_RE.search(value):
            raise PersistenceCredentialError("credential-like text is not persistable")
        try:
            parsed = urlsplit(value)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.scheme in {"http", "https"}:
            if parsed.username is not None or parsed.password is not None:
                raise PersistenceCredentialError("URL userinfo is not persistable")
            query = parsed.query.lower()
            if any(fragment in query for fragment in ("api_key", "apikey", "token", "secret", "password", "key=")):
                raise PersistenceCredentialError("credential-bearing URL query is not persistable")


class SQLiteRoomPersistence:
    """Small SQLite-backed room store.

    The database contains no token plaintext. ``payload_sha256`` detects
    partial writes and alterations where the digest was not recomputed. It is
    an integrity checksum, not an authenticated signature against an attacker
    who can rewrite the database.
    """

    def __init__(self, path: str | Path) -> None:
        raw_path = str(path)
        self._memory = raw_path == ":memory:"
        self.path = Path(raw_path) if not self._memory else None
        if self.path is not None:
            self.path = self.path.expanduser()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Capability databases are private by default.  Do not loosen an
            # existing mode because an operator may intentionally be stricter.
            try:
                self.path.touch(exist_ok=True)
                self.path.chmod(self.path.stat().st_mode & 0o600)
            except OSError as err:
                raise PersistenceError("cannot initialize room persistence path") from err
        try:
            self._conn = sqlite3.connect(
                raw_path,
                check_same_thread=False,
                isolation_level=None,
            )
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=DELETE")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rooms (
                    room_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL
                )
                """
            )
        except sqlite3.Error as err:
            raise PersistenceError("cannot initialize room persistence database") from err
        self._lock = threading.RLock()
        self._closed = False

    def save_record(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        if payload.get("schema_version") != PERSISTENCE_SCHEMA_VERSION:
            raise PersistenceIntegrityError("unsupported room persistence schema")
        room_id = str(payload.get("room_id") or "")
        if not room_id or any(ch in room_id for ch in "/\\\x00"):
            raise PersistenceIntegrityError("invalid persisted room id")
        _assert_safe_payload(payload)
        body = _canonical_json(payload)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        now = time.time()
        with self._lock:
            if self._closed:
                raise PersistenceError("room persistence is closed")
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    """
                    INSERT INTO rooms(room_id, schema_version, updated_at, payload_json, payload_sha256)
                    VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(room_id) DO UPDATE SET
                        schema_version=excluded.schema_version,
                        updated_at=excluded.updated_at,
                        payload_json=excluded.payload_json,
                        payload_sha256=excluded.payload_sha256
                    """,
                    (room_id, PERSISTENCE_SCHEMA_VERSION, now, body, digest),
                )
                self._conn.execute("COMMIT")
            except sqlite3.Error as err:
                try:
                    self._conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise PersistenceError("cannot durably save room") from err

    def load_records(self) -> list[dict[str, Any]]:
        if self._closed:
            raise PersistenceError("room persistence is closed")
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT room_id, schema_version, payload_json, payload_sha256 FROM rooms ORDER BY room_id"
                ).fetchall()
        except sqlite3.Error as err:
            raise PersistenceError("cannot load persisted rooms") from err
        records: list[dict[str, Any]] = []
        for room_id, schema_version, body, expected_digest in rows:
            if schema_version != PERSISTENCE_SCHEMA_VERSION:
                raise PersistenceIntegrityError("unsupported room persistence schema")
            if not isinstance(body, str) or not isinstance(expected_digest, str):
                raise PersistenceIntegrityError("malformed persisted room row")
            actual_digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if not hmac.compare_digest(actual_digest, expected_digest):
                raise PersistenceIntegrityError("persisted room integrity check failed")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as err:
                raise PersistenceIntegrityError("persisted room JSON is invalid") from err
            if not isinstance(payload, dict) or payload.get("room_id") != room_id:
                raise PersistenceIntegrityError("persisted room identity mismatch")
            if payload.get("schema_version") != PERSISTENCE_SCHEMA_VERSION:
                raise PersistenceIntegrityError("persisted room payload schema mismatch")
            _assert_safe_payload(payload)
            records.append(payload)
        return records

    def delete_room(self, room_id: str) -> None:
        if self._closed:
            raise PersistenceError("room persistence is closed")
        try:
            with self._lock:
                self._conn.execute("DELETE FROM rooms WHERE room_id = ?", (str(room_id),))
        except sqlite3.Error as err:
            raise PersistenceError("cannot delete persisted room") from err

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except sqlite3.Error as err:
                raise PersistenceError("cannot close room persistence") from err
            finally:
                self._closed = True

    def __enter__(self) -> "SQLiteRoomPersistence":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


__all__ = [
    "PERSISTENCE_SCHEMA_VERSION",
    "PersistenceCredentialError",
    "PersistenceError",
    "PersistenceIntegrityError",
    "RoomPersistence",
    "SQLiteRoomPersistence",
    "hash_capability",
    "verify_capability",
]
