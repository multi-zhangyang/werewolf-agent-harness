"""Append-only transcript primitives for multi-agent harness runs.

The game orchestrator already emits environment events and decision traces.
This module gives those records one evidence envelope for
read-only projection, integrity checks, offline analysis, and the console.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

TRANSCRIPT_SCHEMA_VERSION = "werewolf.harness.transcript.v1"
_MISSING = object()


class TranscriptIntegrityError(ValueError):
    """Raised when transcript evidence cannot be independently verified."""


@dataclass(frozen=True)
class ValidatedTranscript:
    """Immutable, canonical transcript evidence for offline evaluators.

    The live runner already creates these invariants while appending events,
    but consumers of a reloaded result must not assume that a JSON object is
    trustworthy merely because it has the expected keys.  Keeping the
    canonical rows and digest together prevents an evaluator from mixing a
    transcript from one run with a digest (or run id) from another.
    """

    run_id: str
    stable_digest: str
    metadata: dict[str, Any]
    counts_by_kind: dict[str, int]
    entries: tuple[dict[str, Any], ...]
    enclosing_digest_verified: bool

_REDACTED = "[redacted]"
_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "secret",
    "password",
    "x-api-key",
    "x-room-token",
    "admin_token",
    "seat_token",
    "seat_tokens",
    "access_token",
    "refresh_token",
    "id_token",
    "session_token",
    "client_secret",
    "private_key",
    "cookie",
    "set_cookie",
)
_TIMING_PAYLOAD_KEYS = {"_ts", "deadline_monotonic", "latency_seconds", "elapsed_seconds"}
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_-]{8,}|[A-Za-z0-9._~+/=-]*(?:api[_-]?key|secret|password)[A-Za-z0-9._~+/=-]{6,})\b"
)
_TOKEN_ASSIGNMENT_RE = re.compile(
    r"(?i)(^|[^A-Za-z0-9_])([\"']?(?:access[_-]?token|refresh[_-]?token|id[_-]?token|session[_-]?token|client[_-]?secret|private[_-]?key)[\"']?\s*[:=]\s*[\"']?)[^\s,;\"']{8,}"
)
_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)


def _redact(value: Any) -> Any:
    """Return a JSON-safe copy with credentials and room tokens removed."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key, raw_val in value.items():
            key = str(raw_key)
            lowered = key.lower().replace("-", "_")
            if any(fragment in lowered for fragment in _SENSITIVE_KEY_FRAGMENTS):
                result[key] = _REDACTED
            else:
                result[key] = _redact(raw_val)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def redact_sensitive(value: Any) -> Any:
    """Public redaction helper for harness summaries and transcript evidence."""
    return _redact(deepcopy(value))


def _redact_text(value: str) -> str:
    cleaned = str(value)

    def replace_url(match: re.Match[str]) -> str:
        return _safe_url_origin(match.group(0)) or _REDACTED

    cleaned = _URL_RE.sub(replace_url, cleaned)
    cleaned = _BEARER_RE.sub(f"Bearer {_REDACTED}", cleaned)
    cleaned = _TOKEN_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        cleaned,
    )
    cleaned = _JWT_RE.sub(_REDACTED, cleaned)
    return _SECRET_VALUE_RE.sub(_REDACTED, cleaned)


def _safe_url_origin(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    if not parsed.scheme or not parsed.hostname:
        return ""
    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, "", "", ""))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def payload_digest(payload: dict[str, Any]) -> str:
    """Public payload hash helper used by integrity tooling."""
    return _hash_payload(payload)


def _payload_for_stable_digest(payload: dict[str, Any], *, include_timing: bool) -> dict[str, Any]:
    """Return the payload representation used by transcript stable_digest."""
    if include_timing:
        return payload
    return _without_timing(payload)


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if str(key) not in _TIMING_PAYLOAD_KEYS
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    if isinstance(value, tuple):
        return [_without_timing(item) for item in value]
    return value


def _extract_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class HarnessEvent(BaseModel):
    """One normalized trace item emitted by the harness."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = TRANSCRIPT_SCHEMA_VERSION
    run_id: str
    seq: int
    kind: str
    ts_monotonic: float
    day: int | None = None
    phase: str | None = None
    seat: int | None = None
    visibility: str | None = None
    source_idx: int | None = None
    payload_hash: str
    payload: dict[str, Any] = Field(default_factory=dict)


class Transcript(BaseModel):
    """Append-only transcript with deterministic, redacted serialization."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = TRANSCRIPT_SCHEMA_VERSION
    run_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    entries: list[HarnessEvent] = Field(default_factory=list)

    def append(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        ts_monotonic: float | None = None,
        source_idx: int | None = None,
    ) -> HarnessEvent:
        """Append one normalized event and return the stored envelope."""
        safe_payload = _redact(deepcopy(payload))
        seq = len(self.entries) + 1
        event = HarnessEvent(
            run_id=self.run_id,
            seq=seq,
            kind=str(kind),
            ts_monotonic=float(time.monotonic() if ts_monotonic is None else ts_monotonic),
            day=_extract_int(safe_payload, "day"),
            phase=str(safe_payload.get("phase")) if safe_payload.get("phase") is not None else None,
            seat=_extract_int(safe_payload, "seat"),
            visibility=str(safe_payload.get("visibility")) if safe_payload.get("visibility") is not None else None,
            source_idx=source_idx,
            payload_hash=_hash_payload(safe_payload),
            payload=safe_payload,
        )
        self.entries.append(event)
        return event

    def counts_by_kind(self) -> dict[str, int]:
        return dict(Counter(entry.kind for entry in self.entries))

    def stable_digest(self, *, include_timing: bool = False) -> str:
        """Hash transcript content.

        Timing is excluded by default so equivalent environment runs can
        compare structural content without wall-clock noise.
        """
        rows: list[dict[str, Any]] = []
        for entry in self.entries:
            row = entry.model_dump()
            if not include_timing:
                row.pop("ts_monotonic", None)
                payload = _payload_for_stable_digest(dict(row.get("payload") or {}), include_timing=False)
                row["payload"] = payload
                row["payload_hash"] = _hash_payload(payload)
            rows.append(row)
        body = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "metadata": _redact(self.metadata),
            "entries": rows,
        }
        return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()

    def export(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "metadata": _redact(self.metadata),
            "counts_by_kind": self.counts_by_kind(),
            "stable_digest": self.stable_digest(),
            "entries": [entry.model_dump() for entry in self.entries],
        }

    def to_jsonl_rows(self) -> list[dict[str, Any]]:
        return [entry.model_dump() for entry in self.entries]


def validate_transcript_evidence(run: Any) -> ValidatedTranscript:
    """Validate and canonicalize the transcript attached to one run result.

    This is an in-memory counterpart to artifact verification for evaluators
    that receive a ``HarnessRunResult`` directly.  It verifies every integrity
    field that the transcript format can prove on its own.  The enclosing
    result's ``run_id`` is required.  ``transcript_digest`` is checked when
    present; older in-memory callers may omit it, and the returned
    ``enclosing_digest_verified`` flag lets downstream summary aggregation
    exclude such unbound evidence.
    """

    run_id = _evidence_text(_evidence_value(run, "run_id"))
    expected_digest = _evidence_text(_evidence_value(run, "transcript_digest"))
    if not run_id:
        raise TranscriptIntegrityError("run result is missing run_id provenance")

    raw_transcript = _evidence_value(run, "transcript")
    if not isinstance(raw_transcript, Mapping):
        raise TranscriptIntegrityError("run transcript must be an object")
    schema_version = raw_transcript.get("schema_version")
    if schema_version != TRANSCRIPT_SCHEMA_VERSION:
        raise TranscriptIntegrityError(
            f"unsupported transcript schema_version: {schema_version}"
        )
    if raw_transcript.get("run_id") != run_id:
        raise TranscriptIntegrityError("transcript run_id does not match result")
    embedded_digest = raw_transcript.get("stable_digest")
    if not _is_sha256(_evidence_text(embedded_digest)):
        raise TranscriptIntegrityError("transcript stable_digest is invalid")
    enclosing_digest_verified = bool(expected_digest)
    if expected_digest and embedded_digest != expected_digest:
        raise TranscriptIntegrityError(
            "transcript stable_digest does not match result provenance"
        )
    expected_digest = _evidence_text(embedded_digest)

    raw_metadata = raw_transcript.get("metadata")
    if not isinstance(raw_metadata, dict):
        raise TranscriptIntegrityError("transcript metadata must be an object")
    raw_counts = raw_transcript.get("counts_by_kind")
    if not isinstance(raw_counts, dict) or any(
        not isinstance(kind, str)
        or not kind
        or type(count) is not int
        or count < 0
        for kind, count in raw_counts.items()
    ):
        raise TranscriptIntegrityError("transcript counts_by_kind is invalid")
    raw_entries = raw_transcript.get("entries")
    if not isinstance(raw_entries, list):
        raise TranscriptIntegrityError("transcript entries must be a list")

    events: list[HarnessEvent] = []
    counts: Counter[str] = Counter()
    for expected_seq, raw_entry in enumerate(raw_entries, start=1):
        if not isinstance(raw_entry, dict):
            raise TranscriptIntegrityError("transcript entries must be objects")
        try:
            event = HarnessEvent.model_validate(raw_entry)
        except ValueError as err:
            raise TranscriptIntegrityError(
                f"invalid transcript row at seq {expected_seq}"
            ) from err
        if event.schema_version != TRANSCRIPT_SCHEMA_VERSION:
            raise TranscriptIntegrityError(
                f"transcript schema mismatch at seq {expected_seq}"
            )
        if event.run_id != run_id:
            raise TranscriptIntegrityError(
                f"transcript run_id mismatch at seq {expected_seq}"
            )
        if event.seq != expected_seq:
            raise TranscriptIntegrityError(
                f"transcript sequence mismatch at seq {expected_seq}"
            )
        if event.payload_hash != payload_digest(event.payload):
            raise TranscriptIntegrityError(
                f"transcript payload hash mismatch at seq {expected_seq}"
            )
        events.append(event)
        counts[event.kind] += 1

    if dict(counts) != raw_counts:
        raise TranscriptIntegrityError(
            "transcript counts_by_kind does not match entries"
        )
    transcript = Transcript(
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
        run_id=run_id,
        metadata=deepcopy(raw_metadata),
        entries=events,
    )
    if transcript.stable_digest() != expected_digest:
        raise TranscriptIntegrityError("transcript stable_digest cannot be recomputed")
    if redact_sensitive(dict(raw_transcript)) != dict(raw_transcript):
        raise TranscriptIntegrityError("transcript contains unredacted sensitive content")
    return ValidatedTranscript(
        run_id=run_id,
        stable_digest=expected_digest,
        metadata=deepcopy(raw_metadata),
        counts_by_kind=dict(raw_counts),
        entries=tuple(event.model_dump() for event in events),
        enclosing_digest_verified=enclosing_digest_verified,
    )


def validated_final_analysis(
    evidence: ValidatedTranscript,
    outer_analysis: Any = _MISSING,
) -> dict[str, Any] | None:
    """Return the unique final environment analysis bound to a transcript.

    A top-level ``run.analysis`` is a convenience copy and is not independent
    truth.  When supplied, it must exactly match the environment's final
    transcript event.  Older in-memory fixtures without an enclosing digest may
    fall back to that copy only when no analysis event exists; the evidence flag
    remains false, so persisted-row aggregation will not trust the result.
    """

    event_rows = [
        entry for entry in evidence.entries if entry.get("kind") == "event"
    ]
    candidates: list[tuple[int, Mapping[str, Any]]] = []
    for event_index, entry in enumerate(event_rows):
        payload = entry.get("payload")
        if not isinstance(payload, Mapping) or payload.get("type") != "analysis":
            continue
        analysis = payload.get("analysis")
        if not isinstance(analysis, Mapping):
            raise TranscriptIntegrityError(
                "transcript analysis event must contain an analysis object"
            )
        candidates.append((event_index, analysis))

    if len(candidates) > 1:
        raise TranscriptIntegrityError(
            "transcript must contain exactly one final analysis event"
        )
    if not candidates:
        legacy_outer = (
            dict(outer_analysis)
            if isinstance(outer_analysis, Mapping)
            else None
        )
        if legacy_outer and evidence.enclosing_digest_verified:
            raise TranscriptIntegrityError(
                "verified transcript is missing its final analysis event"
            )
        return deepcopy(legacy_outer) if legacy_outer else None

    event_index, transcript_analysis = candidates[0]
    if event_index != len(event_rows) - 1:
        raise TranscriptIntegrityError(
            "transcript analysis event must be the final environment event"
        )
    canonical = dict(transcript_analysis)
    if outer_analysis is not _MISSING and outer_analysis is not None:
        if not isinstance(outer_analysis, Mapping) or dict(outer_analysis) != canonical:
            raise TranscriptIntegrityError(
                "outer run.analysis does not match transcript analysis evidence"
            )
    return deepcopy(canonical)


def _evidence_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _evidence_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
