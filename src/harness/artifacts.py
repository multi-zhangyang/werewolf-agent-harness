"""Atomic, integrity-checked artifacts for legacy and generic harness runs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

from .core_runner import CORE_RUN_RESULT_VERSION, EnvironmentRunResult
from .core_spec import (
    CORE_RUN_MANIFEST_VERSION,
    ArtifactIntegrity,
    CoreRunManifest,
    CoreRunSpec,
)
from .spec import (
    LEGACY_TRANSCRIPT_INTEGRITY_VERSION,
    MANIFEST_SCHEMA_VERSION,
    RunManifest,
    RunSpec,
)
from .transcript import (
    TRANSCRIPT_SCHEMA_VERSION,
    HarnessEvent,
    Transcript,
    payload_digest,
    redact_sensitive,
)

if TYPE_CHECKING:
    from .runner import HarnessRunResult
    from .results import RunSummaryRow


VerifiedRunManifest: TypeAlias = RunManifest | CoreRunManifest
_RunSpec: TypeAlias = RunSpec | CoreRunSpec

_ARTIFACT_PATHS = {
    "manifest": "manifest.json",
    "summary": "summary.json",
    "transcript_jsonl": "transcript.jsonl",
}
_CONTENT_ARTIFACTS = ("summary", "transcript_jsonl")


class ArtifactIntegrityError(ValueError):
    """Raised when an artifact set is incomplete, unsafe, or tampered."""


@dataclass(frozen=True)
class VerifiedArtifactSnapshot:
    """One coherent, integrity-verified in-memory artifact read.

    Consumers that need semantic evidence must use this snapshot rather than
    re-reading files after verification. The raw bytes are retained solely for
    offline scanners that need to inspect the exact verified documents; they
    must never be logged or returned to an untrusted caller.
    """

    manifest: VerifiedRunManifest
    raw_manifest: dict[str, Any]
    summary: dict[str, Any]
    transcript_rows: list[dict[str, Any]]
    manifest_bytes: bytes
    content_bytes: dict[str, bytes]


@overload
def write_run_artifacts(
    result: HarnessRunResult,
    run_spec: RunSpec,
    root: str | Path,
) -> dict[str, str]: ...


@overload
def write_run_artifacts(
    result: EnvironmentRunResult,
    run_spec: CoreRunSpec,
    root: str | Path,
) -> dict[str, str]: ...


def write_run_artifacts(
    result: HarnessRunResult | EnvironmentRunResult,
    run_spec: _RunSpec,
    root: str | Path,
) -> dict[str, str]:
    """Write one result as exactly three artifacts, committing the manifest last."""
    manifest_kind, safe_spec = _validate_result_spec_pair(result, run_spec)
    (
        transcript_schema_version,
        transcript_rows,
        transcript_metadata,
        transcript_counts,
    ) = _validated_result_transcript(result)

    transcript_bytes = "".join(
        json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        for entry in transcript_rows
    ).encode("utf-8")
    summary = redact_sensitive(result.model_dump(exclude={"transcript"}))
    if manifest_kind == "legacy":
        # The embedded provenance spec has already been validated against the
        # supplied manifest spec and contains only ModelConfigManifest data.
        # Generic text redaction would strip safe endpoint paths and make this
        # canonical spec impossible to verify independently on read.
        summary["run_spec"] = cast(RunSpec, safe_spec).model_dump()
    summary_bytes = (
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n"
    ).encode("utf-8")
    artifact_integrity = {
        "summary": _integrity(summary_bytes),
        "transcript_jsonl": _integrity(transcript_bytes),
    }

    if manifest_kind == "legacy":
        legacy_spec = cast(RunSpec, safe_spec)
        manifest: VerifiedRunManifest = RunManifest(
            run=legacy_spec,
            transcript_schema_version=transcript_schema_version,
            transcript_integrity_version=LEGACY_TRANSCRIPT_INTEGRITY_VERSION,
            transcript_metadata=transcript_metadata,
            transcript_counts_by_kind=transcript_counts,
            artifact_paths=dict(_ARTIFACT_PATHS),
            artifact_integrity=artifact_integrity,
            transcript_digest=result.transcript_digest,
        )
    else:
        core_result = cast(EnvironmentRunResult, result)
        core_spec = cast(CoreRunSpec, safe_spec)
        manifest = CoreRunManifest(
            run=core_spec,
            run_spec_hash=core_spec.spec_hash,
            result_schema_version=core_result.schema_version,
            transcript_schema_version=transcript_schema_version,
            transcript_metadata=transcript_metadata,
            transcript_counts_by_kind=transcript_counts,
            artifact_paths=dict(_ARTIFACT_PATHS),
            artifact_integrity=artifact_integrity,
            transcript_digest=result.transcript_digest,
        )

    manifest_bytes = (
        json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2, default=str) + "\n"
    ).encode("utf-8")
    artifact_root = Path(root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_dir = artifact_root / result.run_id
    if run_dir.is_symlink():
        raise ArtifactIntegrityError("artifact run directory must not be a symlink")
    run_dir.mkdir(parents=False, exist_ok=True)
    if run_dir.resolve().parent != artifact_root:
        raise ArtifactIntegrityError("artifact run directory escapes artifact root")
    transcript_path = run_dir / _ARTIFACT_PATHS["transcript_jsonl"]
    summary_path = run_dir / _ARTIFACT_PATHS["summary"]
    manifest_path = run_dir / _ARTIFACT_PATHS["manifest"]

    _atomic_write_bytes(transcript_path, transcript_bytes)
    _atomic_write_bytes(summary_path, summary_bytes)
    # The manifest is the commit marker. A reader never accepts the content
    # files unless both match the manifest that was replaced last.
    _atomic_write_bytes(manifest_path, manifest_bytes)

    return {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "transcript_jsonl": str(transcript_path),
    }


def load_transcript_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return _parse_transcript_jsonl_bytes(Path(path).read_bytes())


def _parse_transcript_jsonl_bytes(content: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in content.decode("utf-8").splitlines():
        if raw := line.strip():
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("transcript JSONL rows must be objects")
            rows.append(value)
    return rows


def verify_run_artifacts(run_dir: str | Path) -> VerifiedRunManifest:
    """Verify one committed three-file set and return its versioned manifest."""
    return load_verified_artifact_snapshot(run_dir).manifest


def load_verified_artifact_snapshot(
    run_dir: str | Path,
) -> VerifiedArtifactSnapshot:
    """Read every artifact once, verify it, and return that verified read.

    This is the integrity boundary for offline semantic verifiers. In
    particular, callers must not call :func:`verify_run_artifacts` and then
    open artifact paths again, because a concurrent replacement would create a
    time-of-check/time-of-use gap.
    """
    root = Path(run_dir).resolve()
    expected_names = set(_ARTIFACT_PATHS.values())
    try:
        entries = list(root.iterdir())
    except OSError as err:
        raise ArtifactIntegrityError(f"artifact directory is unreadable: {root}") from err
    actual_names = {path.name for path in entries}
    if actual_names != expected_names:
        expected_label = sorted(expected_names)
        actual_label = sorted(actual_names)
        raise ArtifactIntegrityError(
            f"artifact file set mismatch: expected={expected_label} actual={actual_label}"
        )
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise ArtifactIntegrityError("artifact set must contain regular files only")

    try:
        raw_manifest_bytes = (root / _ARTIFACT_PATHS["manifest"]).read_bytes()
    except OSError as err:
        raise ArtifactIntegrityError("manifest is missing or unreadable") from err
    manifest, raw_manifest = _load_manifest_bytes(raw_manifest_bytes)
    if manifest.artifact_paths != _ARTIFACT_PATHS:
        raise ArtifactIntegrityError("manifest artifact paths are not canonical relative paths")
    if set(manifest.artifact_integrity) != set(_CONTENT_ARTIFACTS):
        raise ArtifactIntegrityError(
            "manifest artifact integrity records are incomplete or unexpected"
        )

    content_by_key: dict[str, bytes] = {}
    for key in _CONTENT_ARTIFACTS:
        expected = manifest.artifact_integrity[key]
        path = root / _ARTIFACT_PATHS[key]
        try:
            content = path.read_bytes()
        except OSError as err:
            raise ArtifactIntegrityError(f"artifact is unreadable: {key}") from err
        if _integrity(content) != expected:
            raise ArtifactIntegrityError(f"artifact integrity mismatch: {key}")
        content_by_key[key] = content

    try:
        summary = json.loads(content_by_key["summary"].decode("utf-8"))
        transcript_rows = _parse_transcript_jsonl_bytes(
            content_by_key["transcript_jsonl"]
        )
    except (OSError, json.JSONDecodeError, UnicodeError, ValueError) as err:
        raise ArtifactIntegrityError("artifact content is not valid JSON") from err
    if not isinstance(summary, dict):
        raise ArtifactIntegrityError("summary must be a JSON object")

    _verify_summary(summary, manifest, raw_manifest=raw_manifest)
    _verify_transcript_rows(
        transcript_rows,
        run_id=manifest.run.run_id,
        transcript_schema_version=manifest.transcript_schema_version,
    )
    if isinstance(manifest, CoreRunManifest):
        _verify_core_transcript_digest(transcript_rows, manifest)
    else:
        _verify_legacy_transcript_digest(transcript_rows, manifest, raw_manifest=raw_manifest)
    return VerifiedArtifactSnapshot(
        manifest=manifest,
        raw_manifest=raw_manifest,
        summary=summary,
        transcript_rows=transcript_rows,
        manifest_bytes=raw_manifest_bytes,
        content_bytes=content_by_key,
    )


def load_verified_run_summary(run_dir: str | Path) -> "RunSummaryRow":
    """Rebuild a trusted Werewolf summary from one verified artifact set.

    A summary JSONL row is only a resume/cache record.  This loader is the
    explicit path that restores evaluation trust: it validates the manifest,
    reads the committed summary and transcript once, reconstructs the full
    ``HarnessRunResult``, and asks ``run_summary_from_result`` to derive all
    metrics again.  Generic Core artifacts do not have a Werewolf summary row.
    """

    snapshot = load_verified_artifact_snapshot(run_dir)
    if not isinstance(snapshot.manifest, RunManifest):
        raise ArtifactIntegrityError(
            "verified artifact does not contain a legacy Werewolf run summary"
        )
    if (
        snapshot.raw_manifest.get("transcript_integrity_version")
        != LEGACY_TRANSCRIPT_INTEGRITY_VERSION
        or "transcript_metadata" not in snapshot.raw_manifest
        or "transcript_counts_by_kind" not in snapshot.raw_manifest
    ):
        raise ArtifactIntegrityError(
            "legacy artifact lacks transcript reconstruction provenance"
        )

    from .results import run_summary_from_result
    from .runner import HarnessRunResult

    transcript = {
        "schema_version": snapshot.manifest.transcript_schema_version,
        "run_id": snapshot.manifest.run.run_id,
        "metadata": snapshot.manifest.transcript_metadata,
        "counts_by_kind": snapshot.manifest.transcript_counts_by_kind,
        "stable_digest": snapshot.manifest.transcript_digest,
        "entries": snapshot.transcript_rows,
    }
    payload = dict(snapshot.summary)
    payload["transcript"] = transcript
    # The manifest is the authoritative source for the identity fields that
    # the row factory binds to the transcript.
    payload["run_id"] = snapshot.manifest.run.run_id
    payload["run_spec"] = snapshot.manifest.run.model_dump()
    payload["run_spec_hash"] = snapshot.manifest.run.spec_hash
    payload["role_seed"] = snapshot.manifest.run.role_seed
    payload["actor_seed"] = snapshot.manifest.run.actor_seed
    payload["orchestrator_seed"] = snapshot.manifest.run.orchestrator_seed
    payload["transcript_digest"] = snapshot.manifest.transcript_digest
    payload["event_count"] = sum(
        1 for row in snapshot.transcript_rows if row.get("kind") == "event"
    )
    payload["decision_trace_count"] = sum(
        1 for row in snapshot.transcript_rows if row.get("kind") == "decision"
    )
    result = HarnessRunResult.model_validate(payload)
    return run_summary_from_result(result)


def _validate_result_spec_pair(
    result: HarnessRunResult | EnvironmentRunResult,
    run_spec: _RunSpec,
) -> tuple[str, RunSpec | CoreRunSpec]:
    if isinstance(run_spec, RunSpec) and not isinstance(result, EnvironmentRunResult):
        legacy_result = cast(Any, result)
        if not callable(getattr(legacy_result, "model_dump", None)) or not hasattr(
            legacy_result, "run_spec"
        ):
            raise TypeError("legacy artifacts require a HarnessRunResult-compatible result")
        if legacy_result.run_id != run_spec.run_id:
            raise ArtifactIntegrityError("result run_id does not match RunSpec")
        if legacy_result.run_spec_hash != run_spec.spec_hash:
            raise ArtifactIntegrityError("result run_spec_hash does not match RunSpec")
        try:
            embedded = (
                RunSpec.model_validate(legacy_result.run_spec)
                if legacy_result.run_spec
                else run_spec
            )
        except ValueError as err:
            raise ArtifactIntegrityError("result contains an invalid embedded RunSpec") from err
        if embedded.run_id != run_spec.run_id or embedded.spec_hash != run_spec.spec_hash:
            raise ArtifactIntegrityError("result embedded RunSpec does not match supplied RunSpec")
        return "legacy", embedded

    if isinstance(result, EnvironmentRunResult) and isinstance(run_spec, CoreRunSpec):
        if result.run_id != run_spec.run_id:
            raise ArtifactIntegrityError("result run_id does not match CoreRunSpec")
        if result.run_spec_hash != run_spec.spec_hash:
            raise ArtifactIntegrityError("result run_spec_hash does not match CoreRunSpec")
        if (
            result.environment_id != run_spec.environment.id
            or result.environment_version != run_spec.environment.version
        ):
            raise ArtifactIntegrityError("result environment does not match CoreRunSpec")
        if result.schema_version != CORE_RUN_RESULT_VERSION:
            raise ArtifactIntegrityError(
                f"unsupported core result schema_version: {result.schema_version}"
            )
        return "core", run_spec

    raise TypeError(
        "write_run_artifacts requires HarnessRunResult + RunSpec or "
        "EnvironmentRunResult + CoreRunSpec"
    )


def _validated_result_transcript(
    result: HarnessRunResult | EnvironmentRunResult,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    transcript = result.transcript
    if not isinstance(transcript, dict):
        raise ArtifactIntegrityError("result transcript must be an object")
    if transcript.get("run_id") != result.run_id:
        raise ArtifactIntegrityError("transcript run_id does not match result")
    if transcript.get("stable_digest") != result.transcript_digest:
        raise ArtifactIntegrityError("transcript stable_digest does not match result")
    schema_version = transcript.get("schema_version")
    if schema_version != TRANSCRIPT_SCHEMA_VERSION:
        raise ArtifactIntegrityError(f"unsupported transcript schema_version: {schema_version}")
    rows = transcript.get("entries")
    if not isinstance(rows, list):
        raise ArtifactIntegrityError("result transcript entries must be a list")
    metadata = transcript.get("metadata")
    if not isinstance(metadata, dict):
        raise ArtifactIntegrityError("result transcript metadata must be an object")
    counts = transcript.get("counts_by_kind")
    if not isinstance(counts, dict) or any(
        not isinstance(kind, str) or type(count) is not int or count < 0
        for kind, count in counts.items()
    ):
        raise ArtifactIntegrityError("result transcript counts_by_kind is invalid")
    _verify_transcript_rows(
        rows,
        run_id=result.run_id,
        transcript_schema_version=schema_version,
    )
    actual_counts: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("kind") or "")
        actual_counts[kind] = actual_counts.get(kind, 0) + 1
    if counts != actual_counts:
        raise ArtifactIntegrityError("result transcript counts_by_kind does not match entries")
    try:
        reconstructed = Transcript(
            schema_version=schema_version,
            run_id=result.run_id,
            metadata=metadata,
            entries=[HarnessEvent.model_validate(row) for row in rows],
        )
    except ValueError as err:
        raise ArtifactIntegrityError("result transcript is invalid") from err
    if reconstructed.stable_digest() != result.transcript_digest:
        raise ArtifactIntegrityError("result transcript digest cannot be recomputed")
    if redact_sensitive(transcript) != transcript:
        raise ArtifactIntegrityError("result transcript contains unredacted sensitive content")
    return schema_version, rows, metadata, counts


def _load_manifest_bytes(content: bytes) -> tuple[VerifiedRunManifest, dict[str, Any]]:
    try:
        raw = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeError) as err:
        raise ArtifactIntegrityError("manifest is missing or invalid") from err
    if not isinstance(raw, dict):
        raise ArtifactIntegrityError("manifest must be a JSON object")
    schema_version = raw.get("schema_version")
    try:
        if schema_version == MANIFEST_SCHEMA_VERSION:
            return RunManifest.model_validate(raw), raw
        if schema_version == CORE_RUN_MANIFEST_VERSION:
            return CoreRunManifest.model_validate(raw), raw
    except ValueError as err:
        raise ArtifactIntegrityError("manifest is missing or invalid") from err
    raise ArtifactIntegrityError(f"unsupported manifest schema_version: {schema_version}")


def _verify_summary(
    summary: dict[str, Any],
    manifest: VerifiedRunManifest,
    *,
    raw_manifest: dict[str, Any],
) -> None:
    if summary.get("run_id") != manifest.run.run_id:
        raise ArtifactIntegrityError("summary run_id does not match manifest")
    if isinstance(manifest, RunManifest):
        _verify_legacy_summary_run_spec(summary, manifest)
    expected_spec_hash = (
        manifest.run.spec_hash if isinstance(manifest, RunManifest) else manifest.run_spec_hash
    )
    summary_spec_hash = summary.get("run_spec_hash")
    if summary_spec_hash != expected_spec_hash:
        raw_run = raw_manifest.get("run")
        legacy_raw_hash = (
            _canonical_hash(raw_run)
            if isinstance(manifest, RunManifest) and isinstance(raw_run, dict)
            else None
        )
        if summary_spec_hash != legacy_raw_hash:
            raise ArtifactIntegrityError("summary run_spec_hash does not match manifest")
    if summary.get("transcript_digest") != manifest.transcript_digest:
        raise ArtifactIntegrityError("summary transcript_digest does not match manifest")

    if isinstance(manifest, CoreRunManifest):
        if summary.get("schema_version") != manifest.result_schema_version:
            raise ArtifactIntegrityError("summary schema_version does not match manifest")
        if manifest.result_schema_version != CORE_RUN_RESULT_VERSION:
            raise ArtifactIntegrityError(
                f"unsupported core result schema_version: {manifest.result_schema_version}"
            )
        if (
            summary.get("environment_id") != manifest.run.environment.id
            or summary.get("environment_version") != manifest.run.environment.version
        ):
            raise ArtifactIntegrityError("summary environment does not match manifest")


def _verify_legacy_summary_run_spec(
    summary: dict[str, Any],
    manifest: RunManifest,
) -> None:
    raw_run_spec = summary.get("run_spec")
    if not isinstance(raw_run_spec, dict):
        raise ArtifactIntegrityError("summary embedded RunSpec must be a JSON object")
    try:
        embedded_run_spec = RunSpec.model_validate(raw_run_spec)
    except ValueError as err:
        raise ArtifactIntegrityError("summary contains an invalid embedded RunSpec") from err
    if embedded_run_spec.spec_hash != manifest.run.spec_hash:
        raise ArtifactIntegrityError("summary embedded RunSpec does not match manifest")


def _verify_transcript_rows(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    transcript_schema_version: str,
) -> None:
    if transcript_schema_version != TRANSCRIPT_SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            f"unsupported transcript schema_version: {transcript_schema_version}"
        )
    for expected_seq, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise ArtifactIntegrityError("transcript rows must be JSON objects")
        try:
            event = HarnessEvent.model_validate(raw)
        except ValueError as err:
            raise ArtifactIntegrityError(f"invalid transcript row at seq {expected_seq}") from err
        if event.schema_version != transcript_schema_version:
            raise ArtifactIntegrityError(f"transcript schema mismatch at seq {expected_seq}")
        if event.run_id != run_id:
            raise ArtifactIntegrityError(f"transcript run_id mismatch at seq {expected_seq}")
        if event.seq != expected_seq:
            raise ArtifactIntegrityError(f"transcript sequence mismatch at seq {expected_seq}")
        if event.payload_hash != payload_digest(event.payload):
            raise ArtifactIntegrityError(f"transcript payload hash mismatch at seq {expected_seq}")


def _verify_core_transcript_digest(
    rows: list[dict[str, Any]],
    manifest: CoreRunManifest,
) -> None:
    if redact_sensitive(manifest.transcript_metadata) != manifest.transcript_metadata:
        raise ArtifactIntegrityError("core manifest transcript metadata is not redacted")
    counts: dict[str, int] = {}
    events: list[HarnessEvent] = []
    for row in rows:
        event = HarnessEvent.model_validate(row)
        events.append(event)
        counts[event.kind] = counts.get(event.kind, 0) + 1
    if counts != manifest.transcript_counts_by_kind:
        raise ArtifactIntegrityError("transcript counts do not match core manifest")
    transcript = Transcript(
        schema_version=manifest.transcript_schema_version,
        run_id=manifest.run.run_id,
        metadata=manifest.transcript_metadata,
        entries=events,
    )
    if transcript.stable_digest() != manifest.transcript_digest:
        raise ArtifactIntegrityError("transcript digest does not match core manifest")


def _verify_legacy_transcript_digest(
    rows: list[dict[str, Any]],
    manifest: RunManifest,
    *,
    raw_manifest: dict[str, Any],
) -> None:
    """Rebuild the legacy transcript digest from manifest-owned evidence.

    Older v2 manifests did not carry these fields and remain readable for
    compatibility, but newly written manifests can now be verified without
    loading the original live result object.
    """
    # Keep reading pre-metadata v2 artifacts. They still receive row/hash and
    # file-integrity checks, but their original stable digest cannot be
    # reconstructed without the omitted metadata.
    integrity_version = raw_manifest.get("transcript_integrity_version")
    if integrity_version not in (None, LEGACY_TRANSCRIPT_INTEGRITY_VERSION):
        raise ArtifactIntegrityError(
            f"unsupported legacy transcript integrity version: {integrity_version}"
        )
    metadata_present = "transcript_metadata" in raw_manifest
    counts_present = "transcript_counts_by_kind" in raw_manifest
    if not metadata_present and not counts_present:
        if integrity_version is not None:
            raise ArtifactIntegrityError(
                "legacy transcript integrity version requires reconstruction fields"
            )
        return
    if metadata_present != counts_present:
        raise ArtifactIntegrityError("legacy manifest transcript reconstruction fields are incomplete")
    if redact_sensitive(manifest.transcript_metadata) != manifest.transcript_metadata:
        raise ArtifactIntegrityError("legacy manifest transcript metadata is not redacted")
    counts: dict[str, int] = {}
    events: list[HarnessEvent] = []
    for row in rows:
        event = HarnessEvent.model_validate(row)
        events.append(event)
        counts[event.kind] = counts.get(event.kind, 0) + 1
    if counts != manifest.transcript_counts_by_kind:
        raise ArtifactIntegrityError("legacy transcript counts do not match manifest")
    transcript = Transcript(
        schema_version=manifest.transcript_schema_version,
        run_id=manifest.run.run_id,
        metadata=manifest.transcript_metadata,
        entries=events,
    )
    if transcript.stable_digest() != manifest.transcript_digest:
        raise ArtifactIntegrityError("transcript digest does not match legacy manifest")


def _integrity(content: bytes) -> ArtifactIntegrity:
    return ArtifactIntegrity(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))


def _canonical_hash(value: Any) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
