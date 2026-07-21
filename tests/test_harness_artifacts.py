"""Harness manifest and artifact tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import src.harness.artifacts as artifact_module
from src.harness.artifacts import (
    ArtifactIntegrityError,
    load_transcript_jsonl,
    load_verified_artifact_snapshot,
    load_verified_run_summary,
    verify_run_artifacts,
    write_run_artifacts,
)
from src.harness.core_runner import EnvironmentRunResult
from src.harness.core_spec import CoreRunManifest, CoreRunSpec, EnvironmentRef
from src.harness.runner import HarnessRunResult
from src.harness.results import RunSummaryRow, is_verified_summary_row
from src.harness.spec import ModelConfigManifest, RunSpec
from src.harness.summary import summarize_runs
from src.harness.transcript import Transcript
from src.llm.models import ModelConfig


def _spec() -> RunSpec:
    return RunSpec(
        run_id="artifact-run",
        player_names=["A", "B", "C", "D", "E", "F"],
        role_deck=["werewolf", "werewolf", "seer", "villager", "villager", "villager"],
        role_seed=1,
        actor_seed=2,
        orchestrator_seed=3,
        default_model=ModelConfigManifest.from_config(ModelConfig(
            provider="openai",
            model="model-a",
            api_base="https://user:password@example.invalid/v1?token=secret",
            api_key="never-store-this",
        )),
    )


def _result(spec: RunSpec) -> HarnessRunResult:
    transcript = Transcript(run_id=spec.run_id, metadata={"run_spec_hash": spec.spec_hash})
    transcript.append("event", {"type": "phase_started", "phase": "setup", "day": 0})
    exported = transcript.export()
    return HarnessRunResult(
        run_id=spec.run_id,
        status="completed",
        winner="village",
        days=2,
        elapsed_seconds=1.25,
        run_spec_hash=spec.spec_hash,
        role_seed=1,
        actor_seed=2,
        orchestrator_seed=3,
        run_spec=spec.model_dump(),
        event_count=1,
        transcript_digest=exported["stable_digest"],
        transcript=exported,
        analysis={
            "winner": "village",
            "days": 2,
            "agent_strategy_metrics": {
                "schema_version": "werewolf.agent-strategy-metrics.v1",
                "belief_observation_count": 3,
                "belief_brier": 0.125,
                "seats": [{"seat": 1, "belief_brier": 0.125}],
            },
        },
    )


def _core_spec() -> CoreRunSpec:
    return CoreRunSpec(
        run_id="generic-artifact-run",
        environment=EnvironmentRef(id="counter", version="1"),
        environment_config={"target": 2},
        seeds={"environment": 17},
        metadata={"suite": "generic-artifact-test"},
    )


def _core_result(spec: CoreRunSpec) -> EnvironmentRunResult:
    transcript = Transcript(run_id=spec.run_id, metadata={"run_spec_hash": spec.spec_hash})
    transcript.append("event", {
        "type": "counter_incremented",
        "actor_id": "counter-alpha",
        "value": 1,
        "api_key": "sk-test-secret-must-not-be-written",
    })
    exported = transcript.export()
    return EnvironmentRunResult(
        run_id=spec.run_id,
        status="completed",
        environment_id=spec.environment.id,
        environment_version=spec.environment.version,
        run_spec_hash=spec.spec_hash,
        elapsed_seconds=0.01,
        outcome={"value": 1},
        metrics={"api_key": "sk-test-secret-must-not-be-written"},
        transcript_digest=exported["stable_digest"],
        transcript=exported,
    )


def _replace_summary_and_refresh_integrity(
    paths: dict[str, str],
    summary: dict[str, object],
) -> None:
    summary_path = Path(paths["summary"])
    manifest_path = Path(paths["manifest"])
    summary_bytes = (
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n"
    ).encode("utf-8")
    summary_path.write_bytes(summary_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_integrity"]["summary"] = {
        "sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "bytes": len(summary_bytes),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_model_manifest_preserves_endpoint_path_but_never_credentials():
    manifest = _spec().default_model
    assert manifest is not None
    assert manifest.api_base == "https://example.invalid/v1"
    dumped = json.dumps(manifest.model_dump())
    assert "never-store-this" not in dumped
    assert "password" not in dumped
    assert "token=secret" not in dumped


def test_model_manifest_records_structured_request_options_without_secrets():
    manifest = ModelConfigManifest.from_config(ModelConfig(
        provider="openai",
        model="model-a",
        api_key="sk-manifest-key-123456789",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "decision",
                "description": "do not echo sk-description-secret-123456789",
                "schema": {
                    "type": "object",
                    "properties": {"api_key": {"type": "string"}},
                },
            },
        },
        reasoning={"effort": "high", "api_key": "sk-reasoning-secret-123456789"},
        thinking={"type": "enabled", "secret": "sk-thinking-secret-123456789"},
    ))

    assert manifest.response_format is not None
    assert manifest.response_format["json_schema"]["schema"]["properties"]["api_key"] == {
        "type": "string"
    }
    assert "sk-description-secret" not in json.dumps(manifest.model_dump())
    assert manifest.reasoning == {"effort": "high", "api_key": "[redacted]"}
    assert manifest.thinking == {"type": "enabled", "secret": "[redacted]"}


def test_model_request_options_participate_in_run_spec_hash():
    base = ModelConfig(provider="openai", model="model-a", api_key="test-key")
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "decision",
            "schema": {"type": "object", "additionalProperties": False},
        },
    }
    configs = [
        base,
        base.model_copy(update={"response_format": response_format}),
        base.model_copy(update={"reasoning": {"effort": "high"}}),
        base.model_copy(update={"thinking": {"type": "enabled", "budget_tokens": 1024}}),
    ]
    hashes = {
        _spec().model_copy(update={
            "default_model": ModelConfigManifest.from_config(config),
        }).spec_hash
        for config in configs
    }

    assert len(hashes) == len(configs)


def test_run_artifacts_write_only_manifest_summary_and_transcript(tmp_path: Path):
    spec = _spec()
    paths = write_run_artifacts(_result(spec), spec, tmp_path)
    run_dir = Path(paths["run_dir"])

    assert sorted(path.name for path in run_dir.iterdir()) == [
        "manifest.json",
        "summary.json",
        "transcript.jsonl",
    ]
    assert set(paths) == {"run_dir", "manifest", "summary", "transcript_jsonl"}
    assert load_transcript_jsonl(paths["transcript_jsonl"])[0]["payload"]["type"] == "phase_started"

    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert "transcript" not in summary
    assert summary["analysis"]["agent_strategy_metrics"] == {
        "schema_version": "werewolf.agent-strategy-metrics.v1",
        "belief_observation_count": 3,
        "belief_brier": 0.125,
        "seats": [{"seat": 1, "belief_brier": 0.125}],
    }
    assert manifest["run"]["run_id"] == spec.run_id
    serialized = json.dumps({"summary": summary, "manifest": manifest})
    assert "never-store-this" not in serialized
    assert "social_spec" not in serialized
    assert "interaction_graph" not in serialized
    assert "replay_capability" not in serialized
    verified = verify_run_artifacts(run_dir)
    assert verified.run.run_id == spec.run_id
    assert verified.transcript_digest == _result(spec).transcript_digest
    assert set(verified.artifact_integrity) == {"summary", "transcript_jsonl"}
    assert verified.artifact_paths == {
        "manifest": "manifest.json",
        "summary": "summary.json",
        "transcript_jsonl": "transcript.jsonl",
    }


def test_verified_artifact_snapshot_returns_the_single_verified_read(tmp_path: Path):
    spec = _core_spec()
    paths = write_run_artifacts(_core_result(spec), spec, tmp_path)

    snapshot = load_verified_artifact_snapshot(paths["run_dir"])

    assert isinstance(snapshot.manifest, CoreRunManifest)
    assert snapshot.manifest.run == spec
    assert snapshot.summary["run_id"] == spec.run_id
    assert snapshot.transcript_rows[0]["payload"]["actor_id"] == "counter-alpha"
    assert json.loads(snapshot.manifest_bytes)["run"]["run_id"] == spec.run_id
    assert json.loads(snapshot.content_bytes["summary"])["run_id"] == spec.run_id


def test_legacy_manifest_recomputes_transcript_digest_after_reload(tmp_path: Path):
    spec = _spec()
    paths = write_run_artifacts(_result(spec), spec, tmp_path)
    manifest_path = Path(paths["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["transcript_metadata"] == {"run_spec_hash": spec.spec_hash}
    assert manifest["transcript_counts_by_kind"] == {"event": 1}
    verified = verify_run_artifacts(paths["run_dir"])
    assert verified.transcript_digest == _result(spec).transcript_digest

    # A manifest-only metadata alteration preserves file hashes and summary
    # identity, but must fail the independently reconstructed stable digest.
    manifest["transcript_metadata"]["forged"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="digest does not match legacy manifest"):
        verify_run_artifacts(paths["run_dir"])


def test_verified_artifact_loader_rederives_nonserializable_summary_attestation(
    tmp_path: Path,
) -> None:
    spec = _spec()
    analysis = {
        "winner": "village",
        "days": 2,
        "turn_policy": spec.turn_policy,
        "seats": [],
    }
    transcript = Transcript(
        run_id=spec.run_id,
        metadata={"run_spec_hash": spec.spec_hash},
    )
    transcript.append("event", {"type": "analysis", "analysis": analysis})
    exported = transcript.export()
    result = _result(spec).model_copy(update={
        "analysis": analysis,
        "transcript": exported,
        "transcript_digest": exported["stable_digest"],
        "event_count": 1,
        "decision_trace_count": 0,
    })
    paths = write_run_artifacts(result, spec, tmp_path)

    row = load_verified_run_summary(paths["run_dir"])
    assert is_verified_summary_row(row)
    assert summarize_runs([row]).evaluation_evidence_run_count == 1

    restored = RunSummaryRow.model_validate(row.model_dump(mode="json"))
    assert not is_verified_summary_row(restored)
    restored_summary = summarize_runs([restored])
    assert restored_summary.evaluation_evidence_run_count == 0
    assert restored_summary.operational_evaluation is None


def test_legacy_manifest_rejects_forged_transcript_counts(tmp_path: Path):
    spec = _spec()
    paths = write_run_artifacts(_result(spec), spec, tmp_path)
    manifest_path = Path(paths["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["transcript_counts_by_kind"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="legacy transcript counts"):
        verify_run_artifacts(paths["run_dir"])


def test_legacy_manifest_without_new_default_field_remains_verifiable(tmp_path: Path):
    spec = _spec()
    paths = write_run_artifacts(_result(spec), spec, tmp_path)
    manifest_path = Path(paths["manifest"])
    summary_path = Path(paths["summary"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Simulate a pre-extension agent-harness.manifest.v2 artifact: these
    # reconstruction fields did not exist and must remain backward compatible.
    manifest.pop("transcript_integrity_version", None)
    manifest.pop("transcript_metadata", None)
    manifest.pop("transcript_counts_by_kind", None)
    manifest["run"].pop("ruleset_id")
    raw_run_bytes = json.dumps(
        manifest["run"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    legacy_hash = hashlib.sha256(raw_run_bytes).hexdigest()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_spec"].pop("ruleset_id")
    summary["run_spec_hash"] = legacy_hash
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _replace_summary_and_refresh_integrity(paths, summary)

    verified = verify_run_artifacts(paths["run_dir"])
    assert verified.run.ruleset_id == "classic.v1"
    with pytest.raises(ArtifactIntegrityError, match="transcript reconstruction provenance"):
        load_verified_run_summary(paths["run_dir"])


def test_legacy_verifier_rejects_rehashed_embedded_run_spec_tampering(
    tmp_path: Path,
):
    spec = _spec()
    paths = write_run_artifacts(_result(spec), spec, tmp_path)
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["run_spec_hash"] == spec.spec_hash

    summary["run_spec"]["max_speak_rounds"] += 1
    _replace_summary_and_refresh_integrity(paths, summary)

    with pytest.raises(
        ArtifactIntegrityError,
        match="summary embedded RunSpec does not match manifest",
    ):
        verify_run_artifacts(paths["run_dir"])


def test_legacy_verifier_rejects_rehashed_invalid_embedded_run_spec(
    tmp_path: Path,
):
    spec = _spec()
    paths = write_run_artifacts(_result(spec), spec, tmp_path)
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["run_spec_hash"] == spec.spec_hash

    summary["run_spec"] = {"run_id": spec.run_id}
    _replace_summary_and_refresh_integrity(paths, summary)

    with pytest.raises(
        ArtifactIntegrityError,
        match="summary contains an invalid embedded RunSpec",
    ):
        verify_run_artifacts(paths["run_dir"])


def test_artifact_verifier_rejects_tampering_and_extra_files(tmp_path: Path):
    spec = _spec()
    paths = write_run_artifacts(_result(spec), spec, tmp_path)
    run_dir = Path(paths["run_dir"])
    transcript_path = Path(paths["transcript_jsonl"])
    transcript_path.write_text(
        transcript_path.read_text(encoding="utf-8") + '{"forged":true}\n',
        encoding="utf-8",
    )
    with pytest.raises(ArtifactIntegrityError, match="integrity mismatch"):
        verify_run_artifacts(run_dir)

    write_run_artifacts(_result(spec), spec, tmp_path)
    (run_dir / "unexpected.txt").write_text("not part of the artifact contract", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="file set mismatch"):
        verify_run_artifacts(run_dir)


def test_generic_run_uses_versioned_core_manifest_without_credentials(tmp_path: Path):
    spec = _core_spec()
    result = _core_result(spec)

    paths = write_run_artifacts(result, spec, tmp_path)
    manifest_json = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    summary_json = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    transcript_text = Path(paths["transcript_jsonl"]).read_text(encoding="utf-8")
    verified = verify_run_artifacts(paths["run_dir"])

    assert isinstance(verified, CoreRunManifest)
    assert verified.run == spec
    assert verified.run_spec_hash == spec.spec_hash
    assert verified.result_schema_version == result.schema_version
    assert manifest_json["schema_version"] == "agent-harness.core-manifest.v1"
    assert manifest_json["run"]["environment"] == {"id": "counter", "version": "1"}
    assert "transcript" not in summary_json
    assert summary_json["metrics"]["api_key"] == "[redacted]"
    serialized = json.dumps(manifest_json) + json.dumps(summary_json) + transcript_text
    assert "sk-test-secret-must-not-be-written" not in serialized
    transcript_row = load_transcript_jsonl(paths["transcript_jsonl"])[0]
    assert transcript_row["payload"]["actor_id"] == "counter-alpha"


def test_generic_writer_rejects_mismatched_spec_identity_before_writing(tmp_path: Path):
    spec = _core_spec()
    wrong_hash = _core_result(spec).model_copy(update={"run_spec_hash": "0" * 64})
    with pytest.raises(ArtifactIntegrityError, match="run_spec_hash"):
        write_run_artifacts(wrong_hash, spec, tmp_path)

    wrong_run = _core_result(spec).model_copy(update={"run_id": "different-run"})
    with pytest.raises(ArtifactIntegrityError, match="run_id"):
        write_run_artifacts(wrong_run, spec, tmp_path)

    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_core_spec_rejects_credentials_hidden_in_metadata_values():
    with pytest.raises(ValueError, match="credentials are forbidden"):
        CoreRunSpec(
            run_id="credential-leak",
            environment=EnvironmentRef(id="counter", version="1"),
            metadata={"note": "Bearer credential-that-must-not-enter-a-manifest"},
        )

    with pytest.raises(ValueError, match="credentials are forbidden"):
        CoreRunSpec(
            run_id="credential-url-leak",
            environment=EnvironmentRef(id="counter", version="1"),
            environment_config={
                "endpoint": "https://user:password@example.invalid/v1",
            },
        )


def test_generic_verifier_rejects_rehashed_transcript_payload_tampering(tmp_path: Path):
    spec = _core_spec()
    paths = write_run_artifacts(_core_result(spec), spec, tmp_path)
    transcript_path = Path(paths["transcript_jsonl"])
    manifest_path = Path(paths["manifest"])

    row = load_transcript_jsonl(transcript_path)[0]
    row["payload"]["value"] = 999
    transcript_bytes = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    transcript_path.write_bytes(transcript_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_integrity"]["transcript_jsonl"] = {
        "sha256": hashlib.sha256(transcript_bytes).hexdigest(),
        "bytes": len(transcript_bytes),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="payload hash mismatch"):
        verify_run_artifacts(paths["run_dir"])


def test_verifier_rejects_unknown_manifest_schema_version(tmp_path: Path):
    spec = _core_spec()
    paths = write_run_artifacts(_core_result(spec), spec, tmp_path)
    manifest_path = Path(paths["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "agent-harness.core-manifest.v999"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="unsupported manifest schema_version"):
        verify_run_artifacts(paths["run_dir"])


def test_generic_verifier_rejects_manifest_run_spec_hash_mismatch(tmp_path: Path):
    spec = _core_spec()
    paths = write_run_artifacts(_core_result(spec), spec, tmp_path)
    manifest_path = Path(paths["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_spec_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="manifest is missing or invalid"):
        verify_run_artifacts(paths["run_dir"])


def test_generic_verifier_recomputes_digest_from_rows_and_manifest_metadata(tmp_path: Path):
    spec = _core_spec()
    paths = write_run_artifacts(_core_result(spec), spec, tmp_path)
    manifest_path = Path(paths["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["transcript_metadata"]["forged"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="transcript digest"):
        verify_run_artifacts(paths["run_dir"])


def test_writer_rejects_preexisting_run_directory_symlink(tmp_path: Path):
    spec = _core_spec()
    artifact_root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    artifact_root.mkdir()
    outside.mkdir()
    (artifact_root / spec.run_id).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactIntegrityError, match="must not be a symlink"):
        write_run_artifacts(_core_result(spec), spec, artifact_root)

    assert not list(outside.iterdir())


def test_run_id_cannot_escape_artifact_root():
    data = _spec().model_dump()
    data["run_id"] = "../outside"
    with pytest.raises(ValueError, match="safe path component"):
        RunSpec(**data)


def test_manifest_commit_marker_detects_interrupted_multi_file_replacement(
    tmp_path: Path,
    monkeypatch,
):
    spec = _spec()
    paths = write_run_artifacts(_result(spec), spec, tmp_path)
    changed = _result(spec)
    transcript = Transcript(run_id=spec.run_id, metadata={"run_spec_hash": spec.spec_hash})
    transcript.append("event", {"type": "phase_started", "phase": "setup", "day": 0})
    transcript.append("event", {"type": "game_ended", "winner": "werewolves"})
    exported = transcript.export()
    changed = changed.model_copy(update={
        "winner": "werewolves",
        "event_count": 2,
        "transcript": exported,
        "transcript_digest": exported["stable_digest"],
    })

    real_atomic_write = artifact_module._atomic_write_bytes
    calls = 0

    def fail_before_summary(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated storage interruption")
        real_atomic_write(path, content)

    monkeypatch.setattr(artifact_module, "_atomic_write_bytes", fail_before_summary)
    with pytest.raises(OSError, match="simulated storage interruption"):
        write_run_artifacts(changed, spec, tmp_path)

    with pytest.raises(ArtifactIntegrityError, match="integrity mismatch"):
        verify_run_artifacts(paths["run_dir"])
    assert not list(Path(paths["run_dir"]).glob("*.tmp"))
