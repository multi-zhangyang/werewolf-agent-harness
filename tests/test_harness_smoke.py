"""Offline verification tests for real-model smoke-run evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import src.harness.smoke as smoke_module
from src.harness.artifacts import (
    ArtifactIntegrityError,
    load_verified_artifact_snapshot,
    write_run_artifacts,
)
from src.harness.core_runner import EnvironmentRunResult
from src.harness.core_spec import CoreRunSpec, EnvironmentRef
from src.harness.runner import HarnessRunResult
from src.harness.smoke import (
    REAL_MODEL_SMOKE_REPORT_VERSION,
    RealModelSmokeVerificationError,
    main,
    verify_real_model_smoke_artifacts,
)
from src.harness.spec import RunSpec
from src.harness.transcript import Transcript


def _spec(run_id: str = "real-smoke-artifact") -> RunSpec:
    return RunSpec(
        run_id=run_id,
        player_names=["A", "B", "C", "D", "E", "F"],
        role_deck=[
            "werewolf",
            "werewolf",
            "seer",
            "villager",
            "villager",
            "villager",
        ],
        role_seed=11,
        actor_seed=22,
        orchestrator_seed=33,
    )


def _transcript(
    run_id: str,
    *,
    terminal_count: int = 1,
    include_consumed: bool = True,
    orphan_terminal: bool = False,
    additional_call_id: str | None = None,
) -> dict[str, Any]:
    transcript = Transcript(run_id=run_id, metadata={"suite": "real-model-smoke"})
    request_id = f"{run_id}:request:1"
    transcript.append(
        "decision",
        {
            "kind": "agent_request",
            "request": {
                "request_id": request_id,
                "actor_id": "seat-1",
                "legal_actions": [{"name": "speak"}],
            },
        },
    )
    for _index in range(terminal_count):
        transcript.append(
            "decision",
            {
                "kind": "agent_response",
                "request_id": request_id,
                "envelope": {
                    "request_id": request_id,
                    "actor_id": "seat-1",
                    "model_call_id": "local-test-call-1",
                },
                "validation": {"valid": True, "issues": []},
            },
        )
    if orphan_terminal:
        transcript.append(
            "decision",
            {
                "kind": "agent_response_failed",
                "request_id": f"{run_id}:orphan",
                "failure": {"error_type": "OfflineTestFailure"},
            },
        )
    if include_consumed:
        transcript.append(
            "decision",
            {
                "type": "decision_consumed",
                "request_id": request_id,
                "model_call_id": "local-test-call-1",
                "action": "speak",
            },
        )
    if additional_call_id is not None:
        second_request_id = f"{run_id}:request:2"
        transcript.append("decision", {
            "kind": "agent_request",
            "request": {"request_id": second_request_id, "actor_id": "seat-2"},
        })
        transcript.append("decision", {
            "kind": "agent_response",
            "request_id": second_request_id,
            "envelope": {
                "request_id": second_request_id,
                "actor_id": "seat-2",
                "model_call_id": additional_call_id,
            },
            "validation": {"valid": True, "issues": []},
        })
        transcript.append("decision", {
            "type": "decision_consumed",
            "request_id": second_request_id,
            "model_call_id": additional_call_id,
            "action": "speak",
        })
    return transcript.export()


def _model_attempt(
    *,
    attempt: int,
    call_id: str,
    request_id: str,
    status: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency: float,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "status": status,
        "error_type": "ValidationError" if status == "response_rejected" else None,
        "llm_call": {
            "call_id": call_id,
            "context": {"request_id": request_id},
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            "latency": latency,
        },
    }


def _nested_attempt_transcript(
    run_id: str,
    *,
    call_ids: tuple[str, str, str] = (
        "nested-call-1",
        "nested-call-2",
        "nested-call-3",
    ),
    context_request_ids: tuple[str, str, str] | None = None,
    accepted_attempt_count: int | None = None,
    accepted_attempt_numbers: tuple[int, int] = (1, 2),
    accepted_response_call_id: str | None = None,
    accepted_consumed_call_id: str | None = None,
    include_response_call_id: bool = True,
) -> dict[str, Any]:
    transcript = Transcript(run_id=run_id, metadata={"suite": "nested-real-model-smoke"})
    accepted_request_id = f"{run_id}:request:1"
    failed_request_id = f"{run_id}:request:2"
    contexts = context_request_ids or (
        accepted_request_id,
        accepted_request_id,
        failed_request_id,
    )
    accepted_attempts = [
        _model_attempt(
            attempt=accepted_attempt_numbers[0],
            call_id=call_ids[0],
            request_id=contexts[0],
            status="response_rejected",
            prompt_tokens=10,
            completion_tokens=4,
            latency=1.25,
        ),
        _model_attempt(
            attempt=accepted_attempt_numbers[1],
            call_id=call_ids[1],
            request_id=contexts[1],
            status="accepted",
            prompt_tokens=11,
            completion_tokens=5,
            latency=1.5,
        ),
    ]
    failed_attempts = [
        _model_attempt(
            attempt=1,
            call_id=call_ids[2],
            request_id=contexts[2],
            status="response_rejected",
            prompt_tokens=12,
            completion_tokens=6,
            latency=1.75,
        ),
    ]

    transcript.append("decision", {
        "kind": "agent_request",
        "request": {"request_id": accepted_request_id, "actor_id": "seat-1"},
    })
    transcript.append("decision", {
        "type": "agent_turn_started",
        "request_id": accepted_request_id,
        "session_id": "session-1",
        "seat": 1,
        "visibility": "admin",
    })
    transcript.append("decision", {
        "type": "model_generation",
        "request_id": accepted_request_id,
        "call_id": call_ids[0],
        "step": 1,
        "visibility": "admin",
    })
    transcript.append("decision", {
        "type": "tool_call_requested",
        "request_id": accepted_request_id,
        "call_id": "read-call-1",
        "tool": "get_legal_actions",
        "visibility": "admin",
    })
    transcript.append("decision", {
        "type": "tool_result",
        "request_id": accepted_request_id,
        "call_id": "read-call-1",
        "tool": "get_legal_actions",
        "ok": True,
        "terminal": False,
        "visibility": "admin",
    })
    transcript.append("decision", {
        "type": "model_generation",
        "request_id": accepted_request_id,
        "call_id": call_ids[1],
        "step": 2,
        "visibility": "admin",
    })
    transcript.append("decision", {
        "type": "tool_call_requested",
        "request_id": accepted_request_id,
        "call_id": "terminal-call-1",
        "tool": "speak",
        "visibility": "admin",
    })
    transcript.append("decision", {
        "type": "tool_result",
        "request_id": accepted_request_id,
        "call_id": "terminal-call-1",
        "tool": "speak",
        "ok": True,
        "terminal": True,
        "visibility": "admin",
    })
    transcript.append("decision", {
        "type": "agent_action_submitted",
        "request_id": accepted_request_id,
        "call_id": "terminal-call-1",
        "tool": "speak",
        "action": "speak",
        "visibility": "admin",
    })
    transcript.append("decision", {
        "kind": "agent_response",
        "request_id": accepted_request_id,
        "envelope": {
            "request_id": accepted_request_id,
            "actor_id": "seat-1",
            "model_call_id": (
                accepted_response_call_id or call_ids[1]
                if include_response_call_id
                else None
            ),
        },
        "validation": {"valid": True, "issues": []},
    })
    accepted_call = dict(accepted_attempts[-1]["llm_call"])
    accepted_call.update({
        "actor_response_attempt_count": (
            accepted_attempt_count
            if accepted_attempt_count is not None
            else len(accepted_attempts)
        ),
        "actor_response_attempts": accepted_attempts,
    })
    transcript.append("decision", {
        "type": "decision_consumed",
        "request_id": accepted_request_id,
        "model_call_id": accepted_consumed_call_id or call_ids[1],
        "action": "speak",
        "llm_call": accepted_call,
    })
    transcript.append("decision", {
        "type": "rules_result",
        "request_id": accepted_request_id,
        "rules": {"status": "accepted", "action": "speak"},
    })

    transcript.append("decision", {
        "kind": "agent_request",
        "request": {"request_id": failed_request_id, "actor_id": "seat-2"},
    })
    transcript.append("decision", {
        "type": "agent_turn_started",
        "request_id": failed_request_id,
        "session_id": "session-2",
        "seat": 2,
        "visibility": "admin",
    })
    transcript.append("decision", {
        "type": "model_generation",
        "request_id": failed_request_id,
        "call_id": call_ids[2],
        "step": 1,
        "visibility": "admin",
    })
    transcript.append("decision", {
        "type": "tool_call_requested",
        "request_id": failed_request_id,
        "call_id": "failed-tool-call-1",
        "tool": "speak",
        "visibility": "admin",
    })
    transcript.append("decision", {
        "type": "tool_result",
        "request_id": failed_request_id,
        "call_id": "failed-tool-call-1",
        "tool": "speak",
        "ok": False,
        "terminal": False,
        "visibility": "admin",
    })
    transcript.append("decision", {
        "type": "agent_turn_failed",
        "request_id": failed_request_id,
        "error_code": "response_rejected",
        "visibility": "admin",
    })
    transcript.append("decision", {
        "kind": "agent_response_failed",
        "request_id": failed_request_id,
        "failure": {
            "error_type": "AgentDecisionError",
            "timeout": False,
            "reason": "AgentDecisionError during day/speak",
            "llm_call_attempts": failed_attempts,
        },
    })
    return transcript.export()


def _write_nested_attempt_run(
    tmp_path: Path,
    *,
    call_ids: tuple[str, str, str] = (
        "nested-call-1",
        "nested-call-2",
        "nested-call-3",
    ),
    context_request_ids: tuple[str, str, str] | None = None,
    accepted_attempt_count: int | None = None,
    accepted_attempt_numbers: tuple[int, int] = (1, 2),
    accepted_response_call_id: str | None = None,
    accepted_consumed_call_id: str | None = None,
    include_response_call_id: bool = True,
    router_stats_override: dict[str, Any] | None = None,
) -> Path:
    spec = _spec()
    transcript = _nested_attempt_transcript(
        spec.run_id,
        call_ids=call_ids,
        context_request_ids=context_request_ids,
        accepted_attempt_count=accepted_attempt_count,
        accepted_attempt_numbers=accepted_attempt_numbers,
        accepted_response_call_id=accepted_response_call_id,
        accepted_consumed_call_id=accepted_consumed_call_id,
        include_response_call_id=include_response_call_id,
    )
    router_stats_delta: dict[str, Any] = {
        "calls": 3,
        "successes": 3,
        "failures": 0,
        "retries": 0,
        "total_tokens_in": 33,
        "total_tokens_out": 15,
        "total_latency": 4.5,
    }
    router_stats_delta.update(router_stats_override or {})
    result = HarnessRunResult(
        run_id=spec.run_id,
        status="completed",
        winner="village",
        days=1,
        elapsed_seconds=0.1,
        run_spec_hash=spec.spec_hash,
        role_seed=11,
        actor_seed=22,
        orchestrator_seed=33,
        run_spec=spec.model_dump(),
        decision_trace_count=len(transcript["entries"]),
        transcript_digest=transcript["stable_digest"],
        transcript=transcript,
        router_stats_delta=router_stats_delta,
    )
    paths = write_run_artifacts(result, spec, tmp_path)
    return Path(paths["run_dir"])


def _result(
    spec: RunSpec,
    *,
    calls: int = 1,
    status: str = "completed",
    terminal_count: int = 1,
    include_consumed: bool = True,
    orphan_terminal: bool = False,
    additional_call_id: str | None = None,
) -> HarnessRunResult:
    transcript = _transcript(
        spec.run_id,
        terminal_count=terminal_count,
        include_consumed=include_consumed,
        orphan_terminal=orphan_terminal,
        additional_call_id=additional_call_id,
    )
    return HarnessRunResult(
        run_id=spec.run_id,
        status=status,
        winner="village",
        days=1,
        elapsed_seconds=0.1,
        run_spec_hash=spec.spec_hash,
        role_seed=11,
        actor_seed=22,
        orchestrator_seed=33,
        run_spec=spec.model_dump(),
        decision_trace_count=len(transcript["entries"]),
        transcript_digest=transcript["stable_digest"],
        transcript=transcript,
        router_stats_delta={"calls": calls, "successes": calls},
    )


def _write_legacy(
    tmp_path: Path,
    **result_kwargs: Any,
) -> tuple[Path, RunSpec]:
    spec = _spec()
    paths = write_run_artifacts(_result(spec, **result_kwargs), spec, tmp_path)
    return Path(paths["run_dir"]), spec


def _rewrite_summary_and_integrity(
    run_dir: Path,
    update: dict[str, Any],
) -> None:
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(update)
    summary_bytes = (
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    summary_path.write_bytes(summary_bytes)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_integrity"]["summary"] = {
        "sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "bytes": len(summary_bytes),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_real_model_smoke_rejects_degraded_legacy_evidence_without_nested_attempts(
    tmp_path: Path,
):
    run_dir, spec = _write_legacy(tmp_path, calls=3)

    with pytest.raises(
        RealModelSmokeVerificationError,
        match="complete nested attempt provenance",
    ):
        verify_real_model_smoke_artifacts(run_dir)

    assert spec.run_id == "real-smoke-artifact"


def test_real_model_smoke_reconciles_complete_nested_attempt_provenance(
    tmp_path: Path,
):
    run_dir = _write_nested_attempt_run(tmp_path)

    report = verify_real_model_smoke_artifacts(run_dir)

    assert report.model_calls == 3
    assert report.model_call_id_count == 3
    assert report.request_count == 2
    assert report.terminal_response_count == 2
    assert report.response_count == 1
    assert report.response_failure_count == 1
    assert report.decision_consumed_count == 1
    assert report.model_backed_decision_count == 1


def test_real_model_smoke_uses_the_verified_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    run_dir = _write_nested_attempt_run(tmp_path)
    snapshot = load_verified_artifact_snapshot(run_dir)

    def replace_after_snapshot(_run_dir: str | Path):
        (run_dir / "summary.json").write_text("not verified content\n", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(
        smoke_module,
        "load_verified_artifact_snapshot",
        replace_after_snapshot,
    )
    report = verify_real_model_smoke_artifacts(run_dir)

    assert report.status == "passed"
    assert report.model_calls == 3


def test_real_model_smoke_rejects_duplicate_nested_attempt_call_ids(
    tmp_path: Path,
):
    run_dir = _write_nested_attempt_run(
        tmp_path,
        call_ids=("nested-call-1", "nested-call-2", "nested-call-1"),
    )

    with pytest.raises(RealModelSmokeVerificationError, match="must be unique"):
        verify_real_model_smoke_artifacts(run_dir)


@pytest.mark.parametrize(
    ("run_kwargs", "message"),
    [
        ({"accepted_attempt_count": 3}, "attempt_count"),
        (
            {"accepted_attempt_numbers": (1, 3)},
            "contiguous and one-based",
        ),
        (
            {"accepted_consumed_call_id": "nested-call-1"},
            "decision_consumed model_call_id",
        ),
        (
            {
                "accepted_response_call_id": "nested-call-1",
                "accepted_consumed_call_id": "nested-call-1",
            },
            "does not match its accepted attempt",
        ),
        (
            {"include_response_call_id": False},
            "no model_call_id",
        ),
    ],
)
def test_real_model_smoke_rejects_inconsistent_accepted_attempt_provenance(
    tmp_path: Path,
    run_kwargs: dict[str, Any],
    message: str,
):
    run_dir = _write_nested_attempt_run(tmp_path, **run_kwargs)

    with pytest.raises(RealModelSmokeVerificationError, match=message):
        verify_real_model_smoke_artifacts(run_dir)


@pytest.mark.parametrize(
    ("context_request_ids", "message"),
    [
        (
            (
                "real-smoke-artifact:request:1",
                "real-smoke-artifact:request:1",
                "real-smoke-artifact:request:missing",
            ),
            "references no agent_request",
        ),
        (
            (
                "real-smoke-artifact:request:2",
                "real-smoke-artifact:request:1",
                "real-smoke-artifact:request:2",
            ),
            "does not match its owning request",
        ),
    ],
)
def test_real_model_smoke_rejects_unlinked_nested_attempt_context(
    tmp_path: Path,
    context_request_ids: tuple[str, str, str],
    message: str,
):
    run_dir = _write_nested_attempt_run(
        tmp_path,
        context_request_ids=context_request_ids,
    )

    with pytest.raises(RealModelSmokeVerificationError, match=message):
        verify_real_model_smoke_artifacts(run_dir)


@pytest.mark.parametrize(
    ("router_stats_override", "message"),
    [
        ({"calls": 4}, "model-call count"),
        ({"total_tokens_in": 34}, "input-token total"),
        ({"total_tokens_out": 16}, "output-token total"),
        ({"total_latency": 5.0}, "latency total"),
    ],
)
def test_real_model_smoke_rejects_nested_attempt_router_total_mismatch(
    tmp_path: Path,
    router_stats_override: dict[str, Any],
    message: str,
):
    run_dir = _write_nested_attempt_run(
        tmp_path,
        router_stats_override=router_stats_override,
    )

    with pytest.raises(RealModelSmokeVerificationError, match=message):
        verify_real_model_smoke_artifacts(run_dir)


def test_core_model_call_metric_is_explicitly_supported(tmp_path: Path):
    spec = CoreRunSpec(
        run_id="core-real-smoke-artifact",
        environment=EnvironmentRef(id="test.counter", version="1"),
        seeds={"turn_order": 7},
    )
    transcript = _nested_attempt_transcript(spec.run_id)
    result = EnvironmentRunResult(
        run_id=spec.run_id,
        status="completed",
        environment_id=spec.environment.id,
        environment_version=spec.environment.version,
        run_spec_hash=spec.spec_hash,
        elapsed_seconds=0.1,
        outcome={"winner": "alpha"},
        metrics={
            "model_calls": 3,
            "router_stats_delta": {
                "calls": 3,
                "total_tokens_in": 33,
                "total_tokens_out": 15,
                "total_latency": 4.5,
            },
        },
        transcript_digest=transcript["stable_digest"],
        transcript=transcript,
    )
    paths = write_run_artifacts(result, spec, tmp_path)

    report = verify_real_model_smoke_artifacts(paths["run_dir"])

    assert report.model_calls == 3
    assert report.model_call_metric_sources == [
        "metrics.router_stats_delta.calls",
        "metrics.model_calls",
    ]


def test_real_model_smoke_rejects_zero_model_calls(tmp_path: Path):
    run_dir, _spec_value = _write_legacy(tmp_path, calls=0)

    with pytest.raises(RealModelSmokeVerificationError, match="at least one model call"):
        verify_real_model_smoke_artifacts(run_dir)


@pytest.mark.parametrize(
    ("additional_call_id", "calls", "message"),
    [
        ("local-test-call-1", 2, "must be unique"),
        ("local-test-call-2", 1, "smaller than transcript"),
    ],
)
def test_real_model_smoke_rejects_call_id_and_metric_contradictions(
    tmp_path: Path,
    additional_call_id: str,
    calls: int,
    message: str,
):
    run_dir, _spec_value = _write_legacy(
        tmp_path,
        calls=calls,
        additional_call_id=additional_call_id,
    )

    with pytest.raises(RealModelSmokeVerificationError, match=message):
        verify_real_model_smoke_artifacts(run_dir)


@pytest.mark.parametrize(
    ("result_kwargs", "message"),
    [
        ({"terminal_count": 0}, "exactly one response terminal"),
        ({"terminal_count": 2}, "exactly one response terminal"),
        ({"orphan_terminal": True}, "orphan agent response"),
    ],
)
def test_real_model_smoke_rejects_unpaired_duplicate_and_orphan_terminals(
    tmp_path: Path,
    result_kwargs: dict[str, Any],
    message: str,
):
    run_dir, _spec_value = _write_legacy(tmp_path, **result_kwargs)

    with pytest.raises(RealModelSmokeVerificationError, match=message):
        verify_real_model_smoke_artifacts(run_dir)


@pytest.mark.parametrize(
    ("field", "credential"),
    [
        ("api_key", "not-redacted-test-value"),
        ("note", "Bearer " + "offline-test-token-value"),
        ("note", "sk-" + "offline_test_token_value"),
        ("note", "https://user:password@example.invalid/v1"),
    ],
)
def test_real_model_smoke_rejects_unredacted_or_obvious_credentials(
    tmp_path: Path,
    field: str,
    credential: str,
):
    run_dir, _spec_value = _write_legacy(tmp_path)
    _rewrite_summary_and_integrity(run_dir, {field: credential})

    with pytest.raises(RealModelSmokeVerificationError) as raised:
        verify_real_model_smoke_artifacts(run_dir)
    assert credential not in str(raised.value)


def test_real_model_smoke_rejects_artifact_tampering_before_semantics(tmp_path: Path):
    run_dir, _spec_value = _write_legacy(tmp_path)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = "failed"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="integrity mismatch"):
        verify_real_model_smoke_artifacts(run_dir)


def test_real_model_smoke_cli_prints_only_versioned_report(
    tmp_path: Path,
):
    run_dir = _write_nested_attempt_run(tmp_path)
    spec = _spec()

    completed = subprocess.run(
        [sys.executable, "-m", "src.harness.smoke", str(run_dir)],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    report = json.loads(completed.stdout)
    assert completed.stderr == ""
    assert report["schema_version"] == REAL_MODEL_SMOKE_REPORT_VERSION
    assert report["status"] == "passed"
    assert report["run_id"] == spec.run_id
    assert "local-test-call-1" not in completed.stdout


def test_real_model_smoke_cli_fails_closed_with_safe_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    run_dir, _spec_value = _write_legacy(tmp_path, calls=0)

    assert main([str(run_dir)]) != 0

    captured = capsys.readouterr()
    failure = json.loads(captured.err)
    assert captured.out == ""
    assert failure["schema_version"] == REAL_MODEL_SMOKE_REPORT_VERSION
    assert failure["status"] == "failed"
