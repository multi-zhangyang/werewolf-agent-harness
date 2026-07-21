"""Factual run-row, aggregate, and batch-resume tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.harness.batch import run_experiment_spec
from src.harness.results import RunSummaryRow, run_summary_from_result
from src.harness.runner import HarnessRunResult
from src.harness.runner import resolve_run_spec
from src.harness.spec import ExperimentSpec, RunSpec
from src.harness.summary import summarize_runs
from src.harness.transcript import Transcript
from src.llm.models import ModelConfig


class _Router:
    async def aclose(self) -> None:
        return None


def _run_spec(run_id: str, *, policy: str = "fixed_round_robin") -> RunSpec:
    return RunSpec(
        run_id=run_id,
        player_names=["A", "B", "C", "D", "E", "F"],
        role_deck=["werewolf", "werewolf", "seer", "villager", "villager", "villager"],
        turn_policy=policy,
        role_seed=1,
        actor_seed=2,
        orchestrator_seed=3,
    )


def _result(
    run_id: str,
    *,
    status: str = "completed",
    winner: str | None = "village",
    policy: str = "fixed_round_robin",
    run_spec: RunSpec | None = None,
) -> HarnessRunResult:
    spec = run_spec or _run_spec(run_id, policy=policy)
    policy = spec.turn_policy
    analysis = {
        "winner": winner,
        "days": 3,
        "turn_policy": policy,
        "seats": [],
        "decision_count": 5,
        "parse_metrics": {"decision_count": 5, "parse_recovered_count": 1},
        "decision_failure_metrics": {"failure_count": 2},
    }
    transcript = Transcript(run_id=run_id)
    transcript.append("event", {"type": "game_ended", "winner": winner})
    transcript.append("event", {"type": "analysis", "analysis": analysis})
    exported = transcript.export()
    return HarnessRunResult(
        run_id=run_id,
        status=status,
        winner=winner,
        days=3,
        elapsed_seconds=12.5,
        error_type="ProviderError" if status != "completed" else None,
        error="provider failed" if status != "completed" else None,
        run_spec_hash=spec.spec_hash,
        role_seed=1,
        actor_seed=2,
        orchestrator_seed=3,
        run_spec=spec.model_dump(),
        event_count=7,
        decision_trace_count=10,
        transcript_digest=exported["stable_digest"],
        transcript=exported,
        analysis=analysis,
        router_stats_delta={
            "calls": 5,
            "successes": 4,
            "failures": 1,
            "retries": 2,
            "total_tokens_in": 100,
            "total_tokens_out": 40,
            "total_latency": 8.25,
            "structured_responses": 6,
            "incomplete_responses": 1,
            "response_parse_failures": 2,
            "response_parse_recoveries": 1,
            "lossy_parse_rejections": 1,
        },
    )


def test_run_summary_contains_only_recomputable_outcome_cost_and_failure_fields():
    row = run_summary_from_result(_result("row-1"))

    assert row.model_calls == 5
    assert row.input_tokens == 100
    assert row.output_tokens == 40
    assert row.decision_count == 5
    assert row.consumed_parse_recovery_count == 1
    assert row.structured_response_count == 6
    assert row.incomplete_response_count == 1
    assert row.response_parse_failure_count == 2
    assert row.response_parse_recovery_count == 1
    assert row.lossy_parse_rejection_count == 1
    assert row.decision_failure_count == 2
    dumped = row.model_dump()
    for removed in (
        "posterior_metrics",
        "deception_audit",
        "collusion_audit",
        "quality",
        "social_metrics",
        "replay_capability",
    ):
        assert removed not in dumped


def test_experiment_summary_sums_factual_counts_without_claiming_significance():
    rows = [
        run_summary_from_result(_result("a", winner="village")),
        run_summary_from_result(_result("b", winner="werewolves", policy="bid_reply")),
        run_summary_from_result(_result("c", status="failed", winner=None)),
    ]
    summary = summarize_runs(rows)

    assert summary.run_count == 3
    assert summary.evaluation_evidence_run_count == 3
    assert summary.cache_only_run_count == 0
    assert summary.completed_runs == 2
    assert summary.failed_runs == 1
    assert summary.winner_counts == {"village": 1, "werewolves": 1}
    assert summary.total_model_calls == 15
    assert summary.total_decisions == 15
    assert summary.total_consumed_parse_recoveries == 3
    assert summary.total_structured_responses == 18
    assert summary.total_incomplete_responses == 3
    assert summary.total_response_parse_failures == 6
    assert summary.total_response_parse_recoveries == 3
    assert summary.total_lossy_parse_rejections == 3
    assert not hasattr(summary, "confidence_interval")


@pytest.mark.asyncio
async def test_batch_runs_missing_rows_and_resumes_existing_rows(monkeypatch, tmp_path: Path):
    spec = ExperimentSpec(
        experiment_id="batch",
        player_names=["A", "B", "C", "D", "E", "F"],
        role_deck=["werewolf", "werewolf", "seer", "villager", "villager", "villager"],
        turn_policies=["fixed_round_robin"],
        replicates=2,
        base_seed=100,
    )
    model_config = ModelConfig(provider="openai", model="m", api_key="k")
    runs = [
        resolve_run_spec(run, model_config=model_config)
        for run in spec.expand_runs()
    ]
    resume_path = tmp_path / "summary.jsonl"
    resumed_row = run_summary_from_result(_result(runs[0].run_id, run_spec=runs[0]))
    resume_path.write_text(json.dumps(resumed_row.model_dump()) + "\n", encoding="utf-8")
    called: list[str] = []

    async def fake_run(run_spec, **_kwargs):
        called.append(run_spec.run_id)
        return _result(run_spec.run_id, run_spec=run_spec)

    monkeypatch.setattr("src.harness.batch.run_werewolf_run", fake_run)
    batch = await run_experiment_spec(
        spec,
        model_config=model_config,
        router=_Router(),  # type: ignore[arg-type]
        summary_jsonl=resume_path,
        resume_jsonl=True,
    )

    assert called == [runs[1].run_id]
    assert batch.scheduled_runs == 2
    assert batch.completed_runs == 2
    assert batch.summary.evaluation_evidence_run_count == 1
    assert batch.summary.cache_only_run_count == 1
    assert batch.resumed_run_ids == [runs[0].run_id]
    assert [row.run_id for row in batch.rows] == sorted(run.run_id for run in runs)
    assert len(resume_path.read_text(encoding="utf-8").splitlines()) == 2


def test_summary_collapses_identical_runs_and_rejects_conflicting_duplicates() -> None:
    row = run_summary_from_result(_result("duplicate-summary"))
    summary = summarize_runs([row, row])
    assert summary.run_count == 1
    assert summary.total_model_calls == row.model_calls

    conflicting = row.model_copy(update={"model_calls": row.model_calls + 1})
    with pytest.raises(ValueError, match="conflicting summary rows"):
        summarize_runs([row, conflicting])


@pytest.mark.asyncio
async def test_batch_resume_rejects_stale_run_spec_hash(tmp_path: Path):
    spec = ExperimentSpec(
        experiment_id="stale-resume",
        player_names=["A", "B", "C", "D", "E", "F"],
        role_deck=["werewolf", "werewolf", "seer", "villager", "villager", "villager"],
        turn_policies=["fixed_round_robin"],
        replicates=1,
        base_seed=100,
    )
    scheduled = spec.expand_runs()[0]
    stale = _result(scheduled.run_id, run_spec=scheduled)
    resume_path = tmp_path / "summary.jsonl"
    resume_path.write_text(
        json.dumps(run_summary_from_result(stale).model_dump()) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run_spec_hash mismatch"):
        await run_experiment_spec(
            spec,
            model_config=ModelConfig(provider="openai", model="new-model", api_key="k"),
            router=_Router(),  # type: ignore[arg-type]
            summary_jsonl=resume_path,
            resume_jsonl=True,
        )


@pytest.mark.asyncio
async def test_batch_cyclic_permutation_moves_runtime_and_manifest_model_overrides(monkeypatch):
    spec = ExperimentSpec(
        experiment_id="model-cross-play",
        player_names=["A", "B", "C", "D", "E", "F"],
        role_deck=["werewolf", "werewolf", "seer", "villager", "villager", "villager"],
        turn_policies=["fixed_round_robin", "bid_reply"],
        replicates=2,
        base_seed=100,
        policy_order="abba",
        metadata={"seat_permutation_mode": "cyclic"},
    )
    default = ModelConfig(provider="openai", model="default", api_key="default-key")
    source_overrides = {
        1: ModelConfig(provider="openai", model="source-seat-1", api_key="key-1"),
        2: ModelConfig(provider="openai", model="source-seat-2", api_key="key-2"),
    }
    calls: list[tuple[RunSpec, dict[int, ModelConfig]]] = []

    async def fake_run(run_spec, **kwargs):
        calls.append((run_spec, kwargs["seat_model_configs"]))
        return _result(run_spec.run_id, run_spec=run_spec)

    monkeypatch.setattr("src.harness.batch.run_werewolf_run", fake_run)
    await run_experiment_spec(
        spec,
        model_config=default,
        seat_model_configs=source_overrides,
        router=_Router(),  # type: ignore[arg-type]
    )

    first_spec, first_runtime = calls[0]
    assert first_spec.seat_models[1].model == "source-seat-1"
    assert first_spec.seat_models[2].model == "source-seat-2"
    assert first_runtime[1].model == "source-seat-1"
    assert first_runtime[2].model == "source-seat-2"
    paired_spec, paired_runtime = calls[1]
    assert paired_spec.metadata["pair_id"] == first_spec.metadata["pair_id"]
    assert paired_spec.seat_models == first_spec.seat_models
    assert paired_runtime == first_runtime

    second_spec, second_runtime = calls[2]
    assert second_spec.metadata["seat_permutation"] == [2, 3, 4, 5, 6, 1]
    assert second_spec.seat_models[1].model == "source-seat-2"
    assert second_spec.seat_models[6].model == "source-seat-1"
    assert set(second_spec.seat_models) == {1, 6}
    assert second_runtime[1].model == "source-seat-2"
    assert second_runtime[6].model == "source-seat-1"
    reverse_paired_spec, reverse_paired_runtime = calls[3]
    assert reverse_paired_spec.metadata["pair_id"] == second_spec.metadata["pair_id"]
    assert reverse_paired_spec.seat_models == second_spec.seat_models
    assert reverse_paired_runtime == second_runtime
