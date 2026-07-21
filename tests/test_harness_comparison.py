from __future__ import annotations

from typing import Any

from src.harness.comparison import aggregate_comparative_metrics
from src.harness.results import RunSummaryRow, run_summary_from_result
from src.harness.runner import HarnessRunResult
from src.harness.summary import summarize_runs
from src.harness.transcript import Transcript


_PLAYERS = ["A", "B", "C", "D", "E", "F"]
_DECK = ["werewolf", "werewolf", "seer", "villager", "villager", "villager"]


def _row(
    run_id: str,
    *,
    policy: str,
    pair: int,
    days: int,
    winner: str = "village",
    expected_pairs: int = 2,
    role_seed_offset: int = 0,
    metadata_update: dict[str, Any] | None = None,
) -> RunSummaryRow:
    case_seed = 100 + pair
    metadata = {
        "experiment_id": "paired-evaluation",
        "experiment_spec_hash": "e" * 64,
        "policy_set": ["fixed_round_robin", "bid_reply"],
        "policy_order": "abba",
        "runs_per_policy": expected_pairs,
        "pair_id": f"pair-{pair:04d}",
        "case_seed": case_seed,
    }
    metadata.update(metadata_update or {})
    analysis = {
        "winner": winner,
        "days": days,
        "turn_policy": policy,
        "seats": [],
        "decision_count": 10,
        "decision_failure_metrics": {
            "failure_count": 1 if policy == "fixed_round_robin" else 2,
        },
    }
    transcript = Transcript(
        run_id=run_id,
        metadata={"caller_metadata": metadata},
    )
    transcript.append("event", {"type": "analysis", "analysis": analysis})
    exported = transcript.export()
    result = HarnessRunResult(
        run_id=run_id,
        status="completed",
        winner=winner,
        days=days,
        elapsed_seconds=float(days * 2),
        role_seed=case_seed + role_seed_offset,
        actor_seed=100_000 + case_seed,
        orchestrator_seed=200_000 + case_seed,
        run_spec_hash=("a" if policy == "fixed_round_robin" else "b") * 64,
        run_spec={
            "environment_id": "werewolf.classic",
            "environment_version": "1",
            "turn_policy": policy,
            "player_names": list(_PLAYERS),
            "role_deck": list(_DECK),
            "metadata": metadata,
        },
        transcript_digest=exported["stable_digest"],
        transcript=exported,
        analysis=analysis,
        router_stats_delta={
            "calls": 12,
            "failures": 0 if policy == "fixed_round_robin" else 1,
            "retries": 0,
            "total_tokens_in": 100,
            "total_tokens_out": 50,
            "total_latency": float(days),
        },
    )
    return run_summary_from_result(result)


def test_summary_reports_recomputable_noncausal_paired_effects() -> None:
    rows = [
        _row("a1", policy="fixed_round_robin", pair=1, days=2, winner="village"),
        _row("b1", policy="bid_reply", pair=1, days=4, winner="werewolves"),
        _row("a2", policy="fixed_round_robin", pair=2, days=4, winner="werewolves"),
        _row("b2", policy="bid_reply", pair=2, days=6, winner="village"),
    ]

    evaluation = summarize_runs(rows).comparative_evaluation

    assert evaluation is not None
    assert evaluation.causal is False
    assert evaluation.design == "matched_pair_descriptive"
    assert evaluation.run_count == 4
    assert evaluation.eligible_run_count == 4
    experiment = evaluation.experiments["paired-evaluation"]
    assert experiment.expected_pair_count == 2
    assert experiment.observed_pair_count == 2
    assert experiment.complete_pair_count == 2
    assert experiment.incomplete_pair_count == 0
    assert experiment.invalid_pair_count == 0
    comparison = experiment.comparisons[0]
    assert comparison.baseline_policy == "fixed_round_robin"
    assert comparison.comparison_policy == "bid_reply"

    days = comparison.metrics["days"]
    assert days.pair_count == 2
    assert days.baseline_mean == 3.0
    assert days.comparison_mean == 5.0
    assert days.mean_difference == 2.0
    assert days.bootstrap_ci_low == 2.0
    assert days.bootstrap_ci_high == 2.0
    assert days.positive_difference_count == 2
    assert days.difference_definition == "comparison_minus_baseline"

    wins = comparison.metrics["village_win"]
    assert wins.pair_count == 2
    assert wins.baseline_mean == 0.5
    assert wins.comparison_mean == 0.5
    assert wins.mean_difference == 0.0


def test_pairing_diagnostics_exclude_missing_duplicate_and_mismatched_controls() -> None:
    rows = [
        _row("a1", policy="fixed_round_robin", pair=1, days=2, expected_pairs=4),
        _row("b1", policy="bid_reply", pair=1, days=3, expected_pairs=4),
        # Pair 2 is observed but incomplete.
        _row("a2", policy="fixed_round_robin", pair=2, days=2, expected_pairs=4),
        # Pair 3 has both policies, but the role seed differs and must fail closed.
        _row("a3", policy="fixed_round_robin", pair=3, days=2, expected_pairs=4),
        _row(
            "b3",
            policy="bid_reply",
            pair=3,
            days=3,
            expected_pairs=4,
            role_seed_offset=1,
        ),
    ]

    evaluation = aggregate_comparative_metrics(rows)

    assert evaluation is not None
    experiment = evaluation.experiments["paired-evaluation"]
    assert experiment.expected_pair_count == 4
    assert experiment.observed_pair_count == 3
    assert experiment.complete_pair_count == 1
    # One observed partial pair plus one entirely absent scheduled pair.
    assert experiment.incomplete_pair_count == 2
    assert experiment.invalid_pair_count == 1
    assert experiment.comparisons[0].metrics["days"].pair_count == 1


def test_uncontrolled_rows_do_not_create_comparative_claims() -> None:
    row = _row("single", policy="fixed_round_robin", pair=1, days=2)
    row = row.model_copy(update={"metadata": {"source": "legacy"}})

    assert aggregate_comparative_metrics([row]) is None
    summary = summarize_runs([row])
    assert summary.comparative_evaluation is None


def test_conflicting_policy_declarations_are_reported_without_effect_estimates() -> None:
    first = _row("a1", policy="fixed_round_robin", pair=1, days=2)
    second = _row(
        "b1",
        policy="bid_reply",
        pair=1,
        days=3,
        metadata_update={
            "policy_set": ["bid_reply", "fixed_round_robin"],
        },
    )

    evaluation = aggregate_comparative_metrics([first, second])

    assert evaluation is not None
    experiment = evaluation.experiments["paired-evaluation"]
    assert experiment.metadata_conflict_count == 1
    assert experiment.comparisons == []
