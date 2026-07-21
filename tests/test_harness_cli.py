from __future__ import annotations

import pytest

from src.harness.cli import parse_args
from src.harness.spec import ExperimentSpec


def test_cli_parses_real_harness_schedule_without_pseudo_metric_options():
    args = parse_args([
        "--seed", "100",
        "--runs", "2",
        "--turn-policies", "fixed_round_robin,bid_reply",
        "--policy-order", "abba",
        "--seat-permutation", "cyclic",
    ])
    assert args.seed == 100
    assert args.runs == 2
    assert args.turn_policies == ["fixed_round_robin", "bid_reply"]
    assert args.seat_permutation == "cyclic"
    assert not hasattr(args, "quality")
    assert not hasattr(args, "posterior")
    assert not hasattr(args, "deception")
    assert not hasattr(args, "verbose_thinking")

    runs = ExperimentSpec(
        experiment_id=args.experiment_id,
        player_names=args.names,
        turn_policies=args.turn_policies,
        replicates=args.runs,
        base_seed=args.seed,
        policy_order=args.policy_order,
    ).expand_runs()
    assert len(runs) == 4
    assert [run.turn_policy for run in runs] == [
        "fixed_round_robin",
        "bid_reply",
        "bid_reply",
        "fixed_round_robin",
    ]


def test_cli_rejects_removed_policy():
    with pytest.raises(SystemExit, match="unknown turn policies"):
        parse_args(["--seed", "1", "--turn-policies", "bid_only"])
