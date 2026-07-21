from __future__ import annotations

from copy import deepcopy

import pytest

from src.harness.schedule import (
    apply_seat_permutation,
    build_policy_schedule,
    default_experiment_id,
    experiment_metadata,
    validate_persona_provenance,
)
from src.harness.spec import ExperimentSpec
from src.harness.statistics import bootstrap_mean_ci, router_stats_delta, wilson_ci


def test_harness_statistics_match_known_values() -> None:
    assert wilson_ci(0, 0) == (0.0, 0.0)

    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)

    first = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], iterations=200, seed=123)
    second = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], iterations=200, seed=123)
    assert first == second
    assert first[0] <= 2.5 <= first[1]


def test_harness_router_stats_delta_uses_delta_latency_for_avg() -> None:
    delta = router_stats_delta(
        {"calls": 10, "successes": 8, "total_latency": 15.0, "avg_latency": 1.5, "ignored": "bad"},
        {"calls": 13, "successes": 10, "total_latency": 22.5, "avg_latency": 1.731, "ignored": "still bad"},
    )

    assert delta == {"avg_latency": 2.5, "calls": 3.0, "successes": 2.0, "total_latency": 7.5}


def test_harness_policy_schedule_matches_abba_shape() -> None:
    assert default_experiment_id(["fixed_round_robin", "bid_reply"], policy_order="abba") == "abba:fixed_round_robin,bid_reply"

    schedule = build_policy_schedule(
        2,
        ["fixed_round_robin", "bid_reply"],
        policy_order="abba",
        seed=100,
        experiment_id="exp-abba",
    )

    assert [(row["turn_policy"], row["policy_game_idx"], row["role_seed"]) for row in schedule] == [
        ("fixed_round_robin", 1, 101),
        ("bid_reply", 1, 101),
        ("bid_reply", 2, 102),
        ("fixed_round_robin", 2, 102),
    ]
    assert [row["counterbalance_order"] for row in schedule] == ["AB", "AB", "BA", "BA"]
    assert [row["abba_position"] for row in schedule] == [1, 2, 3, 4]
    assert {row["scheduled_total"] for row in schedule} == {4}
    assert {row["runs_per_policy"] for row in schedule} == {2}
    assert schedule[0]["pair_id"] == schedule[1]["pair_id"] == "pair-0001"
    assert schedule[2]["pair_id"] == schedule[3]["pair_id"] == "pair-0002"


def test_experiment_spec_uses_runs_per_policy_and_canonical_abba_metadata() -> None:
    spec = ExperimentSpec(
        experiment_id="spec-abba",
        player_names=["A", "B", "C", "D", "E", "F"],
        turn_policies=["fixed_round_robin", "bid_reply"],
        replicates=2,
        base_seed=100,
        policy_order="abba",
        metadata={"source": "test", "case_seed": -1},
    )

    runs = spec.expand_runs()

    assert len(runs) == 4
    assert [run.run_id for run in runs] == [
        "spec-abba-g0001",
        "spec-abba-g0002",
        "spec-abba-g0003",
        "spec-abba-g0004",
    ]
    assert [run.turn_policy for run in runs] == [
        "fixed_round_robin",
        "bid_reply",
        "bid_reply",
        "fixed_round_robin",
    ]
    assert [(run.role_seed, run.actor_seed, run.orchestrator_seed) for run in runs] == [
        (101, 100_101, 200_101),
        (101, 100_101, 200_101),
        (102, 100_102, 200_102),
        (102, 100_102, 200_102),
    ]
    assert [run.metadata["policy_game_idx"] for run in runs] == [1, 1, 2, 2]
    assert [run.metadata["pair_id"] for run in runs] == [
        "pair-0001",
        "pair-0001",
        "pair-0002",
        "pair-0002",
    ]
    assert [run.metadata["counterbalance_order"] for run in runs] == ["AB", "AB", "BA", "BA"]
    assert [run.metadata["abba_position"] for run in runs] == [1, 2, 3, 4]
    assert {run.metadata["scheduled_total"] for run in runs} == {4}
    assert {run.metadata["runs_per_policy"] for run in runs} == {2}
    assert [run.metadata["case_seed"] for run in runs] == [101, 101, 102, 102]
    assert all(run.metadata["source"] == "test" for run in runs)
    assert all(run.metadata["protocol_version"] == "turn_policy_ablation.v1" for run in runs)
    assert all(run.metadata["player_names"] == ["A", "B", "C", "D", "E", "F"] for run in runs)
    assert all(run.metadata["experiment_spec_hash"] == spec.spec_hash for run in runs)


def test_experiment_spec_sequential_replicates_are_also_per_policy() -> None:
    runs = ExperimentSpec(
        experiment_id="spec-sequential",
        player_names=["A", "B", "C", "D", "E", "F"],
        turn_policies=["fixed_round_robin", "bid_reply"],
        replicates=2,
        base_seed=10,
    ).expand_runs()

    assert [(run.turn_policy, run.role_seed) for run in runs] == [
        ("fixed_round_robin", 11),
        ("fixed_round_robin", 12),
        ("bid_reply", 11),
        ("bid_reply", 12),
    ]


def test_experiment_spec_rejects_incomplete_abba_block() -> None:
    spec = ExperimentSpec(
        experiment_id="bad-abba",
        player_names=["A", "B", "C", "D", "E", "F"],
        turn_policies=["fixed_round_robin", "bid_reply"],
        replicates=1,
        base_seed=10,
        policy_order="abba",
    )

    with pytest.raises(ValueError, match="even number of runs per policy"):
        spec.expand_runs()


def test_harness_policy_schedule_rejects_unpaired_multi_policy_without_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        build_policy_schedule(
            2,
            ["fixed_round_robin", "bid_reply"],
            policy_order="sequential",
            seed=None,
            experiment_id="bad",
        )


def test_harness_experiment_metadata_is_safe_and_structured() -> None:
    schedule = build_policy_schedule(
        1,
        ["fixed_round_robin"],
        policy_order="sequential",
        seed=10,
        experiment_id="exp",
    )
    meta = experiment_metadata(schedule[0], player_names=["A", "B", "C", "D", "E", "F"])

    assert meta["protocol_version"] == "turn_policy_ablation.v1"
    assert meta["experiment_id"] == "exp"
    assert meta["runs_per_policy"] == 1
    assert meta["global_game_idx"] == 1
    assert meta["policy_game_idx"] == 1
    assert meta["role_seed"] == 11
    assert meta["actor_seed"] == 100011
    assert meta["orchestrator_seed"] == 200011
    assert meta["player_names"] == ["A", "B", "C", "D", "E", "F"]


def test_cyclic_seat_permutation_is_paired_within_abba_and_rotates_between_cases() -> None:
    schedule = build_policy_schedule(
        2,
        ["fixed_round_robin", "bid_reply"],
        policy_order="abba",
        seed=100,
        experiment_id="permuted-abba",
        seat_count=6,
        seat_permutation="cyclic",
    )

    assert schedule[0]["seat_permutation"] == schedule[1]["seat_permutation"] == [1, 2, 3, 4, 5, 6]
    assert schedule[2]["seat_permutation"] == schedule[3]["seat_permutation"] == [2, 3, 4, 5, 6, 1]
    assert [row["permutation_id"] for row in schedule] == [
        "seat-rotation-00",
        "seat-rotation-00",
        "seat-rotation-01",
        "seat-rotation-01",
    ]
    assert apply_seat_permutation(
        ["A", "B", "C", "D", "E", "F"],
        schedule[2],
    ) == ["B", "C", "D", "E", "F", "A"]


def test_experiment_spec_records_cyclic_permutation_in_concrete_run_provenance() -> None:
    source_names = ["A", "B", "C", "D", "E", "F"]
    runs = ExperimentSpec(
        experiment_id="permuted-spec",
        player_names=source_names,
        role_deck=["werewolf", "werewolf", "seer", "villager", "villager", "villager"],
        turn_policies=["fixed_round_robin"],
        replicates=2,
        base_seed=10,
        metadata={"seat_permutation_mode": "cyclic"},
    ).expand_runs()

    assert runs[0].player_names == source_names
    assert runs[1].player_names == ["B", "C", "D", "E", "F", "A"]
    assert runs[1].metadata["seat_rotation"] == 1
    assert runs[1].metadata["seat_permutation"] == [2, 3, 4, 5, 6, 1]
    assert runs[1].metadata["source_player_names"] == source_names


def test_fixed_schedule_does_not_add_permutation_provenance() -> None:
    row = build_policy_schedule(
        1,
        ["fixed_round_robin"],
        policy_order="sequential",
        seed=10,
        experiment_id="fixed",
    )[0]

    assert "seat_permutation" not in row
    assert "seat_rotation" not in row


def test_seat_and_persona_counterbalancing_uses_full_cartesian_control_cycle() -> None:
    schedule = build_policy_schedule(
        36,
        ["fixed_round_robin"],
        policy_order="sequential",
        seed=100,
        experiment_id="cartesian-controls",
        seat_count=6,
        seat_permutation="cyclic",
        role_layout_mode="fixed",
        persona_mode="counterbalanced",
    )

    combinations = {
        (row["seat_rotation"], row["persona_counterbalance_position"])
        for row in schedule
    }
    assert combinations == {
        (rotation, position)
        for rotation in range(6)
        for position in range(1, 7)
    }
    assert {row["role_seed"] for row in schedule} == {101}
    assert {row["role_layout_control_cycle"] for row in schedule} == {36}
    # Seat is the fast axis; persona advances only after every identity has
    # occupied every physical-seat rotation for the current persona position.
    assert [row["seat_rotation"] for row in schedule[:6]] == list(range(6))
    assert {row["persona_counterbalance_position"] for row in schedule[:6]} == {1}
    assert schedule[0]["source_persona_profile_ids"] == schedule[5]["source_persona_profile_ids"]
    assert schedule[6]["persona_counterbalance_position"] == 2
    assert schedule[6]["source_persona_profile_ids"] != schedule[0]["source_persona_profile_ids"]


def test_counterbalanced_role_layout_changes_only_after_complete_control_block() -> None:
    schedule = build_policy_schedule(
        72,
        ["fixed_round_robin"],
        policy_order="sequential",
        seed=100,
        experiment_id="role-blocks",
        seat_count=6,
        seat_permutation="cyclic",
        role_layout_mode="counterbalanced",
        role_layout_seed=900,
        role_layout_count=2,
        persona_mode="counterbalanced",
    )

    assert {row["role_seed"] for row in schedule[:36]} == {900}
    assert {row["role_layout_index"] for row in schedule[:36]} == {1}
    assert {row["role_seed"] for row in schedule[36:]} == {901}
    assert {row["role_layout_index"] for row in schedule[36:]} == {2}
    assert {
        (row["seat_rotation"], row["persona_counterbalance_position"])
        for row in schedule[:36]
    } == {
        (rotation, position)
        for rotation in range(6)
        for position in range(1, 7)
    }


def test_abba_policies_receive_identical_experimental_controls_per_case() -> None:
    schedule = build_policy_schedule(
        36,
        ["fixed_round_robin", "bid_reply"],
        policy_order="abba",
        seed=100,
        experiment_id="paired-controls",
        seat_count=6,
        seat_permutation="cyclic",
        role_layout_mode="fixed",
        persona_mode="counterbalanced",
    )

    by_pair: dict[str, list[dict]] = {}
    for row in schedule:
        by_pair.setdefault(row["pair_id"], []).append(row)
    assert len(by_pair) == 36
    for pair in by_pair.values():
        assert len(pair) == 2
        assert pair[0]["role_seed"] == pair[1]["role_seed"]
        assert pair[0]["seat_permutation"] == pair[1]["seat_permutation"]
        assert (
            pair[0]["source_persona_assignment_id"]
            == pair[1]["source_persona_assignment_id"]
        )


def test_counterbalanced_persona_rejects_diagonal_or_partial_control_cycle() -> None:
    with pytest.raises(ValueError, match=r"seat-by-persona control cycle \(36\)"):
        build_policy_schedule(
            6,
            ["fixed_round_robin"],
            policy_order="sequential",
            seed=100,
            experiment_id="partial-controls",
            seat_count=6,
            seat_permutation="cyclic",
            persona_mode="counterbalanced",
        )


def test_persona_provenance_is_recomputable_and_rejects_source_tampering() -> None:
    run = ExperimentSpec(
        experiment_id="persona-provenance",
        player_names=["A", "B", "C", "D", "E", "F"],
        turn_policies=["fixed_round_robin"],
        replicates=1,
        base_seed=40,
        metadata={"persona_mode": "fixed", "persona_seed": 900},
    ).expand_runs()[0]

    assignments = validate_persona_provenance(
        run.metadata,
        player_names=run.player_names,
    )
    assert len(assignments) == 6

    bad_profile = deepcopy(run.metadata)
    bad_profile["source_persona_profile_ids"][0] = (
        "direct_confrontation"
        if bad_profile["source_persona_profile_ids"][0] != "direct_confrontation"
        else "observe_wait"
    )
    with pytest.raises(ValueError, match="deterministic controls"):
        validate_persona_provenance(bad_profile, player_names=run.player_names)

    bad_source = deepcopy(run.metadata)
    bad_source["persona_assignments"][0]["source_player_name"] = "forged-player"
    with pytest.raises(ValueError, match="source player"):
        validate_persona_provenance(bad_source, player_names=run.player_names)

    bad_position = deepcopy(run.metadata)
    bad_position["persona_counterbalance_position"] = 2
    with pytest.raises(ValueError, match="counterbalance position"):
        validate_persona_provenance(bad_position, player_names=run.player_names)
