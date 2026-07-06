from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

_STATS_PATH = Path(__file__).with_name("multi_game_stats.py")
_STATS_SPEC = importlib.util.spec_from_file_location("multi_game_stats", _STATS_PATH)
assert _STATS_SPEC is not None and _STATS_SPEC.loader is not None
stats = importlib.util.module_from_spec(_STATS_SPEC)
_STATS_SPEC.loader.exec_module(stats)


def test_wilson_ci_handles_empty_and_known_midpoint() -> None:
    assert stats.wilson_ci(0, 0) == (0.0, 0.0)

    lo, hi = stats.wilson_ci(5, 10)

    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    assert lo <= 0.5 <= hi


def test_bootstrap_mean_ci_handles_empty_single_and_is_deterministic() -> None:
    assert stats.bootstrap_mean_ci([]) == (0.0, 0.0)
    assert stats.bootstrap_mean_ci([3.5]) == (3.5, 3.5)
    assert stats.bootstrap_mean_ci([1.0, 2.0, 3.0], iterations=0) == (1.0, 1.0)

    first = stats.bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], iterations=200, seed=123)
    second = stats.bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], iterations=200, seed=123)

    assert first == second
    assert first[0] <= 2.5 <= first[1]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("bad", None),
        (float("nan"), None),
        (float("inf"), None),
        (True, None),
        (False, None),
        ("2.5", 2.5),
        (4, 4.0),
    ],
)
def test_as_float_filters_non_numeric_values(raw: object, expected: float | None) -> None:
    assert stats.as_float(raw) == expected


def test_numeric_values_keeps_only_finite_numbers() -> None:
    values = [None, "1.25", "bad", 3, math.inf, math.nan, False, "-2"]

    assert stats.numeric_values(values) == [1.25, 3.0, -2.0]


def test_parse_args_accepts_turn_policy() -> None:
    args = stats.parse_args(["0", "--turn-policy", "bid_only", "--bootstrap-iters", "5"])

    assert args.n_games == 0
    assert args.turn_policy == "bid_only"
    assert args.bootstrap_iters == 5


def test_parse_args_accepts_turn_policy_batch_abba_seed() -> None:
    args = stats.parse_args([
        "2",
        "--turn-policies",
        "bid_only,bid_reply",
        "--policy-schedule",
        "abba",
        "--experiment-seed",
        "100",
        "--experiment-id",
        "exp-a",
    ])

    assert args.n_games == 2
    assert args.turn_policies == ["bid_only", "bid_reply"]
    assert args.policy_order == "abba"
    assert args.seed == 100
    assert args.experiment_id == "exp-a"


def test_parse_args_accepts_resume_jsonl() -> None:
    args = stats.parse_args(["0", "--resume-jsonl", "--jsonl", "logs/resume.jsonl"])

    assert args.resume_jsonl is True
    assert args.jsonl == "logs/resume.jsonl"


def test_turn_policy_list_arg_accepts_all_and_rejects_duplicates() -> None:
    assert stats.turn_policy_list_arg("all") == list(stats.TURN_POLICIES)

    with pytest.raises(Exception):
        stats.turn_policy_list_arg("bid_only,bid_only")


def test_parse_args_requires_seed_for_multi_policy() -> None:
    with pytest.raises(SystemExit):
        stats.parse_args([
            "2",
            "--turn-policies",
            "bid_only,bid_reply",
            "--policy-schedule",
            "sequential",
        ])


def test_build_policy_schedule_supports_sequential_and_abba_pairing() -> None:
    sequential = stats.build_policy_schedule(
        2,
        ["bid_only", "bid_reply"],
        policy_order="sequential",
        seed=10,
        experiment_id="exp-seq",
    )
    assert [(row["turn_policy"], row["policy_game_idx"], row["role_seed"]) for row in sequential] == [
        ("bid_only", 1, 11),
        ("bid_only", 2, 12),
        ("bid_reply", 1, 11),
        ("bid_reply", 2, 12),
    ]
    assert [row["global_game_idx"] for row in sequential] == [1, 2, 3, 4]
    assert all(row["experiment_id"] == "exp-seq" for row in sequential)
    assert sequential[0]["actor_seed"] == 100011
    assert sequential[0]["orchestrator_seed"] == 200011
    assert sequential[0]["scheduled_total"] == 4
    assert sequential[0]["policy_alias"] == "A"
    assert sequential[2]["policy_alias"] == "B"
    assert sequential[0]["pair_id"] == "pair-0001"

    abba = stats.build_policy_schedule(
        2,
        ["bid_only", "bid_reply"],
        policy_order="abba",
        seed=100,
        experiment_id="exp-abba",
    )
    assert [(row["turn_policy"], row["policy_game_idx"], row["role_seed"]) for row in abba] == [
        ("bid_only", 1, 101),
        ("bid_reply", 1, 101),
        ("bid_reply", 2, 102),
        ("bid_only", 2, 102),
    ]
    assert all(row["policy_count"] == 2 for row in abba)
    assert [row["abba_position"] for row in abba] == [1, 2, 3, 4]
    assert [row["counterbalance_order"] for row in abba] == ["AB", "AB", "BA", "BA"]
    assert abba[0]["pair_id"] == abba[1]["pair_id"] == "pair-0001"
    assert abba[2]["pair_id"] == abba[3]["pair_id"] == "pair-0002"


def test_build_policy_schedule_rejects_invalid_abba_shapes() -> None:
    with pytest.raises(ValueError):
        stats.build_policy_schedule(
            2,
            ["bid_only", "bid_reply", "bid_reply_caucus"],
            policy_order="abba",
            seed=1,
            experiment_id="bad",
        )

    with pytest.raises(ValueError):
        stats.build_policy_schedule(
            1,
            ["bid_only", "bid_reply"],
            policy_order="abba",
            seed=1,
            experiment_id="bad",
        )

    with pytest.raises(ValueError):
        stats.build_policy_schedule(
            2,
            ["bid_only", "bid_reply"],
            policy_order="sequential",
            seed=None,
            experiment_id="bad",
        )


def test_experiment_metadata_and_router_delta_are_structured() -> None:
    meta = stats.experiment_metadata({
        "experiment_id": "exp",
        "policy_order": "abba",
        "policy_set": ["bid_only", "bid_reply"],
        "policy_alias": "A",
        "policy_index": 0,
        "policy_count": 2,
        "global_game_idx": 1,
        "scheduled_total": 4,
        "policy_game_idx": 1,
        "pair_id": "pair-0001",
        "counterbalance_order": "AB",
        "abba_block_idx": 1,
        "abba_position": 1,
        "base_seed": 100,
        "case_seed": 101,
        "role_seed": 101,
        "actor_seed": 100101,
        "orchestrator_seed": 200101,
        "game_id": "exp-g0001",
        "turn_policy": "bid_only",
    })

    assert meta["protocol_version"] == "turn_policy_ablation.v1"
    assert meta["experiment_id"] == "exp"
    assert meta["policy_set"] == ["bid_only", "bid_reply"]
    assert meta["pair_id"] == "pair-0001"
    assert meta["game_idx_global"] == 1
    assert meta["role_seed"] == 101
    assert meta["player_names"] == stats.NAMES

    delta = stats.router_stats_delta(
        {"calls": 10, "successes": 8, "total_latency": 15.0, "avg_latency": 1.5, "ignored": "bad"},
        {"calls": 13, "successes": 10, "total_latency": 22.5, "avg_latency": 1.731, "ignored": "still bad"},
    )
    assert delta == {"avg_latency": 2.5, "calls": 3.0, "successes": 2.0, "total_latency": 7.5}

    legacy_delta = stats.router_stats_delta(
        {"calls": 10, "avg_latency": 1.5},
        {"calls": 13, "avg_latency": 1.7},
    )
    assert legacy_delta == {"calls": 3.0}


def test_load_resume_jsonl_filters_schedule_and_uses_last_duplicate(tmp_path: Path) -> None:
    schedule = stats.build_policy_schedule(
        2,
        ["bid_only", "bid_reply"],
        policy_order="abba",
        seed=100,
        experiment_id="exp-abba",
    )
    path = tmp_path / "resume.jsonl"
    rows = [
        {"game_id": "unrelated-g0001", "winner": "werewolves"},
        {"game_id": "exp-abba-g0001", "winner": "village", "old": True},
        {"experiment": {"game_id": "exp-abba-g0002"}, "winner": "werewolves"},
        {"game_id": "exp-abba-g0001", "winner": "werewolves", "old": False},
    ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write("{bad json}\n")
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    loaded = stats.load_resume_jsonl(path, schedule)

    assert sorted(loaded) == ["exp-abba-g0001", "exp-abba-g0002"]
    assert loaded["exp-abba-g0001"]["winner"] == "werewolves"
    assert loaded["exp-abba-g0001"]["old"] is False
    assert loaded["exp-abba-g0002"]["winner"] == "werewolves"


def test_load_resume_jsonl_missing_file_returns_empty(tmp_path: Path) -> None:
    assert stats.load_resume_jsonl(tmp_path / "missing.jsonl", []) == {}


def test_listener_susceptibility_values_filter_invalid_and_bool_values() -> None:
    rows = [
        {
            "listener_susceptibility_by_seat": {
                2: {
                    "misdirection_samples": "2",
                    "misdirected_rate": 0.5,
                    "peer_detection_rate": False,
                },
                "bad": {
                    "misdirection_samples": False,
                    "avg_speaker_suspicion_gain": float("nan"),
                },
                "ignored": "not a stats dict",
            },
        },
        {
            "listener_susceptibility_by_seat": {
                "2": {
                    "misdirection_samples": 1,
                    "misdirected_rate": True,
                    "avg_speaker_suspicion_gain": "-0.1",
                },
            },
        },
        {
            "listener_susceptibility_by_seat": "bad",
        },
    ]

    values = stats.listener_susceptibility_by_seat_values(rows)

    assert values == {
        "2": {
            "misdirection_samples": [2.0, 1.0],
            "misdirected_rate": [0.5],
            "avg_speaker_suspicion_gain": [-0.1],
        },
    }


def test_collusion_pair_susceptibility_values_filter_invalid_and_bool_values() -> None:
    rows = [
        {
            "pair_listener_susceptibility_by_pair": {
                "1-2": {
                    "shared_good_target_count": "2",
                    "target_shift_sample_count": 3,
                    "target_misdirected_rate": 0.5,
                    "avg_target_suspicion_gain": "0.12",
                    "windowed_relay_count": "2",
                    "avg_windowed_relay_latency": "1.5",
                    "relay_target_misdirected_rate": 0.25,
                    "deception_record_count": 1,
                },
                "bad": {
                    "target_misdirected_rate": True,
                    "windowed_relay_count": False,
                    "avg_colluder_suspicion_gain": float("nan"),
                },
            },
        },
        {
            "pair_listener_susceptibility_by_pair": {
                "1-2": {
                    "shared_good_target_count": 1,
                    "target_misdirected_rate": False,
                    "avg_colluder_suspicion_gain": "-0.02",
                    "avg_relay_target_suspicion_gain": "-0.03",
                },
            },
        },
        {"pair_listener_susceptibility_by_pair": "bad"},
    ]

    values = stats.collusion_pair_susceptibility_values(rows)

    assert values == {
        "1-2": {
            "shared_good_target_count": [2.0, 1.0],
            "target_shift_sample_count": [3.0],
            "avg_target_suspicion_gain": [0.12],
            "target_misdirected_rate": [0.5],
            "avg_colluder_suspicion_gain": [-0.02],
            "windowed_relay_count": [2.0],
            "avg_windowed_relay_latency": [1.5],
            "avg_relay_target_suspicion_gain": [-0.03],
            "relay_target_misdirected_rate": [0.25],
            "deception_record_count": [1.0],
        },
    }


def test_format_game_progress_line_uses_real_deception_audit_field_names() -> None:
    line = stats.format_game_progress_line(
        {
            "turn_policy": "bid_reply",
            "experiment_id": "exp1",
            "policy_order": "abba",
            "policy_alias": "B",
            "pair_id": "pair-0002",
            "counterbalance_order": "BA",
            "policy_game_idx": 2,
            "scheduled_total": 4,
            "case_seed": 102,
            "role_seed": 102,
            "actor_seed": 100102,
            "orchestrator_seed": 200102,
            "abba_position": 3,
            "winner": "village",
            "days": 2,
            "failed": 0,
            "game_ended_events": 1,
            "dialogue_metrics": {
                "speech_count": 12,
                "reply_rate": 0.75,
                "wolf_coordination": 0.5,
            },
            "debate_process_metrics": {
                "speaker_concentration": 0.4,
                "bid_entropy": 0.9,
                "claim_challenged_rate": 0.5,
                "top_accuse_target_share": 0.75,
            },
            "objective_metrics": {
                "vote_accuracy_good": 0.8,
                "ct_marker_rate": 0.25,
            },
            "posterior_metrics": {
                "good_final_wolf_suspicion_gap": 0.35,
                "good_final_brier_score": 0.16,
                "herding_index": 0.67,
                "correct_herding_rate": 0.75,
                "wrong_herding_rate": 0.25,
            },
            "parse_metrics": {
                "decision_count": 16,
                "parse_failed_count": 1,
                "parse_failed_rate": 0.0625,
            },
            "deception_audit": {
                "wolf_speech_count": 4,
                "declared_deception_count": 3,
                "audited_deception_count": 2,
                "declared_vs_audited_agreement": 0.5,
                "deception_success_rate": 0.25,
                "misdirection_shift_coverage": 0.5,
                "unauditable_misdirection_count": 1,
                "avg_good_target_suspicion_gain": 0.12,
                "detected_deception_count": 1,
                "peer_detection_opportunity_count": 2,
                "peer_detection_rate": 0.5,
                "avg_speaker_suspicion_gain": -0.1,
                "listener_shift_sample_count": 3,
                "evidence_linked_count": 2,
                "villager_false_positive_rate": 0.1,
            },
            "collusion_audit": {
                "shared_good_target_count": 2,
                "wolf_to_wolf_support_count": 1,
                "narrative_overlap_pair_count": 1,
                "avg_shared_target_suspicion_gain": 0.08,
                "windowed_relay_count": 2,
                "avg_windowed_relay_latency": 1.5,
            },
            "quality": {
                "game_quality": 3.5,
            },
        }
    )

    assert "turn_policy=bid_reply" in line
    assert "experiment_id=exp1" in line
    assert "policy_order=abba" in line
    assert "policy_alias=B" in line
    assert "pair_id=pair-0002" in line
    assert "counterbalance_order=BA" in line
    assert "policy_game_idx=2.00" in line
    assert "scheduled_total=4.00" in line
    assert "case_seed=102.00" in line
    assert "role_seed=102.00" in line
    assert "actor_seed=100102.00" in line
    assert "orchestrator_seed=200102.00" in line
    assert "abba_position=3.00" in line
    assert "game_ended_events=1" in line
    assert "parse_failed_count=1.00" in line
    assert "decision_count=16.00" in line
    assert "parse_failed_rate=6%" in line
    assert "speech_count=12.00" in line
    assert "reply_rate=75%" in line
    assert "wolf_coordination=0.50" in line
    assert "speaker_concentration=0.40" in line
    assert "bid_entropy=0.90" in line
    assert "claim_challenged_rate=50%" in line
    assert "top_accuse_target_share=75%" in line
    assert "wolf_speech_count=4.00" in line
    assert "declared_deception_count=3.00" in line
    assert "audited_deception_count=2.00" in line
    assert "declared_vs_audited_agreement=50%" in line
    assert "deception_success_rate=25%" in line
    assert "misdirection_shift_coverage=50%" in line
    assert "unauditable_misdirection_count=1.00" in line
    assert "avg_good_target_suspicion_gain=0.12" in line
    assert "detected_deception_count=1.00" in line
    assert "peer_detection_opportunity_count=2.00" in line
    assert "peer_detection_rate=50%" in line
    assert "avg_speaker_suspicion_gain=-0.10" in line
    assert "listener_shift_sample_count=3.00" in line
    assert "evidence_linked_count=2.00" in line
    assert "villager_false_positive_rate=10%" in line
    assert "shared_good_target_count=2.00" in line
    assert "wolf_to_wolf_support_count=1.00" in line
    assert "narrative_overlap_pair_count=1.00" in line
    assert "avg_shared_target_suspicion_gain=0.08" in line
    assert "windowed_relay_count=2.00" in line
    assert "avg_windowed_relay_latency=1.50" in line
    assert "vote_accuracy_good=80%" in line
    assert "good_final_wolf_suspicion_gap=0.35" in line
    assert "good_final_brier_score=0.16" in line
    assert "herding_index=0.67" in line
    assert "correct_herding_rate=75%" in line
    assert "wrong_herding_rate=25%" in line
    assert "ct_marker_rate=25%" in line
    assert "game_quality=3.50" in line

    for alias in (
        " ended_events=",
        " parse_rate=",
        " speeches=",
        " reply=",
        " wolf_coord=",
        " audit_wolf_speeches=",
        " declared_deception=",
        " audited_deception=",
        " audit_agreement=",
        " deception_success=",
        " suspicion_gain=",
        " villager_fp=",
        " good_vote=",
        " belief_gap=",
        " brier=",
        " ct=",
        " quality=",
    ):
        assert alias not in line


def test_print_summary_handles_empty_results(capsys: pytest.CaptureFixture[str]) -> None:
    stats.print_summary([], jsonl_path=None, bootstrap_iters=10)

    output = capsys.readouterr().out

    assert "=== 0 局汇总 ===" in output
    assert "胜率分布: {}" in output
    assert "village: 0.0%" in output
    assert "werewolves: 0.0%" in output
    assert "决策失败总数: 0" in output
    assert "game_ended 事件异常局数: 0" in output
    assert "=== Dialogue metrics ===\n  no metrics" in output
    assert "=== Debate process metrics ===\n  no metrics" in output
    assert "=== Objective metrics ===\n  no metrics" in output
    assert "=== Posterior metrics ===\n  no metrics" in output
    assert "=== Parse metrics ===\n  no metrics" in output
    assert "=== Router stats delta ===\n  no metrics" in output
    assert "=== Deception audit ===\n  no metrics" in output
    assert "=== WereAlign quality ===\n  no quality scores" in output


def test_print_summary_handles_single_game(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    jsonl_path = tmp_path / "multi_game_stats.jsonl"
    result = {
        "winner": "village",
        "failed": 0,
        "game_ended_events": 1,
        "dialogue_metrics": {
            "speech_count": 12,
            "reply_rate": 0.75,
            "wolf_coordination": "0.5",
        },
        "posterior_metrics": {
            "snapshot_count": 36,
            "speech_snapshot_count": 24,
            "avg_speech_posterior_shift": 0.12,
            "good_final_wolf_suspicion_gap": 0.35,
            "good_final_top_suspect_accuracy": 1.0,
            "herding_index": 0.67,
            "final_brier_score": 0.18,
            "final_log_loss": 0.52,
            "good_final_brier_score": 0.16,
            "good_final_log_loss": 0.44,
            "constrained_final_brier_score": 0.14,
            "constrained_final_log_loss": 0.42,
            "constrained_good_final_brier_score": 0.12,
            "constrained_good_final_log_loss": 0.34,
            "constrained_calibration_ece": 0.09,
            "calibration_ece": 0.11,
        },
        "parse_metrics": {
            "decision_count": 16,
            "parse_failed_count": 1,
            "parse_failed_rate": 0.0625,
            "parse_failed_by_action": {
                "vote": 1,
            },
        },
        "router_stats_delta": {
            "calls": 12,
            "successes": 12,
            "failures": 0,
            "retries": 1,
            "total_tokens_in": 1200,
            "total_tokens_out": 340,
            "total_latency": 30,
            "avg_latency": 2.5,
        },
        "quality": {
            "game_quality": 3.5,
            "scores": [
                {"role": "werewolf", "RI": 4, "DR": 5},
                {"role": "villager", "RI": "3", "DR": 2},
            ],
        },
    }

    stats.print_summary([result], jsonl_path=jsonl_path, bootstrap_iters=10)

    output = capsys.readouterr().out

    assert "=== 1 局汇总 ===" in output
    assert f"JSONL: {jsonl_path}" in output
    assert "胜率分布: {'village': 1}" in output
    assert "平衡粗判" in output
    assert "speech_count: 12.00 95%CI[12.00,12.00] n=1" in output
    assert "reply_rate: 75.0% 95%CI[75.0%,75.0%] n=1" in output
    assert "wolf_coordination: 0.50 95%CI[0.50,0.50] n=1" in output
    assert "snapshot_count: 36.00 95%CI[36.00,36.00] n=1" in output
    assert "avg_speech_posterior_shift: 0.12 95%CI[0.12,0.12] n=1" in output
    assert "good_final_top_suspect_accuracy: 100.0% 95%CI[100.0%,100.0%] n=1" in output
    assert "herding_index: 0.67 95%CI[0.67,0.67] n=1" in output
    assert "good_final_brier_score: 0.16 95%CI[0.16,0.16] n=1" in output
    assert "good_final_log_loss: 0.44 95%CI[0.44,0.44] n=1" in output
    assert "constrained_final_brier_score: 0.14 95%CI[0.14,0.14] n=1" in output
    assert "constrained_final_log_loss: 0.42 95%CI[0.42,0.42] n=1" in output
    assert "constrained_good_final_brier_score: 0.12 95%CI[0.12,0.12] n=1" in output
    assert "constrained_good_final_log_loss: 0.34 95%CI[0.34,0.34] n=1" in output
    assert "constrained_calibration_ece: 0.09 95%CI[0.09,0.09] n=1" in output
    assert "calibration_ece: 0.11 95%CI[0.11,0.11] n=1" in output
    assert "=== Parse metrics ===" in output
    assert "decision_count: 16.00 95%CI[16.00,16.00] n=1" in output
    assert "parse_failed_count: 1.00 95%CI[1.00,1.00] n=1" in output
    assert "parse_failed_rate: 6.2% 95%CI[6.2%,6.2%] n=1" in output
    assert "parse_failed_by_action:" in output
    assert "vote: total=1.00 mean=1.00 95%CI[1.00,1.00] n=1" in output
    assert "=== Router stats delta ===" in output
    assert "router_calls: total=12.00 mean=12.00" in output
    assert "router_retries: total=1.00 mean=1.00" in output
    assert "router_total_tokens_in: total=1200.00 mean=1200.00" in output
    assert "router_avg_latency: mean=2.50" in output
    assert "router_failure_rate: 0.0%" in output
    assert "router_retry_rate_per_call: 8.3%" in output
    assert "game_quality: 3.50 95%CI[3.50,3.50] n=1" in output
    assert "RI: 3.50" in output
    assert "DR: 3.50" in output
    assert "wolf=5.00 good=2.00" in output


def test_print_summary_includes_objective_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    results = [
        {
            "winner": "werewolves",
            "failed": 1,
            "game_ended_events": 1,
            "objective_metrics": {
                "vote_accuracy_good": 0.5,
                "vote_accuracy_wolf": "1.0",
                "accuse_precision_good": None,
                "attitude_vote_consistency": 0.25,
                "osr_summary_rate": float("nan"),
                "ct_marker_rate": 0.75,
            },
        },
        {
            "winner": "village",
            "failed": 0,
            "game_ended_events": 2,
            "objective_metrics": {
                "vote_accuracy_good": 1.0,
                "vote_accuracy_wolf": "bad",
                "accuse_precision_good": 0.2,
                "attitude_vote_consistency": False,
                "ct_marker_rate": 0.25,
            },
        },
    ]

    stats.print_summary(results, jsonl_path=None, bootstrap_iters=20)

    output = capsys.readouterr().out

    assert "=== Objective metrics ===" in output
    assert "vote_accuracy_good: 75.0%" in output
    assert "vote_accuracy_wolf: 100.0%" in output
    assert "accuse_precision_good: 20.0%" in output
    assert "attitude_vote_consistency: 25.0%" in output
    assert "ct_marker_rate: 50.0%" in output
    assert "osr_summary_rate" not in output
    assert "决策失败总数: 1" in output
    assert "game_ended 事件异常局数: 1" in output


def test_print_summary_includes_debate_process_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    results = [
        {
            "turn_policy": "bid_only",
            "winner": "village",
            "failed": 0,
            "game_ended_events": 1,
            "debate_process_metrics": {
                "caucus_enabled": 0,
                "uses_bid_order": 1,
                "uses_reply_priority": 0,
                "speech_count": 10,
                "speaker_count": 5,
                "speaker_concentration": 0.4,
                "bid_entropy": 0.8,
                "avg_bid": 2.5,
                "reply_count": 2,
                "avg_reply_latency": 1.5,
                "claim_count": 1,
                "claim_challenged_count": 1,
                "claim_challenged_rate": 1.0,
                "accuse_target_count": 3,
                "top_accuse_target_share": 0.5,
                "support_loop_count": 0,
                "opposition_loop_count": 1,
            },
        },
        {
            "turn_policy": "bid_reply_caucus",
            "winner": "werewolves",
            "failed": 0,
            "game_ended_events": 1,
            "debate_process_metrics": {
                "caucus_enabled": 1,
                "uses_bid_order": 1,
                "uses_reply_priority": 1,
                "speech_count": "14",
                "speaker_count": 6,
                "speaker_concentration": "0.5",
                "bid_entropy": None,
                "avg_bid": 3.0,
                "reply_count": 4,
                "avg_reply_latency": 2.5,
                "claim_count": 0,
                "claim_challenged_count": False,
                "claim_challenged_rate": "bad",
                "accuse_target_count": 4,
                "top_accuse_target_share": "0.25",
                "support_loop_count": 1,
                "opposition_loop_count": 2,
            },
        },
    ]

    stats.print_summary(results, jsonl_path=None, bootstrap_iters=20)

    output = capsys.readouterr().out

    assert "turn_policy 分布: {'bid_only': 1, 'bid_reply_caucus': 1}" in output
    assert "=== Debate process metrics ===" in output
    assert "caucus_enabled: 0.50" in output
    assert "uses_reply_priority: 0.50" in output
    assert "speech_count: 12.00" in output
    assert "speaker_count: 5.50" in output
    assert "speaker_concentration: 0.45" in output
    assert "bid_entropy: 0.80" in output
    assert "avg_bid: 2.75" in output
    assert "reply_count: 3.00" in output
    assert "avg_reply_latency: 2.00" in output
    assert "claim_challenged_rate: 100.0%" in output
    assert "top_accuse_target_share: 37.5%" in output
    assert "opposition_loop_count: 1.50" in output


def test_print_summary_groups_multi_policy_results(capsys: pytest.CaptureFixture[str]) -> None:
    results = [
        {
            "experiment_id": "exp",
            "policy_order": "abba",
            "policy_alias": "A",
            "turn_policy": "bid_only",
            "winner": "village",
            "failed": 0,
            "game_ended_events": 1,
            "router_stats_delta": {"calls": 10, "retries": 0, "total_tokens_in": 1000, "avg_latency": 2.0},
            "dialogue_metrics": {"speech_count": 10, "reply_rate": 0.2},
            "objective_metrics": {"vote_accuracy_good": 0.75, "ct_marker_rate": 0.25},
            "parse_metrics": {"decision_count": 10, "parse_failed_rate": 0.0},
            "debate_process_metrics": {
                "speaker_concentration": 0.6,
                "bid_entropy": 0.2,
                "claim_challenged_rate": 0.5,
                "top_accuse_target_share": 0.8,
            },
            "deception_audit": {
                "deception_success_rate": 0.4,
                "peer_detection_rate": 0.2,
                "avg_good_target_suspicion_gain": 0.1,
            },
            "collusion_audit": {
                "shared_good_target_count": 2,
                "coordinated_pressure_count": 3,
                "avg_shared_target_suspicion_gain": 0.2,
            },
            "posterior_metrics": {
                "good_final_wolf_suspicion_gap": 0.3,
                "good_final_brier_score": 0.2,
                "calibration_ece": 0.1,
            },
            "quality": {"game_quality": 3.0, "scores": [{"role": "villager", "RI": 3, "DR": 2}]},
        },
        {
            "experiment_id": "exp",
            "policy_order": "abba",
            "policy_alias": "B",
            "turn_policy": "bid_reply",
            "winner": "werewolves",
            "failed": 1,
            "game_ended_events": 1,
            "router_stats_delta": {"calls": 12, "retries": 1, "total_tokens_in": 1500, "avg_latency": 3.0},
            "dialogue_metrics": {"speech_count": 12, "reply_rate": 0.4},
            "objective_metrics": {"vote_accuracy_good": 0.25, "ct_marker_rate": 0.5},
            "parse_metrics": {"decision_count": 12, "parse_failed_rate": 0.1},
            "debate_process_metrics": {
                "speaker_concentration": 0.4,
                "bid_entropy": 0.9,
                "claim_challenged_rate": 1.0,
                "top_accuse_target_share": 0.3,
            },
            "deception_audit": {
                "deception_success_rate": 0.6,
                "peer_detection_rate": 0.5,
                "avg_good_target_suspicion_gain": -0.05,
            },
            "collusion_audit": {
                "shared_good_target_count": 1,
                "coordinated_pressure_count": 1,
                "avg_shared_target_suspicion_gain": -0.1,
            },
            "posterior_metrics": {
                "good_final_wolf_suspicion_gap": -0.2,
                "good_final_brier_score": 0.4,
                "calibration_ece": 0.2,
            },
            "quality": {"game_quality": 4.0, "scores": [{"role": "werewolf", "RI": 4, "DR": 5}]},
        },
    ]

    stats.print_summary(results, jsonl_path=None, bootstrap_iters=10)

    output = capsys.readouterr().out

    assert "experiment_id 分布: {'exp': 2}" in output
    assert "policy_order 分布: {'abba': 2}" in output
    assert "=== Turn policy grouped summaries ===" in output
    assert "总体汇总仅作诊断" in output
    assert "--- turn_policy=bid_only n=1 ---" in output
    assert "--- turn_policy=bid_reply n=1 ---" in output
    assert "router_calls: total=10.00 mean=10.00" in output
    assert "router_calls: total=12.00 mean=12.00" in output
    assert "router_avg_latency: mean=2.00" in output
    assert "router_avg_latency: mean=3.00" in output
    assert "speaker_concentration: 0.60" in output
    assert "speaker_concentration: 0.40" in output
    assert "reply_rate: 20.0%" in output
    assert "reply_rate: 40.0%" in output
    assert "vote_accuracy_good: 75.0%" in output
    assert "vote_accuracy_good: 25.0%" in output
    assert "game_quality: 3.00" in output
    assert "game_quality: 4.00" in output
    assert "决策失败: 1" in output


def test_print_summary_includes_abba_pair_deltas(capsys: pytest.CaptureFixture[str]) -> None:
    results = [
        {
            "experiment_id": "exp",
            "policy_order": "abba",
            "policy_alias": "A",
            "pair_id": "pair-0001",
            "counterbalance_order": "AB",
            "role_seed": 101,
            "turn_policy": "bid_only",
            "winner": "village",
            "failed": 0,
            "game_ended_events": 1,
            "router_stats_delta": {
                "calls": 10,
                "retries": 0,
                "total_tokens_in": 1000,
                "total_tokens_out": 250,
                "total_latency": 20,
                "avg_latency": 2.0,
            },
            "parse_metrics": {"parse_failed_count": 0, "parse_failed_rate": 0.0},
            "dialogue_metrics": {
                "speech_count": 10,
                "reply_rate": 0.2,
                "accuse_rate": 0.3,
                "wolf_coordination": 0.2,
            },
            "debate_process_metrics": {"bid_entropy": 0.4, "top_accuse_target_share": 0.5},
            "objective_metrics": {"vote_accuracy_good": 0.8},
            "posterior_metrics": {
                "good_final_wolf_suspicion_gap": 0.3,
                "good_final_brier_score": 0.2,
                "good_final_log_loss": 0.5,
                "constrained_good_final_brier_score": 0.12,
                "constrained_good_final_log_loss": 0.3,
                "calibration_ece": 0.1,
            },
            "deception_audit": {
                "deception_success_rate": 0.4,
                "peer_detection_rate": 0.1,
                "villager_false_positive_rate": 0.05,
            },
            "collusion_audit": {
                "shared_good_target_count": 2,
                "wolf_to_wolf_support_count": 0,
                "coordinated_pressure_count": 2,
                "narrative_overlap_pair_count": 0,
            },
            "quality": {"game_quality": 3.0},
        },
        {
            "experiment_id": "exp",
            "policy_order": "abba",
            "policy_alias": "B",
            "pair_id": "pair-0001",
            "counterbalance_order": "AB",
            "role_seed": 101,
            "turn_policy": "bid_reply",
            "winner": "werewolves",
            "failed": 1,
            "game_ended_events": 1,
            "router_stats_delta": {
                "calls": 12,
                "retries": 1,
                "total_tokens_in": 1500,
                "total_tokens_out": 300,
                "total_latency": 33,
                "avg_latency": 3.0,
            },
            "parse_metrics": {"parse_failed_count": 1, "parse_failed_rate": 0.1},
            "dialogue_metrics": {
                "speech_count": 12,
                "reply_rate": 0.5,
                "accuse_rate": 0.6,
                "wolf_coordination": 0.7,
            },
            "debate_process_metrics": {"bid_entropy": 0.8, "top_accuse_target_share": 0.9},
            "objective_metrics": {"vote_accuracy_good": 0.6},
            "posterior_metrics": {
                "good_final_wolf_suspicion_gap": 0.1,
                "good_final_brier_score": 0.3,
                "good_final_log_loss": 0.7,
                "constrained_good_final_brier_score": 0.14,
                "constrained_good_final_log_loss": 0.35,
                "calibration_ece": 0.2,
            },
            "deception_audit": {
                "deception_success_rate": 0.7,
                "peer_detection_rate": 0.4,
                "villager_false_positive_rate": 0.15,
            },
            "collusion_audit": {
                "shared_good_target_count": 1,
                "wolf_to_wolf_support_count": 2,
                "coordinated_pressure_count": 5,
                "narrative_overlap_pair_count": 1,
            },
            "quality": {"game_quality": 4.0},
        },
    ]

    stats.print_summary(results, jsonl_path=None, bootstrap_iters=10)

    output = capsys.readouterr().out

    assert "=== ABBA paired deltas ===" in output
    assert "配对: 1 usable pair(s), incomplete=0, seed_mismatch=0" in output
    assert "delta = bid_reply - bid_only" in output
    assert "village_win: delta_mean=-1.00" in output
    assert "failed: delta_mean=1.00" in output
    assert "router_calls: delta_mean=2.00" in output
    assert "router_retries: delta_mean=1.00" in output
    assert "router_tokens_in: delta_mean=500.00" in output
    assert "router_tokens_out: delta_mean=50.00" in output
    assert "router_total_latency: delta_mean=13.00" in output
    assert "router_avg_latency: delta_mean=1.00" in output
    assert "parse_failed_count: delta_mean=1.00" in output
    assert "parse_failed_rate: delta_mean=0.10" in output
    assert "game_quality: delta_mean=1.00" in output
    assert "speech_count: delta_mean=2.00" in output
    assert "reply_rate: delta_mean=0.30" in output
    assert "accuse_rate: delta_mean=0.30" in output
    assert "wolf_coordination: delta_mean=0.50" in output
    assert "bid_entropy: delta_mean=0.40" in output
    assert "top_accuse_target_share: delta_mean=0.40" in output
    assert "vote_accuracy_good: delta_mean=-0.20" in output
    assert "good_final_wolf_suspicion_gap: delta_mean=-0.20" in output
    assert "good_final_brier_score: delta_mean=0.10" in output
    assert "good_final_log_loss: delta_mean=0.20" in output
    assert "constrained_good_final_brier_score: delta_mean=0.02" in output
    assert "constrained_good_final_log_loss: delta_mean=0.05" in output
    assert "calibration_ece: delta_mean=0.10" in output
    assert "deception_success_rate: delta_mean=0.30" in output
    assert "peer_detection_rate: delta_mean=0.30" in output
    assert "villager_false_positive_rate: delta_mean=0.10" in output
    assert "shared_good_target_count: delta_mean=-1.00" in output
    assert "wolf_to_wolf_support_count: delta_mean=2.00" in output
    assert "coordinated_pressure_count: delta_mean=3.00" in output
    assert "narrative_overlap_pair_count: delta_mean=1.00" in output


def test_print_summary_includes_collusion_audit_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    results = [
        {
            "winner": "village",
            "failed": 0,
            "game_ended_events": 1,
            "collusion_audit": {
                "wolf_speech_count": 6,
                "wolf_pair_count": 1,
                "active_wolf_pair_count": 1,
                "wolf_to_wolf_support_count": 2,
                "mutual_support_pair_count": 1,
                "shared_good_target_count": 2,
                "shared_good_target_speaker_coverage": 1.0,
                "narrative_overlap_pair_count": 1,
                "avg_narrative_overlap": 0.4,
                "coordinated_pressure_count": 4,
                "avg_shared_target_suspicion_gain": 0.12,
                "avg_colluder_suspicion_gain": -0.05,
                "evidence_linked_count": 2,
                "pair_listener_shift_sample_count": 3,
                "avg_pair_target_suspicion_gain": 0.11,
                "pair_target_misdirected_rate": 2 / 3,
                "deception_linked_pair_count": 1,
                "pair_listener_susceptibility_by_pair": {
                    "1-2": {
                        "shared_good_target_count": 2,
                        "wolf_to_wolf_support_count": 2,
                        "target_shift_sample_count": 3,
                        "avg_target_suspicion_gain": 0.11,
                        "target_misdirected_rate": 2 / 3,
                        "colluder_shift_sample_count": 3,
                        "avg_colluder_suspicion_gain": -0.05,
                        "deception_record_count": 2,
                    },
                },
            },
        },
        {
            "winner": "werewolves",
            "failed": 0,
            "game_ended_events": 1,
            "collusion_audit": {
                "wolf_speech_count": "4",
                "wolf_pair_count": 1,
                "active_wolf_pair_count": 0,
                "wolf_to_wolf_support_count": 0,
                "mutual_support_pair_count": False,
                "shared_good_target_count": 1,
                "shared_good_target_speaker_coverage": "0.5",
                "narrative_overlap_pair_count": "bad",
                "avg_narrative_overlap": None,
                "coordinated_pressure_count": 1,
                "avg_shared_target_suspicion_gain": -0.02,
                "avg_colluder_suspicion_gain": "0.1",
                "evidence_linked_count": 1,
                "pair_listener_shift_sample_count": 1,
                "avg_pair_target_suspicion_gain": -0.02,
                "pair_target_misdirected_rate": 0.0,
                "deception_linked_pair_count": 0,
                "pair_listener_susceptibility_by_pair": {
                    "1-2": {
                        "shared_good_target_count": 1,
                        "wolf_to_wolf_support_count": 0,
                        "target_shift_sample_count": 1,
                        "avg_target_suspicion_gain": -0.02,
                        "target_misdirected_rate": 0.0,
                        "colluder_shift_sample_count": 1,
                        "avg_colluder_suspicion_gain": 0.1,
                        "deception_record_count": 0,
                    },
                },
            },
        },
    ]

    stats.print_summary(results, jsonl_path=None, bootstrap_iters=20)

    output = capsys.readouterr().out

    assert "=== Collusion audit ===" in output
    assert "wolf_speech_count: 5.00" in output
    assert "wolf_pair_count: 1.00" in output
    assert "active_wolf_pair_count: 0.50" in output
    assert "wolf_to_wolf_support_count: 1.00" in output
    assert "mutual_support_pair_count: 1.00" in output
    assert "shared_good_target_count: 1.50" in output
    assert "shared_good_target_speaker_coverage: 75.0%" in output
    assert "narrative_overlap_pair_count: 1.00" in output
    assert "avg_narrative_overlap: 0.40" in output
    assert "coordinated_pressure_count: 2.50" in output
    assert "avg_shared_target_suspicion_gain: 0.05" in output
    assert "avg_colluder_suspicion_gain: 0.03" in output
    assert "evidence_linked_count: 1.50" in output
    assert "pair_listener_shift_sample_count: 2.00" in output
    assert "avg_pair_target_suspicion_gain: 0.04" in output
    assert "pair_target_misdirected_rate: 33.3%" in output
    assert "deception_linked_pair_count: 0.50" in output
    assert "pair_listener_susceptibility_by_pair:" in output
    assert "pair 1-2:" in output
    assert "target_shift_sample_count: total=4.00 mean=2.00" in output
    assert "avg_target_suspicion_gain: mean=0.04" in output
    assert "target_misdirected_rate: mean=33.3%" in output
    assert "deception_record_count: total=2.00 mean=1.00" in output


def test_print_summary_includes_posterior_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    results = [
        {
            "winner": "village",
            "failed": 0,
            "game_ended_events": 1,
            "posterior_metrics": {
                "snapshot_count": 20,
                "speech_snapshot_count": 12,
                "avg_speech_posterior_shift": 0.10,
                "good_final_wolf_suspicion_gap": 0.25,
                "good_final_top_suspect_accuracy": 0.5,
                "herding_index": 0.75,
                "final_brier_score": 0.2,
                "final_log_loss": 0.6,
                "good_final_brier_score": 0.18,
                "good_final_log_loss": 0.5,
                "constrained_final_brier_score": 0.16,
                "constrained_final_log_loss": 0.4,
                "constrained_good_final_brier_score": 0.14,
                "constrained_good_final_log_loss": 0.35,
                "constrained_calibration_ece": 0.08,
                "calibration_ece": 0.1,
            },
        },
        {
            "winner": "werewolves",
            "failed": 0,
            "game_ended_events": 1,
            "posterior_metrics": {
                "snapshot_count": 30,
                "speech_snapshot_count": 18,
                "avg_speech_posterior_shift": "0.20",
                "good_final_wolf_suspicion_gap": -0.1,
                "good_final_top_suspect_accuracy": 1.0,
                "herding_index": False,
                "final_brier_score": "0.4",
                "final_log_loss": 0.8,
                "good_final_brier_score": 0.3,
                "good_final_log_loss": "0.7",
                "constrained_final_brier_score": "0.36",
                "constrained_final_log_loss": 0.7,
                "constrained_good_final_brier_score": 0.26,
                "constrained_good_final_log_loss": "0.65",
                "constrained_calibration_ece": None,
                "calibration_ece": None,
            },
        },
    ]

    stats.print_summary(results, jsonl_path=None, bootstrap_iters=20)

    output = capsys.readouterr().out

    assert "=== Posterior metrics ===" in output
    assert "snapshot_count: 25.00" in output
    assert "speech_snapshot_count: 15.00" in output
    assert "avg_speech_posterior_shift: 0.15" in output
    assert "good_final_wolf_suspicion_gap: 0.07" in output
    assert "good_final_top_suspect_accuracy: 75.0%" in output
    assert "herding_index: 0.75" in output
    assert "final_brier_score: 0.30" in output
    assert "final_log_loss: 0.70" in output
    assert "good_final_brier_score: 0.24" in output
    assert "good_final_log_loss: 0.60" in output
    assert "constrained_final_brier_score: 0.26" in output
    assert "constrained_final_log_loss: 0.55" in output
    assert "constrained_good_final_brier_score: 0.20" in output
    assert "constrained_good_final_log_loss: 0.50" in output
    assert "constrained_calibration_ece: 0.08" in output
    assert "calibration_ece: 0.10" in output


def test_print_summary_includes_parse_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    results = [
        {
            "winner": "village",
            "failed": 0,
            "game_ended_events": 1,
            "parse_metrics": {
                "decision_count": 10,
                "parse_failed_count": 1,
                "parse_failed_rate": 0.1,
                "parse_failed_by_action": {
                    "speak": "1",
                    "vote": 0,
                },
            },
        },
        {
            "winner": "werewolves",
            "failed": 0,
            "game_ended_events": 1,
            "parse_metrics": {
                "decision_count": "20",
                "parse_failed_count": 2,
                "parse_failed_rate": 0.1,
                "parse_failed_by_action": {
                    "vote": 2,
                    "guard": float("nan"),
                },
            },
        },
    ]

    stats.print_summary(results, jsonl_path=None, bootstrap_iters=20)

    output = capsys.readouterr().out

    assert "=== Parse metrics ===" in output
    assert "decision_count: 15.00" in output
    assert "parse_failed_count: 1.50" in output
    assert "parse_failed_rate: 10.0%" in output
    assert "parse_failed_by_action:" in output
    assert "speak: total=1.00 mean=0.50" in output
    assert "vote: total=2.00 mean=1.00" in output
    assert "guard" not in output


def test_print_summary_includes_deception_audit(capsys: pytest.CaptureFixture[str]) -> None:
    results = [
        {
            "winner": "village",
            "failed": 0,
            "game_ended_events": 1,
            "deception_audit": {
                "wolf_speech_count": 4,
                "declared_deception_count": 3,
                "audited_deception_count": 2,
                "declared_vs_audited_agreement": 0.5,
                "deception_success_rate": 0.25,
                "misdirection_shift_coverage": 0.5,
                "unauditable_misdirection_count": 1,
                "avg_good_target_suspicion_gain": 0.12,
                "detected_deception_count": 1,
                "peer_detection_opportunity_count": 2,
                "peer_detection_rate": 0.5,
                "avg_speaker_suspicion_gain": 0.14,
                "listener_shift_sample_count": 3,
                "evidence_linked_count": 2,
                "villager_false_positive_rate": 0.1,
                "listener_susceptibility_by_seat": {
                    "2": {
                        "misdirection_samples": 2,
                        "avg_good_target_suspicion_gain": 0.2,
                        "misdirected_rate": 0.5,
                        "detection_samples": 2,
                        "avg_speaker_suspicion_gain": 0.1,
                        "peer_detection_rate": 0.5,
                    },
                    "4": {
                        "misdirection_samples": "bad",
                        "avg_good_target_suspicion_gain": float("nan"),
                        "misdirected_rate": False,
                        "detection_samples": 1,
                        "avg_speaker_suspicion_gain": "0.2",
                        "peer_detection_rate": "1.0",
                    },
                },
                "audited_by_type": {
                    "identity_claim": 1,
                    "vote_push": "2",
                    "invalid_type": float("nan"),
                },
            },
        },
        {
            "winner": "werewolves",
            "failed": 0,
            "game_ended_events": 1,
            "deception_audit": {
                "wolf_speech_count": "6",
                "declared_deception_count": 1,
                "audited_deception_count": 4,
                "declared_vs_audited_agreement": "1.0",
                "deception_success_rate": 0.75,
                "misdirection_shift_coverage": "1.0",
                "unauditable_misdirection_count": 3,
                "avg_good_target_suspicion_gain": "-0.02",
                "detected_deception_count": 2,
                "peer_detection_opportunity_count": 4,
                "peer_detection_rate": "0.5",
                "avg_speaker_suspicion_gain": "-0.06",
                "listener_shift_sample_count": 7,
                "evidence_linked_count": 4,
                "villager_false_positive_rate": "0.3",
                "listener_susceptibility_by_seat": {
                    "2": {
                        "misdirection_samples": "4",
                        "avg_good_target_suspicion_gain": 0.0,
                        "misdirected_rate": "0.25",
                        "detection_samples": 2,
                        "avg_speaker_suspicion_gain": False,
                        "peer_detection_rate": 1.0,
                    },
                    "4": {
                        "misdirection_samples": 3,
                        "avg_good_target_suspicion_gain": "-0.1",
                        "misdirected_rate": "bad",
                        "detection_samples": False,
                        "avg_speaker_suspicion_gain": None,
                        "peer_detection_rate": float("nan"),
                    },
                },
                "audited_by_type": {
                    "vote_push": 0,
                    "omission": 3,
                    "bool_type": False,
                },
            },
        },
        {
            "winner": "village",
            "failed": 0,
            "game_ended_events": 1,
            "deception_audit": {
                "wolf_speech_count": None,
                "deception_success_rate": float("nan"),
                "misdirection_shift_coverage": False,
                "detected_deception_count": True,
                "peer_detection_rate": "bad",
                "listener_susceptibility_by_seat": {
                    "bool_seat": {
                        "misdirection_samples": False,
                        "peer_detection_rate": True,
                    },
                },
                "audited_by_type": "bad",
            },
        },
    ]

    stats.print_summary(results, jsonl_path=None, bootstrap_iters=20)

    output = capsys.readouterr().out

    assert "=== Deception audit ===" in output
    assert "wolf_speech_count: 5.00" in output
    assert "declared_deception_count: 2.00" in output
    assert "audited_deception_count: 3.00" in output
    assert "declared_vs_audited_agreement: 75.0%" in output
    assert "deception_success_rate: 50.0%" in output
    assert "misdirection_shift_coverage: 75.0%" in output
    assert "unauditable_misdirection_count: 2.00" in output
    assert "avg_good_target_suspicion_gain: 0.05" in output
    assert "detected_deception_count: 1.50" in output
    assert "peer_detection_opportunity_count: 3.00" in output
    assert "peer_detection_rate: 50.0%" in output
    assert "avg_speaker_suspicion_gain: 0.04" in output
    assert "listener_shift_sample_count: 5.00" in output
    assert "evidence_linked_count: 3.00" in output
    assert "villager_false_positive_rate: 20.0%" in output
    assert "audited_by_type:" in output
    assert "identity_claim: total=1.00 mean=0.50" in output
    assert "omission: total=3.00 mean=1.50" in output
    assert "vote_push: total=2.00 mean=1.00" in output
    assert "listener_susceptibility_by_seat:" in output
    assert "seat 2:" in output
    assert "misdirection_samples: total=6.00 mean=3.00" in output
    assert "avg_good_target_suspicion_gain: mean=0.10" in output
    assert "misdirected_rate: mean=37.5%" in output
    assert "detection_samples: total=4.00 mean=2.00" in output
    assert "avg_speaker_suspicion_gain: mean=0.10 95%CI[0.10,0.10] n=1" in output
    assert "peer_detection_rate: mean=75.0%" in output
    assert "seat 4:" in output
    assert "misdirection_samples: total=3.00 mean=3.00" in output
    assert "avg_good_target_suspicion_gain: mean=-0.10" in output
    assert "detection_samples: total=1.00 mean=1.00" in output
    assert "avg_speaker_suspicion_gain: mean=0.20" in output
    assert "peer_detection_rate: mean=100.0%" in output
    assert "bool_type" not in output
    assert "invalid_type" not in output
    assert "bool_seat" not in output
