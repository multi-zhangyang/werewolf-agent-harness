"""Truth-derived, recomputable strategy aggregation tests."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from src.harness.results import (
    LEGACY_RUN_SUMMARY_SCHEMA_VERSION,
    RUN_SUMMARY_SCHEMA_VERSION,
    RunSummaryRow,
    run_summary_from_result,
)
from src.harness.runner import HarnessRunResult
from src.harness.summary import summarize_runs
from src.harness.transcript import Transcript


def _result(
    run_id: str,
    *,
    status: str = "completed",
    policy: str,
    belief_count: int,
    belief_brier: float,
    decision_successes: int,
    decision_failures: int,
    false_role_claims: int,
    false_seer_results: int,
    council_participated: bool,
    agreement: bool,
    contradictions: int = 0,
    wolf_vote_count: int = 1,
    wolf_target_count: int = 1,
    include_strategy: bool = True,
    metadata: dict[str, Any] | None = None,
) -> HarnessRunResult:
    transcript = Transcript(
        run_id=run_id,
        metadata={"caller_metadata": dict(metadata or {})},
    )
    for index in range(decision_successes):
        transcript.append("decision", {
            "type": "decision_consumed",
            "request_id": f"{run_id}-success-{index}",
            "seat": 1,
            "role": "werewolf",
        })
    if council_participated:
        transcript.append("event", {
            "type": "wolf_council_message",
            "visibility": "private",
            "recipients": [f"{run_id}-seat-1"],
            "payload": {"speaker_seat": 1, "target_seat": 2},
        })
    analysis: dict[str, Any] = {
        "turn_policy": policy,
        "decision_count": decision_successes,
        "seats": [{"seat": 1, "name": "A", "role": "werewolf", "team": "werewolves"}],
        "decision_failure_metrics": {
            "failure_count": decision_failures,
            "by_seat": {"1": decision_failures},
            "records": [{"seat": 1} for _ in range(decision_failures)],
        },
        # These model-authored-looking values must never be aggregation inputs.
        "quality": {"game_quality": 1.0},
        "agent_summaries": [{"seat": 1, "self_reported_brier": 0.0}],
    }
    if include_strategy:
        analysis["agent_strategy_metrics"] = {
            "schema_version": "werewolf.agent-strategy-metrics.v1",
            "private_state_seat_count": 1,
            "belief_observation_count": belief_count,
            "belief_brier": belief_brier,
            "structured_claim_count": false_role_claims + false_seer_results,
            "false_role_claim_count": false_role_claims,
            "false_seer_result_count": false_seer_results,
            "seer_result_contradiction_count": contradictions,
            "wolf_council_message_count": int(council_participated),
            "wolf_final_vote_count": wolf_vote_count,
            "wolf_final_vote_target_count": wolf_target_count,
            "wolf_final_vote_agreement": agreement,
            "seats": [{
                "seat": 1,
                "belief_count": belief_count,
                "belief_brier": belief_brier,
                "structured_claim_count": false_role_claims + false_seer_results,
                "false_role_claim_count": false_role_claims,
                "false_seer_result_count": false_seer_results,
                "seer_result_contradiction_count": contradictions,
            }],
        }
    transcript.append("event", {"type": "analysis", "analysis": analysis})
    exported = transcript.export()
    return HarnessRunResult(
        run_id=run_id,
        status=status,
        winner="werewolves",
        days=1,
        elapsed_seconds=1.0,
        run_spec_hash="a" * 64,
        role_seed=1,
        actor_seed=2,
        orchestrator_seed=3,
        run_spec={
            "turn_policy": policy,
            "player_names": ["A"],
            "role_deck": ["werewolf"],
            "metadata": dict(metadata or {}),
        },
        event_count=sum(entry["kind"] == "event" for entry in exported["entries"]),
        decision_trace_count=sum(entry["kind"] == "decision" for entry in exported["entries"]),
        transcript_digest=exported["stable_digest"],
        transcript=exported,
        analysis=analysis,
    )


def test_strategy_summary_uses_weighted_brier_and_explicit_rate_denominators() -> None:
    first = run_summary_from_result(_result(
        "strategy-a",
        policy="fixed_round_robin",
        belief_count=1,
        belief_brier=0.9,
        decision_successes=2,
        decision_failures=1,
        false_role_claims=1,
        false_seer_results=0,
        council_participated=True,
        agreement=True,
        contradictions=1,
        wolf_vote_count=2,
        wolf_target_count=1,
    ))
    second = run_summary_from_result(_result(
        "strategy-b",
        policy="bid_reply",
        belief_count=3,
        belief_brier=0.1,
        decision_successes=1,
        decision_failures=0,
        false_role_claims=0,
        false_seer_results=2,
        council_participated=False,
        agreement=False,
        contradictions=2,
        wolf_vote_count=4,
        wolf_target_count=2,
    ))

    assert first.schema_version == RUN_SUMMARY_SCHEMA_VERSION
    assert first.strategy_metrics is not None
    assert first.strategy_metrics.seats[0].decision_success_count == 2
    assert first.strategy_metrics.seats[0].decision_failure_count == 1

    evaluation = summarize_runs([first, second]).strategy_evaluation
    assert evaluation is not None
    overall = evaluation.overall
    assert overall.run_count == 2
    assert overall.decision_success_count == 3
    assert overall.decision_failure_count == 1
    assert overall.decision_attempt_count == 4
    assert overall.decision_failure_rate == 0.25
    assert overall.belief_observation_count == 4
    assert overall.belief_brier_sum == pytest.approx(1.2)
    assert overall.belief_brier == pytest.approx(0.3)
    assert overall.false_role_claim_count == 1
    assert overall.false_role_claim_rate == pytest.approx(1 / 3)
    assert overall.false_seer_result_count == 2
    assert overall.seer_result_contradiction_count == 3
    assert overall.wolf_council_eligible_seat_count == 2
    assert overall.wolf_council_participant_count == 1
    assert overall.wolf_council_coverage == 0.5
    assert overall.wolf_final_vote_count == 6
    assert overall.wolf_final_vote_target_count == 3
    assert overall.wolf_final_vote_target_diversity == 0.5
    assert overall.wolf_vote_agreement_opportunity_count == 2
    assert overall.wolf_vote_agreement_count == 1
    assert overall.wolf_vote_agreement_rate == 0.5

    assert set(evaluation.by_turn_policy) == {"bid_reply", "fixed_round_robin"}
    assert evaluation.by_turn_policy["fixed_round_robin"].decision_failure_rate == pytest.approx(1 / 3)
    assert evaluation.by_role["werewolf"].belief_brier == pytest.approx(0.3)
    assert evaluation.by_seat["1"].wolf_council_coverage == 0.5


def test_strategy_derivation_accepts_only_bounded_legacy_brier_rounding() -> None:
    result = _result(
        "strategy-rounded-brier",
        policy="fixed_round_robin",
        belief_count=0,
        belief_brier=0.0,
        decision_successes=0,
        decision_failures=0,
        false_role_claims=0,
        false_seer_results=0,
        council_participated=False,
        agreement=False,
        wolf_vote_count=0,
        wolf_target_count=0,
    )
    analysis = deepcopy(result.analysis)
    assert analysis is not None
    analysis["seats"] = [
        {"seat": 1, "name": "A", "role": "werewolf", "team": "werewolves"},
        {"seat": 2, "name": "B", "role": "villager", "team": "village"},
    ]
    analysis["agent_strategy_metrics"] = {
        "schema_version": "werewolf.agent-strategy-metrics.v1",
        "private_state_seat_count": 2,
        "belief_observation_count": 6,
        # Runtime v1 rounded every mean independently to six decimals.
        "belief_brier": 0.333333,
        "structured_claim_count": 0,
        "false_role_claim_count": 0,
        "false_seer_result_count": 0,
        "seer_result_contradiction_count": 0,
        "wolf_council_message_count": 0,
        "wolf_final_vote_count": 0,
        "wolf_final_vote_target_count": 0,
        "wolf_final_vote_agreement": False,
        "seats": [
            {
                "seat": 1,
                "belief_count": 3,
                "belief_brier": 0.333333,
                "structured_claim_count": 0,
                "false_role_claim_count": 0,
                "false_seer_result_count": 0,
                "seer_result_contradiction_count": 0,
            },
            {
                "seat": 2,
                "belief_count": 3,
                "belief_brier": 0.333334,
                "structured_claim_count": 0,
                "false_role_claim_count": 0,
                "false_seer_result_count": 0,
                "seer_result_contradiction_count": 0,
            },
        ],
    }

    def reseal(next_analysis: dict[str, Any]) -> HarnessRunResult:
        transcript = Transcript(run_id=result.run_id)
        transcript.append("event", {"type": "analysis", "analysis": next_analysis})
        exported = transcript.export()
        updated = result.model_copy(deep=True)
        updated.analysis = deepcopy(next_analysis)
        updated.transcript = exported
        updated.transcript_digest = exported["stable_digest"]
        updated.event_count = 1
        updated.decision_trace_count = 0
        updated.run_spec["player_names"] = ["A", "B"]
        updated.run_spec["role_deck"] = ["werewolf", "villager"]
        return updated

    row = run_summary_from_result(reseal(analysis))
    assert row.strategy_metrics is not None
    assert row.strategy_metrics.belief_observation_count == 6
    assert row.strategy_metrics.belief_brier_sum == pytest.approx(2.000001)

    tampered = deepcopy(analysis)
    tampered["agent_strategy_metrics"]["belief_brier"] = 0.30
    with pytest.raises(ValueError, match="belief_brier_sum disagrees"):
        run_summary_from_result(reseal(tampered))


def test_new_failed_row_is_v4_and_legacy_v3_remains_readable() -> None:
    legacy = run_summary_from_result(_result(
        "legacy",
        status="failed",
        policy="fixed_round_robin",
        belief_count=0,
        belief_brier=0.0,
        decision_successes=1,
        decision_failures=0,
        false_role_claims=0,
        false_seer_results=0,
        council_participated=False,
        agreement=False,
        include_strategy=False,
    ))
    payload = legacy.model_dump(exclude_none=True)
    payload["schema_version"] = LEGACY_RUN_SUMMARY_SCHEMA_VERSION
    payload.pop("audit_metrics")

    assert legacy.schema_version == RUN_SUMMARY_SCHEMA_VERSION
    assert payload["schema_version"] == LEGACY_RUN_SUMMARY_SCHEMA_VERSION
    assert "strategy_metrics" not in payload
    restored = RunSummaryRow.model_validate(payload)
    assert restored.schema_version == LEGACY_RUN_SUMMARY_SCHEMA_VERSION
    assert restored.strategy_metrics is None
    assert restored.audit_metrics is None
    assert summarize_runs([restored]).strategy_evaluation is None
    assert summarize_runs([restored]).operational_evaluation is None


def test_per_seat_failures_use_complete_counter_not_capped_records() -> None:
    result = _result(
        "many-failures",
        policy="fixed_round_robin",
        belief_count=1,
        belief_brier=0.25,
        decision_successes=1,
        decision_failures=81,
        false_role_claims=0,
        false_seer_results=0,
        council_participated=False,
        agreement=False,
    )
    row = run_summary_from_result(result)

    assert row.strategy_metrics is not None
    assert row.strategy_metrics.seats[0].decision_failure_count == 81


def test_operational_summary_uses_projection_audit_and_factual_cost_denominators() -> None:
    result = _result(
        "operational",
        policy="fixed_round_robin",
        belief_count=1,
        belief_brier=0.25,
        decision_successes=1,
        decision_failures=0,
        false_role_claims=0,
        false_seer_results=0,
        council_participated=False,
        agreement=False,
    )
    transcript = Transcript(run_id=result.run_id)
    transcript.append("decision", {
        "type": "decision_consumed",
        "request_id": "operational-success-0",
        "seat": 1,
        "role": "werewolf",
    })
    transcript.append("event", {
        "type": "speech",
        "day": 1,
        "seat": 1,
        "text": "public accusation",
        "accuses": [2],
        "reasoning": "must be detected as a public hidden marker",
    })
    transcript.append("event", {
        "type": "vote_cast",
        "day": 1,
        "seat": 3,
        "target_seat": 2,
    })
    transcript.append("event", {
        "type": "seer_result",
        "day": 1,
        "visibility": "private",
        "recipients": [],
        "payload": {"target_seat": 2, "result": "werewolf"},
    })
    transcript.append("event", {"type": "analysis", "analysis": result.analysis})
    exported = transcript.export()
    result.transcript = exported
    result.transcript_digest = exported["stable_digest"]
    result.event_count = 4
    result.decision_trace_count = 1
    result.router_stats_delta = {
        "calls": 4,
        "failures": 1,
        "structured_responses": 2,
        "response_parse_failures": 1,
        "lossy_parse_rejections": 1,
        "incomplete_responses": 1,
        "total_tokens_in": 40,
        "total_tokens_out": 20,
        "total_latency": 8.0,
    }

    row = run_summary_from_result(result)
    assert row.audit_metrics is not None
    assert row.audit_metrics.visibility_audit_error_count == 3
    assert row.audit_metrics.private_information_leak_count == 3
    assert row.audit_metrics.public_vote_count == 1
    assert row.audit_metrics.prior_public_accusation_aligned_vote_count == 1

    operational = summarize_runs([row]).operational_evaluation
    assert operational is not None
    overall = operational.overall
    assert overall.provider_failure_rate == 0.25
    assert overall.structured_response_count == 2
    assert overall.response_parse_failure_rate == 0.5
    assert overall.lossy_parse_rejection_rate == 0.5
    assert overall.incomplete_response_rate == 0.5
    assert overall.input_tokens == 40
    assert overall.output_tokens == 20
    assert overall.average_model_latency_seconds == 2.0
    assert overall.visibility_audited_run_count == 1
    assert overall.private_information_leak_run_rate == 1.0
    assert overall.public_vote_alignment_rate == 1.0


def test_strategy_evaluation_groups_explicit_persona_and_role_layout_controls() -> None:
    first = run_summary_from_result(_result(
        "controlled-a",
        policy="fixed_round_robin",
        belief_count=2,
        belief_brier=0.2,
        decision_successes=1,
        decision_failures=0,
        false_role_claims=1,
        false_seer_results=0,
        council_participated=True,
        agreement=True,
        metadata={
            "role_layout_id": "layout-a",
            "persona_mode": "fixed",
            "persona_assignments": [{"seat": 1, "profile_id": "patient_disguise"}],
        },
    ))
    second = run_summary_from_result(_result(
        "controlled-b",
        policy="bid_reply",
        belief_count=1,
        belief_brier=0.4,
        decision_successes=1,
        decision_failures=1,
        false_role_claims=0,
        false_seer_results=0,
        council_participated=False,
        agreement=False,
        metadata={
            "role_layout_id": "layout-b",
            "persona_mode": "counterbalanced",
            "persona_assignments": [{"seat": 1, "profile_id": "direct_confrontation"}],
        },
    ))

    summary = summarize_runs([first, second])
    evaluation = summary.strategy_evaluation

    assert evaluation is not None
    assert set(evaluation.by_persona) == {"direct_confrontation", "patient_disguise"}
    assert evaluation.by_persona["patient_disguise"].false_role_claim_count == 1
    assert evaluation.by_persona["direct_confrontation"].decision_failure_count == 1
    assert set(evaluation.by_role_layout) == {"layout-a", "layout-b"}
    assert evaluation.by_role_layout["layout-a"].run_count == 1
    assert summary.runs_by_persona_mode == {"counterbalanced": 1, "fixed": 1}
    assert summary.runs_by_role_layout == {"layout-a": 1, "layout-b": 1}


def test_strategy_rows_bind_cached_metrics_to_verified_transcript() -> None:
    row = run_summary_from_result(_result(
        "strategy-provenance",
        policy="fixed_round_robin",
        belief_count=1,
        belief_brier=0.25,
        decision_successes=1,
        decision_failures=0,
        false_role_claims=0,
        false_seer_results=0,
        council_participated=False,
        agreement=False,
    ))
    assert row.strategy_metrics is not None
    assert row.strategy_metrics.source_transcript_digest == row.transcript_digest
    assert row.strategy_metrics.transcript_provenance_verified is True

    detached = row.strategy_metrics.model_copy(update={
        "source_transcript_digest": "f" * 64,
    })
    tampered_row = row.model_copy(update={"strategy_metrics": detached})
    assert summarize_runs([tampered_row]).strategy_evaluation is None

    forged = row.model_dump(mode="json")
    forged_metrics = dict(forged["strategy_metrics"])
    forged_metrics["belief_brier_sum"] = 0.5
    forged_metrics["source_transcript_digest"] = forged["transcript_digest"]
    forged_metrics["transcript_provenance_verified"] = True
    forged["strategy_metrics"] = forged_metrics
    restored = RunSummaryRow.model_validate(forged)
    assert summarize_runs([restored]).strategy_evaluation is None


def test_strategy_derivation_rejects_tampered_transcript_sequence() -> None:
    result = _result(
        "strategy-tampered-transcript",
        policy="fixed_round_robin",
        belief_count=1,
        belief_brier=0.25,
        decision_successes=1,
        decision_failures=0,
        false_role_claims=0,
        false_seer_results=0,
        council_participated=False,
        agreement=False,
    )
    result.transcript["entries"][0]["seq"] = 2
    with pytest.raises(ValueError, match="sequence mismatch"):
        run_summary_from_result(result)


def test_strategy_derivation_rejects_outer_analysis_tampering() -> None:
    result = _result(
        "strategy-tampered-analysis",
        policy="fixed_round_robin",
        belief_count=1,
        belief_brier=0.25,
        decision_successes=1,
        decision_failures=0,
        false_role_claims=0,
        false_seer_results=0,
        council_participated=False,
        agreement=False,
    )
    assert result.analysis is not None
    result.analysis["agent_strategy_metrics"]["belief_brier"] = 0.99
    with pytest.raises(ValueError, match="does not match transcript"):
        run_summary_from_result(result)


def test_strategy_persona_grouping_rejects_duplicate_seat_assignments() -> None:
    row = run_summary_from_result(_result(
        "duplicate-persona",
        policy="fixed_round_robin",
        belief_count=1,
        belief_brier=0.25,
        decision_successes=1,
        decision_failures=0,
        false_role_claims=0,
        false_seer_results=0,
        council_participated=False,
        agreement=False,
        metadata={
            "persona_mode": "fixed",
            "persona_assignments": [
                {"seat": 1, "profile_id": "patient_disguise"},
                {"seat": 1, "profile_id": "direct_confrontation"},
            ],
        },
    ))
    evaluation = summarize_runs([row]).strategy_evaluation
    assert evaluation is not None
    assert evaluation.by_persona == {}
