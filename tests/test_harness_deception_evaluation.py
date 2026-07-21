"""Objective public-deception and belief-shift evaluation tests."""
from __future__ import annotations

import pytest

from src.harness.deception import (
    RunDeceptionMetrics,
    aggregate_deception_metrics,
    analyze_deception,
)
from src.harness.results import RunSummaryRow, run_summary_from_result
from src.harness.runner import HarnessRunResult
from src.harness.transcript import Transcript


def _checkpoint(
    transcript: Transcript,
    *,
    owner: int,
    beliefs: dict[int, float],
) -> None:
    revision = 1 + sum(
        entry.kind == "decision"
        and entry.payload.get("type") == "decision_consumed"
        and entry.payload.get("seat") == owner
        for entry in transcript.entries
    )
    transcript.append("decision", {
        "type": "decision_consumed",
        "seat": owner,
        "belief_state_after": {
            "schema_version": "werewolf.agent-belief-trace.v1",
            "owner_seat": owner,
            "revision": revision,
            "beliefs": {
                str(target): {
                    "wolf_probability": probability,
                    "confidence": 0.5,
                    "likely_role": None,
                    "updated_day": 1,
                    "updated_phase": "day",
                }
                for target, probability in beliefs.items()
            },
        },
    })


def _speech(
    transcript: Transcript,
    *,
    seat: int,
    claim: dict,
) -> None:
    transcript.append("event", {
        "type": "speech",
        "day": 1,
        "seat": seat,
        "text": "public claim",
        "claim": claim,
    })


def _run(transcript: Transcript, *, metadata: dict | None = None) -> dict:
    transcript.metadata = {"caller_metadata": dict(metadata or {})}
    analysis = {
        "seats": [
            {"seat": 1, "role": "werewolf", "team": "werewolves"},
            {"seat": 2, "role": "villager", "team": "village"},
            {"seat": 3, "role": "villager", "team": "village"},
            {"seat": 4, "role": "werewolf", "team": "werewolves"},
            {"seat": 5, "role": "seer", "team": "village"},
        ],
    }
    transcript.append("event", {"type": "analysis", "analysis": analysis})
    exported = transcript.export()
    return {
        "run_id": transcript.run_id,
        "transcript_digest": exported["stable_digest"],
        "transcript": exported,
        "analysis": analysis,
    }


def _summary_row(
    transcript: Transcript,
    *,
    policy: str,
    metadata: dict | None = None,
) -> RunSummaryRow:
    run = _run(transcript, metadata=metadata)
    result = HarnessRunResult(
        run_id=transcript.run_id,
        status="completed",
        winner="werewolves",
        days=1,
        run_spec_hash="a" * 64,
        role_seed=1,
        actor_seed=2,
        orchestrator_seed=3,
        run_spec={
            "turn_policy": policy,
            "player_names": ["A", "B", "C", "D", "E"],
            "role_deck": ["werewolf", "villager", "villager", "werewolf", "seer"],
            "metadata": dict(metadata or {}),
        },
        transcript_digest=run["transcript_digest"],
        transcript=run["transcript"],
        analysis=run["analysis"],
    )
    return run_summary_from_result(result)


def test_false_wolf_cover_claim_measures_opponent_belief_concealment() -> None:
    transcript = Transcript(run_id="deception-cover")
    _checkpoint(transcript, owner=2, beliefs={1: 0.7})
    _checkpoint(transcript, owner=4, beliefs={1: 1.0})
    _speech(transcript, seat=1, claim={"role": "villager"})
    _checkpoint(transcript, owner=2, beliefs={1: 0.2})
    _checkpoint(transcript, owner=4, beliefs={1: 1.0})

    result = analyze_deception(_run(transcript))

    assert result.metrics.causal is False
    assert result.metrics.false_public_role_claim_count == 1
    assert result.metrics.scoreable_signal_count == 1
    assert result.metrics.paired_signal_count == 1
    assert result.metrics.belief_shift_observation_count == 1
    assert result.metrics.beneficial_shift_count == 1
    assert result.metrics.mean_deception_direction_shift == pytest.approx(0.5)
    assert result.shifts[0].observer_seat == 2
    assert result.shifts[0].wolf_probability_delta == pytest.approx(-0.5)
    # The wolf teammate is excluded even though it has checkpoints.
    assert all(shift.observer_seat != 4 for shift in result.shifts)


def test_repeated_same_signal_in_one_observer_window_is_not_double_counted() -> None:
    transcript = Transcript(run_id="deception-repeat")
    _checkpoint(transcript, owner=2, beliefs={1: 0.6})
    _speech(transcript, seat=1, claim={"role": "villager"})
    _speech(transcript, seat=1, claim={"role": "villager"})
    _checkpoint(transcript, owner=2, beliefs={1: 0.4})

    result = analyze_deception(_run(transcript))

    assert result.metrics.false_public_role_claim_count == 2
    assert result.metrics.scoreable_signal_count == 2
    assert result.metrics.paired_signal_count == 1
    assert result.metrics.unpaired_signal_count == 1
    assert result.metrics.belief_shift_observation_count == 1
    assert result.metrics.mean_deception_direction_shift == pytest.approx(0.2)


def test_false_seer_result_scores_target_belief_shift_separately() -> None:
    transcript = Transcript(run_id="deception-seer")
    _checkpoint(transcript, owner=1, beliefs={3: 0.2, 5: 0.3})
    _speech(transcript, seat=5, claim={
        "role": "seer",
        "checked_seat": 3,
        "result": "wolf",
    })
    _checkpoint(transcript, owner=1, beliefs={3: 0.8, 5: 0.3})

    result = analyze_deception(_run(transcript))

    assert result.metrics.false_public_role_claim_count == 0
    assert result.metrics.false_public_seer_result_count == 1
    assert result.metrics.scoreable_signal_count == 1
    assert result.shifts[0].target_seat == 3
    assert result.shifts[0].deception_direction_shift == pytest.approx(0.6)


def test_village_role_cover_is_counted_but_not_mis_scored_as_wolf_alignment() -> None:
    transcript = Transcript(run_id="deception-village-cover")
    _checkpoint(transcript, owner=1, beliefs={5: 0.0})
    _speech(transcript, seat=5, claim={"role": "villager"})
    _checkpoint(transcript, owner=1, beliefs={5: 0.0})

    result = analyze_deception(_run(transcript))

    assert result.metrics.false_public_role_claim_count == 1
    assert result.metrics.unscoreable_false_role_claim_count == 1
    assert result.metrics.scoreable_signal_count == 0
    assert result.metrics.belief_shift_observation_count == 0
    assert result.metrics.mean_deception_direction_shift is None


def test_missing_pre_or_post_checkpoint_stays_explicitly_unpaired() -> None:
    transcript = Transcript(run_id="deception-unpaired")
    _speech(transcript, seat=1, claim={"role": "villager"})
    _checkpoint(transcript, owner=2, beliefs={1: 0.3})

    result = analyze_deception(_run(transcript))

    assert result.metrics.scoreable_signal_count == 1
    assert result.metrics.paired_signal_count == 0
    assert result.metrics.unpaired_signal_count == 1
    assert result.metrics.belief_shift_observation_count == 0


def test_cross_run_deception_aggregate_weights_observer_pairs_not_run_means() -> None:
    first_transcript = Transcript(run_id="aggregate-a")
    _checkpoint(first_transcript, owner=2, beliefs={1: 0.7})
    _speech(first_transcript, seat=1, claim={"role": "villager"})
    _checkpoint(first_transcript, owner=2, beliefs={1: 0.2})
    first = _summary_row(
        first_transcript,
        policy="fixed_round_robin",
        metadata={
            "role_layout_id": "layout-a",
            "persona_assignments": [
                {"seat": 1, "profile_id": "patient_disguise"},
            ],
        },
    )

    second_transcript = Transcript(run_id="aggregate-b")
    _checkpoint(second_transcript, owner=2, beliefs={1: 0.7})
    _checkpoint(second_transcript, owner=3, beliefs={1: 0.5})
    _checkpoint(second_transcript, owner=5, beliefs={1: 0.4})
    _speech(second_transcript, seat=1, claim={"role": "villager"})
    _speech(second_transcript, seat=1, claim={"role": "villager"})
    _checkpoint(second_transcript, owner=2, beliefs={1: 0.3})
    _checkpoint(second_transcript, owner=3, beliefs={1: 0.5})
    _checkpoint(second_transcript, owner=5, beliefs={1: 0.5})
    second = _summary_row(
        second_transcript,
        policy="bid_reply",
        metadata={
            "role_layout_id": "layout-b",
            "persona_assignments": [
                {"seat": 1, "profile_id": "direct_confrontation"},
            ],
        },
    )

    evaluation = aggregate_deception_metrics([first, second])

    assert evaluation is not None
    assert evaluation.causal is False
    assert evaluation.overall.run_count == 2
    assert evaluation.overall.false_public_role_claim_count == 3
    assert evaluation.overall.signal_pairing_rate == pytest.approx(2 / 3)
    assert evaluation.overall.belief_shift_observation_count == 4
    assert evaluation.overall.deception_direction_shift_sum == pytest.approx(0.8)
    assert evaluation.overall.mean_deception_direction_shift == pytest.approx(0.2)
    assert evaluation.by_role["werewolf"].mean_deception_direction_shift == pytest.approx(0.2)
    assert evaluation.by_seat["1"].harmful_shift_rate == pytest.approx(0.25)
    assert set(evaluation.by_turn_policy) == {"bid_reply", "fixed_round_robin"}
    assert set(evaluation.by_persona) == {"direct_confrontation", "patient_disguise"}
    assert evaluation.by_persona["patient_disguise"].mean_deception_direction_shift == 0.5
    assert set(evaluation.by_role_layout) == {"layout-a", "layout-b"}


def test_versioned_deception_rows_cannot_self_attest_with_public_fields() -> None:
    digest = "a" * 64
    detached = RunDeceptionMetrics(
        run_id="detached",
        false_public_role_claim_count=1,
        scoreable_signal_count=1,
        unpaired_signal_count=1,
    )
    row = {
        "schema_version": "werewolf.harness.run_summary.v4",
        "run_id": "detached",
        "transcript_digest": digest,
        "deception_metrics": detached,
    }
    assert aggregate_deception_metrics([row]) is None

    bound = detached.model_copy(update={
        "source_transcript_digest": digest,
        "transcript_provenance_verified": True,
    })
    forged = aggregate_deception_metrics([
        {**row, "deception_metrics": bound},
    ])
    assert forged is None

    mismatched = bound.model_copy(update={"source_transcript_digest": "b" * 64})
    assert aggregate_deception_metrics([
        {**row, "deception_metrics": mismatched},
    ]) is None

    transcript = Transcript(run_id="trusted-from-transcript")
    _speech(transcript, seat=1, claim={"role": "villager"})
    trusted_row = _summary_row(transcript, policy="fixed_round_robin")
    trusted = aggregate_deception_metrics([trusted_row])
    assert trusted is not None
    assert trusted.overall.false_public_role_claim_count == 1


def test_duplicate_bound_deception_rows_do_not_double_count() -> None:
    transcript = Transcript(run_id="duplicate")
    _speech(transcript, seat=1, claim={"role": "villager"})
    row = _summary_row(transcript, policy="fixed_round_robin")
    evaluation = aggregate_deception_metrics([row, row])
    assert evaluation is not None
    assert evaluation.overall.run_count == 1
    assert evaluation.overall.false_public_role_claim_count == 1


def test_deception_derivation_rejects_outer_role_truth_tampering() -> None:
    transcript = Transcript(run_id="deception-tampered-truth")
    _checkpoint(transcript, owner=2, beliefs={1: 0.6})
    _speech(transcript, seat=1, claim={"role": "villager"})
    _checkpoint(transcript, owner=2, beliefs={1: 0.4})
    run = _run(transcript)
    run["analysis"]["seats"][0]["role"] = "villager"

    with pytest.raises(ValueError, match="does not match transcript"):
        analyze_deception(run)


def test_deception_persona_grouping_rejects_duplicate_seat_assignments() -> None:
    transcript = Transcript(run_id="duplicate-persona-deception")
    _speech(transcript, seat=1, claim={"role": "villager"})
    row = _summary_row(
        transcript,
        policy="fixed_round_robin",
        metadata={
            "persona_assignments": [
                {"seat": 1, "profile_id": "patient_disguise"},
                {"seat": 1, "profile_id": "direct_confrontation"},
            ],
        },
    )
    evaluation = aggregate_deception_metrics([row])
    assert evaluation is not None
    assert evaluation.by_persona == {}
