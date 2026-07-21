"""Agent protocol, transcript integrity, redaction, and projection tests."""
from __future__ import annotations

import json
from copy import deepcopy

from pydantic import ValidationError
import pytest

from src.agent.schemas import AgentAction, AgentObservation, Decision
from src.harness.agent_protocol import ActionRequest, AgentProtocol, DecisionEnvelope, LegalAction
from src.harness.agents import validate_decision_against_legal_actions
from src.harness.transcript import (
    Transcript,
    TranscriptIntegrityError,
    validated_final_analysis,
    validate_transcript_evidence,
)
from src.harness.visibility import (
    audit_transcript_visibility,
    project_payload_for_audience,
    project_transcript_rows,
)


def _observation() -> dict:
    return AgentObservation(
        my_seat=1,
        my_role="villager",
        my_team="village",
        seats=[
            {"id": "p1", "seat": 1, "name": "A", "alive": True},
            {"id": "p2", "seat": 2, "name": "B", "alive": True},
        ],
        alive_seats=[1, 2],
        phase="voting",
        day=1,
        available_actions=["vote"],
        candidate_targets=[2],
        vote_targets=[2],
    ).model_dump()


def _request() -> ActionRequest:
    return ActionRequest(
        request_id="request-1",
        run_id="run-1",
        seat=1,
        phase="voting",
        day=1,
        action_kind="vote",
        observation=_observation(),
        legal_actions=[LegalAction(action="vote", target_seats=[2], can_skip=False)],
        deadline_monotonic=100.0,
    )


def test_validate_transcript_evidence_binds_result_and_rejects_row_tampering() -> None:
    transcript = Transcript(run_id="evidence-run")
    transcript.append("event", {"type": "phase_started", "day": 1})
    exported = transcript.export()
    result = {
        "run_id": transcript.run_id,
        "transcript_digest": exported["stable_digest"],
        "transcript": exported,
    }

    validated = validate_transcript_evidence(result)
    assert validated.stable_digest == exported["stable_digest"]
    assert validated.enclosing_digest_verified is True
    assert validated.entries[0]["seq"] == 1

    missing_outer_digest = dict(result)
    missing_outer_digest.pop("transcript_digest")
    legacy = validate_transcript_evidence(missing_outer_digest)
    assert legacy.enclosing_digest_verified is False

    tampered_seq = deepcopy(result)
    tampered_seq["transcript"]["entries"][0]["seq"] = 2
    with pytest.raises(TranscriptIntegrityError, match="sequence mismatch"):
        validate_transcript_evidence(tampered_seq)

    tampered_payload = deepcopy(result)
    tampered_payload["transcript"]["entries"][0]["payload"]["day"] = 2
    with pytest.raises(TranscriptIntegrityError, match="payload hash mismatch"):
        validate_transcript_evidence(tampered_payload)

    tampered_counts = deepcopy(result)
    tampered_counts["transcript"]["counts_by_kind"]["event"] = 2
    with pytest.raises(TranscriptIntegrityError, match="counts_by_kind"):
        validate_transcript_evidence(tampered_counts)


def test_validated_analysis_must_be_unique_final_and_match_outer_copy() -> None:
    transcript = Transcript(run_id="analysis-evidence")
    analysis = {"seats": [{"seat": 1, "role": "werewolf"}]}
    transcript.append("event", {"type": "analysis", "analysis": analysis})
    exported = transcript.export()
    evidence = validate_transcript_evidence({
        "run_id": transcript.run_id,
        "transcript_digest": exported["stable_digest"],
        "transcript": exported,
    })
    assert validated_final_analysis(evidence, analysis) == analysis
    with pytest.raises(TranscriptIntegrityError, match="does not match transcript"):
        validated_final_analysis(evidence, {"seats": []})

    nonfinal = Transcript(run_id="analysis-nonfinal")
    nonfinal.append("event", {"type": "analysis", "analysis": analysis})
    nonfinal.append("event", {"type": "phase_started", "day": 2})
    nonfinal_export = nonfinal.export()
    nonfinal_evidence = validate_transcript_evidence({
        "run_id": nonfinal.run_id,
        "transcript_digest": nonfinal_export["stable_digest"],
        "transcript": nonfinal_export,
    })
    with pytest.raises(TranscriptIntegrityError, match="final environment event"):
        validated_final_analysis(nonfinal_evidence, analysis)


def test_agent_protocol_is_one_request_one_envelope_contract():
    class Agent:
        seat = 1

        async def decide(self, request: ActionRequest) -> DecisionEnvelope:
            return DecisionEnvelope(
                request_id=request.request_id,
                seat=self.seat,
                decision=Decision(action=AgentAction.VOTE, target_seat=2),
            )

    assert isinstance(Agent(), AgentProtocol)


def test_action_request_rejects_unknown_fields_and_assignment():
    request = _request()
    with pytest.raises(ValidationError):
        ActionRequest(**(_request().model_dump() | {"fake": True}))
    with pytest.raises(ValidationError):
        request.day = 2  # type: ignore[misc]


def test_werewolf_protocol_rejects_blank_identity_unknown_version_and_nonfinite_values():
    payload = _request().model_dump()
    payload["request_id"] = "  "
    with pytest.raises(ValidationError):
        ActionRequest(**payload)

    with pytest.raises(ValidationError):
        DecisionEnvelope(
            request_id="request-1",
            seat=1,
            decision=Decision(action=AgentAction.VOTE, target_seat=2),
            latency_seconds=float("inf"),
        )

    valid = DecisionEnvelope(
        request_id="request-1",
        seat=1,
        decision=Decision(action=AgentAction.VOTE, target_seat=2),
    )
    request = _request().model_copy(update={"protocol_version": "unknown.v99"})
    envelope = valid.model_copy(update={"protocol_version": "unknown.v99"})
    result = validate_decision_against_legal_actions(envelope, request)
    assert not result.valid
    assert "unsupported_protocol_version" in {issue.code for issue in result.issues}

    with pytest.raises(ValidationError):
        Decision(
            action=AgentAction.SPEAK,
            speech="exact",
            bid=1,
            claim={"confidence": float("nan")},
        )


def test_action_request_rejects_ambiguous_or_impossible_legal_action_shapes():
    with pytest.raises(ValidationError, match="must be unique"):
        LegalAction(action="vote", target_seats=[2, 2])
    with pytest.raises(ValidationError, match="must allow skip"):
        LegalAction(action="save", target_required=True)
    duplicate_actions = _request().model_dump()
    duplicate_actions["legal_actions"] = [
        {"action": "vote", "target_seats": [2]},
        {"action": "vote", "target_seats": [2]},
    ]
    with pytest.raises(ValidationError, match="legal action names must be unique"):
        ActionRequest(**duplicate_actions)


def test_protocol_validation_checks_identity_action_skip_and_target():
    request = _request()
    valid = DecisionEnvelope(
        request_id=request.request_id,
        seat=1,
        decision=Decision(action=AgentAction.VOTE, target_seat=2),
    )
    assert validate_decision_against_legal_actions(valid, request).valid

    wrong_target = valid.model_copy(update={
        "decision": Decision(action=AgentAction.VOTE, target_seat=3)
    })
    result = validate_decision_against_legal_actions(wrong_target, request)
    assert not result.valid
    assert result.issues[0].code == "target_seat_not_legal"

    skipped = valid.model_copy(update={
        "decision": Decision(action=AgentAction.SKIP, skip_reason="agent_skip")
    })
    result = validate_decision_against_legal_actions(skipped, request)
    assert not result.valid
    assert result.issues[0].code == "skip_not_allowed"

    wrong_version = valid.model_copy(update={"protocol_version": "werewolf.harness.agent_protocol.v1"})
    result = validate_decision_against_legal_actions(wrong_version, request)
    assert not result.valid
    assert result.issues[0].code == "protocol_version_mismatch"


def test_protocol_target_requirement_distinguishes_no_targets_from_target_free_action():
    request = _request().model_copy(update={
        "phase": "night",
        "action_kind": "save",
        "legal_actions": [LegalAction(
            action="save",
            target_seats=[],
            target_required=True,
            can_skip=True,
        )],
    })
    missing_target = DecisionEnvelope(
        request_id=request.request_id,
        seat=1,
        decision=Decision(action=AgentAction.SAVE),
    )
    result = validate_decision_against_legal_actions(missing_target, request)
    assert not result.valid
    assert "target_seat_missing" in {issue.code for issue in result.issues}

    explicit_skip = missing_target.model_copy(update={
        "decision": Decision(action=AgentAction.SKIP, skip_reason="no_legal_target")
    })
    assert validate_decision_against_legal_actions(explicit_skip, request).valid

    target_free = request.model_copy(update={
        "phase": "day",
        "action_kind": "speak",
        "legal_actions": [LegalAction(action="speak", can_skip=True)],
    })
    unexpected_target = DecisionEnvelope(
        request_id=target_free.request_id,
        seat=1,
        decision=Decision(action=AgentAction.SPEAK, target_seat=2, speech="text", bid=1),
    )
    result = validate_decision_against_legal_actions(unexpected_target, target_free)
    assert not result.valid
    assert "target_not_expected" in {issue.code for issue in result.issues}


def test_protocol_validation_rejects_ambiguous_skip_payloads():
    request = _request().model_copy(update={
        "legal_actions": [LegalAction(action="vote", target_seats=[2], can_skip=True)]
    })

    missing_reason = DecisionEnvelope(
        request_id=request.request_id,
        seat=1,
        decision=Decision(action=AgentAction.SKIP),
    )
    result = validate_decision_against_legal_actions(missing_reason, request)
    assert not result.valid
    assert "skip_reason_missing" in {issue.code for issue in result.issues}

    payload_skip = missing_reason.model_copy(update={
        "decision": Decision(
            action=AgentAction.SKIP,
            skip_reason="agent_declined",
            target_seat=2,
            speech="must not be emitted",
        )
    })
    result = validate_decision_against_legal_actions(payload_skip, request)
    assert not result.valid
    assert "skip_payload_not_empty" in {issue.code for issue in result.issues}


def test_protocol_validation_requires_exact_speech_contract():
    request = _request().model_copy(update={
        "phase": "day",
        "action_kind": "speak",
        "legal_actions": [LegalAction(action="speak", can_skip=True)],
    })

    no_text = DecisionEnvelope(
        request_id=request.request_id,
        seat=1,
        decision=Decision(action=AgentAction.SPEAK, bid=2),
    )
    result = validate_decision_against_legal_actions(no_text, request)
    assert not result.valid
    assert "speech_required" in {issue.code for issue in result.issues}

    zero_bid = no_text.model_copy(update={
        "decision": Decision(action=AgentAction.SPEAK, speech="exact output", bid=0)
    })
    result = validate_decision_against_legal_actions(zero_bid, request)
    assert not result.valid
    assert "speak_bid_required" in {issue.code for issue in result.issues}


def test_protocol_validation_rejects_payload_fields_on_wrong_action():
    request = _request()
    envelope = DecisionEnvelope(
        request_id=request.request_id,
        seat=1,
        decision=Decision(
            action=AgentAction.VOTE,
            target_seat=2,
            speech="not a vote field",
            bid=1,
            claim={"role": "seer"},
        ),
    )

    result = validate_decision_against_legal_actions(envelope, request)
    codes = {issue.code for issue in result.issues}
    assert not result.valid
    assert {"speech_not_expected", "bid_not_expected", "speech_metadata_not_expected"} <= codes


def test_transcript_redacts_credentials_and_keeps_stable_content_digest():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0cmFjZSJ9.signaturevalue123456"
    another_jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJvdGhlciJ9.differentsignature123456"
    first = Transcript(run_id="run", metadata={"api_key": "sk-test-secret-value"})
    second = Transcript(run_id="run", metadata={"api_key": "different-secret"})
    first.append(
        "decision",
        {
            "kind": "agent_request",
            "request": {
                "request_id": "r1",
                "deadline_monotonic": 100.0,
                "private_context": {"note": "visible only to admin", "access_token": jwt},
            },
            "_ts": 1.0,
            "authorization": "Bearer token-value",
        },
        ts_monotonic=1.0,
    )
    second.append(
        "decision",
        {
            "kind": "agent_request",
            "request": {
                "request_id": "r1",
                "deadline_monotonic": 999.0,
                "private_context": {"note": "visible only to admin", "access_token": another_jwt},
            },
            "_ts": 9.0,
            "authorization": "Bearer another-token",
        },
        ts_monotonic=9.0,
    )

    first_export = first.export()
    serialized = json.dumps(first_export)
    assert "sk-test-secret-value" not in serialized
    assert "token-value" not in serialized
    assert jwt not in serialized
    assert "[redacted]" in serialized
    assert first.stable_digest() == second.stable_digest()
    assert first.stable_digest(include_timing=True) != second.stable_digest(include_timing=True)


def test_transcript_projection_is_read_only_and_audience_scoped():
    private_reasoning = "model-private-reasoning-sentinel"
    transcript = Transcript(run_id="run")
    transcript.append("event", {
        "type": "speech",
        "seat": 1,
        "text": "public",
        "reasoning": private_reasoning,
        "nested": {
            "thought": private_reasoning,
            "items": [{"private_reasoning": private_reasoning}],
        },
    })
    transcript.append("event", {
        "type": "seer_result",
        "seat": 1,
        "target_seat": 2,
        "visibility": "private",
        "recipients": ["p1"],
        "details": {"reasoning": private_reasoning},
    })
    transcript.append("decision", {
        "kind": "agent_response",
        "seat": 1,
        "private_context": {"hidden": True},
        "envelope": {
            "decision": {"reasoning": private_reasoning},
        },
    })
    rows = transcript.export()["entries"]

    public = project_transcript_rows(rows, audience="public")
    player = project_transcript_rows(rows, audience="player", seat=1, player_id="p1")
    god = project_transcript_rows(rows, audience="god")
    admin = project_transcript_rows(rows, audience="admin")

    assert [row["payload"]["type"] for row in public] == ["speech"]
    assert {row["payload"]["type"] for row in player} == {"speech", "seer_result"}
    assert {row["kind"] for row in god} == {"event"}
    assert {row["kind"] for row in admin} == {"event", "decision"}
    for projection in (public, player, god):
        assert private_reasoning not in json.dumps(projection, ensure_ascii=False)
    admin_events = [row for row in admin if row["kind"] == "event"]
    admin_decision = next(row for row in admin if row["kind"] == "decision")
    assert private_reasoning not in json.dumps(admin_events, ensure_ascii=False)
    assert admin_decision["payload"]["envelope"]["decision"]["reasoning"] == private_reasoning
    assert rows[-1]["payload"]["private_context"] == {"hidden": True}
    assert rows[-1]["payload"]["envelope"]["decision"]["reasoning"] == private_reasoning


def test_validator_failure_is_public_with_harness_attribution_and_sanitization():
    raw = {
        "type": "decision_validation_failed",
        "request_id": "request-1",
        "seat": 1,
        "phase": "voting",
        "action": "vote",
        "error_type": "DecisionValidatorError",
        "agent_kind": "llm",
        "reason": "private validator stack and model response",
        "private_context": {"secret": True},
    }

    public = project_payload_for_audience(raw, audience="public")

    assert public == {
        "type": "decision_validation_failed",
        "phase": "voting",
        "reason": "Harness 校验器失败,本请求未被执行;该故障不归因于 Agent。",
        "request_id": "request-1",
        "error_type": "DecisionValidatorError",
        "agent_kind": "llm",
        "seat": 1,
        "action": "vote",
    }
    assert "private validator stack" not in str(public)


def test_visibility_audit_reports_private_and_public_leaks():
    rows = [
        {"seq": 1, "kind": "event", "payload": {"type": "seer_result", "seat": 1}},
        {"seq": 2, "kind": "event", "payload": {
            "type": "speech",
            "text": "public",
            "reasoning": "must not be public",
        }},
    ]
    issues = audit_transcript_visibility(rows)
    codes = {issue.code for issue in issues}
    assert "private_event_without_private_visibility" in codes
    assert "private_event_without_recipients" in codes
    assert "public_hidden_top_level_field" in codes


def test_explicit_generic_public_events_project_without_weakening_private_guards():
    """Plugins can publish new event names, but cannot relabel core secrets."""
    rows = [
        {
            "seq": 1,
            "kind": "event",
            "payload": {
                "type": "council_proposal_submitted",
                "visibility": "public",
                "proposer": "council:1",
                "mission": 1,
            },
        },
        {
            "seq": 2,
            "kind": "event",
            "payload": {
                "type": "role_assigned",
                "visibility": "public",
                "role": "werewolf",
            },
        },
        {
            "seq": 3,
            "kind": "event",
            "payload": {
                "type": "analysis",
                "visibility": "public",
                "analysis": {"private": True},
            },
        },
        {
            "seq": 4,
            "kind": "event",
            "payload": {
                "type": "council_bad_public_payload",
                "visibility": "public",
                "private_context": {"hidden": True},
            },
        },
    ]

    public = project_transcript_rows(rows, audience="public")
    player = project_transcript_rows(rows, audience="player", player_id="p1")

    assert [row["payload"]["type"] for row in public] == [
        "council_proposal_submitted",
        "council_bad_public_payload",
    ]
    assert [row["payload"]["type"] for row in player] == [
        "council_proposal_submitted",
        "council_bad_public_payload",
    ]
    assert public[0]["payload"] == {
        "type": "council_proposal_submitted",
        "proposer": "council:1",
        "mission": 1,
    }
    assert "private_context" not in public[1]["payload"]

    issues = audit_transcript_visibility(rows)
    codes = {issue.code for issue in issues}
    assert "private_event_without_private_visibility" in codes
    assert "private_event_without_recipients" in codes
    assert "admin_event_public" in codes
    assert "public_private_context_field" in codes


def test_public_projection_recursively_strips_forged_nested_private_state() -> None:
    marker = "forged-public-private-state-sentinel"
    rows = [{
        "seq": 1,
        "kind": "event",
        "payload": {
            "type": "council_bad_public_payload",
            "visibility": "public",
            "payload": {
                "claim": {"role": "seer", "checked_seat": 2},
                "private_state": {"selected_plan": marker},
                "nested": {
                    "secret_notes": marker,
                    "non_claim_role": "werewolf",
                },
            },
        },
    }]

    projected = project_transcript_rows(rows, audience="public")

    assert len(projected) == 1
    visible_payload = projected[0]["payload"]
    assert visible_payload["payload"]["claim"]["role"] == "seer"
    assert "private_state" not in visible_payload["payload"]
    assert "secret_notes" not in visible_payload["payload"]["nested"]
    assert "non_claim_role" not in visible_payload["payload"]["nested"]
    assert marker not in json.dumps(projected, ensure_ascii=False)
    codes = {issue.code for issue in audit_transcript_visibility(rows)}
    assert "public_hidden_nested_field" in codes


def test_cipher_council_team_message_cannot_be_mislabeled_public() -> None:
    rows = [{
        "seq": 1,
        "kind": "event",
        "payload": {
            "type": "council_cipher_message",
            "visibility": "public",
            "actor_id": "council:2",
            "message": "private faction strategy",
        },
    }]

    assert project_transcript_rows(rows, audience="public") == []
    assert project_transcript_rows(rows, audience="player", player_id="council:2") == []
    codes = {issue.code for issue in audit_transcript_visibility(rows)}
    assert "private_event_without_private_visibility" in codes
    assert "private_event_without_recipients" in codes
