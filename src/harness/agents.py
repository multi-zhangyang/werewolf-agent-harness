"""Protocol-level validation for agent decisions.

This module intentionally contains no scripted, replay, or social-simulation
agent implementations.  Test doubles belong in tests; production agents must
implement ``AgentProtocol`` directly.
"""
from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .agent_protocol import (
    AGENT_PROTOCOL_VERSION,
    ActionRequest,
    DecisionEnvelope,
    LegalAction,
    decision_action_value,
    decision_is_skip,
)


class DecisionValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: str = "error"
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class DecisionValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    issues: list[DecisionValidationIssue] = Field(default_factory=list)


def validate_decision_against_legal_actions(
    envelope: DecisionEnvelope,
    request: ActionRequest,
) -> DecisionValidationResult:
    """Check protocol identity, action membership, skip permission, and target scope."""
    issues: list[DecisionValidationIssue] = []
    if envelope.protocol_version != request.protocol_version:
        issues.append(_issue(
            "protocol_version_mismatch",
            "Decision response protocol_version does not match the request.",
            request_protocol_version=request.protocol_version,
            envelope_protocol_version=envelope.protocol_version,
        ))
    if request.protocol_version != AGENT_PROTOCOL_VERSION:
        issues.append(_issue(
            "unsupported_protocol_version",
            "ActionRequest protocol_version is not supported by the Werewolf contract.",
            protocol_version=request.protocol_version,
        ))
    if envelope.protocol_version != AGENT_PROTOCOL_VERSION:
        issues.append(_issue(
            "unsupported_protocol_version",
            "DecisionEnvelope protocol_version is not supported by the Werewolf contract.",
            protocol_version=envelope.protocol_version,
        ))
    if envelope.request_id != request.request_id:
        issues.append(_issue(
            "request_id_mismatch",
            "Decision response request_id does not match the request.",
            request_id=request.request_id,
            envelope_request_id=envelope.request_id,
        ))
    if envelope.seat != request.seat:
        issues.append(_issue(
            "seat_mismatch",
            "Decision response seat does not match the request seat.",
            request_seat=request.seat,
            envelope_seat=envelope.seat,
        ))
    if envelope.latency_seconds is not None and not math.isfinite(envelope.latency_seconds):
        issues.append(_issue(
            "latency_not_finite",
            "Decision latency_seconds must be finite.",
        ))
    try:
        json.dumps(
            envelope.decision.model_dump(exclude={"llm_call_trace"}),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError):
        issues.append(_issue(
            "non_json_payload",
            "Decision payload must contain only finite JSON values.",
        ))

    action = decision_action_value(envelope.decision)
    legal_by_action = {item.action: item for item in request.legal_actions}
    legal = legal_by_action.get(action)
    skipped = decision_is_skip(envelope.decision)
    if skipped:
        if not any(item.can_skip for item in request.legal_actions):
            issues.append(_issue(
                "skip_not_allowed",
                "Agent skipped when the request did not advertise skip as legal.",
                action=action,
            ))
        if not str(envelope.decision.skip_reason or "").strip():
            issues.append(_issue(
                "skip_reason_missing",
                "An explicit SKIP decision must include a factual reason label.",
            ))
        unexpected = {
            key: value
            for key, value in {
                "target_seat": envelope.decision.target_seat,
                "speech": envelope.decision.speech,
                "team_message": envelope.decision.team_message,
                "bid": envelope.decision.bid,
                "claim": envelope.decision.claim,
                "reply_to": envelope.decision.reply_to,
                "accuses": envelope.decision.accuses,
            }.items()
            if value is not None and value != []
        }
        if unexpected:
            issues.append(_issue(
                "skip_payload_not_empty",
                "SKIP cannot carry an executable target or public-output payload.",
                fields=sorted(unexpected),
            ))
    elif legal is None:
        issues.append(_issue(
            "action_not_legal",
            "Agent selected an action outside the advertised legal action space.",
            action=action,
            legal_actions=sorted(legal_by_action),
        ))
    elif envelope.decision.skip_reason is not None:
        issues.append(_issue(
            "skip_reason_on_non_skip",
            "A non-SKIP action cannot carry skip_reason.",
            action=action,
        ))

    if legal is not None and not skipped and legal.requires_target:
        target_seat = envelope.decision.target_seat
        if target_seat is None:
            issues.append(_issue(
                "target_seat_missing",
                "The selected action requires a target from the advertised set.",
                action=action,
                legal_target_seats=legal.target_seats,
                target_seat=envelope.decision.target_seat,
            ))
        elif target_seat not in legal.target_seats:
            issues.append(_issue(
                "target_seat_not_legal",
                "Agent selected a target outside the advertised target set.",
                action=action,
                target_seat=target_seat,
                legal_target_seats=legal.target_seats,
            ))
    elif legal is not None and not skipped and envelope.decision.target_seat is not None:
        issues.append(_issue(
            "target_not_expected",
            "The selected action did not advertise a target set.",
            action=action,
            target_seat=envelope.decision.target_seat,
        ))

    if not skipped and legal is not None:
        speech = str(envelope.decision.speech or "").strip()
        if action == "speak":
            if not speech:
                issues.append(_issue("speech_required", "SPEAK requires non-empty exact public text."))
            if envelope.decision.bid is None or envelope.decision.bid <= 0:
                issues.append(_issue("speak_bid_required", "SPEAK requires bid in the range 1..4."))
        elif action == "last_words":
            if not speech:
                issues.append(_issue("speech_required", "LAST_WORDS requires non-empty exact public text."))
            if envelope.decision.bid is not None:
                issues.append(_issue("bid_not_expected", "Only SPEAK may carry bid.", action=action))
        elif action == "wolf_council":
            if not str(envelope.decision.team_message or "").strip():
                issues.append(_issue(
                    "team_message_required",
                    "WOLF_COUNCIL requires non-empty exact team-private text.",
                ))
            if envelope.decision.speech is not None:
                issues.append(_issue(
                    "speech_not_expected",
                    "WOLF_COUNCIL uses team_message rather than public speech.",
                    action=action,
                ))
            if envelope.decision.bid is not None:
                issues.append(_issue("bid_not_expected", "Only SPEAK may carry bid.", action=action))
        else:
            if envelope.decision.speech is not None:
                issues.append(_issue("speech_not_expected", "This action cannot carry public speech.", action=action))
            if envelope.decision.bid is not None:
                issues.append(_issue("bid_not_expected", "Only SPEAK may carry bid.", action=action))
        if action != "wolf_council" and envelope.decision.team_message is not None:
            issues.append(_issue(
                "team_message_not_expected",
                "Only WOLF_COUNCIL may carry team-private text.",
                action=action,
            ))
        if action != "speak" and any(
            value is not None and value != []
            for value in (
                envelope.decision.claim,
                envelope.decision.reply_to,
                envelope.decision.accuses,
            )
        ):
            issues.append(_issue(
                "speech_metadata_not_expected",
                "claim/reply/accuses metadata is only valid on SPEAK.",
                action=action,
            ))

    if envelope.parse_status not in {"ok", "recovered", "not_applicable"}:
        issues.append(DecisionValidationIssue(
            code="parse_status_not_ok",
            severity="warning",
            message="Decision envelope reports a non-clean parse status.",
            evidence={"parse_status": envelope.parse_status},
        ))
    return DecisionValidationResult(
        valid=not any(issue.severity == "error" for issue in issues),
        issues=issues,
    )


def decision_target_seat(decision: Any) -> int | None:
    """Compatibility helper returning the seat-native target intent."""
    target = getattr(decision, "target_seat", None)
    return int(target) if isinstance(target, int) and not isinstance(target, bool) else None


def _issue(code: str, message: str, **evidence: Any) -> DecisionValidationIssue:
    return DecisionValidationIssue(code=code, message=message, evidence=evidence)
