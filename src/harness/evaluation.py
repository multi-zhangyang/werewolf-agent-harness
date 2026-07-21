"""Recomputable cross-run strategy evaluation for the Werewolf harness.

The runtime owns the truth-bearing ``agent_strategy_metrics`` analysis.  This
module only reshapes those facts into run rows and aggregates; it never asks a
model to grade itself and never infers quality from prose.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import isclose, isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .transcript import validated_final_analysis, validate_transcript_evidence
from .visibility import audit_transcript_visibility
from .evidence_trust import is_verified


STRATEGY_EVALUATION_SCHEMA_VERSION = "werewolf.harness.strategy-evaluation.v1"
RUN_STRATEGY_METRICS_SCHEMA_VERSION = "werewolf.harness.run-strategy-metrics.v1"
RUN_AUDIT_METRICS_SCHEMA_VERSION = "werewolf.harness.run-audit-metrics.v1"
OPERATIONAL_EVALUATION_SCHEMA_VERSION = "werewolf.harness.operational-evaluation.v1"
_KNOWN_ROLES = frozenset({
    "villager",
    "werewolf",
    "seer",
    "doctor",
    "witch",
    "guard",
    "hunter",
})


class RunStrategySeatMetrics(BaseModel):
    """Truth-derived strategy facts for one seat in one run."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    seat: int
    role: str = "unknown"
    decision_success_count: int = Field(default=0, ge=0)
    decision_failure_count: int = Field(default=0, ge=0)
    belief_observation_count: int = Field(default=0, ge=0)
    belief_brier_sum: float = Field(default=0.0, ge=0.0)
    structured_claim_count: int = Field(default=0, ge=0)
    false_role_claim_count: int = Field(default=0, ge=0)
    false_seer_result_count: int = Field(default=0, ge=0)
    seer_result_contradiction_count: int = Field(default=0, ge=0)
    wolf_council_eligible_count: int = Field(default=0, ge=0)
    wolf_council_participation_count: int = Field(default=0, ge=0)
    wolf_final_vote_count: int = Field(default=0, ge=0)
    wolf_final_vote_target_count: int = Field(default=0, ge=0)
    wolf_vote_agreement_opportunity_count: int = Field(default=0, ge=0)
    wolf_vote_agreement_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _consistent_denominators(self) -> "RunStrategySeatMetrics":
        _validate_strategy_counts(self)
        return self


class RunStrategyMetrics(BaseModel):
    """Normalized, credential-free strategy facts for one completed run."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[RUN_STRATEGY_METRICS_SCHEMA_VERSION] = (
        RUN_STRATEGY_METRICS_SCHEMA_VERSION
    )
    run_id: str
    source_transcript_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    transcript_provenance_verified: bool = False
    source_role_layout_id: str | None = None
    source_persona_assignment_id: str | None = None
    turn_policy: str = ""
    decision_success_count: int = Field(default=0, ge=0)
    decision_failure_count: int = Field(default=0, ge=0)
    belief_observation_count: int = Field(default=0, ge=0)
    belief_brier_sum: float = Field(default=0.0, ge=0.0)
    structured_claim_count: int = Field(default=0, ge=0)
    false_role_claim_count: int = Field(default=0, ge=0)
    false_seer_result_count: int = Field(default=0, ge=0)
    seer_result_contradiction_count: int = Field(default=0, ge=0)
    wolf_council_message_count: int = Field(default=0, ge=0)
    wolf_council_eligible_seat_count: int = Field(default=0, ge=0)
    wolf_council_participant_count: int = Field(default=0, ge=0)
    wolf_final_vote_count: int = Field(default=0, ge=0)
    wolf_final_vote_target_count: int = Field(default=0, ge=0)
    wolf_vote_agreement_opportunity_count: int = Field(default=0, ge=0)
    wolf_vote_agreement_count: int = Field(default=0, ge=0)
    seats: list[RunStrategySeatMetrics] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_denominators(self) -> "RunStrategyMetrics":
        _validate_strategy_counts(self)
        seat_ids = [seat.seat for seat in self.seats]
        if len(seat_ids) != len(set(seat_ids)):
            raise ValueError("strategy seat metrics must contain unique seats")
        if self.belief_brier_sum > self.belief_observation_count + 1e-9:
            raise ValueError("Brier sum exceeds its observation denominator")
        return self


class StrategyAggregate(BaseModel):
    """An aggregate with explicit denominators for every derived rate."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = STRATEGY_EVALUATION_SCHEMA_VERSION
    dimension: str
    key: str
    run_count: int = Field(default=0, ge=0)
    seat_run_count: int = Field(default=0, ge=0)
    decision_success_count: int = Field(default=0, ge=0)
    decision_failure_count: int = Field(default=0, ge=0)
    decision_attempt_count: int = Field(default=0, ge=0)
    decision_failure_rate: float | None = None
    belief_observation_count: int = Field(default=0, ge=0)
    belief_brier_sum: float = Field(default=0.0, ge=0.0)
    belief_brier: float | None = None
    structured_claim_count: int = Field(default=0, ge=0)
    false_role_claim_count: int = Field(default=0, ge=0)
    false_role_claim_rate: float | None = None
    false_seer_result_count: int = Field(default=0, ge=0)
    seer_result_contradiction_count: int = Field(default=0, ge=0)
    wolf_council_eligible_seat_count: int = Field(default=0, ge=0)
    wolf_council_participant_count: int = Field(default=0, ge=0)
    wolf_council_coverage: float | None = None
    wolf_final_vote_count: int = Field(default=0, ge=0)
    wolf_final_vote_target_count: int = Field(default=0, ge=0)
    wolf_final_vote_target_diversity: float | None = None
    wolf_vote_agreement_opportunity_count: int = Field(default=0, ge=0)
    wolf_vote_agreement_count: int = Field(default=0, ge=0)
    wolf_vote_agreement_rate: float | None = None


class ExperimentStrategyEvaluation(BaseModel):
    """Cross-run strategy evaluation grouped by controlled dimensions."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = STRATEGY_EVALUATION_SCHEMA_VERSION
    run_count: int = Field(default=0, ge=0)
    overall: StrategyAggregate
    by_turn_policy: dict[str, StrategyAggregate] = Field(default_factory=dict)
    by_role: dict[str, StrategyAggregate] = Field(default_factory=dict)
    by_seat: dict[str, StrategyAggregate] = Field(default_factory=dict)
    by_persona: dict[str, StrategyAggregate] = Field(default_factory=dict)
    by_role_layout: dict[str, StrategyAggregate] = Field(default_factory=dict)


class RunAuditMetrics(BaseModel):
    """Deterministic transcript audit facts attached to every new run row."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[RUN_AUDIT_METRICS_SCHEMA_VERSION] = (
        RUN_AUDIT_METRICS_SCHEMA_VERSION
    )
    visibility_audit_performed: bool = True
    visibility_audit_error_count: int = Field(default=0, ge=0)
    private_information_leak_count: int = Field(default=0, ge=0)
    public_vote_count: int = Field(default=0, ge=0)
    prior_public_accusation_aligned_vote_count: int = Field(default=0, ge=0)


class OperationalAggregate(BaseModel):
    """Provider, schema, cost, vote-alignment, and visibility audit totals."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = OPERATIONAL_EVALUATION_SCHEMA_VERSION
    dimension: str
    key: str
    run_count: int = Field(default=0, ge=0)
    provider_call_count: int = Field(default=0, ge=0)
    provider_failure_count: int = Field(default=0, ge=0)
    provider_failure_rate: float | None = None
    structured_response_count: int = Field(default=0, ge=0)
    response_parse_failure_count: int = Field(default=0, ge=0)
    response_parse_failure_rate: float | None = None
    lossy_parse_rejection_count: int = Field(default=0, ge=0)
    lossy_parse_rejection_rate: float | None = None
    incomplete_response_count: int = Field(default=0, ge=0)
    incomplete_response_rate: float | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    model_latency_seconds: float = Field(default=0.0, ge=0.0)
    average_model_latency_seconds: float | None = None
    visibility_audited_run_count: int = Field(default=0, ge=0)
    visibility_audit_error_count: int = Field(default=0, ge=0)
    private_information_leak_count: int = Field(default=0, ge=0)
    runs_with_private_information_leak: int = Field(default=0, ge=0)
    private_information_leak_run_rate: float | None = None
    public_vote_count: int = Field(default=0, ge=0)
    prior_public_accusation_aligned_vote_count: int = Field(default=0, ge=0)
    public_vote_alignment_rate: float | None = None


class ExperimentOperationalEvaluation(BaseModel):
    """Operational evaluation over all rows, including failed and legacy runs."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = OPERATIONAL_EVALUATION_SCHEMA_VERSION
    run_count: int = Field(default=0, ge=0)
    overall: OperationalAggregate
    by_turn_policy: dict[str, OperationalAggregate] = Field(default_factory=dict)


def strategy_metrics_from_run(run: Any) -> RunStrategyMetrics | None:
    """Normalize one ``HarnessRunResult`` (or mapping) into strategy facts.

    Missing analysis is expected for failed runs and returns ``None``. The
    returned object is placed on an attested ``RunSummaryRow`` by the result
    factory; JSONL-only cache rows are deliberately excluded by aggregation.
    """

    evidence = validate_transcript_evidence(run)
    analysis = _mapping(validated_final_analysis(
        evidence,
        _value(run, "analysis"),
    ))
    source = _mapping(analysis.get("agent_strategy_metrics"))
    if not source:
        return None
    run_id = str(_value(run, "run_id") or "")
    spec = _mapping(_value(run, "run_spec"))
    turn_policy = str(spec.get("turn_policy") or analysis.get("turn_policy") or "")
    truth_seats = _truth_seats(analysis.get("seats"))
    raw_truth_seats = analysis.get("seats")
    if evidence.enclosing_digest_verified and (
        not isinstance(raw_truth_seats, list)
        or not raw_truth_seats
        or len(raw_truth_seats) != len(truth_seats)
    ):
        raise ValueError(
            "strategy analysis contains duplicate or invalid seat-role truth"
        )
    failure_metrics = _mapping(analysis.get("decision_failure_metrics"))
    entries = list(evidence.entries)
    control_metadata = _mapping(evidence.metadata.get("caller_metadata"))
    failure_by_seat = _failure_counts_by_seat(failure_metrics)
    success_by_seat = _consumed_counts_by_seat(entries)
    council_speakers = _council_speakers(entries, truth_seats)
    wolf_vote_targets = _wolf_final_vote_targets_by_seat(entries)

    raw_seats = source.get("seats")
    seat_rows: list[RunStrategySeatMetrics] = []
    source_seats_seen: set[int] = set()
    source_seat_rows_valid = isinstance(raw_seats, list)
    if isinstance(raw_seats, list):
        for raw in raw_seats:
            item = _mapping(raw)
            seat = _as_int(item.get("seat"))
            if seat is None or seat not in truth_seats or seat in source_seats_seen:
                source_seat_rows_valid = False
                continue
            source_seats_seen.add(seat)
            role = truth_seats.get(seat, "unknown")
            is_wolf = role == "werewolf"
            agreement = source.get("wolf_final_vote_agreement")
            seat_vote_targets = wolf_vote_targets.get(seat, [])
            seat_rows.append(RunStrategySeatMetrics(
                seat=seat,
                role=role,
                decision_success_count=success_by_seat.get(seat, 0),
                decision_failure_count=failure_by_seat.get(seat, 0),
                belief_observation_count=_nonnegative_int(item.get("belief_count")),
                belief_brier_sum=_brier_sum_from_mapping(item),
                structured_claim_count=_nonnegative_int(item.get("structured_claim_count")),
                false_role_claim_count=_nonnegative_int(item.get("false_role_claim_count")),
                false_seer_result_count=_nonnegative_int(item.get("false_seer_result_count")),
                seer_result_contradiction_count=_nonnegative_int(
                    item.get("seer_result_contradiction_count")
                ),
                wolf_council_eligible_count=int(is_wolf),
                wolf_council_participation_count=int(seat in council_speakers),
                wolf_final_vote_count=len(seat_vote_targets),
                wolf_final_vote_target_count=len(set(seat_vote_targets)),
                wolf_vote_agreement_opportunity_count=int(is_wolf and isinstance(agreement, bool)),
                wolf_vote_agreement_count=int(is_wolf and agreement is True),
            ))

    source_brier = _as_float(source.get("belief_brier"))
    source_brier_sum = _as_float(source.get("belief_brier_sum"))
    source_observations = _nonnegative_int(source.get("belief_observation_count"))
    seat_brier_sum = sum(row.belief_brier_sum for row in seat_rows)
    # Seat-level sums are the canonical recomputable value.  The historical
    # aggregate only stored a six-decimal mean, so multiplying it by the
    # observation count can differ from the seat rows by legitimate rounding.
    brier_sum = (
        seat_brier_sum
        if isinstance(raw_seats, list) and source_seat_rows_valid
        else (
            max(0.0, source_brier_sum)
            if source_brier_sum is not None
            else _brier_sum(source_brier, source_observations)
        )
    )
    agreement = source.get("wolf_final_vote_agreement")
    agreement_opportunities = int(isinstance(agreement, bool))
    agreement_count = int(agreement is True)
    decision_success = _nonnegative_int(
        analysis.get("decision_count"),
        default=sum(row.decision_success_count for row in seat_rows),
    )
    decision_failure = _nonnegative_int(
        failure_metrics.get("failure_count"),
        default=sum(row.decision_failure_count for row in seat_rows),
    )
    wolf_seats = {row.seat for row in seat_rows if row.role == "werewolf"}
    participant_count = len(council_speakers & wolf_seats)
    if not council_speakers:
        # An older transcript may retain only the aggregate source metric. Do
        # not invent speakers; the missing participant evidence remains zero.
        participant_count = 0
    metrics = RunStrategyMetrics(
        run_id=run_id,
        source_transcript_digest=evidence.stable_digest,
        transcript_provenance_verified=evidence.enclosing_digest_verified,
        source_role_layout_id=_nonempty_text(control_metadata.get("role_layout_id")),
        source_persona_assignment_id=_nonempty_text(
            control_metadata.get("persona_assignment_id")
        ),
        turn_policy=turn_policy,
        decision_success_count=decision_success,
        decision_failure_count=decision_failure,
        belief_observation_count=source_observations,
        belief_brier_sum=brier_sum,
        structured_claim_count=_nonnegative_int(source.get("structured_claim_count")),
        false_role_claim_count=_nonnegative_int(source.get("false_role_claim_count")),
        false_seer_result_count=_nonnegative_int(source.get("false_seer_result_count")),
        seer_result_contradiction_count=_nonnegative_int(
            source.get("seer_result_contradiction_count")
        ),
        wolf_council_message_count=_nonnegative_int(source.get("wolf_council_message_count")),
        wolf_council_eligible_seat_count=len(wolf_seats),
        wolf_council_participant_count=participant_count,
        wolf_final_vote_count=_nonnegative_int(source.get("wolf_final_vote_count")),
        wolf_final_vote_target_count=_nonnegative_int(source.get("wolf_final_vote_target_count")),
        wolf_vote_agreement_opportunity_count=agreement_opportunities,
        wolf_vote_agreement_count=agreement_count,
        seats=sorted(seat_rows, key=lambda row: row.seat),
    )
    if evidence.enclosing_digest_verified:
        _verify_strategy_metric_consistency(
            metrics,
            analysis=analysis,
            source=source,
            source_seat_rows_valid=source_seat_rows_valid,
            truth_seats=truth_seats,
            success_by_seat=success_by_seat,
            failure_by_seat=failure_by_seat,
        )
    return metrics


def aggregate_strategy_metrics(
    rows: Iterable[Any],
) -> ExperimentStrategyEvaluation | None:
    """Aggregate normalized strategy rows without averaging pre-rounded rates."""

    normalized: list[tuple[Any, RunStrategyMetrics]] = []
    seen: dict[str, tuple[Any, RunStrategyMetrics]] = {}
    for row in rows:
        metric = _row_strategy_metrics(row)
        if metric is None:
            continue
        previous = seen.get(metric.run_id)
        if previous is not None:
            if _same_evidence_row(previous[0], row):
                continue
            raise ValueError(
                f"conflicting strategy evidence rows for run_id {metric.run_id}"
            )
        seen[metric.run_id] = (row, metric)
        normalized.append((row, metric))
    if not normalized:
        return None
    overall = _Accumulator("overall", "all")
    by_policy: dict[str, _Accumulator] = {}
    by_role: dict[str, _Accumulator] = {}
    by_seat: dict[str, _Accumulator] = {}
    by_persona: dict[str, _Accumulator] = {}
    by_role_layout: dict[str, _Accumulator] = {}
    for row, metric in normalized:
        overall.add_run(metric)
        policy = metric.turn_policy or "unknown"
        by_policy.setdefault(policy, _Accumulator("turn_policy", policy)).add_run(metric)
        metadata = _mapping(_value(row, "metadata"))
        layout_id = _trusted_control_id(
            metadata,
            field="role_layout_id",
            source=metric.source_role_layout_id,
        )
        if layout_id:
            by_role_layout.setdefault(
                layout_id,
                _Accumulator("role_layout", layout_id),
            ).add_run(metric)
        for seat in metric.seats:
            by_role.setdefault(seat.role, _Accumulator("role", seat.role)).add_seat(metric.run_id, seat)
            seat_key = str(seat.seat)
            by_seat.setdefault(seat_key, _Accumulator("seat", seat_key)).add_seat(metric.run_id, seat)
            persona = (
                _persona_profile_for_seat(metadata, seat.seat)
                if metric.source_persona_assignment_id is None
                or _trusted_control_id(
                    metadata,
                    field="persona_assignment_id",
                    source=metric.source_persona_assignment_id,
                ) is not None
                else None
            )
            if persona:
                by_persona.setdefault(
                    persona,
                    _Accumulator("persona", persona),
                ).add_seat(metric.run_id, seat)
    return ExperimentStrategyEvaluation(
        run_count=len(normalized),
        overall=overall.export(),
        by_turn_policy={key: value.export() for key, value in sorted(by_policy.items())},
        by_role={key: value.export() for key, value in sorted(by_role.items())},
        by_seat={key: value.export() for key, value in sorted(by_seat.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0])},
        by_persona={key: value.export() for key, value in sorted(by_persona.items())},
        by_role_layout={
            key: value.export() for key, value in sorted(by_role_layout.items())
        },
    )


def audit_metrics_from_run(run: Any) -> RunAuditMetrics:
    """Audit public transcript markers and factual vote alignment for one run."""

    evidence = validate_transcript_evidence(run)
    entries = [dict(entry) for entry in evidence.entries]
    issues = audit_transcript_visibility(entries)
    error_issues = [issue for issue in issues if issue.severity == "error"]
    leak_codes = {
        "admin_event_public",
        "private_event_without_private_visibility",
        "private_event_without_recipients",
        "private_visibility_without_recipients",
        "public_hidden_top_level_field",
        "public_private_context_field",
    }
    public_votes, aligned_votes = _public_vote_alignment(entries)
    return RunAuditMetrics(
        visibility_audit_error_count=len(error_issues),
        private_information_leak_count=sum(issue.code in leak_codes for issue in error_issues),
        public_vote_count=public_votes,
        prior_public_accusation_aligned_vote_count=aligned_votes,
    )


def aggregate_operational_metrics(
    rows: Iterable[Any],
) -> ExperimentOperationalEvaluation | None:
    """Aggregate provider/schema/cost and deterministic transcript audit facts."""

    normalized: list[Any] = []
    seen: dict[str, Any] = {}
    for row in rows:
        if not is_verified(row) or _value(row, "audit_metrics") is None:
            continue
        run_id = str(_value(row, "run_id") or "")
        previous = seen.get(run_id)
        if previous is not None:
            if _same_evidence_row(previous, row):
                continue
            raise ValueError(f"conflicting operational evidence rows for run_id {run_id}")
        seen[run_id] = row
        normalized.append(row)
    if not normalized:
        return None
    overall = _OperationalAccumulator("overall", "all")
    by_policy: dict[str, _OperationalAccumulator] = {}
    for row in normalized:
        overall.add(row)
        policy = str(_value(row, "turn_policy") or "unknown")
        by_policy.setdefault(policy, _OperationalAccumulator("turn_policy", policy)).add(row)
    return ExperimentOperationalEvaluation(
        run_count=len(normalized),
        overall=overall.export(),
        by_turn_policy={key: value.export() for key, value in sorted(by_policy.items())},
    )


@dataclass
class _OperationalAccumulator:
    dimension: str
    key: str
    run_count: int = 0
    provider_call_count: int = 0
    provider_failure_count: int = 0
    structured_response_count: int = 0
    response_parse_failure_count: int = 0
    lossy_parse_rejection_count: int = 0
    incomplete_response_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_latency_seconds: float = 0.0
    visibility_audited_run_count: int = 0
    visibility_audit_error_count: int = 0
    private_information_leak_count: int = 0
    runs_with_private_information_leak: int = 0
    public_vote_count: int = 0
    prior_public_accusation_aligned_vote_count: int = 0

    def add(self, row: Any) -> None:
        self.run_count += 1
        self.provider_call_count += _nonnegative_int(_value(row, "model_calls"))
        self.provider_failure_count += _nonnegative_int(_value(row, "model_failures"))
        self.structured_response_count += _nonnegative_int(
            _value(row, "structured_response_count")
        )
        self.response_parse_failure_count += _nonnegative_int(
            _value(row, "response_parse_failure_count")
        )
        self.lossy_parse_rejection_count += _nonnegative_int(
            _value(row, "lossy_parse_rejection_count")
        )
        self.incomplete_response_count += _nonnegative_int(
            _value(row, "incomplete_response_count")
        )
        self.input_tokens += _nonnegative_int(_value(row, "input_tokens"))
        self.output_tokens += _nonnegative_int(_value(row, "output_tokens"))
        self.model_latency_seconds += max(0.0, _as_float(_value(row, "model_latency_seconds")) or 0.0)
        raw_audit = _value(row, "audit_metrics")
        if raw_audit is None:
            return
        audit = raw_audit if isinstance(raw_audit, RunAuditMetrics) else RunAuditMetrics.model_validate(raw_audit)
        if audit.visibility_audit_performed:
            self.visibility_audited_run_count += 1
        self.visibility_audit_error_count += audit.visibility_audit_error_count
        self.private_information_leak_count += audit.private_information_leak_count
        self.runs_with_private_information_leak += int(audit.private_information_leak_count > 0)
        self.public_vote_count += audit.public_vote_count
        self.prior_public_accusation_aligned_vote_count += (
            audit.prior_public_accusation_aligned_vote_count
        )

    def export(self) -> OperationalAggregate:
        return OperationalAggregate(
            dimension=self.dimension,
            key=self.key,
            run_count=self.run_count,
            provider_call_count=self.provider_call_count,
            provider_failure_count=self.provider_failure_count,
            provider_failure_rate=_ratio(
                self.provider_failure_count,
                self.provider_call_count,
            ),
            structured_response_count=self.structured_response_count,
            response_parse_failure_count=self.response_parse_failure_count,
            response_parse_failure_rate=_ratio(
                self.response_parse_failure_count,
                self.structured_response_count,
            ),
            lossy_parse_rejection_count=self.lossy_parse_rejection_count,
            lossy_parse_rejection_rate=_ratio(
                self.lossy_parse_rejection_count,
                self.structured_response_count,
            ),
            incomplete_response_count=self.incomplete_response_count,
            incomplete_response_rate=_ratio(
                self.incomplete_response_count,
                self.structured_response_count,
            ),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            model_latency_seconds=round(self.model_latency_seconds, 12),
            average_model_latency_seconds=_ratio(
                self.model_latency_seconds,
                self.provider_call_count,
            ),
            visibility_audited_run_count=self.visibility_audited_run_count,
            visibility_audit_error_count=self.visibility_audit_error_count,
            private_information_leak_count=self.private_information_leak_count,
            runs_with_private_information_leak=self.runs_with_private_information_leak,
            private_information_leak_run_rate=_ratio(
                self.runs_with_private_information_leak,
                self.visibility_audited_run_count,
            ),
            public_vote_count=self.public_vote_count,
            prior_public_accusation_aligned_vote_count=(
                self.prior_public_accusation_aligned_vote_count
            ),
            public_vote_alignment_rate=_ratio(
                self.prior_public_accusation_aligned_vote_count,
                self.public_vote_count,
            ),
        )


@dataclass
class _Accumulator:
    dimension: str
    key: str
    run_ids: set[str] = field(default_factory=set)
    seat_run_ids: set[tuple[str, int]] = field(default_factory=set)
    run_count: int = 0
    seat_run_count: int = 0
    decision_success_count: int = 0
    decision_failure_count: int = 0
    belief_observation_count: int = 0
    belief_brier_sum: float = 0.0
    structured_claim_count: int = 0
    false_role_claim_count: int = 0
    false_seer_result_count: int = 0
    seer_result_contradiction_count: int = 0
    wolf_council_eligible_seat_count: int = 0
    wolf_council_participant_count: int = 0
    wolf_final_vote_count: int = 0
    wolf_final_vote_target_count: int = 0
    wolf_vote_agreement_opportunity_count: int = 0
    wolf_vote_agreement_count: int = 0

    def _mark_run(self, run_id: str) -> bool:
        normalized = str(run_id or f"anonymous-{self.run_count}")
        if normalized in self.run_ids:
            return False
        self.run_ids.add(normalized)
        self.run_count += 1
        return True

    def add_run(self, metric: RunStrategyMetrics) -> None:
        if not self._mark_run(metric.run_id):
            return
        self.seat_run_count += len(metric.seats)
        self.decision_success_count += metric.decision_success_count
        self.decision_failure_count += metric.decision_failure_count
        self.belief_observation_count += metric.belief_observation_count
        self.belief_brier_sum += metric.belief_brier_sum
        self.structured_claim_count += metric.structured_claim_count
        self.false_role_claim_count += metric.false_role_claim_count
        self.false_seer_result_count += metric.false_seer_result_count
        self.seer_result_contradiction_count += metric.seer_result_contradiction_count
        self.wolf_council_eligible_seat_count += metric.wolf_council_eligible_seat_count
        self.wolf_council_participant_count += metric.wolf_council_participant_count
        self.wolf_final_vote_count += metric.wolf_final_vote_count
        self.wolf_final_vote_target_count += metric.wolf_final_vote_target_count
        self.wolf_vote_agreement_opportunity_count += metric.wolf_vote_agreement_opportunity_count
        self.wolf_vote_agreement_count += metric.wolf_vote_agreement_count

    def add_seat(self, run_id: str, seat: RunStrategySeatMetrics) -> None:
        seat_run_id = (str(run_id), seat.seat)
        if seat_run_id in self.seat_run_ids:
            return
        self.seat_run_ids.add(seat_run_id)
        self._mark_run(run_id)
        self.seat_run_count += 1
        self.decision_success_count += seat.decision_success_count
        self.decision_failure_count += seat.decision_failure_count
        self.belief_observation_count += seat.belief_observation_count
        self.belief_brier_sum += seat.belief_brier_sum
        self.structured_claim_count += seat.structured_claim_count
        self.false_role_claim_count += seat.false_role_claim_count
        self.false_seer_result_count += seat.false_seer_result_count
        self.seer_result_contradiction_count += seat.seer_result_contradiction_count
        self.wolf_council_eligible_seat_count += seat.wolf_council_eligible_count
        self.wolf_council_participant_count += seat.wolf_council_participation_count
        self.wolf_final_vote_count += seat.wolf_final_vote_count
        self.wolf_final_vote_target_count += seat.wolf_final_vote_target_count
        self.wolf_vote_agreement_opportunity_count += seat.wolf_vote_agreement_opportunity_count
        self.wolf_vote_agreement_count += seat.wolf_vote_agreement_count

    def export(self) -> StrategyAggregate:
        attempts = self.decision_success_count + self.decision_failure_count
        return StrategyAggregate(
            dimension=self.dimension,
            key=self.key,
            run_count=self.run_count,
            seat_run_count=self.seat_run_count,
            decision_success_count=self.decision_success_count,
            decision_failure_count=self.decision_failure_count,
            decision_attempt_count=attempts,
            decision_failure_rate=_ratio(self.decision_failure_count, attempts),
            belief_observation_count=self.belief_observation_count,
            belief_brier_sum=round(self.belief_brier_sum, 12),
            belief_brier=_ratio(self.belief_brier_sum, self.belief_observation_count),
            structured_claim_count=self.structured_claim_count,
            false_role_claim_count=self.false_role_claim_count,
            false_role_claim_rate=_ratio(
                self.false_role_claim_count,
                self.structured_claim_count,
            ),
            false_seer_result_count=self.false_seer_result_count,
            seer_result_contradiction_count=self.seer_result_contradiction_count,
            wolf_council_eligible_seat_count=self.wolf_council_eligible_seat_count,
            wolf_council_participant_count=self.wolf_council_participant_count,
            wolf_council_coverage=_ratio(
                self.wolf_council_participant_count,
                self.wolf_council_eligible_seat_count,
            ),
            wolf_final_vote_count=self.wolf_final_vote_count,
            wolf_final_vote_target_count=self.wolf_final_vote_target_count,
            wolf_final_vote_target_diversity=_ratio(
                self.wolf_final_vote_target_count,
                self.wolf_final_vote_count,
            ),
            wolf_vote_agreement_opportunity_count=self.wolf_vote_agreement_opportunity_count,
            wolf_vote_agreement_count=self.wolf_vote_agreement_count,
            wolf_vote_agreement_rate=_ratio(
                self.wolf_vote_agreement_count,
                self.wolf_vote_agreement_opportunity_count,
            ),
        )


def _row_strategy_metrics(row: Any) -> RunStrategyMetrics | None:
    value = _value(row, "strategy_metrics")
    if value is None:
        return None
    metrics = (
        value
        if isinstance(value, RunStrategyMetrics)
        else RunStrategyMetrics.model_validate(value)
    )
    if not _trusted_metric_provenance(row, metrics):
        return None
    return metrics


def _trusted_metric_provenance(row: Any, metrics: RunStrategyMetrics) -> bool:
    if not is_verified(row):
        return False
    row_digest = str(_value(row, "transcript_digest") or "").strip()
    return (
        bool(row_digest)
        and metrics.run_id == str(_value(row, "run_id") or "")
        and metrics.transcript_provenance_verified
        and metrics.source_transcript_digest == row_digest
    )


def _same_evidence_row(first: Any, second: Any) -> bool:
    if first is second:
        return True
    first_dump = getattr(first, "model_dump", None)
    second_dump = getattr(second, "model_dump", None)
    if not callable(first_dump) or not callable(second_dump):
        return False
    return first_dump(mode="json") == second_dump(mode="json")


def _verify_strategy_metric_consistency(
    metrics: RunStrategyMetrics,
    *,
    analysis: Mapping[str, Any],
    source: Mapping[str, Any],
    source_seat_rows_valid: bool,
    truth_seats: Mapping[int, str],
    success_by_seat: Mapping[int, int],
    failure_by_seat: Mapping[int, int],
) -> None:
    if not source_seat_rows_valid:
        raise ValueError("strategy analysis contains invalid or duplicate seat metrics")
    expected_seats = set(truth_seats)
    actual_seats = {seat.seat for seat in metrics.seats}
    if actual_seats != expected_seats:
        raise ValueError(
            "strategy analysis seat metrics do not cover the environment truth seats"
        )
    scalar_matches = {
        "decision_success_count": (
            metrics.decision_success_count,
            sum(success_by_seat.values()),
        ),
        "decision_failure_count": (
            metrics.decision_failure_count,
            sum(failure_by_seat.values()),
        ),
        "belief_observation_count": (
            metrics.belief_observation_count,
            sum(seat.belief_observation_count for seat in metrics.seats),
        ),
        "belief_brier_sum": (
            round(metrics.belief_brier_sum, 12),
            round(sum(seat.belief_brier_sum for seat in metrics.seats), 12),
        ),
        "structured_claim_count": (
            metrics.structured_claim_count,
            sum(seat.structured_claim_count for seat in metrics.seats),
        ),
        "false_role_claim_count": (
            metrics.false_role_claim_count,
            sum(seat.false_role_claim_count for seat in metrics.seats),
        ),
        "false_seer_result_count": (
            metrics.false_seer_result_count,
            sum(seat.false_seer_result_count for seat in metrics.seats),
        ),
        "seer_result_contradiction_count": (
            metrics.seer_result_contradiction_count,
            sum(seat.seer_result_contradiction_count for seat in metrics.seats),
        ),
    }
    mismatched = [
        field
        for field, (observed, expected) in scalar_matches.items()
        if observed != expected
    ]
    if mismatched:
        raise ValueError(
            "strategy analysis aggregate disagrees with seat/transcript evidence: "
            + ",".join(sorted(mismatched))
        )
    source_brier_sum = _as_float(source.get("belief_brier_sum"))
    expected_brier_sum = sum(seat.belief_brier_sum for seat in metrics.seats)
    if source_brier_sum is not None:
        brier_tolerance = 1e-9
    else:
        # v1 artifacts expose only a six-decimal mean. Bound the compatibility
        # tolerance by its maximum per-observation rounding error.
        brier_tolerance = max(
            1e-9,
            metrics.belief_observation_count * 1e-6,
        )
        source_brier_sum = _brier_sum(
            source.get("belief_brier"),
            source.get("belief_observation_count"),
        )
    if not isclose(
        float(source_brier_sum or 0.0),
        float(expected_brier_sum),
        rel_tol=0.0,
        abs_tol=brier_tolerance,
    ):
        raise ValueError(
            "strategy analysis belief_brier_sum disagrees with seat/transcript evidence"
        )
    source_schema = str(source.get("schema_version") or "")
    if source_schema != "werewolf.agent-strategy-metrics.v1":
        raise ValueError("strategy analysis schema_version is unsupported")
    analysis_decisions = _nonnegative_int(analysis.get("decision_count"))
    if analysis_decisions != metrics.decision_success_count:
        raise ValueError(
            "strategy analysis decision_count disagrees with consumed transcript rows"
        )


def _persona_profile_for_seat(metadata: Mapping[str, Any], seat: int) -> str | None:
    assignments = metadata.get("persona_assignments")
    if not isinstance(assignments, list):
        return None
    profiles: dict[int, str] = {}
    for raw in assignments:
        item = _mapping(raw)
        assigned_seat = _as_int(item.get("seat"))
        profile = str(item.get("profile_id") or "").strip()
        if (
            assigned_seat is None
            or assigned_seat < 1
            or not profile
            or assigned_seat in profiles
        ):
            return None
        profiles[assigned_seat] = profile
    return profiles.get(seat)


def _nonempty_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _trusted_control_id(
    metadata: Mapping[str, Any],
    *,
    field: str,
    source: str | None,
) -> str | None:
    row_value = _nonempty_text(metadata.get(field))
    if source is None:
        return row_value
    return row_value if row_value == source else None


def _validate_strategy_counts(
    metrics: RunStrategyMetrics | RunStrategySeatMetrics,
) -> None:
    if metrics.false_role_claim_count > metrics.structured_claim_count:
        raise ValueError("false role claims exceed structured claims")
    if metrics.false_seer_result_count > metrics.structured_claim_count:
        raise ValueError("false seer results exceed structured claims")
    eligible = getattr(
        metrics,
        "wolf_council_eligible_seat_count",
        getattr(metrics, "wolf_council_eligible_count", 0),
    )
    participants = getattr(
        metrics,
        "wolf_council_participant_count",
        getattr(metrics, "wolf_council_participation_count", 0),
    )
    if participants > eligible:
        raise ValueError("wolf council participants exceed eligible seats")
    if metrics.wolf_final_vote_target_count > metrics.wolf_final_vote_count:
        raise ValueError("distinct wolf vote targets exceed wolf votes")
    if metrics.wolf_vote_agreement_count > metrics.wolf_vote_agreement_opportunity_count:
        raise ValueError("wolf vote agreements exceed agreement opportunities")
    if not isfinite(metrics.belief_brier_sum):
        raise ValueError("Brier sum must be finite")
    if metrics.belief_brier_sum > metrics.belief_observation_count + 1e-9:
        raise ValueError("Brier sum exceeds its observation denominator")


def _truth_seats(value: Any) -> dict[int, str]:
    result: dict[int, str] = {}
    if not isinstance(value, list):
        return result
    for raw in value:
        item = _mapping(raw)
        seat = _as_int(item.get("seat"))
        role = str(item.get("role") or "").strip().lower()
        if seat is not None and seat > 0 and role in _KNOWN_ROLES:
            result[seat] = role
    return result


def _transcript_entries(run: Any) -> list[Mapping[str, Any]]:
    transcript = _mapping(_value(run, "transcript"))
    entries = transcript.get("entries")
    return [item for item in (_mapping(raw) for raw in entries or []) if item]


def _public_vote_alignment(entries: Iterable[Mapping[str, Any]]) -> tuple[int, int]:
    accused_by_day: dict[int, set[int]] = {}
    vote_count = 0
    aligned_count = 0
    for entry in entries:
        if entry.get("kind") != "event":
            continue
        payload = _mapping(entry.get("payload"))
        if payload.get("visibility") in {"private", "admin", "god"} or payload.get("recipients"):
            continue
        day = _as_int(payload.get("day"), _as_int(entry.get("day"), 0)) or 0
        event_type = payload.get("type")
        if event_type == "speech":
            raw_accuses = payload.get("accuses")
            if isinstance(raw_accuses, list):
                accused_by_day.setdefault(day, set()).update(
                    seat for raw in raw_accuses if (seat := _as_int(raw)) is not None
                )
        elif event_type == "vote_cast":
            target = _as_int(payload.get("target_seat"))
            if target is None:
                continue
            vote_count += 1
            aligned_count += int(target in accused_by_day.get(day, set()))
    return vote_count, aligned_count


def _consumed_counts_by_seat(entries: Iterable[Mapping[str, Any]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for entry in entries:
        if entry.get("kind") != "decision":
            continue
        payload = _mapping(entry.get("payload"))
        if payload.get("type") != "decision_consumed":
            continue
        seat = _as_int(payload.get("seat"), _as_int(entry.get("seat")))
        if seat is not None:
            counts[seat] += 1
    return counts


def _wolf_final_vote_targets_by_seat(
    entries: Iterable[Mapping[str, Any]],
) -> dict[int, list[int]]:
    targets: dict[int, list[int]] = {}
    for entry in entries:
        if entry.get("kind") != "decision":
            continue
        payload = _mapping(entry.get("payload"))
        if payload.get("type") != "decision_consumed" or payload.get("phase") != "wolf_final_vote":
            continue
        seat = _as_int(payload.get("seat"), _as_int(entry.get("seat")))
        target = _as_int(payload.get("target_seat"))
        if seat is not None and target is not None:
            targets.setdefault(seat, []).append(target)
    return targets


def _failure_counts_by_seat(metrics: Mapping[str, Any]) -> Counter[int]:
    counts: Counter[int] = Counter()
    # ``by_seat`` is the complete counter. ``records`` is intentionally capped
    # by the runtime and is only a compatibility fallback for older summaries.
    raw_by_seat = _mapping(metrics.get("by_seat"))
    for raw_seat, raw_count in raw_by_seat.items():
        seat = _as_int(raw_seat)
        if seat is not None:
            counts[seat] = _nonnegative_int(raw_count)
    if counts:
        return counts
    records = metrics.get("records")
    if isinstance(records, list):
        for raw in records:
            seat = _as_int(_mapping(raw).get("seat"))
            if seat is not None:
                counts[seat] += 1
    return counts


def _council_speakers(
    entries: Iterable[Mapping[str, Any]],
    truth_seats: Mapping[int, str],
) -> set[int]:
    speakers: set[int] = set()
    for entry in entries:
        if entry.get("kind") != "event":
            continue
        payload = _mapping(entry.get("payload"))
        if payload.get("type") != "wolf_council_message":
            continue
        event_payload = _mapping(payload.get("payload"))
        seat = _as_int(
            event_payload.get("speaker_seat"),
            _as_int(payload.get("speaker_seat")),
        )
        if seat is not None and truth_seats.get(seat) == "werewolf":
            speakers.add(seat)
    return speakers


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _nonnegative_int(value: Any, default: int = 0) -> int:
    parsed = _as_int(value)
    return max(0, parsed) if parsed is not None else max(0, default)


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _brier_sum(value: Any, observations: Any) -> float:
    score = _as_float(value)
    count = _nonnegative_int(observations)
    if score is None or not count:
        return 0.0
    return max(0.0, score) * count


def _brier_sum_from_mapping(value: Mapping[str, Any]) -> float:
    exact = _as_float(value.get("belief_brier_sum"))
    if exact is not None:
        return max(0.0, exact)
    return _brier_sum(value.get("belief_brier"), value.get("belief_count"))


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 12)
