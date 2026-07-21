"""Objective, recomputable deception-to-belief-shift analysis.

The evaluator never asks a model whether its lie was convincing. It derives
false public claims from environment truth, then associates each claim with
the nearest recorded belief checkpoint before and after the claim for eligible
opponents. The association is observational: intervening public information
can contribute to the shift, so these metrics must not be reported as causal.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence_trust import is_verified
from .transcript import (
    TranscriptIntegrityError,
    validated_final_analysis,
    validate_transcript_evidence,
)


DECEPTION_ANALYSIS_SCHEMA_VERSION = "werewolf.harness.deception-analysis.v1"
RUN_DECEPTION_METRICS_SCHEMA_VERSION = "werewolf.harness.run-deception-metrics.v1"
DECEPTION_EVALUATION_SCHEMA_VERSION = "werewolf.harness.deception-evaluation.v1"
BELIEF_TRACE_SCHEMA_VERSION = "werewolf.agent-belief-trace.v1"
_KNOWN_ROLES = frozenset({
    "villager",
    "werewolf",
    "seer",
    "doctor",
    "witch",
    "guard",
    "hunter",
})


class DeceptionSignal(BaseModel):
    """One objectively false public alignment proposition."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    signal_id: str
    event_seq: int = Field(ge=1)
    day: int = Field(default=0, ge=0)
    speaker_seat: int = Field(ge=1)
    target_seat: int = Field(ge=1)
    kind: Literal["false_role_claim", "false_seer_result"]
    actual_alignment: Literal["wolf", "village"]
    asserted_alignment: Literal["wolf", "village"]


class DeceptionBeliefShift(BaseModel):
    """Nearest pre/post belief change associated with one deception signal."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    signal_id: str
    speaker_seat: int = Field(ge=1)
    target_seat: int = Field(ge=1)
    observer_seat: int = Field(ge=1)
    pre_checkpoint_seq: int = Field(ge=1)
    post_checkpoint_seq: int = Field(ge=1)
    pre_wolf_probability: float = Field(ge=0.0, le=1.0)
    post_wolf_probability: float = Field(ge=0.0, le=1.0)
    wolf_probability_delta: float
    deception_direction_shift: float


class DeceptionSeatMetrics(BaseModel):
    """Aggregate facts for public deception authored by one seat."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    seat: int = Field(ge=1)
    role: str = "unknown"
    false_public_role_claim_count: int = Field(default=0, ge=0)
    false_public_seer_result_count: int = Field(default=0, ge=0)
    unscoreable_false_role_claim_count: int = Field(default=0, ge=0)
    scoreable_signal_count: int = Field(default=0, ge=0)
    paired_signal_count: int = Field(default=0, ge=0)
    belief_shift_observation_count: int = Field(default=0, ge=0)
    beneficial_shift_count: int = Field(default=0, ge=0)
    neutral_shift_count: int = Field(default=0, ge=0)
    harmful_shift_count: int = Field(default=0, ge=0)
    deception_direction_shift_sum: float = 0.0
    mean_deception_direction_shift: float | None = None

    @model_validator(mode="after")
    def _consistent_denominators(self) -> "DeceptionSeatMetrics":
        _validate_deception_counts(self, has_unpaired_field=False)
        return self


class RunDeceptionMetrics(BaseModel):
    """Explicit-denominator deception metrics for one run."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[RUN_DECEPTION_METRICS_SCHEMA_VERSION] = (
        RUN_DECEPTION_METRICS_SCHEMA_VERSION
    )
    run_id: str
    source_transcript_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    transcript_provenance_verified: bool = False
    source_role_layout_id: str | None = None
    source_persona_assignment_id: str | None = None
    method: str = "nearest_pre_first_post_unique_observer_window"
    causal: Literal[False] = False
    false_public_role_claim_count: int = Field(default=0, ge=0)
    false_public_seer_result_count: int = Field(default=0, ge=0)
    unscoreable_false_role_claim_count: int = Field(default=0, ge=0)
    scoreable_signal_count: int = Field(default=0, ge=0)
    paired_signal_count: int = Field(default=0, ge=0)
    unpaired_signal_count: int = Field(default=0, ge=0)
    belief_shift_observation_count: int = Field(default=0, ge=0)
    beneficial_shift_count: int = Field(default=0, ge=0)
    neutral_shift_count: int = Field(default=0, ge=0)
    harmful_shift_count: int = Field(default=0, ge=0)
    deception_direction_shift_sum: float = 0.0
    mean_deception_direction_shift: float | None = None
    seats: list[DeceptionSeatMetrics] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_denominators(self) -> "RunDeceptionMetrics":
        _validate_deception_counts(self, has_unpaired_field=True)
        seat_ids = [seat.seat for seat in self.seats]
        if len(seat_ids) != len(set(seat_ids)):
            raise ValueError("deception seat metrics must contain unique seats")
        if self.seats:
            additive = (
                "false_public_role_claim_count",
                "false_public_seer_result_count",
                "unscoreable_false_role_claim_count",
                "scoreable_signal_count",
                "paired_signal_count",
                "belief_shift_observation_count",
                "beneficial_shift_count",
                "neutral_shift_count",
                "harmful_shift_count",
            )
            mismatched = [
                field
                for field in additive
                if getattr(self, field)
                != sum(getattr(seat, field) for seat in self.seats)
            ]
            if mismatched:
                raise ValueError(
                    "deception run totals disagree with seat totals: "
                    + ",".join(mismatched)
                )
        return self


class DeceptionAggregate(BaseModel):
    """Cross-run totals with explicit denominators for every rate."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = DECEPTION_EVALUATION_SCHEMA_VERSION
    dimension: str
    key: str
    run_count: int = Field(default=0, ge=0)
    seat_run_count: int = Field(default=0, ge=0)
    false_public_role_claim_count: int = Field(default=0, ge=0)
    false_public_seer_result_count: int = Field(default=0, ge=0)
    unscoreable_false_role_claim_count: int = Field(default=0, ge=0)
    scoreable_signal_count: int = Field(default=0, ge=0)
    paired_signal_count: int = Field(default=0, ge=0)
    unpaired_signal_count: int = Field(default=0, ge=0)
    signal_pairing_rate: float | None = None
    belief_shift_observation_count: int = Field(default=0, ge=0)
    beneficial_shift_count: int = Field(default=0, ge=0)
    beneficial_shift_rate: float | None = None
    neutral_shift_count: int = Field(default=0, ge=0)
    harmful_shift_count: int = Field(default=0, ge=0)
    harmful_shift_rate: float | None = None
    deception_direction_shift_sum: float = 0.0
    mean_deception_direction_shift: float | None = None


class ExperimentDeceptionEvaluation(BaseModel):
    """Recomputable deception evaluation grouped by controlled dimensions."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = DECEPTION_EVALUATION_SCHEMA_VERSION
    run_count: int = Field(default=0, ge=0)
    causal: Literal[False] = False
    overall: DeceptionAggregate
    by_turn_policy: dict[str, DeceptionAggregate] = Field(default_factory=dict)
    by_role: dict[str, DeceptionAggregate] = Field(default_factory=dict)
    by_seat: dict[str, DeceptionAggregate] = Field(default_factory=dict)
    by_persona: dict[str, DeceptionAggregate] = Field(default_factory=dict)
    by_role_layout: dict[str, DeceptionAggregate] = Field(default_factory=dict)


class RunDeceptionAnalysis(BaseModel):
    """Metrics plus event-level evidence for audit and method validation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = DECEPTION_ANALYSIS_SCHEMA_VERSION
    metrics: RunDeceptionMetrics
    signals: list[DeceptionSignal] = Field(default_factory=list)
    shifts: list[DeceptionBeliefShift] = Field(default_factory=list)


@dataclass(frozen=True)
class _Checkpoint:
    seq: int
    owner_seat: int
    revision: int
    beliefs: dict[int, float]


@dataclass
class _SignalExtraction:
    signals: list[DeceptionSignal]
    false_role_by_seat: Counter[int]
    false_seer_by_seat: Counter[int]
    unscoreable_by_seat: Counter[int]


def analyze_deception(run: Any) -> RunDeceptionAnalysis:
    """Recompute deception signals and observer belief shifts from one run."""
    run_id = str(_value(run, "run_id") or "")
    evidence = validate_transcript_evidence(run)
    has_analysis_event = _has_analysis_event(evidence.entries)
    analysis = _mapping(validated_final_analysis(
        evidence,
        _value(run, "analysis"),
    ))
    truth_roles = _truth_roles(analysis.get("seats"))
    if has_analysis_event and not truth_roles:
        raise TranscriptIntegrityError(
            "transcript analysis is missing complete seat-role truth"
        )
    if has_analysis_event:
        raw_seats = analysis.get("seats")
        if not isinstance(raw_seats, list) or len(raw_seats) != len(truth_roles):
            raise TranscriptIntegrityError(
                "transcript analysis contains duplicate or invalid seat-role truth"
            )
    entries = list(evidence.entries)
    control_metadata = _mapping(evidence.metadata.get("caller_metadata"))
    extraction = _extract_signals(entries, truth_roles)
    checkpoints = _extract_checkpoints(entries)
    shifts = _associate_shifts(extraction.signals, checkpoints, truth_roles)
    metrics = _build_metrics(
        run_id=run_id,
        source_transcript_digest=evidence.stable_digest,
        transcript_provenance_verified=evidence.enclosing_digest_verified,
        source_role_layout_id=_nonempty_text(control_metadata.get("role_layout_id")),
        source_persona_assignment_id=_nonempty_text(
            control_metadata.get("persona_assignment_id")
        ),
        truth_roles=truth_roles,
        extraction=extraction,
        shifts=shifts,
    )
    return RunDeceptionAnalysis(
        metrics=metrics,
        signals=extraction.signals,
        shifts=shifts,
    )


def deception_metrics_from_run(run: Any) -> RunDeceptionMetrics:
    """Return only compact run metrics suitable for a summary JSONL row."""
    return analyze_deception(run).metrics


def aggregate_deception_metrics(
    rows: Iterable[Any],
) -> ExperimentDeceptionEvaluation | None:
    """Aggregate run rows without averaging already-derived per-run means."""
    normalized: list[tuple[Any, RunDeceptionMetrics]] = []
    seen: dict[str, tuple[Any, RunDeceptionMetrics]] = {}
    for row in rows:
        raw = _value(row, "deception_metrics")
        if raw is None:
            continue
        metrics = (
            raw
            if isinstance(raw, RunDeceptionMetrics)
            else RunDeceptionMetrics.model_validate(raw)
        )
        if not _trusted_metric_provenance(row, metrics):
            continue
        previous = seen.get(metrics.run_id)
        if previous is not None:
            if _same_evidence_row(previous[0], row):
                continue
            raise ValueError(
                f"conflicting deception evidence rows for run_id {metrics.run_id}"
            )
        seen[metrics.run_id] = (row, metrics)
        normalized.append((row, metrics))
    if not normalized:
        return None

    overall = _DeceptionAccumulator("overall", "all")
    by_policy: dict[str, _DeceptionAccumulator] = {}
    by_role: dict[str, _DeceptionAccumulator] = {}
    by_seat: dict[str, _DeceptionAccumulator] = {}
    by_persona: dict[str, _DeceptionAccumulator] = {}
    by_role_layout: dict[str, _DeceptionAccumulator] = {}
    for row, metrics in normalized:
        overall.add_run(metrics)
        policy = str(_value(row, "turn_policy") or "unknown")
        by_policy.setdefault(
            policy,
            _DeceptionAccumulator("turn_policy", policy),
        ).add_run(metrics)
        metadata = _mapping(_value(row, "metadata"))
        layout_id = _trusted_control_id(
            metadata,
            field="role_layout_id",
            source=metrics.source_role_layout_id,
        )
        if layout_id:
            by_role_layout.setdefault(
                layout_id,
                _DeceptionAccumulator("role_layout", layout_id),
            ).add_run(metrics)
        for seat in metrics.seats:
            by_role.setdefault(
                seat.role,
                _DeceptionAccumulator("role", seat.role),
            ).add_seat(metrics.run_id, seat)
            seat_key = str(seat.seat)
            by_seat.setdefault(
                seat_key,
                _DeceptionAccumulator("seat", seat_key),
            ).add_seat(metrics.run_id, seat)
            persona = (
                _persona_profile_for_seat(metadata, seat.seat)
                if metrics.source_persona_assignment_id is None
                or _trusted_control_id(
                    metadata,
                    field="persona_assignment_id",
                    source=metrics.source_persona_assignment_id,
                ) is not None
                else None
            )
            if persona:
                by_persona.setdefault(
                    persona,
                    _DeceptionAccumulator("persona", persona),
                ).add_seat(metrics.run_id, seat)
    return ExperimentDeceptionEvaluation(
        run_count=len(normalized),
        overall=overall.export(),
        by_turn_policy={key: value.export() for key, value in sorted(by_policy.items())},
        by_role={key: value.export() for key, value in sorted(by_role.items())},
        by_seat={
            key: value.export()
            for key, value in sorted(
                by_seat.items(),
                key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
            )
        },
        by_persona={key: value.export() for key, value in sorted(by_persona.items())},
        by_role_layout={
            key: value.export() for key, value in sorted(by_role_layout.items())
        },
    )


@dataclass
class _DeceptionAccumulator:
    dimension: str
    key: str
    run_ids: set[str] = field(default_factory=set)
    run_count: int = 0
    seat_run_count: int = 0
    false_public_role_claim_count: int = 0
    false_public_seer_result_count: int = 0
    unscoreable_false_role_claim_count: int = 0
    scoreable_signal_count: int = 0
    paired_signal_count: int = 0
    unpaired_signal_count: int = 0
    belief_shift_observation_count: int = 0
    beneficial_shift_count: int = 0
    neutral_shift_count: int = 0
    harmful_shift_count: int = 0
    deception_direction_shift_sum: float = 0.0
    seen_seat_runs: set[tuple[str, int]] = field(default_factory=set)

    def _mark_run(self, run_id: str) -> bool:
        normalized = str(run_id or f"anonymous-{self.run_count}")
        if normalized in self.run_ids:
            return False
        self.run_ids.add(normalized)
        self.run_count += 1
        return True

    def add_run(self, metrics: RunDeceptionMetrics) -> None:
        if not self._mark_run(metrics.run_id):
            return
        self.seat_run_count += len(metrics.seats)
        self._add_values(metrics)

    def add_seat(self, run_id: str, metrics: DeceptionSeatMetrics) -> None:
        self._mark_run(run_id)
        seat_key = (str(run_id or ""), metrics.seat)
        if seat_key in self.seen_seat_runs:
            return
        self.seen_seat_runs.add(seat_key)
        self.seat_run_count += 1
        self._add_values(metrics)

    def _add_values(self, metrics: RunDeceptionMetrics | DeceptionSeatMetrics) -> None:
        self.false_public_role_claim_count += metrics.false_public_role_claim_count
        self.false_public_seer_result_count += metrics.false_public_seer_result_count
        self.unscoreable_false_role_claim_count += (
            metrics.unscoreable_false_role_claim_count
        )
        self.scoreable_signal_count += metrics.scoreable_signal_count
        self.paired_signal_count += metrics.paired_signal_count
        self.unpaired_signal_count += (
            metrics.unpaired_signal_count
            if isinstance(metrics, RunDeceptionMetrics)
            else metrics.scoreable_signal_count - metrics.paired_signal_count
        )
        self.belief_shift_observation_count += metrics.belief_shift_observation_count
        self.beneficial_shift_count += metrics.beneficial_shift_count
        self.neutral_shift_count += metrics.neutral_shift_count
        self.harmful_shift_count += metrics.harmful_shift_count
        self.deception_direction_shift_sum += metrics.deception_direction_shift_sum

    def export(self) -> DeceptionAggregate:
        return DeceptionAggregate(
            dimension=self.dimension,
            key=self.key,
            run_count=self.run_count,
            seat_run_count=self.seat_run_count,
            false_public_role_claim_count=self.false_public_role_claim_count,
            false_public_seer_result_count=self.false_public_seer_result_count,
            unscoreable_false_role_claim_count=self.unscoreable_false_role_claim_count,
            scoreable_signal_count=self.scoreable_signal_count,
            paired_signal_count=self.paired_signal_count,
            unpaired_signal_count=self.unpaired_signal_count,
            signal_pairing_rate=_ratio(
                self.paired_signal_count,
                self.scoreable_signal_count,
            ),
            belief_shift_observation_count=self.belief_shift_observation_count,
            beneficial_shift_count=self.beneficial_shift_count,
            beneficial_shift_rate=_ratio(
                self.beneficial_shift_count,
                self.belief_shift_observation_count,
            ),
            neutral_shift_count=self.neutral_shift_count,
            harmful_shift_count=self.harmful_shift_count,
            harmful_shift_rate=_ratio(
                self.harmful_shift_count,
                self.belief_shift_observation_count,
            ),
            deception_direction_shift_sum=round(
                self.deception_direction_shift_sum,
                12,
            ),
            mean_deception_direction_shift=_ratio(
                self.deception_direction_shift_sum,
                self.belief_shift_observation_count,
            ),
        )


def _extract_signals(
    entries: list[Mapping[str, Any]],
    truth_roles: Mapping[int, str],
) -> _SignalExtraction:
    signals: list[DeceptionSignal] = []
    false_role_by_seat: Counter[int] = Counter()
    false_seer_by_seat: Counter[int] = Counter()
    unscoreable_by_seat: Counter[int] = Counter()
    for fallback_seq, entry in enumerate(entries, start=1):
        if entry.get("kind") != "event":
            continue
        payload = _mapping(entry.get("payload"))
        if not _is_public(payload) or payload.get("type") != "speech":
            continue
        claim = _mapping(payload.get("claim"))
        if not claim:
            continue
        speaker = _as_positive_int(payload.get("seat"))
        actual_role = truth_roles.get(speaker or -1)
        if speaker is None or not actual_role:
            continue
        seq = _as_positive_int(entry.get("seq")) or fallback_seq
        day = _as_nonnegative_int(payload.get("day"))
        claimed_role = str(claim.get("role") or "").strip().lower()
        actual_alignment = _role_alignment(actual_role)
        if claimed_role and claimed_role != actual_role:
            false_role_by_seat[speaker] += 1
            asserted_alignment = _claimed_role_alignment(
                actual_alignment=actual_alignment,
                claimed_role=claimed_role,
            )
            if asserted_alignment is None:
                unscoreable_by_seat[speaker] += 1
            else:
                signals.append(DeceptionSignal(
                    signal_id=f"{seq}:role:{speaker}",
                    event_seq=seq,
                    day=day,
                    speaker_seat=speaker,
                    target_seat=speaker,
                    kind="false_role_claim",
                    actual_alignment=actual_alignment,
                    asserted_alignment=asserted_alignment,
                ))

        if claimed_role != "seer":
            continue
        checked_seat = _as_positive_int(claim.get("checked_seat"))
        asserted_result = _normalize_alignment(claim.get("result"))
        checked_role = truth_roles.get(checked_seat or -1)
        if checked_seat is None or asserted_result is None or not checked_role:
            continue
        checked_alignment = _role_alignment(checked_role)
        if asserted_result == checked_alignment:
            continue
        false_seer_by_seat[speaker] += 1
        signals.append(DeceptionSignal(
            signal_id=f"{seq}:seer:{speaker}:{checked_seat}",
            event_seq=seq,
            day=day,
            speaker_seat=speaker,
            target_seat=checked_seat,
            kind="false_seer_result",
            actual_alignment=checked_alignment,
            asserted_alignment=asserted_result,
        ))
    return _SignalExtraction(
        signals=signals,
        false_role_by_seat=false_role_by_seat,
        false_seer_by_seat=false_seer_by_seat,
        unscoreable_by_seat=unscoreable_by_seat,
    )


def _extract_checkpoints(
    entries: list[Mapping[str, Any]],
) -> dict[int, list[_Checkpoint]]:
    result: dict[int, list[_Checkpoint]] = {}
    for fallback_seq, entry in enumerate(entries, start=1):
        if entry.get("kind") != "decision":
            continue
        payload = _mapping(entry.get("payload"))
        if payload.get("type") != "decision_consumed":
            continue
        snapshot = _mapping(payload.get("belief_state_after"))
        if snapshot.get("schema_version") != BELIEF_TRACE_SCHEMA_VERSION:
            continue
        owner = _as_positive_int(snapshot.get("owner_seat"))
        payload_seat = _as_positive_int(payload.get("seat"))
        if owner is None or (payload_seat is not None and owner != payload_seat):
            continue
        revision = _as_nonnegative_int_or_none(snapshot.get("revision"))
        if revision is None:
            raise TranscriptIntegrityError(
                "belief checkpoint revision must be a non-negative integer"
            )
        raw_beliefs = _mapping(snapshot.get("beliefs"))
        beliefs: dict[int, float] = {}
        for raw_target, raw_belief in raw_beliefs.items():
            target = _as_positive_int(raw_target)
            probability = _as_probability(_mapping(raw_belief).get("wolf_probability"))
            if target is not None and probability is not None:
                beliefs[target] = probability
        seq = _as_positive_int(entry.get("seq")) or fallback_seq
        result.setdefault(owner, []).append(_Checkpoint(
            seq=seq,
            owner_seat=owner,
            revision=revision,
            beliefs=beliefs,
        ))
    for checkpoints in result.values():
        checkpoints.sort(key=lambda checkpoint: checkpoint.seq)
        previous: _Checkpoint | None = None
        for checkpoint in checkpoints:
            if previous is not None and checkpoint.revision < previous.revision:
                raise TranscriptIntegrityError(
                    "belief checkpoint revisions must be monotonic per owner"
                )
            if (
                previous is not None
                and checkpoint.revision == previous.revision
                and checkpoint.beliefs != previous.beliefs
            ):
                raise TranscriptIntegrityError(
                    "one belief revision cannot contain conflicting checkpoints"
                )
            previous = checkpoint
    return result


def _associate_shifts(
    signals: list[DeceptionSignal],
    checkpoints: Mapping[int, list[_Checkpoint]],
    truth_roles: Mapping[int, str],
) -> list[DeceptionBeliefShift]:
    # One observer transition can contain repeated copies of the same public
    # proposition. Keep only the latest such signal so one change is not
    # multiplied by repetition before the observer next updates.
    associated: dict[
        tuple[int, int, int, str, str, int, int],
        DeceptionBeliefShift,
    ] = {}
    for signal in sorted(signals, key=lambda item: item.event_seq):
        speaker_team = _role_team(truth_roles.get(signal.speaker_seat, ""))
        if not speaker_team:
            continue
        direction = 1.0 if signal.asserted_alignment == "wolf" else -1.0
        for observer, observer_checkpoints in checkpoints.items():
            if observer in {signal.speaker_seat, signal.target_seat}:
                continue
            if _role_team(truth_roles.get(observer, "")) == speaker_team:
                continue
            before = _nearest_checkpoint(
                observer_checkpoints,
                target_seat=signal.target_seat,
                event_seq=signal.event_seq,
                before=True,
            )
            after = _nearest_checkpoint(
                observer_checkpoints,
                target_seat=signal.target_seat,
                event_seq=signal.event_seq,
                before=False,
            )
            if before is None or after is None:
                continue
            pre = before.beliefs[signal.target_seat]
            post = after.beliefs[signal.target_seat]
            delta = post - pre
            key = (
                observer,
                signal.speaker_seat,
                signal.target_seat,
                signal.kind,
                signal.asserted_alignment,
                before.seq,
                after.seq,
            )
            associated[key] = DeceptionBeliefShift(
                signal_id=signal.signal_id,
                speaker_seat=signal.speaker_seat,
                target_seat=signal.target_seat,
                observer_seat=observer,
                pre_checkpoint_seq=before.seq,
                post_checkpoint_seq=after.seq,
                pre_wolf_probability=pre,
                post_wolf_probability=post,
                wolf_probability_delta=delta,
                deception_direction_shift=direction * delta,
            )
    return sorted(
        associated.values(),
        key=lambda item: (
            item.pre_checkpoint_seq,
            item.post_checkpoint_seq,
            item.observer_seat,
            item.signal_id,
        ),
    )


def _build_metrics(
    *,
    run_id: str,
    source_transcript_digest: str,
    transcript_provenance_verified: bool,
    source_role_layout_id: str | None,
    source_persona_assignment_id: str | None,
    truth_roles: Mapping[int, str],
    extraction: _SignalExtraction,
    shifts: list[DeceptionBeliefShift],
) -> RunDeceptionMetrics:
    signal_by_id = {signal.signal_id: signal for signal in extraction.signals}
    paired_signal_ids = {shift.signal_id for shift in shifts}
    speakers = sorted(
        set(extraction.false_role_by_seat)
        | set(extraction.false_seer_by_seat)
        | {signal.speaker_seat for signal in extraction.signals}
    )
    seat_rows: list[DeceptionSeatMetrics] = []
    for seat in speakers:
        seat_signals = [signal for signal in extraction.signals if signal.speaker_seat == seat]
        seat_shifts = [shift for shift in shifts if shift.speaker_seat == seat]
        seat_rows.append(_seat_metrics(
            seat=seat,
            role=truth_roles.get(seat, "unknown"),
            false_role_count=extraction.false_role_by_seat[seat],
            false_seer_count=extraction.false_seer_by_seat[seat],
            unscoreable_count=extraction.unscoreable_by_seat[seat],
            signal_count=len(seat_signals),
            paired_signal_count=len({
                signal.signal_id for signal in seat_signals
                if signal.signal_id in paired_signal_ids
            }),
            shifts=seat_shifts,
        ))
    shift_sum = sum(shift.deception_direction_shift for shift in shifts)
    beneficial, neutral, harmful = _shift_direction_counts(shifts)
    return RunDeceptionMetrics(
        run_id=run_id,
        source_transcript_digest=source_transcript_digest,
        transcript_provenance_verified=transcript_provenance_verified,
        source_role_layout_id=source_role_layout_id,
        source_persona_assignment_id=source_persona_assignment_id,
        false_public_role_claim_count=sum(extraction.false_role_by_seat.values()),
        false_public_seer_result_count=sum(extraction.false_seer_by_seat.values()),
        unscoreable_false_role_claim_count=sum(extraction.unscoreable_by_seat.values()),
        scoreable_signal_count=len(signal_by_id),
        paired_signal_count=len(paired_signal_ids),
        unpaired_signal_count=len(signal_by_id) - len(paired_signal_ids),
        belief_shift_observation_count=len(shifts),
        beneficial_shift_count=beneficial,
        neutral_shift_count=neutral,
        harmful_shift_count=harmful,
        deception_direction_shift_sum=round(shift_sum, 12),
        mean_deception_direction_shift=_ratio(shift_sum, len(shifts)),
        seats=seat_rows,
    )


def _seat_metrics(
    *,
    seat: int,
    role: str,
    false_role_count: int,
    false_seer_count: int,
    unscoreable_count: int,
    signal_count: int,
    paired_signal_count: int,
    shifts: list[DeceptionBeliefShift],
) -> DeceptionSeatMetrics:
    shift_sum = sum(shift.deception_direction_shift for shift in shifts)
    beneficial, neutral, harmful = _shift_direction_counts(shifts)
    return DeceptionSeatMetrics(
        seat=seat,
        role=role,
        false_public_role_claim_count=false_role_count,
        false_public_seer_result_count=false_seer_count,
        unscoreable_false_role_claim_count=unscoreable_count,
        scoreable_signal_count=signal_count,
        paired_signal_count=paired_signal_count,
        belief_shift_observation_count=len(shifts),
        beneficial_shift_count=beneficial,
        neutral_shift_count=neutral,
        harmful_shift_count=harmful,
        deception_direction_shift_sum=round(shift_sum, 12),
        mean_deception_direction_shift=_ratio(shift_sum, len(shifts)),
    )


def _nearest_checkpoint(
    checkpoints: list[_Checkpoint],
    *,
    target_seat: int,
    event_seq: int,
    before: bool,
) -> _Checkpoint | None:
    eligible = [
        checkpoint
        for checkpoint in checkpoints
        if target_seat in checkpoint.beliefs
        and (checkpoint.seq < event_seq if before else checkpoint.seq > event_seq)
    ]
    if not eligible:
        return None
    return eligible[-1] if before else eligible[0]


def _shift_direction_counts(
    shifts: list[DeceptionBeliefShift],
) -> tuple[int, int, int]:
    beneficial = sum(shift.deception_direction_shift > 1e-12 for shift in shifts)
    harmful = sum(shift.deception_direction_shift < -1e-12 for shift in shifts)
    neutral = len(shifts) - beneficial - harmful
    return beneficial, neutral, harmful


def _truth_roles(value: Any) -> dict[int, str]:
    result: dict[int, str] = {}
    if not isinstance(value, list):
        return result
    for raw in value:
        item = _mapping(raw)
        seat = _as_positive_int(item.get("seat"))
        role = str(item.get("role") or "").strip().lower()
        if seat is not None and role in _KNOWN_ROLES:
            result[seat] = role
    return result


def _persona_profile_for_seat(metadata: Mapping[str, Any], seat: int) -> str | None:
    assignments = metadata.get("persona_assignments")
    if not isinstance(assignments, list):
        return None
    profiles: dict[int, str] = {}
    for raw in assignments:
        item = _mapping(raw)
        assigned_seat = _as_positive_int(item.get("seat"))
        profile = str(item.get("profile_id") or "").strip()
        if assigned_seat is None or not profile or assigned_seat in profiles:
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


def _validate_deception_counts(
    metrics: RunDeceptionMetrics | DeceptionSeatMetrics,
    *,
    has_unpaired_field: bool,
) -> None:
    if metrics.unscoreable_false_role_claim_count > metrics.false_public_role_claim_count:
        raise ValueError("unscoreable role claims exceed false role claims")
    expected_signals = (
        metrics.false_public_role_claim_count
        - metrics.unscoreable_false_role_claim_count
        + metrics.false_public_seer_result_count
    )
    if metrics.scoreable_signal_count != expected_signals:
        raise ValueError("scoreable signal denominator disagrees with false claims")
    if metrics.paired_signal_count > metrics.scoreable_signal_count:
        raise ValueError("paired signals exceed scoreable signals")
    if has_unpaired_field and (
        metrics.paired_signal_count + getattr(metrics, "unpaired_signal_count")
        != metrics.scoreable_signal_count
    ):
        raise ValueError("paired and unpaired signals must equal scoreable signals")
    if metrics.paired_signal_count > metrics.belief_shift_observation_count:
        raise ValueError("paired signals exceed observer belief-shift evidence")
    direction_count = (
        metrics.beneficial_shift_count
        + metrics.neutral_shift_count
        + metrics.harmful_shift_count
    )
    if direction_count != metrics.belief_shift_observation_count:
        raise ValueError("belief-shift directions must equal their denominator")
    if not isfinite(metrics.deception_direction_shift_sum):
        raise ValueError("deception direction shift sum must be finite")
    if metrics.belief_shift_observation_count == 0:
        if (
            abs(metrics.deception_direction_shift_sum) > 1e-12
            or metrics.mean_deception_direction_shift is not None
        ):
            raise ValueError("zero belief-shift observations require null mean and zero sum")
        return
    expected_mean = round(
        metrics.deception_direction_shift_sum
        / metrics.belief_shift_observation_count,
        12,
    )
    if (
        metrics.mean_deception_direction_shift is None
        or not isfinite(metrics.mean_deception_direction_shift)
        or abs(metrics.mean_deception_direction_shift - expected_mean) > 1e-9
    ):
        raise ValueError("deception direction mean disagrees with sum and denominator")


def _trusted_metric_provenance(row: Any, metrics: RunDeceptionMetrics) -> bool:
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


def _role_alignment(role: str) -> Literal["wolf", "village"]:
    return "wolf" if str(role).strip().lower() == "werewolf" else "village"


def _role_team(role: str) -> str:
    normalized = str(role).strip().lower()
    if not normalized:
        return ""
    return "werewolves" if normalized == "werewolf" else "village"


def _claimed_role_alignment(
    *,
    actual_alignment: Literal["wolf", "village"],
    claimed_role: str,
) -> Literal["wolf", "village"] | None:
    claimed_alignment = _role_alignment(claimed_role)
    return claimed_alignment if claimed_alignment != actual_alignment else None


def _normalize_alignment(value: Any) -> Literal["wolf", "village"] | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"wolf", "werewolf"}:
        return "wolf"
    if normalized in {"village", "villager"}:
        return "village"
    return None


def _is_public(payload: Mapping[str, Any]) -> bool:
    return not payload.get("recipients") and str(payload.get("visibility") or "public") not in {
        "private",
        "team",
        "admin",
        "god",
    }


def _transcript_entries(run: Any) -> list[Mapping[str, Any]]:
    transcript = _mapping(_value(run, "transcript"))
    entries = transcript.get("entries")
    return [item for item in (_mapping(raw) for raw in entries or []) if item]


def _has_analysis_event(entries: Iterable[Mapping[str, Any]]) -> bool:
    return any(
        entry.get("kind") == "event"
        and _mapping(entry.get("payload")).get("type") == "analysis"
        for entry in entries
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _as_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _as_nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _as_nonnegative_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _as_probability(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) and 0.0 <= parsed <= 1.0 else None


def _ratio(numerator: float, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / denominator, 12)
