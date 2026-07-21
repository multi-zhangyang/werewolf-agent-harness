"""Compact factual rows derived from harness run results."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .deception import RunDeceptionMetrics, deception_metrics_from_run
from .evidence_trust import _mark_verified, is_verified
from .runner import HarnessRunResult
from .evaluation import (
    RunAuditMetrics,
    RunStrategyMetrics,
    audit_metrics_from_run,
    strategy_metrics_from_run,
)
from .transcript import (
    redact_sensitive,
    validated_final_analysis,
    validate_transcript_evidence,
)

LEGACY_RUN_SUMMARY_SCHEMA_VERSION = "werewolf.harness.run_summary.v3"
RUN_SUMMARY_SCHEMA_VERSION = "werewolf.harness.run_summary.v4"


class RunSummaryRow(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    _evaluation_attestation_digest: str | None = PrivateAttr(default=None)

    # New rows are v4; the Literal union keeps legacy v3 resume rows readable.
    schema_version: Literal[
        LEGACY_RUN_SUMMARY_SCHEMA_VERSION,
        RUN_SUMMARY_SCHEMA_VERSION,
    ] = RUN_SUMMARY_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    termination_reason: str | None = None
    winner: str | None = None
    days: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    error_type: str | None = None
    error: str | None = None
    environment_id: str = Field(min_length=1)
    environment_version: str = Field(min_length=1)
    turn_policy: str
    player_names: list[str] = Field(default_factory=list)
    role_deck: list[str] = Field(default_factory=list)
    role_seed: int
    actor_seed: int
    orchestrator_seed: int
    run_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcript_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_count: int = Field(default=0, ge=0)
    decision_trace_count: int = Field(default=0, ge=0)
    decision_count: int = Field(default=0, ge=0)
    consumed_parse_recovery_count: int = Field(default=0, ge=0)
    decision_failure_count: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    model_successes: int = Field(default=0, ge=0)
    model_failures: int = Field(default=0, ge=0)
    model_retries: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    model_latency_seconds: float = Field(default=0.0, ge=0.0)
    structured_response_count: int = Field(default=0, ge=0)
    incomplete_response_count: int = Field(default=0, ge=0)
    response_parse_failure_count: int = Field(default=0, ge=0)
    response_parse_recovery_count: int = Field(default=0, ge=0)
    lossy_parse_rejection_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    audit_metrics: RunAuditMetrics | None = None
    strategy_metrics: RunStrategyMetrics | None = None
    deception_metrics: RunDeceptionMetrics | None = None

def run_summary_from_result(
    result: HarnessRunResult | Mapping[str, Any],
) -> RunSummaryRow:
    run = result if isinstance(result, HarnessRunResult) else HarnessRunResult(**dict(result))
    spec = dict(run.run_spec)
    evidence = validate_transcript_evidence(run)
    canonical_analysis = validated_final_analysis(evidence, run.analysis)
    analysis = dict(canonical_analysis or {})
    if "winner" in analysis and run.winner is not None and analysis.get("winner") != run.winner:
        raise ValueError("run result winner does not match transcript analysis")
    if "days" in analysis:
        canonical_days = _integer(analysis.get("days"))
        if canonical_days != run.days:
            raise ValueError("run result days does not match transcript analysis")
    parse = dict(analysis.get("parse_metrics") or {})
    failures = dict(analysis.get("decision_failure_metrics") or {})
    router = dict(run.router_stats_delta or {})
    audit_metrics = audit_metrics_from_run(run)
    strategy_metrics = strategy_metrics_from_run(run)
    truth_seats = analysis.get("seats")
    deception_metrics = (
        deception_metrics_from_run(run)
        if isinstance(truth_seats, list) and bool(truth_seats)
        else None
    )
    row = RunSummaryRow(
        run_id=run.run_id,
        status=run.status,
        termination_reason=run.termination_reason,
        winner=analysis.get("winner", run.winner),
        days=_integer(analysis.get("days"), run.days),
        elapsed_seconds=run.elapsed_seconds,
        error_type=run.error_type,
        error=str(redact_sensitive(run.error)) if run.error is not None else None,
        environment_id=str(spec.get("environment_id") or "werewolf.classic"),
        environment_version=str(spec.get("environment_version") or "1"),
        turn_policy=str(spec.get("turn_policy") or analysis.get("turn_policy") or ""),
        player_names=[str(value) for value in spec.get("player_names") or []],
        role_deck=[str(value) for value in spec.get("role_deck") or []],
        role_seed=run.role_seed,
        actor_seed=run.actor_seed,
        orchestrator_seed=run.orchestrator_seed,
        run_spec_hash=run.run_spec_hash,
        transcript_digest=evidence.stable_digest,
        event_count=evidence.counts_by_kind.get("event", 0),
        decision_trace_count=evidence.counts_by_kind.get("decision", 0),
        decision_count=_integer(analysis.get("decision_count"), _integer(parse.get("decision_count"))),
        consumed_parse_recovery_count=_integer(parse.get("parse_recovered_count")),
        decision_failure_count=_integer(failures.get("failure_count")),
        model_calls=_integer(router.get("calls")),
        model_successes=_integer(router.get("successes")),
        model_failures=_integer(router.get("failures")),
        model_retries=_integer(router.get("retries")),
        input_tokens=_integer(router.get("total_tokens_in")),
        output_tokens=_integer(router.get("total_tokens_out")),
        model_latency_seconds=_number(router.get("total_latency")),
        structured_response_count=_integer(router.get("structured_responses")),
        incomplete_response_count=_integer(router.get("incomplete_responses")),
        response_parse_failure_count=_integer(router.get("response_parse_failures")),
        response_parse_recovery_count=_integer(router.get("response_parse_recoveries")),
        lossy_parse_rejection_count=_integer(router.get("lossy_parse_rejections")),
        metadata=dict(spec.get("metadata") or {}),
        audit_metrics=audit_metrics,
        strategy_metrics=strategy_metrics,
        deception_metrics=deception_metrics,
    )
    return _mark_verified(row)


def is_verified_summary_row(value: Any) -> bool:
    """Return whether a row still matches verified in-process evidence.

    The attestation is intentionally absent from ``model_dump`` and JSON.
    Deserializing a standalone summary row therefore never grants evaluation
    trust; an artifact-backed loader must rederive it from the transcript.
    """

    return isinstance(value, RunSummaryRow) and is_verified(value)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
