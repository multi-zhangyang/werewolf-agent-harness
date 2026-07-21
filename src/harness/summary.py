"""Aggregate only directly observable outcomes and resource usage."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .comparison import (
    ExperimentComparativeEvaluation,
    aggregate_comparative_metrics,
)
from .deception import ExperimentDeceptionEvaluation, aggregate_deception_metrics
from .evaluation import (
    ExperimentOperationalEvaluation,
    ExperimentStrategyEvaluation,
    aggregate_operational_metrics,
    aggregate_strategy_metrics,
)
from .results import RunSummaryRow, is_verified_summary_row


class ExperimentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    run_count: int
    evaluation_evidence_run_count: int = Field(default=0, ge=0)
    cache_only_run_count: int = Field(default=0, ge=0)
    completed_runs: int
    failed_runs: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    winner_counts: dict[str, int] = Field(default_factory=dict)
    total_days: int = 0
    total_elapsed_seconds: float = 0.0
    total_model_calls: int = 0
    total_model_failures: int = 0
    total_model_retries: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_model_latency_seconds: float = 0.0
    total_decisions: int = 0
    total_consumed_parse_recoveries: int = 0
    total_structured_responses: int = 0
    total_incomplete_responses: int = 0
    total_response_parse_failures: int = 0
    total_response_parse_recoveries: int = 0
    total_lossy_parse_rejections: int = 0
    total_decision_failures: int = 0
    runs_by_turn_policy: dict[str, int] = Field(default_factory=dict)
    runs_by_persona_mode: dict[str, int] = Field(default_factory=dict)
    runs_by_role_layout: dict[str, int] = Field(default_factory=dict)
    operational_evaluation: ExperimentOperationalEvaluation | None = None
    strategy_evaluation: ExperimentStrategyEvaluation | None = None
    deception_evaluation: ExperimentDeceptionEvaluation | None = None
    comparative_evaluation: ExperimentComparativeEvaluation | None = None


def summarize_runs(rows: Iterable[RunSummaryRow | Mapping[str, Any]]) -> ExperimentSummary:
    parsed = [
        row if isinstance(row, RunSummaryRow) else RunSummaryRow(**dict(row))
        for row in rows
    ]
    normalized = _deduplicate_rows(parsed)
    evaluation_rows = [row for row in normalized if is_verified_summary_row(row)]
    status_counts = Counter(row.status for row in normalized)
    winner_counts = Counter(row.winner for row in normalized if row.winner)
    policy_counts = Counter(row.turn_policy for row in normalized)
    persona_mode_counts = Counter(
        str(row.metadata.get("persona_mode") or "legacy") for row in normalized
    )
    role_layout_counts = Counter(
        str(row.metadata.get("role_layout_id") or "legacy") for row in normalized
    )
    return ExperimentSummary(
        run_count=len(normalized),
        evaluation_evidence_run_count=len(evaluation_rows),
        cache_only_run_count=len(normalized) - len(evaluation_rows),
        completed_runs=status_counts.get("completed", 0),
        failed_runs=len(normalized) - status_counts.get("completed", 0),
        status_counts=dict(sorted(status_counts.items())),
        winner_counts=dict(sorted(winner_counts.items())),
        total_days=sum(row.days for row in normalized),
        total_elapsed_seconds=round(sum(row.elapsed_seconds for row in normalized), 6),
        total_model_calls=sum(row.model_calls for row in normalized),
        total_model_failures=sum(row.model_failures for row in normalized),
        total_model_retries=sum(row.model_retries for row in normalized),
        total_input_tokens=sum(row.input_tokens for row in normalized),
        total_output_tokens=sum(row.output_tokens for row in normalized),
        total_model_latency_seconds=round(sum(row.model_latency_seconds for row in normalized), 6),
        total_decisions=sum(row.decision_count for row in normalized),
        total_consumed_parse_recoveries=sum(row.consumed_parse_recovery_count for row in normalized),
        total_structured_responses=sum(row.structured_response_count for row in normalized),
        total_incomplete_responses=sum(row.incomplete_response_count for row in normalized),
        total_response_parse_failures=sum(row.response_parse_failure_count for row in normalized),
        total_response_parse_recoveries=sum(row.response_parse_recovery_count for row in normalized),
        total_lossy_parse_rejections=sum(row.lossy_parse_rejection_count for row in normalized),
        total_decision_failures=sum(row.decision_failure_count for row in normalized),
        runs_by_turn_policy=dict(sorted(policy_counts.items())),
        runs_by_persona_mode=dict(sorted(persona_mode_counts.items())),
        runs_by_role_layout=dict(sorted(role_layout_counts.items())),
        operational_evaluation=aggregate_operational_metrics(evaluation_rows),
        strategy_evaluation=aggregate_strategy_metrics(evaluation_rows),
        deception_evaluation=aggregate_deception_metrics(evaluation_rows),
        comparative_evaluation=aggregate_comparative_metrics(evaluation_rows),
    )


def _deduplicate_rows(rows: Iterable[RunSummaryRow]) -> list[RunSummaryRow]:
    unique: dict[str, RunSummaryRow] = {}
    for row in rows:
        previous = unique.get(row.run_id)
        if previous is None:
            unique[row.run_id] = row
            continue
        if previous.model_dump(mode="json") != row.model_dump(mode="json"):
            raise ValueError(f"conflicting summary rows for run_id {row.run_id}")
        if is_verified_summary_row(row) and not is_verified_summary_row(previous):
            unique[row.run_id] = row
    return list(unique.values())
