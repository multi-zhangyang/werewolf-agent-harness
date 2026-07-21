"""Descriptive matched-pair comparisons for harness experiment rows.

The comparison layer is deliberately downstream of execution.  It consumes
credential-free ``RunSummaryRow`` values, requires the schedule's explicit
pair/control provenance, and never treats a difference as causal.  Missing,
duplicated, or control-mismatched pairs remain visible in the report instead
of being silently discarded.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .evidence_trust import is_verified
from .statistics import bootstrap_mean_ci


COMPARATIVE_EVALUATION_SCHEMA_VERSION = "werewolf.harness.comparative-evaluation.v1"
_BOOTSTRAP_ITERATIONS = 2000
_ATTESTED_ROW_KEY = "__harness_attested_summary_row__"


class PairedMetricEstimate(BaseModel):
    """One metric's paired difference, with explicit denominators."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    metric: str
    baseline_policy: str
    comparison_policy: str
    difference_definition: str = "comparison_minus_baseline"
    complete_pair_count: int = Field(default=0, ge=0)
    pair_count: int = Field(default=0, ge=0)
    missing_value_pair_count: int = Field(default=0, ge=0)
    baseline_observation_count: int = Field(default=0, ge=0)
    comparison_observation_count: int = Field(default=0, ge=0)
    baseline_mean: float | None = None
    comparison_mean: float | None = None
    mean_difference: float | None = None
    bootstrap_ci_low: float | None = None
    bootstrap_ci_high: float | None = None
    confidence_level: float = Field(default=0.95, ge=0.0, le=1.0)
    bootstrap_iterations: int = Field(default=_BOOTSTRAP_ITERATIONS, ge=0)
    positive_difference_count: int = Field(default=0, ge=0)
    zero_difference_count: int = Field(default=0, ge=0)
    negative_difference_count: int = Field(default=0, ge=0)


class PairedPolicyComparison(BaseModel):
    """Comparison of two policies over the same valid matched cases."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    baseline_policy: str
    comparison_policy: str
    complete_pair_count: int = Field(default=0, ge=0)
    metrics: dict[str, PairedMetricEstimate] = Field(default_factory=dict)


class ComparativeExperiment(BaseModel):
    """Pairing diagnostics and policy comparisons for one experiment ID."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    experiment_id: str
    policy_set: list[str] = Field(default_factory=list)
    expected_pair_count: int = Field(default=0, ge=0)
    observed_pair_count: int = Field(default=0, ge=0)
    complete_pair_count: int = Field(default=0, ge=0)
    incomplete_pair_count: int = Field(default=0, ge=0)
    invalid_pair_count: int = Field(default=0, ge=0)
    metadata_conflict_count: int = Field(default=0, ge=0)
    comparisons: list[PairedPolicyComparison] = Field(default_factory=list)


class ExperimentComparativeEvaluation(BaseModel):
    """Cross-run matched-pair evidence; ``causal`` is always false."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = COMPARATIVE_EVALUATION_SCHEMA_VERSION
    design: str = "matched_pair_descriptive"
    causal: Literal[False] = False
    run_count: int = Field(default=0, ge=0)
    eligible_run_count: int = Field(default=0, ge=0)
    unpaired_row_count: int = Field(default=0, ge=0)
    experiments: dict[str, ComparativeExperiment] = Field(default_factory=dict)


_BASE_METRICS = (
    "completed",
    "village_win",
    "werewolves_win",
    "days",
    "elapsed_seconds",
    "model_calls",
    "model_failures",
    "model_retries",
    "decision_failures",
    "decision_failure_rate",
    "provider_failure_rate",
    "input_tokens",
    "output_tokens",
    "model_latency_seconds",
)
_STRATEGY_METRICS = (
    "belief_brier",
    "false_role_claims",
    "beneficial_deception_shift_rate",
    "wolf_vote_agreement_rate",
)
_CONTROL_FIELDS = (
    "case_seed",
    "role_seed",
    "actor_seed",
    "orchestrator_seed",
    "experiment_spec_hash",
    "player_names",
    "role_deck",
    "seat_permutation",
    "seat_rotation",
    "role_layout_id",
    "role_layout_index",
    "role_layout_seed_base",
    "role_layout_block_id",
    "persona_assignment_id",
    "source_persona_assignment_id",
)


def aggregate_comparative_metrics(
    rows: Iterable[Any],
) -> ExperimentComparativeEvaluation | None:
    """Build schedule-aware paired comparisons from summary rows.

    A row is eligible only when it has an explicit experiment ID, policy set,
    pair ID, and seeded/control provenance.  This makes a hand-written or
    legacy row unable to create a spurious effect estimate.
    """

    normalized = list(rows)
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    declarations: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    expected_counts: dict[str, set[int]] = defaultdict(set)
    eligible_run_count = 0
    unpaired_row_count = 0
    unique_attested: dict[str, Any] = {}
    conflicted_run_ids: set[str] = set()

    for raw in normalized:
        if not is_verified(raw):
            unpaired_row_count += 1
            continue
        raw_run_id = _text(_value(raw, "run_id"))
        if not raw_run_id:
            unpaired_row_count += 1
            continue
        previous_raw = unique_attested.get(raw_run_id)
        if previous_raw is not None:
            if _same_row(previous_raw, raw):
                continue
            conflicted_run_ids.add(raw_run_id)
            unique_attested.pop(raw_run_id, None)
            continue
        if raw_run_id in conflicted_run_ids:
            continue
        unique_attested[raw_run_id] = raw

    unpaired_row_count += len(conflicted_run_ids) * 2
    for raw in unique_attested.values():
        row = dict(_as_mapping(raw))
        row[_ATTESTED_ROW_KEY] = True
        metadata = _as_mapping(row.get("metadata"))
        experiment_id = _text(metadata.get("experiment_id"))
        pair_id = _text(metadata.get("pair_id"))
        policies = _policy_set(metadata.get("policy_set"))
        policy = _text(row.get("turn_policy"))
        controls = _control_fingerprint(row, metadata)
        run_id = _text(row.get("run_id"))
        if (
            not experiment_id
            or not pair_id
            or len(policies) < 2
            or not policy
            or policy not in policies
            or controls is None
        ):
            unpaired_row_count += 1
            continue
        declarations[experiment_id].add(policies)
        runs_per_policy = _positive_int(metadata.get("runs_per_policy"))
        if runs_per_policy is not None:
            expected_counts[experiment_id].add(runs_per_policy)
        groups[(experiment_id, pair_id)].append({
            "row": row,
            "metadata": metadata,
            "policy": policy,
            "controls": controls,
        })
        eligible_run_count += 1

    if not declarations:
        return None

    reports: dict[str, ComparativeExperiment] = {}
    for experiment_id in sorted(declarations):
        policy_declarations = declarations[experiment_id]
        metadata_conflicts = max(0, len(policy_declarations) - 1)
        if not policy_declarations:
            continue
        # A schedule declaration is canonical provenance.  On conflict, use
        # the deterministic lexical choice for diagnostics but emit no valid
        # policy effect; the conflict remains explicit in the report.
        policy_set = sorted(policy_declarations)[0]
        if len(policy_declarations) == 1:
            policy_set = next(iter(policy_declarations))

        experiment_groups = {
            pair_id: entries
            for (group_experiment, pair_id), entries in groups.items()
            if group_experiment == experiment_id
        }
        observed = len(experiment_groups)
        expected_values = expected_counts.get(experiment_id, set())
        expected = max(expected_values) if expected_values else observed
        if len(expected_values) > 1:
            metadata_conflicts += len(expected_values) - 1

        complete: dict[str, dict[str, Mapping[str, Any]]] = {}
        incomplete = 0
        invalid = 0
        for pair_id, entries in experiment_groups.items():
            by_policy: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for entry in entries:
                by_policy[str(entry["policy"])].append(entry)
            duplicate = any(len(values) != 1 for values in by_policy.values())
            missing = any(policy not in by_policy for policy in policy_set)
            unknown = any(policy not in policy_set for policy in by_policy)
            control_values = {
                _stable_json(entry["controls"])
                for entry in entries
            }
            controls_mismatch = len(control_values) != 1
            if duplicate or unknown or controls_mismatch:
                invalid += 1
                continue
            if missing:
                incomplete += 1
                continue
            complete[pair_id] = {
                policy: by_policy[policy][0]["row"]
                for policy in policy_set
            }

        # A partially written schedule can omit entire pair IDs.  Count those
        # as incomplete rather than silently treating the observed subset as a
        # complete experiment.
        incomplete += max(0, expected - observed)
        comparisons: list[PairedPolicyComparison] = []
        if metadata_conflicts == 0:
            for baseline_index, baseline in enumerate(policy_set[:-1]):
                for comparison in policy_set[baseline_index + 1:]:
                    comparisons.append(_build_policy_comparison(
                        baseline,
                        comparison,
                        complete,
                        experiment_id=experiment_id,
                    ))
        reports[experiment_id] = ComparativeExperiment(
            experiment_id=experiment_id,
            policy_set=list(policy_set),
            expected_pair_count=expected,
            observed_pair_count=observed,
            complete_pair_count=len(complete),
            incomplete_pair_count=incomplete,
            invalid_pair_count=invalid,
            metadata_conflict_count=metadata_conflicts,
            comparisons=comparisons,
        )

    if not reports:
        return None
    return ExperimentComparativeEvaluation(
        run_count=len({_text(_value(raw, "run_id")) for raw in normalized}),
        eligible_run_count=eligible_run_count,
        unpaired_row_count=unpaired_row_count,
        experiments=reports,
    )


def _build_policy_comparison(
    baseline: str,
    comparison: str,
    complete: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    experiment_id: str,
) -> PairedPolicyComparison:
    metrics: dict[str, PairedMetricEstimate] = {}
    metric_names = _BASE_METRICS + _STRATEGY_METRICS
    for metric in metric_names:
        metrics[metric] = _estimate_metric(
            metric,
            baseline,
            comparison,
            complete,
            experiment_id=experiment_id,
        )
    return PairedPolicyComparison(
        baseline_policy=baseline,
        comparison_policy=comparison,
        complete_pair_count=len(complete),
        metrics=metrics,
    )


def _estimate_metric(
    metric: str,
    baseline_policy: str,
    comparison_policy: str,
    complete: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    experiment_id: str,
) -> PairedMetricEstimate:
    baseline_values: list[float] = []
    comparison_values: list[float] = []
    differences: list[float] = []
    baseline_observations = 0
    comparison_observations = 0
    for pair_id in sorted(complete):
        baseline_value = _metric_value(metric, complete[pair_id][baseline_policy])
        comparison_value = _metric_value(metric, complete[pair_id][comparison_policy])
        if baseline_value is not None:
            baseline_observations += 1
        if comparison_value is not None:
            comparison_observations += 1
        if baseline_value is None or comparison_value is None:
            continue
        baseline_values.append(baseline_value)
        comparison_values.append(comparison_value)
        differences.append(comparison_value - baseline_value)

    pair_count = len(differences)
    ci_low: float | None = None
    ci_high: float | None = None
    mean_difference: float | None = None
    baseline_mean: float | None = None
    comparison_mean: float | None = None
    positive = zero = negative = 0
    if pair_count:
        baseline_mean = round(sum(baseline_values) / pair_count, 12)
        comparison_mean = round(sum(comparison_values) / pair_count, 12)
        mean_difference = round(sum(differences) / pair_count, 12)
        seed_material = f"{experiment_id}|{baseline_policy}|{comparison_policy}|{metric}"
        seed = int.from_bytes(
            hashlib.sha256(seed_material.encode("utf-8")).digest()[:4],
            "big",
        )
        ci_low, ci_high = bootstrap_mean_ci(
            differences,
            iterations=_BOOTSTRAP_ITERATIONS,
            seed=seed,
        )
        ci_low = round(ci_low, 12)
        ci_high = round(ci_high, 12)
        positive = sum(value > 0 for value in differences)
        zero = sum(value == 0 for value in differences)
        negative = sum(value < 0 for value in differences)
    return PairedMetricEstimate(
        metric=metric,
        baseline_policy=baseline_policy,
        comparison_policy=comparison_policy,
        complete_pair_count=len(complete),
        pair_count=pair_count,
        missing_value_pair_count=len(complete) - pair_count,
        baseline_observation_count=baseline_observations,
        comparison_observation_count=comparison_observations,
        baseline_mean=baseline_mean,
        comparison_mean=comparison_mean,
        mean_difference=mean_difference,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        positive_difference_count=positive,
        zero_difference_count=zero,
        negative_difference_count=negative,
    )


def _metric_value(metric: str, row: Mapping[str, Any]) -> float | None:
    status = _text(row.get("status"))
    winner = _text(row.get("winner"))
    if metric == "completed":
        return float(status == "completed")
    if metric == "village_win":
        return float(winner == "village") if status == "completed" and winner else None
    if metric == "werewolves_win":
        return float(winner == "werewolves") if status == "completed" and winner else None
    if metric == "days":
        value = _nonnegative_number(row.get("days"))
        return value if status == "completed" else None
    direct_fields = {
        "elapsed_seconds": "elapsed_seconds",
        "model_calls": "model_calls",
        "model_failures": "model_failures",
        "model_retries": "model_retries",
        "decision_failures": "decision_failure_count",
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
        "model_latency_seconds": "model_latency_seconds",
        "false_role_claims": None,
    }
    if metric in direct_fields and direct_fields[metric] is not None:
        return _nonnegative_number(row.get(direct_fields[metric]))
    if metric == "provider_failure_rate":
        calls = _nonnegative_number(row.get("model_calls"))
        failures = _nonnegative_number(row.get("model_failures"))
        if calls is None or failures is None or failures > calls or calls <= 0:
            return None
        return failures / calls
    if metric == "decision_failure_rate":
        successes = _nonnegative_number(row.get("decision_count"))
        failures = _nonnegative_number(row.get("decision_failure_count"))
        if successes is None or failures is None:
            return None
        denominator = successes + failures
        return failures / denominator if denominator and failures <= denominator else None

    strategy = _trusted_nested_metrics(row, "strategy_metrics")
    if metric == "belief_brier":
        count = _nonnegative_number(strategy.get("belief_observation_count"))
        total = _nonnegative_number(strategy.get("belief_brier_sum"))
        if count is None or total is None or count <= 0 or total > count:
            return None
        return total / count
    if metric == "false_role_claims":
        count = _nonnegative_number(strategy.get("false_role_claim_count"))
        claims = _nonnegative_number(strategy.get("structured_claim_count"))
        return count if count is not None and claims is not None and count <= claims else None
    deception = _trusted_nested_metrics(row, "deception_metrics")
    if metric == "beneficial_deception_shift_rate":
        count = _nonnegative_number(deception.get("belief_shift_observation_count"))
        beneficial = _nonnegative_number(deception.get("beneficial_shift_count"))
        if count is None or beneficial is None or count <= 0 or beneficial > count:
            return None
        return beneficial / count
    if metric == "wolf_vote_agreement_rate":
        opportunities = _nonnegative_number(strategy.get("wolf_vote_agreement_opportunity_count"))
        agreements = _nonnegative_number(strategy.get("wolf_vote_agreement_count"))
        if (
            opportunities is None
            or agreements is None
            or opportunities <= 0
            or agreements > opportunities
        ):
            return None
        return agreements / opportunities
    return None


def _control_fingerprint(
    row: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    values: dict[str, Any] = {}
    for field in _CONTROL_FIELDS:
        if field in metadata:
            value = metadata.get(field)
        elif field in {"role_seed", "actor_seed", "orchestrator_seed", "player_names", "role_deck"}:
            value = row.get(field)
        else:
            value = None
        values[field] = value
    required = (
        "case_seed",
        "role_seed",
        "actor_seed",
        "orchestrator_seed",
        "experiment_spec_hash",
        "player_names",
        "role_deck",
    )
    if any(values[field] is None for field in required):
        return None
    if any(type(values[field]) is not int for field in (
        "case_seed",
        "role_seed",
        "actor_seed",
        "orchestrator_seed",
    )):
        return None
    names = values["player_names"]
    deck = values["role_deck"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
        or not isinstance(deck, list)
        or len(deck) != len(names)
        or any(not isinstance(role, str) or not role.strip() for role in deck)
    ):
        return None
    if not _is_hash(values["experiment_spec_hash"]):
        return None
    if str(metadata.get("role_layout_mode") or "legacy") != "legacy" and not values.get(
        "role_layout_id"
    ):
        return None
    if str(metadata.get("persona_mode") or "legacy") != "legacy" and not values.get(
        "persona_assignment_id"
    ):
        return None
    return values


def _trusted_nested_metrics(
    row: Mapping[str, Any],
    field: str,
) -> Mapping[str, Any]:
    if row.get(_ATTESTED_ROW_KEY) is not True:
        return {}
    metrics = _as_mapping(row.get(field))
    if not metrics:
        return {}
    row_digest = _text(row.get("transcript_digest"))
    source_digest = _text(metrics.get("source_transcript_digest"))
    if (
        not row_digest
        or _text(metrics.get("run_id")) != _text(row.get("run_id"))
        or source_digest != row_digest
        or metrics.get("transcript_provenance_verified") is not True
    ):
        return {}
    return metrics


def _policy_set(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    values = tuple(_text(item) for item in value)
    if any(not item for item in values) or len(set(values)) != len(values):
        return ()
    return values


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        result = dump()
        if isinstance(result, Mapping):
            return result
    return {}


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _same_row(first: Any, second: Any) -> bool:
    if first is second:
        return True
    first_dump = getattr(first, "model_dump", None)
    second_dump = getattr(second, "model_dump", None)
    if not callable(first_dump) or not callable(second_dump):
        return False
    return first_dump(mode="json") == second_dump(mode="json")


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _nonnegative_number(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number >= 0 else None


def _is_hash(value: Any) -> bool:
    text = str(value) if value is not None else ""
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _positive_int(value: Any) -> int | None:
    if type(value) is not int or value < 1:
        return None
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
