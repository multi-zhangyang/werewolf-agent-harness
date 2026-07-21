"""Executable, versioned adversarial scenarios for the generic harness core.

The suite layer deliberately knows nothing about Werewolf or any concrete
provider.  A scenario is one exact :class:`CoreRunSpec` plus machine-checkable
expectations.  Environment plugins and agents remain supplied by the caller,
so test doubles do not become production harness behavior.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .core_runner import EnvironmentRunResult, run_environment_run
from .core_spec import CoreRunSpec
from .registry import EnvironmentRegistry
from .transcript import redact_sensitive, validate_transcript_evidence


ADVERSARIAL_SCENARIO_SPEC_VERSION = "agent-harness.adversarial-scenario.v1"
ADVERSARIAL_SCENARIO_REPORT_VERSION = "agent-harness.adversarial-scenario-report.v1"
ADVERSARIAL_SUITE_REPORT_VERSION = "agent-harness.adversarial-suite-report.v1"

ScenarioCategory = Literal[
    "protocol_identity",
    "protocol_action_schema",
    "agent_fault",
    "deadline",
    "confidentiality",
    "outcome_integrity",
    "multi_agent_collusion",
]
InvariantKind = Literal[
    "request_terminal_pairing",
    "minimum_request_count",
    "run_status",
    "error_type",
    "outcome_equals",
    "terminal_kind_count",
    "terminal_error_type_present",
    "validation_issue_present",
    "fact_count",
    "request_observation_key_absent",
    "no_fabricated_choice",
    "secret_absence",
]

TERMINAL_TRACE_KINDS = (
    "agent_response",
    "agent_response_failed",
    "agent_response_cancelled",
    "agent_response_validation_failed",
)
_FACT_COUNT_SUBJECTS = {
    "request",
    "terminal",
    "valid_response",
    "rejected_response",
    "accepted_action",
    "accepted_skip",
    "rejected_action",
    "rejected_skip",
    "target_argument",
    "fabricated_choice_terminal",
}
_REQUIRED_INVARIANTS = {
    ("request_terminal_pairing", None),
    ("minimum_request_count", None),
    ("no_fabricated_choice", None),
    ("secret_absence", None),
}


class ExpectedInvariant(BaseModel):
    """One supported assertion over facts produced by a scenario run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: InvariantKind
    expected: Any
    subject: str | None = None

    @model_validator(mode="after")
    def _well_formed_expectation(self) -> "ExpectedInvariant":
        if self.subject is not None and not self.subject.strip():
            raise ValueError("invariant subject must not be empty")
        boolean_kinds = {
            "request_terminal_pairing",
            "validation_issue_present",
            "terminal_error_type_present",
            "request_observation_key_absent",
            "no_fabricated_choice",
            "secret_absence",
        }
        if self.kind in boolean_kinds and self.expected is not True:
            raise ValueError(f"{self.kind} invariant must expect true")
        subject_kinds = {
            "terminal_kind_count",
            "terminal_error_type_present",
            "validation_issue_present",
            "fact_count",
            "request_observation_key_absent",
        }
        if self.kind in subject_kinds and self.subject is None:
            raise ValueError(f"{self.kind} invariant requires a subject")
        if self.kind not in subject_kinds and self.subject is not None:
            raise ValueError(f"{self.kind} invariant does not accept a subject")
        if self.kind in {"minimum_request_count", "terminal_kind_count", "fact_count"}:
            if type(self.expected) is not int or self.expected < 0:
                raise ValueError(f"{self.kind} invariant expects a non-negative integer")
        if self.kind == "minimum_request_count" and self.expected < 1:
            raise ValueError("minimum_request_count must be at least one")
        if self.kind == "terminal_kind_count" and self.subject not in TERMINAL_TRACE_KINDS:
            raise ValueError("terminal_kind_count subject is not a terminal trace kind")
        if self.kind == "fact_count" and self.subject not in _FACT_COUNT_SUBJECTS:
            raise ValueError("fact_count subject is not supported")
        if self.kind == "run_status" and self.expected not in {
            "completed", "failed", "timed_out"
        }:
            raise ValueError("run_status invariant has an unsupported status")
        if self.kind == "error_type" and not (
            self.expected is None
            or isinstance(self.expected, str) and bool(self.expected.strip())
        ):
            raise ValueError("error_type invariant expects null or a non-empty string")
        if self.kind == "outcome_equals" and not isinstance(self.expected, dict):
            raise ValueError("outcome_equals invariant expects an object")
        return self


class ScenarioSpec(BaseModel):
    """An immutable adversarial run with exact plugin identity and fixed seeds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "agent-harness.adversarial-scenario.v1"
    ] = ADVERSARIAL_SCENARIO_SPEC_VERSION
    scenario_id: str = Field(min_length=1, max_length=200)
    category: ScenarioCategory
    run_spec: CoreRunSpec
    expected_invariants: tuple[ExpectedInvariant, ...] = Field(min_length=1)

    @field_validator("scenario_id")
    @classmethod
    def _safe_scenario_id(cls, value: str) -> str:
        scenario_id = str(value).strip()
        if (
            not scenario_id
            or scenario_id in {".", ".."}
            or any(character in scenario_id for character in ("/", "\\", "\x00"))
        ):
            raise ValueError("scenario_id must be a single safe path component")
        return scenario_id

    @model_validator(mode="after")
    def _fixed_and_auditable(self) -> "ScenarioSpec":
        if not self.run_spec.seeds:
            raise ValueError("adversarial scenarios require at least one fixed seed")
        keys = [(item.kind, item.subject) for item in self.expected_invariants]
        if len(keys) != len(set(keys)):
            raise ValueError("scenario expected invariants must be unique")
        missing = sorted(_REQUIRED_INVARIANTS - set(keys))
        if missing:
            labels = [kind for kind, _subject in missing]
            raise ValueError(
                "scenario is missing required invariants: " + ",".join(labels)
            )
        return self


class InvariantResult(BaseModel):
    """Observed and expected values for one evaluated invariant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: InvariantKind
    subject: str | None = None
    passed: bool
    expected: Any
    observed: Any


class ScenarioReport(BaseModel):
    """Credential-free facts and invariant results for one executed scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "agent-harness.adversarial-scenario-report.v1"
    ] = ADVERSARIAL_SCENARIO_REPORT_VERSION
    scenario_schema_version: str
    scenario_id: str
    category: ScenarioCategory
    run_id: str
    environment_id: str
    environment_version: str
    run_spec_hash: str
    fixed_seeds: dict[str, int]
    passed: bool
    run_status: str
    error_type: str | None = None
    outcome: dict[str, Any] = Field(default_factory=dict)
    transcript_digest: str
    request_count: int = Field(ge=0)
    terminal_count: int = Field(ge=0)
    terminal_counts: dict[str, int] = Field(default_factory=dict)
    validation_issue_counts: dict[str, int] = Field(default_factory=dict)
    terminal_error_type_counts: dict[str, int] = Field(default_factory=dict)
    fact_counts: dict[str, int] = Field(default_factory=dict)
    secret_marker_count: int = Field(ge=0)
    secret_marker_match_count: int = Field(ge=0)
    invariant_results: tuple[InvariantResult, ...]


class ScenarioSuiteReport(BaseModel):
    """Aggregate scenario facts; it makes no claims beyond executed specs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "agent-harness.adversarial-suite-report.v1"
    ] = ADVERSARIAL_SUITE_REPORT_VERSION
    status: Literal["passed", "failed"]
    scenario_count: int = Field(ge=0)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    category_counts: dict[str, int] = Field(default_factory=dict)
    reports: tuple[ScenarioReport, ...]


ScenarioAgentResolver = Callable[[ScenarioSpec, str], Any]


async def run_adversarial_scenario(
    spec: ScenarioSpec,
    *,
    registry: EnvironmentRegistry,
    resolve_agent: Callable[[str], Any],
    sensitive_markers: Sequence[str] = (),
) -> ScenarioReport:
    """Execute one scenario through ``run_environment_run`` and audit facts."""
    markers = _normalize_sensitive_markers(sensitive_markers)
    result = await run_environment_run(
        spec.run_spec,
        registry=registry,
        resolve_agent=resolve_agent,
    )
    # Do not evaluate a detached or reordered trace.  The generic runner emits
    # a digest, but an offline caller may hand this function a substituted
    # result object; validate the evidence before deriving scenario facts.
    validate_transcript_evidence(result)
    facts = _collect_trace_facts(result)
    serialized_result = _canonical_json(result.model_dump())
    marker_match_count = sum(serialized_result.count(marker) for marker in markers)
    redaction_clean = redact_sensitive(result.model_dump()) == result.model_dump()
    invariants = tuple(
        _evaluate_invariant(
            expected,
            result=result,
            facts=facts,
            marker_match_count=marker_match_count,
            redaction_clean=redaction_clean,
        )
        for expected in spec.expected_invariants
    )
    report_payload = {
        "scenario_schema_version": spec.schema_version,
        "scenario_id": spec.scenario_id,
        "category": spec.category,
        "run_id": result.run_id,
        "environment_id": result.environment_id,
        "environment_version": result.environment_version,
        "run_spec_hash": result.run_spec_hash,
        "fixed_seeds": dict(spec.run_spec.seeds),
        "passed": all(item.passed for item in invariants),
        "run_status": result.status,
        "error_type": result.error_type,
        "outcome": result.outcome,
        "transcript_digest": result.transcript_digest,
        "request_count": facts.request_count,
        "terminal_count": facts.terminal_count,
        "terminal_counts": dict(sorted(facts.terminal_counts.items())),
        "validation_issue_counts": dict(sorted(facts.validation_issue_counts.items())),
        "terminal_error_type_counts": dict(sorted(facts.terminal_error_type_counts.items())),
        "fact_counts": dict(sorted(facts.fact_counts.items())),
        "secret_marker_count": len(markers),
        "secret_marker_match_count": marker_match_count,
        "invariant_results": [item.model_dump() for item in invariants],
    }
    safe_report_payload = _redact_sensitive_markers(
        report_payload,
        markers,
    )
    return ScenarioReport.model_validate(safe_report_payload)


async def run_adversarial_scenario_suite(
    specs: Sequence[ScenarioSpec],
    *,
    registry: EnvironmentRegistry,
    resolve_agent: ScenarioAgentResolver,
    sensitive_markers: Mapping[str, Sequence[str]] | None = None,
) -> ScenarioSuiteReport:
    """Execute a deterministic list of scenarios and aggregate only run facts."""
    scenario_ids = [spec.scenario_id for spec in specs]
    run_ids = [spec.run_spec.run_id for spec in specs]
    if not scenario_ids:
        raise ValueError("adversarial suite must contain at least one scenario")
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("adversarial suite scenario_id values must be unique")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("adversarial suite run_id values must be unique")
    marker_map = dict(sensitive_markers or {})
    unknown_marker_ids = sorted(set(marker_map) - set(scenario_ids))
    if unknown_marker_ids:
        raise ValueError(
            "sensitive markers supplied for unknown scenarios: "
            + ",".join(unknown_marker_ids)
        )

    reports: list[ScenarioReport] = []
    for spec in specs:
        reports.append(await run_adversarial_scenario(
            spec,
            registry=registry,
            resolve_agent=lambda actor_id, scenario=spec: resolve_agent(
                scenario, actor_id
            ),
            sensitive_markers=marker_map.get(spec.scenario_id, ()),
        ))
    passed_count = sum(report.passed for report in reports)
    category_counts = Counter(report.category for report in reports)
    return ScenarioSuiteReport(
        status="passed" if passed_count == len(reports) else "failed",
        scenario_count=len(reports),
        passed_count=passed_count,
        failed_count=len(reports) - passed_count,
        category_counts=dict(sorted(category_counts.items())),
        reports=tuple(reports),
    )


class _TraceFacts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_count: int
    terminal_count: int
    pairing_valid: bool
    terminal_counts: dict[str, int]
    validation_issue_counts: dict[str, int]
    terminal_error_type_counts: dict[str, int]
    fact_counts: dict[str, int]
    observation_keys: frozenset[str]


def _collect_trace_facts(result: EnvironmentRunResult) -> _TraceFacts:
    requests: Counter[str] = Counter()
    terminals: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()
    validation_issue_counts: Counter[str] = Counter()
    terminal_error_type_counts: Counter[str] = Counter()
    observation_keys: set[str] = set()
    facts: Counter[str] = Counter()
    request_positions: dict[str, list[int]] = {}
    terminal_positions: dict[str, list[int]] = {}

    entries = result.transcript.get("entries")
    if not isinstance(entries, list):
        entries = []
    for row_index, row in enumerate(entries):
        if not isinstance(row, dict) or row.get("kind") != "decision":
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        trace_kind = payload.get("kind")
        if trace_kind == "agent_request":
            request = payload.get("request")
            request_id = request.get("request_id") if isinstance(request, dict) else None
            if isinstance(request_id, str) and request_id:
                requests[request_id] += 1
                request_positions.setdefault(request_id, []).append(row_index)
            facts["request"] += 1
            observation = request.get("observation") if isinstance(request, dict) else None
            if isinstance(observation, dict):
                observation_keys.update(str(key) for key in observation)
            continue
        if trace_kind not in TERMINAL_TRACE_KINDS:
            continue
        terminal_counts[str(trace_kind)] += 1
        facts["terminal"] += 1
        request_id = payload.get("request_id")
        if isinstance(request_id, str) and request_id:
            terminals[request_id] += 1
            terminal_positions.setdefault(request_id, []).append(row_index)
        if trace_kind in {"agent_response_failed", "agent_response_cancelled"}:
            if any(
                key in payload
                for key in (
                    "envelope", "choice", "arguments", "target", "target_seat",
                    "selected_target",
                )
            ):
                facts["fabricated_choice_terminal"] += 1
        failure = payload.get("failure")
        if isinstance(failure, dict):
            error_type = failure.get("error_type")
            if isinstance(error_type, str) and error_type:
                terminal_error_type_counts[error_type] += 1
        if trace_kind != "agent_response":
            continue
        validation = payload.get("validation")
        valid = isinstance(validation, dict) and validation.get("valid") is True
        facts["valid_response" if valid else "rejected_response"] += 1
        if isinstance(validation, dict):
            issues = validation.get("issues")
            if isinstance(issues, list):
                for issue in issues:
                    code = issue.get("code") if isinstance(issue, dict) else None
                    if isinstance(code, str) and code:
                        validation_issue_counts[code] += 1
        envelope = payload.get("envelope")
        choice = envelope.get("choice") if isinstance(envelope, dict) else None
        choice_kind = choice.get("kind") if isinstance(choice, dict) else None
        if choice_kind in {"action", "skip"}:
            prefix = "accepted" if valid else "rejected"
            facts[f"{prefix}_{choice_kind}"] += 1
        arguments = choice.get("arguments") if isinstance(choice, dict) else None
        if isinstance(arguments, dict) and any(
            key in arguments for key in ("target", "target_id", "target_seat")
        ):
            facts["target_argument"] += 1

    request_ids_valid = facts["request"] == sum(requests.values())
    terminal_ids_valid = facts["terminal"] == sum(terminals.values())
    pairing_valid = bool(requests) and request_ids_valid and terminal_ids_valid
    pairing_valid = pairing_valid and not (set(terminals) - set(requests))
    pairing_valid = pairing_valid and all(count == 1 for count in requests.values())
    pairing_valid = pairing_valid and all(
        terminals.get(request_id, 0) == 1 for request_id in requests
    )
    pairing_valid = pairing_valid and all(
        request_positions[request_id][0] < terminal_positions[request_id][0]
        for request_id in requests
        if request_id in terminal_positions
    )
    return _TraceFacts(
        request_count=facts["request"],
        terminal_count=facts["terminal"],
        pairing_valid=pairing_valid,
        terminal_counts=dict(terminal_counts),
        validation_issue_counts=dict(validation_issue_counts),
        terminal_error_type_counts=dict(terminal_error_type_counts),
        fact_counts={subject: facts[subject] for subject in sorted(_FACT_COUNT_SUBJECTS)},
        observation_keys=frozenset(observation_keys),
    )


def _evaluate_invariant(
    invariant: ExpectedInvariant,
    *,
    result: EnvironmentRunResult,
    facts: _TraceFacts,
    marker_match_count: int,
    redaction_clean: bool,
) -> InvariantResult:
    kind = invariant.kind
    subject = invariant.subject
    if kind == "request_terminal_pairing":
        observed: Any = facts.pairing_valid
    elif kind == "minimum_request_count":
        observed = facts.request_count
        return _invariant_result(invariant, observed, observed >= invariant.expected)
    elif kind == "run_status":
        observed = result.status
    elif kind == "error_type":
        observed = result.error_type
    elif kind == "outcome_equals":
        observed = result.outcome
    elif kind == "terminal_kind_count":
        observed = facts.terminal_counts.get(str(subject), 0)
    elif kind == "terminal_error_type_present":
        observed = facts.terminal_error_type_counts.get(str(subject), 0) > 0
    elif kind == "validation_issue_present":
        observed = facts.validation_issue_counts.get(str(subject), 0) > 0
    elif kind == "fact_count":
        observed = facts.fact_counts.get(str(subject), 0)
    elif kind == "request_observation_key_absent":
        observed = str(subject) not in facts.observation_keys
    elif kind == "no_fabricated_choice":
        observed = facts.fact_counts.get("fabricated_choice_terminal", 0) == 0
    elif kind == "secret_absence":
        observed = {
            "redaction_clean": redaction_clean,
            "marker_match_count": marker_match_count,
        }
        return _invariant_result(
            invariant,
            observed,
            redaction_clean and marker_match_count == 0,
        )
    else:  # pragma: no cover - Literal and Pydantic guard this boundary
        raise AssertionError(f"unsupported invariant kind: {kind}")
    return _invariant_result(invariant, observed, observed == invariant.expected)


def _invariant_result(
    invariant: ExpectedInvariant,
    observed: Any,
    passed: bool,
) -> InvariantResult:
    return InvariantResult(
        kind=invariant.kind,
        subject=invariant.subject,
        passed=bool(passed),
        expected=redact_sensitive(invariant.expected),
        observed=redact_sensitive(observed),
    )


def _normalize_sensitive_markers(values: Sequence[str]) -> tuple[str, ...]:
    markers: list[str] = []
    for value in values:
        marker = str(value)
        if len(marker) < 4:
            raise ValueError("sensitive markers must contain at least four characters")
        if marker not in markers:
            markers.append(marker)
    return tuple(markers)


def _redact_sensitive_markers(value: Any, markers: Sequence[str]) -> Any:
    """Remove caller-known opaque secrets that generic patterns cannot know."""
    if isinstance(value, dict):
        return {
            _redact_marker_text(str(key), markers): _redact_sensitive_markers(
                item, markers
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_markers(item, markers) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_sensitive_markers(item, markers) for item in value)
    if isinstance(value, str):
        return _redact_marker_text(value, markers)
    return value


def _redact_marker_text(value: str, markers: Sequence[str]) -> str:
    safe = value
    for marker in markers:
        safe = safe.replace(marker, "[redacted]")
    return safe


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
