"""Fail-closed verification for completed real-model smoke-run artifacts.

This module is deliberately offline.  It verifies evidence that a run already
produced; it never resolves credentials, constructs a router, or invokes an
Agent.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import load_verified_artifact_snapshot
from .transcript import redact_sensitive


REAL_MODEL_SMOKE_REPORT_VERSION = "agent-harness.real-model-smoke-report.v1"

_REDACTED = "[redacted]"
_TERMINAL_KINDS = (
    "agent_response",
    "agent_response_failed",
    "agent_response_cancelled",
    "agent_response_validation_failed",
)
_MODEL_CALL_METRIC_PATHS = (
    ("router_stats_delta", "calls"),
    ("metrics", "router_stats_delta", "calls"),
    ("metrics", "model_calls"),
    ("metrics", "model_call_count"),
)
_ROUTER_STATS_DELTA_PATHS = (
    ("router_stats_delta",),
    ("metrics", "router_stats_delta"),
)
_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "secret",
    "password",
    "x_api_key",
    "x_room_token",
    "admin_token",
    "seat_token",
    "seat_tokens",
    "access_token",
    "refresh_token",
)
_BEARER_CREDENTIAL_RE = re.compile(
    r"(?i)\bBearer[ \t]+[A-Za-z0-9._~+/=-]{4,}"
)
_SK_CREDENTIAL_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{8,}"
)
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")


class RealModelSmokeVerificationError(ValueError):
    """Raised when real-model smoke evidence is missing or contradictory."""


class RealModelSmokeReport(BaseModel):
    """Safe, versioned proof summary with no prompts, responses, or secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "agent-harness.real-model-smoke-report.v1"
    ] = REAL_MODEL_SMOKE_REPORT_VERSION
    status: Literal["passed"] = "passed"
    run_id: str = Field(min_length=1)
    transcript_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_manifest_schema_version: str = Field(min_length=1)
    model_calls: int = Field(gt=0)
    model_call_metric_sources: list[str] = Field(min_length=1)
    request_count: int = Field(gt=0)
    terminal_response_count: int = Field(gt=0)
    response_count: int = Field(gt=0)
    response_failure_count: int = Field(ge=0)
    response_cancelled_count: int = Field(ge=0)
    response_validation_failure_count: int = Field(ge=0)
    model_call_id_count: int = Field(gt=0)
    decision_consumed_count: int = Field(gt=0)
    model_backed_decision_count: int = Field(gt=0)
    model_generation_count: int = Field(gt=0)
    tool_call_count: int = Field(gt=0)
    tool_result_count: int = Field(gt=0)
    terminal_action_count: int = Field(gt=0)
    environment_consumed_action_count: int = Field(gt=0)


def verify_real_model_smoke_artifacts(
    run_dir: str | Path,
) -> RealModelSmokeReport:
    """Verify one committed run directory as factual real-model smoke evidence.

    Artifact integrity is checked into one in-memory snapshot before any
    semantic evidence is trusted. A report is returned only when every
    invariant passes; partial or ambiguous evidence raises
    :class:`RealModelSmokeVerificationError`.
    """
    snapshot = load_verified_artifact_snapshot(run_dir)
    manifest = snapshot.manifest
    raw_manifest = _load_json_object_bytes(
        snapshot.manifest_bytes,
        label="manifest",
    )
    summary = _load_json_object_bytes(
        snapshot.content_bytes["summary"],
        label="summary",
    )
    transcript_rows = _load_jsonl_objects_bytes(
        snapshot.content_bytes["transcript_jsonl"],
        label="transcript",
    )

    _verify_credential_redaction(
        {
            "manifest": raw_manifest,
            "summary": summary,
            "transcript": transcript_rows,
        }
    )

    if summary.get("status") != "completed":
        raise RealModelSmokeVerificationError(
            "real-model smoke run status must be completed"
        )
    model_calls, metric_sources = _model_call_count(summary)
    evidence = _verify_decision_evidence(
        transcript_rows,
        model_calls=model_calls,
        summary=summary,
    )

    return RealModelSmokeReport(
        run_id=manifest.run.run_id,
        transcript_digest=manifest.transcript_digest,
        artifact_manifest_schema_version=manifest.schema_version,
        model_calls=model_calls,
        model_call_metric_sources=metric_sources,
        request_count=evidence.request_count,
        terminal_response_count=evidence.terminal_response_count,
        response_count=evidence.response_count,
        response_failure_count=evidence.response_failure_count,
        response_cancelled_count=evidence.response_cancelled_count,
        response_validation_failure_count=(
            evidence.response_validation_failure_count
        ),
        model_call_id_count=evidence.model_call_id_count,
        decision_consumed_count=evidence.decision_consumed_count,
        model_backed_decision_count=evidence.model_backed_decision_count,
        model_generation_count=evidence.model_generation_count,
        tool_call_count=evidence.tool_call_count,
        tool_result_count=evidence.tool_result_count,
        terminal_action_count=evidence.terminal_action_count,
        environment_consumed_action_count=evidence.environment_consumed_action_count,
    )


class _DecisionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_count: int
    terminal_response_count: int
    response_count: int
    response_failure_count: int
    response_cancelled_count: int
    response_validation_failure_count: int
    model_call_id_count: int
    decision_consumed_count: int
    model_backed_decision_count: int
    model_generation_count: int
    tool_call_count: int
    tool_result_count: int
    terminal_action_count: int
    environment_consumed_action_count: int


class _ToolLoopEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_generation_count: int
    tool_call_count: int
    tool_result_count: int
    terminal_action_count: int
    environment_consumed_action_count: int


def _verify_decision_evidence(
    rows: list[dict[str, Any]],
    *,
    model_calls: int,
    summary: Mapping[str, Any],
) -> _DecisionEvidence:
    requests: Counter[str] = Counter()
    terminals: Counter[str] = Counter()
    terminal_kinds: dict[str, str] = {}
    responses: dict[str, dict[str, Any]] = {}
    consumed: Counter[str] = Counter()
    consumed_payloads: dict[str, dict[str, Any]] = {}

    for row in rows:
        if row.get("kind") != "decision":
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise RealModelSmokeVerificationError(
                "decision transcript row payload must be an object"
            )
        trace_kind = payload.get("kind")
        if trace_kind == "agent_request":
            request = payload.get("request")
            request_id = request.get("request_id") if isinstance(request, dict) else None
            request_id = _required_request_id(request_id, "agent_request")
            requests[request_id] += 1
        elif trace_kind in _TERMINAL_KINDS:
            request_id = _required_request_id(
                payload.get("request_id"), str(trace_kind)
            )
            terminals[request_id] += 1
            terminal_kinds.setdefault(request_id, str(trace_kind))
            if trace_kind == "agent_response":
                responses.setdefault(request_id, payload)

        if payload.get("type") == "decision_consumed":
            request_id = _required_request_id(
                payload.get("request_id"), "decision_consumed"
            )
            consumed[request_id] += 1
            consumed_payloads.setdefault(request_id, payload)

    if not requests:
        raise RealModelSmokeVerificationError(
            "real-model smoke evidence has no agent_request rows"
        )
    duplicate_requests = sorted(
        request_id for request_id, count in requests.items() if count != 1
    )
    if duplicate_requests:
        raise RealModelSmokeVerificationError(
            "duplicate agent_request request_id values are forbidden"
        )

    orphan_terminals = sorted(set(terminals) - set(requests))
    if orphan_terminals:
        raise RealModelSmokeVerificationError(
            "orphan agent response terminals are forbidden"
        )
    invalid_terminal_counts = sorted(
        request_id
        for request_id in requests
        if terminals.get(request_id, 0) != 1
    )
    if invalid_terminal_counts:
        raise RealModelSmokeVerificationError(
            "every agent_request must have exactly one response terminal"
        )

    if not responses:
        raise RealModelSmokeVerificationError(
            "real-model smoke evidence has no agent_response rows"
        )
    duplicate_consumed = sorted(
        request_id for request_id, count in consumed.items() if count != 1
    )
    if duplicate_consumed:
        raise RealModelSmokeVerificationError(
            "duplicate decision_consumed request_id values are forbidden"
        )
    orphan_consumed = sorted(set(consumed) - set(requests))
    if orphan_consumed:
        raise RealModelSmokeVerificationError(
            "orphan decision_consumed rows are forbidden"
        )
    nonresponse_consumed = sorted(
        request_id
        for request_id in consumed
        if terminal_kinds.get(request_id) != "agent_response"
    )
    if nonresponse_consumed:
        raise RealModelSmokeVerificationError(
            "decision_consumed must reference an agent_response terminal"
        )
    if not consumed:
        raise RealModelSmokeVerificationError(
            "real-model smoke evidence has no decision_consumed rows"
        )
    invalid_consumed = sorted(
        request_id
        for request_id in consumed
        if not isinstance(responses[request_id].get("validation"), dict)
        or responses[request_id]["validation"].get("valid") is not True
    )
    if invalid_consumed:
        raise RealModelSmokeVerificationError(
            "decision_consumed must reference a valid agent_response"
        )

    response_call_ids: dict[str, str] = {}
    for request_id, response in responses.items():
        envelope = response.get("envelope")
        if not isinstance(envelope, dict):
            raise RealModelSmokeVerificationError(
                "agent_response envelope must be an object"
            )
        model_call_id = envelope.get("model_call_id")
        if (
            isinstance(model_call_id, str)
            and model_call_id.strip()
            and model_call_id.strip() != _REDACTED
        ):
            response_call_ids[request_id] = model_call_id.strip()
    if not response_call_ids:
        raise RealModelSmokeVerificationError(
            "real-model smoke evidence has no model_call_id"
        )
    unique_call_ids = set(response_call_ids.values())
    if len(unique_call_ids) != len(response_call_ids):
        raise RealModelSmokeVerificationError(
            "model_call_id values must be unique across agent responses"
        )
    if model_calls < len(unique_call_ids):
        raise RealModelSmokeVerificationError(
            "model-call metric is smaller than transcript model_call_id evidence"
        )

    attempt_call_ids = _verify_nested_attempt_provenance(
        rows,
        request_ids=set(requests),
        response_call_ids=response_call_ids,
        consumed_payloads=consumed_payloads,
        model_calls=model_calls,
        summary=summary,
    )
    tool_evidence = _verify_tool_loop_evidence(
        rows,
        request_ids=set(requests),
        model_backed_request_ids=set(response_call_ids),
        consumed_payloads=consumed_payloads,
    )
    verified_call_id_count = len(attempt_call_ids)

    model_backed_decisions = set(consumed) & set(response_call_ids)
    if not model_backed_decisions:
        raise RealModelSmokeVerificationError(
            "no decision_consumed row is backed by a model_call_id response"
        )

    by_kind = Counter(terminal_kinds.values())
    return _DecisionEvidence(
        request_count=sum(requests.values()),
        terminal_response_count=sum(terminals.values()),
        response_count=by_kind["agent_response"],
        response_failure_count=by_kind["agent_response_failed"],
        response_cancelled_count=by_kind["agent_response_cancelled"],
        response_validation_failure_count=by_kind[
            "agent_response_validation_failed"
        ],
        model_call_id_count=verified_call_id_count,
        decision_consumed_count=sum(consumed.values()),
        model_backed_decision_count=len(model_backed_decisions),
        model_generation_count=tool_evidence.model_generation_count,
        tool_call_count=tool_evidence.tool_call_count,
        tool_result_count=tool_evidence.tool_result_count,
        terminal_action_count=tool_evidence.terminal_action_count,
        environment_consumed_action_count=tool_evidence.environment_consumed_action_count,
    )


def _verify_tool_loop_evidence(
    rows: list[dict[str, Any]],
    *,
    request_ids: set[str],
    model_backed_request_ids: set[str],
    consumed_payloads: Mapping[str, Mapping[str, Any]],
) -> _ToolLoopEvidence:
    """Require a real model-backed request to complete a visible tool loop.

    A nonzero provider counter alone is insufficient evidence for an Agent
    harness.  The transcript must show a model generation selecting a tool,
    the matching tool result, a successful terminal tool, and the environment
    consuming that terminal decision.  All rows are admin-only decision rows;
    this verifier never trusts raw prompts or model self-reports.
    """
    turn_started: set[str] = set()
    requested: dict[str, dict[str, int]] = {}
    results: dict[str, dict[str, int]] = {}
    terminal_results: dict[str, set[str]] = {}
    submitted: dict[str, set[str]] = {}
    rules_statuses: dict[str, set[str]] = {}
    model_generations: dict[str, int] = {}

    for row in rows:
        if row.get("kind") != "decision":
            continue
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            continue
        event_type = str(payload.get("type") or "")
        if event_type in {
            "agent_turn_started",
            "model_generation",
            "tool_call_requested",
            "tool_result",
            "agent_action_submitted",
            "agent_turn_failed",
        }:
            request_id = _required_request_id(
                payload.get("request_id"), event_type
            )
            if request_id not in request_ids:
                raise RealModelSmokeVerificationError(
                    f"{event_type} references no agent_request"
                )
            if event_type == "agent_turn_started":
                turn_started.add(request_id)
            elif event_type == "model_generation":
                model_generations[request_id] = model_generations.get(request_id, 0) + 1
            elif event_type == "tool_call_requested":
                call_id = _required_call_id(payload.get("call_id"))
                requested.setdefault(request_id, {})[call_id] = (
                    requested.setdefault(request_id, {}).get(call_id, 0) + 1
                )
            elif event_type == "tool_result":
                call_id = _required_call_id(payload.get("call_id"))
                results.setdefault(request_id, {})[call_id] = (
                    results.setdefault(request_id, {}).get(call_id, 0) + 1
                )
                if bool(payload.get("terminal")) and bool(payload.get("ok")):
                    terminal_results.setdefault(request_id, set()).add(call_id)
            elif event_type == "agent_action_submitted":
                call_id = _required_call_id(payload.get("call_id"))
                submitted.setdefault(request_id, set()).add(call_id)
            continue
        if event_type == "rules_result":
            request_id = _required_request_id(payload.get("request_id"), "rules_result")
            if request_id not in request_ids:
                raise RealModelSmokeVerificationError("rules_result references no agent_request")
            rules = payload.get("rules")
            if not isinstance(rules, Mapping):
                raise RealModelSmokeVerificationError("rules_result.rules must be an object")
            rules_statuses.setdefault(request_id, set()).add(str(rules.get("status") or ""))

    if not turn_started:
        raise RealModelSmokeVerificationError(
            "real-model smoke evidence has no agent_turn_started tool-loop row"
        )
    missing_turns = sorted(model_backed_request_ids - turn_started)
    if missing_turns:
        raise RealModelSmokeVerificationError(
            "every model-backed request must have an agent_turn_started row"
        )

    for request_id in sorted(model_backed_request_ids):
        if model_generations.get(request_id, 0) < 1:
            raise RealModelSmokeVerificationError(
                "tool-loop request has no model_generation row"
            )
        calls = requested.get(request_id, {})
        tool_results = results.get(request_id, {})
        if not calls:
            raise RealModelSmokeVerificationError(
                "tool-loop request has no tool_call_requested row"
            )
        if not tool_results:
            raise RealModelSmokeVerificationError(
                "tool-loop request has no tool_result row"
            )
        if any(count != 1 for count in calls.values()):
            raise RealModelSmokeVerificationError(
                "tool_call_requested call_id values must be unique per request"
            )
        if any(count != 1 for count in tool_results.values()):
            raise RealModelSmokeVerificationError(
                "tool_result call_id values must be unique per request"
            )
        orphan_results = set(tool_results) - set(calls)
        if orphan_results:
            raise RealModelSmokeVerificationError(
                "tool_result has no matching tool_call_requested row"
            )
        terminal_calls = terminal_results.get(request_id, set())
        if not terminal_calls:
            raise RealModelSmokeVerificationError(
                "tool-loop request has no successful terminal tool result"
            )
        submitted_calls = submitted.get(request_id, set())
        if not terminal_calls & submitted_calls:
            raise RealModelSmokeVerificationError(
                "terminal tool result has no matching agent_action_submitted row"
            )
        if request_id not in consumed_payloads:
            raise RealModelSmokeVerificationError(
                "tool-loop terminal action was not consumed by the environment"
            )

    consumed_rule_count = sum(
        1
        for statuses in rules_statuses.values()
        if statuses & {"accepted", "skipped"}
    )
    if consumed_rule_count < 1:
        raise RealModelSmokeVerificationError(
            "tool-loop evidence has no environment-consumed rules_result"
        )
    return _ToolLoopEvidence(
        model_generation_count=sum(model_generations.values()),
        tool_call_count=sum(sum(items.values()) for items in requested.values()),
        tool_result_count=sum(sum(items.values()) for items in results.values()),
        terminal_action_count=sum(len(items) for items in terminal_results.values()),
        environment_consumed_action_count=consumed_rule_count,
    )


def _required_request_id(value: Any, row_kind: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
    ):
        raise RealModelSmokeVerificationError(
            f"{row_kind} row has no valid request_id"
        )
    return value


def _verify_nested_attempt_provenance(
    rows: list[dict[str, Any]],
    *,
    request_ids: set[str],
    response_call_ids: Mapping[str, str],
    consumed_payloads: Mapping[str, Mapping[str, Any]],
    model_calls: int,
    summary: Mapping[str, Any],
) -> set[str]:
    """Verify complete, request-linked evidence for every logical Router call.

    This verifier is a release gate, not a compatibility reader.  Falling back
    to final-envelope IDs when nested attempts are absent would let an artifact
    delete the stronger proof fields and masquerade as an older run.
    """
    containers: list[tuple[str, Any, str, str, Mapping[str, Any] | None]] = []

    for row in rows:
        if row.get("kind") != "decision":
            continue
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if payload.get("type") == "decision_consumed":
            request_id = _required_request_id(
                payload.get("request_id"), "decision_consumed"
            )
            llm_call = payload.get("llm_call")
            if isinstance(llm_call, Mapping) and "actor_response_attempts" in llm_call:
                containers.append((
                    request_id,
                    llm_call.get("actor_response_attempts"),
                    "decision_consumed.llm_call.actor_response_attempts",
                    "accepted",
                    llm_call,
                ))

        if payload.get("kind") == "agent_response_failed":
            request_id = _required_request_id(
                payload.get("request_id"), "agent_response_failed"
            )
            failure = payload.get("failure")
            if isinstance(failure, Mapping) and "llm_call_attempts" in failure:
                containers.append((
                    request_id,
                    failure.get("llm_call_attempts"),
                    "agent_response_failed.failure.llm_call_attempts",
                    "failed",
                    None,
                ))

    model_backed_request_ids = set(response_call_ids)
    missing_consumed = sorted(model_backed_request_ids - set(consumed_payloads))
    if missing_consumed:
        raise RealModelSmokeVerificationError(
            "every model-backed agent_response must be decision_consumed"
        )
    for request_id in model_backed_request_ids:
        consumed_payload = consumed_payloads[request_id]
        llm_call = consumed_payload.get("llm_call")
        if not isinstance(llm_call, Mapping) or "actor_response_attempts" not in llm_call:
            raise RealModelSmokeVerificationError(
                "complete nested attempt provenance is required for every "
                "model-backed decision"
            )
    if not containers:
        raise RealModelSmokeVerificationError(
            "complete nested attempt provenance is required for real-model smoke evidence"
        )

    call_ids: set[str] = set()
    call_owners: dict[str, str] = {}
    accepted_final_call_ids: dict[str, str] = {}
    container_owners: set[str] = set()
    input_tokens = 0
    output_tokens = 0
    latency_seconds = 0.0
    attempt_count = 0

    for owner_request_id, raw_attempts, label, outcome, outer_llm_call in containers:
        if owner_request_id in container_owners:
            raise RealModelSmokeVerificationError(
                "a request must have exactly one nested attempt container"
            )
        container_owners.add(owner_request_id)
        if not isinstance(raw_attempts, list) or not raw_attempts:
            raise RealModelSmokeVerificationError(
                f"{label} must be a non-empty array"
            )
        if outcome == "accepted":
            assert outer_llm_call is not None
            recorded_count = _positive_evidence_integer(
                outer_llm_call.get("actor_response_attempt_count", _MISSING),
                label="decision_consumed.llm_call.actor_response_attempt_count",
            )
            if recorded_count != len(raw_attempts):
                raise RealModelSmokeVerificationError(
                    "actor_response_attempt_count does not match nested attempt count"
                )

        for expected_attempt, raw_attempt in enumerate(raw_attempts, start=1):
            if not isinstance(raw_attempt, Mapping):
                raise RealModelSmokeVerificationError(
                    f"{label} entries must be objects"
                )
            recorded_attempt = _positive_evidence_integer(
                raw_attempt.get("attempt", _MISSING),
                label="nested model-call attempt.attempt",
            )
            if recorded_attempt != expected_attempt:
                raise RealModelSmokeVerificationError(
                    "nested model-call attempt numbers must be contiguous and one-based"
                )
            status = raw_attempt.get("status")
            if (
                not isinstance(status, str)
                or not status.strip()
                or status != status.strip()
            ):
                raise RealModelSmokeVerificationError(
                    "nested model-call attempt has no valid status"
                )
            is_last = expected_attempt == len(raw_attempts)
            if outcome == "accepted":
                if (status == "accepted") != is_last:
                    raise RealModelSmokeVerificationError(
                        "an accepted decision must end with exactly one accepted attempt"
                    )
            elif status == "accepted":
                raise RealModelSmokeVerificationError(
                    "a failed decision cannot contain an accepted attempt"
                )

            llm_call = raw_attempt.get("llm_call")
            if not isinstance(llm_call, Mapping):
                raise RealModelSmokeVerificationError(
                    "nested model-call attempt has no llm_call object"
                )
            call_id = _required_call_id(llm_call.get("call_id"))
            if call_id in call_ids:
                raise RealModelSmokeVerificationError(
                    "nested model-call attempt call_id values must be unique"
                )
            call_ids.add(call_id)
            call_owners[call_id] = owner_request_id

            context = llm_call.get("context")
            if not isinstance(context, Mapping):
                raise RealModelSmokeVerificationError(
                    "nested model-call attempt has no context object"
                )
            context_request_id = _required_request_id(
                context.get("request_id"), "nested model-call context"
            )
            if context_request_id not in request_ids:
                raise RealModelSmokeVerificationError(
                    "nested model-call context references no agent_request"
                )
            if context_request_id != owner_request_id:
                raise RealModelSmokeVerificationError(
                    "nested model-call context does not match its owning request"
                )

            usage = llm_call.get("usage")
            if not isinstance(usage, Mapping):
                raise RealModelSmokeVerificationError(
                    "nested model-call attempt usage must be an object"
                )
            input_tokens += _nonnegative_evidence_integer(
                usage.get("prompt_tokens", 0),
                label="nested usage.prompt_tokens",
            )
            output_tokens += _nonnegative_evidence_integer(
                usage.get("completion_tokens", 0),
                label="nested usage.completion_tokens",
            )
            latency_seconds += _nonnegative_number(
                llm_call.get("latency", _MISSING),
                label="nested llm_call.latency",
            )
            attempt_count += 1

            if outcome == "accepted" and is_last:
                assert outer_llm_call is not None
                outer_call_id = _required_call_id(outer_llm_call.get("call_id"))
                if outer_call_id != call_id:
                    raise RealModelSmokeVerificationError(
                        "decision_consumed llm_call.call_id does not match its "
                        "accepted attempt"
                    )
                consumed_call_id = _required_call_id(
                    consumed_payloads[owner_request_id].get("model_call_id")
                )
                if consumed_call_id != call_id:
                    raise RealModelSmokeVerificationError(
                        "decision_consumed model_call_id does not match its accepted attempt"
                    )
                accepted_final_call_ids[owner_request_id] = call_id

    for request_id, response_call_id in response_call_ids.items():
        owner_request_id = call_owners.get(response_call_id)
        if owner_request_id is None:
            raise RealModelSmokeVerificationError(
                "agent_response model_call_id is absent from nested attempt provenance"
            )
        if owner_request_id != request_id:
            raise RealModelSmokeVerificationError(
                "agent_response model_call_id belongs to another request"
            )
        if accepted_final_call_ids.get(request_id) != response_call_id:
            raise RealModelSmokeVerificationError(
                "agent_response model_call_id does not match its accepted attempt"
            )
    if set(accepted_final_call_ids) != set(response_call_ids):
        raise RealModelSmokeVerificationError(
            "every accepted nested attempt must match an agent_response model_call_id"
        )

    router_calls, router_input, router_output, router_latency = (
        _router_stats_delta_totals(summary)
    )
    if router_calls != model_calls or attempt_count != router_calls:
        raise RealModelSmokeVerificationError(
            "nested model-call count does not match router_stats_delta.calls"
        )
    if input_tokens != router_input:
        raise RealModelSmokeVerificationError(
            "nested input-token total does not match router_stats_delta.total_tokens_in"
        )
    if output_tokens != router_output:
        raise RealModelSmokeVerificationError(
            "nested output-token total does not match router_stats_delta.total_tokens_out"
        )

    # Each call latency and the Router snapshot are rounded independently. The
    # tolerance covers their cumulative millisecond-scale rounding only.
    latency_tolerance = (0.0005 * (attempt_count + 2)) + 1e-9
    if not math.isclose(
        latency_seconds,
        router_latency,
        rel_tol=0.0,
        abs_tol=latency_tolerance,
    ):
        raise RealModelSmokeVerificationError(
            "nested latency total does not match router_stats_delta.total_latency"
        )
    return call_ids


def _is_present_call_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value.strip() != _REDACTED
    )


def _required_call_id(value: Any) -> str:
    if not _is_present_call_id(value) or value != value.strip():
        raise RealModelSmokeVerificationError(
            "nested model-call attempt has no valid call_id"
        )
    return value


def _router_stats_delta_totals(
    summary: Mapping[str, Any],
) -> tuple[int, int, int, float]:
    found: list[tuple[str, tuple[int, int, int, float]]] = []
    for path in _ROUTER_STATS_DELTA_PATHS:
        raw = _nested_value(summary, path)
        if raw is _MISSING:
            continue
        label = ".".join(path)
        if not isinstance(raw, Mapping):
            raise RealModelSmokeVerificationError(
                f"summary {label} must be an object"
            )
        found.append((label, (
            _nonnegative_evidence_integer(
                raw.get("calls", _MISSING), label=f"{label}.calls"
            ),
            _nonnegative_evidence_integer(
                raw.get("total_tokens_in", _MISSING),
                label=f"{label}.total_tokens_in",
            ),
            _nonnegative_evidence_integer(
                raw.get("total_tokens_out", _MISSING),
                label=f"{label}.total_tokens_out",
            ),
            _nonnegative_number(
                raw.get("total_latency", _MISSING),
                label=f"{label}.total_latency",
            ),
        )))
    if not found:
        raise RealModelSmokeVerificationError(
            "nested attempt provenance requires router_stats_delta totals"
        )
    baseline = found[0][1]
    for _label, current in found[1:]:
        if current[:3] != baseline[:3] or not math.isclose(
            current[3], baseline[3], rel_tol=0.0, abs_tol=1e-9
        ):
            raise RealModelSmokeVerificationError(
                "router_stats_delta metric sources disagree"
            )
    return baseline


def _nonnegative_evidence_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RealModelSmokeVerificationError(
            f"{label} must be a non-negative integer"
        )
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise RealModelSmokeVerificationError(
            f"{label} must be a non-negative integer"
        )
    return int(number)


def _positive_evidence_integer(value: Any, *, label: str) -> int:
    number = _nonnegative_evidence_integer(value, label=label)
    if number <= 0:
        raise RealModelSmokeVerificationError(
            f"{label} must be a positive integer"
        )
    return number


def _nonnegative_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RealModelSmokeVerificationError(
            f"{label} must be a finite non-negative number"
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise RealModelSmokeVerificationError(
            f"{label} must be a finite non-negative number"
        )
    return number


def _model_call_count(summary: Mapping[str, Any]) -> tuple[int, list[str]]:
    found: list[tuple[str, int]] = []
    for path in _MODEL_CALL_METRIC_PATHS:
        value = _nested_value(summary, path)
        if value is _MISSING:
            continue
        label = ".".join(path)
        found.append((label, _nonnegative_integer(value, label=label)))
    if not found:
        supported = ", ".join(".".join(path) for path in _MODEL_CALL_METRIC_PATHS)
        raise RealModelSmokeVerificationError(
            f"summary has no supported model-call metric ({supported})"
        )
    values = {count for _source, count in found}
    if len(values) != 1:
        raise RealModelSmokeVerificationError(
            "supported model-call metrics disagree"
        )
    calls = values.pop()
    if calls <= 0:
        raise RealModelSmokeVerificationError(
            "real-model smoke run must record at least one model call"
        )
    return calls, [source for source, _count in found]


_MISSING = object()


def _nested_value(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RealModelSmokeVerificationError(
            f"model-call metric {label} must be a non-negative integer"
        )
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise RealModelSmokeVerificationError(
            f"model-call metric {label} must be a non-negative integer"
        )
    return int(number)


def _verify_credential_redaction(value: Any, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            current = f"{path}.{key}"
            _reject_obvious_credential(key, current)
            if _is_sensitive_key(key):
                if item != _REDACTED:
                    raise RealModelSmokeVerificationError(
                        f"structured credential field is not redacted at {current}"
                    )
                continue
            _verify_credential_redaction(item, current)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _verify_credential_redaction(item, f"{path}[{index}]")
        return
    if isinstance(value, str):
        _reject_obvious_credential(value, path)


def _is_sensitive_key(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _reject_obvious_credential(value: str, path: str) -> None:
    if _BEARER_CREDENTIAL_RE.search(value) or _SK_CREDENTIAL_RE.search(value):
        raise RealModelSmokeVerificationError(
            f"obvious credential token is forbidden at {path}"
        )
    for match in _URL_RE.finditer(value):
        try:
            parsed = urlsplit(match.group(0))
        except ValueError:
            continue
        if parsed.username is not None or parsed.password is not None:
            raise RealModelSmokeVerificationError(
                f"URL credentials are forbidden at {path}"
            )
        if any(
            _is_sensitive_key(key) and item != _REDACTED
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        ):
            raise RealModelSmokeVerificationError(
                f"URL credential parameters are forbidden at {path}"
            )


def _load_json_object_bytes(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        raw = content.decode("utf-8")
        _reject_obvious_credential(raw, f"artifact.{label}")
        value = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (UnicodeError, json.JSONDecodeError, _DuplicateJsonKey) as err:
        raise RealModelSmokeVerificationError(
            f"{label} could not be read after artifact verification"
        ) from err
    if not isinstance(value, dict):
        raise RealModelSmokeVerificationError(f"{label} must be a JSON object")
    return value


def _load_jsonl_objects_bytes(content: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        raw = content.decode("utf-8")
        _reject_obvious_credential(raw, f"artifact.{label}")
        rows: list[dict[str, Any]] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            value = json.loads(line, object_pairs_hook=_unique_json_object)
            if not isinstance(value, dict):
                raise _JsonStructureError("JSONL row is not an object")
            rows.append(value)
        return rows
    except (
        UnicodeError,
        json.JSONDecodeError,
        _JsonStructureError,
    ) as err:
        raise RealModelSmokeVerificationError(
            f"{label} could not be read after artifact verification"
        ) from err


class _JsonStructureError(ValueError):
    pass


class _DuplicateJsonKey(_JsonStructureError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKey("duplicate JSON object key")
        value[key] = item
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify an existing real-model smoke-run artifact directory."
    )
    parser.add_argument("run_dir", help="Directory containing the three run artifacts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_real_model_smoke_artifacts(args.run_dir)
    except Exception as err:  # noqa: BLE001 - CLI is a fail-closed boundary
        safe_error = str(redact_sensitive(str(err) or type(err).__name__))
        failure = {
            "schema_version": REAL_MODEL_SMOKE_REPORT_VERSION,
            "status": "failed",
            "error_type": type(err).__name__,
            "error": safe_error,
        }
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
