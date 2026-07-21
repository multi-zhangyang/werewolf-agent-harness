"""Environment-neutral decision protocol for the reusable harness core."""
from __future__ import annotations

import json
import math
import time
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CORE_AGENT_PROTOCOL_VERSION = "agent-harness.decision.v1"


class ActionOption(BaseModel):
    """One action advertised by an environment with its complete input schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    input_schema: dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "additionalProperties": False,
    })
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _nonempty_name(cls, value: str) -> str:
        name = str(value).strip()
        if not name:
            raise ValueError("action option name must not be empty")
        return name

    @field_validator("input_schema")
    @classmethod
    def _valid_object_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_json_value(value, label="action input_schema")
        try:
            Draft202012Validator.check_schema(value)
        except (SchemaError, TypeError, ValueError) as err:
            message = getattr(err, "message", str(err))
            raise ValueError(f"invalid action input_schema: {message}") from err
        schema_type = value.get("type")
        if schema_type is not None and not (
            schema_type == "object"
            or isinstance(schema_type, list) and "object" in schema_type
        ):
            raise ValueError("action input_schema must describe an object")
        return value

    @field_validator("metadata")
    @classmethod
    def _json_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_json_value(value, label="action metadata")
        return value


class SkipPolicy(BaseModel):
    """Explicit environment policy for a deliberate no-action decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed: bool = Field(default=False, strict=True)
    reason_required: bool = Field(default=True, strict=True)


class ActionRequest(BaseModel):
    """One immutable, environment-neutral request addressed to an actor ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal[CORE_AGENT_PROTOCOL_VERSION] = CORE_AGENT_PROTOCOL_VERSION
    request_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    observation: dict[str, Any] = Field(default_factory=dict)
    legal_actions: list[ActionOption] = Field(default_factory=list)
    skip_policy: SkipPolicy = Field(default_factory=SkipPolicy)
    deadline_monotonic: float | None = None
    labels: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "run_id", "actor_id")
    @classmethod
    def _nonempty_identity(cls, value: str) -> str:
        identity = str(value).strip()
        if not identity:
            raise ValueError("request/run/actor identity must not be empty")
        return identity

    @field_validator("observation", "metadata")
    @classmethod
    def _json_objects(cls, value: dict[str, Any], info: Any) -> dict[str, Any]:
        _require_json_value(value, label=info.field_name)
        return value

    @field_validator("labels")
    @classmethod
    def _finite_labels(
        cls,
        value: dict[str, str | int | float | bool | None],
    ) -> dict[str, str | int | float | bool | None]:
        _require_json_value(value, label="labels")
        return value

    @field_validator("deadline_monotonic")
    @classmethod
    def _finite_deadline(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("deadline_monotonic must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def _request_has_a_terminal_choice(self) -> "ActionRequest":
        names = [option.name for option in self.legal_actions]
        if len(names) != len(set(names)):
            raise ValueError("legal action names must be unique")
        if not names and not self.skip_policy.allowed:
            raise ValueError("request must advertise an action or allow skip")
        return self

    def seconds_remaining(self, now: float | None = None) -> float | None:
        if self.deadline_monotonic is None:
            return None
        current = time.monotonic() if now is None else now
        return max(0.0, self.deadline_monotonic - current)


class ActionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["action"] = "action"
    action: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action")
    @classmethod
    def _nonempty_action(cls, value: str) -> str:
        action = str(value).strip()
        if not action:
            raise ValueError("action choice name must not be empty")
        return action

    @field_validator("arguments")
    @classmethod
    def _json_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_json_value(value, label="action arguments")
        return value


class SkipChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["skip"] = "skip"
    reason: str = ""


DecisionChoice = Annotated[ActionChoice | SkipChoice, Field(discriminator="kind")]


class DecisionEnvelope(BaseModel):
    """One actor choice plus safe call and parse provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal[CORE_AGENT_PROTOCOL_VERSION] = CORE_AGENT_PROTOCOL_VERSION
    request_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    choice: DecisionChoice
    private_reasoning: str | None = None
    latency_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    model_call_id: str | None = None
    prompt_hash: str | None = None
    response_hash: str | None = None
    parse_status: Literal["ok", "recovered", "not_applicable"] = "ok"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id", "actor_id")
    @classmethod
    def _nonempty_identity(cls, value: str) -> str:
        identity = str(value).strip()
        if not identity:
            raise ValueError("request/actor identity must not be empty")
        return identity

    @field_validator("metadata")
    @classmethod
    def _json_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_json_value(value, label="decision metadata")
        return value


class DecisionValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: Literal["error", "warning"] = "error"
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class DecisionValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    issues: list[DecisionValidationIssue] = Field(default_factory=list)


def validate_decision_envelope(
    envelope: DecisionEnvelope,
    request: ActionRequest,
) -> DecisionValidationResult:
    """Validate identity, action membership, skip policy, and JSON arguments."""
    issues: list[DecisionValidationIssue] = []
    if envelope.protocol_version != request.protocol_version:
        issues.append(_issue(
            "protocol_version_mismatch",
            "Decision protocol_version does not match the request.",
            request_protocol_version=request.protocol_version,
            envelope_protocol_version=envelope.protocol_version,
        ))
    if request.protocol_version != CORE_AGENT_PROTOCOL_VERSION:
        issues.append(_issue(
            "unsupported_protocol_version",
            "ActionRequest protocol_version is not supported by the core contract.",
            protocol_version=request.protocol_version,
        ))
    if envelope.protocol_version != CORE_AGENT_PROTOCOL_VERSION:
        issues.append(_issue(
            "unsupported_protocol_version",
            "DecisionEnvelope protocol_version is not supported by the core contract.",
            protocol_version=envelope.protocol_version,
        ))
    if envelope.request_id != request.request_id:
        issues.append(_issue(
            "request_id_mismatch",
            "Decision request_id does not match the request.",
            request_id=request.request_id,
            envelope_request_id=envelope.request_id,
        ))
    if envelope.actor_id != request.actor_id:
        issues.append(_issue(
            "actor_id_mismatch",
            "Decision actor_id does not match the request.",
            request_actor_id=request.actor_id,
            envelope_actor_id=envelope.actor_id,
        ))

    choice = envelope.choice
    try:
        _require_json_value(
            envelope.model_dump(exclude={"private_reasoning"}),
            label="decision envelope",
        )
    except ValueError as err:
        issues.append(_issue(
            "non_json_payload",
            str(err),
        ))
    if envelope.latency_seconds is not None and not math.isfinite(envelope.latency_seconds):
        issues.append(_issue(
            "latency_not_finite",
            "Decision latency_seconds must be finite.",
        ))
    if isinstance(choice, SkipChoice):
        if not request.skip_policy.allowed:
            issues.append(_issue(
                "skip_not_allowed",
                "The environment did not advertise skip as legal.",
            ))
        if request.skip_policy.reason_required and not choice.reason.strip():
            issues.append(_issue(
                "skip_reason_missing",
                "An explicit skip must include a non-empty reason.",
            ))
    else:
        option = next(
            (candidate for candidate in request.legal_actions if candidate.name == choice.action),
            None,
        )
        if option is None:
            issues.append(_issue(
                "action_not_legal",
                "The selected action was not advertised by the environment.",
                action=choice.action,
                legal_actions=[candidate.name for candidate in request.legal_actions],
            ))
        else:
            validator = Draft202012Validator(option.input_schema)
            for error in sorted(validator.iter_errors(choice.arguments), key=_schema_error_key):
                issues.append(_issue(
                    "action_arguments_invalid",
                    error.message,
                    action=choice.action,
                    instance_path=list(error.absolute_path),
                    schema_path=list(error.absolute_schema_path),
                    validator=error.validator,
                ))

    return DecisionValidationResult(
        valid=not any(issue.severity == "error" for issue in issues),
        issues=issues,
    )


def _require_json_value(value: Any, *, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as err:
        raise ValueError(f"{label} must contain only finite JSON values") from err


def _schema_error_key(error: Any) -> tuple[str, str]:
    return ("/".join(str(part) for part in error.absolute_path), str(error.message))


def _issue(code: str, message: str, **evidence: Any) -> DecisionValidationIssue:
    return DecisionValidationIssue(code=code, message=message, evidence=evidence)


@runtime_checkable
class AgentProtocol(Protocol):
    actor_id: str

    async def decide(self, request: ActionRequest) -> DecisionEnvelope:
        """Return one envelope for the supplied request."""
