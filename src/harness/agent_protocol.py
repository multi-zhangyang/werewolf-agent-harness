"""Versioned decision boundary between the Werewolf environment and an agent."""
from __future__ import annotations

import json
import math
import time
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..agent.schemas import Decision

AGENT_PROTOCOL_VERSION = "werewolf.harness.agent_protocol.v2"


class LegalAction(BaseModel):
    """One action and target set advertised by the environment."""

    model_config = ConfigDict(extra="forbid")

    action: str
    target_seats: list[int] = Field(default_factory=list)
    # An empty target set is not enough to distinguish a target-free action
    # (for example SPEAK) from a targeted action that currently has no legal
    # target (for example SAVE when nobody was attacked).
    target_required: bool = Field(default=False, strict=True)
    can_skip: bool = Field(default=False, strict=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action")
    @classmethod
    def _nonempty_action(cls, value: str) -> str:
        action = str(value).strip()
        if not action:
            raise ValueError("legal action name must not be empty")
        return action

    @field_validator("target_seats", mode="before")
    @classmethod
    def _valid_target_seats(cls, value: Any) -> list[int]:
        if not isinstance(value, list):
            raise ValueError("legal target seats must be a list")
        if any(type(seat) is not int or seat < 1 for seat in value):
            raise ValueError("legal target seats must be positive integers")
        seats = list(value)
        if len(seats) != len(set(seats)):
            raise ValueError("legal target seats must be unique")
        return seats

    @model_validator(mode="after")
    def _targeted_action_is_executable(self) -> "LegalAction":
        if self.target_required and not self.target_seats and not self.can_skip:
            raise ValueError(
                "a target-required action with no legal targets must allow skip"
            )
        return self

    @field_validator("metadata")
    @classmethod
    def _json_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_json_value(value, label="legal action metadata")
        return value

    @property
    def requires_target(self) -> bool:
        """Return the explicit requirement while preserving v2 compatibility."""
        return self.target_required or bool(self.target_seats)


class ActionRequest(BaseModel):
    """One immutable environment-to-agent decision request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal[AGENT_PROTOCOL_VERSION] = AGENT_PROTOCOL_VERSION
    request_id: str = Field(min_length=1)
    run_id: str | None = None
    seat: int = Field(ge=1, strict=True)
    phase: str
    day: int = Field(ge=0, strict=True)
    action_kind: str
    observation: dict[str, Any] = Field(default_factory=dict)
    legal_actions: list[LegalAction] = Field(min_length=1)
    deadline_monotonic: float | None = None
    private_context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("phase", "action_kind")
    @classmethod
    def _nonempty_label(cls, value: str) -> str:
        label = str(value).strip()
        if not label:
            raise ValueError("request phase/action_kind must not be empty")
        return label

    @field_validator("request_id")
    @classmethod
    def _nonempty_request_id(cls, value: str) -> str:
        request_id = str(value).strip()
        if not request_id:
            raise ValueError("request identity must not be empty")
        return request_id

    @field_validator("run_id")
    @classmethod
    def _nonempty_run_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        run_id = str(value).strip()
        if not run_id:
            raise ValueError("run identity must not be empty")
        return run_id

    @field_validator("observation", "private_context", "metadata")
    @classmethod
    def _json_objects(cls, value: dict[str, Any], info: Any) -> dict[str, Any]:
        _require_json_value(value, label=info.field_name)
        return value

    @field_validator("deadline_monotonic")
    @classmethod
    def _finite_deadline(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("deadline_monotonic must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def _unique_legal_actions(self) -> "ActionRequest":
        actions = [item.action for item in self.legal_actions]
        if len(actions) != len(set(actions)):
            raise ValueError("legal action names must be unique within one request")
        return self

    def seconds_remaining(self, now: float | None = None) -> float | None:
        if self.deadline_monotonic is None:
            return None
        current = time.monotonic() if now is None else now
        return max(0.0, self.deadline_monotonic - current)


class DecisionEnvelope(BaseModel):
    """One agent decision plus model-call and parse provenance."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: Literal[AGENT_PROTOCOL_VERSION] = AGENT_PROTOCOL_VERSION
    request_id: str = Field(min_length=1)
    seat: int = Field(ge=1, strict=True)
    decision: Decision
    latency_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    model_call_id: str | None = None
    prompt_hash: str | None = None
    response_hash: str | None = None
    parse_status: Literal["ok", "recovered", "not_applicable"] = "ok"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("request_id")
    @classmethod
    def _nonempty_request_id(cls, value: str) -> str:
        request_id = str(value).strip()
        if not request_id:
            raise ValueError("request identity must not be empty")
        return request_id

    @field_validator("metadata")
    @classmethod
    def _json_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_json_value(value, label="decision metadata")
        return value

    @model_validator(mode="after")
    def _json_decision_payload(self) -> "DecisionEnvelope":
        _require_json_value(
            self.decision.model_dump(exclude={"llm_call_trace"}),
            label="decision payload",
        )
        return self

    @property
    def skipped(self) -> bool:
        return self.decision.is_skip


def decision_action_value(decision: Decision) -> str:
    return str(getattr(decision.action, "value", decision.action)).strip()


def decision_is_skip(decision: Decision) -> bool:
    return decision.is_skip


def _require_json_value(value: Any, *, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as err:
        raise ValueError(f"{label} must contain only finite JSON values") from err


@runtime_checkable
class AgentProtocol(Protocol):
    """The only production decision interface for LLM and human agents."""

    seat: int

    async def decide(self, request: ActionRequest) -> DecisionEnvelope:
        """Return exactly one envelope for the supplied request."""
