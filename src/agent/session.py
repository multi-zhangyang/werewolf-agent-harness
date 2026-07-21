"""Per-seat tool-using Agent session primitives.

This module is deliberately independent from the Werewolf rules engine.  A
single :class:`AgentSession` belongs to one seat and owns the context that is
sent to the model, the tool-loop counters, and the private trace.  The
environment supplies tools through :class:`ToolRegistry`; terminal tools
return a :class:`~src.agent.schemas.Decision` which the environment still
has to validate and consume.

The loop follows the usual agent protocol::

    model turn -> tool call -> tool result -> model turn -> ... -> terminal

Tool failures are model-visible observations.  They do not become an
implicit ``skip`` and they do not terminate the loop unless a configured
budget is exhausted.  Identity (seat/role) is supplied by the execution
context and is intentionally absent from tool schemas and model-authored
arguments.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import re
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from jsonschema import Draft202012Validator

from ..harness.agent_protocol import ActionRequest, decision_action_value
from ..harness.transcript import redact_sensitive
from .schemas import AgentAction, Decision


class ToolKind(StrEnum):
    """Capabilities exposed to one Agent model."""

    READ_ONLY = "read_only"
    PRIVATE_STATE = "private_state"
    TERMINAL = "terminal"


# These names would let a model choose or impersonate the caller.  A target
# such as ``target_seat`` or a belief's ``seat`` remains valid: those refer to
# another player, not to the owner of this session.
_IDENTITY_ARGUMENT_NAMES = frozenset({
    "my_seat",
    "owner_seat",
    "agent_seat",
    "actor_seat",
    "player_seat",
    "self_seat",
    "identity_seat",
    "my_role",
    "owner_role",
    "agent_role",
    "actor_role",
    "self_role",
    "identity_role",
})
# Match the Router's provider-neutral function name contract exactly.
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_RESULT_CHARS = 12_000
_MAX_ERROR_CHARS = 600
_MAX_PRIVATE_TRACE_VALUE_CHARS = 16_000
_MAX_PRIVATE_TRACE_ROW_CHARS = 32_000
_TOOL_CANCELLATION_GRACE_SECONDS = 0.25
_DEFAULT_MAX_MODEL_HISTORY_CHARS = 24_000
_DEFAULT_RECENT_TOOL_GROUPS = 3
_HISTORY_SUMMARY_STRING_CHARS = 240
_HISTORY_SUMMARY_ITEMS = 4
_HISTORY_SUMMARY_KEYS = 12


class AgentSessionError(RuntimeError):
    """A bounded, attributable failure of an Agent session."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.code = str(code)
        self.details = dict(details or {})
        self.cause = cause
        super().__init__(str(message))


class ToolExecutionError(RuntimeError):
    """A safe error raised by a tool handler.

    The message is shown to the owning model as an observation.  It is never
    broadcast as an environment event unless the caller explicitly projects
    it.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.details = dict(details or {})
        super().__init__(str(message))


class SessionStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class _TraceList(list[dict[str, Any]]):
    """Detached trace view compatible with property and method callers."""

    def __call__(self) -> list[dict[str, Any]]:
        return deepcopy(list(self))


@dataclass(frozen=True)
class AgentSessionLimits:
    """Hard bounds for one model/tool loop.

    Values are intentionally conservative defaults.  They bound model turns,
    tool executions, repeated no-progress turns and wall time independently.
    """

    max_steps: int = 12
    # One step can make more than one provider request when a response is
    # malformed and retried.  Keep that amplification independently bounded.
    # ``None`` preserves compatibility for integrations that intentionally
    # delegate the call budget to an outer provider ledger.
    max_model_generations: int | None = None
    max_tool_calls: int = 32
    max_no_progress_steps: int = 3
    max_model_response_retries: int = 2
    # This is a post-response accounting limit over provider-reported usage;
    # it is never translated into a provider ``max_tokens`` parameter.  A
    # response that crosses the limit is recorded but its tools are not run.
    max_total_tokens: int | None = None
    # Some test doubles and non-conforming gateways omit usage.  By default we
    # expose that gap in telemetry and continue; strict harnesses can fail
    # closed whenever a token budget cannot be audited.
    require_token_usage: bool = False
    max_wall_time_seconds: float | None = 180.0
    # Tool handlers are an independent failure boundary.  A hung external
    # lookup must become a model-visible observation before the whole session
    # deadline expires.
    max_tool_time_seconds: float | None = 30.0
    # Full messages remain in the private session record.  Only the copy sent
    # to the next model turn is compacted, and only at complete assistant-call
    # + tool-result boundaries.  This bounds repeated prompt amplification
    # without creating orphaned tool protocol messages.
    max_model_history_chars: int | None = _DEFAULT_MAX_MODEL_HISTORY_CHARS
    keep_recent_tool_groups: int = _DEFAULT_RECENT_TOOL_GROUPS

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_tool_calls", "max_no_progress_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_model_generations is not None:
            value = self.max_model_generations
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError("max_model_generations must be a positive integer or None")
        value = self.max_model_response_retries
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError("max_model_response_retries must be a non-negative integer")
        if self.max_total_tokens is not None:
            value = self.max_total_tokens
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError("max_total_tokens must be a positive integer or None")
        if not isinstance(self.require_token_usage, bool):
            raise ValueError("require_token_usage must be a boolean")
        if self.require_token_usage and self.max_total_tokens is None:
            raise ValueError("require_token_usage requires max_total_tokens")
        if self.max_wall_time_seconds is not None:
            value = float(self.max_wall_time_seconds)
            if value <= 0 or not value == value or value == float("inf"):
                raise ValueError("max_wall_time_seconds must be a finite positive number")
        if self.max_tool_time_seconds is not None:
            value = float(self.max_tool_time_seconds)
            if value <= 0 or not value == value or value == float("inf"):
                raise ValueError("max_tool_time_seconds must be a finite positive number")
        if self.max_model_history_chars is not None:
            value = self.max_model_history_chars
            if isinstance(value, bool) or int(value) != value or int(value) < 256:
                raise ValueError("max_model_history_chars must be an integer >= 256 or None")
        value = self.keep_recent_tool_groups
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError("keep_recent_tool_groups must be a positive integer")


@dataclass(frozen=True)
class TerminalSubmission:
    """A terminal tool's environment-bound decision."""

    decision: Decision
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """Model-visible result of one tool call."""

    call_id: str
    name: str
    kind: ToolKind | None
    ok: bool
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None
    error_details: Mapping[str, Any] = field(default_factory=dict)
    terminal: bool = False

    def observation(self) -> dict[str, Any]:
        """Return bounded JSON-safe data suitable for the next model turn."""
        row: dict[str, Any] = {
            "ok": bool(self.ok),
            "tool": self.name,
            "call_id": self.call_id,
        }
        if self.ok:
            # Tool output is untrusted model input.  Keep the same credential
            # redaction and global size bound used by the admin trace so a
            # custom tool cannot amplify the next provider prompt without a
            # limit.
            row["result"] = _private_trace_view(self.output)
            if self.terminal:
                row["terminal"] = True
        else:
            row["error"] = {
                "code": str(self.error_code or "tool_error"),
                "message": _bounded_text(self.error_message or "tool execution failed", _MAX_ERROR_CHARS),
                "details": _bounded_json(dict(self.error_details)),
            }
        return row

    def model_message(self) -> dict[str, Any]:
        """Encode this observation as a standard tool result message."""
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "name": self.name,
            "content": json.dumps(self.observation(), ensure_ascii=False, sort_keys=True),
        }


ToolHandler = Callable[["ToolExecutionContext", dict[str, Any]], Any | Awaitable[Any]]
DecisionBuilder = Callable[["ToolExecutionContext", dict[str, Any], Any], Any | Awaitable[Any]]
TraceSink = Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    """One registered function and its capability class."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    kind: ToolKind
    handler: ToolHandler
    # Optional environment action name.  When omitted, a terminal tool whose
    # name matches a LegalAction is preflighted against that action.
    terminal_action: str | None = None
    # Override the session default for tools that need a tighter/looser bound.
    timeout_seconds: float | None = None
    # Optional conversion for a terminal handler that returns a payload rather
    # than a Decision.  The builder must still return a Decision or
    # TerminalSubmission; the environment remains the source of truth.
    decision_builder: DecisionBuilder | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not _TOOL_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid tool name: {self.name!r}")
        if not isinstance(self.kind, ToolKind):
            object.__setattr__(self, "kind", ToolKind(self.kind))
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")
        schema = deepcopy(dict(self.parameters))
        if schema.get("type") != "object":
            raise ValueError("tool parameters schema must have type='object'")
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:  # jsonschema raises SchemaError, keep API stable
            raise ValueError("tool parameters schema is invalid") from exc
        forbidden = _identity_fields_in_schema(schema)
        if forbidden:
            raise ValueError(
                "tool schema must not expose caller identity fields: "
                + ", ".join(sorted(forbidden))
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", _bounded_text(self.description, 2_000))
        object.__setattr__(self, "parameters", schema)
        if self.terminal_action is not None:
            object.__setattr__(self, "terminal_action", str(self.terminal_action).strip() or None)
        if self.timeout_seconds is not None:
            timeout = float(self.timeout_seconds)
            if timeout <= 0 or not math.isfinite(timeout):
                raise ValueError("tool timeout_seconds must be a finite positive number")
            object.__setattr__(self, "timeout_seconds", timeout)


class ToolExecutionContext:
    """Private context injected by an :class:`AgentSession`.

    ``seat`` and ``role`` are available to Python handlers but never copied
    into a model-authored argument or a public tool definition.  The request
    and private context are detached copies to prevent handlers from mutating
    the immutable environment request.
    """

    __slots__ = (
        "_request",
        "seat",
        "role",
        "step",
        "state_version",
        "private_state",
        "memory",
        "metadata",
        "_session",
        "_pending_terminal_submission",
    )

    def __init__(
        self,
        *,
        request: ActionRequest,
        seat: int,
        role: str | None,
        step: int,
        state_version: int,
        private_state: Any = None,
        memory: Any = None,
        metadata: Mapping[str, Any] | None = None,
        session: "AgentSession | None" = None,
    ) -> None:
        # ``ActionRequest`` is only shallowly frozen by Pydantic.  Keep a
        # detached private copy and expose another deep copy through the
        # property below, so a custom tool cannot mutate nested observation,
        # legal-action, or metadata values used by later tool calls.
        self._request = _clone_action_request(request)
        self.seat = int(seat)
        self.role = str(role) if role is not None else None
        self.step = int(step)
        self.state_version = int(state_version)
        self.private_state = private_state
        self.memory = memory
        self.metadata = deepcopy(dict(metadata or {}))
        self._session = session
        self._pending_terminal_submission: TerminalSubmission | None = None

    @property
    def request(self) -> ActionRequest:
        """Return a detached request snapshot for a Python tool handler."""
        return _clone_action_request(self._request)

    @property
    def observation(self) -> dict[str, Any]:
        return deepcopy(dict(self.request.observation))

    @property
    def private_context(self) -> dict[str, Any]:
        return deepcopy(dict(self.request.private_context))

    @property
    def legal_actions(self) -> tuple[Any, ...]:
        return tuple(self.request.legal_actions)

    @property
    def terminal_consumed(self) -> bool:
        return bool(
            self._pending_terminal_submission is not None
            or (self._session and self._session.terminal_submission is not None)
        )

    def submit_terminal(self, value: Decision | TerminalSubmission, *, tool_name: str = "", call_id: str = "") -> TerminalSubmission:
        """Stage a terminal candidate for the current tool invocation.

        Returning a Decision from the handler remains the preferred path; this
        explicit method is useful for adapters that perform local work before
        returning.  The Registry commits it only after legality validation, so
        an invalid candidate cannot poison the owning session.
        """
        if self._session is None:
            raise ToolExecutionError("terminal_context_missing", "terminal tools require an AgentSession context")
        if isinstance(value, Decision):
            submission = TerminalSubmission(value)
        elif isinstance(value, TerminalSubmission):
            submission = value
        else:
            raise AgentSessionError(
                "terminal_decision_missing",
                "terminal value must be a Decision or TerminalSubmission",
            )
        if self._pending_terminal_submission is not None:
            raise ToolExecutionError(
                "terminal_already_submitted",
                "a terminal action has already been staged for this tool call",
            )
        self._pending_terminal_submission = submission
        return submission

    def _take_pending_terminal(self) -> TerminalSubmission | None:
        submission = self._pending_terminal_submission
        self._pending_terminal_submission = None
        return submission

    def _clear_pending_terminal(self) -> None:
        self._pending_terminal_submission = None


class ToolRegistry:
    """Validated, capability-labelled registry for one Agent session.

    Registries are safe to share as immutable definitions, but terminal state
    is always held by the session context, never by a registry instance.
    """

    def __init__(self, specs: Sequence[ToolSpec] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs or ():
            self.register(spec)

    def register(
        self,
        spec_or_name: ToolSpec | str,
        handler: ToolHandler | None = None,
        *,
        description: str = "",
        parameters: Mapping[str, Any] | None = None,
        kind: ToolKind = ToolKind.READ_ONLY,
        decision_builder: DecisionBuilder | None = None,
        terminal_action: str | None = None,
        timeout_seconds: float | None = None,
        replace: bool = False,
    ) -> ToolSpec:
        """Register and validate a tool definition.

        Both ``register(ToolSpec(...))`` and the keyword form are supported so
        environment plugins can construct registries without boilerplate.
        """
        if isinstance(spec_or_name, ToolSpec):
            spec = spec_or_name
        else:
            if handler is None:
                raise TypeError("handler is required when registering by name")
            spec = ToolSpec(
                name=str(spec_or_name),
                description=description,
                parameters=parameters or _empty_parameters_schema(),
                kind=kind,
                handler=handler,
                decision_builder=decision_builder,
                terminal_action=terminal_action,
                timeout_seconds=timeout_seconds,
            )
        if spec.name in self._specs and not replace:
            raise ValueError(f"tool already registered: {spec.name}")
        self._specs[spec.name] = spec
        return spec

    def unregister(self, name: str) -> None:
        self._specs.pop(str(name), None)

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(str(name))

    def __contains__(self, name: object) -> bool:
        return str(name) in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs.values())

    def definitions(self) -> list[dict[str, Any]]:
        """Return provider-neutral OpenAI function-tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": deepcopy(dict(spec.parameters)),
                },
            }
            for spec in self._specs.values()
        ]

    # Common aliases used by adapters and tests.
    model_tools = definitions
    tool_definitions = definitions

    async def execute(
        self,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any] | None,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Validate and invoke one model-selected tool.

        Every failure is converted into a :class:`ToolResult`; callers can
        append it to the model history and let the same Agent recover.
        """
        call_id = str(call_id or "")
        name = str(name or "")
        spec = self._specs.get(name)
        if spec is None:
            return _tool_error(call_id, name, None, "unknown_tool", "tool is not registered")
        if not isinstance(arguments, Mapping):
            return _tool_error(call_id, name, spec.kind, "invalid_arguments", "tool arguments must be a JSON object")
        raw_args = deepcopy(dict(arguments))
        forbidden = _identity_fields_in_mapping(raw_args)
        if forbidden:
            return _tool_error(
                call_id,
                name,
                spec.kind,
                "identity_argument_forbidden",
                "caller identity is supplied by the environment",
                details={"fields": sorted(forbidden)},
            )
        validator = Draft202012Validator(dict(spec.parameters))
        errors = sorted(validator.iter_errors(raw_args), key=lambda item: list(item.path))
        if errors:
            first = errors[0]
            path = ".".join(str(item) for item in first.path)
            message = f"invalid tool arguments{f' at {path}' if path else ''}: {first.message}"
            return _tool_error(call_id, name, spec.kind, "invalid_arguments", message)

        if spec.kind == ToolKind.TERMINAL:
            preflight_error = _preflight_terminal_arguments(spec, context.request, raw_args)
            if preflight_error is not None:
                return _tool_error(
                    call_id,
                    name,
                    spec.kind,
                    "illegal_terminal_action",
                    preflight_error,
                )

        # A terminal claim is guarded by the session.  We reserve only around
        # handler execution; failed handlers release the reservation so the
        # model can retry with a corrected target or payload.
        lock = _terminal_lock(context) if spec.kind == ToolKind.TERMINAL else _null_async_lock()
        async with lock:
            if spec.kind == ToolKind.TERMINAL and context.terminal_consumed:
                return _tool_error(
                    call_id,
                    name,
                    spec.kind,
                    "terminal_already_submitted",
                    "a terminal action has already been submitted for this request",
                )
            try:
                value = spec.handler(context, raw_args)
                if inspect.isawaitable(value):
                    timeout = spec.timeout_seconds
                    if timeout is None and context._session is not None:
                        timeout = context._session.limits.max_tool_time_seconds
                    value = await _await_tool_with_timeout(
                        value,
                        timeout,
                        session=context._session,
                    )
                if spec.kind == ToolKind.TERMINAL:
                    submission = await _coerce_terminal_submission(
                        spec,
                        context,
                        raw_args,
                        value,
                    )
                    staged = context._take_pending_terminal()
                    if staged is not None:
                        if submission is not None and not _same_terminal_submission(
                            staged,
                            submission,
                        ):
                            return _tool_error(
                                call_id,
                                name,
                                spec.kind,
                                "terminal_submission_conflict",
                                "terminal handler both staged and returned a submission",
                            )
                        submission = staged
                    if submission is None:
                        return _tool_error(
                            call_id,
                            name,
                            spec.kind,
                            "terminal_decision_missing",
                            "terminal tool must return a Decision or TerminalSubmission",
                        )
                    if not _decision_is_legal(context.request, submission.decision):
                        return _tool_error(
                            call_id,
                            name,
                            spec.kind,
                            "illegal_terminal_action",
                            "terminal decision is not advertised as legal for this request",
                        )
                    if context._session is None:
                        return _tool_error(
                            call_id,
                            name,
                            spec.kind,
                            "terminal_context_missing",
                            "terminal tools require an AgentSession context",
                        )
                    already_committed = context._session.terminal_submission is submission
                    if not already_committed:
                        context._session._commit_terminal(submission, name=name, call_id=call_id)
                    return ToolResult(
                        call_id=call_id,
                        name=name,
                        kind=spec.kind,
                        ok=True,
                        output={
                            "action": decision_action_value(submission.decision),
                            "message": submission.message,
                        },
                        terminal=True,
                    )
                return ToolResult(
                    call_id=call_id,
                    name=name,
                    kind=spec.kind,
                    ok=True,
                    output=value,
                )
            except ToolExecutionError as exc:
                return _tool_error(
                    call_id,
                    name,
                    spec.kind,
                    exc.code,
                    str(exc),
                    details=exc.details,
                )
            except Exception as exc:  # noqa: BLE001 - tool boundary is recoverable
                return _tool_error(
                    call_id,
                    name,
                    spec.kind,
                    "tool_execution_failed",
                    f"{type(exc).__name__}: tool execution failed",
                )
            finally:
                if spec.kind == ToolKind.TERMINAL:
                    context._clear_pending_terminal()


class ToolTurnRouter(Protocol):
    async def complete_tools(
        self,
        messages: list[dict[str, Any]],
        config: Any,
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        ...


@dataclass
class AgentSessionResult:
    """Detached result of a completed or failed Agent loop."""

    session_id: str
    seat: int
    status: SessionStatus
    decision: Decision | None = None
    # ``terminal_value`` is the generic name used by environment adapters;
    # for Werewolf it is the same ``Decision`` exposed by ``decision``.
    terminal_value: Any = None
    terminal_tool: str | None = None
    steps: int = 0
    tool_calls: int = 0
    # Provider generation attempts include response-level retries.  The
    # existing ``steps`` counter intentionally remains model-loop steps for
    # backwards compatibility.
    generation_attempts: int = 0
    model_generations: int = 0
    generation_failures: int = 0
    response_retries: int = 0
    tool_successes: int = 0
    tool_failures: int = 0
    model_latency_seconds: float = 0.0
    tool_latency_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    token_usage_complete: bool = True
    total_tokens: int = 0
    no_progress_steps: int = 0
    state_version: int = 0
    error: AgentSessionError | None = None
    # Alias retained for callers that use failure terminology instead of
    # ``error``.  It is always the same bounded exception object.
    failure: AgentSessionError | None = None
    usage: dict[str, int] = field(default_factory=dict)
    history_compactions: int = 0
    max_compacted_tool_groups: int = 0
    peak_history_chars: int = 0
    peak_model_history_chars: int = 0
    history_limit_misses: int = 0
    telemetry: dict[str, Any] = field(default_factory=dict)
    _private_trace: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def completed(self) -> bool:
        return self.status == SessionStatus.COMPLETED and self.decision is not None

    @property
    def failed(self) -> bool:
        return self.status in {SessionStatus.FAILED, SessionStatus.CANCELLED}

    def require_decision(self) -> Decision:
        if self.decision is None:
            if self.error is not None:
                raise self.error
            raise AgentSessionError("terminal_decision_missing", "session did not submit a terminal decision")
        return self.decision

    @property
    def trace(self) -> list[dict[str, Any]]:
        """Admin-only private trace alias."""
        return self.private_trace()

    def private_trace(self) -> list[dict[str, Any]]:
        """Return a detached trace for an authorized God/Admin consumer."""
        return deepcopy(self._private_trace)

    def public_summary(self) -> dict[str, Any]:
        """Return counters only; no prompts, reasoning, arguments or beliefs."""
        return {
            "session_id": self.session_id,
            "seat": self.seat,
            "status": self.status.value,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "generation_attempts": self.generation_attempts,
            "model_generations": self.model_generations,
            "generation_failures": self.generation_failures,
            "response_retries": self.response_retries,
            "tool_successes": self.tool_successes,
            "tool_failures": self.tool_failures,
            "model_latency_seconds": self.model_latency_seconds,
            "tool_latency_seconds": self.tool_latency_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "token_usage_complete": self.token_usage_complete,
            "total_tokens": self.total_tokens,
            "no_progress_steps": self.no_progress_steps,
            "state_version": self.state_version,
            "terminal_tool": self.terminal_tool,
            "action": decision_action_value(self.decision) if self.decision else None,
            "error_code": self.error.code if self.error else None,
            "usage": dict(self.usage),
            "history_compactions": self.history_compactions,
            "max_compacted_tool_groups": self.max_compacted_tool_groups,
            "peak_history_chars": self.peak_history_chars,
            "peak_model_history_chars": self.peak_model_history_chars,
            "history_limit_misses": self.history_limit_misses,
            "telemetry": deepcopy(self.telemetry),
        }


class AgentSession:
    """One private, bounded model/tool loop for exactly one seat."""

    def __init__(
        self,
        *,
        seat: int,
        role: str | None = None,
        session_id: str | None = None,
        registry: ToolRegistry | None = None,
        limits: AgentSessionLimits | None = None,
        private_state: Any = None,
        memory: Any = None,
        trace_sink: TraceSink | None = None,
    ) -> None:
        if isinstance(seat, bool) or int(seat) < 1:
            raise ValueError("seat must be a positive integer")
        self.seat = int(seat)
        self.role = str(role) if role is not None else None
        self.session_id = str(session_id or uuid.uuid4().hex)
        self.registry = registry or ToolRegistry()
        self.limits = limits or AgentSessionLimits()
        self.private_state = private_state
        self.memory = memory
        self.trace_sink = trace_sink
        self.status = SessionStatus.CREATED
        self.steps = 0
        self.tool_call_count = 0
        self._generation_attempt_count = 0
        self._model_generation_count = 0
        self._generation_failure_count = 0
        self._response_retry_count = 0
        self._tool_success_count = 0
        self._tool_failure_count = 0
        self._model_latency_seconds = 0.0
        self._tool_latency_seconds = 0.0
        self._total_tokens = 0
        self._token_usage_complete = True
        self._finish_emitted = False
        self._budget_failure_code: str | None = None
        self.no_progress_steps = 0
        self.state_version = 0
        self.messages: list[dict[str, Any]] = []
        self._private_trace: list[dict[str, Any]] = []
        self._terminal_submission: TerminalSubmission | None = None
        self._terminal_tool: str | None = None
        self._terminal_call_id: str | None = None
        self._run_result: AgentSessionResult | None = None
        self._terminal_lock = asyncio.Lock()
        self._seen_call_ids: set[str] = set()
        self._usage: dict[str, int] = {}
        self._history_compactions = 0
        self._max_compacted_tool_groups = 0
        self._peak_history_chars = 0
        self._peak_model_history_chars = 0
        self._history_limit_misses = 0
        self._started_monotonic: float | None = None
        self._turn_id: str | None = None
        self._request: ActionRequest | None = None
        self._tool_tasks: set[asyncio.Future[Any]] = set()
        self._unresolved_tool_tasks: set[asyncio.Future[Any]] = set()
        self._trace_tasks: set[asyncio.Future[Any]] = set()

    @property
    def terminal_submission(self) -> TerminalSubmission | None:
        return self._terminal_submission

    @property
    def decision(self) -> Decision | None:
        return self._terminal_submission.decision if self._terminal_submission else None

    @property
    def terminal_tool(self) -> str | None:
        return self._terminal_tool

    @property
    def terminal_submitted(self) -> bool:
        return self._terminal_submission is not None

    @property
    def private_trace(self) -> list[dict[str, Any]]:
        # ``_TraceList`` is both iterable (the Actor integration uses the
        # property form) and callable (older harness consumers use
        # ``session.private_trace()``).  In either form the caller receives a
        # detached copy and cannot mutate the session-owned evidence.
        return _TraceList(deepcopy(self._private_trace))

    def submit_terminal(
        self,
        value: Decision | TerminalSubmission,
        *,
        tool_name: str = "",
        call_id: str = "",
    ) -> TerminalSubmission:
        """Commit exactly one terminal value for this request."""
        if isinstance(value, Decision):
            submission = TerminalSubmission(value)
        elif isinstance(value, TerminalSubmission):
            submission = value
        else:
            raise AgentSessionError("terminal_decision_missing", "terminal value must be a Decision or TerminalSubmission")
        if self._terminal_submission is not None:
            raise AgentSessionError(
                "terminal_already_submitted",
                "a terminal action has already been submitted for this request",
            )
        self._commit_terminal(submission, name=tool_name, call_id=call_id)
        return submission

    async def run(
        self,
        request: ActionRequest,
        router: ToolTurnRouter | Callable[..., Any] | None = None,
        config: Any | None = None,
        *,
        model_config: Any | None = None,
        system: str | None = None,
        messages: Sequence[Mapping[str, Any]] | None = None,
        initial_messages: Sequence[Mapping[str, Any]] | None = None,
        tools: ToolRegistry | Sequence[ToolSpec] | None = None,
        trace_context: Mapping[str, Any] | None = None,
        budget_scope: str | None = None,
    ) -> AgentSessionResult:
        """Run the loop until one terminal tool succeeds or a hard bound fails.

        The method is idempotent after a terminal result: repeated calls return
        a detached copy of the original result and never execute another
        terminal action.
        """
        if self._run_result is not None:
            try:
                repeated_request = request if isinstance(request, ActionRequest) else ActionRequest.model_validate(request)
            except Exception as exc:
                raise AgentSessionError("session_reuse_invalid_request", "completed session received an invalid request", cause=exc) from exc
            if self._request is not None and repeated_request.request_id != self._request.request_id:
                raise AgentSessionError(
                    "session_reused",
                    "one AgentSession can serve only one ActionRequest",
                    details={"original_request_id": self._request.request_id},
                )
            return _clone_result(self._run_result)
        if router is None:
            return self._fail("router_missing", "a tool-turn router is required")
        if not isinstance(request, ActionRequest):
            request = ActionRequest.model_validate(request)
        if request.seat != self.seat:
            return self._fail(
                "seat_mismatch",
                "ActionRequest seat does not match this AgentSession",
                details={"request_seat": request.seat},
            )
        selected_config = config if config is not None else model_config
        if selected_config is None:
            return self._fail("model_config_missing", "a model config is required for tool turns")
        self._request = request
        if messages is not None and initial_messages is not None:
            return self._fail(
                "initial_messages_ambiguous",
                "provide messages or initial_messages, not both",
            )
        if tools is not None:
            self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.status = SessionStatus.RUNNING
        self._started_monotonic = time.monotonic()
        self._turn_id = uuid.uuid4().hex
        selected_messages = messages if messages is not None else initial_messages
        self.messages = [deepcopy(dict(item)) for item in (selected_messages or ())]
        if not self.messages:
            self.messages = [{
                "role": "user",
                "content": _initial_observation_message(request),
            }]
        self._emit(
            "agent_turn_started",
            {
                "turn_id": self._turn_id,
                "request_id": request.request_id,
                "phase": request.phase,
                "day": request.day,
                "tool_count": len(self.registry),
                "context": dict(trace_context or {}),
            },
        )

        try:
            while self._terminal_submission is None:
                budget_code = self._exhausted_budget_code()
                if budget_code is not None:
                    return self._fail_budget(code=budget_code)
                self.steps += 1
                response_retry = 0
                while True:
                    if not self._model_generation_budget_available():
                        return self._fail_budget(code="max_model_generations")
                    self._generation_attempt_count += 1
                    generation_started = time.monotonic()
                    turn_call = self._complete_tools(
                        router,
                        selected_config,
                        system=system,
                        trace_context={
                            **dict(trace_context or {}),
                            "seat": self.seat,
                            "step": self.steps,
                            "response_attempt": response_retry + 1,
                        },
                        budget_scope=budget_scope,
                    )
                    remaining = self._remaining_timeout(request)
                    try:
                        response = (
                            await asyncio.wait_for(turn_call, timeout=remaining)
                            if remaining is not None
                            else await turn_call
                        )
                        generation_elapsed = max(
                            0.0,
                            time.monotonic() - generation_started,
                        )
                        self._model_latency_seconds += generation_elapsed
                        break
                    except asyncio.TimeoutError as exc:
                        generation_elapsed = max(
                            0.0,
                            time.monotonic() - generation_started,
                        )
                        self._model_latency_seconds += generation_elapsed
                        self._generation_failure_count += 1
                        self._emit_model_generation_failure(
                            exc,
                            response_attempt=response_retry + 1,
                            will_retry=False,
                            elapsed_seconds=generation_elapsed,
                        )
                        raise AgentSessionError(
                            "wall_time_exceeded",
                            "agent model turn exceeded its wall-time/deadline budget",
                            cause=exc,
                        ) from exc
                    except Exception as exc:
                        generation_elapsed = max(
                            0.0,
                            time.monotonic() - generation_started,
                        )
                        self._model_latency_seconds += generation_elapsed
                        self._generation_failure_count += 1
                        is_response_error = _is_model_response_error(exc)
                        retry_allowed = (
                            is_response_error
                            and response_retry < self.limits.max_model_response_retries
                            and self._model_generation_budget_available()
                        )
                        self._emit_model_generation_failure(
                            exc,
                            response_attempt=response_retry + 1,
                            will_retry=retry_allowed,
                            elapsed_seconds=generation_elapsed,
                        )
                        if not is_response_error:
                            raise
                        if response_retry >= self.limits.max_model_response_retries:
                            raise
                        if not self._model_generation_budget_available():
                            return self._fail_budget(code="max_model_generations")
                        response_retry += 1
                        self._response_retry_count += 1
                        await asyncio.sleep(min(0.8, 0.05 * (2 ** (response_retry - 1))))
                self._model_generation_count += 1
                response_total_tokens = self._record_response_usage(response)
                self._emit_model_generation(
                    response,
                    elapsed_seconds=generation_elapsed,
                    response_attempt=response_retry + 1,
                    response_total_tokens=response_total_tokens,
                )
                token_failure = self._token_budget_failure(response_total_tokens)
                if token_failure is not None:
                    code, message, details = token_failure
                    self._budget_failure_code = code
                    return self._fail(
                        code,
                        message,
                        details=details,
                    )
                calls = _extract_tool_calls(response)
                if not calls:
                    self._append_assistant_response(response, calls)
                    self.no_progress_steps += 1
                    self._append_model_observation(
                        {
                            "ok": False,
                            "error": {
                                "code": "terminal_tool_required",
                                "message": "use one registered terminal tool to submit the action",
                            },
                        },
                    )
                    self._emit(
                        "tool_result",
                        {
                            "call_id": None,
                            "tool": None,
                            "ok": False,
                            "error_code": "terminal_tool_required",
                        },
                    )
                    continue

                self._append_assistant_response(response, calls)
                turn_had_progress = False
                for call in calls:
                    # ``steps`` was admitted before this model generation. A
                    # terminal tool emitted on the last admitted step must
                    # still execute; rechecking max_steps here would reject a
                    # valid final Decision after the model has already paid
                    # for that step. Keep the independent wall-time bound.
                    if self._wall_time_exhausted():
                        return self._fail(
                            "wall_time_exceeded",
                            "agent session budget exhausted: wall_time_exceeded",
                        )
                    if self.tool_call_count >= self.limits.max_tool_calls:
                        return self._fail_budget(code="max_tool_calls")
                    self.tool_call_count += 1
                    self._emit(
                        "tool_call_requested",
                        {
                            "turn_id": self._turn_id,
                            "call_id": call.call_id,
                            "tool": call.name,
                            "step": self.steps,
                            "arguments_hash": _hash_json(call.arguments),
                            # Tool-capable models commonly express their plan,
                            # evidence and deception choice only through
                            # structured arguments. Keep a bounded, credential-
                            # redacted copy in the admin trace so God/replay can
                            # inspect that reasoning. This row never enters a
                            # player observation or public event stream.
                            "arguments": _private_trace_view(call.arguments),
                        },
                    )
                    tool_elapsed = 0.0
                    if call.parse_error is not None:
                        result = _tool_error(
                            call.call_id,
                            call.name,
                            self.registry.get(call.name).kind if self.registry.get(call.name) else None,
                            "invalid_arguments",
                            call.parse_error,
                        )
                    elif call.call_id in self._seen_call_ids:
                        result = _tool_error(
                            call.call_id,
                            call.name,
                            self.registry.get(call.name).kind if self.registry.get(call.name) else None,
                            "duplicate_call_id",
                            "call_id was already used in this session",
                        )
                    else:
                        self._seen_call_ids.add(call.call_id)
                        context = ToolExecutionContext(
                            request=request,
                            seat=self.seat,
                            role=self.role,
                            step=self.steps,
                            state_version=self.state_version,
                            private_state=self.private_state,
                            memory=self.memory,
                            metadata={"session_id": self.session_id, "turn_id": self._turn_id},
                            session=self,
                        )
                        tool_started = time.monotonic()
                        result = await self.registry.execute(
                            call.call_id,
                            call.name,
                            call.arguments,
                            context,
                        )
                        tool_elapsed = max(0.0, time.monotonic() - tool_started)
                    self.state_version += 1
                    if result.ok:
                        turn_had_progress = True
                        self._tool_success_count += 1
                    else:
                        self._tool_failure_count += 1
                    self._tool_latency_seconds += tool_elapsed
                    self._emit_tool_result(
                        result,
                        latency_seconds=tool_elapsed,
                    )
                    self.messages.append(result.model_message())
                    if result.ok and result.terminal:
                        self._emit(
                            "agent_action_submitted",
                            {
                                "turn_id": self._turn_id,
                                "call_id": call.call_id,
                                "tool": call.name,
                                "action": decision_action_value(self.decision),
                                "decision": self.decision.model_dump(mode="json", exclude={"llm_call_trace"})
                                if self.decision
                                else None,
                            },
                        )
                        break
                if turn_had_progress:
                    self.no_progress_steps = 0
                else:
                    self.no_progress_steps += 1
                if self._terminal_submission is not None:
                    break

            if self._unresolved_tool_tasks:
                return self._fail(
                    "tool_cleanup_pending",
                    "a tool handler ignored bounded cancellation",
                    details={"pending_task_count": len(self._unresolved_tool_tasks)},
                )
            self.status = SessionStatus.COMPLETED
            result = self._result()
            self._run_result = result
            return _clone_result(result)
        except asyncio.CancelledError as exc:
            self.status = SessionStatus.CANCELLED
            self._emit_failure("cancelled", "agent session cancelled")
            result = self._result(
                error=AgentSessionError("cancelled", "agent session cancelled", cause=exc)
            )
            self._run_result = result
            raise
        except AgentSessionError as exc:
            self.status = SessionStatus.FAILED
            self._emit_failure(exc.code, str(exc))
            result = self._result(error=exc)
            self._run_result = result
            return _clone_result(result)
        except Exception as exc:  # provider/router failures are terminal session errors
            wrapped = AgentSessionError(
                "model_turn_failed",
                f"model/tool turn failed: {type(exc).__name__}",
                cause=exc,
            )
            self.status = SessionStatus.FAILED
            self._emit_failure(wrapped.code, str(wrapped))
            result = self._result(error=wrapped)
            self._run_result = result
            return _clone_result(result)

    async def run_or_raise(self, request: ActionRequest, **kwargs: Any) -> AgentSessionResult:
        """Run and raise :class:`AgentSessionError` for a failed session."""
        result = await self.run(request, **kwargs)
        if result.error is not None:
            raise result.error
        return result

    async def aclose(self) -> None:
        """Boundedly cancel tool handlers that outlived this session.

        Most handlers finish during ``run``.  This lifecycle hook exists for
        external adapters and the Actor boundary so a handler that ignores
        cancellation is reported instead of silently continuing with access
        to seat-owned state.
        """
        # Give asynchronous admin trace sinks a short flush window before
        # cancelling anything.  Tool handlers are cancelled immediately below
        # because they may still hold mutable seat state.
        trace_tasks = {task for task in self._trace_tasks if not task.done()}
        if trace_tasks:
            _done, pending_trace = await asyncio.wait(
                trace_tasks,
                timeout=_TOOL_CANCELLATION_GRACE_SECONDS,
            )
            for task in pending_trace:
                task.cancel()
            if pending_trace:
                await asyncio.wait(
                    pending_trace,
                    timeout=_TOOL_CANCELLATION_GRACE_SECONDS,
                )

        tasks = {
            task
            for task in (*self._tool_tasks, *self._unresolved_tool_tasks)
            if not task.done()
        }
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.wait(
            tasks,
            timeout=_TOOL_CANCELLATION_GRACE_SECONDS,
        )
        pending = [task for task in tasks if not task.done()]
        if pending:
            raise AgentSessionError(
                "tool_cleanup_pending",
                "tool handler ignored bounded cancellation",
                details={"pending_task_count": len(pending)},
            )
        for task in tasks:
            _consume_task_result(task)
            self._tool_tasks.discard(task)
            self._unresolved_tool_tasks.discard(task)
        for task in list(self._trace_tasks):
            if task.done():
                _consume_task_result(task)
                self._trace_tasks.discard(task)

    def _forget_tool_task(self, task: asyncio.Future[Any]) -> None:
        self._tool_tasks.discard(task)
        self._unresolved_tool_tasks.discard(task)
        _consume_task_result(task)

    def _forget_trace_task(self, task: asyncio.Future[Any]) -> None:
        self._trace_tasks.discard(task)
        _consume_task_result(task)

    async def _complete_tools(
        self,
        router: ToolTurnRouter | Callable[..., Any],
        config: Any,
        *,
        system: str | None,
        trace_context: Mapping[str, Any],
        budget_scope: str | None,
    ) -> Any:
        method = getattr(router, "complete_tools", None)
        if method is None and callable(router):
            method = router
        if method is None or not callable(method):
            raise AgentSessionError("router_invalid", "router does not provide complete_tools")
        kwargs: dict[str, Any] = {
            # Every productive AgentSession turn is a tool turn. ``required``
            # still lets the model choose which read/private/terminal tool to
            # call, while preventing plain chat from counting as progress.
            "tool_choice": "required",
            # Serial execution keeps terminal/state writes deterministic.  A
            # plugin may explicitly parallelize read-only calls later.
            "parallel_tool_calls": False,
            "system": system,
            "trace_context": dict(trace_context),
            "budget_scope": budget_scope,
        }
        # Test doubles often implement a narrower signature.  Filter only
        # unsupported keyword names without swallowing errors from the call.
        try:
            signature = inspect.signature(method)
            accepts_var_kw = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if not accepts_var_kw:
                kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key in signature.parameters
                }
        except (TypeError, ValueError):
            pass
        model_messages, history_stats = _model_history_view(
            self.messages,
            max_chars=self.limits.max_model_history_chars,
            keep_recent_tool_groups=self.limits.keep_recent_tool_groups,
        )
        self._peak_history_chars = max(
            self._peak_history_chars,
            int(history_stats["original_chars"]),
        )
        self._peak_model_history_chars = max(
            self._peak_model_history_chars,
            int(history_stats["model_chars"]),
        )
        compacted_groups = int(history_stats["compacted_tool_groups"])
        limit_satisfied = bool(history_stats["limit_satisfied"])
        if not limit_satisfied:
            self._history_limit_misses += 1
        if compacted_groups or not limit_satisfied:
            # ``agent_history_compacted`` is also emitted when no complete old
            # group could be replaced.  This preserves provider compatibility
            # (the model still receives the unmodified history) while making a
            # soft-window miss observable to metrics and an admin operator.
            if compacted_groups:
                self._history_compactions += 1
                self._max_compacted_tool_groups = max(
                    self._max_compacted_tool_groups,
                    compacted_groups,
                )
            self._emit(
                "agent_history_compacted",
                {
                    "step": self.steps,
                    "original_message_count": len(self.messages),
                    "model_message_count": len(model_messages),
                    "original_chars": int(history_stats["original_chars"]),
                    "model_chars": int(history_stats["model_chars"]),
                    "compacted_tool_groups": compacted_groups,
                    "limit_satisfied": limit_satisfied,
                    "model_history_hash": _hash_json(model_messages),
                },
            )
        value = method(model_messages, config, self.registry.definitions(), **kwargs)
        if inspect.isawaitable(value):
            return await value
        return value

    def _append_assistant_response(self, response: Any, calls: Sequence["_NormalizedToolCall"]) -> None:
        # History is private but is sent back across the provider boundary on
        # the next step.  Scrub credentials before retaining or replaying model
        # text and raw function arguments.
        content = str(redact_sensitive(str(_response_field(response, "content", "") or "")))
        row: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            row["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": (
                            json.dumps(
                                redact_sensitive(call.arguments),
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            if call.parse_error is None
                            else str(redact_sensitive(call.raw_arguments))
                        ),
                    },
                }
                for call in calls
            ]
        self.messages.append(row)

    def _append_model_observation(self, observation: Mapping[str, Any]) -> None:
        self.messages.append({
            "role": "user",
            "content": json.dumps(_bounded_json(dict(observation)), ensure_ascii=False, sort_keys=True),
        })

    def _emit_model_generation(
        self,
        response: Any,
        *,
        elapsed_seconds: float,
        response_attempt: int,
        response_total_tokens: int | None,
    ) -> None:
        trace = _response_field(response, "trace")
        if not isinstance(trace, Mapping):
            trace = {}
        raw_usage = _response_field(response, "usage", {})
        usage = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
        payload = {
            "turn_id": self._turn_id,
            "step": self.steps,
            "generation_attempt": self._generation_attempt_count,
            "generation_index": self._model_generation_count,
            "response_attempt": int(response_attempt),
            "call_id": _response_field(response, "call_id"),
            "request_hash": _response_field(response, "request_hash"),
            "response_hash": (
                str(trace.get("response_hash"))
                if trace.get("response_hash")
                else _hash_text(str(_response_field(response, "content", "") or ""))
            ),
            # Keep provider text in the private God/Admin trace as well as its
            # hash.  A tool-capable model may put an intent explanation beside
            # a function call even when no separate reasoning channel exists.
            "content": _bounded_text(
                str(_response_field(response, "content", "") or ""),
                _MAX_RESULT_CHARS,
            ),
            "reasoning": _bounded_text(str(_response_field(response, "reasoning", "") or ""), _MAX_RESULT_CHARS),
            "usage": _bounded_json(usage),
            "response_total_tokens": response_total_tokens,
            "token_usage_complete": response_total_tokens is not None,
            "elapsed_seconds": round(float(elapsed_seconds), 6),
            "latency": float(_response_field(response, "latency", 0.0) or 0.0),
            "tool_call_count": len(_extract_tool_calls(response)),
            "router_trace": _private_trace_view(trace),
        }
        self._emit("model_generation", payload)

    def _emit_model_generation_failure(
        self,
        error: BaseException,
        *,
        response_attempt: int,
        will_retry: bool,
        elapsed_seconds: float,
    ) -> None:
        """Record a bounded response-level failure before a fresh model turn."""
        trace = getattr(error, "llm_call_trace", None)
        trace_view = _private_trace_view(trace) if isinstance(trace, Mapping) else {}
        self._emit(
            "model_generation_failed",
            {
                "step": self.steps,
                "generation_attempt": self._generation_attempt_count,
                "response_attempt": int(response_attempt),
                "will_retry": bool(will_retry),
                "elapsed_seconds": round(float(elapsed_seconds), 6),
                "error_type": type(error).__name__,
                "call_id": trace_view.get("call_id") if isinstance(trace_view, Mapping) else None,
                "request_hash": trace_view.get("request_hash") if isinstance(trace_view, Mapping) else None,
                "router_trace": trace_view,
            },
        )

    def _emit_tool_result(self, result: ToolResult, *, latency_seconds: float = 0.0) -> None:
        self._emit(
            "tool_result",
            {
                "turn_id": self._turn_id,
                "call_id": result.call_id,
                "tool": result.name,
                "kind": result.kind.value if result.kind else None,
                "ok": result.ok,
                "terminal": result.terminal,
                "latency": round(float(latency_seconds), 6),
                "output_hash": _hash_json(result.output) if result.ok else None,
                "output": _private_trace_view(result.output) if result.ok else None,
                "error": {
                    "code": result.error_code,
                    "message": result.error_message,
                    "details": _private_trace_view(result.error_details),
                }
                if not result.ok
                else None,
            },
        )

    def _emit_failure(self, code: str, message: str) -> None:
        self._emit(
            "agent_turn_failed",
            {
                "turn_id": self._turn_id,
                "step": self.steps,
                "error_code": str(code),
                "message": _bounded_text(message, _MAX_ERROR_CHARS),
            },
        )

    def _emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        raw_row = {
            "type": str(event_type),
            # Decision transcript rows are routed through the admin capability
            # channel.  Marking them as bare ``private`` would require a
            # recipient list and would fail the transcript visibility audit.
            "visibility": "admin",
            "audience": "admin",
            "session_id": self.session_id,
            "seat": self.seat,
            "state_version": self.state_version,
            "request_id": self._request.request_id if self._request is not None else None,
            **deepcopy(dict(payload)),
        }
        # Trace rows are consumed by more than the HTTP projection (custom
        # sinks and offline harnesses can inspect them directly), so do not
        # rely on a later API redaction pass for credentials or size bounds.
        row = _bounded_trace_row(raw_row)
        self._private_trace.append(row)
        if self.trace_sink is not None:
            try:
                value = self.trace_sink(deepcopy(row))
                # Do not make a sync trace sink mandatory async.
                if inspect.isawaitable(value):
                    # Keep a bounded reference so AgentSession.aclose() can
                    # flush admin evidence before the owning Actor is reused.
                    task = asyncio.create_task(value)
                    self._trace_tasks.add(task)
                    task.add_done_callback(self._forget_trace_task)
            except Exception:
                # Trace is observability, never a second game decision path.
                pass

    def _commit_terminal(self, submission: TerminalSubmission, *, name: str, call_id: str) -> None:
        if self._terminal_submission is not None:
            raise ToolExecutionError(
                "terminal_already_submitted",
                "a terminal action has already been submitted for this request",
            )
        self._terminal_submission = submission
        self._terminal_tool = str(name)
        self._terminal_call_id = str(call_id)

    def _model_generation_budget_available(self) -> bool:
        limit = self.limits.max_model_generations
        return limit is None or self._generation_attempt_count < limit

    def _exhausted_budget_code(self) -> str | None:
        """Return the first request budget that blocks another model turn."""
        if self.steps >= self.limits.max_steps:
            return "max_steps"
        if self.no_progress_steps >= self.limits.max_no_progress_steps:
            return "no_progress"
        if not self._model_generation_budget_available():
            return "max_model_generations"
        if (
            self.limits.max_total_tokens is not None
            and self._total_tokens >= self.limits.max_total_tokens
        ):
            return "max_total_tokens"
        if self._wall_time_exhausted():
            return "wall_time_exceeded"
        return None

    def _wall_time_exhausted(self) -> bool:
        started = self._started_monotonic
        if started is not None and self.limits.max_wall_time_seconds is not None:
            if time.monotonic() - started >= self.limits.max_wall_time_seconds:
                return True
        return False

    def _remaining_timeout(self, request: ActionRequest) -> float | None:
        values: list[float] = []
        if self._started_monotonic is not None and self.limits.max_wall_time_seconds is not None:
            values.append(self.limits.max_wall_time_seconds - (time.monotonic() - self._started_monotonic))
        request_remaining = request.seconds_remaining()
        if request_remaining is not None:
            values.append(float(request_remaining))
        if not values:
            return None
        return max(0.001, min(values))

    def _record_response_usage(self, response: Any) -> int | None:
        """Accumulate one provider response's usage without inventing tokens."""
        raw_usage = _response_field(response, "usage", {})
        usage = dict(raw_usage) if isinstance(raw_usage, Mapping) else {}
        for key, value in usage.items():
            parsed = _nonnegative_int(value)
            if parsed is not None:
                name = str(key)
                self._usage[name] = self._usage.get(name, 0) + parsed

        input_tokens = _usage_token_value(
            usage,
            "prompt_tokens",
            "input_tokens",
            "prompt_token_count",
        )
        output_tokens = _usage_token_value(
            usage,
            "completion_tokens",
            "output_tokens",
            "completion_token_count",
        )
        total_tokens = _usage_token_value(usage, "total_tokens", "total_token_count")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        if total_tokens is None:
            self._token_usage_complete = False
        else:
            self._total_tokens += total_tokens
        return total_tokens

    def _token_budget_failure(
        self,
        response_total_tokens: int | None,
    ) -> tuple[str, str, dict[str, Any]] | None:
        limit = self.limits.max_total_tokens
        if limit is None:
            return None
        if response_total_tokens is None and self.limits.require_token_usage:
            return (
                "token_usage_unavailable",
                "provider response did not include auditable token usage",
                {
                    "max_total_tokens": limit,
                    "generation_attempt": self._generation_attempt_count,
                },
            )
        if self._total_tokens > limit:
            return (
                "max_total_tokens",
                "agent session token budget exhausted: max_total_tokens",
                {
                    "max_total_tokens": limit,
                    "total_tokens": self._total_tokens,
                    "response_total_tokens": response_total_tokens,
                },
            )
        return None

    def _telemetry_snapshot(self) -> dict[str, Any]:
        elapsed = 0.0
        if self._started_monotonic is not None:
            elapsed = max(0.0, time.monotonic() - self._started_monotonic)
        return {
            "request_id": self._request.request_id if self._request is not None else None,
            "session_id": self.session_id,
            "seat": self.seat,
            "status": self.status.value,
            "elapsed_seconds": round(elapsed, 6),
            "generation_attempts": self._generation_attempt_count,
            "model_generations": self._model_generation_count,
            "generation_failures": self._generation_failure_count,
            "response_retries": self._response_retry_count,
            "tool_calls": self.tool_call_count,
            "tool_successes": self._tool_success_count,
            "tool_failures": self._tool_failure_count,
            "model_latency_seconds": round(self._model_latency_seconds, 6),
            "tool_latency_seconds": round(self._tool_latency_seconds, 6),
            "total_tokens": self._total_tokens,
            "token_usage_complete": self._token_usage_complete,
            "usage": dict(self._usage),
            "budget_exhausted": self._budget_failure_code,
            "limits": {
                "max_steps": self.limits.max_steps,
                "max_model_generations": self.limits.max_model_generations,
                "max_tool_calls": self.limits.max_tool_calls,
                "max_no_progress_steps": self.limits.max_no_progress_steps,
                "max_model_response_retries": self.limits.max_model_response_retries,
                "max_total_tokens": self.limits.max_total_tokens,
                "require_token_usage": self.limits.require_token_usage,
                "max_wall_time_seconds": self.limits.max_wall_time_seconds,
                "max_tool_time_seconds": self.limits.max_tool_time_seconds,
            },
        }

    def _emit_session_finished(self, telemetry: Mapping[str, Any]) -> None:
        if self._finish_emitted:
            return
        self._finish_emitted = True
        self._emit("agent_turn_finished", {"telemetry": deepcopy(dict(telemetry))})

    def _fail_budget(self, *, code: str = "agent_budget_exhausted") -> AgentSessionResult:
        if code == "agent_budget_exhausted":
            code = self._exhausted_budget_code() or code
        self._budget_failure_code = code
        return self._fail(code, f"agent session budget exhausted: {code}")

    def _fail(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> AgentSessionResult:
        error = AgentSessionError(code, message, details=details, cause=cause)
        self.status = SessionStatus.FAILED
        self._emit_failure(code, message)
        result = self._result(error=error)
        self._run_result = result
        return _clone_result(result)

    def _result(self, *, error: AgentSessionError | None = None) -> AgentSessionResult:
        telemetry = self._telemetry_snapshot()
        self._emit_session_finished(telemetry)
        return AgentSessionResult(
            session_id=self.session_id,
            seat=self.seat,
            status=self.status,
            decision=self.decision,
            terminal_value=self.decision,
            terminal_tool=self._terminal_tool,
            steps=self.steps,
            tool_calls=self.tool_call_count,
            generation_attempts=self._generation_attempt_count,
            model_generations=self._model_generation_count,
            generation_failures=self._generation_failure_count,
            response_retries=self._response_retry_count,
            tool_successes=self._tool_success_count,
            tool_failures=self._tool_failure_count,
            model_latency_seconds=round(self._model_latency_seconds, 6),
            tool_latency_seconds=round(self._tool_latency_seconds, 6),
            elapsed_seconds=float(telemetry["elapsed_seconds"]),
            token_usage_complete=self._token_usage_complete,
            total_tokens=self._total_tokens,
            no_progress_steps=self.no_progress_steps,
            state_version=self.state_version,
            error=error,
            failure=error,
            usage=dict(self._usage),
            history_compactions=self._history_compactions,
            max_compacted_tool_groups=self._max_compacted_tool_groups,
            peak_history_chars=self._peak_history_chars,
            peak_model_history_chars=self._peak_model_history_chars,
            history_limit_misses=self._history_limit_misses,
            telemetry=deepcopy(telemetry),
            _private_trace=deepcopy(self._private_trace),
        )


@dataclass(frozen=True)
class _NormalizedToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str
    parse_error: str | None = None


async def _coerce_terminal_submission(
    spec: ToolSpec,
    context: ToolExecutionContext,
    arguments: dict[str, Any],
    value: Any,
) -> TerminalSubmission | None:
    if spec.decision_builder is not None:
        built = spec.decision_builder(context, arguments, value)
        if inspect.isawaitable(built):
            built = await built
        value = built
    if isinstance(value, TerminalSubmission):
        return value
    if isinstance(value, Decision):
        return TerminalSubmission(value)
    if isinstance(value, Mapping):
        candidate = value.get("decision") if "decision" in value else value
        if isinstance(candidate, Decision):
            return TerminalSubmission(
                candidate,
                message=str(value.get("message") or "") or None,
                metadata=dict(value.get("metadata") or {}) if isinstance(value.get("metadata"), Mapping) else {},
            )
        try:
            return TerminalSubmission(Decision.model_validate(candidate))
        except Exception:
            return None
    return None


def _same_terminal_submission(
    first: TerminalSubmission,
    second: TerminalSubmission,
) -> bool:
    return (
        first.decision.model_dump(mode="json")
        == second.decision.model_dump(mode="json")
        and first.message == second.message
        and dict(first.metadata) == dict(second.metadata)
    )


def _decision_is_legal(request: ActionRequest, decision: Decision) -> bool:
    action = decision_action_value(decision)
    if action == AgentAction.SKIP.value:
        # Werewolf advertises optional abstention through ``can_skip`` on the
        # requested action; it does not add a separate ``skip`` LegalAction.
        return any(bool(item.can_skip) for item in request.legal_actions)
    legal = [item for item in request.legal_actions if str(item.action) == action]
    if not legal:
        return False
    descriptor = legal[0]
    if descriptor.requires_target:
        target = decision.target_seat
        return target is not None and target in set(descriptor.target_seats)
    if decision.target_seat is not None and descriptor.target_seats:
        return decision.target_seat in set(descriptor.target_seats)
    return True


def _preflight_terminal_arguments(
    spec: ToolSpec,
    request: ActionRequest,
    arguments: Mapping[str, Any],
) -> str | None:
    """Reject obviously illegal terminal targets before running side effects."""
    action_name = spec.terminal_action or spec.name
    descriptors = [item for item in request.legal_actions if str(item.action) == str(action_name)]
    if not descriptors:
        # The handler may intentionally map a differently named tool to an
        # action, so defer authoritative validation until it returns a
        # Decision.
        return None
    descriptor = descriptors[0]
    target = arguments.get("target_seat")
    if descriptor.requires_target:
        if target is None:
            return "terminal action requires a target seat"
        try:
            target_int = int(target)
        except (TypeError, ValueError):
            return "terminal target seat must be an integer"
        if target_int not in set(descriptor.target_seats):
            return "terminal target seat is not legal for this request"
    elif target is not None and descriptor.target_seats:
        try:
            target_int = int(target)
        except (TypeError, ValueError):
            return "terminal target seat must be an integer"
        if target_int not in set(descriptor.target_seats):
            return "terminal target seat is not legal for this request"
    return None


def _extract_tool_calls(response: Any) -> list[_NormalizedToolCall]:
    raw_calls = getattr(response, "tool_calls", None)
    if raw_calls is None and isinstance(response, Mapping):
        raw_calls = response.get("tool_calls")
    if not isinstance(raw_calls, (list, tuple)):
        return []
    result: list[_NormalizedToolCall] = []
    for index, raw in enumerate(raw_calls):
        if isinstance(raw, Mapping):
            call_id = str(raw.get("call_id") or raw.get("id") or f"tool-call-{index + 1}")
            function = raw.get("function")
            if isinstance(function, Mapping):
                name = str(raw.get("name") or function.get("name") or "")
                raw_arguments = function.get("arguments")
            else:
                name = str(raw.get("name") or "")
                raw_arguments = raw.get("raw_arguments", raw.get("arguments"))
        else:
            call_id = str(getattr(raw, "call_id", None) or getattr(raw, "id", None) or f"tool-call-{index + 1}")
            function = getattr(raw, "function", None)
            if function is not None:
                name = str(getattr(raw, "name", None) or getattr(function, "name", "") or "")
                raw_arguments = getattr(function, "arguments", None)
            else:
                name = str(getattr(raw, "name", "") or "")
                raw_arguments = getattr(raw, "raw_arguments", getattr(raw, "arguments", None))
        parse_error: str | None = None
        if isinstance(raw_arguments, Mapping):
            arguments = deepcopy(dict(raw_arguments))
            encoded = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        else:
            encoded = str(raw_arguments or "")
            try:
                parsed = json.loads(encoded)
            except (TypeError, ValueError):
                parsed = None
            if not isinstance(parsed, Mapping):
                arguments = {}
                parse_error = "tool arguments must be one complete JSON object"
            else:
                arguments = dict(parsed)
        result.append(_NormalizedToolCall(
            call_id=call_id,
            name=name,
            arguments=arguments,
            raw_arguments=encoded,
            parse_error=parse_error,
        ))
    return result


def _identity_fields_in_schema(schema: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    required = schema.get("required")
    if isinstance(required, list):
        found.update(
            str(name)
            for name in required
            if str(name).lower() in _IDENTITY_ARGUMENT_NAMES
        )
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for name, child in properties.items():
            if str(name).lower() in _IDENTITY_ARGUMENT_NAMES:
                found.add(str(name))
            if isinstance(child, Mapping):
                found.update(_identity_fields_in_schema(child))
    for key in ("items", "additionalProperties", "if", "then", "else", "not"):
        child = schema.get(key)
        if isinstance(child, Mapping):
            found.update(_identity_fields_in_schema(child))
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        children = schema.get(key)
        if isinstance(children, list):
            for child in children:
                if isinstance(child, Mapping):
                    found.update(_identity_fields_in_schema(child))
    return found


def _identity_fields_in_mapping(value: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    for key, child in value.items():
        if str(key).lower() in _IDENTITY_ARGUMENT_NAMES:
            found.add(str(key))
        if isinstance(child, Mapping):
            found.update(_identity_fields_in_mapping(child))
        elif isinstance(child, (list, tuple)):
            for item in child:
                if isinstance(item, Mapping):
                    found.update(_identity_fields_in_mapping(item))
    return found


def _empty_parameters_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _tool_error(
    call_id: str,
    name: str,
    kind: ToolKind | None,
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        call_id=str(call_id),
        name=str(name),
        kind=kind,
        ok=False,
        error_code=str(code),
        error_message=_bounded_private_text(message, _MAX_ERROR_CHARS),
        error_details=redact_sensitive(deepcopy(dict(details or {}))),
    )


async def _await_tool_with_timeout(
    awaitable: Awaitable[Any],
    timeout_seconds: float | None,
    *,
    session: "AgentSession | None" = None,
) -> Any:
    if timeout_seconds is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    if session is not None:
        session._tool_tasks.add(task)
    try:
        done, _pending = await asyncio.wait({task}, timeout=timeout_seconds)
        if task in done:
            return task.result()

        # Do not use ``wait_for`` here: Python waits for a cancelled coroutine
        # to finish, so a handler that swallows CancelledError could defeat the
        # tool bound.  A second bounded wait lets cooperative handlers cleanly
        # exit while keeping hostile ones attributable and detached.
        task.cancel()
        done, _pending = await asyncio.wait(
            {task},
            timeout=_TOOL_CANCELLATION_GRACE_SECONDS,
        )
        if task not in done:
            if session is not None:
                session._unresolved_tool_tasks.add(task)
                session._tool_tasks.discard(task)
                task.add_done_callback(session._forget_tool_task)
            raise ToolExecutionError(
                "tool_timeout",
                f"tool execution exceeded {timeout_seconds:g} seconds",
                details={
                    "timeout_seconds": timeout_seconds,
                    "cleanup_pending": True,
                },
            )
        _consume_task_result(task)
        raise ToolExecutionError(
            "tool_timeout",
            f"tool execution exceeded {timeout_seconds:g} seconds",
            details={"timeout_seconds": timeout_seconds},
        )
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
            done, _pending = await asyncio.wait(
                {task},
                timeout=_TOOL_CANCELLATION_GRACE_SECONDS,
            )
            if task not in done and session is not None:
                session._unresolved_tool_tasks.add(task)
                session._tool_tasks.discard(task)
                task.add_done_callback(session._forget_tool_task)
        raise
    finally:
        if task.done():
            _consume_task_result(task)
            if session is not None:
                session._tool_tasks.discard(task)
                session._unresolved_tool_tasks.discard(task)


class _NullAsyncLock:
    async def __aenter__(self) -> "_NullAsyncLock":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None


def _null_async_lock() -> _NullAsyncLock:
    return _NullAsyncLock()


def _terminal_lock(context: ToolExecutionContext) -> asyncio.Lock:
    if context._session is None:
        # A lock that allows validation/handler execution, followed by the
        # explicit terminal_context_missing result in ``execute``.
        return asyncio.Lock()
    return context._session._terminal_lock


def _initial_observation_message(request: ActionRequest) -> str:
    payload = {
        "type": "agent_observation",
        "request_id": request.request_id,
        "phase": request.phase,
        "day": request.day,
        "action_kind": request.action_kind,
        "observation": _private_trace_view(request.observation),
        "legal_actions": [item.model_dump(mode="json") for item in request.legal_actions],
        "private_context": _private_trace_view(request.private_context),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _model_history_view(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_chars: int | None,
    keep_recent_tool_groups: int,
) -> tuple[list[dict[str, Any]], dict[str, int | bool]]:
    """Build a bounded model view without mutating the audit history.

    Tool protocols require every assistant function call to stay adjacent to a
    corresponding tool result.  Compaction therefore removes only *complete*
    exchange groups and replaces the whole group with one ordinary user
    history summary.  Incomplete or malformed groups are retained verbatim so
    this layer can never manufacture an orphaned call/result pair.
    """
    original = [deepcopy(dict(message)) for message in messages]
    original_chars = _history_chars(original)
    base_stats: dict[str, int | bool] = {
        "original_chars": original_chars,
        "model_chars": original_chars,
        "compacted_tool_groups": 0,
        "limit_satisfied": max_chars is None or original_chars <= max_chars,
    }
    if max_chars is None or original_chars <= max_chars:
        return original, base_stats

    groups = _completed_tool_history_groups(original)
    eligible = groups[:-keep_recent_tool_groups]
    if not eligible:
        return original, base_stats

    selected: set[tuple[int, int]] = set()
    current = original
    current_chars = original_chars
    for start, end in eligible:
        proposed = set(selected)
        proposed.add((start, end))
        candidate = _replace_tool_history_groups(original, proposed)
        candidate_chars = _history_chars(candidate)
        # A tiny exchange can be shorter than its structured summary.  Keep it
        # untouched unless replacing the atomic group actually reduces input.
        if candidate_chars >= current_chars:
            continue
        selected = proposed
        current = candidate
        current_chars = candidate_chars
        if current_chars <= max_chars:
            break

    return current, {
        "original_chars": original_chars,
        "model_chars": current_chars,
        "compacted_tool_groups": len(selected),
        "limit_satisfied": current_chars <= max_chars,
    }


def _completed_tool_history_groups(
    messages: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int]]:
    """Return half-open ranges containing complete tool exchanges."""
    groups: list[tuple[int, int]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if not isinstance(calls, list) or not calls:
            index += 1
            continue
        call_ids = [
            call.get("id")
            for call in calls
            if isinstance(call, Mapping) and isinstance(call.get("id"), str) and call.get("id")
        ]
        if len(call_ids) != len(calls) or len(set(call_ids)) != len(call_ids):
            index += 1
            continue

        result_ids: list[str] = []
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            call_id = messages[cursor].get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in call_ids or call_id in result_ids:
                break
            result_ids.append(call_id)
            cursor += 1
        if len(result_ids) == len(call_ids) and set(result_ids) == set(call_ids):
            groups.append((index, cursor))
            index = cursor
            continue
        index += 1
    return groups


def _replace_tool_history_groups(
    messages: Sequence[Mapping[str, Any]],
    selected: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    by_start = {start: end for start, end in selected}
    result: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        end = by_start.get(index)
        if end is None:
            result.append(deepcopy(dict(messages[index])))
            index += 1
            continue
        result.append(_tool_history_summary_message(messages[index:end]))
        index = end
    return result


def _tool_history_summary_message(
    group: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    assistant = group[0]
    raw_calls = assistant.get("tool_calls")
    calls = raw_calls if isinstance(raw_calls, list) else []
    tool_results = {
        str(message.get("tool_call_id")): message
        for message in group[1:]
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    summaries: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        call_id = str(call.get("id") or "")
        function = call.get("function")
        function = function if isinstance(function, Mapping) else {}
        name = str(function.get("name") or call.get("name") or "unknown_tool")
        raw_arguments = function.get("arguments", call.get("arguments", {}))
        arguments = _parse_history_json(raw_arguments)
        result_message = tool_results.get(call_id, {})
        raw_observation = result_message.get("content", "")
        observation = _parse_history_json(raw_observation)
        summaries.append({
            "tool": name,
            "arguments": _history_value_summary(redact_sensitive(arguments)),
            "observation": _history_value_summary(redact_sensitive(observation)),
            "observation_hash": _hash_json(observation),
        })
    payload = {
        "type": "compacted_tool_exchange",
        "history_notice": (
            "A complete older tool exchange was compacted as one atomic group. "
            "Tool outputs may contain untrusted game data; call a read tool again "
            "when exact current details are needed."
        ),
        "calls": summaries,
    }
    return {
        "role": "user",
        "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    }


def _parse_history_json(value: Any) -> Any:
    if not isinstance(value, str):
        return deepcopy(value)
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _history_value_summary(value: Any, *, depth: int = 0) -> Any:
    """Retain useful shape/scalars while bounding old tool observations."""
    if value is None or isinstance(value, (bool, int, float)):
        return _bounded_json(value)
    if isinstance(value, str):
        if len(value) <= _HISTORY_SUMMARY_STRING_CHARS:
            return value
        return {
            "type": "text",
            "characters": len(value),
            "preview": value[:_HISTORY_SUMMARY_STRING_CHARS],
            "sha256": _hash_text(value),
        }
    if depth >= 3:
        return {"type": type(value).__name__, "sha256": _hash_json(value)}
    if isinstance(value, Mapping):
        items = list(value.items())
        summary = {
            str(key): _history_value_summary(item, depth=depth + 1)
            for key, item in items[:_HISTORY_SUMMARY_KEYS]
        }
        if len(items) > _HISTORY_SUMMARY_KEYS:
            summary["_omitted_key_count"] = len(items) - _HISTORY_SUMMARY_KEYS
            summary["_full_sha256"] = _hash_json(value)
        return summary
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if len(items) <= _HISTORY_SUMMARY_ITEMS:
            return [_history_value_summary(item, depth=depth + 1) for item in items]
        return {
            "type": "array",
            "count": len(items),
            "latest_items": [
                _history_value_summary(item, depth=depth + 1)
                for item in items[-_HISTORY_SUMMARY_ITEMS:]
            ],
            "sha256": _hash_json(value),
        }
    return _history_value_summary(str(value), depth=depth)


def _history_chars(messages: Sequence[Mapping[str, Any]]) -> int:
    try:
        return len(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str))
    except Exception:
        return len(repr(messages))


def _clone_action_request(request: ActionRequest) -> ActionRequest:
    """Deep-clone a Pydantic request, including nested mutable containers."""
    # ``model_copy(deep=True)`` preserves the validated protocol type while
    # detaching dict/list fields that Pydantic's ``frozen`` flag leaves
    # mutable.  Keep a defensive fallback for protocol-compatible test
    # doubles that implement only ``model_dump``/``model_validate``.
    try:
        return request.model_copy(deep=True)
    except (AttributeError, TypeError):
        return ActionRequest.model_validate(deepcopy(request.model_dump(mode="python")))


def _clone_result(result: AgentSessionResult) -> AgentSessionResult:
    cloned_error = None
    if result.error is not None:
        cloned_error = AgentSessionError(
            result.error.code,
            str(result.error),
            details=result.error.details,
            cause=result.error.cause,
        )
    return AgentSessionResult(
        session_id=result.session_id,
        seat=result.seat,
        status=result.status,
        decision=Decision.model_validate(result.decision.model_dump(mode="python")) if result.decision else None,
        terminal_value=(
            Decision.model_validate(result.terminal_value.model_dump(mode="python"))
            if isinstance(result.terminal_value, Decision)
            else deepcopy(result.terminal_value)
        ),
        terminal_tool=result.terminal_tool,
        steps=result.steps,
        tool_calls=result.tool_calls,
        generation_attempts=result.generation_attempts,
        model_generations=result.model_generations,
        generation_failures=result.generation_failures,
        response_retries=result.response_retries,
        tool_successes=result.tool_successes,
        tool_failures=result.tool_failures,
        model_latency_seconds=result.model_latency_seconds,
        tool_latency_seconds=result.tool_latency_seconds,
        elapsed_seconds=result.elapsed_seconds,
        token_usage_complete=result.token_usage_complete,
        total_tokens=result.total_tokens,
        no_progress_steps=result.no_progress_steps,
        state_version=result.state_version,
        error=cloned_error,
        failure=cloned_error,
        usage=dict(result.usage),
        history_compactions=result.history_compactions,
        max_compacted_tool_groups=result.max_compacted_tool_groups,
        peak_history_chars=result.peak_history_chars,
        peak_model_history_chars=result.peak_model_history_chars,
        history_limit_misses=result.history_limit_misses,
        telemetry=deepcopy(result.telemetry),
        _private_trace=deepcopy(result._private_trace),
    )


def _response_field(response: Any, name: str, default: Any = None) -> Any:
    if isinstance(response, Mapping):
        return response.get(name, default)
    return getattr(response, name, default)


def _nonnegative_int(value: Any) -> int | None:
    """Parse provider counters conservatively; booleans/fractions are unknown."""
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed < 0:
        return None
    try:
        if float(value) != parsed:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed


def _usage_token_value(usage: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in usage:
            value = _nonnegative_int(usage.get(key))
            if value is not None:
                return value
    return None


def _is_model_response_error(error: BaseException) -> bool:
    """Identify Router response-shape failures without retrying transport faults."""
    try:
        from ..llm.router import LLMResponseError
    except Exception:
        return False
    return isinstance(error, LLMResponseError)


def _private_trace_view(value: Any) -> Any:
    """Detach, redact and globally bound one private God/Admin value."""
    safe = _bounded_json(redact_sensitive(value))
    return _fit_private_trace_value(safe, limit=_MAX_PRIVATE_TRACE_VALUE_CHARS)


def _bounded_trace_row(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return one credential-safe trace row with a hard serialized bound.

    The row keeps protocol identity fields intact.  Oversized arbitrary values
    become a digest marker computed from the already-redacted representation,
    so a marker cannot become an offline oracle for a raw credential.
    """
    redacted = redact_sensitive(deepcopy(dict(value)))
    safe = {
        str(key): _fit_private_trace_value(
            _bounded_json(item),
            limit=_MAX_PRIVATE_TRACE_VALUE_CHARS,
        )
        for key, item in redacted.items()
    }
    encoded = _trace_json_text(safe)
    if len(encoded) <= _MAX_PRIVATE_TRACE_ROW_CHARS:
        return safe

    essential = {
        "type", "visibility", "audience", "session_id", "seat",
        "state_version", "request_id", "turn_id", "step", "phase", "day",
        "call_id", "tool", "kind", "ok", "terminal", "action",
        "target_seat", "response_attempt", "will_retry", "error_code",
        "arguments_hash", "output_hash", "request_hash", "response_hash",
    }
    original_chars = len(encoded)
    original_hash = _hash_text(encoded)
    candidates = sorted(
        (key for key in safe if key not in essential),
        key=lambda key: len(_trace_json_text(safe[key])),
        reverse=True,
    )
    for key in candidates:
        field_text = _trace_json_text(safe[key])
        if len(field_text) <= 256:
            continue
        safe[key] = _private_trace_truncation_marker(field_text)
        if len(_trace_json_text(safe)) <= _MAX_PRIVATE_TRACE_ROW_CHARS:
            return safe

    # Pathological custom metadata can make even field-level markers too wide.
    # Preserve the linkage needed by verifiers and make the loss explicit.
    minimal = {
        key: _bounded_trace_metadata(safe[key])
        for key in essential
        if key in safe
    }
    minimal.update({
        "trace_truncated": True,
        "trace_original_chars": original_chars,
        "trace_sha256": original_hash,
    })
    return minimal


def _fit_private_trace_value(value: Any, *, limit: int) -> Any:
    encoded = _trace_json_text(value)
    if len(encoded) <= limit:
        return value
    return _private_trace_truncation_marker(encoded)


def _private_trace_truncation_marker(encoded: str) -> dict[str, Any]:
    return {
        "type": "trace_value_truncated",
        "characters": len(encoded),
        "sha256": _hash_text(encoded),
    }


def _bounded_trace_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _bounded_text(value, 512)


def _trace_json_text(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        return repr(value)


def _bounded_private_text(value: Any, limit: int) -> str:
    safe = redact_sensitive(str(value or ""))
    return _bounded_text(safe, limit)


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "")
    return text[:limit]


def _bounded_json(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        return value[:_MAX_RESULT_CHARS]
    if depth >= 5:
        return _bounded_text(value, _MAX_RESULT_CHARS)
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_json(item, depth=depth + 1)
            for key, item in list(value.items())[:100]
        }
    if isinstance(value, (list, tuple, set)):
        return [_bounded_json(item, depth=depth + 1) for item in list(value)[:100]]
    return _bounded_text(value, _MAX_RESULT_CHARS)


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    """Retrieve a finished task result so detached failures stay observed."""
    if not task.done():
        return
    try:
        task.result()
    except BaseException:
        return


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _hash_json(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        encoded = repr(value)
    return _hash_text(encoded)


# Compatibility aliases make the primitive discoverable under the terminology
# used by different harness integrations.
AgentToolRegistry = ToolRegistry
AgentToolSpec = ToolSpec
AgentToolKind = ToolKind
ToolLoopSession = AgentSession


__all__ = [
    "AgentSession",
    "AgentSessionError",
    "AgentSessionLimits",
    "AgentSessionResult",
    "AgentToolKind",
    "AgentToolRegistry",
    "AgentToolSpec",
    "SessionStatus",
    "TerminalSubmission",
    "ToolExecutionContext",
    "ToolExecutionError",
    "ToolKind",
    "ToolLoopSession",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
]
