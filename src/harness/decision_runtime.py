"""Single execution path for one ActionRequest and its terminal response."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from copy import deepcopy
import inspect
import logging
import math
import time
from typing import Any

from .errors import AgentDecisionError
from .transcript import redact_sensitive

TraceCallback = Callable[[dict[str, Any]], None]
EnvelopeValidator = Callable[[Any, Any], Any]
logger = logging.getLogger(__name__)


class DecisionRuntimeCleanupError(RuntimeError):
    """DecisionRuntime still owns tasks after bounded cooperative cleanup."""


class DecisionRuntime:
    """Invoke an AgentProtocol once, enforce its deadline, validate, and trace.

    Provider/response retries remain inside the Agent adapter. This runtime owns
    the environment boundary only: exactly one request row followed by one
    terminal row containing a validated/rejected envelope, a boundary failure,
    a cancellation, or an envelope-validator failure. Deadline cancellation is
    cooperative and bounded by ``cancellation_grace_seconds``; a task that
    remains alive is a fatal cleanup failure, never a successful decision or
    implicit SKIP.
    """

    def __init__(
        self,
        *,
        on_trace: TraceCallback,
        envelope_type: type[Any] | None = None,
        validate_envelope: EnvelopeValidator | None = None,
        default_timeout_seconds: float | None = None,
        cancellation_grace_seconds: float = 1.0,
        expected_run_id: str | None = None,
    ) -> None:
        if (envelope_type is None) != (validate_envelope is None):
            raise ValueError("envelope_type and validate_envelope must be supplied together")
        if envelope_type is None:
            # Compatibility is resolved lazily so the reusable runtime module
            # does not import the Werewolf decision schema at module import.
            from .agent_protocol import DecisionEnvelope
            from .agents import validate_decision_against_legal_actions

            envelope_type = DecisionEnvelope
            validate_envelope = validate_decision_against_legal_actions
        self._on_trace = on_trace
        self._trace_listeners: list[TraceCallback] = []
        self._envelope_type = envelope_type
        self._validate_envelope = validate_envelope
        self._default_timeout_seconds = _optional_positive_duration(
            default_timeout_seconds,
            name="default_timeout_seconds",
        )
        self._cancellation_grace_seconds = _bounded_nonnegative_duration(
            cancellation_grace_seconds,
            name="cancellation_grace_seconds",
            maximum=60.0,
        )
        if expected_run_id is None:
            self._expected_run_id = None
        else:
            normalized_run_id = str(expected_run_id).strip()
            if not normalized_run_id:
                raise ValueError("expected_run_id must not be empty")
            self._expected_run_id = normalized_run_id
        self._accepted_request_ids: set[str] = set()
        # Concurrent calls are accepted deterministically, but their agents may
        # finish in any order. Commit terminal rows in acceptance order so an
        # equivalent run produces the same append-only evidence transcript.
        self._request_trace_sequence: dict[str, int] = {}
        self._terminal_trace_buffer: dict[int, dict[str, Any]] = {}
        self._next_terminal_trace_sequence = 1
        self._unresolved_tasks: dict[asyncio.Future[Any], str] = {}

    async def execute(self, actor: Any, request: Any) -> Any:
        started_monotonic = time.monotonic()
        request = self._apply_default_deadline(request)
        request = _mark_deadline_owner(request)
        trusted_request = _deep_copy_protocol_value(request)
        if trusted_request.request_id in self._accepted_request_ids:
            raise _decision_error(
                f"duplicate ActionRequest request_id: {trusted_request.request_id}",
                error_type="DuplicateRequestId",
                request=trusted_request,
            )
        # There is no await between the check and insertion, so concurrent
        # decisions on one event loop cannot accept the same request ID twice.
        self._accepted_request_ids.add(trusted_request.request_id)
        self._request_trace_sequence[trusted_request.request_id] = len(
            self._accepted_request_ids
        )
        self._emit_trace({
            "kind": "agent_request",
            "request": trusted_request.model_dump(),
        })
        try:
            self._assert_request_run_id(trusted_request)
            _assert_actor_matches_request(actor, trusted_request)
            envelope = await self._invoke_with_deadline(
                actor,
                _deep_copy_protocol_value(trusted_request),
            )
        except asyncio.CancelledError as err:
            # Cancellation is terminal for pairing but remains distinct from an
            # Agent/provider failure and must propagate to the owning run.
            self._trace_cancelled(
                trusted_request,
                actor,
                err,
                started_monotonic=started_monotonic,
            )
            raise
        except Exception as err:  # noqa: BLE001 - normalize boundary failures
            _attach_request_context(err, trusted_request)
            self._trace_failure(
                trusted_request,
                actor,
                err,
                started_monotonic=started_monotonic,
            )
            raise

        if not isinstance(envelope, self._envelope_type):
            err = _decision_error(
                "AgentProtocol.decide returned an unexpected envelope type "
                f"(expected {self._envelope_type.__name__})",
                error_type="DecisionEnvelopeTypeError",
                request=trusted_request,
            )
            self._trace_failure(
                trusted_request,
                actor,
                err,
                started_monotonic=started_monotonic,
            )
            raise err

        trusted_envelope = _deep_copy_protocol_value(envelope)
        try:
            validation = self._validate_envelope(
                trusted_envelope,
                _deep_copy_protocol_value(trusted_request),
            )
            if not hasattr(validation, "valid") or not callable(
                getattr(validation, "model_dump", None)
            ):
                raise TypeError("envelope validator returned an invalid result")
        except Exception as err:  # noqa: BLE001 - normalize validator defects
            self._trace_validation_failure(
                trusted_request,
                actor,
                trusted_envelope,
                err,
                started_monotonic=started_monotonic,
            )
            failure = _decision_error(
                "DecisionEnvelope validator failed",
                error_type="DecisionValidatorError",
                request=trusted_request,
            )
            setattr(failure, "validator_error_type", type(err).__name__)
            raise failure from err
        response_trace: dict[str, Any] = {
            "kind": "agent_response",
            "request_id": trusted_request.request_id,
            "envelope": trusted_envelope.model_dump(),
            "validation": validation.model_dump(),
            "request_telemetry": _request_telemetry(
                started_monotonic,
                outcome="accepted" if validation.valid else "rejected",
            ),
        }
        _attach_actor_trace_identity(response_trace, actor, trusted_request)
        self._emit_terminal_trace(response_trace)
        if not validation.valid:
            raise _rejected_envelope_error(trusted_request, validation)
        return trusted_envelope

    def _assert_request_run_id(self, request: Any) -> None:
        """Bind a generic runtime to the run that owns its transcript.

        Legacy room runtimes leave ``expected_run_id`` unset.  Generic core
        runs always set it, so an environment plugin cannot execute a request
        labelled as belonging to another run and contaminate replay evidence.
        """
        if self._expected_run_id is None:
            return
        actual = getattr(request, "run_id", None)
        if not isinstance(actual, str) or actual != self._expected_run_id:
            raise _decision_error(
                "ActionRequest run_id does not match the owning harness run",
                error_type="RunIdMismatch",
                request=request,
            )

    async def _invoke_with_deadline(self, actor: Any, request: Any) -> Any:
        decide = getattr(actor, "decide", None)
        if not callable(decide):
            raise _decision_error(
                f"agent {_request_actor_label(request)} does not implement AgentProtocol.decide",
                error_type="AgentProtocolMissing",
                request=request,
            )

        remaining = request.seconds_remaining()
        if remaining is not None and remaining <= 0:
            deadline = _deadline_error(
                request,
                0.0,
                when="before decision start",
                actor=actor,
            )
            hook_failure = await self._notify_agent_timeout(actor, request)
            if hook_failure is not None:
                raise _deadline_cleanup_error(deadline, [hook_failure])
            raise deadline

        decision = decide(request)
        if not inspect.isawaitable(decision):
            raise _decision_error(
                "AgentProtocol.decide must return an awaitable",
                error_type="AgentProtocolInvalid",
                request=request,
            )
        task = asyncio.ensure_future(decision)
        _set_task_name(task, f"decision:{request.request_id}")
        try:
            if remaining is None:
                # Shield lets this boundary receive external cancellation
                # immediately and then reclaim the child within a bounded grace.
                return await asyncio.shield(task)
            done, _pending = await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError as err:
            terminated, _interrupted = await self._cancel_task(task)
            if not terminated:
                self._track_unresolved(task, f"decision:{request.request_id}")
                setattr(err, "cleanup_pending_task_count", 1)
            raise

        if task in done:
            return task.result()

        terminated, caller_cancelled = await self._cancel_task(task)
        cleanup_failures: list[dict[str, Any]] = []
        if not terminated:
            self._track_unresolved(task, f"decision:{request.request_id}")
            cleanup_failures.append({
                "stage": "agent_decide",
                "error_type": "TaskIgnoredCancellation",
                "pending_task_count": 1,
            })
        if caller_cancelled:
            cancelled = asyncio.CancelledError()
            if not terminated:
                setattr(cancelled, "cleanup_pending_task_count", 1)
            raise cancelled
        hook_failure = await self._notify_agent_timeout(actor, request)
        if hook_failure is not None:
            cleanup_failures.append(hook_failure)
        deadline = _deadline_error(
            request,
            remaining,
            when="during decision",
            actor=actor,
        )
        if cleanup_failures:
            raise _deadline_cleanup_error(deadline, cleanup_failures)
        raise deadline

    async def _cancel_task(
        self,
        task: asyncio.Future[Any],
    ) -> tuple[bool, bool]:
        """Cancel twice within one grace, deferring repeated caller cancels."""
        if task.done():
            _consume_task_result(task)
            return True, False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._cancellation_grace_seconds
        task.cancel()
        first_budget = self._cancellation_grace_seconds / 2
        caller_cancelled = await _wait_for_task_until(
            task,
            loop.time() + first_budget,
        )
        if not task.done():
            # A second cancellation interrupts cleanup code that swallowed the
            # first CancelledError. Both waits share the configured grace.
            task.cancel()
            caller_cancelled = (
                await _wait_for_task_until(task, deadline)
                or caller_cancelled
            )
        if not task.done():
            return False, caller_cancelled
        _consume_task_result(task)
        return True, caller_cancelled

    async def _notify_agent_timeout(
        self,
        actor: Any,
        request: Any,
    ) -> dict[str, Any] | None:
        """Run the optional timeout hook within the same cleanup contract."""
        callback = getattr(actor, "on_decision_timeout", None)
        if not callable(callback):
            return None
        try:
            result = callback(request)
        except asyncio.CancelledError:
            return {
                "stage": "agent_timeout_hook",
                "error_type": "CallbackCancelled",
                "pending_task_count": 0,
            }
        except Exception:  # noqa: BLE001 - keep provider details out of traces
            logger.warning(
                "agent timeout lifecycle hook failed (actor=%s request_id=%s error_type=%s)",
                _actor_identity(actor, request),
                request.request_id,
                "CallbackError",
            )
            return {
                "stage": "agent_timeout_hook",
                "error_type": "CallbackError",
                "pending_task_count": 0,
            }
        if not inspect.isawaitable(result):
            return None
        task = asyncio.ensure_future(result)
        _set_task_name(task, f"decision-timeout-hook:{request.request_id}")
        try:
            done, _pending = await asyncio.wait(
                {task},
                timeout=self._cancellation_grace_seconds,
            )
        except asyncio.CancelledError:
            terminated, _interrupted = await self._cancel_task(task)
            if not terminated:
                self._track_unresolved(
                    task,
                    f"decision-timeout-hook:{request.request_id}",
                )
            raise
        if task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                return {
                    "stage": "agent_timeout_hook",
                    "error_type": "CallbackCancelled",
                    "pending_task_count": 0,
                }
            except Exception:  # noqa: BLE001 - keep hook detail private
                logger.warning(
                    "agent timeout lifecycle hook failed (actor=%s request_id=%s error_type=%s)",
                    _actor_identity(actor, request),
                    request.request_id,
                    "CallbackError",
                )
                return {
                    "stage": "agent_timeout_hook",
                    "error_type": "CallbackError",
                    "pending_task_count": 0,
                }
            return None
        terminated, caller_cancelled = await self._cancel_task(task)
        if not terminated:
            self._track_unresolved(task, f"decision-timeout-hook:{request.request_id}")
        if caller_cancelled:
            cancelled = asyncio.CancelledError()
            if not terminated:
                setattr(cancelled, "cleanup_pending_task_count", 1)
            raise cancelled
        return {
            "stage": "agent_timeout_hook",
            "error_type": (
                "TaskIgnoredCancellation" if not terminated else "CleanupTimeout"
            ),
            "pending_task_count": 0 if terminated else 1,
        }

    def _apply_default_deadline(self, request: Any) -> Any:
        if request.deadline_monotonic is not None or self._default_timeout_seconds is None:
            return request
        metadata = dict(request.metadata)
        metadata.setdefault("deadline_source", "decision")
        metadata.setdefault("effective_timeout_seconds", self._default_timeout_seconds)
        return request.model_copy(update={
            "deadline_monotonic": (
                asyncio.get_running_loop().time() + self._default_timeout_seconds
            ),
            "metadata": metadata,
        })

    def _track_unresolved(self, task: asyncio.Future[Any], owner: str) -> None:
        if task.done():
            _consume_task_result(task)
            return
        self._unresolved_tasks[task] = owner

        def forget(done: asyncio.Future[Any]) -> None:
            self._unresolved_tasks.pop(done, None)
            _consume_task_result(done)

        task.add_done_callback(forget)
        logger.critical(
            "decision task ignored bounded cancellation (owner=%s)",
            owner,
        )

    @property
    def unresolved_task_count(self) -> int:
        return sum(not task.done() for task in self._unresolved_tasks)

    @property
    def unresolved_task_details(self) -> list[dict[str, Any]]:
        return [
            {"owner": owner, "task_name": _task_name(task)}
            for task, owner in self._unresolved_tasks.items()
            if not task.done()
        ]

    @property
    def unresolved_task_items(
        self,
    ) -> tuple[tuple[asyncio.Future[Any], str], ...]:
        """Return live task identities for transfer to a lifecycle owner."""
        return tuple(
            (task, owner)
            for task, owner in self._unresolved_tasks.items()
            if not task.done()
        )

    async def aclose(self) -> None:
        """Retry bounded cancellation for every task that previously resisted."""
        tasks = [task for task in self._unresolved_tasks if not task.done()]
        caller_cancelled = False
        for task in tasks:
            _terminated, interrupted = await self._cancel_task(task)
            caller_cancelled = caller_cancelled or interrupted
        pending = self.unresolved_task_details
        if caller_cancelled:
            raise asyncio.CancelledError
        if pending:
            raise DecisionRuntimeCleanupError(
                f"DecisionRuntime cleanup left {len(pending)} task(s) pending"
            )

    def _trace_failure(
        self,
        request: Any,
        actor: Any,
        err: BaseException,
        *,
        started_monotonic: float,
    ) -> None:
        error_type = str(getattr(err, "error_type", type(err).__name__))
        timeout = bool(getattr(err, "timeout", False))
        stage = _request_stage(request)
        action = _request_action(request)
        failure: dict[str, Any] = {
            "error_type": error_type,
            "timeout": timeout,
            "reason": (
                str(err)
                if isinstance(err, AgentDecisionError) and timeout
                else f"{error_type} during {stage}/{action}"
            ),
        }
        timeout_seconds = getattr(err, "timeout_seconds", None)
        if timeout_seconds is not None:
            failure["timeout_seconds"] = float(timeout_seconds)
        llm_call_attempts = getattr(err, "llm_call_attempts", None)
        if isinstance(llm_call_attempts, list):
            failure["llm_call_attempts"] = redact_sensitive(llm_call_attempts)
        cleanup_failures = getattr(err, "cleanup_failures", None)
        if isinstance(cleanup_failures, list):
            failure["cleanup"] = {
                "fatal": bool(getattr(err, "fatal_cleanup_failure", False)),
                "failures": redact_sensitive(cleanup_failures),
            }
        failure_trace: dict[str, Any] = {
            "kind": "agent_response_failed",
            "request_id": request.request_id,
            "stage": stage,
            "action": action,
            "failure": failure,
            "request_telemetry": _request_telemetry(
                started_monotonic,
                outcome="failed",
            ),
        }
        session_telemetry = getattr(err, "agent_session_telemetry", None)
        if isinstance(session_telemetry, dict):
            failure_trace["agent_session"] = redact_sensitive(
                deepcopy(session_telemetry)
            )
        legacy_phase = getattr(request, "phase", None)
        if legacy_phase is not None:
            failure_trace["phase"] = str(legacy_phase)
        _attach_actor_trace_identity(failure_trace, actor, request)
        self._emit_terminal_trace(failure_trace)

    def _trace_cancelled(
        self,
        request: Any,
        actor: Any,
        err: BaseException,
        *,
        started_monotonic: float,
    ) -> None:
        cancellation: dict[str, Any] = {"reason": "run_or_room_cancelled"}
        pending = getattr(err, "cleanup_pending_task_count", None)
        if pending:
            cancellation["cleanup"] = {
                "fatal": True,
                "pending_task_count": int(pending),
            }
        trace: dict[str, Any] = {
            "kind": "agent_response_cancelled",
            "request_id": request.request_id,
            "stage": _request_stage(request),
            "action": _request_action(request),
            "cancellation": cancellation,
            "request_telemetry": _request_telemetry(
                started_monotonic,
                outcome="cancelled",
            ),
        }
        legacy_phase = getattr(request, "phase", None)
        if legacy_phase is not None:
            trace["phase"] = str(legacy_phase)
        _attach_actor_trace_identity(trace, actor, request)
        self._emit_terminal_trace(trace)

    def _trace_validation_failure(
        self,
        request: Any,
        actor: Any,
        envelope: Any,
        err: BaseException,
        *,
        started_monotonic: float,
    ) -> None:
        trace: dict[str, Any] = {
            "kind": "agent_response_validation_failed",
            "request_id": request.request_id,
            "stage": _request_stage(request),
            "action": _request_action(request),
            "envelope": envelope.model_dump(),
            "failure": {
                "error_type": type(err).__name__,
                "reason": "DecisionEnvelope validator failed.",
            },
            "request_telemetry": _request_telemetry(
                started_monotonic,
                outcome="validation_failed",
            ),
        }
        legacy_phase = getattr(request, "phase", None)
        if legacy_phase is not None:
            trace["phase"] = str(legacy_phase)
        _attach_actor_trace_identity(trace, actor, request)
        self._emit_terminal_trace(trace)

    def add_trace_listener(self, listener: TraceCallback) -> None:
        if listener is self._on_trace or listener in self._trace_listeners:
            return
        self._trace_listeners.append(listener)

    def _emit_trace(self, payload: dict[str, Any]) -> None:
        self._on_trace(payload)
        for listener in tuple(self._trace_listeners):
            listener(payload)

    def _emit_terminal_trace(self, payload: dict[str, Any]) -> None:
        """Commit concurrent terminal rows in deterministic acceptance order."""
        request_id = str(payload.get("request_id") or "")
        sequence = self._request_trace_sequence.get(request_id)
        if sequence is None:
            raise RuntimeError("terminal trace does not belong to an accepted request")
        if sequence in self._terminal_trace_buffer:
            raise RuntimeError("request produced more than one terminal trace")
        self._terminal_trace_buffer[sequence] = payload
        while self._next_terminal_trace_sequence in self._terminal_trace_buffer:
            terminal = self._terminal_trace_buffer.pop(
                self._next_terminal_trace_sequence
            )
            self._next_terminal_trace_sequence += 1
            self._emit_trace(terminal)


def _rejected_envelope_error(
    request: Any,
    validation: Any,
) -> AgentDecisionError:
    codes = ",".join(issue.code for issue in validation.issues)
    return _decision_error(
        f"invalid decision envelope: {codes}",
        error_type="DecisionEnvelopeRejected",
        request=request,
    )


def _deadline_error(
    request: Any,
    remaining: float,
    *,
    when: str,
    actor: Any | None = None,
) -> AgentDecisionError:
    source = str(request.metadata.get("deadline_source") or "decision")
    stage = _request_stage(request)
    action = _request_action(request)
    if source == "phase":
        error_type = "PhaseDeadlineExceeded"
        message = (
            f"{stage}/{action} phase deadline exhausted "
            f"{when} after {{timeout_seconds:.3f}}s"
        )
    elif bool(getattr(actor, "is_human", False)):
        error_type = "HumanDecisionTimeout"
        message = (
            f"human decision timeout for request {request.request_id} "
            f"after {{timeout_seconds:.3f}}s"
        )
    else:
        error_type = "DecisionTimeout"
        message = (
            f"{stage}/{action} decision timeout "
            f"after {{timeout_seconds:.3f}}s"
        )
    configured = request.metadata.get("effective_timeout_seconds")
    try:
        timeout_seconds = float(configured)
    except (TypeError, ValueError):
        timeout_seconds = max(0.0, float(remaining))
    err = _decision_error(
        message.format(timeout_seconds=timeout_seconds),
        error_type=error_type,
        request=request,
    )
    setattr(err, "timeout", True)
    setattr(err, "timeout_seconds", timeout_seconds)
    if source == "phase":
        setattr(err, "phase_deadline_exhausted", True)
    return err


def _deadline_cleanup_error(
    deadline: AgentDecisionError,
    failures: list[dict[str, Any]],
) -> AgentDecisionError:
    pending_count = sum(int(item.get("pending_task_count") or 0) for item in failures)
    error_type = (
        "DecisionTaskCleanupTimeout" if pending_count else "DecisionCleanupFailed"
    )
    request_id = str(getattr(deadline, "request_id", "<unknown>"))
    err = AgentDecisionError(
        f"{deadline}; bounded decision cleanup failed for request {request_id}"
    )
    setattr(err, "error_type", error_type)
    setattr(err, "request_id", request_id)
    setattr(err, "timeout", True)
    setattr(err, "timeout_seconds", float(getattr(deadline, "timeout_seconds", 0.0)))
    setattr(err, "deadline_error_type", getattr(deadline, "error_type", "DecisionTimeout"))
    setattr(err, "fatal_cleanup_failure", True)
    setattr(err, "cleanup_failures", [dict(item) for item in failures])
    if bool(getattr(deadline, "phase_deadline_exhausted", False)):
        setattr(err, "phase_deadline_exhausted", True)
    return err


def _decision_error(
    message: str,
    *,
    error_type: str,
    request: Any,
) -> AgentDecisionError:
    err = AgentDecisionError(message)
    setattr(err, "error_type", error_type)
    setattr(err, "request_id", request.request_id)
    return err


def _attach_request_context(err: BaseException, request: Any) -> None:
    """Make every propagated boundary failure linkable to its ActionRequest."""
    if getattr(err, "request_id", None):
        return
    try:
        setattr(err, "request_id", request.request_id)
    except Exception:  # pragma: no cover - unusual immutable third-party errors
        return


def _deep_copy_protocol_value(value: Any) -> Any:
    """Detach mutable protocol payloads at every trust-boundary handoff."""
    model_copy = getattr(value, "model_copy", None)
    if callable(model_copy):
        return model_copy(deep=True)
    return deepcopy(value)


def _mark_deadline_owner(request: Any) -> Any:
    """Tell deadline-aware adapters that this runtime owns the wall clock."""
    if request.deadline_monotonic is None:
        return request
    metadata = dict(request.metadata)
    if metadata.get("deadline_owner") == "decision_runtime":
        return request
    metadata["deadline_owner"] = "decision_runtime"
    return request.model_copy(update={"metadata": metadata})


def _set_task_name(task: asyncio.Future[Any], name: str) -> None:
    setter = getattr(task, "set_name", None)
    if callable(setter):
        setter(name)


def _task_name(task: asyncio.Future[Any]) -> str:
    getter = getattr(task, "get_name", None)
    return str(getter()) if callable(getter) else "unnamed"


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    if not task.done():
        return
    try:
        task.result()
    except BaseException:  # retrieval prevents unhandled task diagnostics
        return


async def _wait_for_task_until(
    task: asyncio.Future[Any],
    deadline: float,
) -> bool:
    """Wait without letting a repeated outer cancel skip task bookkeeping."""
    caller_cancelled = False
    while not task.done():
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        if remaining <= 0:
            break
        try:
            await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError:
            caller_cancelled = True
    return caller_cancelled


def _optional_positive_duration(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{name} must be a positive finite number or None")
    return duration


def _request_telemetry(started_monotonic: float, *, outcome: str) -> dict[str, Any]:
    """Return bounded wall-clock telemetry common to every actor protocol."""
    return {
        "outcome": str(outcome),
        "elapsed_seconds": round(
            max(0.0, time.monotonic() - float(started_monotonic)),
            6,
        ),
    }


def _bounded_nonnegative_duration(
    value: float,
    *,
    name: str,
    maximum: float,
) -> float:
    duration = float(value)
    if not math.isfinite(duration) or duration < 0 or duration > maximum:
        raise ValueError(
            f"{name} must be a finite number between 0 and {maximum:g} seconds"
        )
    return duration


def _request_labels(request: Any) -> dict[str, Any]:
    labels = getattr(request, "labels", None)
    return dict(labels) if isinstance(labels, dict) else {}


def _request_stage(request: Any) -> str:
    labels = _request_labels(request)
    return str(labels.get("stage") or labels.get("phase") or getattr(request, "phase", "decision"))


def _request_action(request: Any) -> str:
    labels = _request_labels(request)
    return str(labels.get("action") or getattr(request, "action_kind", "decide"))


def _request_actor_label(request: Any) -> str:
    actor_id = getattr(request, "actor_id", None)
    if actor_id is not None:
        return str(actor_id)
    seat = getattr(request, "seat", None)
    return f"seat {seat}" if seat is not None else "<unknown>"


def _assert_actor_matches_request(actor: Any, request: Any) -> None:
    """Fail closed when the invoked object is not the addressed actor.

    Envelope validation proves what the model *returned*.  This independent
    check proves which long-lived object (and therefore which private memory)
    the environment actually invoked.
    """
    requested_actor_id = getattr(request, "actor_id", None)
    if requested_actor_id is not None:
        actual_actor_id = getattr(actor, "actor_id", None)
        if (
            actual_actor_id is None
            or str(actual_actor_id).strip() != str(requested_actor_id).strip()
        ):
            raise _decision_error(
                "agent object identity does not match ActionRequest actor_id",
                error_type="AgentBindingMismatch",
                request=request,
            )
        return

    requested_seat = getattr(request, "seat", None)
    if requested_seat is None:
        return
    actual_seat = getattr(actor, "seat", None)
    if (
        isinstance(actual_seat, bool)
        or not isinstance(actual_seat, int)
        or actual_seat != requested_seat
    ):
        raise _decision_error(
            "agent object seat does not match ActionRequest seat",
            error_type="AgentBindingMismatch",
            request=request,
        )


def _actor_identity(actor: Any, request: Any) -> str:
    actor_id = getattr(actor, "actor_id", None)
    if actor_id is not None:
        return str(actor_id)
    seat = getattr(actor, "seat", getattr(request, "seat", None))
    return f"seat:{seat}" if seat is not None else _request_actor_label(request)


def _attach_actor_trace_identity(trace: dict[str, Any], actor: Any, request: Any) -> None:
    seat = getattr(actor, "seat", getattr(request, "seat", None))
    if seat is not None:
        trace["seat"] = int(seat)
    actor_id = getattr(actor, "actor_id", None)
    if actor_id is not None:
        trace["actor_id"] = str(actor_id)
    elif seat is None:
        trace["actor_id"] = _actor_identity(actor, request)
