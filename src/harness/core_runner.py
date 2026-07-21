"""Generic lifecycle runner shared by registered adversarial environments."""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .core_spec import CoreRunSpec
from .decision_runtime import DecisionRuntime
from .environment import (
    AgentRegistry,
    AgentResolver,
    EnvironmentDescriptor,
    EnvironmentRunEvidence,
    EnvironmentOutcome,
    EnvironmentRunContext,
    EnvironmentSession,
)
from .registry import EnvironmentRegistry
from .transcript import Transcript, redact_sensitive


CORE_RUN_RESULT_VERSION = "agent-harness.run-result.v1"
logger = logging.getLogger(__name__)

# Python cannot force-kill a coroutine that repeatedly swallows CancelledError.
# Keep such tasks strongly referenced and consume their eventual result while
# the returned result/transcript reports the fatal cleanup failure explicitly.
_QUARANTINED_TASKS: dict[asyncio.Future[Any], str] = {}
TaskQuarantineSink = Callable[[asyncio.Future[Any], str], None]


def environment_cancellation_budget_seconds(spec: CoreRunSpec) -> float:
    """Upper bound for cancellation plus both sequential cleanup stages."""
    grace = float(spec.execution.cancellation_grace_seconds)
    cleanup = float(spec.execution.cleanup_timeout_seconds)
    scheduling_margin = max(0.1, grace)
    return grace + 2.0 * (cleanup + grace) + scheduling_margin


class EnvironmentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = CORE_RUN_RESULT_VERSION
    run_id: str
    status: str
    termination_reason: str | None = None
    environment_id: str
    environment_version: str
    run_spec_hash: str
    elapsed_seconds: float
    outcome: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    harness_metrics: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    error: str | None = None
    transcript_digest: str
    transcript: dict[str, Any] = Field(default_factory=dict)


@dataclass
class PreparedEnvironmentRun:
    """Resources prepared by an interactive owner for one Core lifecycle.

    ``session`` may wrap an already-dealt, room-owned environment state.  The
    caller constructs it with ``evidence.emit_event`` / ``evidence.emit_trace``
    and the same ``decision_runtime`` carried here.  Once passed to
    :func:`run_prepared_environment_run`, Core takes exclusive lifecycle
    ownership: the object is single-use, and Core closes both the session and
    runtime after success, failure, timeout, or cancellation.
    """

    descriptor: EnvironmentDescriptor
    session: EnvironmentSession
    decision_runtime: DecisionRuntime
    evidence: EnvironmentRunEvidence
    agent_registry: AgentRegistry
    task_quarantine_sink: TaskQuarantineSink | None = None
    _claimed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.task_quarantine_sink is not None and not callable(
            self.task_quarantine_sink
        ):
            raise TypeError("task_quarantine_sink must be callable")

    def claim(self) -> None:
        if self._claimed:
            raise RuntimeError("prepared environment run has already been consumed")
        self._claimed = True


async def run_environment_run(
    spec: CoreRunSpec,
    *,
    registry: EnvironmentRegistry,
    resolve_agent: AgentResolver,
) -> EnvironmentRunResult:
    """Resolve, create, and run one exact-version environment plugin.

    This compatibility entry point still owns session creation.  Its execution
    and cleanup path is shared with :func:`run_prepared_environment_run`.
    """
    plugin = registry.get(spec.environment.id, spec.environment.version)
    descriptor = EnvironmentDescriptor.model_validate(plugin.descriptor)
    missing_seeds = sorted(set(descriptor.required_seeds) - set(spec.seeds))
    if missing_seeds:
        raise ValueError("missing required environment seeds: " + ",".join(missing_seeds))
    config = plugin.resolve_config(spec.environment_config, spec.seeds)

    transcript = Transcript(
        run_id=spec.run_id,
        metadata={
            "core_run_spec_version": spec.schema_version,
            "run_spec_hash": spec.spec_hash,
            "environment_id": descriptor.id,
            "environment_version": descriptor.version,
            "seeds": dict(spec.seeds),
            "caller_metadata": redact_sensitive(spec.metadata),
        },
    )
    evidence = EnvironmentRunEvidence(transcript=transcript)

    contract = plugin.decision_contract
    runtime = DecisionRuntime(
        on_trace=evidence.emit_trace,
        envelope_type=contract.envelope_type,
        validate_envelope=contract.validate_envelope,
        default_timeout_seconds=spec.execution.decision_timeout_seconds,
        cancellation_grace_seconds=spec.execution.cancellation_grace_seconds,
        expected_run_id=spec.run_id,
    )
    agent_registry = AgentRegistry(resolve_agent)
    context = EnvironmentRunContext(
        run_id=spec.run_id,
        config=config,
        seeds=dict(spec.seeds),
        actor_spec=spec.actors,
        decision_runtime=runtime,
        emit_event=evidence.emit_event,
        emit_trace=evidence.emit_trace,
        resolve_agent=agent_registry.resolve,
        metadata=dict(spec.metadata),
    )

    return await _run_environment_lifecycle(
        spec,
        descriptor=descriptor,
        decision_runtime=runtime,
        evidence=evidence,
        agent_registry=agent_registry,
        create_session=lambda: plugin.create_session(context),
    )


async def run_prepared_environment_run(
    spec: CoreRunSpec,
    *,
    prepared: PreparedEnvironmentRun,
) -> EnvironmentRunResult:
    """Run a caller-prepared session through the canonical Core lifecycle.

    Preparation is intentionally outside this function so an interactive host
    can bind its existing state, human queues, actors, and source-indexed full
    evidence sinks.  Execution is not: after validation and the single-use
    claim, Core exclusively owns run timeout, cancellation, session close,
    DecisionRuntime close, cleanup reporting, and result construction.
    """
    if isinstance(prepared, PreparedEnvironmentRun) and prepared._claimed:
        raise RuntimeError("prepared environment run has already been consumed")
    run_started_recorded, recorded_binding_actor_ids = _validate_prepared_environment_run(
        spec,
        prepared,
    )
    prepared.claim()
    return await _run_environment_lifecycle(
        spec,
        descriptor=EnvironmentDescriptor.model_validate(prepared.descriptor),
        decision_runtime=prepared.decision_runtime,
        evidence=prepared.evidence,
        agent_registry=prepared.agent_registry,
        session=prepared.session,
        run_started_recorded=run_started_recorded,
        recorded_binding_actor_ids=recorded_binding_actor_ids,
        task_quarantine_sink=prepared.task_quarantine_sink,
    )


async def _run_environment_lifecycle(
    spec: CoreRunSpec,
    *,
    descriptor: EnvironmentDescriptor,
    decision_runtime: DecisionRuntime,
    evidence: EnvironmentRunEvidence,
    agent_registry: AgentRegistry,
    session: EnvironmentSession | None = None,
    create_session: Callable[[], Awaitable[EnvironmentSession]] | None = None,
    run_started_recorded: bool = False,
    recorded_binding_actor_ids: tuple[str, ...] | None = None,
    task_quarantine_sink: TaskQuarantineSink | None = None,
) -> EnvironmentRunResult:
    if (session is None) == (create_session is None):
        raise ValueError("provide exactly one prepared session or session factory")

    started = time.monotonic()
    outcome: EnvironmentOutcome | None = None
    error_type: str | None = None
    error: str | None = None
    cleanup_failures: list[dict[str, Any]] = []
    cleanup_cancellation: asyncio.CancelledError | None = None
    owned_tasks: dict[asyncio.Future[Any], str] = {}
    final_pending_task_count = 0
    cancelled = False
    run_deadline = (
        started + spec.execution.run_timeout_seconds
        if spec.execution.run_timeout_seconds is not None
        else None
    )
    try:
        if not run_started_recorded:
            await evidence.emit_harness({
                "type": "run_started",
                "environment_id": descriptor.id,
                "environment_version": descriptor.version,
            })
        if create_session is not None:
            session = await _run_stage(
                create_session(),
                timeout_seconds=_seconds_until(run_deadline),
                cancellation_grace_seconds=spec.execution.cancellation_grace_seconds,
                stage="session_create",
                owned_tasks=owned_tasks,
            )
        if session is None:  # Defensive: a hostile factory may return None.
            raise TypeError("environment session factory returned no session")
        outcome = await _run_stage(
            session.run(),
            timeout_seconds=_seconds_until(run_deadline),
            cancellation_grace_seconds=spec.execution.cancellation_grace_seconds,
            stage="session_run",
            owned_tasks=owned_tasks,
        )
        if not isinstance(outcome, EnvironmentOutcome):
            raise TypeError("EnvironmentSession.run must return EnvironmentOutcome")
        if not outcome.terminal:
            raise RuntimeError("environment returned a non-terminal outcome")
    except asyncio.CancelledError as cancel_err:
        # A create_session coroutine may cooperatively finish after receiving
        # cancellation (for example while unwinding an SDK/client setup).  If
        # it produced a session inside the grace window, retain it so the
        # finally block can still close the resource before propagating the
        # caller's cancellation.
        late_session = getattr(cancel_err, "environment_late_result", None)
        if (
            getattr(cancel_err, "environment_stage", None) == "session_create"
            and late_session is not None
            and callable(getattr(late_session, "aclose", None))
        ):
            session = late_session
        cancelled = True
        await _emit_harness_best_effort(evidence, {"type": "run_cancelled"})
        raise
    except _EnvironmentTaskCleanupTimeout as timeout_err:
        if timeout_err.stage == "session_create" and timeout_err.late_result is not None:
            session = timeout_err.late_result
        cleanup_failures.append(timeout_err.failure)
        error_type = "RunTaskCleanupTimeout"
        error = (
            "environment run deadline expired and its task ignored bounded cancellation"
        )
        await _emit_harness_best_effort(evidence, {
            "type": "run_failed",
            "error_type": error_type,
            "error": error,
            "during": timeout_err.stage,
            "fatal_cleanup_failure": True,
            "pending_task_count": timeout_err.failure["pending_task_count"],
        })
    except _EnvironmentRunTimeout as timeout_err:
        if timeout_err.stage == "session_create" and timeout_err.late_result is not None:
            session = timeout_err.late_result
        error_type = "RunTimeout"
        error = f"environment run exceeded {spec.execution.run_timeout_seconds:.3f}s"
        await _emit_harness_best_effort(evidence, {
            "type": "run_failed",
            "error_type": error_type,
            "error": error,
            "during": timeout_err.stage,
        })
    except Exception as err:  # noqa: BLE001 - normalize the plugin boundary
        error_type = type(err).__name__
        error = str(redact_sensitive(str(err) or error_type))
        await _emit_harness_best_effort(evidence, {
            "type": "run_failed",
            "error_type": error_type,
            "error": error,
        })
    finally:
        if session is not None:
            try:
                close_failure = await _run_cleanup(
                    session.aclose,
                    timeout_seconds=spec.execution.cleanup_timeout_seconds,
                    cancellation_grace_seconds=spec.execution.cancellation_grace_seconds,
                    stage="session_close",
                    owned_tasks=owned_tasks,
                )
            except asyncio.CancelledError as cancel_err:
                cleanup_cancellation = cancel_err
                cleanup_failures.append(
                    _cleanup_cancellation_failure("session_close", cancel_err)
                )
            else:
                if close_failure is not None:
                    cleanup_failures.append(close_failure)
        try:
            runtime_failure = await _run_cleanup(
                decision_runtime.aclose,
                timeout_seconds=spec.execution.cleanup_timeout_seconds,
                cancellation_grace_seconds=spec.execution.cancellation_grace_seconds,
                stage="decision_runtime_close",
                owned_tasks=owned_tasks,
            )
        except asyncio.CancelledError as cancel_err:
            if cleanup_cancellation is None:
                cleanup_cancellation = cancel_err
            cleanup_failures.append(
                _cleanup_cancellation_failure("decision_runtime_close", cancel_err)
            )
        else:
            if runtime_failure is not None:
                cleanup_failures.append(runtime_failure)

        # Closing the session can release a task that initially resisted the
        # deadline cancellation. Do not grant a hidden second grace here: tasks
        # still alive are quarantined and reported as an in-process limitation.
        transferred_tasks: set[asyncio.Future[Any]] = set()
        for task, stage in list(owned_tasks.items()):
            if task.done():
                owned_tasks.pop(task, None)
                _consume_task_result(task)
                continue
            _quarantine_task(task, stage, sink=task_quarantine_sink)
            transferred_tasks.add(task)

        runtime_pending = decision_runtime.unresolved_task_details
        for task, owner in decision_runtime.unresolved_task_items:
            if task in transferred_tasks:
                continue
            _quarantine_task(task, owner, sink=task_quarantine_sink)
            transferred_tasks.add(task)
        if runtime_pending and not any(
            failure.get("stage") == "decision_runtime_tasks"
            for failure in cleanup_failures
        ):
            cleanup_failures.append({
                "stage": "decision_runtime_tasks",
                "error_type": "TaskIgnoredCancellation",
                "timeout": True,
                "pending_task_count": len(runtime_pending),
            })

        final_pending_task_count = sum(
            not task.done() for task in owned_tasks
        ) + len(runtime_pending)

        if cleanup_failures:
            await _emit_harness_best_effort(evidence, {
                "type": "run_cleanup_failed",
                "fatal": True,
                "failure_count": len(cleanup_failures),
                "pending_task_count": final_pending_task_count,
                "failures": cleanup_failures,
            })
            if cancelled or cleanup_cancellation is not None:
                logger.critical(
                    "cancelled environment run had cleanup failures "
                    "(run_id=%s failure_count=%d pending_task_count=%d)",
                    spec.run_id,
                    len(cleanup_failures),
                    final_pending_task_count,
                )
            else:
                error_type = "EnvironmentCleanupError"
                error = "environment cleanup failed after bounded cancellation"

        # If cancellation first arrived during cleanup, repay it only after all
        # owned cleanup stages have been attempted.  When the run was already
        # cancelled, leaving the finally block preserves that original error.
        if cleanup_cancellation is not None and not cancelled:
            raise cleanup_cancellation

    resolved_actor_ids = sorted(agent_registry.snapshot())
    if (
        recorded_binding_actor_ids is not None
        and resolved_actor_ids != list(recorded_binding_actor_ids)
    ):
        if error_type is None:
            error_type = "AgentBindingDriftError"
            error = "resolved actor set changed after startup attestation"
        await _emit_harness_best_effort(evidence, {
            "type": "run_failed",
            "error_type": "AgentBindingDriftError",
            "error": "resolved actor set changed after startup attestation",
        })
    final_sink_failure = None
    if recorded_binding_actor_ids is None:
        final_sink_failure = await _emit_harness_best_effort(evidence, {
            "type": "agent_bindings_finalized",
            "actor_count": len(resolved_actor_ids),
            "actor_ids": resolved_actor_ids,
        })
    if final_sink_failure is not None and error_type is None:
        error_type = "EvidenceSinkError"
        error = "harness evidence sink failed"
    if error_type is None and outcome is not None:
        final_sink_failure = await _emit_harness_best_effort(evidence, {
            "type": "run_completed" if outcome.status == "completed" else "run_incomplete",
            "status": outcome.status,
            "termination_reason": outcome.termination_reason,
            "outcome": outcome.outcome,
            "metrics": outcome.metrics,
        })
        if final_sink_failure is not None:
            error_type = "EvidenceSinkError"
            error = "harness evidence sink failed"
    elapsed = round(time.monotonic() - started, 6)
    exported = evidence.transcript.export()
    status = outcome.status if error_type is None and outcome is not None else (
        "timed_out" if error_type == "RunTimeout" else "failed"
    )
    return EnvironmentRunResult(
        run_id=spec.run_id,
        status=status,
        termination_reason=(
            outcome.termination_reason
            if error_type is None and outcome is not None
            else None
        ),
        environment_id=descriptor.id,
        environment_version=descriptor.version,
        run_spec_hash=spec.spec_hash,
        elapsed_seconds=elapsed,
        outcome=redact_sensitive(outcome.outcome) if outcome is not None else {},
        metrics=redact_sensitive(outcome.metrics) if outcome is not None else {},
        harness_metrics={
            "cancellation_grace_seconds": spec.execution.cancellation_grace_seconds,
            "cleanup_timeout_seconds": spec.execution.cleanup_timeout_seconds,
            "cleanup_failure_count": len(cleanup_failures),
            "pending_task_count": final_pending_task_count,
            "resolved_actor_count": len(resolved_actor_ids),
            "resolved_actor_ids": resolved_actor_ids,
        },
        error_type=error_type,
        error=error,
        transcript_digest=exported["stable_digest"],
        transcript=exported,
    )


def _validate_prepared_environment_run(
    spec: CoreRunSpec,
    prepared: PreparedEnvironmentRun,
) -> tuple[bool, tuple[str, ...] | None]:
    if not isinstance(prepared, PreparedEnvironmentRun):
        raise TypeError("prepared must be a PreparedEnvironmentRun")
    descriptor = EnvironmentDescriptor.model_validate(prepared.descriptor)
    if (
        descriptor.id != spec.environment.id
        or descriptor.version != spec.environment.version
    ):
        raise ValueError("prepared environment descriptor does not match CoreRunSpec")
    missing_seeds = sorted(set(descriptor.required_seeds) - set(spec.seeds))
    if missing_seeds:
        raise ValueError(
            "missing required environment seeds: " + ",".join(missing_seeds)
        )
    if not isinstance(prepared.evidence, EnvironmentRunEvidence):
        raise TypeError("prepared evidence must be EnvironmentRunEvidence")
    transcript = prepared.evidence.transcript
    if transcript.run_id != spec.run_id:
        raise ValueError("prepared transcript run_id does not match CoreRunSpec")
    transcript_spec_hash = transcript.metadata.get("run_spec_hash")
    if transcript_spec_hash is not None and transcript_spec_hash != spec.spec_hash:
        raise ValueError("prepared transcript run_spec_hash does not match CoreRunSpec")
    if not isinstance(prepared.decision_runtime, DecisionRuntime):
        raise TypeError("prepared decision_runtime must be a DecisionRuntime")
    expected_run_id = getattr(prepared.decision_runtime, "_expected_run_id", None)
    if expected_run_id != spec.run_id:
        raise ValueError(
            "prepared DecisionRuntime must set expected_run_id to CoreRunSpec.run_id"
        )
    runtime_trace_callbacks = [
        getattr(prepared.decision_runtime, "_on_trace", None),
        *list(getattr(prepared.decision_runtime, "_trace_listeners", ())),
    ]
    if not any(
        callback == prepared.evidence.emit_trace
        for callback in runtime_trace_callbacks
    ):
        raise ValueError(
            "prepared DecisionRuntime must emit through EnvironmentRunEvidence"
        )
    if not isinstance(prepared.agent_registry, AgentRegistry):
        raise TypeError("prepared agent_registry must be an AgentRegistry")
    if not callable(getattr(prepared.session, "run", None)):
        raise TypeError("prepared session has no run method")
    if not callable(getattr(prepared.session, "aclose", None)):
        raise TypeError("prepared session has no aclose method")
    resolved_actor_ids = sorted(prepared.agent_registry.snapshot())
    run_started_rows = _harness_rows(transcript, "run_started")
    if len(run_started_rows) > 1:
        raise ValueError("prepared transcript has duplicate run_started evidence")
    if run_started_rows:
        row = run_started_rows[0]
        if (
            row.get("environment_id") != descriptor.id
            or row.get("environment_version") != descriptor.version
        ):
            raise ValueError("prepared run_started evidence does not match environment")
    binding_rows = _harness_rows(transcript, "agent_bindings_finalized")
    if len(binding_rows) > 1:
        raise ValueError(
            "prepared transcript has duplicate agent_bindings_finalized evidence"
        )
    if binding_rows:
        row = binding_rows[0]
        if (
            row.get("actor_count") != len(resolved_actor_ids)
            or row.get("actor_ids") != resolved_actor_ids
        ):
            raise ValueError(
                "prepared agent_bindings_finalized evidence does not match registry"
            )
    terminal_rows = [
        entry
        for entry in transcript.entries
        if entry.kind == "harness"
        and entry.payload.get("type")
        in {"run_completed", "run_incomplete", "run_cancelled", "run_failed"}
    ]
    if terminal_rows:
        raise ValueError("prepared transcript already contains terminal run evidence")
    recorded_actor_ids = tuple(resolved_actor_ids) if binding_rows else None
    return bool(run_started_rows), recorded_actor_ids


def _harness_rows(transcript: Transcript, event_type: str) -> list[dict[str, Any]]:
    return [
        entry.payload
        for entry in transcript.entries
        if entry.kind == "harness" and entry.payload.get("type") == event_type
    ]


async def _emit_harness_best_effort(
    evidence: EnvironmentRunEvidence,
    payload: dict[str, Any],
) -> str | None:
    """Keep lifecycle cleanup deterministic when an external full sink fails."""
    try:
        await evidence.emit_harness(payload)
    except Exception as err:  # noqa: BLE001 - external evidence boundary
        logger.error(
            "harness evidence sink failed (run_id=%s event_type=%s error_type=%s)",
            evidence.transcript.run_id,
            payload.get("type"),
            type(err).__name__,
        )
        return type(err).__name__
    return None


class _EnvironmentRunTimeout(TimeoutError):
    def __init__(self, stage: str, *, late_result: Any | None = None) -> None:
        super().__init__(f"environment run timed out during {stage}")
        self.stage = stage
        self.late_result = late_result


class _EnvironmentTaskCleanupTimeout(_EnvironmentRunTimeout):
    def __init__(self, stage: str, *, late_result: Any | None = None) -> None:
        super().__init__(stage, late_result=late_result)
        self.failure = {
            "stage": stage,
            "error_type": "TaskIgnoredCancellation",
            "timeout": True,
            "pending_task_count": 1,
        }


async def _run_stage(
    awaitable: Any,
    *,
    timeout_seconds: float | None,
    cancellation_grace_seconds: float,
    stage: str,
    owned_tasks: dict[asyncio.Future[Any], str],
) -> Any:
    if not inspect.isawaitable(awaitable):
        raise TypeError(f"{stage} must return an awaitable")
    task = asyncio.ensure_future(awaitable)
    _set_task_name(task, f"environment:{stage}")
    try:
        if timeout_seconds is None:
            return await asyncio.shield(task)
        done, _pending = await asyncio.wait(
            {task},
            timeout=max(0.0, timeout_seconds),
        )
    except asyncio.CancelledError as err:
        terminated, _interrupted = await _cancel_task(
            task,
            cancellation_grace_seconds,
        )
        setattr(err, "environment_stage", stage)
        if terminated:
            late_result = _late_task_result(task)
            if late_result is not None:
                setattr(err, "environment_late_result", late_result)
        if not terminated:
            _track_owned_task(task, stage, owned_tasks)
            setattr(err, "cleanup_pending_task_count", 1)
        raise
    if task in done:
        return task.result()

    terminated, caller_cancelled = await _cancel_task(
        task,
        cancellation_grace_seconds,
    )
    late_result = _late_task_result(task) if terminated else None
    if not terminated:
        _track_owned_task(task, stage, owned_tasks)
    if caller_cancelled:
        cancelled = asyncio.CancelledError()
        if not terminated:
            setattr(cancelled, "cleanup_pending_task_count", 1)
        raise cancelled
    if not terminated:
        raise _EnvironmentTaskCleanupTimeout(stage)
    raise _EnvironmentRunTimeout(stage, late_result=late_result)


async def _run_cleanup(
    callback: Any,
    *,
    timeout_seconds: float,
    cancellation_grace_seconds: float,
    stage: str,
    owned_tasks: dict[asyncio.Future[Any], str],
) -> dict[str, Any] | None:
    try:
        awaitable = callback()
    except Exception as err:  # noqa: BLE001 - sanitize plugin cleanup failure
        return {
            "stage": stage,
            "error_type": type(err).__name__,
            "timeout": False,
            "pending_task_count": 0,
        }
    if not inspect.isawaitable(awaitable):
        return {
            "stage": stage,
            "error_type": "CleanupProtocolError",
            "timeout": False,
            "pending_task_count": 0,
        }
    task = asyncio.ensure_future(awaitable)
    _set_task_name(task, f"environment:{stage}")
    try:
        done, _pending = await asyncio.wait({task}, timeout=timeout_seconds)
    except asyncio.CancelledError as err:
        terminated, _interrupted = await _cancel_task(
            task,
            cancellation_grace_seconds,
        )
        if not terminated:
            _track_owned_task(task, stage, owned_tasks)
        setattr(err, "cleanup_stage", stage)
        setattr(err, "cleanup_pending_task_count", 0 if terminated else 1)
        raise
    if task in done:
        try:
            task.result()
        except asyncio.CancelledError:
            return {
                "stage": stage,
                "error_type": "CleanupCancelled",
                "timeout": False,
                "pending_task_count": 0,
            }
        except Exception as err:  # noqa: BLE001 - sanitize plugin cleanup failure
            return {
                "stage": stage,
                "error_type": type(err).__name__,
                "timeout": False,
                "pending_task_count": 0,
            }
        return None

    terminated, caller_cancelled = await _cancel_task(
        task,
        cancellation_grace_seconds,
    )
    if not terminated:
        _track_owned_task(task, stage, owned_tasks)
    if caller_cancelled:
        cancelled = asyncio.CancelledError()
        setattr(cancelled, "cleanup_stage", stage)
        setattr(cancelled, "cleanup_pending_task_count", 0 if terminated else 1)
        raise cancelled
    return {
        "stage": stage,
        "error_type": "CleanupTimeout" if terminated else "TaskIgnoredCancellation",
        "timeout": True,
        "pending_task_count": 0 if terminated else 1,
    }


def _cleanup_cancellation_failure(
    stage: str,
    error: asyncio.CancelledError,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "error_type": "CleanupCancelled",
        "timeout": False,
        "pending_task_count": int(
            getattr(error, "cleanup_pending_task_count", 0)
        ),
    }


async def _cancel_task(
    task: asyncio.Future[Any],
    cancellation_grace_seconds: float,
) -> tuple[bool, bool]:
    """Cancel twice in one grace and defer repeated caller cancellation."""
    if task.done():
        _consume_task_result(task)
        return True, False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + cancellation_grace_seconds
    task.cancel()
    first_budget = cancellation_grace_seconds / 2
    caller_cancelled = await _wait_task_until(
        task,
        loop.time() + first_budget,
    )
    if not task.done():
        task.cancel()
        caller_cancelled = (
            await _wait_task_until(task, deadline)
            or caller_cancelled
        )
    if not task.done():
        return False, caller_cancelled
    _consume_task_result(task)
    return True, caller_cancelled


def _track_owned_task(
    task: asyncio.Future[Any],
    stage: str,
    owned_tasks: dict[asyncio.Future[Any], str],
) -> None:
    owned_tasks[task] = stage

    def forget(done: asyncio.Future[Any]) -> None:
        owned_tasks.pop(done, None)
        _QUARANTINED_TASKS.pop(done, None)
        _consume_task_result(done)

    task.add_done_callback(forget)


def _quarantine_task(
    task: asyncio.Future[Any],
    stage: str,
    *,
    sink: TaskQuarantineSink | None = None,
) -> None:
    if task.done():
        _consume_task_result(task)
        return
    if sink is not None:
        try:
            sink(task, stage)
            return
        except Exception as err:  # noqa: BLE001 - retain task in fallback registry
            logger.error(
                "task quarantine sink failed; using Core fallback registry "
                "(stage=%s error_type=%s)",
                stage,
                type(err).__name__,
            )
    _QUARANTINED_TASKS[task] = stage
    logger.critical(
        "environment task ignored bounded cancellation and remains in-process "
        "(stage=%s task_name=%s)",
        stage,
        _task_name(task),
    )


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    if not task.done():
        return
    try:
        task.result()
    except BaseException:
        return


async def _wait_task_until(
    task: asyncio.Future[Any],
    deadline: float,
) -> bool:
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


def _late_task_result(task: asyncio.Future[Any]) -> Any | None:
    if not task.done() or task.cancelled():
        return None
    try:
        return task.result()
    except BaseException:
        return None


def _set_task_name(task: asyncio.Future[Any], name: str) -> None:
    setter = getattr(task, "set_name", None)
    if callable(setter):
        setter(name)


def _task_name(task: asyncio.Future[Any]) -> str:
    getter = getattr(task, "get_name", None)
    return str(getter()) if callable(getter) else "unnamed"


def _seconds_until(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())
