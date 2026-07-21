"""DecisionRuntime one-request/one-terminal-row contract tests."""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from src.agent.schemas import AgentAction, Decision
from src.harness.agent_protocol import ActionRequest, DecisionEnvelope, LegalAction
from src.harness.core_protocol import (
    ActionChoice as CoreActionChoice,
    ActionOption as CoreActionOption,
    ActionRequest as CoreActionRequest,
    DecisionEnvelope as CoreDecisionEnvelope,
    validate_decision_envelope as validate_core_decision_envelope,
)
from src.harness.decision_runtime import DecisionRuntime
from src.harness.errors import AgentDecisionError


def _request(
    *,
    request_id: str = "request-1",
    deadline: float | None = None,
    deadline_source: str | None = None,
    timeout_seconds: float | None = None,
) -> ActionRequest:
    metadata: dict[str, Any] = {}
    if deadline_source is not None:
        metadata["deadline_source"] = deadline_source
    if timeout_seconds is not None:
        metadata["effective_timeout_seconds"] = timeout_seconds
    return ActionRequest(
        request_id=request_id,
        run_id="run-1",
        seat=1,
        phase="voting",
        day=1,
        action_kind="vote",
        observation={},
        legal_actions=[LegalAction(action="vote", target_seats=[2])],
        deadline_monotonic=deadline,
        metadata=metadata,
    )


class _ValidAgent:
    seat = 1

    async def decide(self, request: ActionRequest) -> DecisionEnvelope:
        return DecisionEnvelope(
            request_id=request.request_id,
            seat=self.seat,
            decision=Decision(action=AgentAction.VOTE, target_seat=2),
        )


@pytest.mark.asyncio
async def test_runtime_records_one_request_and_one_valid_envelope_terminal():
    trace: list[dict[str, Any]] = []
    envelope = await DecisionRuntime(on_trace=trace.append).execute(_ValidAgent(), _request())

    assert envelope.decision.target_seat == 2
    assert [row["kind"] for row in trace] == ["agent_request", "agent_response"]
    assert trace[0]["request"]["request_id"] == "request-1"
    assert trace[1]["request_id"] == "request-1"
    assert trace[1]["validation"]["valid"] is True
    assert trace[1]["request_telemetry"]["outcome"] == "accepted"
    assert trace[1]["request_telemetry"]["elapsed_seconds"] >= 0


@pytest.mark.asyncio
async def test_runtime_rejects_wrong_agent_object_before_generic_decision():
    class WrongAgent:
        actor_id = "wrong"
        called = False

        async def decide(self, request: CoreActionRequest) -> CoreDecisionEnvelope:
            self.called = True
            # Even a forged envelope matching the request cannot repair the
            # fact that the environment invoked another actor's state object.
            return CoreDecisionEnvelope(
                request_id=request.request_id,
                actor_id=request.actor_id,
                choice=CoreActionChoice(action="commit", arguments={}),
            )

    request = CoreActionRequest(
        request_id="generic-request-1",
        run_id="generic-run",
        actor_id="expected",
        legal_actions=[CoreActionOption(name="commit")],
    )
    agent = WrongAgent()
    trace: list[dict[str, Any]] = []
    runtime = DecisionRuntime(
        on_trace=trace.append,
        envelope_type=CoreDecisionEnvelope,
        validate_envelope=validate_core_decision_envelope,
    )

    with pytest.raises(AgentDecisionError) as raised:
        await runtime.execute(agent, request)

    assert getattr(raised.value, "error_type") == "AgentBindingMismatch"
    assert agent.called is False
    assert [row["kind"] for row in trace] == [
        "agent_request",
        "agent_response_failed",
    ]


@pytest.mark.asyncio
async def test_runtime_actor_exception_records_failed_terminal_and_links_error():
    class FailingAgent:
        seat = 1

        async def decide(self, _request: ActionRequest) -> DecisionEnvelope:
            raise RuntimeError("private provider response must not enter trace")

    trace: list[dict[str, Any]] = []
    with pytest.raises(RuntimeError) as raised:
        await DecisionRuntime(on_trace=trace.append).execute(FailingAgent(), _request())

    assert getattr(raised.value, "request_id") == "request-1"
    assert [row["kind"] for row in trace] == ["agent_request", "agent_response_failed"]
    failure = trace[1]
    assert failure["request_id"] == "request-1"
    assert failure["failure"] == {
        "error_type": "RuntimeError",
        "timeout": False,
        "reason": "RuntimeError during voting/vote",
    }
    assert "private provider response" not in str(failure)


@pytest.mark.asyncio
async def test_runtime_failure_keeps_redacted_llm_attempt_provenance():
    class ProviderFailureAgent:
        seat = 1

        async def decide(self, _request: ActionRequest) -> DecisionEnvelope:
            err = AgentDecisionError("provider failed")
            setattr(err, "llm_call_attempts", [{
                "attempt": 1,
                "status": "provider_failed",
                "llm_call": {
                    "call_id": "call-failed",
                    "request_hash": "a" * 64,
                    "api_key": "must-not-enter-trace",
                    "transport_attempts": [{
                        "attempt": 1,
                        "status": "failed",
                        "error_type": "LLMError",
                        "will_retry": False,
                    }],
                },
            }])
            raise err

    trace: list[dict[str, Any]] = []
    with pytest.raises(AgentDecisionError):
        await DecisionRuntime(on_trace=trace.append).execute(ProviderFailureAgent(), _request())

    attempts = trace[-1]["failure"]["llm_call_attempts"]
    assert attempts[0]["llm_call"]["call_id"] == "call-failed"
    assert attempts[0]["llm_call"]["api_key"] == "[redacted]"
    assert "must-not-enter-trace" not in str(trace)


@pytest.mark.asyncio
async def test_runtime_failure_carries_seat_session_telemetry_without_changing_error_shape():
    class SessionFailureAgent:
        seat = 1

        async def decide(self, _request: ActionRequest) -> DecisionEnvelope:
            error = AgentDecisionError("agent session budget exhausted")
            setattr(error, "agent_session_telemetry", {
                "request_id": "request-1",
                "generation_attempts": 2,
                "tool_calls": 1,
                "budget_exhausted": "max_model_generations",
                "api_key": "must-be-redacted",
            })
            raise error

    trace: list[dict[str, Any]] = []
    with pytest.raises(AgentDecisionError):
        await DecisionRuntime(on_trace=trace.append).execute(
            SessionFailureAgent(),
            _request(),
        )

    terminal = trace[-1]
    assert terminal["request_telemetry"]["outcome"] == "failed"
    assert terminal["agent_session"]["generation_attempts"] == 2
    assert terminal["agent_session"]["api_key"] == "[redacted]"
    assert terminal["failure"]["error_type"] == "AgentDecisionError"


@pytest.mark.asyncio
async def test_runtime_decision_timeout_records_failed_terminal():
    class SlowAgent:
        seat = 1

        async def decide(self, _request: ActionRequest) -> DecisionEnvelope:
            await asyncio.sleep(1)
            raise AssertionError("unreachable")

    timeout = 0.02
    request = _request(
        deadline=time.monotonic() + timeout,
        deadline_source="decision",
        timeout_seconds=timeout,
    )
    trace: list[dict[str, Any]] = []
    with pytest.raises(AgentDecisionError) as raised:
        await DecisionRuntime(on_trace=trace.append).execute(SlowAgent(), request)

    assert getattr(raised.value, "error_type") == "DecisionTimeout"
    assert getattr(raised.value, "request_id") == request.request_id
    assert getattr(raised.value, "timeout") is True
    assert [row["kind"] for row in trace] == ["agent_request", "agent_response_failed"]
    assert trace[1]["failure"]["error_type"] == "DecisionTimeout"
    assert trace[1]["failure"]["timeout"] is True
    assert "decision timeout" in trace[1]["failure"]["reason"]


@pytest.mark.asyncio
async def test_runtime_deadline_waits_only_bounded_grace_for_delayed_cancellation():
    class SlowCleanupAgent:
        seat = 1
        task: asyncio.Task[Any] | None = None

        async def decide(self, _request: ActionRequest) -> DecisionEnvelope:
            self.task = asyncio.current_task()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.02)
                raise

    agent = SlowCleanupAgent()
    timeout = 0.01
    trace: list[dict[str, Any]] = []
    runtime = DecisionRuntime(
        on_trace=trace.append,
        cancellation_grace_seconds=0.1,
    )
    started = time.monotonic()

    with pytest.raises(AgentDecisionError) as raised:
        await runtime.execute(
            agent,
            _request(
                deadline=time.monotonic() + timeout,
                deadline_source="decision",
                timeout_seconds=timeout,
            ),
        )

    assert time.monotonic() - started < 0.2
    assert getattr(raised.value, "error_type") == "DecisionTimeout"
    assert agent.task is not None and agent.task.done()
    assert runtime.unresolved_task_count == 0


@pytest.mark.asyncio
async def test_runtime_reports_fatal_cleanup_when_agent_swallows_cancellation():
    release = asyncio.Event()

    class CancellationIgnoringAgent:
        seat = 1
        task: asyncio.Task[Any] | None = None

        async def decide(self, _request: ActionRequest) -> DecisionEnvelope:
            self.task = asyncio.current_task()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
            raise asyncio.CancelledError

    agent = CancellationIgnoringAgent()
    trace: list[dict[str, Any]] = []
    runtime = DecisionRuntime(
        on_trace=trace.append,
        cancellation_grace_seconds=0.02,
    )
    started = time.monotonic()
    try:
        with pytest.raises(AgentDecisionError) as raised:
            await runtime.execute(
                agent,
                _request(
                    deadline=time.monotonic() + 0.01,
                    deadline_source="decision",
                    timeout_seconds=0.01,
                ),
            )

        assert time.monotonic() - started < 0.15
        assert getattr(raised.value, "error_type") == "DecisionTaskCleanupTimeout"
        assert getattr(raised.value, "fatal_cleanup_failure") is True
        assert runtime.unresolved_task_count == 1
        assert trace[-1]["failure"]["cleanup"] == {
            "fatal": True,
            "failures": [{
                "stage": "agent_decide",
                "error_type": "TaskIgnoredCancellation",
                "pending_task_count": 1,
            }],
        }
    finally:
        release.set()
        await asyncio.sleep(0)
        await runtime.aclose()

    assert agent.task is not None and agent.task.done()
    assert runtime.unresolved_task_count == 0


@pytest.mark.asyncio
async def test_runtime_expired_phase_deadline_does_not_call_agent_but_is_terminal():
    class UncalledAgent:
        seat = 1
        called = False

        async def decide(self, _request: ActionRequest) -> DecisionEnvelope:
            self.called = True
            raise AssertionError("expired request must not invoke the agent")

    agent = UncalledAgent()
    request = _request(
        deadline=time.monotonic() - 1,
        deadline_source="phase",
        timeout_seconds=0,
    )
    trace: list[dict[str, Any]] = []
    with pytest.raises(AgentDecisionError) as raised:
        await DecisionRuntime(on_trace=trace.append).execute(agent, request)

    assert agent.called is False
    assert getattr(raised.value, "error_type") == "PhaseDeadlineExceeded"
    assert getattr(raised.value, "phase_deadline_exhausted") is True
    assert [row["kind"] for row in trace] == ["agent_request", "agent_response_failed"]
    assert trace[1]["request_id"] == request.request_id
    assert "before decision start" in trace[1]["failure"]["reason"]


@pytest.mark.asyncio
async def test_runtime_invalid_envelope_is_the_terminal_row_not_a_second_failure():
    class InvalidTargetAgent:
        seat = 1

        async def decide(self, request: ActionRequest) -> DecisionEnvelope:
            return DecisionEnvelope(
                request_id=request.request_id,
                seat=self.seat,
                decision=Decision(action=AgentAction.VOTE, target_seat=999),
            )

    trace: list[dict[str, Any]] = []
    with pytest.raises(AgentDecisionError) as raised:
        await DecisionRuntime(on_trace=trace.append).execute(InvalidTargetAgent(), _request())

    assert getattr(raised.value, "error_type") == "DecisionEnvelopeRejected"
    assert getattr(raised.value, "request_id") == "request-1"
    assert [row["kind"] for row in trace] == ["agent_request", "agent_response"]
    response = trace[1]
    assert response["envelope"]["decision"]["target_seat"] == 999
    assert response["validation"]["valid"] is False
    assert [issue["code"] for issue in response["validation"]["issues"]] == [
        "target_seat_not_legal"
    ]


@pytest.mark.asyncio
async def test_runtime_agent_cannot_mutate_trusted_request_before_validation():
    class MutatingAgent:
        seat = 1
        received: ActionRequest | None = None

        async def decide(self, request: ActionRequest) -> DecisionEnvelope:
            self.received = request
            request.legal_actions[0].target_seats.append(999)
            request.observation["private_marker"] = {"nested": ["mutated"]}
            request.private_context["killed_seat"] = 999
            request.metadata["agent_injected"] = True
            return DecisionEnvelope(
                request_id=request.request_id,
                seat=self.seat,
                decision=Decision(action=AgentAction.VOTE, target_seat=999),
            )

    agent = MutatingAgent()
    original = _request()
    trace: list[dict[str, Any]] = []

    with pytest.raises(AgentDecisionError) as raised:
        await DecisionRuntime(on_trace=trace.append).execute(agent, original)

    assert getattr(raised.value, "error_type") == "DecisionEnvelopeRejected"
    assert agent.received is not None
    assert agent.received.legal_actions[0].target_seats == [2, 999]
    assert agent.received.observation["private_marker"]["nested"] == ["mutated"]
    assert original.legal_actions[0].target_seats == [2]
    assert original.observation == {}
    assert original.private_context == {}
    assert original.metadata == {}
    assert trace[0]["request"]["legal_actions"][0]["target_seats"] == [2]
    assert trace[0]["request"]["observation"] == {}
    assert trace[0]["request"]["private_context"] == {}
    assert trace[0]["request"]["metadata"] == {}
    assert [row["kind"] for row in trace] == ["agent_request", "agent_response"]
    assert trace[1]["validation"]["valid"] is False
    assert [
        issue["code"] for issue in trace[1]["validation"]["issues"]
    ] == ["target_seat_not_legal"]


@pytest.mark.asyncio
async def test_runtime_validator_exception_preserves_envelope_in_distinct_terminal():
    def broken_validator(_envelope: Any, _request: Any) -> Any:
        raise RuntimeError("private validator implementation detail")

    trace: list[dict[str, Any]] = []
    runtime = DecisionRuntime(
        on_trace=trace.append,
        envelope_type=DecisionEnvelope,
        validate_envelope=broken_validator,
    )

    with pytest.raises(AgentDecisionError) as raised:
        await runtime.execute(_ValidAgent(), _request())

    assert getattr(raised.value, "error_type") == "DecisionValidatorError"
    assert getattr(raised.value, "validator_error_type") == "RuntimeError"
    assert getattr(raised.value, "request_id") == "request-1"
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert [row["kind"] for row in trace] == [
        "agent_request",
        "agent_response_validation_failed",
    ]
    terminal = trace[1]
    assert terminal["request_id"] == "request-1"
    assert terminal["envelope"]["decision"]["target_seat"] == 2
    assert terminal["failure"] == {
        "error_type": "RuntimeError",
        "reason": "DecisionEnvelope validator failed.",
    }
    assert "private validator implementation detail" not in str(terminal)
    assert not any(row["kind"] == "agent_response_failed" for row in trace)


@pytest.mark.asyncio
async def test_runtime_non_envelope_value_records_failed_terminal():
    class WrongTypeAgent:
        seat = 1

        async def decide(self, _request: ActionRequest) -> Any:
            return {"action": "vote", "target_seat": 2}

    trace: list[dict[str, Any]] = []
    with pytest.raises(AgentDecisionError) as raised:
        await DecisionRuntime(on_trace=trace.append).execute(WrongTypeAgent(), _request())

    assert getattr(raised.value, "error_type") == "DecisionEnvelopeTypeError"
    assert [row["kind"] for row in trace] == ["agent_request", "agent_response_failed"]
    assert trace[1]["failure"]["error_type"] == "DecisionEnvelopeTypeError"


@pytest.mark.asyncio
async def test_runtime_rejects_duplicate_request_id_before_invoking_agent_twice():
    class CountingAgent(_ValidAgent):
        calls = 0

        async def decide(self, request: ActionRequest) -> DecisionEnvelope:
            self.calls += 1
            return await super().decide(request)

    trace: list[dict[str, Any]] = []
    actor = CountingAgent()
    runtime = DecisionRuntime(on_trace=trace.append)
    request = _request()

    first = await runtime.execute(actor, request)
    assert first.request_id == request.request_id

    with pytest.raises(AgentDecisionError) as raised:
        await runtime.execute(actor, request)

    assert getattr(raised.value, "error_type", None) == "DuplicateRequestId"
    assert getattr(raised.value, "request_id", None) == request.request_id
    assert actor.calls == 1
    assert [row["kind"] for row in trace] == ["agent_request", "agent_response"]


@pytest.mark.asyncio
async def test_runtime_records_cancellation_as_a_distinct_terminal_and_propagates():
    started = asyncio.Event()

    class BlockingAgent:
        seat = 1

        async def decide(self, _request: ActionRequest) -> DecisionEnvelope:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    trace: list[dict[str, Any]] = []
    task = asyncio.create_task(
        DecisionRuntime(on_trace=trace.append).execute(BlockingAgent(), _request())
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [row["kind"] for row in trace] == [
        "agent_request",
        "agent_response_cancelled",
    ]
    terminal = trace[-1]
    assert terminal["request_id"] == "request-1"
    assert terminal["cancellation"] == {"reason": "run_or_room_cancelled"}
    assert "failure" not in terminal


@pytest.mark.asyncio
async def test_runtime_external_cancellation_reclaims_delayed_agent_task():
    started = asyncio.Event()

    class DelayedCancellationAgent:
        seat = 1
        task: asyncio.Task[Any] | None = None

        async def decide(self, _request: ActionRequest) -> DecisionEnvelope:
            self.task = asyncio.current_task()
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0.01)
                raise

    agent = DelayedCancellationAgent()
    trace: list[dict[str, Any]] = []
    runtime = DecisionRuntime(
        on_trace=trace.append,
        cancellation_grace_seconds=0.1,
    )
    execution = asyncio.create_task(runtime.execute(agent, _request()))
    await started.wait()

    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution

    assert agent.task is not None and agent.task.done()
    assert runtime.unresolved_task_count == 0
    assert trace[-1]["kind"] == "agent_response_cancelled"
