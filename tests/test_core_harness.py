"""Environment-neutral protocol, registry, and runner vertical-slice tests."""
from __future__ import annotations

import asyncio
import ast
from pathlib import Path
import random
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError
import pytest

from src.harness.core_protocol import (
    ActionChoice,
    ActionOption,
    ActionRequest,
    DecisionEnvelope,
    SkipChoice,
    SkipPolicy,
    validate_decision_envelope,
)
from src.harness.core_runner import (
    PreparedEnvironmentRun,
    run_environment_run,
    run_prepared_environment_run,
)
from src.harness.core_spec import CoreRunSpec, EnvironmentRef, ExecutionSpec
from src.harness.decision_runtime import DecisionRuntime
from src.harness.environment import (
    AgentBindingError,
    AgentRegistry,
    DecisionContract,
    EnvironmentDescriptor,
    EnvironmentOutcome,
    EnvironmentRunEvidence,
    EnvironmentRunContext,
)
from src.harness.registry import EnvironmentRegistry, EnvironmentRegistryError
from src.harness.transcript import Transcript


class CounterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_ids: list[str] = Field(min_length=1)
    delay_seconds: float = Field(default=0, ge=0)
    request_run_id: str | None = None


class CounterAgent:
    def __init__(self, actor_id: str, amount: int) -> None:
        self.actor_id = actor_id
        self.amount = amount
        self.calls = 0

    async def decide(self, request: ActionRequest) -> DecisionEnvelope:
        self.calls += 1
        return DecisionEnvelope(
            request_id=request.request_id,
            actor_id=self.actor_id,
            choice=ActionChoice(action="increment", arguments={"amount": self.amount}),
            parse_status="not_applicable",
        )


class CounterSession:
    def __init__(self, context: EnvironmentRunContext, owner: "CounterPlugin") -> None:
        self.context = context
        self.owner = owner

    async def run(self) -> EnvironmentOutcome:
        config = CounterConfig.model_validate(self.context.config)
        if config.delay_seconds:
            await asyncio.sleep(config.delay_seconds)
        actor_ids = list(config.actor_ids)
        self.context.rng("turn_order").shuffle(actor_ids)
        total = 0
        for step, actor_id in enumerate(actor_ids, start=1):
            request = ActionRequest(
                request_id=f"{self.context.run_id}:step:{step}",
                run_id=config.request_run_id or self.context.run_id,
                actor_id=actor_id,
                observation={"current_total": total},
                legal_actions=[ActionOption(
                    name="increment",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "amount": {"type": "integer", "minimum": 1, "maximum": 5},
                        },
                        "required": ["amount"],
                        "additionalProperties": False,
                    },
                )],
                labels={"stage": "counter_round", "step": step, "action": "increment"},
            )
            agent = self.context.resolve_agent(actor_id)
            envelope = await self.context.decision_runtime.execute(agent, request)
            assert isinstance(envelope.choice, ActionChoice)
            amount = int(envelope.choice.arguments["amount"])
            total += amount
            await self.context.emit_event({
                "type": "counter_incremented",
                "actor_id": actor_id,
                "step": step,
                "amount": amount,
                "total": total,
            })
        return EnvironmentOutcome(
            terminal=True,
            outcome={"total": total},
            metrics={"decision_count": len(actor_ids)},
        )

    async def aclose(self) -> None:
        self.owner.closed_sessions += 1


class CounterPlugin:
    descriptor = EnvironmentDescriptor(
        id="test.counter",
        version="1",
        required_seeds=("turn_order",),
        capabilities=("multi_agent", "adversarial_test"),
    )
    decision_contract = DecisionContract(
        envelope_type=DecisionEnvelope,
        validate_envelope=validate_decision_envelope,
    )

    def __init__(self) -> None:
        self.closed_sessions = 0

    def resolve_config(
        self,
        raw_config: Mapping[str, Any],
        _seeds: Mapping[str, int],
    ) -> BaseModel:
        return CounterConfig(**dict(raw_config))

    async def create_session(self, context: EnvironmentRunContext) -> CounterSession:
        return CounterSession(context, self)


class LifecycleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_delay_seconds: float = Field(default=0, ge=0)
    cancellation_delay_seconds: float = Field(default=0, ge=0)
    close_raises: bool = False
    close_ignores_cancellation: bool = False


class LifecycleSession:
    def __init__(self, config: LifecycleConfig, owner: "LifecyclePlugin") -> None:
        self.config = config
        self.owner = owner

    async def run(self) -> EnvironmentOutcome:
        self.owner.run_task = asyncio.current_task()
        self.owner.run_started.set()
        try:
            if self.config.run_delay_seconds:
                await asyncio.sleep(self.config.run_delay_seconds)
        except asyncio.CancelledError:
            if self.config.cancellation_delay_seconds:
                await asyncio.sleep(self.config.cancellation_delay_seconds)
            raise
        return EnvironmentOutcome(terminal=True, outcome={"ok": True})

    async def aclose(self) -> None:
        self.owner.close_calls += 1
        self.owner.close_task = asyncio.current_task()
        if self.config.close_raises:
            raise RuntimeError("private session cleanup detail")
        if self.config.close_ignores_cancellation:
            while not self.owner.release_close.is_set():
                try:
                    await self.owner.release_close.wait()
                except asyncio.CancelledError:
                    continue


class LifecyclePlugin:
    descriptor = EnvironmentDescriptor(id="test.lifecycle", version="1")
    decision_contract = DecisionContract(
        envelope_type=DecisionEnvelope,
        validate_envelope=validate_decision_envelope,
    )

    def __init__(self) -> None:
        self.close_calls = 0
        self.run_started = asyncio.Event()
        self.release_close = asyncio.Event()
        self.run_task: asyncio.Task[Any] | None = None
        self.close_task: asyncio.Task[Any] | None = None

    def resolve_config(
        self,
        raw_config: Mapping[str, Any],
        _seeds: Mapping[str, int],
    ) -> BaseModel:
        return LifecycleConfig.model_validate(raw_config)

    async def create_session(self, context: EnvironmentRunContext) -> LifecycleSession:
        return LifecycleSession(LifecycleConfig.model_validate(context.config), self)


class RecordingDecisionRuntime(DecisionRuntime):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        await super().aclose()


def _spec(*, timeout: float = 2, delay: float = 0) -> CoreRunSpec:
    return CoreRunSpec(
        run_id="counter-run",
        environment=EnvironmentRef(id="test.counter", version="1"),
        environment_config={
            "actor_ids": ["alpha", "opponent:beta"],
            "delay_seconds": delay,
        },
        seeds={"turn_order": 17},
        execution=ExecutionSpec(run_timeout_seconds=timeout),
    )


def _lifecycle_spec(
    config: dict[str, Any],
    *,
    run_timeout: float = 1.0,
    cancellation_grace: float = 0.1,
    cleanup_timeout: float = 0.1,
) -> CoreRunSpec:
    return CoreRunSpec(
        run_id="lifecycle-run",
        environment=EnvironmentRef(id="test.lifecycle", version="1"),
        environment_config=config,
        execution=ExecutionSpec(
            run_timeout_seconds=run_timeout,
            cancellation_grace_seconds=cancellation_grace,
            cleanup_timeout_seconds=cleanup_timeout,
        ),
    )


def test_core_protocol_validates_identity_skip_and_json_schema_arguments() -> None:
    request = ActionRequest(
        request_id="r1",
        run_id="run",
        actor_id="alpha",
        legal_actions=[ActionOption(
            name="bid",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "integer", "minimum": 0}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )],
        skip_policy=SkipPolicy(allowed=True),
    )
    valid = DecisionEnvelope(
        request_id="r1",
        actor_id="alpha",
        choice=ActionChoice(action="bid", arguments={"value": 3}),
    )
    assert validate_decision_envelope(valid, request).valid

    invalid = valid.model_copy(update={
        "actor_id": "beta",
        "choice": ActionChoice(action="bid", arguments={"value": -1, "extra": True}),
    })
    codes = {issue.code for issue in validate_decision_envelope(invalid, request).issues}
    assert {"actor_id_mismatch", "action_arguments_invalid"} <= codes

    empty_skip = valid.model_copy(update={"choice": SkipChoice(reason="")})
    result = validate_decision_envelope(empty_skip, request)
    assert not result.valid
    assert "skip_reason_missing" in {issue.code for issue in result.issues}


@pytest.mark.parametrize("identity", [" ", "\t\n"])
def test_core_protocol_rejects_blank_identity_fields(identity: str) -> None:
    with pytest.raises(ValidationError):
        ActionRequest(
            request_id=identity,
            run_id="run",
            actor_id="actor",
            legal_actions=[ActionOption(name="commit")],
        )
    with pytest.raises(ValidationError):
        DecisionEnvelope(
            request_id=identity,
            actor_id="actor",
            choice=ActionChoice(action="commit"),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_core_protocol_rejects_nonfinite_json_and_latency(value: float) -> None:
    with pytest.raises(ValidationError):
        ActionChoice(action="commit", arguments={"value": value})
    with pytest.raises(ValidationError):
        DecisionEnvelope(
            request_id="request",
            actor_id="actor",
            choice=ActionChoice(action="commit"),
            latency_seconds=value,
        )


def test_core_protocol_rejects_unknown_version_after_unsafe_model_copy() -> None:
    request = ActionRequest(
        request_id="request",
        run_id="run",
        actor_id="actor",
        legal_actions=[ActionOption(name="commit")],
    ).model_copy(update={"protocol_version": "unknown.v99"})
    envelope = DecisionEnvelope(
        request_id="request",
        actor_id="actor",
        choice=ActionChoice(action="commit"),
    ).model_copy(update={"protocol_version": "unknown.v99"})

    result = validate_decision_envelope(envelope, request)

    assert not result.valid
    assert "unsupported_protocol_version" in {issue.code for issue in result.issues}


def test_registry_is_exact_versioned_and_fail_closed() -> None:
    registry = EnvironmentRegistry()
    plugin = CounterPlugin()
    registry.register(plugin)
    assert registry.get("test.counter", "1") is plugin
    assert registry.descriptors() == [plugin.descriptor]
    with pytest.raises(EnvironmentRegistryError, match="already registered"):
        registry.register(CounterPlugin())
    with pytest.raises(EnvironmentRegistryError, match="unknown environment"):
        registry.get("test.counter", "2")

    class MissingContract:
        descriptor = EnvironmentDescriptor(id="broken", version="1")

        def resolve_config(self, raw_config, seeds):
            return CounterConfig(actor_ids=["a"])

        async def create_session(self, context):
            raise AssertionError("must not be registered")

    with pytest.raises(EnvironmentRegistryError, match="decision contract"):
        EnvironmentRegistry().register(MissingContract())  # type: ignore[arg-type]


def test_agent_registry_keeps_one_object_per_actor_and_rejects_aliasing() -> None:
    class BareAgent:
        pass

    shared = BareAgent()
    calls: list[str] = []

    def resolver(actor_id: str) -> CounterAgent:
        calls.append(actor_id)
        return shared

    registry = AgentRegistry(resolver)
    assert registry.resolve("alpha") is shared
    assert registry.resolve("alpha") is shared
    assert calls == ["alpha"]
    with pytest.raises(AgentBindingError, match="multiple actors"):
        registry.resolve("beta")


def test_agent_registry_missing_binding_fails_closed_and_is_stable() -> None:
    calls: list[str] = []

    def resolver(actor_id: str) -> None:
        calls.append(actor_id)
        return None

    registry = AgentRegistry(resolver)
    with pytest.raises(AgentBindingError, match="returned no agent"):
        registry.resolve("missing")
    with pytest.raises(AgentBindingError, match="returned no agent"):
        registry.resolve(" missing ")
    assert calls == ["missing"]


def test_environment_context_rng_keeps_one_advancing_stream_per_namespace() -> None:
    async def emit_event(_payload: dict[str, Any]) -> None:
        return None

    context = EnvironmentRunContext(
        run_id="rng-run",
        config=CounterConfig(actor_ids=["alpha"]),
        seeds={"turn_order": 17, "sampling": 31},
        decision_runtime=object(),  # type: ignore[arg-type]
        emit_event=emit_event,
        emit_trace=lambda _payload: None,
        resolve_agent=lambda _actor_id: None,
        metadata={},
    )
    expected = random.Random(17)
    turn_order = context.rng("turn_order")

    assert turn_order.random() == expected.random()
    assert context.rng("turn_order") is turn_order
    assert context.rng(" turn_order ").random() == expected.random()
    assert context.rng("sampling") is not turn_order

    replay_context = EnvironmentRunContext(
        run_id="rng-run-replay",
        config=CounterConfig(actor_ids=["alpha"]),
        seeds={"turn_order": 17, "sampling": 31},
        decision_runtime=object(),  # type: ignore[arg-type]
        emit_event=emit_event,
        emit_trace=lambda _payload: None,
        resolve_agent=lambda _actor_id: None,
        metadata={},
    )
    replay_stream = replay_context.rng("turn_order")
    expected_replay = random.Random(17)
    assert [replay_stream.random(), replay_context.rng("turn_order").random()] == [
        expected_replay.random(),
        expected_replay.random(),
    ]


@pytest.mark.asyncio
async def test_generic_runner_rejects_cross_run_request_before_agent_call() -> None:
    registry = EnvironmentRegistry()
    plugin = CounterPlugin()
    registry.register(plugin)
    agent = CounterAgent("alpha", amount=1)
    spec = CoreRunSpec(
        run_id="owning-run",
        environment=EnvironmentRef(id="test.counter", version="1"),
        environment_config={
            "actor_ids": ["alpha"],
            "request_run_id": "foreign-run",
        },
        seeds={"turn_order": 17},
    )

    result = await run_environment_run(
        spec,
        registry=registry,
        resolve_agent=lambda _actor_id: agent,
    )

    assert result.status == "failed"
    assert result.error_type == "AgentDecisionError"
    assert agent.calls == 0
    decision_rows = [
        entry["payload"]
        for entry in result.transcript["entries"]
        if entry["kind"] == "decision"
    ]
    assert [row["kind"] for row in decision_rows] == [
        "agent_request",
        "agent_response_failed",
    ]
    assert decision_rows[0]["request"]["run_id"] == "foreign-run"
    assert decision_rows[1]["request_id"] == decision_rows[0]["request"]["request_id"]
    assert decision_rows[1]["failure"]["error_type"] == "RunIdMismatch"


def test_core_spec_rejects_credentials_and_non_integer_seeds() -> None:
    with pytest.raises(ValidationError, match="credentials are forbidden"):
        CoreRunSpec(
            run_id="bad",
            environment=EnvironmentRef(id="test.counter", version="1"),
            environment_config={"api_key": "must-not-enter-spec"},
            seeds={"turn_order": 1},
        )
    with pytest.raises(ValidationError, match="must be an integer"):
        CoreRunSpec(
            run_id="bad-seed",
            environment=EnvironmentRef(id="test.counter", version="1"),
            seeds={"turn_order": "1"},
        )
    with pytest.raises(ValidationError, match="safe path component"):
        CoreRunSpec(
            run_id="../../escape",
            environment=EnvironmentRef(id="test.counter", version="1"),
            seeds={"turn_order": 1},
        )


@pytest.mark.asyncio
async def test_non_werewolf_environment_runs_through_shared_runtime_and_transcript() -> None:
    plugin = CounterPlugin()
    registry = EnvironmentRegistry()
    registry.register(plugin)
    agents = {
        "alpha": CounterAgent("alpha", 2),
        "opponent:beta": CounterAgent("opponent:beta", 3),
    }

    result = await run_environment_run(
        _spec(),
        registry=registry,
        resolve_agent=agents.__getitem__,
    )

    assert result.status == "completed"
    assert result.outcome == {"total": 5}
    assert result.metrics == {"decision_count": 2}
    assert plugin.closed_sessions == 1
    assert all(agent.calls == 1 for agent in agents.values())
    rows = result.transcript["entries"]
    decision_rows = [row["payload"] for row in rows if row["kind"] == "decision"]
    assert [row["kind"] for row in decision_rows] == [
        "agent_request", "agent_response", "agent_request", "agent_response",
    ]
    assert {row.get("actor_id") for row in decision_rows} >= {"alpha", "opponent:beta"}
    assert all(row["day"] is None and row["phase"] is None and row["seat"] is None for row in rows)


@pytest.mark.asyncio
async def test_prepared_session_uses_full_sinks_without_duplicate_evidence() -> None:
    spec = _spec()
    plugin = CounterPlugin()
    transcript = Transcript(
        run_id=spec.run_id,
        metadata={"run_spec_hash": spec.spec_hash},
    )
    event_sources: list[dict[str, Any]] = []
    trace_sources: list[dict[str, Any]] = []
    harness_sources: list[dict[str, Any]] = []

    async def event_sink(payload: dict[str, Any]) -> None:
        event_sources.append(dict(payload))
        transcript.append(
            "event",
            payload,
            source_idx=len(event_sources) - 1,
        )

    def trace_sink(payload: dict[str, Any]) -> None:
        trace_sources.append(dict(payload))
        transcript.append(
            "decision",
            payload,
            source_idx=len(trace_sources) - 1,
        )

    async def harness_sink(payload: dict[str, Any]) -> None:
        harness_sources.append(dict(payload))
        transcript.append(
            "harness",
            payload,
            source_idx=len(harness_sources) - 1,
        )

    evidence = EnvironmentRunEvidence(
        transcript=transcript,
        event_sink=event_sink,
        trace_sink=trace_sink,
        harness_sink=harness_sink,
    )
    runtime = RecordingDecisionRuntime(
        on_trace=evidence.emit_trace,
        envelope_type=DecisionEnvelope,
        validate_envelope=validate_decision_envelope,
        expected_run_id=spec.run_id,
    )
    agents = {
        "alpha": CounterAgent("alpha", 2),
        "opponent:beta": CounterAgent("opponent:beta", 3),
    }
    agent_registry = AgentRegistry(agents.__getitem__)
    for actor_id in agents:
        agent_registry.resolve(actor_id)
    context = EnvironmentRunContext(
        run_id=spec.run_id,
        config=CounterConfig.model_validate(spec.environment_config),
        seeds=dict(spec.seeds),
        actor_spec=spec.actors,
        decision_runtime=runtime,
        emit_event=evidence.emit_event,
        emit_trace=evidence.emit_trace,
        resolve_agent=agent_registry.resolve,
        metadata=dict(spec.metadata),
    )
    # Interactive startup may commit these attestations before publishing the
    # running room. Core validates and reuses them instead of appending again.
    await harness_sink({
        "type": "run_started",
        "environment_id": plugin.descriptor.id,
        "environment_version": plugin.descriptor.version,
    })
    await harness_sink({
        "type": "agent_bindings_finalized",
        "actor_count": 2,
        "actor_ids": sorted(agents),
    })
    prepared = PreparedEnvironmentRun(
        descriptor=plugin.descriptor,
        session=CounterSession(context, plugin),
        decision_runtime=runtime,
        evidence=evidence,
        agent_registry=agent_registry,
    )

    result = await run_prepared_environment_run(spec, prepared=prepared)

    assert result.status == "completed"
    assert result.outcome == {"total": 5}
    assert plugin.closed_sessions == 1
    assert runtime.close_calls == 1
    assert [row["type"] for row in harness_sources] == [
        "run_started",
        "agent_bindings_finalized",
        "run_completed",
    ]
    assert len(event_sources) == 2
    assert len(trace_sources) == 4
    assert transcript.counts_by_kind() == {
        "harness": 3,
        "decision": 4,
        "event": 2,
    }
    for kind in ("event", "decision", "harness"):
        rows = [entry for entry in transcript.entries if entry.kind == kind]
        assert [entry.source_idx for entry in rows] == list(range(len(rows)))
    assert result.transcript_digest == transcript.stable_digest()
    with pytest.raises(RuntimeError, match="already been consumed"):
        await run_prepared_environment_run(spec, prepared=prepared)


@pytest.mark.asyncio
async def test_prepared_session_timeout_closes_session_and_shared_runtime() -> None:
    spec = _lifecycle_spec(
        {"run_delay_seconds": 10},
        run_timeout=0.01,
        cancellation_grace=0.05,
    )
    plugin = LifecyclePlugin()
    transcript = Transcript(
        run_id=spec.run_id,
        metadata={"run_spec_hash": spec.spec_hash},
    )
    evidence = EnvironmentRunEvidence(transcript=transcript)
    runtime = RecordingDecisionRuntime(
        on_trace=evidence.emit_trace,
        envelope_type=DecisionEnvelope,
        validate_envelope=validate_decision_envelope,
        expected_run_id=spec.run_id,
    )
    prepared = PreparedEnvironmentRun(
        descriptor=plugin.descriptor,
        session=LifecycleSession(
            LifecycleConfig.model_validate(spec.environment_config),
            plugin,
        ),
        decision_runtime=runtime,
        evidence=evidence,
        agent_registry=AgentRegistry(lambda _actor_id: None),
    )

    result = await run_prepared_environment_run(spec, prepared=prepared)

    assert result.status == "timed_out"
    assert result.error_type == "RunTimeout"
    assert plugin.close_calls == 1
    assert runtime.close_calls == 1
    assert plugin.run_task is not None and plugin.run_task.done()
    assert [
        entry.payload["type"]
        for entry in transcript.entries
        if entry.kind == "harness"
    ] == ["run_started", "run_failed", "agent_bindings_finalized"]


@pytest.mark.asyncio
async def test_prepared_session_rejects_cross_run_transcript_before_claim() -> None:
    spec = _lifecycle_spec({})
    plugin = LifecyclePlugin()
    evidence = EnvironmentRunEvidence(transcript=Transcript(run_id="other-run"))
    runtime = RecordingDecisionRuntime(
        on_trace=evidence.emit_trace,
        envelope_type=DecisionEnvelope,
        validate_envelope=validate_decision_envelope,
        expected_run_id=spec.run_id,
    )
    prepared = PreparedEnvironmentRun(
        descriptor=plugin.descriptor,
        session=LifecycleSession(LifecycleConfig(), plugin),
        decision_runtime=runtime,
        evidence=evidence,
        agent_registry=AgentRegistry(lambda _actor_id: None),
    )

    with pytest.raises(ValueError, match="transcript run_id"):
        await run_prepared_environment_run(spec, prepared=prepared)

    assert prepared._claimed is False
    assert plugin.close_calls == 0
    assert runtime.close_calls == 0


@pytest.mark.asyncio
async def test_non_terminal_environment_outcome_is_rejected_and_runner_fails() -> None:
    with pytest.raises(ValidationError, match="terminal result"):
        EnvironmentOutcome(terminal=False)

    class NonTerminalSession(CounterSession):
        async def run(self) -> EnvironmentOutcome:
            # A hostile plugin can bypass Pydantic construction; the runner's
            # independent lifecycle check must still fail closed.
            return EnvironmentOutcome.model_construct(
                terminal=False,
                status="completed",
                termination_reason=None,
                outcome={"total": 0},
                metrics={},
            )

    class NonTerminalPlugin(CounterPlugin):
        async def create_session(self, context: EnvironmentRunContext) -> CounterSession:
            return NonTerminalSession(context, self)

    plugin = NonTerminalPlugin()
    registry = EnvironmentRegistry()
    registry.register(plugin)
    result = await run_environment_run(
        _spec(),
        registry=registry,
        resolve_agent=lambda actor_id: CounterAgent(actor_id, 1),
    )

    assert result.status == "failed"
    assert result.error_type == "RuntimeError"
    assert result.termination_reason is None
    assert any(
        row["payload"].get("type") == "run_failed"
        for row in result.transcript["entries"]
    )


@pytest.mark.asyncio
async def test_generic_runner_times_out_and_always_closes_session() -> None:
    plugin = CounterPlugin()
    registry = EnvironmentRegistry()
    registry.register(plugin)
    result = await run_environment_run(
        _spec(timeout=0.01, delay=0.2),
        registry=registry,
        resolve_agent=lambda actor_id: CounterAgent(actor_id, 1),
    )

    assert result.status == "timed_out"
    assert result.error_type == "RunTimeout"
    assert plugin.closed_sessions == 1
    assert any(
        row["payload"].get("type") == "run_failed"
        for row in result.transcript["entries"]
    )


@pytest.mark.asyncio
async def test_generic_runner_delayed_cancellation_is_bounded_and_reclaimed() -> None:
    plugin = LifecyclePlugin()
    registry = EnvironmentRegistry()
    registry.register(plugin)
    started = asyncio.get_running_loop().time()

    result = await run_environment_run(
        _lifecycle_spec(
            {
                "run_delay_seconds": 10,
                "cancellation_delay_seconds": 0.02,
            },
            run_timeout=0.01,
            cancellation_grace=0.1,
        ),
        registry=registry,
        resolve_agent=lambda _actor_id: None,
    )

    assert asyncio.get_running_loop().time() - started < 0.25
    assert result.status == "timed_out"
    assert result.error_type == "RunTimeout"
    assert result.harness_metrics["cleanup_failure_count"] == 0
    assert plugin.close_calls == 1
    assert plugin.run_task is not None and plugin.run_task.done()
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("environment:")
    ]


@pytest.mark.asyncio
async def test_generic_runner_cleanup_exception_is_fatal_and_sanitized() -> None:
    plugin = LifecyclePlugin()
    registry = EnvironmentRegistry()
    registry.register(plugin)

    result = await run_environment_run(
        _lifecycle_spec({"close_raises": True}),
        registry=registry,
        resolve_agent=lambda _actor_id: None,
    )

    assert result.status == "failed"
    assert result.error_type == "EnvironmentCleanupError"
    assert result.harness_metrics["cleanup_failure_count"] == 1
    cleanup = next(
        row["payload"]
        for row in result.transcript["entries"]
        if row["payload"].get("type") == "run_cleanup_failed"
    )
    assert cleanup["fatal"] is True
    assert cleanup["failures"] == [{
        "stage": "session_close",
        "error_type": "RuntimeError",
        "timeout": False,
        "pending_task_count": 0,
    }]
    assert "private session cleanup detail" not in str(result.model_dump())


@pytest.mark.asyncio
async def test_generic_runner_external_cancellation_closes_and_reclaims_session() -> None:
    plugin = LifecyclePlugin()
    registry = EnvironmentRegistry()
    registry.register(plugin)
    execution = asyncio.create_task(
        run_environment_run(
            _lifecycle_spec({
                "run_delay_seconds": 10,
                "cancellation_delay_seconds": 0.01,
            }),
            registry=registry,
            resolve_agent=lambda _actor_id: None,
        )
    )
    await plugin.run_started.wait()

    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution

    assert plugin.close_calls == 1
    assert plugin.run_task is not None and plugin.run_task.done()
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("environment:")
    ]


@pytest.mark.asyncio
async def test_repeated_cancellation_attempts_all_cleanup_and_preserves_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.harness.core_runner as core_runner

    runtimes: list[Any] = []

    class RecordingRuntime(core_runner.DecisionRuntime):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.close_calls = 0
            runtimes.append(self)

        async def aclose(self) -> None:
            self.close_calls += 1
            await super().aclose()

    class BlockingCloseSession(LifecycleSession):
        async def aclose(self) -> None:
            self.owner.close_calls += 1
            self.owner.close_task = asyncio.current_task()
            self.owner.close_started.set()
            await asyncio.Event().wait()

    class BlockingClosePlugin(LifecyclePlugin):
        def __init__(self) -> None:
            super().__init__()
            self.close_started = asyncio.Event()

        async def create_session(
            self,
            context: EnvironmentRunContext,
        ) -> LifecycleSession:
            return BlockingCloseSession(
                LifecycleConfig.model_validate(context.config),
                self,
            )

    monkeypatch.setattr(core_runner, "DecisionRuntime", RecordingRuntime)
    plugin = BlockingClosePlugin()
    registry = EnvironmentRegistry()
    registry.register(plugin)
    execution = asyncio.create_task(
        run_environment_run(
            _lifecycle_spec({"run_delay_seconds": 10}),
            registry=registry,
            resolve_agent=lambda _actor_id: None,
        )
    )
    await plugin.run_started.wait()

    execution.cancel("first")
    await plugin.close_started.wait()
    execution.cancel("second")
    with pytest.raises(asyncio.CancelledError) as caught:
        await execution

    assert caught.value.args == ("first",)
    assert plugin.close_calls == 1
    assert len(runtimes) == 1
    assert runtimes[0].close_calls == 1
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("environment:")
    ]


@pytest.mark.asyncio
async def test_external_cancellation_closes_session_completed_during_create_cleanup() -> None:
    class LateCreatePlugin(CounterPlugin):
        def __init__(self) -> None:
            super().__init__()
            self.create_started = asyncio.Event()
            self.create_task: asyncio.Task[Any] | None = None

        async def create_session(self, context: EnvironmentRunContext) -> CounterSession:
            self.create_task = asyncio.current_task()
            self.create_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Resource setup completed while the runner was cooperatively
                # reclaiming the cancelled create task.
                await asyncio.sleep(0.01)
                return CounterSession(context, self)

    plugin = LateCreatePlugin()
    registry = EnvironmentRegistry()
    registry.register(plugin)
    execution = asyncio.create_task(
        run_environment_run(
            _spec(),
            registry=registry,
            resolve_agent=lambda _actor_id: None,
        )
    )
    await plugin.create_started.wait()

    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution

    assert plugin.closed_sessions == 1
    assert plugin.create_task is not None and plugin.create_task.done()
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("environment:")
    ]


@pytest.mark.asyncio
async def test_generic_runner_reports_unforceable_cleanup_task_before_returning() -> None:
    plugin = LifecyclePlugin()
    registry = EnvironmentRegistry()
    registry.register(plugin)
    started = asyncio.get_running_loop().time()
    try:
        result = await run_environment_run(
            _lifecycle_spec(
                {"close_ignores_cancellation": True},
                cleanup_timeout=0.01,
                cancellation_grace=0.02,
            ),
            registry=registry,
            resolve_agent=lambda _actor_id: None,
        )

        assert asyncio.get_running_loop().time() - started < 0.2
        assert result.status == "failed"
        assert result.error_type == "EnvironmentCleanupError"
        assert result.harness_metrics["pending_task_count"] == 1
        cleanup = next(
            row["payload"]
            for row in result.transcript["entries"]
            if row["payload"].get("type") == "run_cleanup_failed"
        )
        assert cleanup["pending_task_count"] == 1
        assert cleanup["failures"][0]["error_type"] == "TaskIgnoredCancellation"
        assert plugin.close_task is not None and not plugin.close_task.done()
    finally:
        plugin.release_close.set()
        await asyncio.sleep(0)

    assert plugin.close_task is not None and plugin.close_task.done()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cancellation_grace_seconds", -0.01),
        ("cancellation_grace_seconds", 61),
        ("cleanup_timeout_seconds", 0),
        ("cleanup_timeout_seconds", 301),
    ],
)
def test_execution_spec_rejects_unbounded_cleanup_configuration(field, value) -> None:
    with pytest.raises(ValidationError):
        ExecutionSpec(**{field: value})


def test_generic_harness_modules_do_not_import_game_domain() -> None:
    root = Path(__file__).parents[1] / "src" / "harness"
    for filename in (
        "core_protocol.py",
        "core_spec.py",
        "environment.py",
        "registry.py",
        "core_runner.py",
        "decision_runtime.py",
    ):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        assert not any(module == "game" or module.startswith("game.") for module in imports)
