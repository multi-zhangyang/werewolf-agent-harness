"""Executable adversarial scenarios for the environment-neutral harness."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import pytest

from src.harness.core_protocol import (
    ActionChoice,
    ActionOption,
    ActionRequest,
    DecisionEnvelope,
    validate_decision_envelope,
)
from src.harness.core_spec import CoreRunSpec, EnvironmentRef, ExecutionSpec
from src.harness.environment import (
    DecisionContract,
    EnvironmentDescriptor,
    EnvironmentOutcome,
    EnvironmentRunContext,
)
from src.harness.registry import EnvironmentRegistry, EnvironmentRegistryError
from src.harness.scenarios import (
    ADVERSARIAL_SCENARIO_REPORT_VERSION,
    ADVERSARIAL_SCENARIO_SPEC_VERSION,
    ADVERSARIAL_SUITE_REPORT_VERSION,
    ExpectedInvariant,
    ScenarioSpec,
    run_adversarial_scenario,
    run_adversarial_scenario_suite,
)


ScenarioMode = Literal[
    "identity_mismatch",
    "illegal_action",
    "invalid_arguments",
    "provider_fault",
    "deadline_timeout",
    "hostile_leak",
    "contradictory_report",
    "collusion",
]
ENVIRONMENT_ID = "test.adversarial-boundary"
ENVIRONMENT_VERSION = "2026.07.1"
LEAK_SECRET = "sk-adversarial-leak-123456789"
FAULT_SECRET = "sk-provider-fault-987654321"


class AdversarialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ScenarioMode
    actor_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _right_actor_count(self) -> "AdversarialConfig":
        expected = 2 if self.mode == "collusion" else 1
        if len(self.actor_ids) != expected:
            raise ValueError(f"{self.mode} requires {expected} actors")
        return self


class ProviderTransportFault(RuntimeError):
    pass


class ScriptedAdversarialAgent:
    def __init__(
        self,
        actor_id: str,
        behavior: ScenarioMode,
        *,
        shared_strategy: dict[str, Any] | None = None,
        opaque_leak: str | None = None,
    ) -> None:
        self.actor_id = actor_id
        self.behavior = behavior
        self.shared_strategy = shared_strategy
        self.opaque_leak = opaque_leak
        self.calls = 0
        self.observations: list[dict[str, Any]] = []

    async def decide(self, request: ActionRequest) -> DecisionEnvelope:
        self.calls += 1
        self.observations.append(dict(request.observation))
        if self.behavior == "provider_fault":
            raise ProviderTransportFault(
                f"provider rejected Authorization: Bearer {FAULT_SECRET}"
            )
        if self.behavior == "deadline_timeout":
            await asyncio.sleep(0.2)
            raise AssertionError("the decision deadline should cancel this coroutine")
        if self.behavior == "identity_mismatch":
            return self._envelope(
                request,
                actor_id="unauthorized-actor",
                action="commit",
                arguments={"value": 1},
            )
        if self.behavior == "illegal_action":
            return self._envelope(request, action="delete_all", arguments={})
        if self.behavior == "invalid_arguments":
            return self._envelope(
                request,
                action="commit",
                arguments={"value": 0, "unexpected": True},
            )
        if self.behavior == "hostile_leak":
            if self.opaque_leak is not None:
                return DecisionEnvelope(
                    request_id=request.request_id,
                    actor_id=self.actor_id,
                    choice=ActionChoice(action="commit", arguments={"value": 1}),
                    private_reasoning=self.opaque_leak,
                    metadata={"hostile": {"note": self.opaque_leak}},
                )
            return DecisionEnvelope(
                request_id=request.request_id,
                actor_id=self.actor_id,
                choice=ActionChoice(action="commit", arguments={"value": 1}),
                private_reasoning=f"credential payload {LEAK_SECRET}",
                metadata={
                    "hostile": {
                        "api_key": LEAK_SECRET,
                        "message": f"Bearer {LEAK_SECRET}",
                    }
                },
            )
        if self.behavior == "contradictory_report":
            return self._envelope(
                request,
                action="declare",
                arguments={"winner": "attackers"},
            )
        if self.behavior == "collusion":
            assert self.shared_strategy is not None
            signal = int(self.shared_strategy.setdefault("signal", 1))
            self.shared_strategy.setdefault("actors", []).append(self.actor_id)
            return self._envelope(
                request,
                action="signal",
                arguments={"bit": signal},
            )
        raise AssertionError(f"unsupported scripted behavior: {self.behavior}")

    def _envelope(
        self,
        request: ActionRequest,
        *,
        action: str,
        arguments: dict[str, Any],
        actor_id: str | None = None,
    ) -> DecisionEnvelope:
        return DecisionEnvelope(
            request_id=request.request_id,
            actor_id=actor_id or self.actor_id,
            choice=ActionChoice(action=action, arguments=arguments),
            parse_status="not_applicable",
        )


class AdversarialSession:
    def __init__(
        self,
        context: EnvironmentRunContext,
        owner: "AdversarialPlugin",
    ) -> None:
        self.context = context
        self.owner = owner

    async def run(self) -> EnvironmentOutcome:
        config = AdversarialConfig.model_validate(self.context.config)
        if config.mode == "collusion":
            return await self._run_collusion(config)

        actor_id = config.actor_ids[0]
        request = self._request(config.mode, actor_id, step=1)
        try:
            envelope = await self.context.decision_runtime.execute(
                self.context.resolve_agent(actor_id),
                request,
            )
        except Exception as err:  # scenario records the rejected boundary fact
            return EnvironmentOutcome(
                terminal=True,
                outcome={
                    "resolution": "rejected",
                    "error_type": str(
                        getattr(err, "error_type", type(err).__name__)
                    ),
                },
            )

        assert isinstance(envelope.choice, ActionChoice)
        if config.mode == "hostile_leak":
            hostile = dict(envelope.metadata["hostile"])
            await self.context.emit_event({"type": "hostile_echo", "echo": hostile})
            return EnvironmentOutcome(
                terminal=True,
                outcome={"resolution": "accepted", "echo": hostile},
            )
        if config.mode == "contradictory_report":
            await self.context.emit_event({
                "type": "claim_observed",
                "claim": envelope.choice.arguments["winner"],
            })
            # The terminal result is owned by the environment, not by an
            # agent's self-reported winner.
            return EnvironmentOutcome(
                terminal=True,
                outcome={"winner": "defenders", "score": 1},
            )
        raise AssertionError(f"{config.mode} unexpectedly passed validation")

    async def _run_collusion(
        self,
        config: AdversarialConfig,
    ) -> EnvironmentOutcome:
        private_truth = f"private-truth-{self.context.seeds['scenario']}"
        self.owner.private_truths[self.context.run_id] = private_truth
        signals: list[int] = []
        for step, actor_id in enumerate(config.actor_ids, start=1):
            envelope = await self.context.decision_runtime.execute(
                self.context.resolve_agent(actor_id),
                self._request(config.mode, actor_id, step=step),
            )
            assert isinstance(envelope.choice, ActionChoice)
            signals.append(int(envelope.choice.arguments["bit"]))
        return EnvironmentOutcome(
            terminal=True,
            outcome={
                "coordinated": len(set(signals)) == 1,
                "signal_count": len(signals),
            },
        )

    def _request(
        self,
        mode: ScenarioMode,
        actor_id: str,
        *,
        step: int,
    ) -> ActionRequest:
        deadline = time.monotonic() + 0.01 if mode == "deadline_timeout" else None
        if mode == "contradictory_report":
            option = ActionOption(
                name="declare",
                input_schema={
                    "type": "object",
                    "properties": {
                        "winner": {"enum": ["attackers", "defenders"]},
                    },
                    "required": ["winner"],
                    "additionalProperties": False,
                },
            )
        elif mode == "collusion":
            option = ActionOption(
                name="signal",
                input_schema={
                    "type": "object",
                    "properties": {"bit": {"type": "integer", "enum": [0, 1]}},
                    "required": ["bit"],
                    "additionalProperties": False,
                },
            )
        else:
            option = ActionOption(
                name="commit",
                input_schema={
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer", "minimum": 1, "maximum": 3},
                    },
                    "required": ["value"],
                    "additionalProperties": False,
                },
            )
        metadata: dict[str, Any] = {}
        if deadline is not None:
            metadata["effective_timeout_seconds"] = 0.01
        return ActionRequest(
            request_id=f"{self.context.run_id}:request:{step}",
            run_id=self.context.run_id,
            actor_id=actor_id,
            observation={"public_round": step, "public_seed_draw": self.context.rng("scenario").randrange(100)},
            legal_actions=[option],
            deadline_monotonic=deadline,
            labels={"stage": "adversarial", "action": option.name, "step": step},
            metadata=metadata,
        )

    async def aclose(self) -> None:
        self.owner.closed_session_count += 1


class AdversarialPlugin:
    descriptor = EnvironmentDescriptor(
        id=ENVIRONMENT_ID,
        version=ENVIRONMENT_VERSION,
        required_seeds=("scenario",),
        capabilities=("adversarial_test", "multi_agent"),
    )
    decision_contract = DecisionContract(
        envelope_type=DecisionEnvelope,
        validate_envelope=validate_decision_envelope,
    )

    def __init__(self) -> None:
        self.closed_session_count = 0
        self.private_truths: dict[str, str] = {}

    def resolve_config(
        self,
        raw_config: Mapping[str, Any],
        _seeds: Mapping[str, int],
    ) -> BaseModel:
        return AdversarialConfig.model_validate(raw_config)

    async def create_session(
        self,
        context: EnvironmentRunContext,
    ) -> AdversarialSession:
        return AdversarialSession(context, self)


def _common_invariants(
    *,
    outcome: dict[str, Any],
    request_count: int = 1,
) -> list[ExpectedInvariant]:
    return [
        ExpectedInvariant(kind="request_terminal_pairing", expected=True),
        ExpectedInvariant(kind="minimum_request_count", expected=request_count),
        ExpectedInvariant(kind="no_fabricated_choice", expected=True),
        ExpectedInvariant(kind="secret_absence", expected=True),
        ExpectedInvariant(kind="run_status", expected="completed"),
        ExpectedInvariant(kind="error_type", expected=None),
        ExpectedInvariant(kind="outcome_equals", expected=outcome),
        ExpectedInvariant(kind="fact_count", subject="accepted_skip", expected=0),
        ExpectedInvariant(kind="fact_count", subject="rejected_skip", expected=0),
        ExpectedInvariant(kind="fact_count", subject="target_argument", expected=0),
    ]


def _scenario(
    mode: ScenarioMode,
    *,
    category: str,
    outcome: dict[str, Any],
    extra: list[ExpectedInvariant],
    actor_ids: tuple[str, ...] = ("actor:primary",),
    seed: int,
) -> ScenarioSpec:
    request_count = len(actor_ids)
    return ScenarioSpec(
        scenario_id=mode.replace("_", "-"),
        category=category,
        run_spec=CoreRunSpec(
            run_id=f"adversarial-{mode.replace('_', '-')}",
            environment=EnvironmentRef(id=ENVIRONMENT_ID, version=ENVIRONMENT_VERSION),
            environment_config={"mode": mode, "actor_ids": list(actor_ids)},
            seeds={"scenario": seed},
            execution=ExecutionSpec(run_timeout_seconds=1),
        ),
        expected_invariants=tuple(
            _common_invariants(outcome=outcome, request_count=request_count) + extra
        ),
    )


def _scenario_specs() -> list[ScenarioSpec]:
    rejected = lambda error_type: {  # noqa: E731 - compact factual fixture
        "resolution": "rejected",
        "error_type": error_type,
    }
    return [
        _scenario(
            "identity_mismatch",
            category="protocol_identity",
            outcome=rejected("DecisionEnvelopeRejected"),
            seed=101,
            extra=[
                ExpectedInvariant(kind="validation_issue_present", subject="actor_id_mismatch", expected=True),
                ExpectedInvariant(kind="terminal_kind_count", subject="agent_response", expected=1),
                ExpectedInvariant(kind="fact_count", subject="rejected_action", expected=1),
                ExpectedInvariant(kind="fact_count", subject="accepted_action", expected=0),
            ],
        ),
        _scenario(
            "illegal_action",
            category="protocol_action_schema",
            outcome=rejected("DecisionEnvelopeRejected"),
            seed=102,
            extra=[
                ExpectedInvariant(kind="validation_issue_present", subject="action_not_legal", expected=True),
                ExpectedInvariant(kind="terminal_kind_count", subject="agent_response", expected=1),
                ExpectedInvariant(kind="fact_count", subject="rejected_action", expected=1),
                ExpectedInvariant(kind="fact_count", subject="accepted_action", expected=0),
            ],
        ),
        _scenario(
            "invalid_arguments",
            category="protocol_action_schema",
            outcome=rejected("DecisionEnvelopeRejected"),
            seed=103,
            extra=[
                ExpectedInvariant(kind="validation_issue_present", subject="action_arguments_invalid", expected=True),
                ExpectedInvariant(kind="terminal_kind_count", subject="agent_response", expected=1),
                ExpectedInvariant(kind="fact_count", subject="rejected_action", expected=1),
                ExpectedInvariant(kind="fact_count", subject="accepted_action", expected=0),
            ],
        ),
        _scenario(
            "provider_fault",
            category="agent_fault",
            outcome=rejected("ProviderTransportFault"),
            seed=104,
            extra=[
                ExpectedInvariant(kind="terminal_kind_count", subject="agent_response_failed", expected=1),
                ExpectedInvariant(kind="terminal_error_type_present", subject="ProviderTransportFault", expected=True),
                ExpectedInvariant(kind="fact_count", subject="accepted_action", expected=0),
            ],
        ),
        _scenario(
            "deadline_timeout",
            category="deadline",
            outcome=rejected("DecisionTimeout"),
            seed=105,
            extra=[
                ExpectedInvariant(kind="terminal_kind_count", subject="agent_response_failed", expected=1),
                ExpectedInvariant(kind="terminal_error_type_present", subject="DecisionTimeout", expected=True),
                ExpectedInvariant(kind="fact_count", subject="accepted_action", expected=0),
            ],
        ),
        _scenario(
            "hostile_leak",
            category="confidentiality",
            outcome={
                "resolution": "accepted",
                "echo": {
                    "api_key": "[redacted]",
                    "message": "Bearer [redacted]",
                },
            },
            seed=106,
            extra=[
                ExpectedInvariant(kind="terminal_kind_count", subject="agent_response", expected=1),
                ExpectedInvariant(kind="fact_count", subject="accepted_action", expected=1),
            ],
        ),
        _scenario(
            "contradictory_report",
            category="outcome_integrity",
            outcome={"winner": "defenders", "score": 1},
            seed=107,
            extra=[
                ExpectedInvariant(kind="terminal_kind_count", subject="agent_response", expected=1),
                ExpectedInvariant(kind="fact_count", subject="accepted_action", expected=1),
            ],
        ),
        _scenario(
            "collusion",
            category="multi_agent_collusion",
            outcome={"coordinated": True, "signal_count": 2},
            actor_ids=("coalition:a", "coalition:b"),
            seed=108,
            extra=[
                ExpectedInvariant(kind="terminal_kind_count", subject="agent_response", expected=2),
                ExpectedInvariant(kind="fact_count", subject="accepted_action", expected=2),
                ExpectedInvariant(kind="request_observation_key_absent", subject="private_truth", expected=True),
            ],
        ),
    ]


@pytest.mark.asyncio
async def test_versioned_adversarial_suite_executes_core_runs_and_reports_facts() -> None:
    plugin = AdversarialPlugin()
    registry = EnvironmentRegistry()
    registry.register(plugin)
    specs = _scenario_specs()
    agents: dict[str, dict[str, ScriptedAdversarialAgent]] = {}
    collusion_strategy: dict[str, Any] = {}
    for spec in specs:
        config = AdversarialConfig.model_validate(spec.run_spec.environment_config)
        shared = collusion_strategy if config.mode == "collusion" else None
        agents[spec.scenario_id] = {
            actor_id: ScriptedAdversarialAgent(
                actor_id,
                config.mode,
                shared_strategy=shared,
            )
            for actor_id in config.actor_ids
        }

    collusion_truth = "private-truth-108"
    report = await run_adversarial_scenario_suite(
        specs,
        registry=registry,
        resolve_agent=lambda scenario, actor_id: agents[scenario.scenario_id][actor_id],
        sensitive_markers={
            "provider-fault": [FAULT_SECRET],
            "hostile-leak": [LEAK_SECRET],
            "collusion": [collusion_truth],
        },
    )

    assert report.schema_version == ADVERSARIAL_SUITE_REPORT_VERSION
    assert report.status == "passed"
    assert (report.scenario_count, report.passed_count, report.failed_count) == (8, 8, 0)
    assert report.category_counts == {
        "agent_fault": 1,
        "confidentiality": 1,
        "deadline": 1,
        "multi_agent_collusion": 1,
        "outcome_integrity": 1,
        "protocol_action_schema": 2,
        "protocol_identity": 1,
    }
    assert plugin.closed_session_count == len(specs)

    expected_outcomes = {
        spec.scenario_id: next(
            item.expected
            for item in spec.expected_invariants
            if item.kind == "outcome_equals"
        )
        for spec in specs
    }
    for item in report.reports:
        assert item.schema_version == ADVERSARIAL_SCENARIO_REPORT_VERSION
        assert item.scenario_schema_version == ADVERSARIAL_SCENARIO_SPEC_VERSION
        assert item.environment_id == ENVIRONMENT_ID
        assert item.environment_version == ENVIRONMENT_VERSION
        assert item.run_status == "completed"
        assert item.error_type is None
        assert item.outcome == expected_outcomes[item.scenario_id]
        assert item.request_count == item.terminal_count
        assert item.request_count >= 1
        assert item.fact_counts["fabricated_choice_terminal"] == 0
        assert item.fact_counts["accepted_skip"] == 0
        assert item.fact_counts["rejected_skip"] == 0
        assert item.fact_counts["target_argument"] == 0
        assert item.secret_marker_match_count == 0
        assert all(invariant.passed for invariant in item.invariant_results)

    assert collusion_strategy == {
        "signal": 1,
        "actors": ["coalition:a", "coalition:b"],
    }
    for agent in agents["collusion"].values():
        assert agent.calls == 1
        assert all("private_truth" not in observation for observation in agent.observations)
        assert all(collusion_truth not in str(observation) for observation in agent.observations)
    assert plugin.private_truths["adversarial-collusion"] == collusion_truth

    serialized_report = report.model_dump_json()
    assert LEAK_SECRET not in serialized_report
    assert FAULT_SECRET not in serialized_report
    assert collusion_truth not in serialized_report
    assert "coverage" not in type(report).model_fields


@pytest.mark.asyncio
async def test_scenario_uses_exact_environment_version() -> None:
    registry = EnvironmentRegistry()
    registry.register(AdversarialPlugin())
    spec = _scenario_specs()[0]
    wrong_version = spec.run_spec.model_copy(update={
        "environment": EnvironmentRef(id=ENVIRONMENT_ID, version="2026.07.2")
    })
    mismatched = spec.model_copy(update={"run_spec": wrong_version})

    with pytest.raises(EnvironmentRegistryError, match="unknown environment plugin"):
        await run_adversarial_scenario(
            mismatched,
            registry=registry,
            resolve_agent=lambda actor_id: ScriptedAdversarialAgent(
                actor_id, "identity_mismatch"
            ),
        )


@pytest.mark.asyncio
async def test_opaque_secret_marker_fails_closed_without_leaking_into_report() -> None:
    opaque_secret = "CLASSIFIED-MARKER-7f3c9a"
    plugin = AdversarialPlugin()
    registry = EnvironmentRegistry()
    registry.register(plugin)
    spec = _scenario(
        "hostile_leak",
        category="confidentiality",
        outcome={
            "resolution": "accepted",
            "echo": {"note": "[redacted]"},
        },
        seed=201,
        extra=[
            ExpectedInvariant(
                kind="terminal_kind_count",
                subject="agent_response",
                expected=1,
            ),
            ExpectedInvariant(
                kind="fact_count",
                subject="accepted_action",
                expected=1,
            ),
        ],
    )

    report = await run_adversarial_scenario(
        spec,
        registry=registry,
        resolve_agent=lambda actor_id: ScriptedAdversarialAgent(
            actor_id,
            "hostile_leak",
            opaque_leak=opaque_secret,
        ),
        sensitive_markers=[opaque_secret],
    )

    assert not report.passed
    assert report.secret_marker_match_count > 0
    assert report.outcome == {
        "resolution": "accepted",
        "echo": {"note": "[redacted]"},
    }
    secret_invariant = next(
        item for item in report.invariant_results if item.kind == "secret_absence"
    )
    assert not secret_invariant.passed
    assert opaque_secret not in report.model_dump_json()


def test_scenario_spec_requires_fixed_seed_and_foundational_invariants() -> None:
    template = _scenario_specs()[0]
    seedless_run = CoreRunSpec(
        run_id="seedless",
        environment=EnvironmentRef(id=ENVIRONMENT_ID, version=ENVIRONMENT_VERSION),
        environment_config={"mode": "identity_mismatch", "actor_ids": ["actor"]},
        seeds={},
    )
    with pytest.raises(ValueError, match="at least one fixed seed"):
        ScenarioSpec(
            scenario_id="seedless",
            category="protocol_identity",
            run_spec=seedless_run,
            expected_invariants=template.expected_invariants,
        )

    with pytest.raises(ValueError, match="missing required invariants"):
        ScenarioSpec(
            scenario_id="under-specified",
            category="protocol_identity",
            run_spec=template.run_spec,
            expected_invariants=(
                ExpectedInvariant(kind="run_status", expected="completed"),
            ),
        )
