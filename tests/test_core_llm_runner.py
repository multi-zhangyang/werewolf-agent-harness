"""Generic real-model runner and smoke-evidence integration tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.environments.cipher_council import (
    CipherCouncilV2EnvironmentPlugin,
    verify_cipher_council_v2_artifacts,
)
from src.environments.werewolf.plugin import WerewolfEnvironmentPlugin
from src.harness.artifacts import write_run_artifacts
from src.harness.core_llm_runner import run_core_llm_environment
from src.harness.core_spec import ActorSpec, CoreRunSpec, EnvironmentRef, ExecutionSpec
from src.harness.model_manifest import ModelConfigManifest
from src.harness.registry import EnvironmentRegistry
from src.harness.smoke import verify_real_model_smoke_artifacts
from src.llm.models import ModelConfig
from src.llm.router import LLMToolCall, LLMToolResponse


class FakeRouterStats:
    def __init__(self) -> None:
        self.calls = 0
        self.successes = 0
        self.failures = 0
        self.retries = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_latency = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "retries": self.retries,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_latency": self.total_latency,
            "avg_latency": (
                self.total_latency / self.calls if self.calls else 0.0
            ),
        }


class ToolOnlyFakeRouter:
    """A deterministic normal tool-turn Router with auditable usage facts."""

    def __init__(self) -> None:
        self.stats = FakeRouterStats()
        self.closed = False

    async def complete_tools(
        self,
        messages: list[dict[str, Any]],
        _config: ModelConfig,
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMToolResponse:
        self.stats.calls += 1
        self.stats.successes += 1
        self.stats.total_tokens_in += 11
        self.stats.total_tokens_out += 3
        self.stats.total_latency += 0.01
        call_number = self.stats.calls
        prompt = json.loads(messages[-1]["content"])
        stage = prompt["labels"]["stage"]
        public_state = prompt["observation"]["public_state"]
        action_name = tools[0]["function"]["name"]
        if stage == "cipher_council":
            arguments = {"message": f"private Cipher strategy {call_number}"}
        elif stage == "deliberation":
            arguments = {"message": f"public statement {call_number}"}
        elif stage == "nomination":
            arguments = {
                "members": public_state["actor_ids"][:public_state["mission_size"]],
            }
        elif stage == "vote":
            arguments = {"approve": True}
        elif stage == "mission_commitment":
            arguments = {"commitment": "support"}
        else:  # pragma: no cover - detects an unrecognized environment stage.
            raise AssertionError(stage)
        usage = {"prompt_tokens": 11, "completion_tokens": 3}
        trace_context = dict(kwargs["trace_context"])
        return LLMToolResponse(
            content="",
            finish_reason="tool_calls",
            usage=usage,
            latency=0.01,
            call_id=f"provider-call-{call_number}",
            request_hash=f"request-hash-{call_number}",
            tool_calls=(LLMToolCall(
                call_id=f"tool-call-{call_number}",
                name=action_name,
                arguments=arguments,
                raw_arguments=json.dumps(arguments, sort_keys=True),
            ),),
            trace={
                "call_id": f"provider-call-{call_number}",
                "context": trace_context,
                "request_hash": f"request-hash-{call_number}",
                "response_hash": f"response-hash-{call_number}",
                "usage": usage,
                "latency": 0.01,
                "finish_reason": "tool_calls",
                "transport_attempt_count": 1,
                "transport_attempts": [],
                "tool_call_count": 1,
                "tool_calls": [{
                    "call_id": f"tool-call-{call_number}",
                    "name": action_name,
                    "arguments_hash": f"arguments-hash-{call_number}",
                }],
                "parse": None,
            },
        )

    async def aclose(self) -> None:
        self.closed = True


def _model_config(*, model: str = "ordinary-tool-model") -> ModelConfig:
    return ModelConfig(
        provider="openai",
        model=model,
        api_base="https://example.invalid/v1",
        api_key="test-key-not-written-to-artifacts",
        max_tokens=0,
    )


def _spec(
    *,
    default_model: ModelConfig | None = None,
    model_overrides: dict[str, ModelConfig] | None = None,
    actors: ActorSpec | None = None,
) -> CoreRunSpec:
    declared_actors = actors or ActorSpec(
        default_model=ModelConfigManifest.from_config(
            default_model or _model_config()
        ).model_dump(mode="json"),
        model_overrides={
            actor_id: ModelConfigManifest.from_config(config).model_dump(mode="json")
            for actor_id, config in (model_overrides or {}).items()
        },
    )
    return CoreRunSpec(
        run_id="core-llm-council",
        environment=EnvironmentRef(id="council.cipher", version="2"),
        environment_config={
            "player_names": ["A", "B", "C", "D", "E"],
            "cipher_count": 2,
            "mission_sizes": [2],
            "victory_target": 1,
            "max_proposals_per_mission": 1,
        },
        seeds={"roles": 17, "order": 31},
        execution=ExecutionSpec(decision_timeout_seconds=1),
        actors=declared_actors,
        metadata={"suite": "core-llm-runner"},
    )


@pytest.mark.asyncio
async def test_generic_llm_runner_creates_one_tool_actor_per_environment_actor_and_smoke_artifact(
    tmp_path: Path,
):
    registry = EnvironmentRegistry()
    registry.register(CipherCouncilV2EnvironmentPlugin())
    router = ToolOnlyFakeRouter()
    spec = _spec()

    result = await run_core_llm_environment(
        spec,
        registry=registry,
        model_config=_model_config(),
        router=router,  # type: ignore[arg-type]
        close_router=True,
    )

    assert result.status == "completed"
    assert result.outcome["winner"] == "council"
    assert result.harness_metrics["model_actor_count"] == 5
    assert result.harness_metrics["resolved_actor_count"] == 5
    assert result.metrics["model_calls"] == router.stats.calls == 15
    assert result.metrics["cipher_council_faction_size"] == 2
    assert result.metrics["cipher_council_round_count"] == 1
    assert result.metrics["cipher_council_request_count"] == 2
    assert result.metrics["cipher_council_message_count"] == 2
    assert result.metrics["cipher_council_absent_count"] == 0
    assert result.metrics["router_stats_delta"]["total_tokens_in"] == 15 * 11
    assert result.metrics["router_stats_delta"]["total_tokens_out"] == 15 * 3
    assert router.closed is True

    decision_payloads = [
        row["payload"]
        for row in result.transcript["entries"]
        if row["kind"] == "decision"
    ]
    assert sum(row.get("type") == "agent_turn_started" for row in decision_payloads) == 15
    assert sum(row.get("type") == "model_generation" for row in decision_payloads) == 15
    assert sum(row.get("type") == "tool_call_requested" for row in decision_payloads) == 15
    assert sum(row.get("type") == "tool_result" for row in decision_payloads) == 15
    assert sum(row.get("type") == "agent_action_submitted" for row in decision_payloads) == 15
    assert sum(row.get("type") == "decision_consumed" for row in decision_payloads) == 15
    assert sum(row.get("type") == "rules_result" for row in decision_payloads) == 15

    paths = write_run_artifacts(result, spec, tmp_path)
    report = verify_real_model_smoke_artifacts(paths["run_dir"])
    coordination = verify_cipher_council_v2_artifacts(paths["run_dir"])
    assert report.model_calls == 15
    assert report.model_backed_decision_count == 15
    assert report.tool_call_count == 15
    assert report.tool_result_count == 15
    assert report.environment_consumed_action_count == 15
    assert coordination.cipher_council_message_count == 2
    assert coordination.cipher_council_absent_count == 0


@pytest.mark.asyncio
async def test_generic_llm_runner_rejects_missing_real_model_configuration():
    registry = EnvironmentRegistry()
    registry.register(CipherCouncilV2EnvironmentPlugin())

    router = ToolOnlyFakeRouter()
    result = await run_core_llm_environment(
        _spec(),
        registry=registry,
        model_config=ModelConfig(provider="openai", model="", api_key=""),
        router=router,  # type: ignore[arg-type]
    )

    assert result.status == "failed"
    assert result.error_type == "ValueError"
    assert result.error == "incomplete real model configuration for council:1: model,api_key"
    assert result.metrics["model_calls"] == 0
    assert router.stats.calls == 0


@pytest.mark.asyncio
async def test_generic_llm_runner_requires_a_declared_safe_model_binding():
    registry = EnvironmentRegistry()
    registry.register(CipherCouncilV2EnvironmentPlugin())
    router = ToolOnlyFakeRouter()

    result = await run_core_llm_environment(
        _spec(actors=ActorSpec()),
        registry=registry,
        model_config=_model_config(),
        router=router,  # type: ignore[arg-type]
    )

    assert result.status == "failed"
    assert result.error_type == "ValueError"
    assert result.error == "CoreRunSpec ActorSpec has no model binding for council:1"
    assert result.metrics["model_calls"] == 0
    assert router.stats.calls == 0


@pytest.mark.asyncio
async def test_generic_llm_runner_rejects_a_runtime_model_that_disagrees_with_default_binding():
    registry = EnvironmentRegistry()
    registry.register(CipherCouncilV2EnvironmentPlugin())
    router = ToolOnlyFakeRouter()

    result = await run_core_llm_environment(
        _spec(default_model=_model_config(model="declared-model")),
        registry=registry,
        model_config=_model_config(model="runtime-model"),
        router=router,  # type: ignore[arg-type]
    )

    assert result.status == "failed"
    assert result.error_type == "ValueError"
    assert result.error == (
        "resolved model actor does not match CoreRunSpec ActorSpec for council:1"
    )
    assert result.metrics["model_calls"] == 0
    assert router.stats.calls == 0


@pytest.mark.asyncio
async def test_generic_llm_runner_rejects_a_runtime_override_that_disagrees_with_binding():
    registry = EnvironmentRegistry()
    registry.register(CipherCouncilV2EnvironmentPlugin())
    router = ToolOnlyFakeRouter()

    result = await run_core_llm_environment(
        _spec(model_overrides={"council:2": _model_config(model="declared-seat-two")}),
        registry=registry,
        model_config=_model_config(),
        actor_model_configs={"council:2": _model_config(model="runtime-seat-two")},
        router=router,  # type: ignore[arg-type]
    )

    assert result.status == "failed"
    assert result.error_type == "ValueError"
    assert result.error == (
        "resolved model actor does not match CoreRunSpec ActorSpec for council:2"
    )
    assert result.metrics["model_calls"] == 0
    assert router.stats.calls == 0


@pytest.mark.asyncio
async def test_generic_llm_runner_rejects_a_legacy_environment_contract_before_transport():
    registry = EnvironmentRegistry()
    registry.register(WerewolfEnvironmentPlugin())
    router = ToolOnlyFakeRouter()
    spec = _spec().model_copy(update={
        "environment": EnvironmentRef(id="werewolf.classic", version="1"),
        "environment_config": {},
        "seeds": {},
    })

    with pytest.raises(
        ValueError,
        match="not compatible with the Core tool decision protocol",
    ):
        await run_core_llm_environment(
            spec,
            registry=registry,
            model_config=_model_config(),
            router=router,  # type: ignore[arg-type]
        )

    assert router.stats.calls == 0
