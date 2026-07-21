"""Unit tests for the environment-neutral real tool-calling actor."""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import json
from typing import Any

import pytest

from src.harness.core_llm_actor import CoreToolActor
from src.harness.core_protocol import (
    ActionChoice,
    ActionOption,
    ActionRequest,
    SkipChoice,
    SkipPolicy,
    validate_decision_envelope,
)
from src.harness.environment import AgentBindingError, AgentRegistry
from src.harness.errors import AgentDecisionError
from src.harness.visibility import project_payload_for_audience
from src.llm.models import ModelConfig
from src.llm.router import LLMResponseError, LLMToolCall, LLMToolResponse


def _request(*, allow_skip: bool = False) -> ActionRequest:
    return ActionRequest(
        request_id="core-request-1",
        run_id="core-run-1",
        actor_id="council:1",
        observation={
            "private_identity": {"team": "cipher"},
            "public_state": {"mission": 1},
        },
        legal_actions=[
            ActionOption(
                name="submit_vote",
                input_schema={
                    "type": "object",
                    "properties": {"approve": {"type": "boolean"}},
                    "required": ["approve"],
                    "additionalProperties": False,
                },
                metadata={"stage": "vote"},
            )
        ],
        skip_policy=SkipPolicy(allowed=allow_skip, reason_required=True),
        labels={"environment": "council.cipher", "stage": "vote"},
    )


def _response(
    name: str,
    arguments: dict[str, Any],
    *,
    reasoning: str = "I will keep this private.",
) -> LLMToolResponse:
    return LLMToolResponse(
        content="",
        finish_reason="tool_calls",
        reasoning=reasoning,
        tool_calls=(LLMToolCall(
            call_id="tool-call-1",
            name=name,
            arguments=deepcopy(arguments),
            raw_arguments="{}",
        ),),
        trace={
            "call_id": "provider-call-1",
            "request_hash": "request-hash-1",
            "response_hash": "response-hash-1",
        },
    )


class FakeToolRouter:
    def __init__(self, responses: Sequence[LLMToolResponse | BaseException]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete_tools(
        self,
        messages: list[dict[str, Any]],
        config: ModelConfig,
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMToolResponse:
        self.calls.append({
            "messages": deepcopy(messages),
            "config": config,
            "tools": deepcopy(tools),
            "kwargs": deepcopy(kwargs),
        })
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _actor(
    router: FakeToolRouter,
    *,
    trace: list[dict[str, Any]] | None = None,
    memory_entry_limit: int = 12,
) -> CoreToolActor:
    return CoreToolActor(
        actor_id="council:1",
        model_config=ModelConfig(provider="openai", model="test-model"),
        router=router,
        budget_scope="core-run-1",
        trace_sink=None if trace is None else trace.append,
        memory_entry_limit=memory_entry_limit,
    )


@pytest.mark.asyncio
async def test_core_tool_actor_uses_exact_required_terminal_function_contract():
    traces: list[dict[str, Any]] = []
    router = FakeToolRouter([_response("submit_action_1", {"approve": True})])
    request = _request()

    envelope = await _actor(router, trace=traces).decide(request)

    assert isinstance(envelope.choice, ActionChoice)
    assert envelope.choice.action == "submit_vote"
    assert envelope.choice.arguments == {"approve": True}
    assert envelope.model_call_id == "provider-call-1"
    assert envelope.prompt_hash == "request-hash-1"
    assert envelope.response_hash == "response-hash-1"
    assert validate_decision_envelope(envelope, request).valid

    call = router.calls[0]
    assert call["kwargs"]["tool_choice"] == "required"
    assert call["kwargs"]["parallel_tool_calls"] is False
    assert "max_tokens" not in call["kwargs"]
    assert call["kwargs"]["budget_scope"] == "core-run-1"
    assert "alliances" in call["kwargs"]["system"]
    assert call["tools"] == [{
        "type": "function",
        "function": {
            "name": "submit_action_1",
            "description": 'Submit the terminal action exactly named "submit_vote". This call ends the current request.',
            "parameters": request.legal_actions[0].input_schema,
        },
    }]
    assert [row["type"] for row in traces] == [
        "agent_turn_started",
        "model_generation",
        "tool_call_requested",
        "tool_result",
        "agent_action_submitted",
    ]
    generation = traces[1]
    assert generation["visibility"] == "admin"
    assert generation["reasoning"] == "I will keep this private."
    assert traces[2]["call_id"] == "tool-call-1"
    assert traces[3]["terminal"] is True
    assert traces[4]["action"] == {
        "kind": "action",
        "action": "submit_vote",
        "arguments": {"approve": True},
    }
    assert envelope.metadata["llm_call"]["actor_response_attempt_count"] == 1
    assert envelope.metadata["llm_call"]["actor_response_attempts"][0]["status"] == "accepted"
    assert project_payload_for_audience(
        generation, kind="decision", audience="public"
    ) is None


@pytest.mark.asyncio
async def test_core_tool_actor_supports_only_advertised_explicit_skip():
    router = FakeToolRouter([_response("submit_skip", {"reason": "No safe vote."})])

    envelope = await _actor(router).decide(_request(allow_skip=True))

    assert isinstance(envelope.choice, SkipChoice)
    assert envelope.choice.reason == "No safe vote."
    assert router.calls[0]["tools"][-1]["function"]["name"] == "submit_skip"


@pytest.mark.asyncio
async def test_core_tool_actor_retries_unadvertised_or_ambiguous_tool_responses():
    traces: list[dict[str, Any]] = []
    router = FakeToolRouter([
        _response("unknown_action", {}),
        _response("submit_action_1", {"approve": False}),
    ])

    envelope = await _actor(router, trace=traces).decide(_request())

    assert isinstance(envelope.choice, ActionChoice)
    assert envelope.choice.arguments == {"approve": False}
    assert len(router.calls) == 2
    failures = [row for row in traces if row["type"] == "model_generation_failed"]
    assert len(failures) == 1
    assert failures[0]["will_retry"] is True


@pytest.mark.asyncio
async def test_core_tool_actor_does_not_rewrite_invalid_action_arguments():
    router = FakeToolRouter([_response("submit_action_1", {"approve": "yes"})])
    request = _request()

    envelope = await _actor(router).decide(request)

    assert isinstance(envelope.choice, ActionChoice)
    assert envelope.choice.arguments == {"approve": "yes"}
    result = validate_decision_envelope(envelope, request)
    assert not result.valid
    assert {issue.code for issue in result.issues} == {"action_arguments_invalid"}


@pytest.mark.asyncio
async def test_core_tool_actor_fails_after_bounded_malformed_response_attempts():
    router = FakeToolRouter([
        _response("unknown_action", {}),
        _response("unknown_action", {}),
        _response("unknown_action", {}),
    ])

    with pytest.raises(AgentDecisionError) as exc_info:
        await _actor(router).decide(_request())

    assert getattr(exc_info.value, "error_type") == "CoreToolActorResponseFailure"
    assert getattr(exc_info.value, "response_attempt_count") == 3
    assert len(router.calls) == 3


@pytest.mark.asyncio
async def test_core_tool_actor_identity_is_fail_closed_and_instances_cannot_be_shared():
    router = FakeToolRouter([_response("submit_action_1", {"approve": True})])
    actor = _actor(router)
    wrong_request = _request().model_copy(update={"actor_id": "council:2"})

    with pytest.raises(AgentDecisionError) as exc_info:
        await actor.decide(wrong_request)

    assert getattr(exc_info.value, "error_type") == "CoreToolActorIdentityMismatch"
    assert router.calls == []

    registry = AgentRegistry(lambda _actor_id: actor)
    assert registry.resolve("council:1") is actor
    with pytest.raises(AgentBindingError, match="identity does not match"):
        registry.resolve("council:2")


@pytest.mark.asyncio
async def test_core_tool_actor_retries_router_response_shape_failures_only():
    router = FakeToolRouter([
        LLMResponseError("tool stream was malformed"),
        _response("submit_action_1", {"approve": True}),
    ])

    envelope = await _actor(router).decide(_request())

    assert isinstance(envelope.choice, ActionChoice)
    assert len(router.calls) == 2


@pytest.mark.asyncio
async def test_core_tool_actor_keeps_bounded_private_episodic_memory_without_reasoning():
    router = FakeToolRouter([
        _response("submit_action_1", {"approve": True}, reasoning="private model chain"),
        _response("submit_action_1", {"approve": False}),
    ])
    actor = _actor(router)
    first = _request().model_copy(update={
        "request_id": "core-memory-first",
        "observation": {
            "private_identity": {"team": "cipher", "plan": "cipher-plan-alpha"},
            "public_state": {"mission": 1},
        },
    })
    second = _request().model_copy(update={
        "request_id": "core-memory-second",
        "observation": {
            "private_identity": {"team": "cipher"},
            "public_state": {"mission": 2},
        },
    })

    await actor.decide(first)
    await actor.decide(second)

    initial_prompt = json.loads(router.calls[0]["messages"][0]["content"])
    second_prompt = json.loads(router.calls[1]["messages"][0]["content"])
    assert initial_prompt["private_memory"] == {
        "schema_version": "agent-harness.core-actor-memory.v1",
        "revision": 0,
        "own_recent_turns": [],
        "history_truncated": False,
    }
    memory = second_prompt["private_memory"]
    assert memory["revision"] == 1
    assert memory["history_truncated"] is False
    assert memory["own_recent_turns"] == [{
        "request_id": "core-memory-first",
        "labels": {"environment": "council.cipher", "stage": "vote"},
        "observation": {
            "private_identity": {"team": "cipher", "plan": "cipher-plan-alpha"},
            "public_state": {"mission": 1},
        },
        "choice": {
            "kind": "action",
            "action": "submit_vote",
            "arguments": {"approve": True},
        },
    }]
    assert "private model chain" not in json.dumps(memory)


@pytest.mark.asyncio
async def test_core_tool_actor_private_memory_never_crosses_actor_instances():
    router = FakeToolRouter([
        _response("submit_action_1", {"approve": True}),
        _response("submit_action_1", {"approve": False}),
        _response("submit_action_1", {"approve": True}),
    ])
    first_actor = _actor(router)
    second_actor = CoreToolActor(
        actor_id="council:2",
        model_config=ModelConfig(provider="openai", model="test-model"),
        router=router,
        budget_scope="core-run-1",
    )
    first_request = _request().model_copy(update={
        "request_id": "core-isolation-first",
        "observation": {"private_identity": {"plan": "cipher-plan-only-one"}},
    })
    second_request = _request().model_copy(update={
        "request_id": "core-isolation-second",
        "actor_id": "council:2",
        "observation": {"private_identity": {"team": "council"}},
    })
    first_followup = _request().model_copy(update={
        "request_id": "core-isolation-followup",
        "observation": {"private_identity": {"plan": "cipher-plan-only-one"}},
    })

    await first_actor.decide(first_request)
    await second_actor.decide(second_request)
    await first_actor.decide(first_followup)

    second_prompt = json.loads(router.calls[1]["messages"][0]["content"])
    followup_prompt = json.loads(router.calls[2]["messages"][0]["content"])
    assert second_prompt["private_memory"]["revision"] == 0
    assert "cipher-plan-only-one" not in json.dumps(second_prompt)
    assert followup_prompt["private_memory"]["revision"] == 1
    assert "cipher-plan-only-one" in json.dumps(followup_prompt["private_memory"])


@pytest.mark.asyncio
async def test_core_tool_actor_only_records_one_memory_turn_after_response_retry():
    router = FakeToolRouter([
        _response("unadvertised", {}),
        _response("submit_action_1", {"approve": True}),
        _response("submit_action_1", {"approve": False}),
    ])
    actor = _actor(router, memory_entry_limit=1)
    first = _request().model_copy(update={"request_id": "core-retry-first"})
    second = _request().model_copy(update={"request_id": "core-retry-second"})

    await actor.decide(first)
    await actor.decide(second)

    retry_prompt = json.loads(router.calls[1]["messages"][0]["content"])
    followup_prompt = json.loads(router.calls[2]["messages"][0]["content"])
    assert retry_prompt["private_memory"]["revision"] == 0
    assert followup_prompt["private_memory"]["revision"] == 1
    assert [
        row["request_id"]
        for row in followup_prompt["private_memory"]["own_recent_turns"]
    ] == ["core-retry-first"]
