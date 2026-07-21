"""LLMRouter 边界测试 —— 不调用真实 LLM。"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic
import httpx
import openai
import pytest

from src.llm.models import ModelConfig
from src.llm.router import (
    LLMCallCleanupError,
    LLMError,
    LLMResponse,
    LLMResponseError,
    LLMRouter,
    LLMToolCall,
    _safe_status_code,
)
from src.agent.actor import AgentActor, AgentDecisionError, DECISION_MAX_ATTEMPTS
from src.agent.information import build_observation
from src.agent.schemas import AgentAction
from src.game.roles import Role
from src.game.state import new_game
from src.harness.agent_protocol import ActionRequest, LegalAction
from src.harness.agents import validate_decision_against_legal_actions


def _agent_request(
    actor: AgentActor,
    state,
    action_kind: str,
    *,
    target_seats: list[int],
    private_context: dict | None = None,
    in_pk: bool = False,
) -> ActionRequest:
    player_id = next(player.id for player in state.players if player.seat == actor.seat)
    observation = build_observation(
        state,
        player_id,
        available_actions=[action_kind],
        vote_targets=target_seats if action_kind == "vote" else None,
        in_pk=in_pk,
    )
    observation.candidate_targets = list(target_seats)
    protocol_action = "night_kill" if action_kind == "hunter_shot" else action_kind
    return ActionRequest(
        request_id=f"test-{action_kind}",
        run_id=state.id,
        seat=actor.seat,
        phase=state.phase.value,
        day=state.day,
        action_kind=action_kind,
        observation=observation.model_dump(),
        legal_actions=[LegalAction(
            action=protocol_action,
            target_seats=target_seats,
            can_skip=True,
        )],
        private_context=private_context or {},
    )


def _private_state_update(*, seat: int = 2) -> dict:
    return {
        "beliefs": [{
            "seat": seat,
            "wolf_probability": 0.5,
            "likely_role": None,
            "confidence": 0.3,
            "evidence": ["test-visible-evidence"],
        }],
        "candidate_plans": ["test-plan-a", "test-plan-b"],
        "selected_plan": "test-plan-a",
        "public_cover_role": None,
        "perceived_image": "test-perceived-image",
        "deception_plan": None,
        "team_plan": None,
    }


def test_model_config_normalizes_provider_without_vendor_gatekeeping():
    assert ModelConfig(provider="openai").provider == "openai"
    assert ModelConfig(provider="openai_responses").provider == "openai_responses"
    assert ModelConfig(provider="anthropic").provider == "anthropic"
    assert ModelConfig(provider="OPENAI_RESPONSES").provider == "openai_responses"
    assert ModelConfig(provider="Vendor_Special").provider == "vendor_special"


@pytest.mark.parametrize("field_name", ["extra_body", "reasoning_effort", "top_k"])
def test_model_config_rejects_non_standard_provider_fields(field_name):
    with pytest.raises(ValueError, match=field_name):
        ModelConfig(provider="openai", **{field_name: {"enabled": True}})


def test_model_config_accepts_standard_reasoning_and_thinking_fields():
    cfg = ModelConfig(
        provider="openai_responses",
        reasoning={"effort": "high", "summary": "auto"},
        thinking={"type": "enabled", "budget_tokens": 2048},
    )

    assert cfg.reasoning == {"effort": "high", "summary": "auto"}
    assert cfg.thinking == {"type": "enabled", "budget_tokens": 2048}


def test_model_config_accepts_standard_json_schema_response_format():
    config = ModelConfig(
        provider="openai",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "agent_decision",
                "schema": {
                    "type": "object",
                    "properties": {"target_seat": {"type": "integer"}},
                    "required": ["target_seat"],
                    "additionalProperties": False,
                },
            },
        },
    )

    assert config.response_format is not None
    assert config.response_format["type"] == "json_schema"
    assert config.response_format["json_schema"]["strict"] is True


def test_model_config_rejects_malformed_json_schema_before_provider_call():
    with pytest.raises(ValueError, match="valid Draft 2020-12 JSON Schema"):
        ModelConfig(
            provider="openai",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_decision",
                    "schema": {"type": "not-a-json-schema-type"},
                },
            },
        )


def test_model_config_merge_allows_explicit_zero_max_tokens_override():
    base = ModelConfig(provider="openai", model="room-model", api_base="https://example.invalid/v1", api_key="key", max_tokens=1024)

    merged = base.merge({"model": "seat-model", "max_tokens": 0})

    assert merged.model == "seat-model"
    assert merged.max_tokens == 0


def test_model_config_merge_detaches_nested_runtime_options():
    base = ModelConfig(
        model="base-model",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "decision",
                "schema": {
                    "type": "object",
                    "properties": {"action": {"type": "string"}},
                },
            },
        },
        reasoning={"effort": "high", "nested": {"enabled": True}},
        thinking={"type": "enabled", "budget_tokens": 1024},
    )

    first = base.merge(None)
    second = base.merge({"model": "seat-model"})

    assert first.response_format is not base.response_format
    assert second.response_format is not base.response_format
    assert first.reasoning is not base.reasoning
    assert second.reasoning is not base.reasoning
    assert first.thinking is not base.thinking
    assert second.thinking is not base.thinking
    assert first.reasoning is not second.reasoning
    first.reasoning["nested"]["enabled"] = False
    assert base.reasoning["nested"]["enabled"] is True
    assert second.reasoning["nested"]["enabled"] is True


def test_model_config_repr_never_contains_api_key():
    secret = "sk-test-repr-secret-123456789"
    config = ModelConfig(provider="openai", model="m", api_key=secret)

    assert secret not in repr(config)
    assert "api_key" not in repr(config)


@pytest.mark.parametrize("provider", ["anthropic", "openai_responses"])
def test_model_config_merge_model_only_inherits_protocol_boundary(provider):
    base = ModelConfig(
        provider=provider,
        model="room-model",
        api_base="https://default.example/v1",
        api_key="room-key",
    )

    merged = base.merge({"model": "seat-model"})

    assert merged.provider == provider
    assert merged.model == "seat-model"
    assert merged.api_base == "https://default.example/v1"
    assert merged.api_key == "room-key"


def test_agent_protocol_uses_one_explicit_decision_entrypoint_with_retry_budget():
    assert hasattr(AgentActor, "decide")
    assert not hasattr(AgentActor, "decide_vote")
    assert not hasattr(AgentActor, "decide_speak")
    assert not hasattr(AgentActor, "decide_night_action")
    assert DECISION_MAX_ATTEMPTS == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("observation_field", "value", "error_type"),
    [
        ("my_seat", 2, "AgentObservationSeatMismatch"),
        ("my_role", Role.WEREWOLF.value, "AgentObservationRoleMismatch"),
    ],
)
async def test_agent_rejects_observation_identity_mismatch_before_model_call(
    observation_field,
    value,
    error_type,
):
    class FailIfCalledRouter:
        async def complete_json(self, *_args, **_kwargs):
            raise AssertionError("identity mismatch must fail before a model call")

    state = new_game(["A", "B", "C", "D", "E", "F"])
    for player in state.players:
        player.role = Role.VILLAGER
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="m", api_key="test"),
        router=FailIfCalledRouter(),  # type: ignore[arg-type]
    )
    request = _agent_request(actor, state, "speak", target_seats=[])
    observation = dict(request.observation)
    observation[observation_field] = value
    request = request.model_copy(update={"observation": observation})

    with pytest.raises(AgentDecisionError) as exc_info:
        await actor.decide(request)

    assert getattr(exc_info.value, "error_type", None) == error_type


@pytest.mark.asyncio
async def test_agent_envelope_reports_lossless_parse_recovery_from_router_metadata():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FakeRouter:
        async def complete_json(self, *_args, **_kwargs):
            return {
                "thought": "private",
                "speech": "public",
                "bid": 2,
                "private_state": _private_state_update(),
                "_parse_recovered": True,
                "_parse_lossy": False,
                "_parse_method": "fenced_json",
                "_llm_call_trace": {
                    "call_id": "call-1",
                    "request_hash": "request-hash",
                    "response_hash": "response-hash",
                    "parse": {
                        "method": "fenced_json",
                        "recovered": True,
                        "lossy": False,
                    },
                },
            }

    state = new_game(["A", "B", "C", "D", "E", "F"])
    for player in state.players:
        player.role = Role.VILLAGER
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=config,
        router=FakeRouter(),  # type: ignore[arg-type]
    )

    envelope = await actor.decide(_agent_request(actor, state, "speak", target_seats=[]))

    assert envelope.parse_status == "recovered"
    assert envelope.decision.speech == "public"
    assert envelope.decision.llm_call_trace["parse"]["method"] == "fenced_json"
    assert "parse_failed" not in envelope.decision.model_dump()


@pytest.mark.asyncio
async def test_router_trace_records_each_transport_attempt_without_credentials():
    secret = "sk-transport-attempt-secret"
    config = ModelConfig(
        provider="openai",
        api_base="https://gateway.example.invalid/v1",
        api_key=secret,
        model="model-a",
    )
    router = LLMRouter(timeout=2, max_retries=2)
    calls = 0

    async def flaky_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.TimeoutError
        return LLMResponse(
            content='{"target_seat": 2}',
            finish_reason="stop",
            usage={"prompt_tokens": 3, "completion_tokens": 4},
        )

    router._call_openai = flaky_call  # type: ignore[method-assign]
    router._backoff_delay = lambda _attempt: 0.0  # type: ignore[method-assign]

    parsed = await router.complete_json(
        [{"role": "user", "content": "choose"}],
        config,
        trace_context={"request_id": "request-1", "seat": 1},
    )

    trace = parsed["_llm_call_trace"]
    assert calls == 2
    assert trace["transport_attempt_count"] == 2
    assert [row["status"] for row in trace["transport_attempts"]] == [
        "failed",
        "succeeded",
    ]
    assert trace["transport_attempts"][0]["will_retry"] is True
    assert trace["transport_attempts"][0]["timeout"] is True
    assert trace["transport_attempts"][1]["will_retry"] is False
    assert trace["context"] == {"request_id": "request-1", "seat": 1}
    assert secret not in json.dumps(trace)
    assert "choose" not in json.dumps(trace)
    assert router.stats.snapshot()["retries"] == 1


@pytest.mark.asyncio
async def test_router_failure_preserves_safe_exhausted_attempt_trace():
    secret = "sk-exhausted-attempt-secret"
    config = ModelConfig(
        provider="openai",
        api_base="https://gateway.example.invalid/v1",
        api_key=secret,
        model="model-a",
    )
    router = LLMRouter(timeout=2, max_retries=2)

    async def timed_out(*_args, **_kwargs):
        raise asyncio.TimeoutError

    router._call_openai = timed_out  # type: ignore[method-assign]
    router._backoff_delay = lambda _attempt: 0.0  # type: ignore[method-assign]

    with pytest.raises(LLMError, match="总超时") as captured:
        await router.complete_json(
            [{"role": "user", "content": "choose"}],
            config,
            trace_context={"request_id": "request-failed"},
        )

    trace = captured.value.llm_call_trace
    assert trace["transport_attempt_count"] == 2
    assert [row["status"] for row in trace["transport_attempts"]] == [
        "failed",
        "failed",
    ]
    assert [row["will_retry"] for row in trace["transport_attempts"]] == [True, False]
    assert trace["response_hash"] is None
    assert trace["context"] == {"request_id": "request-failed"}
    assert secret not in json.dumps(trace)
    assert "choose" not in json.dumps(trace)


@pytest.mark.asyncio
async def test_router_retries_sse_api_error_with_structured_503_then_succeeds():
    config = ModelConfig(
        provider="openai",
        api_base="https://gateway.example.invalid/v1",
        api_key="test-key",
        model="model-a",
    )
    router = LLMRouter(timeout=2, max_retries=2)
    calls = 0

    async def stream_error_then_success(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            request = httpx.Request(
                "POST",
                "https://gateway.example.invalid/v1/chat/completions",
            )
            raise openai.APIError(
                "stream error envelope",
                request,
                body={"type": "server_error", "code": "503", "message": "upstream"},
            )
        return LLMResponse(
            content="",
            finish_reason="tool_calls",
            raw_provider="openai",
            tool_calls=(LLMToolCall(
                call_id="call-probe",
                name="probe",
                arguments={},
                raw_arguments="{}",
            ),),
        )

    router._call_openai_tools = stream_error_then_success  # type: ignore[method-assign]
    router._backoff_delay = lambda _attempt: 0.0  # type: ignore[method-assign]
    try:
        response = await router.complete_tools(
            [{"role": "user", "content": "probe"}],
            config,
            [{
                "type": "function",
                "function": {
                    "name": "probe",
                    "description": "probe",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            }],
            tool_choice="required",
            parallel_tool_calls=False,
        )
    finally:
        await router.aclose()

    assert calls == 2
    trace = response.trace
    assert [row.get("status_code") for row in trace["transport_attempts"]] == [503, None]
    assert [row["will_retry"] for row in trace["transport_attempts"]] == [True, False]
    assert [row["status"] for row in trace["transport_attempts"]] == ["failed", "succeeded"]
    assert response.tool_calls[0].name == "probe"
    assert router.stats.snapshot()["retries"] == 1


@pytest.mark.asyncio
async def test_router_does_not_retry_generic_api_error_with_structured_400():
    config = ModelConfig(
        provider="openai",
        api_base="https://gateway.example.invalid/v1",
        api_key="test-key",
        model="model-a",
    )
    router = LLMRouter(timeout=2, max_retries=3)
    calls = 0

    async def invalid_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        request = httpx.Request(
            "POST",
            "https://gateway.example.invalid/v1/chat/completions",
        )
        raise openai.APIError(
            "stream error envelope",
            request,
            body={"type": "invalid_request_error", "code": "400"},
        )

    router._call_openai = invalid_request  # type: ignore[method-assign]
    try:
        with pytest.raises(LLMError) as captured:
            await router.complete_json([{"role": "user", "content": "probe"}], config)
    finally:
        await router.aclose()

    assert calls == 1
    attempts = captured.value.llm_call_trace["transport_attempts"]
    assert attempts[0]["status_code"] == 400
    assert attempts[0]["retryable"] is False
    assert attempts[0]["will_retry"] is False


@pytest.mark.parametrize(
    ("error_type", "body", "expected_status"),
    [
        (
            openai.APIError,
            {"error": {"type": "server_error", "code": "503"}},
            503,
        ),
        (
            openai.APIError,
            {"error": {"type": "rate_limit_error"}},
            429,
        ),
        (
            anthropic.APIError,
            {"type": "error", "error": {"type": "overloaded_error"}},
            529,
        ),
    ],
)
def test_router_retries_nested_and_semantic_api_error_statuses(
    error_type,
    body,
    expected_status,
):
    request = httpx.Request("POST", "https://gateway.example.invalid/v1/messages")
    error = error_type("stream error envelope", request, body=body)

    assert _safe_status_code(error) == expected_status
    assert LLMRouter._is_retryable(error) is True


@pytest.mark.parametrize(("status", "retryable"), [(400, False), (503, True), (529, True)])
def test_router_classifies_raw_http_status_errors_before_broad_http_errors(
    status,
    retryable,
):
    request = httpx.Request("POST", "https://gateway.example.invalid/v1/messages")
    error = httpx.HTTPStatusError(
        f"HTTP {status}",
        request=request,
        response=httpx.Response(status, request=request),
    )
    wrapped = LLMError("wrapped transport error")
    wrapped.__cause__ = error

    assert _safe_status_code(error) == status
    assert LLMRouter._is_retryable(error) is retryable
    assert LLMRouter._is_retryable(wrapped) is retryable


@pytest.mark.asyncio
async def test_router_attempt_timeout_reclaims_delayed_cancellation_task():
    config = ModelConfig(
        provider="openai",
        api_base="https://gateway.example.invalid/v1",
        api_key="test-key",
        model="model-a",
    )
    router = LLMRouter(
        timeout=0.01,
        max_retries=1,
        cancellation_grace_seconds=0.1,
    )
    provider_task: asyncio.Task | None = None

    async def delayed_cancel(*_args, **_kwargs):
        nonlocal provider_task
        provider_task = asyncio.current_task()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.02)
            raise

    router._call_openai = delayed_cancel  # type: ignore[method-assign]
    started = time.monotonic()

    with pytest.raises(LLMError, match="总超时"):
        await router.complete_json([{"role": "user", "content": "choose"}], config)

    assert time.monotonic() - started < 0.2
    assert provider_task is not None and provider_task.done()
    assert router.unresolved_task_count == 0
    await router.aclose()


@pytest.mark.asyncio
async def test_router_reports_provider_task_that_ignores_bounded_cancellation():
    config = ModelConfig(
        provider="openai",
        api_base="https://gateway.example.invalid/v1",
        api_key="test-key",
        model="model-a",
    )
    router = LLMRouter(
        timeout=0.01,
        max_retries=2,
        cancellation_grace_seconds=0.02,
    )
    release = asyncio.Event()
    provider_task: asyncio.Task | None = None

    async def ignore_cancellation(*_args, **_kwargs):
        nonlocal provider_task
        provider_task = asyncio.current_task()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        raise asyncio.CancelledError

    router._call_openai = ignore_cancellation  # type: ignore[method-assign]
    started = time.monotonic()
    try:
        with pytest.raises(LLMCallCleanupError) as raised:
            await router.complete_json(
                [{"role": "user", "content": "choose"}],
                config,
                trace_context={"request_id": "cleanup-failed"},
            )

        assert time.monotonic() - started < 0.15
        assert raised.value.fatal_cleanup_failure is True
        assert router.unresolved_task_count == 1
        trace = raised.value.llm_call_trace
        assert trace["transport_attempt_count"] == 1
        assert trace["transport_attempts"][0]["error_type"] == "LLMCallCleanupError"
        assert trace["transport_attempts"][0]["will_retry"] is False
        assert trace["context"] == {"request_id": "cleanup-failed"}
        with pytest.raises(LLMCallCleanupError) as quarantined:
            await router.complete_json(
                [{"role": "user", "content": "must not start"}],
                config,
            )
        assert quarantined.value.pending_task_count == 1
    finally:
        release.set()
        await asyncio.sleep(0)

    assert provider_task is not None and provider_task.done()
    assert router.unresolved_task_count == 0

    async def recovered(*_args, **_kwargs):
        return LLMResponse(content='{"ok": true}', finish_reason="stop")

    router._call_openai = recovered  # type: ignore[method-assign]
    assert await router.complete_json(
        [{"role": "user", "content": "after cleanup"}],
        config,
    ) == {"ok": True}
    await router.aclose()

    assert "test-key" not in json.dumps(trace)
    assert "choose" not in json.dumps(trace)


@pytest.mark.asyncio
async def test_router_external_cancellation_reclaims_provider_attempt():
    config = ModelConfig(
        provider="openai",
        api_base="https://gateway.example.invalid/v1",
        api_key="test-key",
        model="model-a",
    )
    router = LLMRouter(
        timeout=10,
        max_retries=1,
        cancellation_grace_seconds=0.1,
    )
    started = asyncio.Event()
    provider_task: asyncio.Task | None = None

    async def delayed_cancel(*_args, **_kwargs):
        nonlocal provider_task
        provider_task = asyncio.current_task()
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            raise

    router._call_openai = delayed_cancel  # type: ignore[method-assign]
    execution = asyncio.create_task(
        router.complete_json([{"role": "user", "content": "choose"}], config)
    )
    await started.wait()

    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution

    assert provider_task is not None and provider_task.done()
    assert router.unresolved_task_count == 0
    await router.aclose()


@pytest.mark.asyncio
async def test_router_close_rejects_new_calls_and_reclaims_active_provider_task():
    config = ModelConfig(
        provider="openai",
        api_base="https://gateway.example.invalid/v1",
        api_key="test-key",
        model="model-a",
    )
    router = LLMRouter(
        timeout=10,
        max_retries=1,
        cancellation_grace_seconds=0.1,
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class SharedClient:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    shared_client = SharedClient()
    router._openai_clients["chat"] = shared_client  # type: ignore[assignment]
    router._anthropic_clients["messages"] = shared_client  # type: ignore[assignment]

    async def blocked_provider(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    router._call_openai = blocked_provider  # type: ignore[method-assign]
    execution = asyncio.create_task(
        router.complete_json([{"role": "user", "content": "choose"}], config)
    )
    await started.wait()
    assert router.active_task_count == 1

    await router.aclose()

    with pytest.raises(asyncio.CancelledError):
        await execution
    assert cancelled.is_set()
    assert router.closed is True
    assert router.closing is False
    assert router.active_task_count == 0
    assert router.unresolved_task_count == 0
    assert shared_client.close_calls == 1

    with pytest.raises(LLMError, match="closing or closed"):
        await router.complete_json([{"role": "user", "content": "choose"}], config)
    await router.aclose()  # Idempotent and does not close a shared client twice.
    assert shared_client.close_calls == 1


@pytest.mark.asyncio
async def test_actor_trace_links_rejected_and_accepted_response_calls():
    class ResponseRetryRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, *_args, **_kwargs):
            self.calls += 1
            return {
                "speech": "public",
                "bid": 9 if self.calls == 1 else 2,
                "_llm_call_trace": {
                    "call_id": f"call-{self.calls}",
                    "request_hash": "request-hash",
                    "response_hash": f"response-hash-{self.calls}",
                    "transport_attempt_count": 1,
                    "transport_attempts": [{
                        "attempt": 1,
                        "status": "succeeded",
                        "retryable": False,
                        "will_retry": False,
                    }],
                },
            }

    router = ResponseRetryRouter()
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="m", api_key="test"),
        router=router,  # type: ignore[arg-type]
    )

    raw = await actor._call_with_retry(
        [],
        "",
        max_attempts=2,
        required_fields=["speech?", "bid"],
    )

    trace = raw["_llm_call_trace"]
    assert router.calls == 2
    assert trace["actor_response_attempt_count"] == 2
    assert [row["status"] for row in trace["actor_response_attempts"]] == [
        "response_rejected",
        "accepted",
    ]
    assert [row["llm_call"]["call_id"] for row in trace["actor_response_attempts"]] == [
        "call-1",
        "call-2",
    ]


@pytest.mark.asyncio
async def test_actor_validation_retry_feedback_is_field_only_and_secret_free():
    private_value = "seer-check-seat-6-must-stay-private"
    credential = "sk-retry-feedback-secret-123456789"
    invalid_private_state = _private_state_update()
    invalid_private_state["candidate_plans"] = [private_value]
    invalid_private_state[private_value] = credential

    class ValidationRetryRouter:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, str]]] = []

        async def complete_json(self, messages, *_args, **_kwargs):
            self.calls.append([dict(message) for message in messages])
            if len(self.calls) == 1:
                return {
                    "private_state": invalid_private_state,
                    "_llm_call_trace": {"call_id": "validation-call-1"},
                }
            return {
                "private_state": _private_state_update(),
                "_llm_call_trace": {"call_id": "validation-call-2"},
            }

    router = ValidationRetryRouter()
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="m", api_key=credential),
        router=router,  # type: ignore[arg-type]
    )

    async def no_retry_delay(_attempt: int) -> None:
        return None

    actor._sleep_before_retry = no_retry_delay  # type: ignore[method-assign]
    original_messages = [{"role": "user", "content": "return a decision"}]
    result = await actor._call_with_retry(
        original_messages,
        "system",
        max_attempts=2,
        required_fields=["private_state"],
    )

    assert result["private_state"] == _private_state_update()
    assert len(router.calls) == 2
    assert router.calls[0] == original_messages
    assert len(router.calls[1]) == 2
    feedback = router.calls[1][-1]["content"]
    assert "private_state.candidate_plans" in feedback
    assert "too_short" in feedback
    assert "<field>: extra_forbidden" in feedback
    assert private_value not in feedback
    assert credential not in feedback
    assert original_messages == [{"role": "user", "content": "return a decision"}]
    response_attempts = result["_llm_call_trace"]["actor_response_attempts"]
    assert response_attempts[0]["validation_issues"] == [
        {"path": "private_state.candidate_plans", "code": "too_short"},
        {"path": "private_state.<field>", "code": "extra_forbidden"},
    ]
    assert "validation_issues" not in response_attempts[1]
    assert private_value not in json.dumps(response_attempts, ensure_ascii=False)
    assert credential not in json.dumps(response_attempts, ensure_ascii=False)


@pytest.mark.asyncio
async def test_agent_vote_with_unresolvable_target_does_not_pick_first_candidate():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FakeRouter:
        async def complete_json(self, *_args, **_kwargs):
            return {
                "thought": "没有形成合法投票目标",
                "target_seat": 999,
                "private_state": _private_state_update(),
            }

    state = new_game(["A", "B", "C", "D", "E", "F"])
    for player in state.players:
        player.role = Role.VILLAGER
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=config,
        router=FakeRouter(),  # type: ignore[arg-type]
    )

    decision = (
        await actor.decide(_agent_request(actor, state, "vote", target_seats=[2, 3, 4, 5, 6]))
    ).decision

    assert decision.action == AgentAction.VOTE
    assert decision.target_seat == 999
    assert decision.skip_reason is None


@pytest.mark.asyncio
async def test_agent_pk_vote_without_valid_target_does_not_pick_first_candidate():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FakeRouter:
        async def complete_json(self, *_args, **_kwargs):
            return {
                "thought": "我想投 PK 外的人",
                "target_seat": 4,
                "private_state": _private_state_update(),
            }

    state = new_game(["A", "B", "C", "D", "E", "F"])
    for player in state.players:
        player.role = Role.VILLAGER
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=config,
        router=FakeRouter(),  # type: ignore[arg-type]
    )

    decision = (
        await actor.decide(_agent_request(
            actor,
            state,
            "vote",
            target_seats=[2, 3],
            in_pk=True,
        ))
    ).decision

    assert decision.action == AgentAction.VOTE
    assert decision.target_seat == 4
    assert decision.skip_reason is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "expected_action", "expected_issue"),
    [
        (
            {"thought": "contradictory", "speech": "must remain visible in envelope", "bid": 0},
            AgentAction.SKIP,
            "skip_payload_not_empty",
        ),
        (
            {"thought": "contradictory", "speech": None, "bid": 2},
            AgentAction.SPEAK,
            "speech_required",
        ),
    ],
)
async def test_agent_preserves_contradictory_speech_intent_for_protocol_rejection(
    raw,
    expected_action,
    expected_issue,
):
    class FakeRouter:
        async def complete_json(self, *_args, **_kwargs):
            return {**raw, "private_state": _private_state_update()}

    state = new_game(["A", "B", "C", "D", "E", "F"])
    for player in state.players:
        player.role = Role.VILLAGER
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=ModelConfig(provider="openai", model="m", api_key="test"),
        router=FakeRouter(),  # type: ignore[arg-type]
    )
    request = _agent_request(actor, state, "speak", target_seats=[])

    envelope = await actor.decide(request)
    validation = validate_decision_against_legal_actions(envelope, request)

    assert envelope.decision.action == expected_action
    assert not validation.valid
    assert expected_issue in {issue.code for issue in validation.issues}
    if raw["speech"]:
        assert envelope.decision.speech == raw["speech"]


@pytest.mark.asyncio
async def test_hunter_requested_action_uses_real_llm_target_without_role_fallback():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FakeRouter:
        async def complete_json(self, *_args, **_kwargs):
            return {
                "thought": "猎人决定带走2号",
                "target_seat": 2,
                "private_state": _private_state_update(),
            }

    state = new_game(["A", "B", "C", "D", "E", "F"])
    roles = [Role.HUNTER, Role.VILLAGER, Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.VILLAGER]
    for player, role in zip(state.players, roles):
        player.role = role
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.HUNTER,
        model_config=config,
        router=FakeRouter(),  # type: ignore[arg-type]
    )

    decision = (
        await actor.decide(_agent_request(
            actor,
            state,
            "hunter_shot",
            target_seats=[2, 3, 4, 5, 6],
        ))
    ).decision

    assert decision.action == AgentAction.NIGHT_KILL
    assert decision.target_seat == 2


@pytest.mark.asyncio
async def test_witch_save_requested_action_rejects_poison_shape_as_parse_failure():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FakeRouter:
        async def complete_json(self, *_args, **_kwargs):
            return {
                "thought": "此阶段却想用毒",
                "use_poison": True,
                "poison_target": 2,
                "private_state": _private_state_update(),
            }

    state = new_game(["A", "B", "C", "D", "E", "F"])
    roles = [Role.WITCH, Role.VILLAGER, Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.VILLAGER]
    for player, role in zip(state.players, roles):
        player.role = role
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.WITCH,
        model_config=config,
        router=FakeRouter(),  # type: ignore[arg-type]
    )

    with pytest.raises(AgentDecisionError, match="缺少必需字段组"):
        await actor.decide(_agent_request(
            actor,
            state,
            "save",
            target_seats=[2],
            private_context={"killed_seat": 2},
        ))


@pytest.mark.asyncio
async def test_doctor_save_uses_protection_prompt_and_exact_target_shape():
    captured: dict[str, str] = {}

    class FakeRouter:
        async def complete_json(self, messages, *_args, **_kwargs):
            captured["messages"] = json.dumps(messages, ensure_ascii=False)
            return {
                "thought": "保护自己",
                "target_seat": 1,
                "private_state": _private_state_update(),
            }

    state = new_game(["A", "B", "C", "D", "E", "F"])
    roles = [Role.DOCTOR, Role.VILLAGER, Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.VILLAGER]
    for player, role in zip(state.players, roles):
        player.role = role
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.DOCTOR,
        model_config=ModelConfig(provider="openai", model="m", api_key="test"),
        router=FakeRouter(),  # type: ignore[arg-type]
    )

    envelope = await actor.decide(_agent_request(
        actor,
        state,
        "save",
        target_seats=[1, 2, 3, 4, 5, 6],
    ))

    assert envelope.decision.action == AgentAction.SAVE
    assert envelope.decision.target_seat == 1
    assert "本夜要保护的存活座位" in captured["messages"]
    assert "是否使用解药" not in captured["messages"]
    actor.observe_event(
        1,
        "night",
        "doctor_protect_target",
        "第1夜你选择保护1号",
        target_seat=1,
    )
    assert actor._role_extras()["doctor_state"] == "第1夜你选择保护1号"


class _KeepaliveSSEHandler(BaseHTTPRequestHandler):
    """持续发送 SSE keepalive,模拟有字节但永不完成的网关。"""

    def log_message(self, *_args) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        while True:
            try:
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
            except OSError:
                return
            time.sleep(0.05)


class _RecordingSSEHandler(BaseHTTPRequestHandler):
    """记录请求并返回预设 SSE 事件。"""

    events: list[object] = []
    requests: list[dict[str, object]] = []

    def log_message(self, *_args) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8") if length else ""
        body = json.loads(raw_body) if raw_body else {}
        type(self).requests.append({
            "path": self.path,
            "headers": dict(self.headers),
            "body": body,
        })
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in type(self).events:
            if event == "[DONE]":
                data = "[DONE]"
            else:
                data = json.dumps(event, ensure_ascii=False)
                if isinstance(event, dict) and event.get("type"):
                    self.wfile.write(f"event: {event['type']}\n".encode("utf-8"))
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()


def _start_recording_sse_server(events: list[object]) -> tuple[ThreadingHTTPServer, type[_RecordingSSEHandler]]:
    class Handler(_RecordingSSEHandler):
        pass

    Handler.events = events
    Handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, Handler


@pytest.mark.asyncio
async def test_openai_lazy_resource_resolution_does_not_starve_attempt_timeout():
    """SDK lazy imports must not monopolize the event loop past router.timeout."""
    entered = threading.Event()
    release = threading.Event()

    class Completions:
        async def create(self, **_kwargs):
            raise AssertionError("timed-out lazy resource must not start a request")

    class Chat:
        completions = Completions()

    class SlowLazyClient:
        @property
        def chat(self):
            entered.set()
            release.wait(timeout=10.0)
            return Chat()

    client = SlowLazyClient()
    router = LLMRouter(
        timeout=0.5,
        chunk_timeout=2.0,
        max_retries=1,
        cancellation_grace_seconds=0.1,
    )
    config = ModelConfig(
        provider="openai",
        api_base="https://gateway.example.invalid/v1",
        api_key="test-key",
        model="test-model",
    )
    router._get_openai_client = lambda *_args, **_kwargs: client  # type: ignore[method-assign]

    await asyncio.to_thread(lambda: None)
    started = time.monotonic()
    try:
        with pytest.raises(LLMError, match="总超时"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
        elapsed = time.monotonic() - started
    finally:
        release.set()
        await router.aclose()

    assert entered.is_set()
    assert elapsed < 2.0
    assert router.stats.snapshot()["failures"] == 1


@pytest.mark.asyncio
async def test_stream_call_has_wall_clock_attempt_timeout():
    """持续 keepalive 不能绕过 router.timeout 的单次调用总时限。"""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _KeepaliveSSEHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    router = LLMRouter(timeout=0.25, chunk_timeout=2.0, max_retries=1)
    config = ModelConfig(
        provider="openai",
        api_base=f"http://127.0.0.1:{server.server_address[1]}",
        api_key="test-key",
        model="test-model",
    )

    started = time.monotonic()
    elapsed = 0.0
    try:
        with pytest.raises(LLMError, match="总超时"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
        elapsed = time.monotonic() - started
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert elapsed < 3.0
    assert router.stats.snapshot()["failures"] == 1


@pytest.mark.asyncio
async def test_openai_stream_closes_immediately_on_malformed_completion():
    """A non-cancellation parse failure must not hold the provider stream open."""
    class BrokenStream:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self):
            async def events():
                yield type("Chunk", (), {"choices": []})()

            return events()

        async def close(self) -> None:
            self.closed = True

    stream = BrokenStream()

    class Completions:
        async def create(self, **_kwargs):
            return stream

    router = LLMRouter(timeout=2, max_retries=1, cleanup_timeout_seconds=0.2)
    router._get_openai_client = lambda *_args, **_kwargs: object()  # type: ignore[method-assign]

    async def resource(*_args, **_kwargs):
        return Completions()

    router._get_openai_resource = resource  # type: ignore[method-assign]
    config = ModelConfig(
        provider="openai",
        api_base="https://gateway.example.invalid/v1",
        api_key="test-key",
        model="test-model",
    )

    try:
        with pytest.raises(LLMError, match="without finish_reason"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
        assert stream.closed is True
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_structured_call_timeout_is_not_bypassed_by_keepalive():
    """Agent JSON 调用不能被 SSE keepalive 拖过总时限。"""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _KeepaliveSSEHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    router = LLMRouter(timeout=0.25, chunk_timeout=2.0, max_retries=1)
    config = ModelConfig(
        provider="openai",
        api_base=f"http://127.0.0.1:{server.server_address[1]}",
        api_key="test-key",
        model="test-model",
    )

    started = time.monotonic()
    try:
        with pytest.raises(LLMError, match="总超时"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
        elapsed = time.monotonic() - started
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert elapsed < 3.0
    assert router.stats.snapshot()["failures"] == 1


@pytest.mark.asyncio
async def test_complete_routes_to_standard_protocol_adapter():
    router = LLMRouter(max_retries=1)
    calls: list[str] = []

    async def fake_openai(*_args, **_kwargs):
        calls.append("openai")
        return LLMResponse(content="chat", finish_reason="stop", raw_provider="openai")

    async def fake_responses(*_args, **_kwargs):
        calls.append("openai_responses")
        return LLMResponse(content="responses", finish_reason="completed", raw_provider="openai_responses")

    async def fake_anthropic(*_args, **_kwargs):
        calls.append("anthropic")
        return LLMResponse(content="anthropic", finish_reason="end_turn", raw_provider="anthropic")

    router._call_openai = fake_openai  # type: ignore[method-assign]
    router._call_openai_responses = fake_responses  # type: ignore[method-assign]
    router._call_anthropic = fake_anthropic  # type: ignore[method-assign]

    for provider in ("openai", "openai_responses", "anthropic"):
        resp = await router._complete(
            [{"role": "user", "content": "hi"}],
            ModelConfig(provider=provider, api_base="http://example.invalid", api_key="x", model="m"),
        )
        assert resp.raw_provider == provider

    assert calls == ["openai", "openai_responses", "anthropic"]


@pytest.mark.asyncio
async def test_complete_rejects_unknown_standard_protocol_without_fallback():
    router = LLMRouter(max_retries=1)
    with pytest.raises(LLMError, match="未知标准协议"):
        await router.complete_json(
            [{"role": "user", "content": "hi"}],
            ModelConfig(provider="vendor_special", api_base="http://example.invalid", api_key="x", model="m"),
        )


@pytest.mark.asyncio
async def test_openai_chat_completions_wire_contract_and_sse_parse():
    events = [
        {"choices": [{"delta": {"content": '{"ok"'}}]},
        {
            "choices": [{"delta": {"content": ": true}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        },
        "[DONE]",
    ]
    server, handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
        max_tokens=0,
        use_json_format=True,
        reasoning={"effort": "high", "summary": "auto"},
        thinking={"type": "enabled", "budget_tokens": 2048},
    )

    try:
        resp = await router._complete(
            [{"role": "user", "content": "hi"}],
            config,
            system="system prompt",
            schema_hint='{"ok": true}',
            json_mode=True,
        )
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert resp.content == '{"ok": true}'
    assert resp.usage == {"prompt_tokens": 3, "completion_tokens": 4}
    assert resp.raw_provider == "openai"
    req = handler.requests[0]
    body = req["body"]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer test-key"
    assert body["model"] == "test-model"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["response_format"] == {"type": "json_object"}
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body
    assert body["messages"][0] == {"role": "system", "content": "system prompt"}
    assert body["messages"][1]["role"] == "system"
    assert body["messages"][2] == {"role": "user", "content": "hi"}
    assert "input" not in body
    assert "max_output_tokens" not in body
    assert "reasoning" not in body
    assert "thinking" not in body


@pytest.mark.asyncio
async def test_openai_chat_transport_never_sends_max_token_limit():
    server, handler = _start_recording_sse_server([
        {"choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}]},
        "[DONE]",
    ])
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
        max_tokens=64,
        use_json_format=True,
    )

    try:
        response = await router._complete([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert response.content == "hello"
    body = handler.requests[0]["body"]
    assert "max_completion_tokens" not in body
    assert "max_tokens" not in body
    assert "response_format" not in body
    assert "reasoning" not in body
    assert "thinking" not in body


@pytest.mark.asyncio
async def test_openai_chat_json_mode_adds_json_instruction_without_schema_hint():
    events = [
        {"choices": [{"delta": {"content": '{"ok": true}'}, "finish_reason": "stop"}]},
        "[DONE]",
    ]
    server, handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
        use_json_format=True,
    )

    try:
        parsed = await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert parsed == {"ok": True}
    body = handler.requests[0]["body"]
    assert body["response_format"] == {"type": "json_object"}
    assert any("json" in message["content"].lower() for message in body["messages"])


@pytest.mark.asyncio
async def test_openai_chat_json_schema_response_format_wire_payload():
    events = [
        {"choices": [{"delta": {"content": '{"target_seat": 2}'}, "finish_reason": "stop"}]},
        "[DONE]",
    ]
    server, handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"target_seat": {"type": "integer"}},
                "required": ["target_seat"],
                "additionalProperties": False,
            },
        },
    }
    config = ModelConfig(
        provider="openai",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
        use_json_format=False,
        response_format=response_format,
    )

    try:
        parsed = await router.complete_json([{"role": "user", "content": "choose"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert parsed == {"target_seat": 2}
    body = handler.requests[0]["body"]
    assert body["response_format"] == response_format
    assert any("json" in message["content"].lower() for message in body["messages"])


@pytest.mark.asyncio
async def test_openai_responses_wire_contract_and_sse_parse():
    events = [
        {"type": "response.output_text.delta", "delta": '{"ok"'},
        {"type": "response.output_text.delta", "delta": ": true}"},
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 5, "output_tokens": 6},
            },
        },
        "[DONE]",
    ]
    server, handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai_responses",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
        max_tokens=64,
        use_json_format=True,
        reasoning={"effort": "high", "summary": "auto"},
        thinking={"type": "enabled", "budget_tokens": 2048},
    )

    try:
        parsed = await router.complete_json(
            [
                {"role": "system", "content": "message system"},
                {"role": "user", "content": "hi"},
            ],
            config,
            system="top system",
            schema_hint='{"ok": true}',
        )
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert parsed == {"ok": True}
    req = handler.requests[0]
    body = req["body"]
    assert req["path"] == "/v1/responses"
    assert req["headers"]["Authorization"] == "Bearer test-key"
    assert body["model"] == "test-model"
    assert body["stream"] is True
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["input"][-1] == {"role": "user", "content": "hi"}
    assert any("json" in item["content"].lower() for item in body["input"])
    assert "max_output_tokens" not in body
    assert body["text"] == {"format": {"type": "json_object"}}
    assert "top system" in body["instructions"]
    assert "message system" in body["instructions"]
    assert '{"ok": true}' in body["instructions"]
    assert "messages" not in body
    assert "response_format" not in body
    assert "max_tokens" not in body
    assert "thinking" not in body


@pytest.mark.asyncio
async def test_openai_responses_completed_payload_reasoning_fills_thought():
    events = [
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"text": "Responses 最终推理。"}],
                    },
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": '{"target_seat": 4}'},
                        ],
                    },
                ],
                "usage": {"input_tokens": 5, "output_tokens": 6},
            },
        },
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai_responses",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
        max_tokens=0,
    )

    try:
        parsed = await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert parsed == {"target_seat": 4, "thought": "Responses 最终推理。"}


@pytest.mark.asyncio
async def test_openai_responses_never_sends_max_output_tokens():
    events = [
        {"type": "response.output_text.delta", "delta": '{"ok": true}'},
        {"type": "response.completed", "response": {"status": "completed"}},
        "[DONE]",
    ]
    server, handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai_responses",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
        max_tokens=64,
    )

    try:
        parsed = await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert parsed == {"ok": True}
    body = handler.requests[0]["body"]
    assert "max_output_tokens" not in body
    assert "max_tokens" not in body


@pytest.mark.asyncio
async def test_openai_responses_reasoning_text_delta_fills_thought_without_content_pollution():
    events = [
        {"type": "response.reasoning_summary_text.delta", "delta": "Responses 流式推理。"},
        {"type": "response.output_text.delta", "delta": '{"target_seat": 5}'},
        {"type": "response.completed", "response": {"status": "completed"}},
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai_responses",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
        max_tokens=0,
    )

    try:
        resp = await router._complete([{"role": "user", "content": "hi"}], config)
        parsed = await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert resp.content == '{"target_seat": 5}'
    assert resp.reasoning == "Responses 流式推理。"
    assert parsed == {"target_seat": 5, "thought": "Responses 流式推理。"}


@pytest.mark.asyncio
async def test_openai_responses_failed_event_retries_structured_503():
    events = [
        {
            "type": "response.failed",
            "response": {
                "status": "failed",
                "error": {
                    "type": "server_error",
                    "code": "503",
                    "message": "upstream failed",
                },
            },
        },
        "[DONE]",
    ]
    server, handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=2)
    router._backoff_delay = lambda _attempt: 0.0  # type: ignore[method-assign]
    config = ModelConfig(
        provider="openai_responses",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
    )

    try:
        with pytest.raises(LLMError, match="responses stream failed") as captured:
            await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    attempts = captured.value.llm_call_trace["transport_attempts"]
    assert len(handler.requests) == 2
    assert [row["status_code"] for row in attempts] == [503, 503]
    assert [row["will_retry"] for row in attempts] == [True, False]


@pytest.mark.asyncio
async def test_openai_responses_error_event_does_not_retry_structured_400():
    server, handler = _start_recording_sse_server([
        {
            "type": "error",
            "code": "400",
            "message": "invalid request",
            "param": None,
        },
    ])
    router = LLMRouter(timeout=5, max_retries=3)
    router._backoff_delay = lambda _attempt: 0.0  # type: ignore[method-assign]
    config = ModelConfig(
        provider="openai_responses",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
    )

    try:
        with pytest.raises(LLMError) as captured:
            await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    attempts = captured.value.llm_call_trace["transport_attempts"]
    assert len(handler.requests) == 1
    assert attempts[0]["status_code"] == 400
    assert attempts[0]["retryable"] is False
    assert attempts[0]["will_retry"] is False


@pytest.mark.asyncio
async def test_openai_responses_incomplete_event_rejects_structured_call():
    events = [
        {"type": "response.output_text.delta", "delta": "partial"},
        {
            "type": "response.incomplete",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        },
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai_responses",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
    )

    try:
        with pytest.raises(LLMError, match="未完整结束"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert router.stats.snapshot()["incomplete_responses"] == 1

@pytest.mark.asyncio
async def test_openai_responses_stream_without_completed_is_not_silent_success():
    events = [
        {"type": "response.output_text.delta", "delta": '{"ok": true}'},
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai_responses",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
    )

    try:
        with pytest.raises(LLMError, match="without response.completed"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_openai_responses_missing_completed_rejects_every_structured_call():
    events = [
        {"type": "response.output_text.delta", "delta": "partial"},
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai_responses",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
    )

    try:
        with pytest.raises(LLMError, match="without response.completed"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

@pytest.mark.asyncio
async def test_openai_chat_stream_without_finish_reason_is_not_silent_success():
    events = [
        {"choices": [{"delta": {"content": '{"ok": true}'}, "finish_reason": None}]},
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
    )

    try:
        with pytest.raises(LLMError, match="without finish_reason"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_openai_chat_missing_finish_reason_rejects_structured_call():
    events = [
        {"choices": [{"delta": {"content": "partial text"}, "finish_reason": None}]},
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
    )

    try:
        with pytest.raises(LLMError, match="without finish_reason"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

@pytest.mark.asyncio
async def test_openai_chat_structured_call_rejects_length_finish_reason():
    events = [
        {"choices": [{"delta": {"content": "partial text"}, "finish_reason": "length"}]},
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
    )

    try:
        with pytest.raises(LLMError, match="未完整结束"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

@pytest.mark.asyncio
async def test_anthropic_messages_wire_contract_and_sse_parse():
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 7, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 8},
        },
        {"type": "message_stop"},
    ]
    server, handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="anthropic",
        api_base=f"http://127.0.0.1:{server.server_address[1]}",
        api_key="test-key",
        model="claude-test",
        max_tokens=0,
        use_json_format=True,
        reasoning={"effort": "high", "summary": "auto"},
        thinking={"type": "enabled", "budget_tokens": 2048},
    )

    try:
        resp = await router._complete(
            [
                {"role": "system", "content": "message system"},
                {"role": "user", "content": "hi"},
            ],
            config,
            system="top system",
            schema_hint='{"ok": true}',
        )
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert resp.content == "hello"
    assert resp.finish_reason == "end_turn"
    assert resp.usage == {"prompt_tokens": 7, "completion_tokens": 8}
    assert resp.raw_provider == "anthropic"
    req = handler.requests[0]
    body = req["body"]
    assert req["path"] == "/v1/messages"
    headers = {key.lower(): value for key, value in req["headers"].items()}
    assert headers["x-api-key"] == "test-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-test"
    assert body["stream"] is True
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 8192
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in body
    assert "top system" in body["system"]
    assert "message system" in body["system"]
    assert '{"ok": true}' in body["system"]
    assert "response_format" not in body
    assert "stream_options" not in body
    assert "max_output_tokens" not in body
    assert "reasoning" not in body


@pytest.mark.asyncio
async def test_anthropic_transport_with_thinking_omits_temperature():
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 7, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 8},
        },
        {"type": "message_stop"},
    ]
    server, handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="anthropic",
        api_base=f"http://127.0.0.1:{server.server_address[1]}",
        api_key="test-key",
        model="claude-test",
        thinking={"type": "enabled", "budget_tokens": 2048},
    )

    try:
        response = await router._complete([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert response.content == "hello"
    body = handler.requests[0]["body"]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in body


@pytest.mark.asyncio
async def test_anthropic_thinking_delta_fills_thought_without_content_pollution():
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 7, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Anthropic 推理。"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": '{"vote": 2}'}},
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 8},
        },
        {"type": "message_stop"},
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="anthropic",
        api_base=f"http://127.0.0.1:{server.server_address[1]}",
        api_key="test-key",
        model="claude-test",
        max_tokens=0,
    )

    try:
        resp = await router._complete([{"role": "user", "content": "hi"}], config)
        parsed = await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert resp.content == '{"vote": 2}'
    assert resp.reasoning == "Anthropic 推理。"
    assert parsed == {"vote": 2, "thought": "Anthropic 推理。"}


@pytest.mark.asyncio
async def test_anthropic_stream_overloaded_event_retries_as_529():
    server, handler = _start_recording_sse_server([
        {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
    ])
    router = LLMRouter(timeout=5, max_retries=2)
    router._backoff_delay = lambda _attempt: 0.0  # type: ignore[method-assign]
    config = ModelConfig(
        provider="anthropic",
        api_base=f"http://127.0.0.1:{server.server_address[1]}",
        api_key="test-key",
        model="claude-test",
    )

    try:
        with pytest.raises(LLMError, match="anthropic stream error") as captured:
            await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    attempts = captured.value.llm_call_trace["transport_attempts"]
    assert len(handler.requests) == 2
    assert [row["status_code"] for row in attempts] == [529, 529]
    assert [row["will_retry"] for row in attempts] == [True, False]


@pytest.mark.asyncio
async def test_anthropic_stream_without_message_stop_is_not_silent_success():
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 7, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": '{"ok": true}'}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 8},
        },
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="anthropic",
        api_base=f"http://127.0.0.1:{server.server_address[1]}",
        api_key="test-key",
        model="claude-test",
    )

    try:
        with pytest.raises(LLMError, match="without message_stop or stop_reason"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_anthropic_structured_call_without_message_stop_is_not_silent_success():
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 7, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 8},
        },
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="anthropic",
        api_base=f"http://127.0.0.1:{server.server_address[1]}",
        api_key="test-key",
        model="claude-test",
    )

    try:
        with pytest.raises(LLMError, match="without message_stop or stop_reason"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

@pytest.mark.asyncio
async def test_anthropic_stream_without_stop_reason_is_not_silent_success():
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 7, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": '{"ok": true}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="anthropic",
        api_base=f"http://127.0.0.1:{server.server_address[1]}",
        api_key="test-key",
        model="claude-test",
    )

    try:
        with pytest.raises(LLMError, match="without message_stop or stop_reason"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_anthropic_structured_call_without_stop_reason_is_not_silent_success():
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 7, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="anthropic",
        api_base=f"http://127.0.0.1:{server.server_address[1]}",
        api_key="test-key",
        model="claude-test",
    )

    try:
        with pytest.raises(LLMError, match="without message_stop or stop_reason"):
            await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

def test_parse_json_rejects_lossy_truncated_field_by_default():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")
    content = '{"target_seat": 2, "bid": 5, "speech": "unterminated'

    with pytest.raises(LLMError, match="有损恢复"):
        LLMRouter._parse_json(content, config)


def test_parse_json_can_expose_lossy_metadata_when_explicitly_allowed():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")
    content = '{"target_seat": 2, "bid": 5, "speech": "unterminated'

    parsed = LLMRouter._parse_json(
        content,
        config,
        allow_lossy=True,
        include_parse_metadata=True,
    )

    assert parsed["target_seat"] == 2
    assert parsed["bid"] == 5
    assert "speech" not in parsed
    assert parsed["_parse_lossy"] is True
    assert parsed["_parse_recovered"] is True
    assert parsed["_parse_method"] == "lossy_kv"


def test_parse_json_accepts_non_lossy_recovery():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    parsed = LLMRouter._parse_json('{"target_seat": 2, "bid": 5', config, include_parse_metadata=True)

    assert parsed["target_seat"] == 2
    assert parsed["bid"] == 5
    assert parsed["_parse_lossy"] is False
    assert parsed["_parse_recovered"] is True
    assert parsed["_parse_method"] == "balanced_literal"


def test_parse_json_marks_markdown_fence_as_non_lossy_recovery():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    parsed = LLMRouter._parse_json(
        '```json\n{"target_seat": 2}\n```',
        config,
        include_parse_metadata=True,
    )

    assert parsed["target_seat"] == 2
    assert parsed["_parse_recovered"] is True
    assert parsed["_parse_lossy"] is False
    assert parsed["_parse_method"] == "fenced_json"


@pytest.mark.asyncio
async def test_complete_json_rejects_lossy_then_accepts_complete_json():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")
    router = LLMRouter(max_retries=1)
    responses = [
        LLMResponse(content='{"target_seat": 2, "speech": "unterminated', finish_reason="stop"),
        LLMResponse(content='{"target_seat": 3, "speech": "ok"}', finish_reason="stop"),
    ]

    async def fake_complete(*_args, **_kwargs):
        return responses.pop(0)

    router._complete = fake_complete  # type: ignore[method-assign]

    with pytest.raises(LLMError, match="有损恢复"):
        await router.complete_json([{"role": "user", "content": "hi"}], config)

    parsed = await router.complete_json([{"role": "user", "content": "hi"}], config)
    assert parsed == {"target_seat": 3, "speech": "ok"}
    stats = router.stats.snapshot()
    assert stats["structured_responses"] == 2
    assert stats["lossy_parse_rejections"] == 1
    assert stats["response_parse_failures"] == 0
    assert stats["response_parse_recoveries"] == 0


@pytest.mark.asyncio
async def test_complete_json_stats_distinguish_parse_failure_and_accepted_recovery():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")
    router = LLMRouter(max_retries=1)
    responses = [
        LLMResponse(content="not json", finish_reason="stop"),
        LLMResponse(content='```json\n{"target_seat": 2}\n```', finish_reason="stop"),
    ]

    async def fake_complete(*_args, **_kwargs):
        return responses.pop(0)

    router._complete = fake_complete  # type: ignore[method-assign]

    with pytest.raises(LLMResponseError, match="JSON 解析失败"):
        await router.complete_json([{"role": "user", "content": "hi"}], config)
    parsed = await router.complete_json(
        [{"role": "user", "content": "hi"}],
        config,
        include_parse_metadata=True,
    )

    assert parsed["target_seat"] == 2
    assert parsed["_parse_recovered"] is True
    stats = router.stats.snapshot()
    assert stats["structured_responses"] == 2
    assert stats["response_parse_failures"] == 1
    assert stats["response_parse_recoveries"] == 1
    assert stats["lossy_parse_rejections"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "finish_reason"),
    [
        ("openai", ""),
        ("openai", "length"),
        ("openai_responses", "incomplete"),
        ("anthropic", "max_tokens"),
    ],
)
async def test_complete_json_rejects_incomplete_finish_status(provider, finish_reason):
    config = ModelConfig(provider=provider, api_base="http://example.invalid", api_key="x", model="m")
    router = LLMRouter(max_retries=1)

    async def fake_complete(*_args, **_kwargs):
        return LLMResponse(content='{"target_seat": 2}', finish_reason=finish_reason)

    router._complete = fake_complete  # type: ignore[method-assign]

    with pytest.raises(LLMError, match="未完整结束|缺少完成状态"):
        await router.complete_json([{"role": "user", "content": "hi"}], config)

    assert router.stats.snapshot()["incomplete_responses"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "finish_reason"),
    [
        ("openai", ""),
        ("openai", "length"),
        ("openai_responses", "incomplete"),
        ("anthropic", "max_tokens"),
    ],
)
async def test_structured_surface_rejects_incomplete_finish_status(provider, finish_reason):
    config = ModelConfig(provider=provider, api_base="http://example.invalid", api_key="x", model="m")
    router = LLMRouter(max_retries=1)

    async def fake_complete(*_args, **_kwargs):
        return LLMResponse(content="partial", finish_reason=finish_reason)

    router._complete = fake_complete  # type: ignore[method-assign]

    with pytest.raises(LLMError, match="未完整结束|缺少完成状态"):
        await router.complete_json([{"role": "user", "content": "hi"}], config)


@pytest.mark.asyncio
async def test_agent_call_with_retry_never_accepts_lossy_decision_json():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FakeRouter:
        def __init__(self) -> None:
            self.calls = 0
            self.allow_lossy_flags: list[bool] = []

        async def complete_json(self, *_args, **kwargs):
            self.calls += 1
            self.allow_lossy_flags.append(bool(kwargs.get("allow_lossy")))
            if self.calls == 1:
                raise LLMResponseError("JSON 有损恢复被拒绝(provider=openai method=lossy_kv)")
            return {"action": "speak", "speech": "ok", "_parse_lossy": True, "_parse_method": "lossy_kv"}

    fake_router = FakeRouter()
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=config,
        router=fake_router,  # type: ignore[arg-type]
    )

    with pytest.raises(AgentDecisionError, match="有损恢复"):
        await actor._call_with_retry([], "", max_attempts=2)

    assert fake_router.calls == 2
    assert fake_router.allow_lossy_flags == [False, False]


@pytest.mark.asyncio
async def test_agent_retries_out_of_range_bid_instead_of_clamping_intent():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FakeRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, *_args, **_kwargs):
            self.calls += 1
            return {"speech": "public", "bid": 9}

    router = FakeRouter()
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=config,
        router=router,  # type: ignore[arg-type]
    )

    with pytest.raises(AgentDecisionError, match="bid 超出合法范围"):
        await actor._call_with_retry(
            [],
            "",
            max_attempts=2,
            required_fields=["speech?", "bid"],
        )

    assert router.calls == 2


@pytest.mark.asyncio
async def test_agent_retries_fractional_target_instead_of_truncating_intent():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FakeRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, *_args, **_kwargs):
            self.calls += 1
            return {"target_seat": 2.9}

    router = FakeRouter()
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=config,
        router=router,  # type: ignore[arg-type]
    )

    with pytest.raises(AgentDecisionError, match="不是整数座位"):
        await actor._call_with_retry(
            [],
            "",
            max_attempts=2,
            required_fields=["target_seat"],
        )

    assert router.calls == 2


@pytest.mark.asyncio
async def test_agent_does_not_multiply_router_transport_failures():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FailedRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_json(self, *_args, **_kwargs):
            self.calls += 1
            raise LLMError("transport retries exhausted")

    router = FailedRouter()
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=config,
        router=router,  # type: ignore[arg-type]
    )

    with pytest.raises(AgentDecisionError, match="transport retries exhausted"):
        await actor._call_with_retry([], "", max_attempts=5)

    assert router.calls == 1


def _function_tools_for_router_tests() -> list[dict[str, object]]:
    return [{
        "type": "function",
        "function": {
            "name": "inspect_seat",
            "description": "Inspect one seat.",
            "parameters": {
                "type": "object",
                "properties": {"seat": {"type": "integer"}},
                "required": ["seat"],
                "additionalProperties": False,
            },
        },
    }]


@pytest.mark.asyncio
async def test_complete_tools_rejects_unsupported_definition_before_transport():
    router = LLMRouter(max_retries=1)
    config = ModelConfig(
        provider="openai",
        api_base="https://gateway.example.invalid/v1",
        api_key="test-key",
        model="test-model",
    )
    with pytest.raises(LLMResponseError, match="standard type=function"):
        await router.complete_tools(
            [{"role": "user", "content": "inspect"}],
            config,
            [{"name": "inspect_seat", "parameters": {"type": "object"}}],
        )
    assert router.stats.calls == 0


@pytest.mark.asyncio
async def test_openai_chat_complete_tools_rejects_missing_finish_marker():
    events = [
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call-no-finish",
            "type": "function",
            "function": {"name": "inspect_seat", "arguments": "{}"},
        }]}}]},
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
    )
    try:
        with pytest.raises(LLMResponseError, match="without finish_reason"):
            await router.complete_tools(
                [{"role": "user", "content": "inspect"}],
                config,
                _function_tools_for_router_tests(),
            )
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_openai_chat_complete_tools_accumulates_fragmented_arguments_without_token_caps():
    events = [
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call-chat-1",
                        "type": "function",
                        "function": {"name": "inspect_seat", "arguments": '{"se'},
                    }]
                }
            }]
        },
        {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": 'at": 3}'},
                    }]
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
        # Some OpenAI-compatible gateways repeat an empty finish frame after
        # the semantic tool-call completion; it must not invalidate the call.
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        "[DONE]",
    ]
    server, handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
        max_tokens=123,
    )
    try:
        result = await router.complete_tools(
            [{"role": "user", "content": "inspect"}],
            config,
            _function_tools_for_router_tests(),
            tool_choice="required",
            parallel_tool_calls=False,
            trace_context={"request_id": "tool-chat"},
        )
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert result.content == ""
    assert result.reasoning == ""
    assert result.finish_reason == "tool_calls"
    assert result.usage == {"prompt_tokens": 11, "completion_tokens": 7}
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].call_id == "call-chat-1"
    assert result.tool_calls[0].name == "inspect_seat"
    assert result.tool_calls[0].arguments == {"seat": 3}
    assert result.tool_calls[0].raw_arguments == '{"seat": 3}'
    assert result.trace["tool_call_count"] == 1
    body = handler.requests[0]["body"]
    assert body["tools"] == _function_tools_for_router_tests()
    assert body["tool_choice"] == "required"
    assert body["parallel_tool_calls"] is False
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body
    assert "max_output_tokens" not in body


@pytest.mark.asyncio
async def test_openai_responses_complete_tools_accumulates_argument_events():
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "fc-item-1",
                "type": "function_call",
                "call_id": "call-resp-1",
                "name": "inspect_seat",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc-item-1",
            "output_index": 0,
            "delta": '{"seat":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc-item-1",
            "output_index": 0,
            "delta": " 4}",
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc-item-1",
            "output_index": 0,
            "name": "inspect_seat",
            "arguments": '{"seat": 4}',
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 13, "output_tokens": 8},
            },
        },
        "[DONE]",
    ]
    server, handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai_responses",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
        max_tokens=123,
    )
    try:
        result = await router.complete_tools(
            [{"role": "user", "content": "inspect"}],
            config,
            _function_tools_for_router_tests(),
            tool_choice={"type": "function", "function": {"name": "inspect_seat"}},
            parallel_tool_calls=True,
        )
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert result.tool_calls[0].arguments == {"seat": 4}
    body = handler.requests[0]["body"]
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["name"] == "inspect_seat"
    assert body["tool_choice"] == {"type": "function", "name": "inspect_seat"}
    assert body["parallel_tool_calls"] is True
    assert "max_tokens" not in body
    assert "max_output_tokens" not in body


@pytest.mark.asyncio
async def test_openai_responses_tool_item_done_repeats_complete_arguments_without_conflict():
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "fc-item-repeat",
                "type": "function_call",
                "call_id": "call-repeat",
                "name": "inspect_seat",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc-item-repeat",
            "output_index": 0,
            "delta": '{"seat":',
        },
        {
            "type": "response.function_call_arguments.done",
            "item_id": "fc-item-repeat",
            "output_index": 0,
            "name": "inspect_seat",
            "arguments": '{"seat": 6}',
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "fc-item-repeat",
                "type": "function_call",
                "call_id": "call-repeat",
                "name": "inspect_seat",
                "arguments": '{"seat": 6}',
            },
        },
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{
                    "id": "fc-item-repeat",
                    "type": "function_call",
                    "call_id": "call-repeat",
                    "name": "inspect_seat",
                    "arguments": '{"seat": 6}',
                }],
            },
        },
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai_responses",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
    )
    try:
        result = await router.complete_tools(
            [{"role": "user", "content": "inspect"}],
            config,
            _function_tools_for_router_tests(),
        )
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].call_id == "call-repeat"
    assert result.tool_calls[0].arguments == {"seat": 6}


@pytest.mark.asyncio
async def test_anthropic_complete_tools_accumulates_input_json_delta_and_sends_required_max_tokens():
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg-tool",
                "type": "message",
                "role": "assistant",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 9, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call-ant-1",
                "name": "inspect_seat",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"seat":'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": " 5}"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 6},
        },
        {"type": "message_stop"},
    ]
    server, handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="anthropic",
        api_base=f"http://127.0.0.1:{server.server_address[1]}",
        api_key="test-key",
        model="claude-test",
        max_tokens=64,
    )
    try:
        result = await router.complete_tools(
            [{"role": "user", "content": "inspect"}],
            config,
            _function_tools_for_router_tests(),
            tool_choice="required",
            parallel_tool_calls=False,
        )
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert result.tool_calls[0].arguments == {"seat": 5}
    body = handler.requests[0]["body"]
    assert body["tools"][0]["name"] == "inspect_seat"
    assert body["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
    assert body["max_tokens"] == 64
    assert "max_output_tokens" not in body


@pytest.mark.asyncio
async def test_complete_tools_rejects_invalid_json_and_conflicting_fragments():
    invalid_events = [
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call-invalid",
            "type": "function",
            "function": {"name": "inspect_seat", "arguments": "not-json"},
        }]}, "finish_reason": "tool_calls"}]},
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(invalid_events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
    )
    try:
        with pytest.raises(LLMResponseError, match="valid complete JSON"):
            await router.complete_tools(
                [{"role": "user", "content": "inspect"}],
                config,
                _function_tools_for_router_tests(),
            )
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    conflicting_events = [
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call-conflict-a",
            "type": "function",
            "function": {"name": "inspect_seat", "arguments": "{}"},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call-conflict-b",
            "function": {"arguments": ""},
        }]}, "finish_reason": "tool_calls"}]},
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(conflicting_events)
    router = LLMRouter(timeout=5, max_retries=1)
    config.api_base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        with pytest.raises(LLMResponseError, match="conflicting"):
            await router.complete_tools(
                [{"role": "user", "content": "inspect"}],
                config,
                _function_tools_for_router_tests(),
            )
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_tool_history_translates_assistant_calls_and_results_for_responses_and_anthropic():
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-history",
                "type": "function",
                "function": {"name": "inspect_seat", "arguments": '{"seat": 2}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call-history", "content": "trusted result"},
        {"role": "user", "content": "continue"},
    ]
    events = [
        {"type": "response.output_text.delta", "delta": "ok"},
        {"type": "response.completed", "response": {"status": "completed"}},
        "[DONE]",
    ]
    server, handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai_responses",
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="test-key",
        model="test-model",
    )
    try:
        await router.complete_tools(messages, config, _function_tools_for_router_tests())
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()
    input_items = handler.requests[0]["body"]["input"]
    assert {"type": "function_call", "call_id": "call-history", "name": "inspect_seat", "arguments": '{"seat": 2}'} in input_items
    assert {"type": "function_call_output", "call_id": "call-history", "output": "trusted result"} in input_items

    events = [
        {
            "type": "message_start",
            "message": {"content": [], "usage": {"input_tokens": 1, "output_tokens": 0}},
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "ok"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    server, handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config.provider = "anthropic"
    config.api_base = f"http://127.0.0.1:{server.server_address[1]}"
    config.model = "claude-test"
    try:
        await router.complete_tools(messages, config, _function_tools_for_router_tests())
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()
    anthropic_messages = handler.requests[0]["body"]["messages"]
    assert any(
        isinstance(item.get("content"), list)
        and any(block.get("type") == "tool_use" for block in item["content"])
        for item in anthropic_messages
    )
    assert any(
        isinstance(item.get("content"), list)
        and any(block.get("type") == "tool_result" for block in item["content"])
        for item in anthropic_messages
    )
