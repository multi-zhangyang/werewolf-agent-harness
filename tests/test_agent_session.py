"""Focused contracts for the per-seat Agent tool loop."""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import pytest

from src.agent.session import (
    AgentSession,
    AgentSessionError,
    AgentSessionLimits,
    _MAX_PRIVATE_TRACE_ROW_CHARS,
    SessionStatus,
    TerminalSubmission,
    ToolExecutionContext,
    ToolKind,
    ToolRegistry,
    ToolSpec,
)
from src.agent.schemas import AgentAction, Decision
from src.harness.agent_protocol import ActionRequest, LegalAction
from src.llm.router import LLMResponseError


def _request(*, action: str = "vote", target: int = 2) -> ActionRequest:
    return ActionRequest(
        request_id="req-1",
        run_id="run-1",
        seat=1,
        phase="voting",
        day=1,
        action_kind=action,
        observation={"my_seat": 1, "my_role": "villager", "public_events": []},
        legal_actions=[LegalAction(action=action, target_seats=[target], target_required=True)],
    )


def _terminal_registry(calls: list[tuple[str, dict]]) -> ToolRegistry:
    registry = ToolRegistry()

    async def read_public(ctx: ToolExecutionContext, args: dict):
        calls.append(("read", args))
        return {"events": ctx.observation.get("public_events", [])}

    def vote(ctx: ToolExecutionContext, args: dict):
        calls.append(("vote", args))
        return Decision(action=AgentAction.VOTE, target_seat=args["target_seat"])

    registry.register(
        "read_public_events",
        read_public,
        description="Read public events",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        kind=ToolKind.READ_ONLY,
    )
    registry.register(
        "vote",
        vote,
        description="Submit vote",
        parameters={
            "type": "object",
            "properties": {"target_seat": {"type": "integer"}},
            "required": ["target_seat"],
            "additionalProperties": False,
        },
        kind=ToolKind.TERMINAL,
    )
    return registry


class FakeRouter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete_tools(self, messages, config, tools, **kwargs):
        self.calls.append((deepcopy(messages), config, deepcopy(tools), deepcopy(kwargs)))
        value = self.responses.pop(0)
        if callable(value):
            value = value(messages)
        return value


@pytest.mark.asyncio
async def test_session_runs_read_then_terminal_with_private_trace():
    calls: list[tuple[str, dict]] = []
    router = FakeRouter([
        {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "read_public_events", "arguments": "{}"}}]},
        {"content": "", "tool_calls": [{"id": "c2", "function": {"name": "vote", "arguments": '{"target_seat": 2}'}}]},
    ])
    session = AgentSession(
        seat=1,
        role="villager",
        registry=_terminal_registry(calls),
        limits=AgentSessionLimits(max_steps=4),
    )
    result = await session.run(_request(), router=router, config=object())

    assert result.status == SessionStatus.COMPLETED
    assert result.completed
    assert result.require_decision().target_seat == 2
    assert [name for name, _ in calls] == ["read", "vote"]
    assert result.steps == 2
    assert result.tool_calls == 2
    assert {row["type"] for row in result.private_trace()} >= {
        "agent_turn_started",
        "model_generation",
        "tool_call_requested",
        "tool_result",
        "agent_action_submitted",
    }
    assert all(row["visibility"] == "admin" for row in result.private_trace())
    requested_rows = [
        row for row in result.private_trace()
        if row.get("type") == "tool_call_requested"
    ]
    assert requested_rows[0]["arguments"] == {}
    assert requested_rows[1]["arguments"] == {"target_seat": 2}
    assert router.calls[0][2][0]["function"]["name"] == "read_public_events"
    assert router.calls[0][3]["tool_choice"] == "required"
    assert router.calls[0][3]["parallel_tool_calls"] is False
    assert "my_seat" not in json.dumps(router.calls[0][2])
    continued_messages = router.calls[1][0]
    assert continued_messages[-2]["role"] == "assistant"
    assert continued_messages[-2]["tool_calls"][0]["id"] == "c1"
    assert continued_messages[-1]["role"] == "tool"
    assert continued_messages[-1]["tool_call_id"] == "c1"


@pytest.mark.asyncio
async def test_model_history_compacts_only_complete_old_tool_groups():
    calls: list[tuple[str, dict]] = []
    registry = _terminal_registry(calls)

    async def read_large(_ctx: ToolExecutionContext, _args: dict):
        return {
            "source": "large-test-result",
            "events": [
                {"index": index, "text": f"event-{index}-" + ("x" * 500)}
                for index in range(8)
            ],
        }

    registry.register(
        "read_large",
        read_large,
        description="Return a deliberately large observation",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        kind=ToolKind.READ_ONLY,
    )
    router = FakeRouter([
        {"tool_calls": [{"id": "large-1", "function": {"name": "read_large", "arguments": "{}"}}]},
        {"tool_calls": [{"id": "large-2", "function": {"name": "read_large", "arguments": "{}"}}]},
        {"tool_calls": [{"id": "large-3", "function": {"name": "read_large", "arguments": "{}"}}]},
        {"tool_calls": [{"id": "terminal", "function": {"name": "vote", "arguments": '{"target_seat": 2}'}}]},
    ])
    session = AgentSession(
        seat=1,
        registry=registry,
        limits=AgentSessionLimits(
            max_steps=5,
            max_model_history_chars=700,
            keep_recent_tool_groups=1,
        ),
    )

    result = await session.run(_request(), router=router, config=object())

    assert result.completed
    # The session-owned audit history remains complete and unmodified.
    assert [
        message["tool_call_id"]
        for message in session.messages
        if message.get("role") == "tool"
    ] == ["large-1", "large-2", "large-3", "terminal"]

    model_history = router.calls[-1][0]
    summaries = [
        json.loads(message["content"])
        for message in model_history
        if message.get("role") == "user"
        and "compacted_tool_exchange" in str(message.get("content"))
    ]
    assert summaries
    assert all(summary["type"] == "compacted_tool_exchange" for summary in summaries)
    summarized_tools = [
        call["tool"]
        for summary in summaries
        for call in summary["calls"]
    ]
    assert summarized_tools == ["read_large", "read_large"]

    # Every tool call still sent to the provider has all of its results. Old
    # groups disappeared atomically; the most recent raw group stayed intact.
    for index, message in enumerate(model_history):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        expected = {call["id"] for call in message["tool_calls"]}
        observed: set[str] = set()
        cursor = index + 1
        while cursor < len(model_history) and model_history[cursor].get("role") == "tool":
            observed.add(model_history[cursor]["tool_call_id"])
            cursor += 1
        assert observed == expected
    assert any(
        message.get("role") == "assistant"
        and message.get("tool_calls", [{}])[0].get("id") == "large-3"
        for message in model_history
    )
    assert not any(
        message.get("role") == "assistant"
        and message.get("tool_calls", [{}])[0].get("id") in {"large-1", "large-2"}
        for message in model_history
    )
    assert result.history_compactions >= 1
    assert result.max_compacted_tool_groups == 2
    assert result.peak_model_history_chars < result.peak_history_chars
    assert any(
        row.get("type") == "agent_history_compacted"
        and row.get("compacted_tool_groups") == 2
        for row in result.private_trace()
    )


@pytest.mark.asyncio
async def test_history_limit_miss_is_visible_without_compacting_recent_group():
    calls: list[tuple[str, dict]] = []
    registry = _terminal_registry(calls)

    async def read_large(_ctx: ToolExecutionContext, _args: dict):
        return {"events": [{"text": "x" * 1_500}]}

    registry.register(
        "read_large",
        read_large,
        description="Return a deliberately large observation",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        kind=ToolKind.READ_ONLY,
    )
    router = FakeRouter([
        {"tool_calls": [{"id": "large", "function": {"name": "read_large", "arguments": "{}"}}]},
        {"tool_calls": [{"id": "terminal", "function": {"name": "vote", "arguments": '{"target_seat": 2}'}}]},
    ])
    session = AgentSession(
        seat=1,
        registry=registry,
        limits=AgentSessionLimits(
            max_steps=3,
            max_model_history_chars=256,
            keep_recent_tool_groups=1,
        ),
    )

    result = await session.run(_request(), router=router, config=object())

    assert result.completed
    assert result.history_compactions == 0
    assert result.history_limit_misses >= 1
    miss_rows = [
        row for row in result.private_trace()
        if row.get("type") == "agent_history_compacted"
        and row.get("compacted_tool_groups") == 0
    ]
    assert miss_rows and miss_rows[0]["limit_satisfied"] is False
    # The soft limit remains provider-compatible: the complete recent group is
    # retained and sent instead of manufacturing an orphaned tool protocol.
    assert len(json.dumps(router.calls[1][0], ensure_ascii=False)) > 256


@pytest.mark.asyncio
async def test_private_trace_redacts_credentials_and_bounds_large_values():
    secret = "sk-session-trace-secret-123456789"
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzZWF0LTEifQ.signaturevalue123456"
    calls: list[tuple[str, dict]] = []
    registry = _terminal_registry(calls)
    registry.unregister("vote")

    def vote(ctx: ToolExecutionContext, args: dict):
        return Decision(
            action=AgentAction.VOTE,
            target_seat=args["target_seat"],
            reasoning=secret,
        )

    registry.register(
        "vote",
        vote,
        description="Submit vote",
        parameters={
            "type": "object",
            "properties": {"target_seat": {"type": "integer"}},
            "required": ["target_seat"],
            "additionalProperties": False,
        },
        kind=ToolKind.TERMINAL,
    )
    router = FakeRouter([
        {"content": secret, "reasoning": f"private {secret}", "tool_calls": [{
            "id": "bad",
            "function": {"name": "vote", "arguments": json.dumps({"target_seat": 2, "access_token": jwt})},
        }]},
        {"content": "ok", "reasoning": secret, "tool_calls": [{
            "id": "good",
            "function": {"name": "vote", "arguments": '{"target_seat": 2}'},
        }]},
    ])
    session = AgentSession(seat=1, registry=registry)
    result = await session.run(
        _request(),
        router=router,
        config=object(),
        trace_context={"api_key": secret, "nested": {"access_token": jwt}},
    )

    serialized_trace = json.dumps(result.private_trace(), ensure_ascii=False)
    serialized_messages = json.dumps(session.messages, ensure_ascii=False)
    assert result.completed
    assert secret not in serialized_trace
    assert jwt not in serialized_trace
    assert secret not in serialized_messages
    assert any(row.get("type") == "model_generation" for row in result.private_trace())

    huge_registry = _terminal_registry([])

    async def huge(_ctx: ToolExecutionContext, _args: dict):
        return {f"field_{index}": "x" * 1_000 for index in range(100)}

    huge_registry.register(
        "huge",
        huge,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        kind=ToolKind.READ_ONLY,
    )
    huge_router = FakeRouter([
        {"tool_calls": [{"id": "huge", "function": {"name": "huge", "arguments": "{}"}}]},
        {"tool_calls": [{"id": "vote", "function": {"name": "vote", "arguments": '{"target_seat": 2}'}}]},
    ])
    huge_session = AgentSession(seat=1, registry=huge_registry)
    huge_result = await huge_session.run(_request(), router=huge_router, config=object())
    huge_tool_row = next(row for row in huge_result.private_trace() if row.get("type") == "tool_result")
    assert len(json.dumps(huge_tool_row, ensure_ascii=False)) <= _MAX_PRIVATE_TRACE_ROW_CHARS
    assert huge_tool_row["output"]["type"] == "trace_value_truncated"


@pytest.mark.asyncio
async def test_terminal_tool_on_last_admitted_step_is_not_rejected_by_budget():
    """The final allowed model step still gets to execute its terminal tool."""
    calls: list[tuple[str, dict]] = []
    router = FakeRouter([
        {"tool_calls": [{"id": "read", "function": {"name": "read_public_events", "arguments": "{}"}}]},
        {"tool_calls": [{"id": "vote", "function": {"name": "vote", "arguments": '{"target_seat": 2}'}}]},
    ])
    session = AgentSession(
        seat=1,
        role="villager",
        registry=_terminal_registry(calls),
        limits=AgentSessionLimits(max_steps=2),
    )

    result = await session.run(_request(), router=router, config=object())

    assert result.completed
    assert result.require_decision().target_seat == 2
    assert [name for name, _ in calls] == ["read", "vote"]
    assert not any(
        row.get("type") == "agent_turn_failed" and row.get("error_code") == "max_steps"
        for row in result.private_trace()
    )


@pytest.mark.asyncio
async def test_tool_error_is_observation_and_model_can_recover():
    calls: list[tuple[str, dict]] = []
    router = FakeRouter([
        {"tool_calls": [{"id": "bad", "function": {"name": "vote", "arguments": '{"target_seat": 9}'}}]},
        {"tool_calls": [{"id": "good", "function": {"name": "vote", "arguments": '{"target_seat": 2}'}}]},
    ])
    session = AgentSession(
        seat=1,
        registry=_terminal_registry(calls),
        limits=AgentSessionLimits(max_steps=4),
    )
    result = await session.run(_request(), router=router, config=object())

    assert result.completed
    assert [name for name, _ in calls] == ["vote"]
    assert any(
        row["type"] == "tool_result"
        and isinstance(row.get("error"), dict)
        and row["error"].get("code") == "illegal_terminal_action"
        for row in result.private_trace()
    )
    # The invalid tool result is retained in the next model context.
    assert any(
        message.get("role") == "tool"
        and "illegal_terminal_action" in message.get("content", "")
        for message in session.messages
    )


@pytest.mark.asyncio
async def test_terminal_submission_is_exactly_once_and_run_is_idempotent():
    calls: list[tuple[str, dict]] = []
    router = FakeRouter([
        {
            "tool_calls": [
                {"id": "first", "function": {"name": "vote", "arguments": '{"target_seat": 2}'}},
                {"id": "second", "function": {"name": "vote", "arguments": '{"target_seat": 2}'}},
            ]
        }
    ])
    session = AgentSession(seat=1, registry=_terminal_registry(calls))
    result = await session.run(_request(), router=router, config=object())
    again = await session.run(_request(), router=router, config=object())

    assert result.completed and again.completed
    assert result.require_decision().target_seat == 2
    assert again.require_decision().target_seat == 2
    assert [name for name, _ in calls] == ["vote"]
    assert len(router.calls) == 1

    with pytest.raises(AgentSessionError, match="only one ActionRequest"):
        await session.run(
            _request(action="speak", target=2).model_copy(update={"request_id": "req-2"}),
            router=router,
            config=object(),
        )


@pytest.mark.asyncio
async def test_no_progress_budget_fails_without_synthetic_skip():
    router = FakeRouter([{"content": "I am thinking", "tool_calls": []}] * 3)
    session = AgentSession(
        seat=1,
        registry=ToolRegistry(),
        limits=AgentSessionLimits(max_steps=5, max_no_progress_steps=2),
    )
    result = await session.run(_request(), router=router, config=object())

    assert result.status == SessionStatus.FAILED
    assert result.error is not None and result.error.code == "no_progress"
    assert result.decision is None


@pytest.mark.asyncio
async def test_wall_time_budget_cancels_a_stuck_model_turn():
    class SlowRouter:
        async def complete_tools(self, *_args, **_kwargs):
            await asyncio.Event().wait()

    session = AgentSession(
        seat=1,
        registry=_terminal_registry([]),
        limits=AgentSessionLimits(max_wall_time_seconds=0.02),
    )
    result = await session.run(_request(), router=SlowRouter(), config=object())

    assert result.status == SessionStatus.FAILED
    assert result.failure is not None and result.failure.code == "wall_time_exceeded"
    assert result.decision is None


@pytest.mark.asyncio
async def test_tool_timeout_is_model_visible_and_loop_can_recover():
    registry = _terminal_registry([])

    async def stuck_tool(_ctx: ToolExecutionContext, _args: dict):
        await asyncio.Event().wait()

    registry.register("stuck_tool", stuck_tool, kind=ToolKind.READ_ONLY)
    router = FakeRouter([
        {
            "tool_calls": [{
                "id": "slow",
                "function": {"name": "stuck_tool", "arguments": "{}"},
            }],
        },
        {
            "tool_calls": [{
                "id": "recover",
                "function": {
                    "name": "vote",
                    "arguments": '{"target_seat": 2}',
                },
            }],
        },
    ])
    session = AgentSession(
        seat=1,
        registry=registry,
        limits=AgentSessionLimits(
            max_steps=4,
            max_wall_time_seconds=1,
            max_tool_time_seconds=0.01,
        ),
    )

    result = await session.run(_request(), router=router, config=object())

    assert result.completed
    assert result.require_decision().target_seat == 2
    assert len(router.calls) == 2
    assert any(
        row["type"] == "tool_result"
        and isinstance(row.get("error"), dict)
        and row["error"].get("code") == "tool_timeout"
        for row in result.private_trace()
    )
    assert any(
        message.get("role") == "tool"
        and "tool_timeout" in message.get("content", "")
        for message in session.messages
    )


@pytest.mark.asyncio
async def test_tool_ignoring_cancellation_fails_closed_without_hanging_session():
    registry = _terminal_registry([])
    release = asyncio.Event()
    finished = asyncio.Event()

    async def cancellation_resistant(_ctx: ToolExecutionContext, _args: dict):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
        finally:
            finished.set()

    registry.register(
        "cancellation_resistant",
        cancellation_resistant,
        kind=ToolKind.READ_ONLY,
    )
    router = FakeRouter([
        {
            "tool_calls": [{
                "id": "resistant",
                "function": {
                    "name": "cancellation_resistant",
                    "arguments": "{}",
                },
            }],
        },
        {
            "tool_calls": [{
                "id": "terminal",
                "function": {
                    "name": "vote",
                    "arguments": '{"target_seat": 2}',
                },
            }],
        },
    ])
    session = AgentSession(
        seat=1,
        registry=registry,
        limits=AgentSessionLimits(
            max_steps=4,
            max_wall_time_seconds=2,
            max_tool_time_seconds=0.01,
        ),
    )
    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await session.run(_request(), router=router, config=object())

    assert loop.time() - started < 1
    assert result.failed
    assert result.error is not None
    assert result.error.code == "tool_cleanup_pending"
    assert result.error.details == {"pending_task_count": 1}

    release.set()
    await asyncio.wait_for(finished.wait(), timeout=1)
    await session.aclose()


@pytest.mark.asyncio
async def test_async_trace_sink_is_flushed_by_session_close():
    received: list[dict] = []

    async def sink(row: dict):
        await asyncio.sleep(0.01)
        received.append(row)

    router = FakeRouter([
        {"tool_calls": [{"id": "terminal", "function": {
            "name": "vote",
            "arguments": '{"target_seat": 2}',
        }}]},
    ])
    session = AgentSession(
        seat=1,
        registry=_terminal_registry([]),
        trace_sink=sink,
    )
    result = await session.run(_request(), router=router, config=object())
    assert result.completed
    await session.aclose()
    assert received
    assert all(row["visibility"] == "admin" for row in received)


@pytest.mark.asyncio
async def test_response_level_router_error_retries_same_agent_turn():
    class ResponseRetryRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_tools(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                error = LLMResponseError("malformed provider tool stream")
                error.llm_call_trace = {
                    "call_id": "failed-call",
                    "request_hash": "a" * 64,
                    "response_hash": None,
                    "usage": {},
                }
                raise error
            return {
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                "tool_calls": [{"id": "terminal", "function": {
                    "name": "vote",
                    "arguments": '{"target_seat": 2}',
                }}],
            }

    router = ResponseRetryRouter()
    session = AgentSession(
        seat=1,
        registry=_terminal_registry([]),
        limits=AgentSessionLimits(max_steps=3, max_model_response_retries=1),
    )
    result = await session.run(_request(), router=router, config=object())

    assert result.completed
    assert router.calls == 2
    failed_rows = [
        row for row in result.private_trace()
        if row.get("type") == "model_generation_failed"
    ]
    assert len(failed_rows) == 1
    assert failed_rows[0]["error_type"] == "LLMResponseError"
    assert failed_rows[0]["will_retry"] is True
    assert result.generation_attempts == 2
    assert result.model_generations == 1
    assert result.generation_failures == 1
    assert result.response_retries == 1
    assert result.tool_successes == 1
    assert result.tool_failures == 0
    assert result.total_tokens == 8
    assert result.token_usage_complete is True
    assert result.telemetry["limits"]["max_model_generations"] is None
    finished = [
        row for row in result.private_trace()
        if row.get("type") == "agent_turn_finished"
    ]
    assert len(finished) == 1
    assert finished[0]["telemetry"]["generation_attempts"] == 2


@pytest.mark.asyncio
async def test_generation_budget_counts_response_retries_and_fails_explicitly():
    class AlwaysMalformedRouter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete_tools(self, *_args, **_kwargs):
            self.calls += 1
            raise LLMResponseError("malformed provider tool stream")

    router = AlwaysMalformedRouter()
    session = AgentSession(
        seat=1,
        registry=_terminal_registry([]),
        limits=AgentSessionLimits(
            max_steps=3,
            max_model_generations=2,
            max_model_response_retries=5,
        ),
    )

    result = await session.run(_request(), router=router, config=object())

    assert result.failed
    assert result.error is not None and result.error.code == "max_model_generations"
    assert router.calls == 2
    assert result.generation_attempts == 2
    assert result.generation_failures == 2
    assert result.response_retries == 1
    assert result.telemetry["budget_exhausted"] == "max_model_generations"
    failed = [
        row for row in result.private_trace()
        if row.get("type") == "model_generation_failed"
    ]
    assert [row["will_retry"] for row in failed] == [True, False]


@pytest.mark.asyncio
async def test_token_budget_rejects_crossing_response_before_tool_side_effect():
    calls: list[tuple[str, dict]] = []
    router = FakeRouter([{
        "usage": {"prompt_tokens": 5, "completion_tokens": 4},
        "tool_calls": [{"id": "terminal", "function": {
            "name": "vote",
            "arguments": '{"target_seat": 2}',
        }}],
    }])
    session = AgentSession(
        seat=1,
        registry=_terminal_registry(calls),
        limits=AgentSessionLimits(max_total_tokens=8),
    )

    result = await session.run(_request(), router=router, config=object())

    assert result.failed
    assert result.error is not None and result.error.code == "max_total_tokens"
    assert result.total_tokens == 9
    assert result.tool_calls == 0
    assert calls == []
    assert result.decision is None
    assert result.telemetry["budget_exhausted"] == "max_total_tokens"


@pytest.mark.asyncio
async def test_token_budget_allows_terminal_tool_at_exact_limit():
    calls: list[tuple[str, dict]] = []
    router = FakeRouter([{
        "usage": {"prompt_tokens": 5, "completion_tokens": 4},
        "tool_calls": [{"id": "terminal", "function": {
            "name": "vote",
            "arguments": '{"target_seat": 2}',
        }}],
    }])
    session = AgentSession(
        seat=1,
        registry=_terminal_registry(calls),
        limits=AgentSessionLimits(max_total_tokens=9),
    )

    result = await session.run(_request(), router=router, config=object())

    assert result.completed
    assert result.total_tokens == 9
    assert [name for name, _ in calls] == ["vote"]
    assert result.telemetry["budget_exhausted"] is None


@pytest.mark.asyncio
async def test_strict_token_budget_fails_closed_when_usage_is_missing():
    router = FakeRouter([{
        "tool_calls": [{"id": "terminal", "function": {
            "name": "vote",
            "arguments": '{"target_seat": 2}',
        }}],
    }])
    session = AgentSession(
        seat=1,
        registry=_terminal_registry([]),
        limits=AgentSessionLimits(
            max_total_tokens=100,
            require_token_usage=True,
        ),
    )

    result = await session.run(_request(), router=router, config=object())

    assert result.failed
    assert result.error is not None
    assert result.error.code == "token_usage_unavailable"
    assert result.token_usage_complete is False
    assert result.tool_calls == 0


def test_request_budget_limit_validation_is_fail_closed():
    with pytest.raises(ValueError, match="max_model_generations"):
        AgentSessionLimits(max_model_generations=0)
    with pytest.raises(ValueError, match="max_total_tokens"):
        AgentSessionLimits(max_total_tokens=-1)
    with pytest.raises(ValueError, match="requires max_total_tokens"):
        AgentSessionLimits(require_token_usage=True)


def test_registry_rejects_identity_fields_but_allows_target_seat():
    with pytest.raises(ValueError, match="identity"):
        ToolSpec(
            name="bad",
            description="bad",
            parameters={
                "type": "object",
                "properties": {"my_seat": {"type": "integer"}},
            },
            kind=ToolKind.READ_ONLY,
            handler=lambda _ctx, _args: None,
        )
    spec = ToolSpec(
        name="good",
        description="good",
        parameters={
            "type": "object",
            "properties": {"target_seat": {"type": "integer"}},
        },
        kind=ToolKind.TERMINAL,
        handler=lambda _ctx, _args: Decision(action=AgentAction.VOTE, target_seat=2),
    )
    assert spec.name == "good"


@pytest.mark.asyncio
async def test_handler_exception_is_recoverable_tool_result():
    registry = ToolRegistry()

    async def broken(_ctx, _args):
        raise RuntimeError("secret provider detail")

    registry.register("broken", broken, kind=ToolKind.READ_ONLY)
    context = ToolExecutionContext(request=_request(), seat=1, role="villager", step=1, state_version=0)
    result = await registry.execute("x", "broken", {}, context)
    assert not result.ok
    assert result.error_code == "tool_execution_failed"
    assert "secret provider detail" not in result.error_message


@pytest.mark.asyncio
async def test_tool_context_cannot_mutate_request_used_by_later_tools():
    registry = ToolRegistry()
    request = _request()

    def mutate_snapshot(ctx: ToolExecutionContext, _args: dict):
        exposed = ctx.request
        exposed.observation["my_seat"] = 999
        exposed.observation.setdefault("public_events", []).append("forged")
        exposed.legal_actions[0].target_seats.clear()
        exposed.private_context["injected"] = True
        exposed.metadata["injected"] = True
        return {"mutated_local_snapshot": True}

    def vote(_ctx: ToolExecutionContext, args: dict):
        return Decision(action=AgentAction.VOTE, target_seat=args["target_seat"])

    registry.register("mutate", mutate_snapshot, kind=ToolKind.READ_ONLY)
    registry.register(
        "vote",
        vote,
        kind=ToolKind.TERMINAL,
        parameters={
            "type": "object",
            "properties": {"target_seat": {"type": "integer"}},
            "required": ["target_seat"],
            "additionalProperties": False,
        },
    )
    context = ToolExecutionContext(
        request=request,
        seat=1,
        role="villager",
        step=1,
        state_version=0,
    )

    mutated = await registry.execute("read-1", "mutate", {}, context)
    terminal = await registry.execute(
        "vote-1", "vote", {"target_seat": 2}, context
    )

    assert mutated.ok
    assert terminal.error_code != "illegal_terminal_action"
    assert context.request.observation["my_seat"] == 1
    assert context.request.observation["public_events"] == []
    assert context.request.legal_actions[0].target_seats == [2]
    assert context.request.private_context == {}
    assert context.request.metadata == {}
    assert request.observation["my_seat"] == 1
    assert request.legal_actions[0].target_seats == [2]


@pytest.mark.asyncio
async def test_handler_can_submit_terminal_through_context_exactly_once():
    registry = ToolRegistry()

    def submit(ctx: ToolExecutionContext, _args):
        return ctx.submit_terminal(
            Decision(action=AgentAction.VOTE, target_seat=2),
            tool_name="submit",
            call_id="call-1",
        )

    registry.register(
        "submit",
        submit,
        kind=ToolKind.TERMINAL,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    session = AgentSession(seat=1, registry=registry)
    context = ToolExecutionContext(
        request=_request(),
        seat=1,
        role="villager",
        step=1,
        state_version=0,
        session=session,
    )
    result = await registry.execute("call-1", "submit", {}, context)
    assert result.ok and result.terminal
    assert session.decision is not None and session.decision.target_seat == 2


@pytest.mark.asyncio
async def test_explicit_submit_illegal_candidate_rolls_back_and_retry_succeeds():
    registry = ToolRegistry()

    def submit(ctx: ToolExecutionContext, args: dict):
        ctx.submit_terminal(
            Decision(action=AgentAction.VOTE, target_seat=args["target_seat"]),
        )
        return None

    registry.register(
        "submit",
        submit,
        kind=ToolKind.TERMINAL,
        parameters={
            "type": "object",
            "properties": {"target_seat": {"type": "integer"}},
            "required": ["target_seat"],
            "additionalProperties": False,
        },
    )
    session = AgentSession(seat=1, registry=registry)
    context = ToolExecutionContext(
        request=_request(), seat=1, role="villager", step=1, state_version=0, session=session
    )

    rejected = await registry.execute("bad-call", "submit", {"target_seat": 9}, context)
    assert not rejected.ok and rejected.error_code == "illegal_terminal_action"
    assert session.terminal_submission is None

    accepted = await registry.execute("good-call", "submit", {"target_seat": 2}, context)
    assert accepted.ok and accepted.terminal
    assert session.decision is not None and session.decision.target_seat == 2
    assert session.terminal_tool == "submit"


@pytest.mark.asyncio
async def test_explicit_submit_handler_failure_and_conflict_do_not_commit():
    registry = ToolRegistry()

    def broken(ctx: ToolExecutionContext, _args: dict):
        ctx.submit_terminal(Decision(action=AgentAction.VOTE, target_seat=2))
        raise RuntimeError("handler failed")

    def conflicting(ctx: ToolExecutionContext, _args: dict):
        ctx.submit_terminal(Decision(action=AgentAction.VOTE, target_seat=2))
        return Decision(action=AgentAction.VOTE, target_seat=3)

    registry.register(
        "broken",
        broken,
        kind=ToolKind.TERMINAL,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    registry.register(
        "conflicting",
        conflicting,
        kind=ToolKind.TERMINAL,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    session = AgentSession(seat=1, registry=registry)
    context = ToolExecutionContext(
        request=_request(), seat=1, role="villager", step=1, state_version=0, session=session
    )

    failed = await registry.execute("broken-call", "broken", {}, context)
    conflicted = await registry.execute("conflict-call", "conflicting", {}, context)
    assert not failed.ok and failed.error_code == "tool_execution_failed"
    assert not conflicted.ok and conflicted.error_code == "terminal_submission_conflict"
    assert session.terminal_submission is None


@pytest.mark.asyncio
async def test_explicit_submit_timeout_rolls_back_pending_candidate():
    registry = ToolRegistry()

    async def hanging(ctx: ToolExecutionContext, _args: dict):
        ctx.submit_terminal(Decision(action=AgentAction.VOTE, target_seat=2))
        await asyncio.Event().wait()

    registry.register(
        "hanging",
        hanging,
        kind=ToolKind.TERMINAL,
        timeout_seconds=0.01,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    session = AgentSession(seat=1, registry=registry)
    context = ToolExecutionContext(
        request=_request(), seat=1, role="villager", step=1, state_version=0, session=session
    )

    result = await registry.execute("timeout-call", "hanging", {}, context)
    assert not result.ok and result.error_code == "tool_timeout"
    assert session.terminal_submission is None
    await session.aclose()


@pytest.mark.asyncio
async def test_explicit_submit_pending_state_isolated_between_sessions():
    registry = ToolRegistry()

    def submit(ctx: ToolExecutionContext, _args: dict):
        ctx.submit_terminal(Decision(action=AgentAction.VOTE, target_seat=2))
        return None

    registry.register(
        "submit",
        submit,
        kind=ToolKind.TERMINAL,
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    first = AgentSession(seat=1, registry=registry)
    second = AgentSession(seat=1, registry=registry)
    first_context = ToolExecutionContext(
        request=_request().model_copy(update={"request_id": "first"}),
        seat=1,
        role="villager",
        step=1,
        state_version=0,
        session=first,
    )
    second_context = ToolExecutionContext(
        request=_request().model_copy(update={"request_id": "second"}),
        seat=1,
        role="villager",
        step=1,
        state_version=0,
        session=second,
    )

    first_result = await registry.execute("first-call", "submit", {}, first_context)
    second_result = await registry.execute("second-call", "submit", {}, second_context)
    assert first_result.ok and second_result.ok
    assert first.terminal_submission is not None
    assert second.terminal_submission is not None
    assert first._terminal_call_id == "first-call"
    assert second._terminal_call_id == "second-call"
