"""LLMRouter 边界测试 —— 不调用真实 LLM。"""
from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.llm.models import ModelConfig
from src.llm.router import LLMError, LLMResponse, LLMRouter
from src.agent.actor import AgentActor, DECISION_MAX_ATTEMPTS, REFLECTION_MAX_ATTEMPTS
from src.agent.schemas import AgentAction
from src.game.roles import Role
from src.game.state import new_game


def test_model_config_accepts_three_standard_providers():
    assert ModelConfig(provider="openai").provider == "openai"
    assert ModelConfig(provider="openai_responses").provider == "openai_responses"
    assert ModelConfig(provider="anthropic").provider == "anthropic"
    assert ModelConfig(provider="OPENAI_RESPONSES").provider == "openai_responses"

    with pytest.raises(ValueError, match="不支持"):
        ModelConfig(provider="vendor_special")


@pytest.mark.parametrize("field_name", ["extra_body", "thinking", "reasoning_effort", "top_k"])
def test_model_config_rejects_non_standard_provider_fields(field_name):
    with pytest.raises(ValueError, match=field_name):
        ModelConfig(provider="openai", **{field_name: {"enabled": True}})


def test_model_config_merge_allows_explicit_zero_max_tokens_override():
    base = ModelConfig(provider="openai", model="room-model", api_base="https://example.invalid/v1", api_key="key", max_tokens=1024)

    merged = base.merge({"model": "seat-model", "max_tokens": 0})

    assert merged.model == "seat-model"
    assert merged.max_tokens == 0


def test_agent_real_decision_defaults_use_deep_retry_budget():
    decision_methods = (
        AgentActor.decide_night_action,
        AgentActor.decide_speak,
        AgentActor.decide_wolf_caucus,
        AgentActor.decide_vote,
        AgentActor.decide_last_words,
    )

    for method in decision_methods:
        assert inspect.signature(method).parameters["max_attempts"].default == DECISION_MAX_ATTEMPTS
    assert DECISION_MAX_ATTEMPTS >= 5


def test_agent_reflection_keeps_non_fatal_retry_budget_separate():
    assert inspect.signature(AgentActor.reflect).parameters["max_attempts"].default == REFLECTION_MAX_ATTEMPTS
    assert REFLECTION_MAX_ATTEMPTS < DECISION_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_agent_vote_without_target_or_suspicion_does_not_pick_first_candidate():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FakeRouter:
        async def complete_json(self, *_args, **_kwargs):
            return {"thought": "没有形成明确投票目标", "objective_summary": "公开发言不足以锁定目标"}

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

    decision = await actor.decide_vote(state, state.players[0].id)

    assert decision.action == AgentAction.SKIP
    assert decision.target_id is None
    assert decision.skip_reason == "vote_target_unresolved"


@pytest.mark.asyncio
async def test_agent_pk_vote_without_valid_target_or_suspicion_does_not_pick_first_candidate():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FakeRouter:
        async def complete_json(self, *_args, **_kwargs):
            return {
                "thought": "我想投 PK 外的人",
                "objective_summary": "PK 内两人都没有明确证据",
                "target_seat": 4,
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

    decision = await actor.decide_vote(
        state,
        state.players[0].id,
        pk_candidates=[state.players[1].id, state.players[2].id],
    )

    assert decision.action == AgentAction.SKIP
    assert decision.target_id is None
    assert decision.skip_reason == "vote_target_unresolved"


@pytest.mark.asyncio
async def test_hunter_requested_action_uses_real_llm_target_without_role_fallback():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FakeRouter:
        async def complete_json(self, *_args, **_kwargs):
            return {"thought": "猎人决定带走2号", "target_seat": 2}

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

    decision = await actor.decide_night_action(
        state,
        state.players[0].id,
        requested_action="hunter_shot",
    )

    assert decision.action == AgentAction.NIGHT_KILL
    assert decision.target_id == state.players[1].id


@pytest.mark.asyncio
async def test_witch_save_requested_action_rejects_poison_shape_transparently():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FakeRouter:
        async def complete_json(self, *_args, **_kwargs):
            return {"thought": "此阶段却想用毒", "use_poison": True, "poison_target": 2}

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

    decision = await actor.decide_night_action(
        state,
        state.players[0].id,
        requested_action="save",
        human_context={"killed_seat": 2},
    )

    assert decision.action == AgentAction.SKIP
    assert decision.skip_reason == "requested_action_mismatch"
    assert decision.target_id is None


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
    try:
        with pytest.raises(LLMError, match="总超时"):
            await router.complete_text([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert time.monotonic() - started < 1.5
    assert router.stats.snapshot()["failures"] == 1


@pytest.mark.asyncio
async def test_complete_routes_to_provider_specific_adapter():
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


@pytest.mark.asyncio
async def test_openai_chat_completions_reasoning_content_becomes_thought_without_content_pollution():
    events = [
        {"choices": [{"delta": {"reasoning_content": "先分析身份。"}}]},
        {"choices": [{"delta": {"content": '{"target_seat": 2}'}, "finish_reason": "stop"}]},
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai",
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

    assert resp.content == '{"target_seat": 2}'
    assert resp.reasoning == "先分析身份。"
    assert parsed == {"target_seat": 2, "thought": "先分析身份。"}


@pytest.mark.asyncio
async def test_openai_chat_completions_reasoning_delta_becomes_thought_without_content_pollution():
    events = [
        {"choices": [{"delta": {"reasoning_delta": "增量推理。"}}]},
        {"choices": [{"delta": {"reasoning": "结构化推理。"}}]},
        {"choices": [{"delta": {"content": '{"target_seat": 3}'}, "finish_reason": "stop"}]},
        "[DONE]",
    ]
    server, _handler = _start_recording_sse_server(events)
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="openai",
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

    assert resp.content == '{"target_seat": 3}'
    assert resp.reasoning == "增量推理。结构化推理。"
    assert parsed == {"target_seat": 3, "thought": "增量推理。结构化推理。"}


@pytest.mark.asyncio
async def test_openai_chat_completions_uses_max_completion_tokens_when_limited():
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
        text = await router.complete_text([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert text == "hello"
    body = handler.requests[0]["body"]
    assert body["max_completion_tokens"] == 64
    assert "max_tokens" not in body
    assert "response_format" not in body


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
    assert body["input"][-1] == {"role": "user", "content": "hi"}
    assert any("json" in item["content"].lower() for item in body["input"])
    assert body["max_output_tokens"] == 64
    assert body["text"] == {"format": {"type": "json_object"}}
    assert "top system" in body["instructions"]
    assert "message system" in body["instructions"]
    assert '{"ok": true}' in body["instructions"]
    assert "messages" not in body
    assert "response_format" not in body
    assert "max_tokens" not in body


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
async def test_openai_responses_zero_max_tokens_omits_max_output_tokens():
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
        max_tokens=0,
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
async def test_openai_responses_reasoning_delta_fills_thought_without_content_pollution():
    events = [
        {"type": "response.reasoning_summary.delta", "delta": {"text": "Responses 流式推理。"}},
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
async def test_anthropic_messages_wire_contract_and_sse_parse():
    events = [
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 7, "output_tokens": 0}},
        },
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 8},
        },
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
    assert req["headers"]["x-api-key"] == "test-key"
    assert req["headers"]["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-test"
    assert body["stream"] is True
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 8192
    assert "top system" in body["system"]
    assert "message system" in body["system"]
    assert '{"ok": true}' in body["system"]
    assert "response_format" not in body
    assert "stream_options" not in body
    assert "max_output_tokens" not in body


@pytest.mark.asyncio
async def test_anthropic_thinking_delta_fills_thought_without_content_pollution():
    events = [
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 7, "output_tokens": 0}},
        },
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "Anthropic 推理。"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": '{"vote": 2}'}},
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
async def test_anthropic_reasoning_delta_alias_fills_thought_without_content_pollution():
    events = [
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 7, "output_tokens": 0}},
        },
        {"type": "content_block_delta", "delta": {"type": "reasoning_delta", "reasoning": "兼容推理。"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": '{"vote": 3}'}},
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
        max_tokens=0,
    )

    try:
        resp = await router._complete([{"role": "user", "content": "hi"}], config)
        parsed = await router.complete_json([{"role": "user", "content": "hi"}], config)
    finally:
        await router.aclose()
        server.shutdown()
        server.server_close()

    assert resp.content == '{"vote": 3}'
    assert resp.reasoning == "兼容推理。"
    assert parsed == {"vote": 3, "thought": "兼容推理。"}


@pytest.mark.asyncio
async def test_anthropic_stream_error_event_raises():
    server, _handler = _start_recording_sse_server([
        {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
    ])
    router = LLMRouter(timeout=5, max_retries=1)
    config = ModelConfig(
        provider="anthropic",
        api_base=f"http://127.0.0.1:{server.server_address[1]}",
        api_key="test-key",
        model="claude-test",
    )

    try:
        with pytest.raises(LLMError, match="anthropic stream error"):
            await router.complete_text([{"role": "user", "content": "hi"}], config)
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


@pytest.mark.asyncio
async def test_agent_call_with_retry_retries_lossy_parse_error():
    config = ModelConfig(provider="openai", api_base="http://example.invalid", api_key="x", model="m")

    class FakeRouter:
        def __init__(self) -> None:
            self.calls = 0
            self.allow_lossy_flags: list[bool] = []

        async def complete_json(self, *_args, **kwargs):
            self.calls += 1
            self.allow_lossy_flags.append(bool(kwargs.get("allow_lossy")))
            if self.calls == 1:
                raise LLMError("JSON 有损恢复被拒绝(provider=openai method=lossy_kv)")
            return {"action": "speak", "speech": "ok", "_parse_lossy": True, "_parse_method": "lossy_kv"}

    fake_router = FakeRouter()
    actor = AgentActor(
        seat=1,
        name="A",
        role=Role.VILLAGER,
        model_config=config,
        router=fake_router,  # type: ignore[arg-type]
    )

    raw = await actor._call_with_retry([], "", max_attempts=2)

    assert raw["speech"] == "ok"
    assert raw["_parse_lossy"] is True
    assert fake_router.calls == 2
    assert fake_router.allow_lossy_flags == [False, True]

    decision = actor._sanitize_last_words(raw)
    assert decision.parse_failed is True
