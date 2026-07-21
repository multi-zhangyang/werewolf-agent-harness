"""Admission and provider-budget integration tests without real model calls."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import src.config as config_module
from src.api.limits import (
    AdmissionLimiter,
    ProviderBudgetLedger,
    ProviderBudgetPolicy,
    RateLimitConfig,
)
from src.api.room_manager import RoomManager
from src.api.server import create_app
from src.llm.models import ModelConfig
from src.llm.router import (
    LLMBudgetError,
    LLMResponse,
    LLMRouter,
    LLMRouterClosedError,
)


PLAYER_NAMES = ["A", "B", "C", "D", "E", "F"]


def _admission_limiter(*, rest_capacity: float = 100, ws_capacity: float = 100) -> AdmissionLimiter:
    return AdmissionLimiter(
        rest=RateLimitConfig(capacity=rest_capacity, refill_rate=0.001),
        ws=RateLimitConfig(capacity=ws_capacity, refill_rate=0.001),
    )


def test_rest_middleware_returns_bounded_429_and_keeps_health_available() -> None:
    manager = RoomManager(admission_limiter=_admission_limiter(rest_capacity=1))
    with TestClient(create_app(manager=manager)) as client:
        first = client.get("/api/providers")
        denied = client.get("/api/providers")
        health = client.get("/healthz")

    assert first.status_code == 200
    assert denied.status_code == 429
    assert denied.json()["reason"] == "rate_limited"
    assert int(denied.headers["retry-after"]) >= 1
    assert health.status_code == 200


def test_websocket_connection_rate_limit_closes_with_4429() -> None:
    manager = RoomManager(admission_limiter=_admission_limiter(ws_capacity=1))
    room = manager.create_room(player_names=PLAYER_NAMES)
    with TestClient(create_app(manager=manager)) as client:
        with client.websocket_connect(f"/ws/{room.id}?mode=spectate") as websocket:
            assert websocket.receive_json()["type"] == "snapshot"
        with pytest.raises(WebSocketDisconnect) as denied:
            with client.websocket_connect(f"/ws/{room.id}?mode=spectate"):
                pass

    assert denied.value.code == 4429


def test_websocket_inbound_message_rate_limit_closes_with_4429() -> None:
    manager = RoomManager(admission_limiter=_admission_limiter(ws_capacity=2))
    room = manager.create_room(player_names=PLAYER_NAMES)
    with TestClient(create_app(manager=manager)) as client:
        with client.websocket_connect(f"/ws/{room.id}?mode=spectate") as websocket:
            assert websocket.receive_json()["type"] == "snapshot"
            websocket.send_text("ping")
            assert websocket.receive_text() == "pong"
            websocket.send_text("ping")
            with pytest.raises(WebSocketDisconnect) as denied:
                websocket.receive_text()

    assert denied.value.code == 4429


def test_websocket_concurrent_room_capacity_closes_with_4429() -> None:
    manager = RoomManager(
        admission_limiter=_admission_limiter(ws_capacity=100),
        max_ws_clients_per_room=1,
    )
    room = manager.create_room(player_names=PLAYER_NAMES)
    with TestClient(create_app(manager=manager)) as client:
        with client.websocket_connect(f"/ws/{room.id}?mode=spectate") as first:
            assert first.receive_json()["type"] == "snapshot"
            with pytest.raises(WebSocketDisconnect) as denied:
                with client.websocket_connect(f"/ws/{room.id}?mode=spectate"):
                    pass

    assert denied.value.code == 4429


@pytest.mark.asyncio
async def test_terminal_room_closes_its_provider_budget_scope() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=5))
    manager = RoomManager(provider_budget_ledger=ledger)
    room = manager.create_room(player_names=PLAYER_NAMES)
    scope_id = ledger.room_scope(room.id)

    before = ledger.snapshot(scope_id)
    assert before is not None and not before.closed

    room.status = "failed"
    room.end_reason = "test_terminal"
    await manager._broadcast_room_status(room)

    after = ledger.snapshot(scope_id)
    assert after is not None and after.closed
    assert ledger.try_reserve(scope_id).reason == "scope_closed"
    await manager.aclose()


@pytest.mark.asyncio
async def test_router_reserves_each_transport_attempt_and_rejects_next_call() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=1))
    router = LLMRouter(max_retries=2, budget_ledger=ledger)
    transport_calls = 0

    async def fake_openai(*_args):
        nonlocal transport_calls
        transport_calls += 1
        return LLMResponse(
            content='{"choice":"ok"}',
            finish_reason="stop",
            usage={"prompt_tokens": 3, "completion_tokens": 2},
        )

    router._call_openai = fake_openai  # type: ignore[method-assign]
    model = ModelConfig(provider="openai", model="test", api_key="test-key")

    result = await router.complete_json(
        [{"role": "user", "content": "choose"}],
        model,
        trace_context={"run_id": "budget-run"},
    )
    with pytest.raises(LLMBudgetError) as denied:
        await router.complete_json(
            [{"role": "user", "content": "choose"}],
            model,
            trace_context={"run_id": "budget-run"},
        )

    assert result["choice"] == "ok"
    assert denied.value.budget_reason == "call_limit"
    assert transport_calls == 1
    snapshot = ledger.snapshot("run:budget-run")
    assert snapshot is not None
    assert snapshot.calls == 1
    assert snapshot.total_tokens == 5
    assert snapshot.reserved_calls == 0
    await router.aclose()


@pytest.mark.asyncio
async def test_router_budget_cannot_be_bypassed_by_omitting_scope() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=1))
    router = LLMRouter(max_retries=1, budget_ledger=ledger)
    transport_calls = 0

    async def fake_openai(*_args):
        nonlocal transport_calls
        transport_calls += 1
        return LLMResponse(content="{}", finish_reason="stop", usage={})

    router._call_openai = fake_openai  # type: ignore[method-assign]
    model = ModelConfig(provider="openai", model="test", api_key="test-key")

    with pytest.raises(LLMBudgetError) as denied:
        await router.complete_json([{"role": "user", "content": "choose"}], model)

    assert denied.value.budget_reason == "budget_scope_required"
    assert transport_calls == 0
    assert ledger.scope_count == 0
    await router.aclose()


@pytest.mark.asyncio
async def test_router_close_race_cancels_unstarted_budget_reservation() -> None:
    policy = ProviderBudgetPolicy(max_calls=1)
    ledger = ProviderBudgetLedger(default_policy=policy)
    router = LLMRouter(
        timeout=1,
        max_retries=1,
        budget_ledger=ledger,
        budget_policy=policy,
    )
    scope = ledger.run_scope("close-race")
    ledger.register_scope(scope, policy)
    config = ModelConfig(
        provider="openai",
        model="model-a",
        api_base="https://example.invalid/v1",
        api_key="unit-test-key",
    )

    async def must_not_start():
        raise AssertionError("provider transport must not start after close admission")

    def close_between_reservation_and_task(*_args, **_kwargs):
        with router._lifecycle_guard:
            router._closing = True
        return must_not_start()

    router._call_openai = close_between_reservation_and_task  # type: ignore[method-assign]
    with pytest.raises(LLMRouterClosedError, match="closing or closed"):
        await router.complete_json(
            [{"role": "user", "content": "choose"}],
            config,
            budget_scope=scope,
        )

    snapshot = ledger.snapshot(scope)
    assert snapshot is not None
    assert snapshot.calls == 0
    assert snapshot.reserved_calls == 0
    assert ledger.inflight_reservation_count == 0
    with router._lifecycle_guard:
        router._closing = False
    await router.aclose()


@pytest.mark.asyncio
async def test_missing_provider_usage_blocks_token_limited_scope() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=2, max_tokens=100))
    router = LLMRouter(max_retries=1, budget_ledger=ledger)

    async def fake_anthropic(*_args):
        return LLMResponse(
            content='{"choice":"ok"}',
            finish_reason="end_turn",
            usage={},
        )

    router._call_anthropic = fake_anthropic  # type: ignore[method-assign]
    model = ModelConfig(
        provider="anthropic",
        model="test",
        api_key="test-key",
        max_tokens=20,
    )

    with pytest.raises(LLMBudgetError) as failed:
        await router.complete_json(
            [{"role": "user", "content": "choose"}],
            model,
            trace_context={"run_id": "unknown-usage"},
        )

    assert failed.value.budget_reason == "usage_unknown"
    snapshot = ledger.snapshot("run:unknown-usage")
    assert snapshot is not None
    assert snapshot.calls == 1
    assert snapshot.unknown_usage_records == 1
    assert snapshot.blocked_reason == "usage_unknown"
    assert snapshot.reserved_tokens == 0
    await router.aclose()


@pytest.mark.asyncio
async def test_anthropic_input_and_output_usage_cannot_hide_output_only_overrun() -> None:
    ledger = ProviderBudgetLedger(ProviderBudgetPolicy(max_calls=2, max_tokens=100))
    router = LLMRouter(max_retries=1, budget_ledger=ledger)

    async def fake_anthropic(*_args):
        return LLMResponse(
            content='{"choice":"ok"}',
            finish_reason="end_turn",
            usage={"prompt_tokens": 8, "completion_tokens": 5},
        )

    router._call_anthropic = fake_anthropic  # type: ignore[method-assign]
    model = ModelConfig(
        provider="anthropic",
        model="test",
        api_key="test-key",
        max_tokens=10,
    )

    with pytest.raises(LLMBudgetError) as failed:
        await router.complete_json(
            [{"role": "user", "content": "choose"}],
            model,
            trace_context={"run_id": "total-usage"},
        )

    assert failed.value.budget_reason == "usage_exceeded_reservation"
    snapshot = ledger.snapshot("run:total-usage")
    assert snapshot is not None
    assert snapshot.input_tokens == 8
    assert snapshot.output_tokens == 5
    assert snapshot.total_tokens == 13
    await router.aclose()


def test_limit_environment_parsers_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEREWOLF_TEST_RATE", "nan")
    with pytest.raises(ValueError, match="positive finite"):
        config_module._positive_float_alias(("WEREWOLF_TEST_RATE",), "1")

    monkeypatch.setenv("WEREWOLF_TEST_BUDGET", "-1")
    with pytest.raises(ValueError, match="non-negative integer"):
        config_module._non_negative_int_alias(("WEREWOLF_TEST_BUDGET",), "0")

    monkeypatch.setenv("WEREWOLF_TEST_BUDGET", "0")
    assert config_module._non_negative_int_alias(("WEREWOLF_TEST_BUDGET",), "1") is None
