from __future__ import annotations

import asyncio

import pytest

import src.llm.router as router_module
from src.llm.models import ModelConfig
from src.llm.router import LLMError, LLMResponse, LLMRouter


class _FakeClient:
    instances: list["_FakeClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.close_calls = 0
        self.instances.append(self)

    async def close(self) -> None:
        self.close_calls += 1


def _config(key: str) -> ModelConfig:
    return ModelConfig(
        provider="openai",
        api_base="https://gateway.example.invalid/v1",
        api_key=key,
        model="test-model",
    )


@pytest.mark.asyncio
async def test_client_cache_uses_opaque_bounded_keys(monkeypatch):
    _FakeClient.instances.clear()
    monkeypatch.setattr(router_module.openai, "AsyncOpenAI", _FakeClient)
    router = LLMRouter(client_cache_size=2)
    secrets = [f"sk-cache-secret-{index}" for index in range(6)]

    try:
        for secret in secrets:
            router._get_openai_client(_config(secret), endpoint="chat")

        assert router.cached_client_count == 2
        assert len(router._openai_clients) == 2
        assert all(secret not in key for secret in secrets for key in router._openai_clients)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert sum(client.close_calls for client in _FakeClient.instances) == 4
        assert not router._client_close_scheduled
    finally:
        await router.aclose()

    assert sum(client.close_calls for client in _FakeClient.instances) == 6


@pytest.mark.asyncio
async def test_active_client_is_never_evicted_and_ephemeral_client_is_closed(monkeypatch):
    _FakeClient.instances.clear()
    monkeypatch.setattr(router_module.openai, "AsyncOpenAI", _FakeClient)
    router = LLMRouter(client_cache_size=1, max_retries=1)
    acquired = asyncio.Event()
    release = asyncio.Event()

    async def first_call():
        router._get_openai_client(_config("sk-active"), endpoint="chat")
        acquired.set()
        await release.wait()
        return LLMResponse(content="{}", finish_reason="stop")

    first = asyncio.create_task(router._await_provider_attempt(first_call(), owner="first"))
    try:
        await acquired.wait()

        async def second_call():
            router._get_openai_client(_config("sk-ephemeral"), endpoint="chat")
            return LLMResponse(content="{}", finish_reason="stop")

        await router._await_provider_attempt(second_call(), owner="second")
        assert len(_FakeClient.instances) == 2
        assert _FakeClient.instances[0].close_calls == 0
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert _FakeClient.instances[1].close_calls == 1
        assert not router._client_close_scheduled

        release.set()
        await first
        assert _FakeClient.instances[0].close_calls == 0
    finally:
        release.set()
        if not first.done():
            await first
        await router.aclose()


@pytest.mark.asyncio
async def test_failed_client_construction_still_closes_evicted_client(monkeypatch):
    _FakeClient.instances.clear()

    class FailingClient(_FakeClient):
        def __init__(self, **kwargs):
            if kwargs.get("api_key") == "sk-fail":
                raise RuntimeError("constructor failure")
            super().__init__(**kwargs)

    monkeypatch.setattr(router_module.openai, "AsyncOpenAI", FailingClient)
    router = LLMRouter(client_cache_size=1)
    try:
        first = router._get_openai_client(_config("sk-first"), endpoint="chat")
        with pytest.raises(RuntimeError, match="constructor failure"):
            router._get_openai_client(_config("sk-fail"), endpoint="chat")
        await asyncio.sleep(0)
        assert first.close_calls == 1
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_evicted_client_close_obeys_cleanup_deadline(monkeypatch):
    class SlowCloseClient(_FakeClient):
        async def close(self) -> None:
            self.close_calls += 1
            if self.kwargs.get("api_key") == "sk-slow-first":
                await asyncio.sleep(10)

    monkeypatch.setattr(router_module.openai, "AsyncOpenAI", SlowCloseClient)
    router = LLMRouter(
        client_cache_size=1,
        cleanup_timeout_seconds=0.01,
        cancellation_grace_seconds=0.01,
    )
    try:
        router._get_openai_client(_config("sk-slow-first"), endpoint="chat")
        router._get_openai_client(_config("sk-slow-second"), endpoint="chat")
        await asyncio.sleep(0.08)
        assert router.unresolved_task_count == 0
    finally:
        await router.aclose()


def test_client_cache_size_is_explicitly_bounded():
    with pytest.raises(ValueError):
        LLMRouter(client_cache_size=-1)
    with pytest.raises(ValueError):
        LLMRouter(client_cache_size=4097)
    with pytest.raises(ValueError):
        LLMRouter(client_cache_size=True)


@pytest.mark.asyncio
async def test_client_getters_reject_router_after_shutdown(monkeypatch):
    monkeypatch.setattr(router_module.openai, "AsyncOpenAI", _FakeClient)
    router = LLMRouter(client_cache_size=1)
    await router.aclose()
    with pytest.raises(LLMError, match="closing or closed"):
        router._get_openai_client(_config("sk-closed"), endpoint="chat")
