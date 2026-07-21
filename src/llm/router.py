"""LLM 调用层 —— 多 provider 真实调用,绝不伪造。

- 每个 AI 决策必须来自真实 LLM 调用,绝不 fallback 出假决策。
- 瞬时 transport/provider 失败在 Router 内有限重试；耗尽后抛 LLMError。
- OpenAI Chat/Responses 不传 max_tokens/max_completion_tokens/max_output_tokens;
  Anthropic Messages 按官方接口要求必须传 max_tokens,0 使用本项目默认上限。
- 宽松超时(默认 180s),不误杀真实思考。
- 凭据只来自 per-seat/房间 config(WEREWOLF_ 前缀),绝不回退读系统 OPENAI_/ANTHROPIC_ env。
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import random
import re
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx
import openai
from jsonschema import Draft202012Validator, SchemaError

from ..api.limits import (
    BudgetRecordResult,
    BudgetReservation,
    ProviderBudgetLedger,
    ProviderBudgetPolicy,
)
from .models import ModelConfig

logger = logging.getLogger(__name__)

# 可重试的 HTTP 状态码(瞬时错误,重试即可,不触发 skip)
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
ANTHROPIC_DEFAULT_MAX_TOKENS = 8192
MAX_TOOL_CALL_ID_CHARS = 256
MAX_TOOL_ARGUMENT_CHARS = 64 * 1024
STANDARD_PROTOCOLS = {"openai", "openai_responses", "anthropic"}
_TRACE_CONTEXT_KEYS = {
    "request_id",
    "run_id",
    "actor_id",
    "seat",
    "role",
    "day",
    "phase",
    "stage",
    "action",
    "budget_scope",
    "response_attempt",
}


@dataclass
class CallStats:
    """调用统计,供复盘/监控。"""

    calls: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency: float = 0.0
    structured_responses: int = 0
    incomplete_responses: int = 0
    response_parse_failures: int = 0
    response_parse_recoveries: int = 0
    lossy_parse_rejections: int = 0

    def record(self, *, ok: bool, retries: int, latency: float, tok_in: int = 0, tok_out: int = 0) -> None:
        self.calls += 1
        if ok:
            self.successes += 1
        else:
            self.failures += 1
        self.retries += retries
        self.total_latency += latency
        self.total_tokens_in += tok_in
        self.total_tokens_out += tok_out

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "retries": self.retries,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_latency": round(self.total_latency, 3),
            "avg_latency": round(self.total_latency / self.calls, 3) if self.calls else 0.0,
            "structured_responses": self.structured_responses,
            "incomplete_responses": self.incomplete_responses,
            "response_parse_failures": self.response_parse_failures,
            "response_parse_recoveries": self.response_parse_recoveries,
            "lossy_parse_rejections": self.lossy_parse_rejections,
        }


@dataclass
class LLMResponse:
    """一次完整调用的结果。"""

    content: str
    finish_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    raw_provider: str = ""
    latency: float = 0.0
    reasoning: str = ""  # provider 返回的私有 reasoning；只进入授权 decision trace。
    call_id: str = ""
    request_hash: str = ""
    transport_attempts: list[dict[str, Any]] = field(default_factory=list)
    # Function/tool calls are populated only by ``complete_tools``.  Keeping
    # this on the common response object lets the existing retry, budget, and
    # provenance machinery remain provider-neutral.
    tool_calls: tuple["LLMToolCall", ...] = ()


@dataclass(frozen=True)
class LLMToolCall:
    """One fully assembled function call emitted by a model stream."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str


@dataclass
class LLMToolResponse(LLMResponse):
    """Typed result returned by :meth:`LLMRouter.complete_tools`."""

    trace: dict[str, Any] = field(default_factory=dict)


# Short aliases are useful to callers that use the generic ``ToolCall`` name,
# while the explicit LLM-prefixed names remain the documented API.
ToolCall = LLMToolCall
ToolResponse = LLMToolResponse
ToolCallResult = LLMToolCall
LLMToolCallResult = LLMToolCall


@dataclass
class JSONParseResult:
    """JSON 解析结果及透明审计标记。"""

    data: dict[str, Any]
    method: str
    recovered: bool = False
    lossy: bool = False


class LLMError(RuntimeError):
    """LLM call failed after Router-owned retries; no Decision was produced."""


class LLMResponseError(LLMError):
    """Provider returned a response that cannot become a complete decision.

    Transport/provider retries belong to :class:`LLMRouter`. The actor may
    issue a fresh decision request only for this response-level failure class.
    """


class _ProviderStreamError(LLMError):
    """Structured provider error emitted as an SSE event.

    Some SDKs surface an SSE ``error``/``response.failed`` event as a normal
    stream object instead of raising ``APIError``.  Keep only status-relevant
    fields so Router retry classification and failure traces remain useful
    without retaining an arbitrary provider payload.
    """

    def __init__(self, message: str, *, payload: Any) -> None:
        super().__init__(message)
        self.body = _structured_error_payload(payload)


class LLMCallCleanupError(LLMError):
    """A provider attempt remained alive after bounded cancellation."""

    def __init__(self, message: str, *, pending_task_count: int = 1) -> None:
        super().__init__(message)
        self.timeout = True
        self.fatal_cleanup_failure = True
        self.pending_task_count = pending_task_count


class LLMRouterClosedError(LLMError):
    """The Router has stopped admitting provider work during shutdown."""


class LLMBudgetError(LLMError):
    """A provider call was denied or invalidated by a hard usage budget."""

    def __init__(self, message: str, *, reason: str, scope_id: str | None = None) -> None:
        super().__init__(message)
        self.budget_reason = reason
        self.budget_scope_id = scope_id


DEFAULT_CLIENT_CACHE_SIZE = 32
MAX_CLIENT_CACHE_SIZE = 4096


@dataclass
class _ClientCacheEntry:
    """Router-owned metadata kept separate from the SDK client object.

    The public-ish ``_openai_clients``/``_anthropic_clients`` maps deliberately
    continue to contain raw SDK clients for compatibility with integrations and
    tests.  Secrets never enter their keys; lifecycle state lives here instead.
    """

    family: str
    cache_key: str
    client: Any
    active_leases: int = 0
    last_used: float = field(default_factory=time.monotonic)
    cached: bool = True
    ephemeral: bool = False


class LLMRouter:
    """多 provider 路由器。

    openai 路径:OpenAI SDK Chat Completions 兼容格式。
    openai_responses 路径:OpenAI SDK Responses API 格式。
    anthropic 路径:Anthropic SDK Messages API 格式。
    凭据只来自 ModelConfig,绝不回退系统 env。
    """

    def __init__(
        self,
        *,
        timeout: float = 180.0,
        max_retries: int = 3,
        concurrency: int = 4,
        chunk_timeout: float = 60.0,
        cancellation_grace_seconds: float = 1.0,
        cleanup_timeout_seconds: float = 5.0,
        client_cache_size: int = DEFAULT_CLIENT_CACHE_SIZE,
        budget_ledger: ProviderBudgetLedger | None = None,
        provider_budget_ledger: ProviderBudgetLedger | None = None,
        budget_policy: ProviderBudgetPolicy | None = None,
        provider_budget_policy: ProviderBudgetPolicy | None = None,
    ) -> None:
        self.timeout = timeout
        # timeout 是每次 attempt 的 wall-clock 总时限;chunk_timeout 是两个
        # SSE chunk 之间的帧间超时。两者分开,避免 keepalive/空 delta 让
        # 一次真实流式调用无限挂住。
        self.chunk_timeout = chunk_timeout
        self.cancellation_grace_seconds = _bounded_duration(
            cancellation_grace_seconds,
            name="cancellation_grace_seconds",
            minimum=0.0,
            maximum=60.0,
        )
        self.cleanup_timeout_seconds = _bounded_duration(
            cleanup_timeout_seconds,
            name="cleanup_timeout_seconds",
            minimum=0.0,
            maximum=300.0,
            minimum_inclusive=False,
        )
        self.client_cache_size = _bounded_int(
            client_cache_size,
            name="client_cache_size",
            minimum=0,
            maximum=MAX_CLIENT_CACHE_SIZE,
        )
        self.max_retries = max(1, max_retries)
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._stats = CallStats()
        # 官方 SDK client 复用(按协议 + base_url + key 缓存)。
        self._openai_clients: dict[str, openai.AsyncOpenAI] = {}
        self._anthropic_clients: dict[str, anthropic.AsyncAnthropic] = {}
        self._active_call_tasks: set[asyncio.Future[Any]] = set()
        self._unresolved_call_tasks: dict[asyncio.Future[Any], str] = {}
        self._deferred_streams: dict[int, tuple[Any, str]] = {}
        # Cache keys are opaque fingerprints.  Lease metadata is keyed by the
        # same digest and never stores a provider credential.
        self._client_cache_entries: dict[tuple[str, str], _ClientCacheEntry] = {}
        self._task_client_leases: dict[asyncio.Future[Any], list[tuple[str, str]]] = {}
        self._client_close_tasks: set[asyncio.Future[Any]] = set()
        self._client_close_scheduled: set[int] = set()
        self._ephemeral_client_sequence = 0
        self._lifecycle_guard = threading.RLock()
        self._close_lock = asyncio.Lock()
        self._closing = False
        self._closed = False
        if budget_ledger is not None and provider_budget_ledger is not None and budget_ledger is not provider_budget_ledger:
            raise ValueError("provide budget_ledger or provider_budget_ledger, not both")
        if budget_policy is not None and provider_budget_policy is not None and budget_policy != provider_budget_policy:
            raise ValueError("provide budget_policy or provider_budget_policy, not both")
        selected_ledger = budget_ledger or provider_budget_ledger
        if selected_ledger is not None and not isinstance(selected_ledger, ProviderBudgetLedger):
            raise TypeError("budget_ledger must be a ProviderBudgetLedger")
        selected_policy = budget_policy or provider_budget_policy
        if selected_policy is not None and not isinstance(selected_policy, ProviderBudgetPolicy):
            raise TypeError("budget_policy must be a ProviderBudgetPolicy")
        self.budget_ledger = selected_ledger
        self.budget_policy = selected_policy

    @property
    def stats(self) -> CallStats:
        return self._stats

    @property
    def closing(self) -> bool:
        with self._lifecycle_guard:
            return self._closing

    @property
    def closed(self) -> bool:
        with self._lifecycle_guard:
            return self._closed

    @property
    def active_task_count(self) -> int:
        with self._lifecycle_guard:
            return sum(not task.done() for task in self._active_call_tasks)

    def _ensure_open(self) -> None:
        with self._lifecycle_guard:
            if self._closing or self._closed:
                raise LLMRouterClosedError("LLMRouter is closing or closed")
            pending = sum(
                not task.done() for task in self._unresolved_call_tasks
            )
            if pending:
                raise LLMCallCleanupError(
                    "LLMRouter has unresolved provider cleanup; new calls are paused",
                    pending_task_count=pending,
                )

    # ------------------------------------------------------------------
    # 客户端管理
    # ------------------------------------------------------------------
    def _sdk_timeout(self) -> httpx.Timeout:
        """SDK/httpx read timeout is per-frame; the Router owns wall time."""
        return httpx.Timeout(self.chunk_timeout, connect=15.0, write=15.0, pool=15.0)

    @staticmethod
    def _require_api_key(config: ModelConfig) -> str:
        key = (config.api_key or "").strip()
        if not key:
            raise LLMError("ModelConfig.api_key 为空: 本项目只使用房间/座位配置里的凭据,不读取系统环境变量")
        return key

    @property
    def cached_client_count(self) -> int:
        """Number of clients currently retained in the bounded shared cache."""
        with self._lifecycle_guard:
            return self._cached_entry_count_locked()

    def _client_cache_fingerprint(
        self,
        *,
        family: str,
        endpoint: str,
        base_url: str | None,
        api_key: str,
    ) -> str:
        # Hash the complete identity rather than concatenating the credential
        # into a dict key.  This keeps keys, URLs with userinfo, and provider
        # configuration out of long-lived Python objects and repr/debug dumps.
        return _hash_json({
            "family": family,
            "endpoint": endpoint,
            "base_url": base_url or "",
            "api_key": api_key,
        })

    def _client_map(self, family: str) -> dict[str, Any]:
        if family == "openai":
            return self._openai_clients
        if family == "anthropic":
            return self._anthropic_clients
        raise ValueError(f"unknown client family: {family}")

    def _cached_entry_count_locked(self) -> int:
        return len(self._openai_clients) + len(self._anthropic_clients)

    def _aggregate_client_leases_locked(self, client: Any) -> int:
        client_id = id(client)
        return sum(
            entry.active_leases
            for entry in self._client_cache_entries.values()
            if id(entry.client) == client_id
        )

    def _oldest_idle_entry_locked(self) -> tuple[str, str] | None:
        candidates: list[tuple[float, str, str]] = []
        for family, cache in (
            ("openai", self._openai_clients),
            ("anthropic", self._anthropic_clients),
        ):
            for cache_key, client in cache.items():
                entry_key = (family, cache_key)
                entry = self._client_cache_entries.get(entry_key)
                active = self._aggregate_client_leases_locked(client)
                if active:
                    continue
                # Entries inserted by older integrations/tests may not have
                # side metadata; treat those as idle and oldest.
                last_used = entry.last_used if entry is not None else 0.0
                candidates.append((last_used, family, cache_key))
        if not candidates:
            return None
        _last_used, family, cache_key = min(candidates)
        return family, cache_key

    def _remove_cached_entry_locked(self, entry_key: tuple[str, str]) -> Any | None:
        family, cache_key = entry_key
        cache = self._client_map(family)
        client = cache.pop(cache_key, None)
        entry = self._client_cache_entries.pop(entry_key, None)
        if client is None and entry is not None:
            client = entry.client
        if entry is not None:
            entry.cached = False
        return client

    def _lease_entry_locked(self, entry: _ClientCacheEntry) -> None:
        """Associate a cached client with the provider task that acquired it."""
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is None or task not in self._active_call_tasks:
            # Direct cache inspection/construction is not a provider lease.
            # Real calls are always admitted through _await_provider_attempt.
            return
        entry.active_leases += 1
        self._task_client_leases.setdefault(task, []).append(
            (entry.family, entry.cache_key)
        )

    def _get_cached_client(
        self,
        *,
        family: str,
        cache_key: str,
        factory: Any,
    ) -> Any:
        """Return a shared client or a bounded-lifetime ephemeral client.

        This method is synchronous because SDK resource lookup happens inside
        provider coroutines and existing callers monkeypatch ``_get_*``. Idle
        evictions are removed atomically, then their async close is scheduled;
        no active lease is ever evicted.
        """
        entry_key = (family, cache_key)
        cache = self._client_map(family)
        evicted: list[Any] = []
        with self._lifecycle_guard:
            client = cache.get(cache_key)
            if client is not None:
                entry = self._client_cache_entries.get(entry_key)
                if entry is None or entry.client is not client:
                    entry = _ClientCacheEntry(
                        family=family,
                        cache_key=cache_key,
                        client=client,
                    )
                    self._client_cache_entries[entry_key] = entry
                entry.last_used = time.monotonic()
                self._lease_entry_locked(entry)
                return client

            while self.client_cache_size > 0 and (
                self._cached_entry_count_locked() >= self.client_cache_size
            ):
                oldest = self._oldest_idle_entry_locked()
                if oldest is None:
                    break
                removed = self._remove_cached_entry_locked(oldest)
                if removed is not None:
                    evicted.append(removed)

            try:
                client = factory()
            except BaseException:
                # The room may rotate credentials while the SDK constructor is
                # unavailable.  Evictions already detached above still need a
                # close attempt before the constructor error escapes.
                for old_client in evicted:
                    self._schedule_client_close(old_client, owner="cache_eviction")
                raise
            can_cache = (
                self.client_cache_size > 0
                and self._cached_entry_count_locked() < self.client_cache_size
            )
            if can_cache:
                cache[cache_key] = client
                entry = _ClientCacheEntry(
                    family=family,
                    cache_key=cache_key,
                    client=client,
                )
            else:
                self._ephemeral_client_sequence += 1
                ephemeral_key = f"ephemeral:{self._ephemeral_client_sequence}"
                entry_key = (family, ephemeral_key)
                entry = _ClientCacheEntry(
                    family=family,
                    cache_key=ephemeral_key,
                    client=client,
                    cached=False,
                    ephemeral=True,
                )
            self._client_cache_entries[entry_key] = entry
            self._lease_entry_locked(entry)

        for old_client in evicted:
            self._schedule_client_close(old_client, owner="cache_eviction")
        return client

    def _get_openai_client(self, config: ModelConfig, *, endpoint: str) -> openai.AsyncOpenAI:
        self._ensure_open()
        base_url = _openai_base_url(config.api_base, endpoint=endpoint)
        api_key = self._require_api_key(config)
        cache_key = self._client_cache_fingerprint(
            family="openai",
            endpoint=endpoint,
            base_url=base_url,
            api_key=api_key,
        )

        def factory() -> openai.AsyncOpenAI:
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": self._sdk_timeout(),
                "max_retries": 0,
            }
            if base_url:
                kwargs["base_url"] = base_url
            return openai.AsyncOpenAI(**kwargs)

        return self._get_cached_client(
            family="openai",
            cache_key=cache_key,
            factory=factory,
        )

    async def _get_openai_resource(self, client: openai.AsyncOpenAI, *, endpoint: str) -> Any:
        """Resolve SDK lazy resources without starving the attempt deadline."""
        if endpoint == "chat":
            resolver = lambda: client.chat.completions
        elif endpoint == "responses":
            resolver = lambda: client.responses
        else:
            raise ValueError(f"unknown OpenAI endpoint: {endpoint}")

        # The first property access imports the generated OpenAI resource/type
        # tree synchronously. Under load that can occupy the event loop for
        # seconds before ``create()`` reaches its first await, preventing the
        # Router's wall-clock timer from running at all.
        return await asyncio.to_thread(resolver)

    def _get_anthropic_client(self, config: ModelConfig) -> anthropic.AsyncAnthropic:
        self._ensure_open()
        base_url = _anthropic_base_url(config.api_base)
        api_key = self._require_api_key(config)
        cache_key = self._client_cache_fingerprint(
            family="anthropic",
            endpoint="messages",
            base_url=base_url,
            api_key=api_key,
        )

        def factory() -> anthropic.AsyncAnthropic:
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "timeout": self._sdk_timeout(),
                "max_retries": 0,
            }
            if base_url:
                kwargs["base_url"] = base_url
            return anthropic.AsyncAnthropic(**kwargs)

        return self._get_cached_client(
            family="anthropic",
            cache_key=cache_key,
            factory=factory,
        )

    def _release_task_client_leases(self, task: asyncio.Future[Any]) -> None:
        """Release all clients acquired by a provider task.

        This callback is also used for tasks that ignored cancellation.  Such
        a task keeps its client lease until it really finishes, so cache
        eviction and Router shutdown cannot close an in-use HTTP pool.
        """
        to_close: list[Any] = []
        with self._lifecycle_guard:
            lease_keys = self._task_client_leases.pop(task, [])
            for entry_key in lease_keys:
                entry = self._client_cache_entries.get(entry_key)
                if entry is None:
                    continue
                entry.active_leases = max(0, entry.active_leases - 1)
                entry.last_used = time.monotonic()
                family, cache_key = entry_key
                cached_client = self._client_map(family).get(cache_key)
                if entry.active_leases == 0 and (
                    entry.ephemeral or cached_client is not entry.client
                ):
                    self._client_cache_entries.pop(entry_key, None)
                    to_close.append(entry.client)
        for client in to_close:
            self._schedule_client_close(client, owner="client_lease_release")

    def _schedule_client_close(self, client: Any, *, owner: str) -> None:
        """Start one best-effort async close and retain it for shutdown drain."""
        if client is None:
            return
        client_id = id(client)
        with self._lifecycle_guard:
            if client_id in self._client_close_scheduled:
                return
            self._client_close_scheduled.add(client_id)
        close = getattr(client, "close", None)
        if not callable(close):
            with self._lifecycle_guard:
                self._client_close_scheduled.discard(client_id)
            logger.warning("LLM client has no close method (owner=%s)", owner)
            return
        try:
            value = close()
        except Exception:  # noqa: BLE001 - provider resource is untrusted
            with self._lifecycle_guard:
                self._client_close_scheduled.discard(client_id)
            logger.warning("LLM client close could not start (owner=%s)", owner)
            return
        if not inspect.isawaitable(value):
            with self._lifecycle_guard:
                self._client_close_scheduled.discard(client_id)
            return
        try:
            close_task = asyncio.ensure_future(value)
        except Exception:  # noqa: BLE001 - provider resource is untrusted
            _dispose_unstarted_awaitable(value)
            with self._lifecycle_guard:
                self._client_close_scheduled.discard(client_id)
            logger.warning("LLM client close task could not start (owner=%s)", owner)
            return
        _set_task_name(close_task, f"llm-client-close-inner:{owner}")

        def clear_close_marker(done: asyncio.Future[Any]) -> None:
            with self._lifecycle_guard:
                self._client_close_scheduled.discard(client_id)
            _consume_task_result(done)

        close_task.add_done_callback(clear_close_marker)
        runner = self._run_scheduled_client_close(close_task, owner=owner)
        try:
            task = asyncio.ensure_future(runner)
        except Exception:  # noqa: BLE001 - provider resource is untrusted
            _dispose_unstarted_awaitable(runner)
            close_task.cancel()
            with self._lifecycle_guard:
                self._client_close_scheduled.discard(client_id)
            logger.warning("LLM client close task could not start (owner=%s)", owner)
            return
        _set_task_name(task, f"llm-client-close:{owner}")
        with self._lifecycle_guard:
            self._client_close_tasks.add(task)

        def forget(done: asyncio.Future[Any]) -> None:
            with self._lifecycle_guard:
                self._client_close_tasks.discard(done)
                self._client_close_scheduled.discard(client_id)
            _consume_task_result(done)

        task.add_done_callback(forget)

    async def _run_scheduled_client_close(self, value: Any, *, owner: str) -> None:
        """Bound an eviction/lease-release close even before Router shutdown."""
        try:
            close_task = asyncio.ensure_future(value)
        except Exception:  # noqa: BLE001 - provider resource is untrusted
            _dispose_unstarted_awaitable(value)
            logger.warning("LLM client close awaitable was invalid (owner=%s)", owner)
            return
        _set_task_name(close_task, f"llm-client-close-inner:{owner}")
        try:
            done, pending = await asyncio.wait(
                {close_task},
                timeout=self.cleanup_timeout_seconds,
            )
        except asyncio.CancelledError:
            pending, _interrupted = await _cancel_tasks_bounded(
                [close_task],
                self.cancellation_grace_seconds,
            )
            for task in pending:
                self._track_unresolved_call(task, f"{owner}_client_close")
            raise
        if pending:
            still_pending, caller_cancelled = await _cancel_tasks_bounded(
                pending,
                self.cancellation_grace_seconds,
            )
            for task in still_pending:
                self._track_unresolved_call(task, f"{owner}_client_close")
            logger.warning("LLM client close exceeded cleanup timeout (owner=%s)", owner)
            if caller_cancelled:
                raise asyncio.CancelledError
            return
        try:
            next(iter(done)).result()
        except asyncio.CancelledError:
            logger.warning("LLM client close was cancelled (owner=%s)", owner)
        except Exception:  # noqa: BLE001 - never expose provider detail
            logger.warning("LLM client close raised an exception (owner=%s)", owner)

    async def _drain_scheduled_client_closes(self) -> tuple[list[str], bool]:
        """Wait for evicted/ephemeral client closes under the cleanup budget."""
        with self._lifecycle_guard:
            tasks = [task for task in self._client_close_tasks if not task.done()]
        if not tasks:
            return [], False
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.cleanup_timeout_seconds,
            )
        except asyncio.CancelledError:
            pending, interrupted = await _cancel_tasks_bounded(
                tasks,
                self.cancellation_grace_seconds,
            )
            for task in pending:
                self._track_unresolved_call(task, "client_close")
            raise
        failures: list[str] = []
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                failures.append("client_close task was cancelled")
            except Exception:  # noqa: BLE001 - never expose provider detail
                failures.append("client_close task raised an exception")
        if pending:
            still_pending, caller_cancelled = await _cancel_tasks_bounded(
                pending,
                self.cancellation_grace_seconds,
            )
            failures.append(f"{len(pending)} client_close task(s) exceeded cleanup timeout")
            for task in still_pending:
                self._track_unresolved_call(task, "client_close")
            return failures, caller_cancelled
        return failures, False

    async def aclose(self) -> None:
        """Stop admissions, reclaim provider work, and close every SDK resource."""
        async with self._close_lock:
            with self._lifecycle_guard:
                if self._closed:
                    return
                self._closing = True
            try:
                await self._aclose_resources()
            finally:
                # A failed close is terminal. Reusing a Router whose transport
                # cleanup was incomplete could create calls beside leaked work.
                with self._lifecycle_guard:
                    self._closing = False
                    self._closed = True

    async def _aclose_resources(self) -> None:
        failures: list[str] = []
        caller_cancelled = False
        with self._lifecycle_guard:
            active = [task for task in self._active_call_tasks if not task.done()]
        active_pending, interrupted = await _cancel_tasks_bounded(
            active,
            self.cancellation_grace_seconds,
        )
        caller_cancelled = caller_cancelled or interrupted
        for task in active_pending:
            self._track_unresolved_call(task, "router_close_active_call")

        unresolved = [
            task for task in self._unresolved_call_tasks if not task.done()
        ]
        pending, interrupted = await _cancel_tasks_bounded(
            unresolved,
            self.cancellation_grace_seconds,
        )
        caller_cancelled = caller_cancelled or interrupted
        if pending:
            failures.append(f"{len(pending)} provider task(s) still pending")

        deferred_streams = list(self._deferred_streams.values())
        self._deferred_streams.clear()
        if deferred_streams:
            stream_failures, interrupted = await self._close_resources_bounded(
                [stream for stream, _owner in deferred_streams],
                owner="stream_close",
            )
            failures.extend(stream_failures)
            caller_cancelled = caller_cancelled or interrupted

        # Detach cached entries first.  Entries leased by a provider task are
        # deliberately left in side metadata; their done callback closes them
        # only after the underlying task has really stopped.
        with self._lifecycle_guard:
            clients_by_id = {
                id(client): client
                for client in (
                    *self._openai_clients.values(),
                    *self._anthropic_clients.values(),
                )
            }
            for entry in self._client_cache_entries.values():
                entry.cached = False
                clients_by_id.setdefault(id(entry.client), entry.client)
            self._openai_clients.clear()
            self._anthropic_clients.clear()
            closable_clients = [
                client
                for client in clients_by_id.values()
                if self._aggregate_client_leases_locked(client) == 0
            ]
            closable_ids = {id(client) for client in closable_clients}
            for entry_key, entry in list(self._client_cache_entries.items()):
                if id(entry.client) in closable_ids:
                    self._client_cache_entries.pop(entry_key, None)
            # Preserve the in-flight marker for clients already being closed by
            # an eviction task. New shutdown closes have no remaining lease and
            # therefore need no marker once their map entry is detached.
            already_scheduled_ids = set(self._client_close_scheduled)
            clients_to_start = [
                client
                for client in closable_clients
                if id(client) not in already_scheduled_ids
            ]
        client_failures, interrupted = await self._close_resources_bounded(
            clients_to_start,
            owner="client_close",
        )
        failures.extend(client_failures)
        caller_cancelled = caller_cancelled or interrupted

        close_failures, interrupted = await self._drain_scheduled_client_closes()
        failures.extend(close_failures)
        caller_cancelled = caller_cancelled or interrupted

        remaining = self.unresolved_task_count
        if remaining:
            failures.append(f"{remaining} task(s) remain in-process")
        if caller_cancelled:
            raise asyncio.CancelledError
        if failures:
            raise LLMCallCleanupError(
                "LLMRouter cleanup failed: " + "; ".join(failures),
                pending_task_count=remaining,
            )

    async def _close_resources_bounded(
        self,
        resources: list[Any],
        *,
        owner: str,
    ) -> tuple[list[str], bool]:
        """Close SDK resources under the router-wide cleanup budget."""
        tasks: list[asyncio.Future[Any]] = []
        failures: list[str] = []
        for index, resource in enumerate(resources):
            close = getattr(resource, "close", None)
            if not callable(close):
                failures.append(f"{owner} resource has no close method")
                continue
            try:
                task = asyncio.ensure_future(close())
            except Exception:  # noqa: BLE001 - provider resource is untrusted
                failures.append(f"{owner} resource close could not start")
                continue
            _set_task_name(task, f"llm-{owner}:{index}")
            tasks.append(task)
        if not tasks:
            return failures, False
        try:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.cleanup_timeout_seconds,
            )
        except asyncio.CancelledError:
            pending, _interrupted = await _cancel_tasks_bounded(
                tasks,
                self.cancellation_grace_seconds,
            )
            for task in pending:
                self._track_unresolved_call(task, owner)
            raise
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                failures.append(f"{owner} task was cancelled")
            except Exception:  # noqa: BLE001 - never expose provider detail
                failures.append(f"{owner} task raised an exception")
        caller_cancelled = False
        if pending:
            initially_pending = set(pending)
            still_pending, caller_cancelled = await _cancel_tasks_bounded(
                pending,
                self.cancellation_grace_seconds,
            )
            failures.append(
                f"{len(initially_pending)} {owner} task(s) exceeded cleanup timeout"
            )
            for task in still_pending:
                self._track_unresolved_call(task, owner)
        return failures, caller_cancelled

    # ------------------------------------------------------------------
    # 主入口:结构化 JSON 调用(agent 决策用)
    # ------------------------------------------------------------------
    async def complete_json(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
        *,
        system: str | None = None,
        schema_hint: str | None = None,
        allow_lossy: bool = False,
        include_parse_metadata: bool = False,
        trace_context: dict[str, Any] | None = None,
        budget_scope: str | None = None,
    ) -> dict[str, Any]:
        """调用 LLM 并解析为 JSON dict。

        use_json_format=True 时要求 response_format=json_object(网关支持则用,降级则靠解析)。
        失败抛 LLMError，由 DecisionRuntime 记录无-envelope 失败终态；绝不返回伪造结果或 SKIP。
        默认拒绝有损 JSON 恢复,避免截断输出被静默当成完整决策。
        """
        try:
            resp = await self._complete(
                messages,
                config,
                system=system,
                schema_hint=schema_hint,
                json_mode=config.use_json_format,
                trace_context=trace_context,
                budget_scope=budget_scope,
            )
        except LLMError as err:
            _attach_trace_context(err, trace_context)
            raise
        self._stats.structured_responses += 1
        try:
            self._ensure_complete_finish(resp, config)
        except LLMResponseError as err:
            self._stats.incomplete_responses += 1
            _attach_response_trace(
                err,
                messages=messages,
                system=system,
                schema_hint=schema_hint,
                config=config,
                response=resp,
                parse=None,
                context=trace_context,
            )
            raise
        try:
            parse = self._parse_json_result(resp.content, config)
        except LLMResponseError as err:
            self._stats.response_parse_failures += 1
            _attach_response_trace(
                err,
                messages=messages,
                system=system,
                schema_hint=schema_hint,
                config=config,
                response=resp,
                parse=None,
                context=trace_context,
            )
            raise
        if parse.lossy and not allow_lossy:
            self._stats.lossy_parse_rejections += 1
            err = LLMResponseError(
                "JSON 有损恢复被拒绝"
                f"(provider={config.provider} method={parse.method}): {resp.content[:300]!r}"
            )
            _attach_response_trace(
                err,
                messages=messages,
                system=system,
                schema_hint=schema_hint,
                config=config,
                response=resp,
                parse=parse,
                context=trace_context,
            )
            raise err
        if parse.recovered:
            self._stats.response_parse_recoveries += 1
        data = dict(parse.data)
        if include_parse_metadata:
            data["_parse_recovered"] = parse.recovered
            data["_parse_lossy"] = parse.lossy
            data["_parse_method"] = parse.method
        if trace_context is not None:
            data["_llm_call_trace"] = _llm_call_trace(
                messages=messages,
                system=system,
                schema_hint=schema_hint,
                config=config,
                response=resp,
                parse=parse,
                context=trace_context,
            )
        if resp.reasoning and not str(data.get("thought") or "").strip():
            data["thought"] = resp.reasoning
        return data

    async def complete_tools(
        self,
        messages: list[dict[str, Any]],
        config: ModelConfig,
        tools: list[dict[str, Any]],
        *,
        tool_choice: Any = "auto",
        parallel_tool_calls: bool | None = None,
        system: str | None = None,
        trace_context: dict[str, Any] | None = None,
        budget_scope: str | None = None,
    ) -> LLMToolResponse:
        """Run one streamed, provider-neutral function/tool turn.

        ``tools`` uses the standard OpenAI function-tool shape.  The router
        validates and normalizes it once, then translates it to the selected
        standard protocol.  No provider-specific endpoint or model names are
        consulted.  Tool arguments are parsed strictly as JSON objects; a
        partial, conflicting, or malformed stream raises
        :class:`LLMResponseError` and never yields an executable call.
        """
        normalized_tools = _normalize_tool_definitions(tools)
        normalized_choice = _normalize_tool_choice(tool_choice, normalized_tools)
        if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
            raise LLMResponseError("parallel_tool_calls must be a boolean or null")

        try:
            response = await self._complete(
                messages,
                config,
                system=system,
                trace_context=trace_context,
                budget_scope=budget_scope,
                tools=normalized_tools,
                tool_choice=normalized_choice,
                parallel_tool_calls=parallel_tool_calls,
            )
        except LLMError as err:
            _attach_trace_context(err, trace_context)
            raise

        try:
            _ensure_tool_finish(response, config)
        except LLMResponseError as err:
            _attach_tool_response_trace(
                err,
                messages=messages,
                system=system,
                config=config,
                response=response,
                tools=normalized_tools,
                tool_choice=normalized_choice,
                parallel_tool_calls=parallel_tool_calls,
                context=trace_context,
            )
            raise

        trace = _tool_call_trace(
            messages=messages,
            system=system,
            config=config,
            response=response,
            tools=normalized_tools,
            tool_choice=normalized_choice,
            parallel_tool_calls=parallel_tool_calls,
            context=trace_context or {},
        )
        return LLMToolResponse(
            content=response.content,
            finish_reason=response.finish_reason,
            usage=dict(response.usage),
            raw_provider=response.raw_provider,
            latency=response.latency,
            reasoning=response.reasoning,
            call_id=response.call_id,
            request_hash=response.request_hash,
            transport_attempts=[dict(row) for row in response.transport_attempts],
            tool_calls=tuple(response.tool_calls),
            trace=trace,
        )

    # ------------------------------------------------------------------
    # 核心调用(带重试)
    # ------------------------------------------------------------------
    async def _complete(
        self,
        messages: list[dict[str, Any]],
        config: ModelConfig,
        *,
        system: str | None = None,
        schema_hint: str | None = None,
        json_mode: bool = False,
        trace_context: dict[str, Any] | None = None,
        budget_scope: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        parallel_tool_calls: bool | None = None,
    ) -> LLMResponse:
        self._ensure_open()
        last_err: Exception | None = None
        retries = 0
        start = time.monotonic()
        request_view = _llm_request_view(
            messages,
            system=system,
            schema_hint=schema_hint,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        request_hash = _hash_json(request_view)
        call_id = _hash_text(f"{time.time_ns()}:{request_hash}")[:16]
        attempts: list[dict[str, Any]] = []
        resolved_budget_scope = self._resolve_budget_scope(
            trace_context,
            explicit_scope=budget_scope,
        )

        async with self._sem:
            for attempt in range(self.max_retries):
                attempt_started = time.monotonic()
                reservation: BudgetReservation | None = None
                transport_started = False
                try:
                    self._ensure_open()
                    protocol = _standard_protocol(config)
                    if self.budget_ledger is not None and resolved_budget_scope is None:
                        raise LLMBudgetError(
                            "provider usage budget requires an explicit run or room scope",
                            reason="budget_scope_required",
                        )
                    if self.budget_ledger is not None and resolved_budget_scope is not None:
                        token_budget = _config_token_budget(config)
                        reservation_decision = self.budget_ledger.try_reserve(
                            resolved_budget_scope,
                            token_budget=token_budget,
                            policy=self.budget_policy,
                        )
                        if not reservation_decision.allowed or reservation_decision.reservation is None:
                            raise LLMBudgetError(
                                "provider usage budget rejected the call "
                                f"(reason={reservation_decision.reason})",
                                reason=reservation_decision.reason,
                                scope_id=resolved_budget_scope,
                            )
                        reservation = reservation_decision.reservation
                    if tools is not None and protocol == "anthropic":
                        call = self._call_anthropic_tools(
                            messages, config, system, tools, tool_choice, parallel_tool_calls
                        )
                    elif tools is not None and protocol == "openai_responses":
                        call = self._call_openai_responses_tools(
                            messages, config, system, tools, tool_choice, parallel_tool_calls
                        )
                    elif tools is not None:
                        call = self._call_openai_tools(
                            messages, config, system, tools, tool_choice, parallel_tool_calls
                        )
                    elif protocol == "anthropic":
                        call = self._call_anthropic(messages, config, system, schema_hint, json_mode)
                    elif protocol == "openai_responses":
                        call = self._call_openai_responses(messages, config, system, schema_hint, json_mode)
                    else:
                        call = self._call_openai(messages, config, system, schema_hint, json_mode)
                    # Calling the adapter creates the transport coroutine. Any
                    # exception after this point is conservatively charged as
                    # an attempted provider call; only synchronous adapter
                    # construction failures release the reservation.
                    transport_started = True
                    try:
                        resp = await self._await_provider_attempt(
                            call,
                            owner=f"{protocol}:{call_id}:attempt:{attempt + 1}",
                        )
                    except asyncio.TimeoutError as err:
                        raise LLMError(
                            f"LLM 调用总超时(provider={config.provider} >{self.timeout:.1f}s)"
                        ) from err
                    latency = time.monotonic() - start
                    resp.latency = latency
                    budget_record = self._record_budget_success(
                        reservation,
                        response=resp,
                        scope_id=resolved_budget_scope,
                    )
                    reservation = None
                    if budget_record is not None and not budget_record.accepted:
                        raise LLMBudgetError(
                            "provider usage exceeded the reserved budget "
                            f"(reason={budget_record.reason})",
                            reason=budget_record.reason,
                            scope_id=resolved_budget_scope,
                        )
                    attempts.append({
                        "attempt": attempt + 1,
                        "status": "succeeded",
                        "latency_seconds": round(time.monotonic() - attempt_started, 6),
                        "retryable": False,
                        "will_retry": False,
                    })
                    if budget_record is not None:
                        attempts[-1].update({
                            "budget_scope": resolved_budget_scope,
                            "budget_scope_total_tokens": budget_record.snapshot.total_tokens,
                            "budget_reason": budget_record.reason,
                        })
                    resp.call_id = call_id
                    resp.request_hash = request_hash
                    resp.transport_attempts = [dict(row) for row in attempts]
                    self._stats.record(
                        ok=True,
                        retries=retries,
                        latency=latency,
                        tok_in=resp.usage.get("prompt_tokens", 0),
                        tok_out=resp.usage.get("completion_tokens", 0),
                    )
                    return resp
                except asyncio.CancelledError:
                    self._finalize_budget_failure(
                        reservation,
                        transport_started=transport_started,
                    )
                    raise
                except Exception as err:  # noqa: BLE001 — 统一重试
                    self._finalize_budget_failure(
                        reservation,
                        transport_started=(
                            transport_started
                            and not isinstance(err, LLMRouterClosedError)
                        ),
                    )
                    last_err = err
                    retryable = self._is_retryable(err)
                    will_retry = attempt < self.max_retries - 1 and retryable
                    attempt_row = {
                        "attempt": attempt + 1,
                        "status": "budget_rejected" if isinstance(err, LLMBudgetError) else "failed",
                        "latency_seconds": round(time.monotonic() - attempt_started, 6),
                        "error_type": type(err).__name__,
                        "timeout": _is_timeout_error(err),
                        "retryable": retryable,
                        "will_retry": will_retry,
                    }
                    if isinstance(err, LLMBudgetError):
                        attempt_row["budget_reason"] = err.budget_reason
                        if err.budget_scope_id:
                            attempt_row["budget_scope"] = err.budget_scope_id
                    status_code = _safe_status_code(err)
                    if status_code is not None:
                        attempt_row["status_code"] = status_code
                    if will_retry:
                        retries += 1
                        delay = self._backoff_delay(attempt)
                        attempt_row["backoff_seconds"] = round(delay, 6)
                        attempts.append(attempt_row)
                        logger.warning(
                            "LLM 调用失败(provider=%s attempt=%d/%d %.2fs后重试 error_type=%s)",
                            config.provider,
                            attempt + 1,
                            self.max_retries,
                            delay,
                            type(err).__name__,
                        )
                        await asyncio.sleep(delay)
                        continue
                    attempts.append(attempt_row)
                    break  # 不可重试或重试耗尽

        self._stats.record(ok=False, retries=retries, latency=time.monotonic() - start)
        trace = _failed_llm_call_trace(
            call_id=call_id,
            request_hash=request_hash,
            config=config,
            attempts=attempts,
            elapsed_seconds=time.monotonic() - start,
        )
        if isinstance(last_err, LLMError):
            setattr(last_err, "llm_call_trace", trace)
            raise last_err
        failure = LLMError(
            "LLM 调用彻底失败"
            f"(provider={config.provider} retries={retries} error_type={type(last_err).__name__})"
        )
        setattr(failure, "llm_call_trace", trace)
        raise failure from last_err

    def _resolve_budget_scope(
        self,
        trace_context: dict[str, Any] | None,
        *,
        explicit_scope: str | None,
    ) -> str | None:
        if self.budget_ledger is None:
            return None
        raw = explicit_scope
        if raw is None and isinstance(trace_context, dict):
            raw = trace_context.get("budget_scope") or trace_context.get("run_id")
        if raw is None:
            return None
        value = str(raw).strip()
        if not value:
            return None
        if value.startswith("room:") or value.startswith("run:"):
            return value
        return self.budget_ledger.run_scope(value)

    def _record_budget_success(
        self,
        reservation: BudgetReservation | None,
        *,
        response: LLMResponse,
        scope_id: str | None,
    ) -> BudgetRecordResult | None:
        if reservation is None or self.budget_ledger is None:
            return None
        return self.budget_ledger.record(
            reservation,
            input_tokens=_usage_value(response.usage, "prompt_tokens"),
            output_tokens=_usage_value(response.usage, "completion_tokens"),
        )

    def _finalize_budget_failure(
        self,
        reservation: BudgetReservation | None,
        *,
        transport_started: bool,
    ) -> None:
        if reservation is None or self.budget_ledger is None:
            return
        try:
            if transport_started:
                # Failed/aborted provider attempts are charged with unknown
                # usage.  The ledger blocks future calls when a hard token
                # limit is configured, rather than guessing a token amount.
                self.budget_ledger.record(
                    reservation,
                    input_tokens=None,
                    output_tokens=None,
                )
            else:
                self.budget_ledger.cancel(reservation)
        except Exception as err:  # noqa: BLE001 - preserve the original provider error
            logger.error("provider budget finalization failed error_type=%s", type(err).__name__)

    async def _await_provider_attempt(self, call: Any, *, owner: str) -> LLMResponse:
        """Apply an attempt deadline without an unbounded cancellation wait."""
        if not inspect.isawaitable(call):
            raise LLMError("provider adapter must return an awaitable")
        with self._lifecycle_guard:
            if self._closing or self._closed:
                _dispose_unstarted_awaitable(call)
                raise LLMRouterClosedError("LLMRouter is closing or closed")
            task = asyncio.ensure_future(call)
            self._active_call_tasks.add(task)
        _set_task_name(task, f"llm-call:{owner}")
        try:
            try:
                if not self.timeout or self.timeout <= 0:
                    return await asyncio.shield(task)
                done, _pending = await asyncio.wait({task}, timeout=self.timeout)
            except asyncio.CancelledError:
                terminated, _interrupted = await self._cancel_call_task(task)
                if not terminated:
                    self._track_unresolved_call(task, owner)
                raise
            if task in done:
                return task.result()

            terminated, caller_cancelled = await self._cancel_call_task(task)
            if not terminated:
                self._track_unresolved_call(task, owner)
            if caller_cancelled:
                cancelled = asyncio.CancelledError()
                if not terminated:
                    setattr(cancelled, "cleanup_pending_task_count", 1)
                raise cancelled
            if not terminated:
                raise LLMCallCleanupError(
                    "LLM provider attempt ignored cancellation after its wall-clock timeout"
                )
            raise asyncio.TimeoutError
        finally:
            # A provider task that ignored cancellation still owns its SDK
            # client.  Release immediately only when it is done; otherwise the
            # callback runs at the actual task completion boundary.
            if task.done():
                self._release_task_client_leases(task)
            else:
                task.add_done_callback(self._release_task_client_leases)
            with self._lifecycle_guard:
                self._active_call_tasks.discard(task)

    async def _cancel_call_task(
        self,
        task: asyncio.Future[Any],
    ) -> tuple[bool, bool]:
        pending, interrupted = await _cancel_tasks_bounded(
            [task],
            self.cancellation_grace_seconds,
        )
        return not pending, interrupted

    def _defer_stream_close(self, stream: Any, *, owner: str) -> None:
        """Register stream teardown without delaying provider cancellation."""
        if stream is None or not callable(getattr(stream, "close", None)):
            return
        self._deferred_streams.setdefault(id(stream), (stream, owner))

    async def _close_stream_after_iteration(self, stream: Any, *, owner: str) -> None:
        """Close an acquired stream promptly without an unbounded cleanup wait."""
        close = getattr(stream, "close", None)
        if not callable(close):
            return
        try:
            value = close()
        except Exception:  # noqa: BLE001 - provider stream is untrusted
            self._defer_stream_close(stream, owner=owner)
            logger.warning("LLM stream close could not start (owner=%s)", owner)
            return
        if not inspect.isawaitable(value):
            return
        task = asyncio.ensure_future(value)
        _set_task_name(task, f"llm-stream-close:{owner}")
        timeout = max(
            0.05,
            min(self.cleanup_timeout_seconds, self.cancellation_grace_seconds or 0.05),
        )
        try:
            done, pending = await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            pending, _interrupted = await _cancel_tasks_bounded(
                [task],
                self.cancellation_grace_seconds,
            )
            for unfinished in pending:
                self._track_unresolved_call(unfinished, owner)
            self._defer_stream_close(stream, owner=owner)
            raise
        if pending:
            still_pending, caller_cancelled = await _cancel_tasks_bounded(
                pending,
                self.cancellation_grace_seconds,
            )
            for unfinished in still_pending:
                self._track_unresolved_call(unfinished, owner)
            self._defer_stream_close(stream, owner=owner)
            logger.warning("LLM stream close exceeded cleanup timeout (owner=%s)", owner)
            if caller_cancelled:
                raise asyncio.CancelledError
            return
        try:
            next(iter(done)).result()
        except asyncio.CancelledError:
            self._defer_stream_close(stream, owner=owner)
            raise
        except Exception:  # noqa: BLE001 - never expose provider detail
            self._defer_stream_close(stream, owner=owner)
            logger.warning("LLM stream close raised an exception (owner=%s)", owner)

    def _track_unresolved_call(self, task: asyncio.Future[Any], owner: str) -> None:
        if task.done():
            _consume_task_result(task)
            return
        self._unresolved_call_tasks[task] = owner

        def forget(done: asyncio.Future[Any]) -> None:
            self._unresolved_call_tasks.pop(done, None)
            _consume_task_result(done)

        task.add_done_callback(forget)
        logger.critical(
            "LLM provider task ignored bounded cancellation (owner=%s)",
            owner,
        )

    @property
    def unresolved_task_count(self) -> int:
        return sum(not task.done() for task in self._unresolved_call_tasks)

    # ------------------------------------------------------------------
    # OpenAI Chat Completions 路径
    # ------------------------------------------------------------------
    async def _call_openai(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
        system: str | None,
        schema_hint: str | None,
        json_mode: bool,
    ) -> LLMResponse:
        """通过 OpenAI SDK 调用 Chat Completions 标准流式接口。"""
        response_format = _openai_chat_response_format(config, json_mode=json_mode)
        full_messages: list[dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        if schema_hint:
            full_messages.append(
                {"role": "system", "content": f"必须以严格 JSON 格式响应。结构要求:\n{schema_hint}"}
            )
        full_messages.extend(messages)
        if response_format is not None and not any(
            "json" in str(message.get("content", "")).lower()
            for message in full_messages
        ):
            full_messages.insert(0, {"role": "system", "content": "请输出严格 JSON 对象。"})

        params: dict[str, Any] = {
            "model": config.model,
            "messages": full_messages,
            "temperature": config.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if response_format is not None:
            params["response_format"] = response_format

        content_parts: list[str] = []
        finish_reason = ""
        usage: dict[str, int] = {}

        stream: Any | None = None
        stream_close_deferred = False
        try:
            client = self._get_openai_client(config, endpoint="chat")
            completions = await self._get_openai_resource(client, endpoint="chat")
            stream = await completions.create(**params)
            async for chunk in stream:
                u = getattr(chunk, "usage", None)
                if u:
                    prompt_tokens = getattr(u, "prompt_tokens", None)
                    completion_tokens = getattr(u, "completion_tokens", None)
                    if isinstance(prompt_tokens, int) and prompt_tokens >= 0:
                        usage["prompt_tokens"] = prompt_tokens
                    if isinstance(completion_tokens, int) and completion_tokens >= 0:
                        usage["completion_tokens"] = completion_tokens
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                if piece:
                    content_parts.append(str(piece))
                fr = getattr(choice, "finish_reason", None)
                if fr:
                    finish_reason = str(fr)
        except asyncio.CancelledError:
            if stream is not None:
                self._defer_stream_close(stream, owner="openai_chat_stream")
                stream_close_deferred = True
            raise
        except (openai.APITimeoutError, httpx.TimeoutException) as err:
            raise LLMError(f"stream 帧间超时(>{self.chunk_timeout}s 无数据): {err}") from err
        except openai.APIStatusError as err:
            if err.status_code in _RETRYABLE_STATUS:
                raise _RetryableHTTP(err.status_code, str(err)) from err
            raise LLMError(f"stream HTTP {err.status_code}: {err}") from err
        except (openai.APIConnectionError, httpx.HTTPError) as err:
            raise LLMError(f"stream 网络错误: {err}") from err
        finally:
            if stream is not None and not stream_close_deferred:
                await self._close_stream_after_iteration(
                    stream,
                    owner="openai_chat_stream",
                )

        if not finish_reason:
            raise LLMError("openai chat stream ended without finish_reason")

        content = "".join(content_parts)

        return LLMResponse(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            raw_provider="openai",
        )

    async def _call_openai_tools(
        self,
        messages: list[dict[str, Any]],
        config: ModelConfig,
        system: str | None,
        tools: list[dict[str, Any]],
        tool_choice: Any,
        parallel_tool_calls: bool | None,
    ) -> LLMResponse:
        """Stream OpenAI Chat tool-call deltas and assemble each call."""
        full_messages: list[dict[str, Any]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        params: dict[str, Any] = {
            "model": config.model,
            "messages": full_messages,
            "temperature": config.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": deepcopy(tools),
            "tool_choice": _openai_chat_tool_choice(tool_choice),
        }
        if parallel_tool_calls is not None:
            params["parallel_tool_calls"] = parallel_tool_calls

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        states: dict[int, dict[str, Any]] = {}
        finish_reason = ""
        usage: dict[str, int] = {}
        stream: Any | None = None
        stream_close_deferred = False
        try:
            client = self._get_openai_client(config, endpoint="chat")
            completions = await self._get_openai_resource(client, endpoint="chat")
            stream = await completions.create(**params)
            async for chunk in stream:
                u = _field(chunk, "usage")
                if u:
                    _merge_usage_openai_chat(usage, u)
                choices = _field(chunk, "choices") or []
                if finish_reason and choices:
                    # Several OpenAI-compatible gateways emit a final empty
                    # choice (or repeat the finish marker) after the semantic
                    # finish chunk.  Ignore that bookkeeping frame, but keep
                    # rejecting any real content/tool delta after completion.
                    meaningful_after_finish = False
                    conflicting_finish = False
                    for trailing_choice in choices:
                        trailing_delta = _field(trailing_choice, "delta")
                        if _field(trailing_delta, "content"):
                            meaningful_after_finish = True
                        if any(
                            _field(trailing_delta, attr)
                            for attr in ("reasoning", "reasoning_content", "thinking")
                        ):
                            meaningful_after_finish = True
                        if _field(trailing_delta, "tool_calls"):
                            meaningful_after_finish = True
                        trailing_finish = _field(trailing_choice, "finish_reason")
                        if trailing_finish and str(trailing_finish) != finish_reason:
                            conflicting_finish = True
                    if meaningful_after_finish or conflicting_finish:
                        raise LLMResponseError(
                            "openai chat stream emitted content after its finish marker"
                        )
                    continue
                for choice in choices:
                    delta = _field(choice, "delta")
                    if delta is None:
                        continue
                    piece = _field(delta, "content")
                    if piece:
                        content_parts.append(str(piece))
                    for attr in ("reasoning", "reasoning_content", "thinking"):
                        thought_piece = _field(delta, attr)
                        if thought_piece:
                            reasoning_parts.append(str(thought_piece))
                    fragments = _field(delta, "tool_calls") or []
                    for fragment in fragments:
                        _ingest_chat_tool_fragment(states, fragment)
                    fr = _field(choice, "finish_reason")
                    if fr:
                        if finish_reason and str(fr) != finish_reason:
                            raise LLMResponseError("openai chat stream emitted conflicting finish markers")
                        finish_reason = str(fr)
        except asyncio.CancelledError:
            if stream is not None:
                self._defer_stream_close(stream, owner="openai_chat_tool_stream")
                stream_close_deferred = True
            raise
        except (openai.APITimeoutError, httpx.TimeoutException) as err:
            raise LLMError(f"stream 帧间超时(>{self.chunk_timeout}s 无数据): {err}") from err
        except openai.APIStatusError as err:
            if err.status_code in _RETRYABLE_STATUS:
                raise _RetryableHTTP(err.status_code, str(err)) from err
            raise LLMError(f"stream HTTP {err.status_code}: {err}") from err
        except (openai.APIConnectionError, httpx.HTTPError) as err:
            raise LLMError(f"stream 网络错误: {err}") from err
        finally:
            if stream is not None and not stream_close_deferred:
                await self._close_stream_after_iteration(
                    stream,
                    owner="openai_chat_tool_stream",
                )

        if not finish_reason:
            raise LLMResponseError("openai chat tool stream ended without finish_reason")
        tool_calls = _finalize_tool_states(states, protocol="openai")
        if finish_reason == "tool_calls" and not tool_calls:
            raise LLMResponseError("openai chat stream marked tool_calls without a tool call")
        if finish_reason in {"length", "content_filter"}:
            raise LLMResponseError(
                f"openai chat tool stream ended incompletely (finish_reason={finish_reason!r})"
            )
        return LLMResponse(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            finish_reason=finish_reason,
            usage=usage,
            raw_provider="openai",
            tool_calls=tool_calls,
        )

    # ------------------------------------------------------------------
    # OpenAI Responses API 路径
    # ------------------------------------------------------------------
    async def _call_openai_responses(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
        system: str | None,
        schema_hint: str | None,
        json_mode: bool,
    ) -> LLMResponse:
        """通过 OpenAI SDK 调用 Responses 标准流式接口。"""
        text_format = _openai_responses_text_format(config, json_mode=json_mode)
        input_items, extracted_instructions = _openai_response_input(messages)
        instructions = _join_instructions(
            system,
            extracted_instructions,
            schema_hint and f"必须以严格 JSON 格式响应。结构要求:\n{schema_hint}",
        )
        params: dict[str, Any] = {
            "model": config.model,
            "input": input_items,
            "temperature": config.temperature,
            "stream": True,
        }
        if instructions:
            params["instructions"] = instructions
        if config.reasoning:
            params["reasoning"] = dict(config.reasoning)
        if text_format is not None:
            params["text"] = {"format": text_format}
            if not any("json" in str(item.get("content", "")).lower() for item in input_items):
                input_items.insert(0, {"role": "user", "content": "请输出严格 JSON 对象。"})

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = ""
        usage: dict[str, int] = {}

        stream: Any | None = None
        stream_close_deferred = False
        try:
            client = self._get_openai_client(config, endpoint="responses")
            responses = await self._get_openai_resource(client, endpoint="responses")
            stream = await responses.create(**params)
            async for event in stream:
                etype = str(getattr(event, "type", "") or "")
                if etype == "response.output_text.delta":
                    piece = getattr(event, "delta", None)
                    if piece:
                        content_parts.append(str(piece))
                elif etype in {"response.reasoning_text.delta", "response.reasoning_summary_text.delta"}:
                    piece = getattr(event, "delta", None)
                    if piece:
                        reasoning_parts.append(str(piece))
                elif etype == "response.completed":
                    response = _object_to_dict(getattr(event, "response", None))
                    finish_reason = str(response.get("status") or finish_reason)
                    usage.update(_openai_response_usage(response.get("usage")))
                    if not content_parts:
                        extracted = _openai_response_output_text(response)
                        if extracted:
                            content_parts.append(extracted)
                    if not reasoning_parts:
                        extracted_reasoning = _openai_response_reasoning_text(response)
                        if extracted_reasoning:
                            reasoning_parts.append(extracted_reasoning)
                elif etype == "response.incomplete":
                    response = _object_to_dict(getattr(event, "response", None))
                    finish_reason = str(response.get("status") or "incomplete")
                    usage.update(_openai_response_usage(response.get("usage")))
                elif etype == "response.failed":
                    response = _object_to_dict(getattr(event, "response", None))
                    usage.update(_openai_response_usage(response.get("usage")))
                    raise _ProviderStreamError(
                        f"responses stream failed: {_openai_response_error_text(response)}",
                        payload=response,
                    )
                elif etype == "error":
                    payload = _object_to_dict(event)
                    raise _ProviderStreamError(
                        f"responses stream error: {_provider_error_text(payload)}",
                        payload=payload,
                    )
        except asyncio.CancelledError:
            if stream is not None:
                self._defer_stream_close(stream, owner="openai_responses_stream")
                stream_close_deferred = True
            raise
        except (openai.APITimeoutError, httpx.TimeoutException) as err:
            raise LLMError(f"responses stream 帧间超时(>{self.chunk_timeout}s 无数据): {err}") from err
        except openai.APIStatusError as err:
            if err.status_code in _RETRYABLE_STATUS:
                raise _RetryableHTTP(err.status_code, str(err)) from err
            raise LLMError(f"responses stream HTTP {err.status_code}: {err}") from err
        except (openai.APIConnectionError, httpx.HTTPError) as err:
            raise LLMError(f"responses stream 网络错误: {err}") from err
        finally:
            if stream is not None and not stream_close_deferred:
                await self._close_stream_after_iteration(
                    stream,
                    owner="openai_responses_stream",
                )

        if not finish_reason:
            raise LLMError("responses stream ended without response.completed")
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        return LLMResponse(
            content=content,
            reasoning=reasoning,
            finish_reason=finish_reason,
            usage=usage,
            raw_provider="openai_responses",
        )

    async def _call_openai_responses_tools(
        self,
        messages: list[dict[str, Any]],
        config: ModelConfig,
        system: str | None,
        tools: list[dict[str, Any]],
        tool_choice: Any,
        parallel_tool_calls: bool | None,
    ) -> LLMResponse:
        """Stream standard Responses function-call item and argument events."""
        input_items, extracted_instructions = _openai_tool_response_input(messages)
        instructions = _join_instructions(system, extracted_instructions)
        params: dict[str, Any] = {
            "model": config.model,
            "input": input_items,
            "temperature": config.temperature,
            "stream": True,
            "tools": _openai_responses_tools(tools),
            "tool_choice": _openai_responses_tool_choice(tool_choice),
        }
        if instructions:
            params["instructions"] = instructions
        if config.reasoning:
            params["reasoning"] = dict(config.reasoning)
        if parallel_tool_calls is not None:
            params["parallel_tool_calls"] = parallel_tool_calls

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        states: dict[str, dict[str, Any]] = {}
        aliases: dict[str, str] = {}
        finish_reason = ""
        usage: dict[str, int] = {}
        saw_completed = False
        stream: Any | None = None
        stream_close_deferred = False
        try:
            client = self._get_openai_client(config, endpoint="responses")
            responses = await self._get_openai_resource(client, endpoint="responses")
            stream = await responses.create(**params)
            async for event in stream:
                etype = str(_field(event, "type") or "")
                if saw_completed:
                    raise LLMResponseError(
                        "responses stream emitted an event after response.completed"
                    )
                if etype == "response.output_text.delta":
                    piece = _field(event, "delta")
                    if piece:
                        content_parts.append(str(piece))
                elif etype in {
                    "response.reasoning_text.delta",
                    "response.reasoning_summary_text.delta",
                }:
                    piece = _field(event, "delta")
                    if piece:
                        reasoning_parts.append(str(piece))
                elif etype in {"response.output_item.added", "response.output_item.done"}:
                    item = _field(event, "item")
                    if str(_field(item, "type") or "") == "function_call":
                        _ingest_responses_tool_item(
                            states,
                            aliases,
                            item,
                            output_index=_field(event, "output_index"),
                            terminal=etype.endswith(".done"),
                            added=etype.endswith(".added"),
                        )
                elif etype == "response.function_call_arguments.delta":
                    _ingest_responses_argument_delta(states, aliases, event)
                elif etype == "response.function_call_arguments.done":
                    _ingest_responses_arguments_done(states, aliases, event)
                elif etype == "response.completed":
                    if saw_completed:
                        raise LLMResponseError("responses stream emitted duplicate completion markers")
                    saw_completed = True
                    response = _object_to_dict(_field(event, "response"))
                    finish_reason = str(response.get("status") or "completed")
                    usage.update(_openai_response_usage(response.get("usage")))
                    if not content_parts:
                        extracted = _openai_response_output_text(response)
                        if extracted:
                            content_parts.append(extracted)
                    if not reasoning_parts:
                        extracted_reasoning = _openai_response_reasoning_text(response)
                        if extracted_reasoning:
                            reasoning_parts.append(extracted_reasoning)
                    output = response.get("output")
                    if isinstance(output, list):
                        for index, item in enumerate(output):
                            if isinstance(item, dict) and item.get("type") == "function_call":
                                _ingest_responses_tool_item(
                                    states,
                                    aliases,
                                    item,
                                    output_index=index,
                                    terminal=True,
                                    added=False,
                                )
                elif etype == "response.incomplete":
                    response = _object_to_dict(_field(event, "response"))
                    usage.update(_openai_response_usage(response.get("usage")))
                    raise LLMResponseError(
                        "responses tool stream ended incomplete: "
                        + _openai_response_error_text(response)
                    )
                elif etype == "response.failed":
                    response = _object_to_dict(_field(event, "response"))
                    usage.update(_openai_response_usage(response.get("usage")))
                    raise _ProviderStreamError(
                        f"responses stream failed: {_openai_response_error_text(response)}",
                        payload=response,
                    )
                elif etype == "error":
                    payload = _object_to_dict(event)
                    raise _ProviderStreamError(
                        f"responses stream error: {_provider_error_text(payload)}",
                        payload=payload,
                    )
        except asyncio.CancelledError:
            if stream is not None:
                self._defer_stream_close(stream, owner="openai_responses_tool_stream")
                stream_close_deferred = True
            raise
        except (openai.APITimeoutError, httpx.TimeoutException) as err:
            raise LLMError(
                f"responses stream 帧间超时(>{self.chunk_timeout}s 无数据): {err}"
            ) from err
        except openai.APIStatusError as err:
            if err.status_code in _RETRYABLE_STATUS:
                raise _RetryableHTTP(err.status_code, str(err)) from err
            raise LLMError(f"responses stream HTTP {err.status_code}: {err}") from err
        except (openai.APIConnectionError, httpx.HTTPError) as err:
            raise LLMError(f"responses stream 网络错误: {err}") from err
        finally:
            if stream is not None and not stream_close_deferred:
                await self._close_stream_after_iteration(
                    stream,
                    owner="openai_responses_tool_stream",
                )

        if not saw_completed or finish_reason != "completed":
            raise LLMResponseError("responses tool stream ended without response.completed")
        return LLMResponse(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            finish_reason=finish_reason,
            usage=usage,
            raw_provider="openai_responses",
            tool_calls=_finalize_tool_states(states, protocol="openai_responses"),
        )

    # ------------------------------------------------------------------
    # Anthropic Messages 路径
    # ------------------------------------------------------------------
    async def _call_anthropic(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
        system: str | None,
        schema_hint: str | None,
        json_mode: bool,
    ) -> LLMResponse:
        """通过 Anthropic SDK 调用 Messages 标准流式接口。"""
        anthropic_messages, extracted_system = _anthropic_messages(messages)
        sys_parts = [
            p
            for p in [
                system,
                extracted_system,
                schema_hint and f"必须以严格 JSON 响应:\n{schema_hint}",
            ]
            if p
        ]
        params: dict[str, Any] = {
            "model": config.model,
            "messages": anthropic_messages,
            "max_tokens": _anthropic_max_tokens(config),
        }
        if config.thinking:
            params["thinking"] = dict(config.thinking)
        else:
            params["temperature"] = config.temperature
        if sys_parts:
            params["system"] = "\n\n".join(sys_parts)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = ""
        saw_message_stop = False
        in_tok: int | None = None
        out_tok: int | None = None

        try:
            client = self._get_anthropic_client(config)
            async with client.messages.stream(**params) as stream:
                async for event in stream:
                    etype = str(getattr(event, "type", "") or "")
                    if etype == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        dtype = str(getattr(delta, "type", "") or "")
                        if dtype == "text_delta":
                            piece = getattr(delta, "text", None)
                            if piece:
                                content_parts.append(str(piece))
                        elif dtype == "thinking_delta":
                            piece = getattr(delta, "thinking", None)
                            if piece:
                                reasoning_parts.append(str(piece))
                    elif etype == "message_delta":
                        delta = getattr(event, "delta", None)
                        stop_reason = getattr(delta, "stop_reason", None) if delta is not None else None
                        if stop_reason:
                            finish_reason = str(stop_reason)
                        u = getattr(event, "usage", None)
                        if u:
                            candidate = getattr(u, "output_tokens", None)
                            if isinstance(candidate, int) and candidate >= 0:
                                out_tok = candidate
                    elif etype == "message_start":
                        message = getattr(event, "message", None)
                        u = getattr(message, "usage", None) if message is not None else None
                        if u:
                            input_candidate = getattr(u, "input_tokens", None)
                            output_candidate = getattr(u, "output_tokens", None)
                            if isinstance(input_candidate, int) and input_candidate >= 0:
                                in_tok = input_candidate
                            if isinstance(output_candidate, int) and output_candidate >= 0:
                                out_tok = output_candidate
                    elif etype == "message_stop":
                        saw_message_stop = True
                    elif etype == "error":
                        payload = _object_to_dict(event)
                        raise _ProviderStreamError(
                            f"anthropic stream error: {_provider_error_text(payload)}",
                            payload=payload,
                        )
        except (anthropic.APITimeoutError, httpx.TimeoutException) as err:
            raise LLMError(f"anthropic stream 帧间超时(>{self.chunk_timeout}s 无数据): {err}") from err
        except anthropic.APIStatusError as err:
            if err.status_code == 200:
                payload = _object_to_dict(err.body)
                raise _ProviderStreamError(
                    f"anthropic stream error: {_provider_error_text(payload)}",
                    payload=payload,
                ) from err
            if err.status_code in _RETRYABLE_STATUS:
                raise _RetryableHTTP(err.status_code, str(err)) from err
            raise LLMError(f"anthropic stream HTTP {err.status_code}: {err}") from err
        except (anthropic.APIConnectionError, httpx.HTTPError) as err:
            raise LLMError(f"anthropic stream 网络错误: {err}") from err

        if not finish_reason or not saw_message_stop:
            raise LLMError("anthropic stream ended without message_stop or stop_reason")

        return LLMResponse(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            finish_reason=finish_reason,
            usage={
                **({"prompt_tokens": in_tok} if in_tok is not None else {}),
                **({"completion_tokens": out_tok} if out_tok is not None else {}),
            },
            raw_provider="anthropic",
        )

    async def _call_anthropic_tools(
        self,
        messages: list[dict[str, Any]],
        config: ModelConfig,
        system: str | None,
        tools: list[dict[str, Any]],
        tool_choice: Any,
        parallel_tool_calls: bool | None,
    ) -> LLMResponse:
        """Stream Anthropic ``tool_use`` blocks and JSON input deltas."""
        anthropic_messages, extracted_system = _anthropic_tool_messages(messages)
        params: dict[str, Any] = {
            "model": config.model,
            "messages": anthropic_messages,
            "max_tokens": _anthropic_max_tokens(config),
            "tools": _anthropic_tools(tools),
            "tool_choice": _anthropic_tool_choice(tool_choice, parallel_tool_calls),
        }
        if config.thinking:
            params["thinking"] = dict(config.thinking)
        else:
            params["temperature"] = config.temperature
        instructions = _join_instructions(system, extracted_system)
        if instructions:
            params["system"] = instructions

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        states: dict[int, dict[str, Any]] = {}
        finish_reason = ""
        saw_message_stop = False
        in_tok: int | None = None
        out_tok: int | None = None
        try:
            client = self._get_anthropic_client(config)
            async with client.messages.stream(**params) as stream:
                async for event in stream:
                    etype = str(_field(event, "type") or "")
                    if saw_message_stop:
                        raise LLMResponseError(
                            "anthropic stream emitted an event after message_stop"
                        )
                    if etype == "content_block_start":
                        block = _field(event, "content_block")
                        if str(_field(block, "type") or "") == "tool_use":
                            _ingest_anthropic_tool_start(
                                states,
                                index=_field(event, "index"),
                                block=block,
                            )
                    elif etype == "content_block_delta":
                        delta = _field(event, "delta")
                        dtype = str(_field(delta, "type") or "")
                        if dtype == "text_delta":
                            piece = _field(delta, "text")
                            if piece:
                                content_parts.append(str(piece))
                        elif dtype == "thinking_delta":
                            piece = _field(delta, "thinking")
                            if piece:
                                reasoning_parts.append(str(piece))
                        elif dtype == "input_json_delta":
                            _ingest_anthropic_input_delta(
                                states,
                                index=_field(event, "index"),
                                partial_json=_field(delta, "partial_json"),
                            )
                    elif etype == "content_block_stop":
                        index = _field(event, "index")
                        if isinstance(index, int) and not isinstance(index, bool) and index in states:
                            _finish_anthropic_tool_block(states, index=index)
                    elif etype == "message_delta":
                        delta = _field(event, "delta")
                        stop_reason = _field(delta, "stop_reason")
                        if stop_reason:
                            if finish_reason and str(stop_reason) != finish_reason:
                                raise LLMResponseError(
                                    "anthropic stream emitted conflicting stop reasons"
                                )
                            finish_reason = str(stop_reason)
                        usage = _field(event, "usage")
                        candidate = _field(usage, "output_tokens")
                        if (
                            isinstance(candidate, int)
                            and not isinstance(candidate, bool)
                            and candidate >= 0
                        ):
                            out_tok = candidate
                    elif etype == "message_start":
                        message = _field(event, "message")
                        usage = _field(message, "usage")
                        input_candidate = _field(usage, "input_tokens")
                        output_candidate = _field(usage, "output_tokens")
                        if (
                            isinstance(input_candidate, int)
                            and not isinstance(input_candidate, bool)
                            and input_candidate >= 0
                        ):
                            in_tok = input_candidate
                        if (
                            isinstance(output_candidate, int)
                            and not isinstance(output_candidate, bool)
                            and output_candidate >= 0
                        ):
                            out_tok = output_candidate
                    elif etype == "message_stop":
                        if saw_message_stop:
                            raise LLMResponseError(
                                "anthropic stream emitted duplicate message_stop markers"
                            )
                        saw_message_stop = True
                    elif etype == "error":
                        payload = _object_to_dict(event)
                        raise _ProviderStreamError(
                            f"anthropic stream error: {_provider_error_text(payload)}",
                            payload=payload,
                        )
        except (anthropic.APITimeoutError, httpx.TimeoutException) as err:
            raise LLMError(
                f"anthropic stream 帧间超时(>{self.chunk_timeout}s 无数据): {err}"
            ) from err
        except anthropic.APIStatusError as err:
            if err.status_code == 200:
                payload = _object_to_dict(err.body)
                raise _ProviderStreamError(
                    f"anthropic stream error: {_provider_error_text(payload)}",
                    payload=payload,
                ) from err
            if err.status_code in _RETRYABLE_STATUS:
                raise _RetryableHTTP(err.status_code, str(err)) from err
            raise LLMError(f"anthropic stream HTTP {err.status_code}: {err}") from err
        except (anthropic.APIConnectionError, httpx.HTTPError) as err:
            raise LLMError(f"anthropic stream 网络错误: {err}") from err

        if not finish_reason or not saw_message_stop:
            raise LLMResponseError(
                "anthropic tool stream ended without message_stop or stop_reason"
            )
        if finish_reason in {"max_tokens", "refusal"}:
            raise LLMResponseError(
                f"anthropic tool stream ended incompletely (stop_reason={finish_reason!r})"
            )
        tool_calls = _finalize_tool_states(states, protocol="anthropic")
        if finish_reason == "tool_use" and not tool_calls:
            raise LLMResponseError("anthropic stream marked tool_use without a tool call")
        return LLMResponse(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            finish_reason=finish_reason,
            usage={
                **({"prompt_tokens": in_tok} if in_tok is not None else {}),
                **({"completion_tokens": out_tok} if out_tok is not None else {}),
            },
            raw_provider="anthropic",
            tool_calls=tool_calls,
        )

    # ------------------------------------------------------------------
    # 重试与解析工具
    # ------------------------------------------------------------------
    @staticmethod
    def _is_retryable(err: Exception) -> bool:
        if isinstance(err, LLMCallCleanupError):
            return False
        if isinstance(err, _RetryableHTTP):
            return True
        if isinstance(err, httpx.HTTPStatusError):
            return _safe_status_code(err) in _RETRYABLE_STATUS
        if isinstance(err, httpx.HTTPError):
            return True
        if isinstance(
            err,
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
            ),
        ):
            return True
        if isinstance(err, openai.APIStatusError):
            return _safe_status_code(err) in _RETRYABLE_STATUS
        if isinstance(err, anthropic.APIStatusError):
            return _safe_status_code(err) in _RETRYABLE_STATUS
        if isinstance(err, _ProviderStreamError):
            return _safe_status_code(err) in _RETRYABLE_STATUS
        # Some OpenAI-compatible gateways send an error envelope inside the
        # SSE stream instead of an HTTP response. The SDK then raises the
        # generic APIError (without ``status_code``), even when the envelope
        # carries a standard 429/5xx code. Recover that structured code so
        # bounded transport retries and attempt provenance remain correct.
        if isinstance(err, (openai.APIError, anthropic.APIError)):
            status_code = _safe_status_code(err)
            return status_code in _RETRYABLE_STATUS
        if isinstance(err, asyncio.TimeoutError):
            return True
        # LLMError 包装的瞬时错误也允许重试(除非是不可恢复的 4xx)。
        # 流式路径把 httpx 网络错误(incomplete chunked read / 连接重置等)包成 LLMError,
        # 这里看 __cause__ 是否为可重试的 httpx 错误,是则重试。
        cause = getattr(err, "__cause__", None)
        if isinstance(cause, httpx.HTTPStatusError):
            return _safe_status_code(cause) in _RETRYABLE_STATUS
        if isinstance(
            cause,
            (openai.APIStatusError, anthropic.APIStatusError, _ProviderStreamError),
        ):
            return _safe_status_code(cause) in _RETRYABLE_STATUS
        if isinstance(
            cause,
            (
                httpx.HTTPError,
                asyncio.TimeoutError,
                _RetryableHTTP,
                openai.APIConnectionError,
                openai.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
            ),
        ):
            return True
        if isinstance(cause, (openai.APIError, anthropic.APIError)):
            return _safe_status_code(cause) in _RETRYABLE_STATUS
        return False

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """指数退避 + 抖动:base * 2^attempt + jitter,上限 20s。"""
        base = 0.8
        delay = min(20.0, base * (2**attempt))
        return delay + random.uniform(0, 0.4)  # noqa: S311 — 抖动无需密码学安全

    @staticmethod
    def _parse_json(
        content: str,
        config: ModelConfig,
        *,
        allow_lossy: bool = False,
        include_parse_metadata: bool = False,
    ) -> dict[str, Any]:
        """解析 JSON dict。

        默认只接受完整 JSON 或无损恢复(围栏/嵌入 JSON/Python 字面量/缺少末尾 })。
        末尾字段截断兜底会丢弃不完整字段,标记为 lossy;除非调用方显式
        allow_lossy=True,否则抛 LLMResponseError 触发上层重新请求。
        """
        result = LLMRouter._parse_json_result(content, config)
        if result.lossy and not allow_lossy:
            raise LLMResponseError(
                "JSON 有损恢复被拒绝"
                f"(provider={config.provider} method={result.method}): {content[:300]!r}"
            )
        data = dict(result.data)
        if include_parse_metadata:
            data["_parse_recovered"] = result.recovered
            data["_parse_lossy"] = result.lossy
            data["_parse_method"] = result.method
        return data

    @staticmethod
    def _parse_json_result(content: str, config: ModelConfig) -> JSONParseResult:
        """容错解析并返回恢复方式。失败抛 LLMResponseError。"""
        import ast

        text = content.strip()
        fenced = False
        if text.startswith("```"):
            parts = text.split("```", 2)
            text = parts[1] if len(parts) > 1 else content
            if text.startswith("json"):
                text = text[4:]
            fenced = True
        text = text.strip()

        # 1) 标准 JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return JSONParseResult(
                    obj,
                    method="fenced_json" if fenced else "json",
                    recovered=fenced,
                )
            raise LLMResponseError(f"JSON 解析结果非对象(provider={config.provider}): {type(obj)}")
        except json.JSONDecodeError:
            pass

        # 2) 单引号 JSON 或 Python 字面量
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return JSONParseResult(
                    obj,
                    method="fenced_literal" if fenced else "literal",
                    recovered=True,
                )
            raise LLMResponseError(f"JSON 解析结果非对象(provider={config.provider}): {type(obj)}")
        except (ValueError, SyntaxError):
            pass

        # 3) 从原始内容中提取首个可能的 {...} 块,尝试截断补全
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            for candidate_end in range(end, start, -1):
                if content[candidate_end] != "}":
                    continue
                snippet = content[start : candidate_end + 1]
                try:
                    obj = json.loads(snippet)
                    if isinstance(obj, dict):
                        return JSONParseResult(obj, method="embedded_json", recovered=True)
                    raise LLMResponseError(f"JSON 解析结果非对象(provider={config.provider}): {type(obj)}")
                except json.JSONDecodeError:
                    pass
                try:
                    obj = ast.literal_eval(snippet)
                    if isinstance(obj, dict):
                        return JSONParseResult(obj, method="embedded_literal", recovered=True)
                    raise LLMResponseError(f"JSON 解析结果非对象(provider={config.provider}): {type(obj)}")
                except (ValueError, SyntaxError):
                    pass

        # 4) 截断补全:模型输出未闭合时,尝试补充缺少的 } 让 Python 字面量可用
        if start != -1:
            snippet = content[start:]
            open_braces = snippet.count("{")
            close_braces = snippet.count("}")
            for extra in range(1, max(1, open_braces - close_braces) + 3):
                candidate = snippet + "}" * extra
                try:
                    obj = ast.literal_eval(candidate)
                    if isinstance(obj, dict):
                        return JSONParseResult(obj, method="balanced_literal", recovered=True)
                    raise LLMResponseError(f"JSON 解析结果非对象(provider={config.provider}): {type(obj)}")
                except (ValueError, SyntaxError):
                    pass

        # 5) 末尾字段截断兜底:逐个丢弃末尾不完整的 key-value,保留已完整的字段
        #    场景:speech 字段值被 max_tokens 截断在字符串中间,引号未配对。
        #    用正则提取所有完整的 "key": value 对,重建一个 dict。
        import re
        if start != -1:
            snippet = content[start:]
            # 匹配 "key" 或 'key' 后跟 : 和 value(字符串/数字/bool/null/嵌套{})
            # 逐个提取,遇到不完整的最后一个就停
            kv_pattern = re.compile(
                r"""['"](\w+)['"]\s*:\s*("""
                r"""(?:['"])(?:\\.|[^'"])*['"]"""      # 完整字符串(双/单引号,含转义)
                r"""|-?\d+(?:\.\d+)?"""                  # 数字
                r"""|true|false|null"""
                r"""|\{[^{}]*\})"""                       # 单层嵌套 dict
                r"""\s*(?=,|\}|$)""",
                re.DOTALL,
            )
            found = {}
            for m in kv_pattern.finditer(snippet):
                key = m.group(1)
                val_raw = m.group(2).strip()
                try:
                    found[key] = ast.literal_eval(val_raw)
                except (ValueError, SyntaxError):
                    try:
                        found[key] = json.loads(val_raw)
                    except json.JSONDecodeError:
                        continue
            if found:
                return JSONParseResult(found, method="lossy_kv", recovered=True, lossy=True)

        raise LLMResponseError(f"JSON 解析失败(provider={config.provider}): {content[:300]!r}")

    @staticmethod
    def _ensure_complete_finish(resp: LLMResponse, config: ModelConfig) -> None:
        """Reject known incomplete structured responses before sanitizing."""
        reason = (resp.finish_reason or "").strip().lower()
        protocol = _standard_protocol(config)
        allowed = {
            "openai": {"stop"},
            "openai_responses": {"completed"},
            "anthropic": {"end_turn", "stop_sequence"},
        }.get(protocol, {"stop"})
        if not reason:
            raise LLMResponseError(
                f"LLM 输出缺少完成状态(provider={config.provider})"
            )
        if reason not in allowed:
            raise LLMResponseError(
                f"LLM 输出未完整结束(provider={config.provider} finish_reason={resp.finish_reason!r})"
            )


class _RetryableHTTP(Exception):
    """可重试的 HTTP 状态错误。"""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:200]}")


def _openai_base_url(api_base: str, *, endpoint: str) -> str:
    base = (api_base or "").strip().rstrip("/")
    if endpoint == "chat" and base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if endpoint == "responses" and base.endswith("/responses"):
        return base[: -len("/responses")]
    return base


def _anthropic_base_url(api_base: str) -> str:
    base = (api_base or "").strip().rstrip("/")
    if base.endswith("/v1/messages"):
        return base[: -len("/v1/messages")]
    if base.endswith("/v1"):
        return base[: -len("/v1")]
    return base


def _standard_protocol(config: ModelConfig) -> str:
    protocol = (config.provider or "openai").strip().lower()
    if protocol in STANDARD_PROTOCOLS:
        return protocol
    raise LLMError(
        "未知标准协议 provider="
        f"{protocol!r}; 仅支持 openai(Chat Completions), openai_responses(Responses), anthropic(Messages)"
    )


_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _normalize_tool_definitions(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(tools, list) or not tools:
        raise LLMResponseError("tools must be a non-empty list of function definitions")
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise LLMResponseError(f"tools[{index}] must be an object")
        if set(tool) != {"type", "function"} or tool.get("type") != "function":
            raise LLMResponseError(
                f"tools[{index}] must use the standard type=function/function shape"
            )
        function = tool.get("function")
        if not isinstance(function, dict):
            raise LLMResponseError(f"tools[{index}].function must be an object")
        allowed = {"name", "description", "parameters", "strict"}
        if set(function) - allowed:
            raise LLMResponseError(
                f"tools[{index}].function contains unsupported fields"
            )
        name = function.get("name")
        if not isinstance(name, str) or _TOOL_NAME_PATTERN.fullmatch(name) is None:
            raise LLMResponseError(
                f"tools[{index}].function.name must match [A-Za-z0-9_-]{{1,64}}"
            )
        if name in seen_names:
            raise LLMResponseError(f"duplicate function tool name: {name}")
        seen_names.add(name)
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            raise LLMResponseError(
                f"tools[{index}].function.parameters must be a JSON Schema object"
            )
        try:
            encoded = json.dumps(parameters, ensure_ascii=True, allow_nan=False)
            safe_parameters = json.loads(encoded)
            Draft202012Validator.check_schema(safe_parameters)
        except (TypeError, ValueError, SchemaError) as err:
            raise LLMResponseError(
                f"tools[{index}].function.parameters is not valid JSON Schema"
            ) from err
        if safe_parameters.get("type") != "object":
            raise LLMResponseError(
                f"tools[{index}].function.parameters must describe an object"
            )
        description = function.get("description")
        if description is not None and not isinstance(description, str):
            raise LLMResponseError(f"tools[{index}].function.description must be a string")
        strict = function.get("strict")
        if strict is not None and not isinstance(strict, bool):
            raise LLMResponseError(f"tools[{index}].function.strict must be a boolean")
        safe_function: dict[str, Any] = {
            "name": name,
            "parameters": safe_parameters,
        }
        if description is not None:
            safe_function["description"] = description
        if strict is not None:
            safe_function["strict"] = strict
        normalized.append({"type": "function", "function": safe_function})
    return normalized


def _normalize_tool_choice(
    tool_choice: Any,
    tools: list[dict[str, Any]],
) -> str | dict[str, Any]:
    if tool_choice is None:
        tool_choice = "auto"
    if isinstance(tool_choice, str):
        value = tool_choice.strip().lower()
        if value not in {"auto", "none", "required"}:
            raise LLMResponseError("tool_choice must be auto, none, required, or a function")
        return value
    if not isinstance(tool_choice, dict):
        raise LLMResponseError("tool_choice must be a string or function choice object")
    name: Any = None
    if set(tool_choice) == {"type", "function"} and tool_choice.get("type") == "function":
        function = tool_choice.get("function")
        if isinstance(function, dict) and set(function) == {"name"}:
            name = function.get("name")
    elif set(tool_choice) == {"type", "name"} and tool_choice.get("type") == "function":
        name = tool_choice.get("name")
    if not isinstance(name, str) or _TOOL_NAME_PATTERN.fullmatch(name) is None:
        raise LLMResponseError("tool_choice function shape is unsupported")
    names = {str(tool["function"]["name"]) for tool in tools}
    if name not in names:
        raise LLMResponseError(f"tool_choice names an undefined function: {name}")
    return {"type": "function", "function": {"name": name}}


def _openai_chat_tool_choice(tool_choice: Any) -> Any:
    return deepcopy(tool_choice)


def _openai_responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    for tool in tools:
        function = tool["function"]
        item: dict[str, Any] = {
            "type": "function",
            "name": function["name"],
            "parameters": deepcopy(function["parameters"]),
        }
        for key in ("description", "strict"):
            if key in function:
                item[key] = function[key]
        translated.append(item)
    return translated


def _openai_responses_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        return tool_choice
    return {
        "type": "function",
        "name": tool_choice["function"]["name"],
    }


def _anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    for tool in tools:
        function = tool["function"]
        item: dict[str, Any] = {
            "name": function["name"],
            "input_schema": deepcopy(function["parameters"]),
        }
        if "description" in function:
            item["description"] = function["description"]
        translated.append(item)
    return translated


def _anthropic_tool_choice(
    tool_choice: Any,
    parallel_tool_calls: bool | None,
) -> dict[str, Any]:
    if isinstance(tool_choice, str):
        choice: dict[str, Any] = {
            "type": {"required": "any"}.get(tool_choice, tool_choice),
        }
    else:
        choice = {"type": "tool", "name": tool_choice["function"]["name"]}
    if parallel_tool_calls is not None and choice["type"] != "none":
        choice["disable_parallel_tool_use"] = not parallel_tool_calls
    return choice


def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _merge_usage_openai_chat(usage: dict[str, int], value: Any) -> None:
    for source, destination in (
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
    ):
        candidate = _field(value, source)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            usage[destination] = candidate


def _set_tool_state_value(
    state: dict[str, Any],
    field_name: str,
    value: Any,
    *,
    label: str,
) -> None:
    if value is None or value == "":
        return
    if not isinstance(value, str):
        raise LLMResponseError(f"{label} must be a string")
    if field_name == "call_id" and len(value) > MAX_TOOL_CALL_ID_CHARS:
        raise LLMResponseError(f"{label} exceeds {MAX_TOOL_CALL_ID_CHARS} characters")
    if field_name == "name" and len(value) > 64:
        raise LLMResponseError(f"{label} exceeds 64 characters")
    existing = state.get(field_name)
    if existing and existing != value:
        raise LLMResponseError(f"conflicting {label} fragments")
    state[field_name] = value


def _ingest_chat_tool_fragment(
    states: dict[int, dict[str, Any]],
    fragment: Any,
) -> None:
    index = _field(fragment, "index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise LLMResponseError("openai tool-call fragment has an invalid index")
    state = states.setdefault(index, {"order": index, "raw": ""})
    fragment_type = _field(fragment, "type")
    if fragment_type not in (None, "", "function"):
        raise LLMResponseError("openai emitted an unsupported tool-call type")
    _set_tool_state_value(
        state,
        "call_id",
        _field(fragment, "id"),
        label="openai tool call id",
    )
    function = _field(fragment, "function")
    if function is None:
        return
    _set_tool_state_value(
        state,
        "name",
        _field(function, "name"),
        label="openai tool name",
    )
    arguments = _field(function, "arguments")
    if arguments is not None:
        if not isinstance(arguments, str):
            raise LLMResponseError("openai tool arguments fragment must be a string")
        _append_tool_arguments(state, arguments, label="openai tool arguments")


def _responses_state(
    states: dict[str, dict[str, Any]],
    aliases: dict[str, str],
    *,
    item_id: Any = None,
    call_id: Any = None,
    output_index: Any = None,
) -> dict[str, Any]:
    identifiers: list[str] = []
    for prefix, value in (("item", item_id), ("call", call_id)):
        if value not in (None, ""):
            if not isinstance(value, str):
                raise LLMResponseError(f"responses {prefix} id must be a string")
            identifiers.append(f"{prefix}:{value}")
    if output_index is not None:
        if isinstance(output_index, bool) or not isinstance(output_index, int) or output_index < 0:
            raise LLMResponseError("responses function call has an invalid output index")
        identifiers.append(f"index:{output_index}")
    call_identifier = next((item for item in identifiers if item.startswith("call:")), None)
    if call_identifier is not None and call_identifier in aliases:
        bound_call = aliases[call_identifier]
        # A call id is the provider's stable identity.  Seeing that id with a
        # previously unseen item/index means a second call illegally reused
        # the id; do not silently merge the two calls.
        for identifier in identifiers:
            if identifier == call_identifier:
                continue
            bound_identifier = aliases.get(identifier)
            if bound_identifier is None or bound_identifier != bound_call:
                raise LLMResponseError("responses emitted a duplicate function call id")
    known = {aliases[value] for value in identifiers if value in aliases}
    if len(known) > 1:
        raise LLMResponseError("responses function-call identifiers conflict")
    key = next(iter(known), identifiers[0] if identifiers else "")
    if not key:
        raise LLMResponseError("responses function call has no stable identifier")
    state = states.setdefault(
        key,
        {"order": output_index if isinstance(output_index, int) else len(states), "raw": ""},
    )
    for identifier in identifiers:
        prior = aliases.get(identifier)
        if prior is not None and prior != key:
            raise LLMResponseError("responses function-call identifier was reused")
        aliases[identifier] = key
    return state


def _set_terminal_arguments(state: dict[str, Any], arguments: Any, *, label: str) -> None:
    if arguments is None:
        return
    if not isinstance(arguments, str):
        raise LLMResponseError(f"{label} must be a string")
    if len(arguments) > MAX_TOOL_ARGUMENT_CHARS:
        raise LLMResponseError(f"{label} exceeds {MAX_TOOL_ARGUMENT_CHARS} characters")
    existing = state.get("raw", "")
    if existing and existing != arguments:
        # Responses emits argument deltas followed by a ``done`` event that
        # repeats the complete JSON string.  A complete value may therefore
        # legitimately extend the already-accumulated prefix; unrelated text
        # is a real conflict and must fail closed.
        if arguments.startswith(existing):
            state["raw"] = arguments
            return
        raise LLMResponseError(f"conflicting {label} fragments")
    state["raw"] = arguments


def _ingest_responses_tool_item(
    states: dict[str, dict[str, Any]],
    aliases: dict[str, str],
    item: Any,
    *,
    output_index: Any,
    terminal: bool,
    added: bool,
) -> None:
    item_id = _field(item, "id")
    call_id = _field(item, "call_id")
    state = _responses_state(
        states,
        aliases,
        item_id=item_id,
        call_id=call_id,
        output_index=output_index,
    )
    if added:
        if state.get("added_seen"):
            raise LLMResponseError("responses emitted duplicate function-call item fragments")
        state["added_seen"] = True
    _set_tool_state_value(state, "call_id", call_id, label="responses call id")
    _set_tool_state_value(state, "name", _field(item, "name"), label="responses tool name")
    arguments = _field(item, "arguments")
    if terminal:
        _set_terminal_arguments(state, arguments, label="responses tool arguments")
        state["terminal"] = True
    elif arguments:
        _set_terminal_arguments(state, arguments, label="responses tool arguments")


def _ingest_responses_argument_delta(
    states: dict[str, dict[str, Any]],
    aliases: dict[str, str],
    event: Any,
) -> None:
    state = _responses_state(
        states,
        aliases,
        item_id=_field(event, "item_id"),
        output_index=_field(event, "output_index"),
    )
    delta = _field(event, "delta")
    if not isinstance(delta, str):
        raise LLMResponseError("responses function arguments delta must be a string")
    if state.get("terminal"):
        raise LLMResponseError("responses emitted arguments after a terminal fragment")
    _append_tool_arguments(state, delta, label="responses tool arguments")


def _ingest_responses_arguments_done(
    states: dict[str, dict[str, Any]],
    aliases: dict[str, str],
    event: Any,
) -> None:
    state = _responses_state(
        states,
        aliases,
        item_id=_field(event, "item_id"),
        output_index=_field(event, "output_index"),
    )
    if state.get("terminal"):
        raise LLMResponseError("responses emitted duplicate terminal argument fragments")
    _set_tool_state_value(
        state,
        "name",
        _field(event, "name"),
        label="responses tool name",
    )
    _set_terminal_arguments(
        state,
        _field(event, "arguments"),
        label="responses tool arguments",
    )
    state["terminal"] = True


def _ingest_anthropic_tool_start(
    states: dict[int, dict[str, Any]],
    *,
    index: Any,
    block: Any,
) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise LLMResponseError("anthropic tool_use block has an invalid index")
    if index in states:
        raise LLMResponseError("anthropic emitted duplicate tool_use block starts")
    state: dict[str, Any] = {"order": index, "raw": ""}
    states[index] = state
    _set_tool_state_value(state, "call_id", _field(block, "id"), label="anthropic tool id")
    _set_tool_state_value(state, "name", _field(block, "name"), label="anthropic tool name")
    initial_input = _field(block, "input")
    if initial_input not in (None, {}):
        if not isinstance(initial_input, dict):
            raise LLMResponseError("anthropic tool input must be an object")
        state["raw"] = json.dumps(
            initial_input,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        state["input_from_start"] = True


def _ingest_anthropic_input_delta(
    states: dict[int, dict[str, Any]],
    *,
    index: Any,
    partial_json: Any,
) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or index not in states:
        raise LLMResponseError("anthropic input_json_delta has no tool_use block")
    if not isinstance(partial_json, str):
        raise LLMResponseError("anthropic input_json_delta must be a string")
    state = states[index]
    if state.get("terminal") or state.get("input_from_start"):
        raise LLMResponseError("anthropic emitted JSON deltas after complete tool input")
    state["saw_input_delta"] = True
    state["raw"] += partial_json


def _finish_anthropic_tool_block(
    states: dict[int, dict[str, Any]],
    *,
    index: Any,
) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or index not in states:
        raise LLMResponseError("anthropic content_block_stop has no tool_use block")
    state = states[index]
    if state.get("terminal"):
        raise LLMResponseError("anthropic emitted duplicate tool content_block_stop events")
    # ``input: {}`` in content_block_start is both the streaming placeholder
    # and the valid final value for a zero-argument tool.  Only block_stop can
    # disambiguate it when no input_json_delta was emitted.
    if not state.get("raw"):
        state["raw"] = "{}"
    state["terminal"] = True


def _strict_json_object(raw: str, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    if not isinstance(raw, str) or len(raw) > MAX_TOOL_ARGUMENT_CHARS:
        raise LLMResponseError(f"{label} exceed {MAX_TOOL_ARGUMENT_CHARS} characters")
    try:
        parsed = json.loads(raw, parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as err:
        raise LLMResponseError(f"{label} are not valid complete JSON") from err
    if not isinstance(parsed, dict):
        raise LLMResponseError(f"{label} must decode to an object")
    return parsed


def _append_tool_arguments(state: dict[str, Any], fragment: str, *, label: str) -> None:
    existing = state.get("raw", "")
    if not isinstance(existing, str):
        raise LLMResponseError(f"{label} are malformed")
    if len(existing) + len(fragment) > MAX_TOOL_ARGUMENT_CHARS:
        raise LLMResponseError(f"{label} exceed {MAX_TOOL_ARGUMENT_CHARS} characters")
    state["raw"] = existing + fragment


def _finalize_tool_states(
    states: dict[Any, dict[str, Any]],
    *,
    protocol: str,
) -> tuple[LLMToolCall, ...]:
    calls: list[LLMToolCall] = []
    seen_call_ids: set[str] = set()
    for state in sorted(states.values(), key=lambda value: value.get("order", 0)):
        call_id = state.get("call_id")
        name = state.get("name")
        raw = state.get("raw")
        if protocol == "anthropic" and not state.get("raw"):
            # Some compatibility gateways omit the per-block stop event while
            # still sending the required message_stop marker.  An empty
            # tool input is represented by the same `{}` placeholder used by
            # content_block_start, so preserve that valid zero-argument call.
            state["raw"] = "{}"
        if not isinstance(call_id, str) or not call_id:
            raise LLMResponseError(f"{protocol} tool call is missing call_id")
        if call_id in seen_call_ids:
            raise LLMResponseError(f"{protocol} emitted duplicate tool call ids")
        seen_call_ids.add(call_id)
        if not isinstance(name, str) or _TOOL_NAME_PATTERN.fullmatch(name) is None:
            raise LLMResponseError(f"{protocol} tool call is missing a valid function name")
        if not isinstance(raw, str):
            raise LLMResponseError(f"{protocol} tool arguments are missing")
        calls.append(LLMToolCall(
            call_id=call_id,
            name=name,
            arguments=_strict_json_object(raw, label=f"{protocol} tool arguments"),
            raw_arguments=raw,
        ))
    return tuple(calls)


def _ensure_tool_finish(response: LLMResponse, config: ModelConfig) -> None:
    reason = (response.finish_reason or "").strip().lower()
    allowed = {
        "openai": {"stop", "tool_calls"},
        "openai_responses": {"completed"},
        "anthropic": {"end_turn", "stop_sequence", "tool_use"},
    }[_standard_protocol(config)]
    if not reason:
        raise LLMResponseError(f"LLM tool output lacks a completion marker ({config.provider})")
    if reason not in allowed:
        raise LLMResponseError(
            f"LLM tool output ended incompletely ({config.provider} {reason!r})"
        )


def _openai_chat_response_format(
    config: ModelConfig,
    *,
    json_mode: bool,
) -> dict[str, Any] | None:
    if config.response_format is not None:
        return deepcopy(config.response_format)
    if json_mode:
        return {"type": "json_object"}
    return None


def _openai_responses_text_format(
    config: ModelConfig,
    *,
    json_mode: bool,
) -> dict[str, Any] | None:
    response_format = _openai_chat_response_format(config, json_mode=json_mode)
    if response_format is None or response_format.get("type") == "json_object":
        return response_format
    descriptor = response_format["json_schema"]
    return {"type": "json_schema", **descriptor}


def _config_token_budget(config: ModelConfig) -> int | None:
    """Return a provider-enforced upper bound for one transport attempt.

    OpenAI paths intentionally do not send output-token caps in this runtime,
    so a configured ``ModelConfig.max_tokens`` is not treated as a hard budget
    reservation there.  Anthropic always receives ``max_tokens`` and therefore
    has a bounded reservation even when its compatibility default is used.
    """
    protocol = _standard_protocol(config)
    if protocol == "anthropic":
        return _anthropic_max_tokens(config)
    return None


def _usage_value(usage: dict[str, int] | None, key: str) -> int | None:
    if not isinstance(usage, dict) or key not in usage:
        return None
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _join_instructions(*parts: str | None) -> str:
    return "\n\n".join(str(part).strip() for part in parts if part and str(part).strip())


def _openai_response_input(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    input_items: list[dict[str, str]] = []
    instruction_parts: list[str] = []
    for message in messages:
        role = (message.get("role") or "user").lower()
        content = str(message.get("content") or "")
        if role in {"system", "developer"}:
            if content.strip():
                instruction_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        input_items.append({"role": role, "content": content})
    return input_items, _join_instructions(*instruction_parts)


def _anthropic_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    anthropic_messages: list[dict[str, str]] = []
    system_parts: list[str] = []
    for message in messages:
        role = (message.get("role") or "user").lower()
        content = str(message.get("content") or "")
        if role in {"system", "developer"}:
            if content.strip():
                system_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        anthropic_messages.append({"role": role, "content": content})
    return anthropic_messages, _join_instructions(*system_parts)


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, (dict, list)):
        try:
            return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as err:
            raise LLMResponseError("tool result content must be JSON-serializable") from err
    return str(content)


def _openai_tool_response_input(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Translate standard chat history, including prior tool turns, to Responses."""
    input_items: list[dict[str, Any]] = []
    instruction_parts: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise LLMResponseError(f"messages[{index}] must be an object")
        role = str(message.get("role") or "user").lower()
        content = message.get("content")
        if role in {"system", "developer"}:
            text = _message_content_text(content)
            if text.strip():
                instruction_parts.append(text)
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise LLMResponseError(f"messages[{index}] tool message lacks tool_call_id")
            input_items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": _message_content_text(content),
            })
            continue
        if role == "assistant" and message.get("tool_calls") is not None:
            if content not in (None, ""):
                input_items.append({
                    "role": "assistant",
                    "content": deepcopy(content),
                })
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                raise LLMResponseError(f"messages[{index}].tool_calls must be a list")
            for call_index, call in enumerate(calls):
                if not isinstance(call, dict):
                    raise LLMResponseError(
                        f"messages[{index}].tool_calls[{call_index}] must be an object"
                    )
                function = call.get("function")
                call_id = call.get("id")
                if not isinstance(function, dict) or not isinstance(call_id, str) or not call_id:
                    raise LLMResponseError("assistant tool call has invalid id/function shape")
                name = function.get("name")
                arguments = function.get("arguments")
                if not isinstance(name, str) or not name or not isinstance(arguments, str):
                    raise LLMResponseError("assistant tool call has invalid name/arguments")
                input_items.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                })
            continue
        normalized_role = role if role in {"user", "assistant"} else "user"
        input_items.append({
            "role": normalized_role,
            "content": deepcopy(content) if content is not None else "",
        })
    return input_items, _join_instructions(*instruction_parts)


def _anthropic_tool_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Translate standard chat history, including tool results, to Messages."""
    result: list[dict[str, Any]] = []
    system_parts: list[str] = []

    def append_message(role: str, content: Any) -> None:
        if result and result[-1]["role"] == role:
            previous = result[-1]["content"]
            if not isinstance(previous, list):
                previous = [{"type": "text", "text": str(previous)}]
                result[-1]["content"] = previous
            if isinstance(content, list):
                previous.extend(content)
            else:
                previous.append({"type": "text", "text": str(content)})
        else:
            result.append({"role": role, "content": content})

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise LLMResponseError(f"messages[{index}] must be an object")
        role = str(message.get("role") or "user").lower()
        content = message.get("content")
        if role in {"system", "developer"}:
            text = _message_content_text(content)
            if text.strip():
                system_parts.append(text)
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise LLMResponseError(f"messages[{index}] tool message lacks tool_call_id")
            block = {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": deepcopy(content) if content is not None else "",
            }
            append_message("user", [block])
            continue
        if role == "assistant" and message.get("tool_calls") is not None:
            blocks: list[dict[str, Any]] = []
            if content not in (None, ""):
                blocks.append({"type": "text", "text": _message_content_text(content)})
            calls = message.get("tool_calls")
            if not isinstance(calls, list):
                raise LLMResponseError(f"messages[{index}].tool_calls must be a list")
            for call_index, call in enumerate(calls):
                if not isinstance(call, dict):
                    raise LLMResponseError(
                        f"messages[{index}].tool_calls[{call_index}] must be an object"
                    )
                function = call.get("function")
                call_id = call.get("id")
                if not isinstance(function, dict) or not isinstance(call_id, str) or not call_id:
                    raise LLMResponseError("assistant tool call has invalid id/function shape")
                name = function.get("name")
                raw_arguments = function.get("arguments")
                if not isinstance(name, str) or not name or not isinstance(raw_arguments, str):
                    raise LLMResponseError("assistant tool call has invalid name/arguments")
                blocks.append({
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": _strict_json_object(raw_arguments, label="assistant tool arguments"),
                })
            append_message("assistant", blocks)
            continue
        normalized_role = role if role in {"user", "assistant"} else "user"
        append_message(
            normalized_role,
            deepcopy(content) if content is not None else "",
        )
    return result, _join_instructions(*system_parts)


def _anthropic_max_tokens(config: ModelConfig) -> int:
    return config.max_tokens if config.max_tokens and config.max_tokens > 0 else ANTHROPIC_DEFAULT_MAX_TOKENS


def _llm_call_trace(
    *,
    messages: list[dict[str, str]],
    system: str | None,
    schema_hint: str | None,
    config: ModelConfig,
    response: LLMResponse,
    parse: JSONParseResult | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    request_view = _llm_request_view(messages, system=system, schema_hint=schema_hint)
    parse_view = None
    if parse is not None:
        parse_view = {
            "method": parse.method,
            "recovered": parse.recovered,
            "lossy": parse.lossy,
        }
    return {
        "call_id": response.call_id or _hash_text(
            f"{time.time_ns()}:{_stable_json(request_view)}"
        )[:16],
        "context": _safe_trace_context(context),
        "protocol": _standard_protocol(config),
        "provider": config.provider,
        "model": config.model,
        "api_base_fingerprint": _fingerprint(config.api_base),
        "request_hash": response.request_hash or _hash_json(request_view),
        "response_hash": _hash_text(response.content),
        "reasoning_hash": _hash_text(response.reasoning) if response.reasoning else None,
        "finish_reason": response.finish_reason,
        "usage": dict(response.usage),
        "latency": round(float(response.latency or 0.0), 3),
        "transport_attempt_count": len(response.transport_attempts) or 1,
        "transport_attempts": [dict(row) for row in response.transport_attempts],
        "parse": parse_view,
    }


def _tool_call_trace(
    *,
    messages: list[dict[str, Any]],
    system: str | None,
    config: ModelConfig,
    response: LLMResponse,
    tools: list[dict[str, Any]],
    tool_choice: Any,
    parallel_tool_calls: bool | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    request_view = _llm_request_view(
        messages,
        system=system,
        schema_hint=None,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
    )
    response_view = {
        "content": response.content,
        "tool_calls": [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in response.tool_calls
        ],
    }
    return {
        "call_id": response.call_id or _hash_text(
            f"{time.time_ns()}:{_stable_json(request_view)}"
        )[:16],
        "context": _safe_trace_context(context),
        "protocol": _standard_protocol(config),
        "provider": config.provider,
        "model": config.model,
        "api_base_fingerprint": _fingerprint(config.api_base),
        "request_hash": response.request_hash or _hash_json(request_view),
        "response_hash": _hash_json(response_view),
        "reasoning_hash": _hash_text(response.reasoning) if response.reasoning else None,
        "finish_reason": response.finish_reason,
        "usage": dict(response.usage),
        "latency": round(float(response.latency or 0.0), 3),
        "transport_attempt_count": len(response.transport_attempts) or 1,
        "transport_attempts": [dict(row) for row in response.transport_attempts],
        "tool_call_count": len(response.tool_calls),
        "tool_calls": [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments_hash": _hash_text(call.raw_arguments),
            }
            for call in response.tool_calls
        ],
        "parse": None,
    }


def _attach_tool_response_trace(
    err: BaseException,
    *,
    messages: list[dict[str, Any]],
    system: str | None,
    config: ModelConfig,
    response: LLMResponse,
    tools: list[dict[str, Any]],
    tool_choice: Any,
    parallel_tool_calls: bool | None,
    context: dict[str, Any] | None,
) -> None:
    setattr(err, "llm_call_trace", _tool_call_trace(
        messages=messages,
        system=system,
        config=config,
        response=response,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        context=dict(context or {}),
    ))


def _llm_request_view(
    messages: list[dict[str, Any]],
    *,
    system: str | None,
    schema_hint: str | None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    parallel_tool_calls: bool | None = None,
) -> dict[str, Any]:
    view: dict[str, Any] = {
        "system": system,
        "schema_hint": schema_hint,
        "messages": messages,
    }
    if tools is not None:
        view.update({
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        })
    return view


def _failed_llm_call_trace(
    *,
    call_id: str,
    request_hash: str,
    config: ModelConfig,
    attempts: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "context": {},
        "protocol": _protocol_or_unknown(config),
        "provider": config.provider,
        "model": config.model,
        "api_base_fingerprint": _fingerprint(config.api_base),
        "request_hash": request_hash,
        "response_hash": None,
        "reasoning_hash": None,
        "finish_reason": None,
        "usage": {},
        "latency": round(max(0.0, float(elapsed_seconds)), 3),
        "transport_attempt_count": len(attempts),
        "transport_attempts": [dict(row) for row in attempts],
        "parse": None,
    }


def _attach_response_trace(
    err: BaseException,
    *,
    messages: list[dict[str, str]],
    system: str | None,
    schema_hint: str | None,
    config: ModelConfig,
    response: LLMResponse,
    parse: JSONParseResult | None,
    context: dict[str, Any] | None,
) -> None:
    setattr(err, "llm_call_trace", _llm_call_trace(
        messages=messages,
        system=system,
        schema_hint=schema_hint,
        config=config,
        response=response,
        parse=parse,
        context=dict(context or {}),
    ))


def _attach_trace_context(
    err: BaseException,
    context: dict[str, Any] | None,
) -> None:
    trace = getattr(err, "llm_call_trace", None)
    if not isinstance(trace, dict):
        return
    updated = dict(trace)
    updated["context"] = _safe_trace_context(context or {})
    setattr(err, "llm_call_trace", updated)


def _protocol_or_unknown(config: ModelConfig) -> str:
    try:
        return _standard_protocol(config)
    except LLMError:
        return "unknown"


def _structured_error_payload(value: Any) -> dict[str, Any]:
    """Project an SDK/event error to status-bearing, non-message fields."""
    raw = _object_to_dict(value)
    if not raw:
        return {}
    projected: dict[str, Any] = {}
    for key in (
        "type",
        "code",
        "status",
        "status_code",
        "statusCode",
        "http_status",
        "httpStatus",
    ):
        candidate = raw.get(key)
        if candidate is None or isinstance(candidate, (dict, list, tuple, set)):
            continue
        projected[key] = candidate
    nested = _object_to_dict(raw.get("error"))
    if nested:
        projected["error"] = _structured_error_payload(nested)
    return projected


def _provider_error_text(payload: dict[str, Any]) -> str:
    """Return a bounded status/type label without exposing provider messages."""
    candidates = [payload]
    nested = _object_to_dict(payload.get("error"))
    if nested:
        candidates.append(nested)
    for item in reversed(candidates):
        for key in ("type", "code", "status"):
            value = item.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()[:120]
    return "provider error"


def _coerce_http_status(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if 100 <= parsed <= 599 else None


def _semantic_error_status(value: Any) -> int | None:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if "rate_limit" in normalized or "ratelimit" in normalized:
        return 429
    if "overloaded" in normalized:
        return 529
    if normalized == "api_error" or normalized.endswith("server_error"):
        return 500
    return None


def _safe_status_code(err: BaseException) -> int | None:
    for current in _error_chain(err):
        successful_transport_status: int | None = None
        status = getattr(current, "status", None)
        if status is None:
            status = getattr(current, "status_code", None)
        parsed_status = _coerce_http_status(status)
        if parsed_status is not None:
            if not 200 <= parsed_status < 300:
                return parsed_status
            successful_transport_status = parsed_status
        response = getattr(current, "response", None)
        response_status = _coerce_http_status(getattr(response, "status_code", None))
        if response_status is not None:
            if not 200 <= response_status < 300:
                return response_status
            successful_transport_status = response_status
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            candidates: list[Any] = [body]
            nested = body.get("error")
            if isinstance(nested, dict):
                candidates.append(nested)
            for item in candidates:
                for key in (
                    "status",
                    "status_code",
                    "statusCode",
                    "http_status",
                    "httpStatus",
                    "code",
                ):
                    parsed = _coerce_http_status(item.get(key))
                    if parsed is not None:
                        return parsed
                for key in ("type", "code"):
                    semantic_status = _semantic_error_status(item.get(key))
                    if semantic_status is not None:
                        return semantic_status
        if successful_transport_status is not None:
            return successful_transport_status
    return None


def _is_timeout_error(err: BaseException) -> bool:
    if bool(getattr(err, "timeout", False)):
        return True
    timeout_types = (
        asyncio.TimeoutError,
        httpx.TimeoutException,
        openai.APITimeoutError,
        anthropic.APITimeoutError,
    )
    return any(isinstance(current, timeout_types) for current in _error_chain(err))


async def _cancel_tasks_bounded(
    tasks: list[asyncio.Future[Any]] | set[asyncio.Future[Any]],
    grace_seconds: float,
) -> tuple[set[asyncio.Future[Any]], bool]:
    """Cancel twice in one grace and defer repeated caller cancellation."""
    pending = {task for task in tasks if not task.done()}
    for task in tasks:
        if task.done():
            _consume_task_result(task)
    if not pending:
        return set(), False
    loop = asyncio.get_running_loop()
    deadline = loop.time() + grace_seconds
    caller_cancelled = False
    for task in pending:
        task.cancel()
    first_budget = grace_seconds / 2
    pending, interrupted = await _wait_tasks_until(
        pending,
        loop.time() + first_budget,
    )
    caller_cancelled = caller_cancelled or interrupted
    if pending:
        for task in pending:
            task.cancel()
        pending, interrupted = await _wait_tasks_until(pending, deadline)
        caller_cancelled = caller_cancelled or interrupted
    return {task for task in pending if not task.done()}, caller_cancelled


async def _wait_tasks_until(
    tasks: set[asyncio.Future[Any]],
    deadline: float,
) -> tuple[set[asyncio.Future[Any]], bool]:
    pending = set(tasks)
    caller_cancelled = False
    while pending:
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        if remaining <= 0:
            break
        try:
            done, pending = await asyncio.wait(pending, timeout=remaining)
        except asyncio.CancelledError:
            caller_cancelled = True
            continue
        for task in done:
            _consume_task_result(task)
    return pending, caller_cancelled


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    if not task.done():
        return
    try:
        task.result()
    except BaseException:
        return


def _dispose_unstarted_awaitable(awaitable: Any) -> None:
    """Prevent an admission-rejected coroutine/future from leaking a warning."""
    if isinstance(awaitable, asyncio.Future):
        awaitable.cancel()
        return
    if inspect.iscoroutine(awaitable):
        awaitable.close()


def _set_task_name(task: asyncio.Future[Any], name: str) -> None:
    setter = getattr(task, "set_name", None)
    if callable(setter):
        setter(name)


def _bounded_duration(
    value: float,
    *,
    name: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float:
    duration = float(value)
    valid_minimum = duration >= minimum if minimum_inclusive else duration > minimum
    if not math.isfinite(duration) or not valid_minimum or duration > maximum:
        comparator = ">=" if minimum_inclusive else ">"
        raise ValueError(
            f"{name} must be finite and {comparator} {minimum:g} and <= {maximum:g}"
        )
    return duration


def _bounded_int(
    value: int,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    """Validate a finite integer resource bound before router startup."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as err:
        raise ValueError(f"{name} must be an integer") from err
    if integer != value or integer < minimum or integer > maximum:
        raise ValueError(
            f"{name} must be an integer between {minimum} and {maximum}"
        )
    return integer


def _error_chain(err: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        cause = getattr(current, "__cause__", None)
        current = cause if isinstance(cause, BaseException) else None
    return chain


def _safe_trace_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in context.items()
        if str(key) in _TRACE_CONTEXT_KEYS
        and (value is None or isinstance(value, (str, int, float, bool)))
    }


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _hash_json(value: Any) -> str:
    return _hash_text(_stable_json(value))


def _hash_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _fingerprint(value: str) -> str:
    text = (value or "").strip()
    return f"sha256:{_hash_text(text)[:16]}" if text else ""


def _openai_response_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    usage: dict[str, int] = {}
    for destination, candidates in (
        ("prompt_tokens", ("input_tokens", "prompt_tokens")),
        ("completion_tokens", ("output_tokens", "completion_tokens")),
    ):
        for candidate in candidates:
            if candidate not in value:
                continue
            raw = value.get(candidate)
            if isinstance(raw, bool):
                break
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                break
            if parsed >= 0:
                usage[destination] = parsed
            break
    return usage


def _openai_response_error_text(response: dict[str, Any]) -> str:
    for key in ("error", "incomplete_details", "status_details"):
        text = _text_from_any(response.get(key))
        if text:
            return text
    status = response.get("status")
    return str(status or "unknown response error")


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        data = value.model_dump()
        return data if isinstance(data, dict) else {}
    if hasattr(value, "dict"):
        data = value.dict()
        return data if isinstance(data, dict) else {}
    return {}


def _openai_response_output_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                        text = block.get("text")
                        if text:
                            parts.append(str(text))
            text = item.get("text")
            if text:
                parts.append(str(text))
    output_text = response.get("output_text")
    if output_text:
        parts.append(str(output_text))
    return "".join(parts)


def _openai_response_reasoning_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if "reasoning" in str(item.get("type") or ""):
                parts.append(_text_from_any(item.get("summary")))
                parts.append(_text_from_any(item.get("content")))
                parts.append(_text_from_any(item.get("text")))
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "reasoning" in str(block.get("type") or ""):
                        parts.append(_text_from_any(block.get("summary")))
                        parts.append(_text_from_any(block.get("text")))
                        parts.append(_text_from_any(block.get("content")))
    reasoning = response.get("reasoning")
    if reasoning:
        parts.append(_text_from_any(reasoning))
    return "".join(part for part in parts if part)


def _text_from_any(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "".join(_text_from_any(item) for item in value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("text", "content", "delta", "thinking", "reasoning", "summary"):
            if key in value:
                parts.append(_text_from_any(value.get(key)))
        if parts:
            return "".join(parts)
    return ""
