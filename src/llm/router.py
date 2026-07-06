"""LLM 调用层 —— 多 provider 真实调用,绝不伪造。

铁律(承 [[no-fallback-design]]):
- 每个 AI 决策必须来自真实 LLM 调用,绝不 fallback 出假决策。
- 失败走深度重试(指数退避 + 抖动),彻底失败才抛 LLMError,由上层决定 skip。
- OpenAI Chat/Responses 在 max_tokens=0/None 时不传输出上限字段;
  Anthropic Messages 按官方接口要求必须传 max_tokens,0 使用本项目默认上限。
- 宽松超时(默认 180s),不误杀真实思考。
- 凭据只来自 per-seat/房间 config(WEREWOLF_ 前缀),绝不回退读系统 OPENAI_/ANTHROPIC_ env。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

import httpx

from .models import ModelConfig

logger = logging.getLogger(__name__)

# 可重试的 HTTP 状态码(瞬时错误,重试即可,不触发 skip)
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_DEFAULT_MAX_TOKENS = 8192

Provider = Literal["openai", "openai_responses", "anthropic"]


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
        }


@dataclass
class LLMResponse:
    """一次完整调用的结果。"""

    content: str
    finish_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    raw_provider: str = ""
    latency: float = 0.0
    reasoning: str = ""  # reasoning 模型的私有思考(AI思考流/复盘可见,不广播)


@dataclass
class JSONParseResult:
    """JSON 解析结果及透明审计标记。"""

    data: dict[str, Any]
    method: str
    recovered: bool = False
    lossy: bool = False


class LLMError(RuntimeError):
    """LLM 调用彻底失败(重试耗尽)。由上层决定 _legal_skip,非伪造决策。"""


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
        max_retries: int = 5,
        concurrency: int = 4,
        chunk_timeout: float = 60.0,
    ) -> None:
        self.timeout = timeout
        # timeout 是每次 attempt 的 wall-clock 总时限;chunk_timeout 是两个
        # SSE chunk 之间的帧间超时。两者分开,避免 keepalive/空 delta 让
        # 一次真实流式调用无限挂住。
        self.chunk_timeout = chunk_timeout
        self.max_retries = max(1, max_retries)
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._stats = CallStats()
        # httpx 客户端复用(按 base_url+key 缓存)。流式与非流式各一套。
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._stream_clients: dict[str, httpx.AsyncClient] = {}

    @property
    def stats(self) -> CallStats:
        return self._stats

    # ------------------------------------------------------------------
    # 客户端管理
    # ------------------------------------------------------------------
    def _client_key(self, config: ModelConfig) -> str:
        key_hash = hashlib.sha256(config.api_key.encode("utf-8")).hexdigest() if config.api_key else ""
        return f"{config.provider}|{config.api_base}|{key_hash}"

    def _get_client(self, config: ModelConfig) -> httpx.AsyncClient:
        key = self._client_key(config)
        client = self._clients.get(key)
        if client is None:
            client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=15.0))
            self._clients[key] = client
        return client

    def _get_stream_client(self, config: ModelConfig) -> httpx.AsyncClient:
        """流式专用 client:read 超时 = chunk_timeout(帧间)。

        httpx 在流式模式下 read 超时作用于每两个 chunk 之间,而非整个请求;
        每次 attempt 的总时限由 _complete 外层 asyncio.wait_for(self.timeout)
        负责。
        """
        key = self._client_key(config)
        client = self._stream_clients.get(key)
        if client is None:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.chunk_timeout, connect=15.0, write=15.0, pool=15.0)
            )
            self._stream_clients[key] = client
        return client

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        for client in self._stream_clients.values():
            await client.aclose()
        self._clients.clear()
        self._stream_clients.clear()

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
    ) -> dict[str, Any]:
        """调用 LLM 并解析为 JSON dict。

        use_json_format=True 时要求 response_format=json_object(网关支持则用,降级则靠解析)。
        失败抛 LLMError(上层决定 skip),绝不返回伪造结果。
        默认拒绝有损 JSON 恢复,避免截断输出被静默当成完整决策。
        """
        resp = await self._complete(messages, config, system=system, schema_hint=schema_hint, json_mode=config.use_json_format)
        data = self._parse_json(
            resp.content,
            config,
            allow_lossy=allow_lossy,
            include_parse_metadata=include_parse_metadata,
        )
        if resp.reasoning and not str(data.get("thought") or "").strip():
            data["thought"] = resp.reasoning
        return data

    async def complete_text(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
        *,
        system: str | None = None,
    ) -> str:
        """纯文本调用(发言/遗言等自由文本)。"""
        resp = await self._complete(messages, config, system=system, json_mode=False)
        return resp.content

    async def stream_text(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
        *,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """流式文本(AI 思考摘要实时推送前端)。

        降级策略:网关不支持流式则一次性返回。
        """
        if config.provider == "anthropic":
            async for chunk in self._stream_anthropic(messages, config, system):
                yield chunk
        elif config.provider == "openai_responses":
            async for chunk in self._stream_openai_responses(messages, config, system):
                yield chunk
        else:
            async for chunk in self._stream_openai(messages, config, system):
                yield chunk

    # ------------------------------------------------------------------
    # 核心调用(带重试)
    # ------------------------------------------------------------------
    async def _complete(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
        *,
        system: str | None = None,
        schema_hint: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        last_err: Exception | None = None
        retries = 0
        start = time.monotonic()

        async with self._sem:
            for attempt in range(self.max_retries):
                try:
                    if config.provider == "anthropic":
                        call = self._call_anthropic(messages, config, system, schema_hint, json_mode)
                    elif config.provider == "openai_responses":
                        call = self._call_openai_responses(messages, config, system, schema_hint, json_mode)
                    else:
                        call = self._call_openai(messages, config, system, schema_hint, json_mode)
                    try:
                        if self.timeout and self.timeout > 0:
                            resp = await asyncio.wait_for(call, timeout=self.timeout)
                        else:
                            resp = await call
                    except asyncio.TimeoutError as err:
                        raise LLMError(
                            f"LLM 调用总超时(provider={config.provider} >{self.timeout:.1f}s)"
                        ) from err
                    self._stats.record(
                        ok=True,
                        retries=retries,
                        latency=time.monotonic() - start,
                        tok_in=resp.usage.get("prompt_tokens", 0),
                        tok_out=resp.usage.get("completion_tokens", 0),
                    )
                    return resp
                except Exception as err:  # noqa: BLE001 — 统一重试
                    last_err = err
                    if attempt < self.max_retries - 1 and self._is_retryable(err):
                        retries += 1
                        delay = self._backoff_delay(attempt)
                        logger.warning(
                            "LLM 调用失败(provider=%s attempt=%d/%d %.2fs后重试): %s",
                            config.provider,
                            attempt + 1,
                            self.max_retries,
                            delay,
                            err,
                        )
                        await asyncio.sleep(delay)
                        continue
                    break  # 不可重试或重试耗尽

        self._stats.record(ok=False, retries=retries, latency=time.monotonic() - start)
        raise LLMError(f"LLM 调用彻底失败(provider={config.provider} retries={retries}): {last_err}") from last_err

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
        """流式调用 OpenAI Chat Completions 格式,边收边累积。

        stream:True 让网关持续推送 SSE chunk,连接始终有数据流动,
        不会触发中间层(LB/反代)的 idle 超时。
        """
        url = _openai_chat_completions_url(config.api_base)
        full_messages: list[dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        if schema_hint:
            full_messages.append(
                {"role": "system", "content": f"必须以严格 JSON 格式响应。结构要求:\n{schema_hint}"}
            )
        full_messages.extend(messages)
        if json_mode and not any("json" in str(message.get("content", "")).lower() for message in full_messages):
            full_messages.insert(0, {"role": "system", "content": "请输出严格 JSON 对象。"})

        payload: dict[str, Any] = {
            "model": config.model,
            "messages": full_messages,
            "temperature": config.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # Chat Completions:0/None 不传输出上限字段。
        if config.max_tokens and config.max_tokens > 0:
            payload["max_completion_tokens"] = config.max_tokens
        # json_object 结构化输出(网关支持时;流式下部分网关会忽略,靠 _parse_json 容错)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        client = self._get_stream_client(config)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = "stop"
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

        try:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    text = body.decode("utf-8", "replace")[:500]
                    if resp.status_code in _RETRYABLE_STATUS:
                        raise _RetryableHTTP(resp.status_code, text)
                    raise LLMError(f"stream HTTP {resp.status_code}: {text}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    # usage 可能在最后一个 chunk(stream_options.include_usage)
                    u = obj.get("usage")
                    if u:
                        usage["prompt_tokens"] = u.get("prompt_tokens", 0)
                        usage["completion_tokens"] = u.get("completion_tokens", 0)
                    choices = obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        piece = delta.get("content")
                        if piece:
                            content_parts.append(piece)
                        reasoning_piece = _openai_chat_reasoning_delta(delta)
                        if reasoning_piece:
                            reasoning_parts.append(reasoning_piece)
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr
        except httpx.TimeoutException as err:
            raise LLMError(f"stream 帧间超时(>{self.chunk_timeout}s 无数据): {err}") from err
        except httpx.HTTPError as err:
            raise LLMError(f"stream 网络错误: {err}") from err

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)

        return LLMResponse(
            content=content,
            reasoning=reasoning,
            finish_reason=finish_reason,
            usage=usage,
            raw_provider="openai",
        )

    async def _stream_openai(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
        system: str | None,
    ) -> AsyncIterator[str]:
        url = _openai_chat_completions_url(config.api_base)
        full_messages: list[dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": full_messages,
            "temperature": config.temperature,
            "stream": True,
        }
        if config.max_tokens and config.max_tokens > 0:
            payload["max_completion_tokens"] = config.max_tokens
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        client = self._get_stream_client(config)

        try:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise LLMError(f"stream HTTP {resp.status_code}: {body[:300]!r}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        piece = delta.get("content")
                        if piece:
                            yield piece
        except httpx.HTTPError as err:
            raise LLMError(f"stream 网络错误: {err}") from err

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
        """流式调用 OpenAI Responses API。

        按 SDK 语义使用 input/instructions/text.format/max_output_tokens,
        不混用 Chat Completions 的 messages/response_format/max_tokens。
        """
        url = _openai_responses_url(config.api_base)
        input_items, extracted_instructions = _openai_response_input(messages)
        instructions = _join_instructions(
            system,
            extracted_instructions,
            schema_hint and f"必须以严格 JSON 格式响应。结构要求:\n{schema_hint}",
        )
        payload: dict[str, Any] = {
            "model": config.model,
            "input": input_items,
            "temperature": config.temperature,
            "stream": True,
        }
        if instructions:
            payload["instructions"] = instructions
        if config.max_tokens and config.max_tokens > 0:
            payload["max_output_tokens"] = config.max_tokens
        if json_mode:
            payload["text"] = {"format": {"type": "json_object"}}
            if not any("json" in str(item.get("content", "")).lower() for item in input_items):
                input_items.insert(0, {"role": "user", "content": "请输出严格 JSON 对象。"})

        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        client = self._get_stream_client(config)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = "completed"
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

        try:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    text = body.decode("utf-8", "replace")[:500]
                    if resp.status_code in _RETRYABLE_STATUS:
                        raise _RetryableHTTP(resp.status_code, text)
                    raise LLMError(f"responses stream HTTP {resp.status_code}: {text}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    etype = str(obj.get("type") or "")
                    if etype == "response.output_text.delta":
                        piece = obj.get("delta")
                        if piece:
                            content_parts.append(str(piece))
                    elif "reasoning" in etype and etype.endswith(".delta"):
                        piece = _text_from_any(obj.get("delta") or obj.get("text"))
                        if piece:
                            reasoning_parts.append(piece)
                    elif etype == "response.completed":
                        response = obj.get("response") if isinstance(obj.get("response"), dict) else {}
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
                        response = obj.get("response") if isinstance(obj.get("response"), dict) else {}
                        finish_reason = str(response.get("status") or "incomplete")
                        usage.update(_openai_response_usage(response.get("usage")))
                    elif etype == "error":
                        raise LLMError(f"responses stream error: {obj.get('error') or obj}")
        except httpx.TimeoutException as err:
            raise LLMError(f"responses stream 帧间超时(>{self.chunk_timeout}s 无数据): {err}") from err
        except httpx.HTTPError as err:
            raise LLMError(f"responses stream 网络错误: {err}") from err

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        return LLMResponse(
            content=content,
            reasoning=reasoning,
            finish_reason=finish_reason,
            usage=usage,
            raw_provider="openai_responses",
        )

    async def _stream_openai_responses(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
        system: str | None,
    ) -> AsyncIterator[str]:
        url = _openai_responses_url(config.api_base)
        input_items, extracted_instructions = _openai_response_input(messages)
        payload: dict[str, Any] = {
            "model": config.model,
            "input": input_items,
            "temperature": config.temperature,
            "stream": True,
        }
        instructions = _join_instructions(system, extracted_instructions)
        if instructions:
            payload["instructions"] = instructions
        if config.max_tokens and config.max_tokens > 0:
            payload["max_output_tokens"] = config.max_tokens

        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
        client = self._get_stream_client(config)
        try:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise LLMError(f"responses stream HTTP {resp.status_code}: {body[:300]!r}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") == "response.output_text.delta":
                        piece = obj.get("delta")
                        if piece:
                            yield str(piece)
                    elif obj.get("type") == "error":
                        raise LLMError(f"responses stream error: {obj.get('error') or obj}")
        except httpx.HTTPError as err:
            raise LLMError(f"responses stream 网络错误: {err}") from err

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
        """流式调用 Anthropic Messages API,边收边累积,避免长连接 idle 断连。"""
        url = _anthropic_messages_url(config.api_base)
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
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": anthropic_messages,
            "temperature": config.temperature,
            "max_tokens": _anthropic_max_tokens(config),
            "stream": True,
        }
        if sys_parts:
            payload["system"] = "\n\n".join(sys_parts)
        headers = {
            "x-api-key": config.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        client = self._get_stream_client(config)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = "end_turn"
        in_tok = out_tok = 0

        try:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    text = body.decode("utf-8", "replace")[:500]
                    if resp.status_code in _RETRYABLE_STATUS:
                        raise _RetryableHTTP(resp.status_code, text)
                    raise LLMError(f"anthropic stream HTTP {resp.status_code}: {text}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        obj = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    etype = obj.get("type")
                    if etype == "content_block_delta":
                        delta = obj.get("delta", {})
                        if delta.get("type") == "text_delta":
                            piece = delta.get("text")
                            if piece:
                                content_parts.append(piece)
                        # Anthropic official thinking streams use thinking_delta; reasoning_delta is a tolerated gateway alias.
                        elif delta.get("type") in {"thinking_delta", "reasoning_delta"}:
                            piece = _text_from_any(delta.get("thinking") or delta.get("reasoning") or delta.get("text"))
                            if piece:
                                reasoning_parts.append(piece)
                    elif etype == "message_delta":
                        delta = obj.get("delta", {})
                        if delta.get("stop_reason"):
                            finish_reason = delta["stop_reason"]
                        u = obj.get("usage")
                        if u:
                            out_tok = u.get("output_tokens", out_tok)
                    elif etype == "message_start":
                        m = obj.get("message", {})
                        u = m.get("usage", {})
                        in_tok = u.get("input_tokens", in_tok)
                        out_tok = u.get("output_tokens", out_tok)
                    elif etype == "error":
                        raise LLMError(f"anthropic stream error: {obj.get('error') or obj}")
        except httpx.TimeoutException as err:
            raise LLMError(f"anthropic stream 帧间超时(>{self.chunk_timeout}s 无数据): {err}") from err
        except httpx.HTTPError as err:
            raise LLMError(f"anthropic stream 网络错误: {err}") from err

        return LLMResponse(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            finish_reason=finish_reason,
            usage={"prompt_tokens": in_tok, "completion_tokens": out_tok},
            raw_provider="anthropic",
        )

    async def _stream_anthropic(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
        system: str | None,
    ) -> AsyncIterator[str]:
        url = _anthropic_messages_url(config.api_base)
        anthropic_messages, extracted_system = _anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": anthropic_messages,
            "temperature": config.temperature,
            "max_tokens": _anthropic_max_tokens(config),
            "stream": True,
        }
        system_text = _join_instructions(system, extracted_system)
        if system_text:
            payload["system"] = system_text
        headers = {
            "x-api-key": config.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        client = self._get_stream_client(config)
        try:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise LLMError(f"anthropic stream HTTP {resp.status_code}: {body[:300]!r}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    try:
                        obj = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") == "content_block_delta":
                        delta = obj.get("delta", {})
                        if delta.get("type") == "text_delta":
                            piece = delta.get("text")
                            if piece:
                                yield piece
                    elif obj.get("type") == "error":
                        raise LLMError(f"anthropic stream error: {obj.get('error') or obj}")
        except httpx.HTTPError as err:
            raise LLMError(f"anthropic stream 网络错误: {err}") from err

    # ------------------------------------------------------------------
    # HTTP 工具
    # ------------------------------------------------------------------
    async def _http_post(self, url: str, payload: dict, headers: dict, config: ModelConfig) -> dict:
        client = self._get_client(config)
        try:
            resp = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as err:
            raise LLMError(f"网络错误: {err}") from err
        if resp.status_code != 200:
            body = resp.text[:500]
            if resp.status_code in _RETRYABLE_STATUS:
                raise _RetryableHTTP(resp.status_code, body)
            raise LLMError(f"HTTP {resp.status_code}: {body}")
        try:
            return resp.json()
        except Exception as err:  # noqa: BLE001
            raise LLMError(f"响应非 JSON: {resp.text[:300]!r}") from err

    # ------------------------------------------------------------------
    # 重试与解析工具
    # ------------------------------------------------------------------
    @staticmethod
    def _is_retryable(err: Exception) -> bool:
        if isinstance(err, _RetryableHTTP):
            return True
        if isinstance(err, httpx.HTTPError):
            return True
        if isinstance(err, httpx.TimeoutException):
            return True
        if isinstance(err, asyncio.TimeoutError):
            return True
        # LLMError 包装的瞬时错误也允许重试(除非是不可恢复的 4xx)。
        # 流式路径把 httpx 网络错误(incomplete chunked read / 连接重置等)包成 LLMError,
        # 这里看 __cause__ 是否为可重试的 httpx 错误,是则重试。
        cause = getattr(err, "__cause__", None)
        if cause is not None and isinstance(
            cause,
            (httpx.HTTPError, httpx.TimeoutException, asyncio.TimeoutError, _RetryableHTTP),
        ):
            return True
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
        allow_lossy=True,否则抛 LLMError 触发上层重试。
        """
        result = LLMRouter._parse_json_result(content, config)
        if result.lossy and not allow_lossy:
            raise LLMError(
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
        """容错解析并返回恢复方式。失败抛 LLMError。"""
        import ast

        text = content.strip()
        if text.startswith("```"):
            parts = text.split("```", 2)
            text = parts[1] if len(parts) > 1 else content
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        # 1) 标准 JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return JSONParseResult(obj, method="json")
            raise LLMError(f"JSON 解析结果非对象(provider={config.provider}): {type(obj)}")
        except json.JSONDecodeError:
            pass

        # 2) 单引号 JSON 或 Python 字面量
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return JSONParseResult(obj, method="literal", recovered=True)
            raise LLMError(f"JSON 解析结果非对象(provider={config.provider}): {type(obj)}")
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
                    raise LLMError(f"JSON 解析结果非对象(provider={config.provider}): {type(obj)}")
                except json.JSONDecodeError:
                    pass
                try:
                    obj = ast.literal_eval(snippet)
                    if isinstance(obj, dict):
                        return JSONParseResult(obj, method="embedded_literal", recovered=True)
                    raise LLMError(f"JSON 解析结果非对象(provider={config.provider}): {type(obj)}")
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
                    raise LLMError(f"JSON 解析结果非对象(provider={config.provider}): {type(obj)}")
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

        raise LLMError(f"JSON 解析失败(provider={config.provider}): {content[:300]!r}")


class _RetryableHTTP(Exception):
    """可重试的 HTTP 状态错误。"""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:200]}")


def _openai_chat_completions_url(api_base: str) -> str:
    base = (api_base or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _openai_responses_url(api_base: str) -> str:
    base = (api_base or "").strip().rstrip("/")
    if base.endswith("/responses"):
        return base
    return base + "/responses"


def _anthropic_messages_url(api_base: str) -> str:
    base = (api_base or "").strip().rstrip("/")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return base + "/messages"
    return base + "/v1/messages"


def _join_instructions(*parts: str | None) -> str:
    return "\n\n".join(str(part).strip() for part in parts if part and str(part).strip())


def _openai_chat_reasoning_delta(delta: dict[str, Any]) -> str:
    """Extract reasoning text from OpenAI-compatible Chat Completions deltas.

    Several OpenAI-compatible providers expose model reasoning in non-content
    delta fields while preserving the Chat Completions envelope. Treat these as
    standard optional reasoning parts and keep them separate from final content.
    """
    parts: list[str] = []
    for key in ("reasoning_content", "reasoning", "reasoning_delta"):
        value = delta.get(key)
        text = _text_from_any(value)
        if text:
            parts.append(text)
    return "".join(parts)


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


def _anthropic_max_tokens(config: ModelConfig) -> int:
    return config.max_tokens if config.max_tokens and config.max_tokens > 0 else ANTHROPIC_DEFAULT_MAX_TOKENS


def _openai_response_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0}
    return {
        "prompt_tokens": int(value.get("input_tokens") or value.get("prompt_tokens") or 0),
        "completion_tokens": int(value.get("output_tokens") or value.get("completion_tokens") or 0),
    }


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
