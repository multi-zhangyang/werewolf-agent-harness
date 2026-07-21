"""模型配置数据模型。

provider 字段表示标准协议选择(openai/openai_responses/anthropic),不是供应商白名单。
模型层只做字符串归一化;实际调用层按标准协议执行或报错。
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Optional

from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConfig(BaseModel):
    """单个座位/房间的 LLM 配置。

    凭据只存在于内存房间对象,不落盘。
    provider 决定走哪个标准 API 路径:
    openai=Chat Completions, openai_responses=Responses API, anthropic=Messages API。
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = "openai"
    model: str = ""
    api_base: str = ""
    api_key: str = Field(default="", repr=False)
    temperature: float = 0.85
    # OpenAI paths omit a cap. Anthropic maps 0 to its required compatibility default.
    max_tokens: int = 0
    use_json_format: bool = True
    # Standard OpenAI Chat Completions response_format. ``json_schema`` is
    # translated to the corresponding Responses API ``text.format`` shape.
    response_format: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None  # OpenAI Responses API standard reasoning config.
    thinking: dict[str, Any] | None = None  # Anthropic Messages API standard thinking config.

    @field_validator("provider")
    @classmethod
    def _normalize_provider(cls, v: str) -> str:
        return (v or "openai").strip().lower()

    @field_validator("response_format")
    @classmethod
    def _validate_response_format(
        cls,
        value: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        try:
            encoded = json.dumps(value, ensure_ascii=True, allow_nan=False)
            normalized = json.loads(encoded)
        except (TypeError, ValueError) as err:
            raise ValueError("response_format must be a JSON-serializable object") from err
        if len(encoded) > 65_536:
            raise ValueError("response_format exceeds the 65536-byte limit")
        if not isinstance(normalized, dict):
            raise ValueError("response_format must be an object")

        format_type = normalized.get("type")
        if format_type == "json_object":
            if set(normalized) != {"type"}:
                raise ValueError("json_object response_format only accepts the type field")
            return {"type": "json_object"}
        if format_type != "json_schema" or set(normalized) != {"type", "json_schema"}:
            raise ValueError(
                "response_format must be json_object or a standard json_schema descriptor"
            )

        descriptor = normalized.get("json_schema")
        if not isinstance(descriptor, dict):
            raise ValueError("json_schema response_format requires a json_schema object")
        allowed = {"name", "description", "schema", "strict"}
        if set(descriptor) - allowed:
            raise ValueError("json_schema descriptor contains unsupported fields")
        name = descriptor.get("name")
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) is None:
            raise ValueError("json_schema name must match [A-Za-z0-9_-]{1,64}")
        schema = descriptor.get("schema")
        if not isinstance(schema, dict):
            raise ValueError("json_schema schema must be an object")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError:
            raise ValueError(
                "json_schema schema must be valid Draft 2020-12 JSON Schema"
            ) from None
        strict = descriptor.get("strict", True)
        if not isinstance(strict, bool):
            raise ValueError("json_schema strict must be a boolean")
        description = descriptor.get("description")
        if description is not None and (
            not isinstance(description, str) or len(description) > 1024
        ):
            raise ValueError("json_schema description must be a string of at most 1024 characters")

        safe_descriptor: dict[str, Any] = {
            "name": name,
            "schema": schema,
            "strict": strict,
        }
        if description:
            safe_descriptor["description"] = description
        return {"type": "json_schema", "json_schema": safe_descriptor}

    def merge(self, override: Optional["ModelConfig | dict[str, Any]"]) -> "ModelConfig":
        """用 override 的非空字段覆盖自身(留空字段沿用默认)。

        per-seat 配置的留空字段默认继承房间配置;但 provider/api_base 是
        凭据边界。一旦座位显式切换协议或 endpoint,不能把房间 key 静默
        带到新的目标。
        """
        if not override:
            return self.model_copy(deep=True)
        ov = (
            override.model_copy(deep=True)
            if isinstance(override, ModelConfig)
            else ModelConfig(**override)
        )
        provided = getattr(ov, "model_fields_set", set())
        merged = self.model_copy(deep=True)
        endpoint_boundary_changed = False
        if "provider" in provided and ov.provider and ov.provider != self.provider:
            endpoint_boundary_changed = True
        if "api_base" in provided and ov.api_base and ov.api_base != self.api_base:
            endpoint_boundary_changed = True
        for f in ("provider", "model", "api_base", "api_key"):
            if f not in provided:
                continue
            val = getattr(ov, f)
            if val:
                setattr(merged, f, val)
        if endpoint_boundary_changed and "api_key" not in provided:
            merged.api_key = ""
        # 数值/布尔字段:override 显式提供才覆盖,否则继承房间默认。
        if "temperature" in provided and ov.temperature is not None:
            merged.temperature = ov.temperature
        if "max_tokens" in provided:
            merged.max_tokens = ov.max_tokens
        if "use_json_format" in provided:
            merged.use_json_format = ov.use_json_format
        if "response_format" in provided:
            merged.response_format = deepcopy(ov.response_format)
        if "reasoning" in provided:
            merged.reasoning = deepcopy(ov.reasoning)
        if "thinking" in provided:
            merged.thinking = deepcopy(ov.thinking)
        return merged

    def safe_view(self) -> dict[str, Any]:
        """对外展示视图(隐藏 api_key)。"""
        return {
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_json_format": self.use_json_format,
            "response_format": self.response_format,
            "reasoning": self.reasoning,
            "thinking": self.thinking,
            "configured": bool(self.model and self.api_key),
        }
