"""模型配置数据模型 + 多 provider 支持。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SUPPORTED_PROVIDERS = {
    "openai": "OpenAI Chat Completions",
    "openai_responses": "OpenAI Responses API",
    "anthropic": "Anthropic Messages API",
}


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
    api_key: str = ""
    temperature: float = 0.85
    max_tokens: int = 0  # 0 = 不限制(不传该参数)
    use_json_format: bool = True

    @field_validator("provider")
    @classmethod
    def _check_provider(cls, v: str) -> str:
        v = (v or "openai").lower()
        if v not in SUPPORTED_PROVIDERS:
            raise ValueError(f"不支持的 provider: {v}(仅 {list(SUPPORTED_PROVIDERS)})")
        return v

    def merge(self, override: Optional["ModelConfig | dict[str, Any]"]) -> "ModelConfig":
        """用 override 的非空字段覆盖自身(留空字段沿用默认)。

        per-seat 配置的留空字段应继承房间默认,而非清空。
        但 endpoint(provider/api_base) 被显式改动时,不得继承默认 api_key,
        否则会把后端默认 key 发给用户指定的第三方 endpoint。
        """
        if not override:
            return self.model_copy()
        ov = override if isinstance(override, ModelConfig) else ModelConfig(**override)
        provided = getattr(ov, "model_fields_set", set())
        provider_changed = "provider" in provided and ov.provider and ov.provider != self.provider
        api_base_changed = (
            "api_base" in provided
            and bool(ov.api_base)
            and _normalize_api_base(ov.api_base) != _normalize_api_base(self.api_base)
        )
        api_key_provided = "api_key" in provided and bool(ov.api_key)
        if self.api_key and (provider_changed or api_base_changed) and not api_key_provided:
            raise ValueError("修改 provider/api_base 时必须显式提供该 endpoint 的 api_key")
        merged = self.model_copy()
        for f in ("provider", "model", "api_base", "api_key"):
            val = getattr(ov, f)
            if val:
                setattr(merged, f, val)
        # 数值/布尔字段:override 显式提供才覆盖,否则继承房间默认。
        if "temperature" in provided and ov.temperature is not None:
            merged.temperature = ov.temperature
        if "max_tokens" in provided:
            merged.max_tokens = ov.max_tokens
        if "use_json_format" in provided:
            merged.use_json_format = ov.use_json_format
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
            "configured": bool(self.model and (self.api_key or self.provider)),
        }


def _normalize_api_base(value: str | None) -> str:
    return (value or "").strip().rstrip("/")
