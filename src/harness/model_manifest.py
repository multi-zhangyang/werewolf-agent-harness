"""Credential-free LLM configuration provenance shared by harness runtimes."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, field_validator

from ..llm.models import ModelConfig
from .transcript import redact_sensitive


def _safe_api_base(value: str) -> str:
    """Return an endpoint URL without credentials, query parameters, or fragments."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    if not parsed.scheme or not parsed.hostname:
        return ""
    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, host, path, "", ""))


class ModelConfigManifest(BaseModel):
    """Safe model manifest for provenance; it never stores API keys."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    provider: str
    model: str
    api_base: str = ""
    temperature: float = 0.85
    max_tokens: int = 0
    use_json_format: bool = False
    response_format: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    configured: bool = False

    @field_validator("api_base")
    @classmethod
    def _sanitize_api_base(cls, value: str) -> str:
        return _safe_api_base(value)

    @field_validator("response_format", "reasoning", "thinking", mode="before")
    @classmethod
    def _require_json_options(cls, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        try:
            normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as err:
            raise ValueError("model request options must be JSON-serializable objects") from err
        if not isinstance(normalized, dict):
            raise ValueError("model request options must be objects")
        return normalized

    @classmethod
    def from_config(cls, config: ModelConfig) -> "ModelConfigManifest":
        return cls(
            provider=config.provider,
            model=config.model,
            api_base=_safe_api_base(config.api_base),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            use_json_format=config.use_json_format,
            # Preserve JSON-schema field names (for example a property named
            # api_key) while redacting any secret-looking text values.
            response_format=_safe_schema_options(config.response_format),
            reasoning=_safe_model_options(config.reasoning),
            thinking=_safe_model_options(config.thinking),
            configured=bool(config.model and config.api_key),
        )


def _safe_model_options(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return redact_sensitive(value)


def _safe_schema_options(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Redact schema text without treating schema property names as secrets."""
    if value is None:
        return None

    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): visit(raw_value) for key, raw_value in item.items()}
        if isinstance(item, list):
            return [visit(raw) for raw in item]
        if isinstance(item, str):
            return redact_sensitive(item)
        return item

    return visit(value)
