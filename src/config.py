"""配置:仅读取 WEREWOLF_* 前缀环境变量。

铁律:绝不读取/写入系统 OPENAI_API_KEY / ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL
(那些被其他进程 / Claude Code 自身占用)。本项目一切配置走 WEREWOLF_ 前缀。
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values

from .api.limits import ProviderBudgetPolicy, RateLimitConfig


def _load_werewolf_dotenv() -> None:
    """只导入 .env 中的 WEREWOLF_* 变量,不污染 OPENAI_/ANTHROPIC_。"""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for key, value in dotenv_values(env_path).items():
        if key.startswith("WEREWOLF_") and value is not None and key not in os.environ:
            os.environ[key] = value


_load_werewolf_dotenv()


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _optional_json_object(key: str) -> dict | None:
    raw = _get(key, "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ValueError(f"{key} must contain a JSON object") from err
    if not isinstance(value, dict):
        raise ValueError(f"{key} must contain a JSON object")
    return value


def parse_cors_origins(value: str) -> tuple[str, ...]:
    """Parse a comma-separated list of exact HTTP(S) origins.

    CORS origins are deliberately limited to scheme + host + optional port.
    Paths, credentials, wildcards, and malformed ports are rejected at
    process startup instead of broadening browser access accidentally.
    """
    parts = [part.strip() for part in str(value).split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("WEREWOLF_CORS_ORIGINS must contain one or more exact origins")

    origins: list[str] = []
    for origin in parts:
        if "*" in origin:
            raise ValueError("WEREWOLF_CORS_ORIGINS does not permit wildcards")
        try:
            parsed = urlsplit(origin)
            hostname = parsed.hostname
            parsed.port  # Validate the port even though urlsplit is otherwise permissive.
        except ValueError as err:
            raise ValueError(f"invalid CORS origin: {origin!r}") from err
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.netloc.endswith(":")
            or parsed.path
            or parsed.query
            or parsed.fragment
            or origin.lower() != f"{parsed.scheme}://{parsed.netloc}".lower()
        ):
            raise ValueError(f"invalid CORS origin: {origin!r}")
        normalized = urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), "", "", ""))
        if normalized not in origins:
            origins.append(normalized)
    return tuple(origins)


def _strict_bool(key: str, default: str) -> bool:
    raw = _get(key, default).strip().lower()
    if raw not in {"true", "false"}:
        raise ValueError(f"{key} must be 'true' or 'false'")
    return raw == "true"


def _positive_int(key: str, default: str) -> int:
    try:
        value = int(_get(key, default))
    except ValueError as err:
        raise ValueError(f"{key} must be a positive integer") from err
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _non_negative_float(key: str, default: str) -> float:
    try:
        value = float(_get(key, default))
    except ValueError as err:
        raise ValueError(f"{key} must be a non-negative finite number") from err
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"{key} must be a non-negative finite number")
    return value


def _env_first(keys: tuple[str, ...], default: str) -> str:
    """Read the first explicitly-set alias without exposing its value."""
    for key in keys:
        if key in os.environ:
            return os.environ[key]
    return default


def _non_negative_int_alias(keys: tuple[str, ...], default: str = "0") -> int | None:
    raw = _env_first(keys, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{keys[0]} must be a non-negative integer") from err
    if value < 0:
        raise ValueError(f"{keys[0]} must be a non-negative integer")
    # Zero is the explicit configuration spelling for an unlimited dimension.
    return None if value == 0 else value


def _positive_float_alias(keys: tuple[str, ...], default: str) -> float:
    raw = _env_first(keys, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{keys[0]} must be a positive finite number") from err
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"{keys[0]} must be a positive finite number")
    return value


def _positive_int_alias(keys: tuple[str, ...], default: str) -> int:
    raw = _env_first(keys, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{keys[0]} must be a positive integer") from err
    if value <= 0:
        raise ValueError(f"{keys[0]} must be a positive integer")
    return value


# —— 服务 ——
HOST: str = _get("WEREWOLF_HOST", "127.0.0.1")
PORT: int = int(_get("WEREWOLF_PORT", "8000"))
LOG_LEVEL: str = _get("WEREWOLF_LOG_LEVEL", "info")
# The production FastAPI build is served on the API port itself. Include that
# same-origin browser surface by default while retaining the Vite dev origins;
# public deployments should still set an explicit exact-origin allowlist.
_DEFAULT_LOCAL_CORS_ORIGINS = ",".join(
    (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        f"http://127.0.0.1:{PORT}",
        f"http://localhost:{PORT}",
    )
)
CORS_ORIGINS: tuple[str, ...] = parse_cors_origins(
    _get("WEREWOLF_CORS_ORIGINS", _DEFAULT_LOCAL_CORS_ORIGINS)
)
CORS_ALLOW_CREDENTIALS: bool = _strict_bool("WEREWOLF_CORS_ALLOW_CREDENTIALS", "false")
if CORS_ALLOW_CREDENTIALS and "*" in CORS_ORIGINS:
    # Kept as a separate invariant so a future parser relaxation cannot create
    # the unsafe wildcard + credential combination accepted by some stacks.
    raise ValueError("credentialed CORS cannot be combined with a wildcard origin")
# Browser WebSocket upgrades carry Origin, while native/TestClient clients may
# omit it. Keep the omission policy explicit so public deployments can require
# an Origin header instead of inheriting an accidental default.
WS_ALLOW_MISSING_ORIGIN: bool = _strict_bool("WEREWOLF_WS_ALLOW_MISSING_ORIGIN", "true")

# —— 房间资源治理 ——
MAX_ROOMS: int = _positive_int("WEREWOLF_MAX_ROOMS", "64")
# 0 disables automatic TTL cleanup; explicit authenticated cleanup remains.
TERMINAL_ROOM_TTL: float = _non_negative_float("WEREWOLF_TERMINAL_ROOM_TTL", "3600")

# —— HTTP/WebSocket admission and provider spend governance ——
# These are process-local controls.  A zero provider dimension means
# unlimited; rate buckets are always enabled with a bounded default burst.
REST_RATE_LIMIT_CONFIG = RateLimitConfig(
    capacity=_positive_float_alias(
        ("WEREWOLF_REST_RATE_LIMIT_CAPACITY", "WEREWOLF_RATE_LIMIT_REST_CAPACITY"),
        "120",
    ),
    refill_rate=_positive_float_alias(
        (
            "WEREWOLF_REST_RATE_LIMIT_REFILL_RATE",
            "WEREWOLF_REST_RATE_LIMIT_REFILL",
            "WEREWOLF_RATE_LIMIT_REST_REFILL_RATE",
        ),
        "20",
    ),
    key_ttl_seconds=_positive_float_alias(
        ("WEREWOLF_REST_RATE_LIMIT_KEY_TTL_SECONDS", "WEREWOLF_RATE_LIMIT_REST_KEY_TTL_SECONDS"),
        "900",
    ),
    max_keys=_positive_int_alias(
        ("WEREWOLF_REST_RATE_LIMIT_MAX_KEYS", "WEREWOLF_RATE_LIMIT_REST_MAX_KEYS"),
        "10000",
    ),
)
WS_RATE_LIMIT_CONFIG = RateLimitConfig(
    capacity=_positive_float_alias(
        ("WEREWOLF_WS_RATE_LIMIT_CAPACITY", "WEREWOLF_RATE_LIMIT_WS_CAPACITY"),
        "60",
    ),
    refill_rate=_positive_float_alias(
        (
            "WEREWOLF_WS_RATE_LIMIT_REFILL_RATE",
            "WEREWOLF_WS_RATE_LIMIT_REFILL",
            "WEREWOLF_RATE_LIMIT_WS_REFILL_RATE",
        ),
        "10",
    ),
    key_ttl_seconds=_positive_float_alias(
        ("WEREWOLF_WS_RATE_LIMIT_KEY_TTL_SECONDS", "WEREWOLF_RATE_LIMIT_WS_KEY_TTL_SECONDS"),
        "900",
    ),
    max_keys=_positive_int_alias(
        ("WEREWOLF_WS_RATE_LIMIT_MAX_KEYS", "WEREWOLF_RATE_LIMIT_WS_MAX_KEYS"),
        "10000",
    ),
)
PROVIDER_BUDGET_POLICY = ProviderBudgetPolicy(
    max_calls=_non_negative_int_alias(
        ("WEREWOLF_PROVIDER_BUDGET_MAX_CALLS", "WEREWOLF_LLM_BUDGET_MAX_CALLS"),
        "0",
    ),
    max_tokens=_non_negative_int_alias(
        ("WEREWOLF_PROVIDER_BUDGET_MAX_TOKENS", "WEREWOLF_LLM_BUDGET_MAX_TOKENS"),
        "0",
    ),
)
PROVIDER_BUDGET_MAX_SCOPES: int = _positive_int_alias(
    ("WEREWOLF_PROVIDER_BUDGET_MAX_SCOPES",),
    "10000",
)
PROVIDER_BUDGET_MAX_INFLIGHT_RESERVATIONS: int = _positive_int_alias(
    ("WEREWOLF_PROVIDER_BUDGET_MAX_INFLIGHT_RESERVATIONS",),
    "100000",
)
PROVIDER_BUDGET_CLOSED_SCOPE_TTL_SECONDS: float = _positive_float_alias(
    ("WEREWOLF_PROVIDER_BUDGET_CLOSED_SCOPE_TTL_SECONDS",),
    "3600",
)

# —— 默认 LLM(房间级,可被 per-seat 覆盖) ——
DEFAULT_MODEL_CONFIG = {
    "provider": _get("WEREWOLF_LLM_PROVIDER", "openai"),
    "model": _get("WEREWOLF_LLM_MODEL", ""),
    "api_base": _get("WEREWOLF_LLM_API_BASE", ""),
    "api_key": _get("WEREWOLF_LLM_API_KEY", ""),
    "temperature": float(_get("WEREWOLF_LLM_TEMPERATURE", "0.85")),
    # OpenAI:0 means omit the output cap; Anthropic:0 selects the Router's
    # required compatibility default because Messages requires max_tokens.
    "max_tokens": int(_get("WEREWOLF_LLM_MAX_TOKENS", "0")),
    "use_json_format": _get("WEREWOLF_LLM_USE_JSON_FORMAT", "false").lower() == "true",
    "response_format": _optional_json_object("WEREWOLF_LLM_RESPONSE_FORMAT"),
}

# —— 调用参数 ——
LLM_TIMEOUT: float = float(_get("WEREWOLF_LLM_TIMEOUT", "180"))  # 宽松,不误杀
LLM_MAX_RETRIES: int = int(_get("WEREWOLF_LLM_MAX_RETRIES", "3"))  # 429/5xx/网络
LLM_CONCURRENCY: int = int(_get("WEREWOLF_LLM_CONCURRENCY", "4"))
AGENT_DECISION_TIMEOUT: float = float(_get("WEREWOLF_AGENT_DECISION_TIMEOUT", "240"))
AGENT_DECISION_TIMEOUT_BY_PHASE: dict[str, float] = {
    phase: float(_get(f"WEREWOLF_AGENT_DECISION_TIMEOUT_{phase.upper()}", str(AGENT_DECISION_TIMEOUT)))
    for phase in ("night", "day", "voting", "pk", "last_words", "hunter")
}
AGENT_PHASE_DEADLINE: float = float(_get("WEREWOLF_AGENT_PHASE_DEADLINE", "0"))
AGENT_PHASE_DEADLINE_BY_PHASE: dict[str, float] = {
    phase: float(_get(f"WEREWOLF_AGENT_PHASE_DEADLINE_{phase.upper()}", str(AGENT_PHASE_DEADLINE)))
    for phase in ("night", "day", "voting", "pk", "last_words", "hunter")
}

# —— 人类玩家操作超时(秒) ——
HUMAN_TIMEOUT: int = int(_get("WEREWOLF_HUMAN_TIMEOUT", "90"))


@lru_cache(maxsize=1)
def providers_meta() -> dict:
    """前端 /api/providers 返回的提供商元信息。"""
    return {
        "openai": {
            "label": "OpenAI Chat Completions",
            "hint": "OpenAI SDK Chat Completions 兼容接口: base_url + /chat/completions",
            "default_api_base": "https://api.openai.com/v1",
            "default_model": "",
        },
        "openai_responses": {
            "label": "OpenAI Responses API",
            "hint": "OpenAI SDK Responses 接口: base_url + /responses",
            "default_api_base": "https://api.openai.com/v1",
            "default_model": "",
        },
        "anthropic": {
            "label": "Anthropic Messages API",
            "hint": "Anthropic SDK Messages 接口: base_url + /v1/messages",
            "default_api_base": "https://api.anthropic.com",
            "default_model": "",
        },
    }
