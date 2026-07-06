"""配置:仅读取 WEREWOLF_* 前缀环境变量。

铁律:绝不读取/写入系统 OPENAI_API_KEY / ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL
(那些被其他进程 / Claude Code 自身占用)。本项目一切配置走 WEREWOLF_ 前缀。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values


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


# —— 服务 ——
HOST: str = _get("WEREWOLF_HOST", "127.0.0.1")
PORT: int = int(_get("WEREWOLF_PORT", "8000"))
LOG_LEVEL: str = _get("WEREWOLF_LOG_LEVEL", "info")

# —— 默认 LLM(房间级,可被 per-seat 覆盖) ——
DEFAULT_MODEL_CONFIG = {
    "provider": _get("WEREWOLF_LLM_PROVIDER", "openai"),
    "model": _get("WEREWOLF_LLM_MODEL", ""),
    "api_base": _get("WEREWOLF_LLM_API_BASE", ""),
    "api_key": _get("WEREWOLF_LLM_API_KEY", ""),
    "temperature": float(_get("WEREWOLF_LLM_TEMPERATURE", "0.85")),
    "max_tokens": int(_get("WEREWOLF_LLM_MAX_TOKENS", "0")),  # 0 = 不限制
    "use_json_format": _get("WEREWOLF_LLM_USE_JSON_FORMAT", "false").lower() == "true",
}

# —— 调用参数 ——
LLM_TIMEOUT: float = float(_get("WEREWOLF_LLM_TIMEOUT", "180"))  # 宽松,不误杀
LLM_MAX_RETRIES: int = int(_get("WEREWOLF_LLM_MAX_RETRIES", "5"))  # 429/5xx/网络
LLM_CONCURRENCY: int = int(_get("WEREWOLF_LLM_CONCURRENCY", "4"))
AGENT_DECISION_TIMEOUT: float = float(_get("WEREWOLF_AGENT_DECISION_TIMEOUT", "240"))
AGENT_DECISION_TIMEOUT_BY_PHASE: dict[str, float] = {
    phase: float(_get(f"WEREWOLF_AGENT_DECISION_TIMEOUT_{phase.upper()}", str(AGENT_DECISION_TIMEOUT)))
    for phase in ("night", "day", "voting", "pk", "last_words", "hunter", "reflection")
}
AGENT_PHASE_DEADLINE: float = float(_get("WEREWOLF_AGENT_PHASE_DEADLINE", "0"))
AGENT_PHASE_DEADLINE_BY_PHASE: dict[str, float] = {
    phase: float(_get(f"WEREWOLF_AGENT_PHASE_DEADLINE_{phase.upper()}", str(AGENT_PHASE_DEADLINE)))
    for phase in ("night", "day", "voting", "pk", "last_words", "hunter", "reflection")
}

# —— 人类玩家操作超时(秒) ——
HUMAN_TIMEOUT: int = int(_get("WEREWOLF_HUMAN_TIMEOUT", "90"))
AI_THINKING_VISIBLE: bool = _get("WEREWOLF_AI_THINKING_VISIBLE", "true").lower() == "true"


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
            "default_model": "claude-sonnet-5",
        },
    }
