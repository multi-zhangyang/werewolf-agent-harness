"""FastAPI server: REST + WebSocket + static frontend.

承 ARCHITECTURE.md §6 + §7。同一进程服务:
- REST  /api/*         房间创建 / 启动 / 元信息 / 回放
- WS    /ws/{room_id}  实时事件流(信息隔离广播由 RoomManager 接管)
- STATIC /             前端 SPA(frontend/ 目录)

启动:python -m src.api.server
"""
from __future__ import annotations

import logging
import json
import math
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from ..config import (
    CORS_ALLOW_CREDENTIALS,
    CORS_ORIGINS,
    DEFAULT_MODEL_CONFIG,
    WS_ALLOW_MISSING_ORIGIN,
    parse_cors_origins,
    providers_meta,
)
from ..harness.spec import _safe_api_base
from ..llm.models import ModelConfig
from .room_manager import (
    CapabilityAuthorizationError,
    DeliveryHistoryGapError,
    FutureDeliveryCursorError,
    InvalidDeliveryCursorError,
    Room,
    RoomCapacityError,
    RoomClientCapacityError,
    RoomInUseError,
    RoomManager,
    RoomManagerUnavailableError,
    TERMINAL_ROOM_STATUSES,
)

logger = logging.getLogger(__name__)

# 前端根目录(React build 产物)
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"
ROOM_TOKEN_HEADER = "X-Room-Token"
WS_PROTOCOL = "werewolf.v1"
WS_CAPABILITY_PROTOCOL_PREFIX = "werewolf.cap."
_REDACTED = "[redacted]"
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_-]{8,}|[A-Za-z0-9._~+/=-]*(?:api[_-]?key|secret|password)[A-Za-z0-9._~+/=-]{6,})\b"
)
_TOKEN_ASSIGNMENT_RE = re.compile(
    r"(?i)(^|[^A-Za-z0-9_])([\"']?(?:access[_-]?token|refresh[_-]?token|id[_-]?token|session[_-]?token|client[_-]?secret|private[_-]?key)[\"']?\s*[:=]\s*[\"']?)[^\s,;\"']{8,}"
)
_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)


class AdmissionRateLimitMiddleware:
    """Apply one process-local REST token bucket before route dispatch.

    Health/readiness probes and CORS preflights stay available during a burst;
    all other HTTP requests use the untrusted peer address as the key.  The
    middleware never trusts forwarded headers, which keeps the default useful
    behind a proxy only when that proxy pins the client connection itself.
    """

    def __init__(self, app: ASGIApp, *, limiter: Any) -> None:
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path") or "")
        method = str(scope.get("method") or "GET").upper()
        if method == "OPTIONS" or path in {"/healthz", "/readyz"}:
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        host = str(client[0]) if isinstance(client, (tuple, list)) and client else "unknown"
        key = f"ip:{host[:240]}"
        decision = self.limiter.admit_rest(key)
        if decision.allowed:
            await self.app(scope, receive, send)
            return
        retry_after = decision.retry_after_seconds
        retry_header = (
            str(max(1, math.ceil(retry_after)))
            if math.isfinite(retry_after)
            else "1"
        )
        payload = json.dumps(
            {
                "detail": "request rate limit exceeded",
                "reason": decision.reason,
                "retry_after_seconds": retry_after if math.isfinite(retry_after) else None,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
                (b"retry-after", retry_header.encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": payload})


def _safe_frontend_file(path: str) -> Path | None:
    """Resolve a frontend asset path without allowing traversal outside dist."""
    root = FRONTEND_DIR.resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _room_players(room: Room, *, reveal_roles: bool) -> list[dict[str, Any]]:
    players: list[dict[str, Any]] = []
    for player in room.state.players:
        item: dict[str, Any] = {
            "seat": player.seat,
            "name": player.name,
            "alive": player.alive,
        }
        if reveal_roles:
            item["role"] = player.role
        players.append(item)
    return players


def _strip_internal_event_fields(payload: dict[str, Any], *, extra_secrets: tuple[str, ...] = ()) -> dict[str, Any]:
    return _redact_sensitive_fields({
        key: value
        for key, value in payload.items()
        if not str(key).startswith("_")
    }, extra_secrets=extra_secrets)


def _redact_sensitive_fields(value: Any, *, extra_secrets: tuple[str, ...] = ()) -> Any:
    """Remove credentials/tokens from admin trace and replay payloads."""
    sensitive_fragments = (
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "secret",
        "password",
        "x-api-key",
        "x-room-token",
        "admin_token",
        "seat_token",
        "seat_tokens",
        "access_token",
        "refresh_token",
        "id_token",
        "session_token",
        "client_secret",
        "private_key",
        "cookie",
        "set_cookie",
    )
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for raw_key, raw_val in value.items():
            key = str(raw_key)
            lowered = key.lower().replace("-", "_")
            if any(fragment in lowered for fragment in sensitive_fragments):
                redacted[key] = _REDACTED
            elif lowered == "api_base" and isinstance(raw_val, str):
                # Provenance manifests may retain the endpoint path. They are
                # structured config fields, not arbitrary log/error strings.
                redacted[key] = _safe_api_base(raw_val)
            else:
                redacted[key] = _redact_sensitive_fields(raw_val, extra_secrets=extra_secrets)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_fields(item, extra_secrets=extra_secrets) for item in value]
    if isinstance(value, tuple):
        return [_redact_sensitive_fields(item, extra_secrets=extra_secrets) for item in value]
    if isinstance(value, str):
        return _redact_sensitive_text(value, extra_secrets=extra_secrets)
    return value


def _redact_sensitive_text(text: str, *, extra_secrets: tuple[str, ...] = ()) -> str:
    cleaned = str(text)
    for secret in extra_secrets:
        if secret:
            cleaned = cleaned.replace(secret, _REDACTED)

    def replace_url(match: re.Match[str]) -> str:
        # Paths in arbitrary provider errors can themselves contain tenant
        # IDs, signed routes, or credential-like material. Keep origin only.
        safe = _safe_url_origin(match.group(0))
        return safe or _REDACTED

    cleaned = _URL_RE.sub(replace_url, cleaned)
    cleaned = _BEARER_RE.sub(f"Bearer {_REDACTED}", cleaned)
    cleaned = _TOKEN_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        cleaned,
    )
    cleaned = _JWT_RE.sub(_REDACTED, cleaned)
    return _SECRET_VALUE_RE.sub(_REDACTED, cleaned)


def _safe_url_origin(value: str) -> str:
    """Return only scheme + host + optional port for an arbitrary URL."""
    try:
        parsed = urlsplit(str(value or "").strip())
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    host = parsed.hostname
    try:
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
    except ValueError:
        return ""
    return urlunsplit((parsed.scheme, host, "", "", ""))


def _room_secret_values(room: Room) -> tuple[str, ...]:
    return tuple(secret for secret in (room.admin_token, *room.seat_tokens.values()) if secret)


def _header_token(token: str | None) -> str | None:
    return token.strip() if token and token.strip() else None


def _websocket_capability(
    websocket: WebSocket,
    query_token: str | None,
) -> tuple[str | None, str | None, bool]:
    """Resolve a browser capability without placing it in the request URL.

    Query tokens remain supported for existing native/CLI clients. Browsers
    send the capability as a non-selected `Sec-WebSocket-Protocol` value while
    the server selects only the stable protocol name.
    """
    raw_protocols = str(websocket.headers.get("sec-websocket-protocol") or "")
    offered = [item.strip() for item in raw_protocols.split(",") if item.strip()]
    capability_values = [
        item[len(WS_CAPABILITY_PROTOCOL_PREFIX):]
        for item in offered
        if item.startswith(WS_CAPABILITY_PROTOCOL_PREFIX)
    ]
    malformed = (
        len(capability_values) > 1
        or any(not value or len(value) > 256 for value in capability_values)
    )
    protocol_token = _header_token(capability_values[0]) if len(capability_values) == 1 else None
    legacy_token = _header_token(query_token)
    if protocol_token and legacy_token and protocol_token != legacy_token:
        malformed = True
    selected_protocol = WS_PROTOCOL if WS_PROTOCOL in offered else None
    return protocol_token or legacy_token, selected_protocol, malformed


def _valid_admin_token(room: Room, token: str | None) -> bool:
    token = _header_token(token)
    return RoomManager.valid_admin_token(room, token)


def _valid_seat_token(room: Room, seat: int | None, token: str | None) -> bool:
    token = _header_token(token)
    return RoomManager.valid_seat_token(room, seat, token)


def _require_admin_token(room: Room, token: str | None) -> None:
    if not _valid_admin_token(room, token):
        raise HTTPException(status_code=403, detail="缺少房间管理 token")


def _capability_response(payload: dict[str, Any]) -> JSONResponse:
    """Return a one-time capability with cache/referrer protections."""
    return JSONResponse(
        payload,
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "Referrer-Policy": "no-referrer",
        },
    )


def _authorized_read_response(payload: dict[str, Any]) -> JSONResponse:
    """Return a protected trace/replay projection that cannot be cached.

    The room capability is carried in a custom header rather than the standard
    ``Authorization`` header.  Generic HTTP caches therefore must not be
    allowed to infer that the response is private.  Trace and replay payloads
    contain roles and, for an authorized God/Admin, model-private reasoning.
    """
    return JSONResponse(
        payload,
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "Referrer-Policy": "no-referrer",
            "Vary": ROOM_TOKEN_HEADER,
        },
    )


def _ensure_known_model_config_fields(override: dict[str, Any] | None) -> None:
    if not isinstance(override, dict):
        return
    allowed = set(ModelConfig.model_fields)
    unknown = sorted(str(key) for key in override if key not in allowed)
    if unknown:
        raise HTTPException(status_code=400, detail=f"未知模型配置字段: {', '.join(unknown)}")


def _merged_default_model_config(override: dict[str, Any] | None) -> ModelConfig:
    cfg_dict = dict(DEFAULT_MODEL_CONFIG)
    boundary_changed = False
    if isinstance(override, dict):
        default_provider = str(cfg_dict.get("provider") or "")
        default_base = str(cfg_dict.get("api_base") or "")
        override_provider = str(override.get("provider") or "")
        override_base = str(override.get("api_base") or "")
        if override_provider and override_provider != default_provider:
            boundary_changed = True
        if override_base and override_base != default_base:
            boundary_changed = True
        for k, v in override.items():
            # 空字符串视为未填,不覆盖默认(尤其 api_key)
            if v in (None, ""):
                continue
            cfg_dict[k] = v
        if boundary_changed and not override.get("api_key"):
            cfg_dict["api_key"] = ""
    return ModelConfig(**cfg_dict)


# =============================================================================
# Pydantic 请求体
# =============================================================================
class CreateRoomRequest(BaseModel):
    player_names: list[str] = Field(..., min_length=6, max_length=12)
    model_config_dict: dict[str, Any] | None = Field(default=None, alias="model_config")
    human_seats: list[int] = Field(default_factory=list)
    experiment_seed: int | None = None
    # 可选:角色板子(默认走 default_role_deck)
    deck: list[str] | None = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# =============================================================================
# FastAPI 应用工厂
# =============================================================================
def create_app(
    manager: RoomManager | None = None,
    *,
    persistence_path: str | os.PathLike[str] | None = None,
) -> FastAPI:
    """Build the API app; persistence is explicit and disabled by default."""
    if manager is not None and persistence_path is not None:
        raise ValueError("provide manager or persistence_path, not both")
    rooms = manager or RoomManager(persistence_path=persistence_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await rooms.aclose()

    app = FastAPI(title="Werewolf MAS", version="0.1.0", lifespan=lifespan)
    app.state.room_manager = rooms
    app.add_middleware(
        AdmissionRateLimitMiddleware,
        limiter=rooms.admission_limiter,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(CORS_ORIGINS),
        allow_credentials=CORS_ALLOW_CREDENTIALS,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", ROOM_TOKEN_HEADER],
    )

    # ------------------------------------------------------------------
    # REST API
    # ------------------------------------------------------------------
    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Process liveness only; no dependency or configuration details."""
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        ready, checks = rooms.readiness()
        return JSONResponse(
            {"status": "ready" if ready else "not_ready", "checks": checks},
            status_code=200 if ready else 503,
        )

    @app.get("/api/providers")
    async def get_providers() -> dict[str, Any]:
        return providers_meta()

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        """返回前端需要的默认模型配置,但绝不下发任何 key 片段。"""
        raw_cfg = dict(DEFAULT_MODEL_CONFIG)
        api_key_configured = bool(raw_cfg.get("api_key"))
        cfg = _redact_sensitive_fields(raw_cfg)
        if not isinstance(cfg, dict):  # Defensive: the source contract is a mapping.
            cfg = {}
        cfg["api_key_configured"] = api_key_configured
        cfg["api_key"] = ""
        cfg["api_base"] = _safe_api_base(str(cfg.get("api_base") or ""))
        return cfg

    @app.post("/api/rooms")
    async def create_room(req: CreateRoomRequest) -> dict[str, Any]:
        if req.deck is not None:
            raise HTTPException(status_code=400, detail="自定义角色板暂不支持")
        invalid_human_seats = sorted({seat for seat in req.human_seats if seat < 1 or seat > len(req.player_names)})
        if invalid_human_seats:
            raise HTTPException(
                status_code=400,
                detail=f"human_seats 超出座位范围: {invalid_human_seats}",
            )
        _ensure_known_model_config_fields(req.model_config_dict)
        try:
            cfg = _merged_default_model_config(req.model_config_dict)
            logger.info(
                "创建房间 llm_configured=%s provider=%s",
                bool(cfg.api_key and cfg.model and cfg.api_base),
                cfg.provider,
            )
            room = rooms.create_room(
                player_names=req.player_names,
                default_model_config=cfg,
                human_seats=set(req.human_seats),
                experiment_seed=req.experiment_seed,
            )
        except RoomCapacityError:
            logger.warning("创建房间被拒绝: 已达到房间容量")
            raise HTTPException(status_code=429, detail="房间容量已满,请稍后重试")
        except RoomManagerUnavailableError:
            logger.warning("创建房间被拒绝: 服务正在关闭")
            raise HTTPException(status_code=503, detail="服务暂不接受新房间")
        except Exception as err:
            logger.error("创建房间失败 error_type=%s", type(err).__name__)
            raise HTTPException(status_code=400, detail=_redact_sensitive_text(str(err)))
        return _capability_response({
            "room_id": room.id,
            "status": room.status,
            "end_reason": room.end_reason,
            "error": room.error,
            "human_seats": sorted(room.human_seats),
            "experiment_seed": room.base_seed,
            "role_seed": room.role_seed,
            "actor_seed": room.actor_seed,
            "orchestrator_seed": room.orchestrator_seed,
            "admin_token": room.admin_token,
            "seat_tokens": {str(seat): token for seat, token in sorted(room.seat_tokens.items())},
            "players": [
                {"seat": p.seat, "name": p.name, "alive": p.alive}
                for p in room.state.players
            ],
        })

    @app.get("/api/rooms/{room_id}")
    async def get_room(room_id: str) -> dict[str, Any]:
        room = rooms.get_room(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="房间不存在")
        return {
            "room_id": room.id,
            "status": room.status,
            "end_reason": room.end_reason,
            "error": room.error,
            "phase": room.state.phase,
            "day": room.state.day,
            "human_seats": sorted(room.human_seats),
            # Room metadata is a public polling surface. Role truth is exposed
            # only by an authorized God/Admin projection, including replay.
            "players": _room_players(room, reveal_roles=False),
            "winner": room.state.winner.value if room.state.winner else None,
        }

    @app.delete("/api/rooms/{room_id}")
    async def delete_room(
        room_id: str,
        x_room_token: str | None = Header(default=None, alias=ROOM_TOKEN_HEADER),
    ) -> dict[str, str]:
        room = rooms.get_room(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="房间不存在")
        _require_admin_token(room, x_room_token)
        try:
            removed = rooms.delete_room(room_id)
        except RoomInUseError:
            raise HTTPException(status_code=409, detail="运行中或仍有连接的房间不能清理")
        except RoomManagerUnavailableError:
            raise HTTPException(status_code=503, detail="服务正在关闭")
        if removed is None:
            raise HTTPException(status_code=404, detail="房间不存在")
        return {"room_id": room_id, "status": "deleted"}

    @app.post("/api/rooms/{room_id}/tokens/rotate")
    async def rotate_room_token(
        room_id: str,
        body: dict[str, Any] | None = None,
        x_room_token: str | None = Header(default=None, alias=ROOM_TOKEN_HEADER),
    ) -> dict[str, Any]:
        """Rotate an admin or human-seat capability.

        The new plaintext capability is returned exactly once in this response
        and is never written to the optional persistence store.
        """
        room = rooms.get_room(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="房间不存在")
        with room.capability_lock:
            # Recheck and mutate under the same lock. Concurrent requests that
            # presented the old admin token cannot rotate the replacement.
            _require_admin_token(room, x_room_token)
            requested_seat = (body or {}).get("seat") if isinstance(body, dict) else None
            try:
                if requested_seat is None:
                    token = rooms.rotate_admin_token(room)
                    return _capability_response({
                        "room_id": room.id,
                        "scope": "admin",
                        "token": token,
                        "version": room.admin_token_version,
                    })
                seat = int(requested_seat)
                token = rooms.rotate_seat_token(room, seat)
                return _capability_response({
                    "room_id": room.id,
                    "scope": "seat",
                    "seat": seat,
                    "token": token,
                    "version": room.seat_token_versions[seat],
                })
            except (TypeError, ValueError) as err:
                raise HTTPException(status_code=400, detail=_redact_sensitive_text(str(err)))

    @app.post("/api/rooms/{room_id}/tokens/revoke")
    async def revoke_room_token(
        room_id: str,
        body: dict[str, Any] | None = None,
        x_room_token: str | None = Header(default=None, alias=ROOM_TOKEN_HEADER),
    ) -> dict[str, Any]:
        """Revoke a human-seat capability without exposing any secret."""
        room = rooms.get_room(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="房间不存在")
        with room.capability_lock:
            _require_admin_token(room, x_room_token)
            requested_seat = (body or {}).get("seat") if isinstance(body, dict) else None
            if requested_seat is None:
                raise HTTPException(status_code=400, detail="只能显式吊销 seat capability")
            try:
                seat = int(requested_seat)
                rooms.revoke_seat_token(room, seat)
            except (TypeError, ValueError) as err:
                raise HTTPException(status_code=400, detail=_redact_sensitive_text(str(err)))
            return _capability_response({
                "room_id": room.id,
                "scope": "seat",
                "seat": seat,
                "status": "revoked",
                "version": room.seat_token_versions[seat],
            })

    @app.post("/api/rooms/{room_id}/start")
    async def start_room(room_id: str, x_room_token: str | None = Header(default=None, alias=ROOM_TOKEN_HEADER)) -> dict[str, Any]:
        room = rooms.get_room(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="房间不存在")
        _require_admin_token(room, x_room_token)
        try:
            await rooms.start_game(room)
        except RoomManagerUnavailableError:
            raise HTTPException(status_code=503, detail="服务正在关闭")
        except ValueError as err:
            raise HTTPException(status_code=400, detail=_redact_sensitive_text(str(err)))
        return {
            "room_id": room.id,
            "status": room.status,
            "end_reason": room.end_reason,
            "error": room.error,
            "human_seats": sorted(room.human_seats),
            "experiment_seed": room.base_seed,
            "role_seed": room.role_seed,
            "actor_seed": room.actor_seed,
            "orchestrator_seed": room.orchestrator_seed,
        }

    @app.post("/api/rooms/{room_id}/seats/{seat}/model_config")
    async def set_seat_config(
        room_id: str,
        seat: int,
        body: dict[str, Any],
        x_room_token: str | None = Header(default=None, alias=ROOM_TOKEN_HEADER),
    ) -> dict[str, Any]:
        room = rooms.get_room(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="房间不存在")
        _require_admin_token(room, x_room_token)
        _ensure_known_model_config_fields(body)
        try:
            rooms.set_seat_model_config(room, seat, body)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=_redact_sensitive_text(str(err)))
        return {"ok": True}

    @app.get("/api/rooms/{room_id}/replay")
    async def get_replay(room_id: str, x_room_token: str | None = Header(default=None, alias=ROOM_TOKEN_HEADER)) -> JSONResponse:
        room = rooms.get_room(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="房间不存在")
        _require_admin_token(room, x_room_token)
        if room.status not in TERMINAL_ROOM_STATUSES:
            raise HTTPException(status_code=409, detail="replay is available only after room termination")
        analysis = next(
            (
                ev.get("analysis")
                for ev in reversed(room.event_history)
                if ev.get("type") == "analysis" and isinstance(ev.get("analysis"), dict)
            ),
            None,
        )
        extra_secrets = _room_secret_values(room)
        safe_analysis = _redact_sensitive_fields(analysis, extra_secrets=extra_secrets) if analysis is not None else None
        return _authorized_read_response({
            "room_id": room.id,
            "status": room.status,
            "end_reason": room.end_reason,
            "error": room.error,
            "phase": room.state.phase,
            "day": room.state.day,
            "winner": room.state.winner.value if room.state.winner else None,
            "human_seats": sorted(room.human_seats),
            "experiment_seed": room.base_seed,
            "role_seed": room.role_seed,
            "actor_seed": room.actor_seed,
            "orchestrator_seed": room.orchestrator_seed,
            "events": [_strip_internal_event_fields(ev, extra_secrets=extra_secrets) for ev in room.event_history],
            "analysis": safe_analysis,
            "players": _room_players(room, reveal_roles=True),
            "run_spec": _redact_sensitive_fields(room.run_spec.model_dump(), extra_secrets=extra_secrets) if room.run_spec else None,
            "core_run_spec": _redact_sensitive_fields(
                room.core_run_spec.model_dump(mode="json"),
                extra_secrets=extra_secrets,
            ) if room.core_run_spec else None,
            "transcript": _redact_sensitive_fields(room.transcript.export(), extra_secrets=extra_secrets) if room.transcript else None,
        })

    @app.get("/api/rooms/{room_id}/trace")
    async def get_trace(
        room_id: str,
        since: str | None = None,
        x_room_token: str | None = Header(default=None, alias=ROOM_TOKEN_HEADER),
    ) -> JSONResponse:
        """Admin-only machine trace for harness replay and experiment audit."""
        room = rooms.get_room(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="房间不存在")
        _require_admin_token(room, x_room_token)
        trace_since: int | None = None
        if since is not None:
            if len(since) > 20 or not re.fullmatch(r"0|[1-9][0-9]*", since):
                raise HTTPException(status_code=400, detail="invalid trace cursor")
            try:
                trace_since = int(since)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid trace cursor")
        extra_secrets = _room_secret_values(room)
        items: list[dict[str, Any]] = []
        for idx, ev in enumerate(room.event_history):
            event_trace_seq = ev.get("_trace_seq")
            if trace_since is not None and (
                type(event_trace_seq) is not int or event_trace_seq <= trace_since
            ):
                continue
            items.append({
                "kind": "event",
                "idx": idx,
                "trace_seq": event_trace_seq,
                "ts": ev.get("_ts"),
                "payload": _strip_internal_event_fields(ev, extra_secrets=extra_secrets),
            })
        for idx, item in enumerate(room.decision_trace):
            decision_trace_seq = item.get("_trace_seq")
            if trace_since is not None and (
                type(decision_trace_seq) is not int or decision_trace_seq <= trace_since
            ):
                continue
            items.append({
                "kind": "decision",
                "idx": idx,
                "trace_seq": decision_trace_seq,
                "ts": item.get("_ts"),
                "payload": _strip_internal_event_fields(item, extra_secrets=extra_secrets),
            })
        items.sort(key=lambda item: (
            int(item.get("trace_seq") or 0),
            float(item.get("ts") or 0),
            int(item.get("idx") or 0),
        ))
        response: dict[str, Any] = {
            "room_id": room.id,
            "status": room.status,
            "end_reason": room.end_reason,
            "error": room.error,
            "phase": room.state.phase,
            "day": room.state.day,
            "winner": room.state.winner.value if room.state.winner else None,
            "experiment_seed": room.base_seed,
            "role_seed": room.role_seed,
            "actor_seed": room.actor_seed,
            "orchestrator_seed": room.orchestrator_seed,
            "event_count": len(room.event_history),
            "decision_trace_count": len(room.decision_trace),
            "trace_seq": int(room.trace_seq),
            "since": trace_since,
            "incremental": trace_since is not None,
            "trace": items,
            "run_spec": _redact_sensitive_fields(room.run_spec.model_dump(), extra_secrets=extra_secrets) if trace_since is None and room.run_spec else None,
            "core_run_spec": _redact_sensitive_fields(
                room.core_run_spec.model_dump(mode="json"),
                extra_secrets=extra_secrets,
            ) if trace_since is None and room.core_run_spec else None,
            "transcript": _redact_sensitive_fields(room.transcript.export(), extra_secrets=extra_secrets) if trace_since is None and room.transcript else None,
        }
        return _authorized_read_response(response)

    # ------------------------------------------------------------------
    # WebSocket:实时事件流
    # ------------------------------------------------------------------
    @app.websocket("/ws/{room_id}")
    async def ws_endpoint(
        websocket: WebSocket,
        room_id: str,
        seat: int | None = None,
        mode: str = "spectate",
        token: str | None = None,
        since: str | None = None,
    ):
        origin = websocket.headers.get("origin")
        if origin is None:
            if not WS_ALLOW_MISSING_ORIGIN:
                await websocket.close(code=4403, reason="WebSocket Origin required")
                return
        else:
            try:
                normalized_origin = parse_cors_origins(origin)
            except ValueError:
                normalized_origin = ()
            if len(normalized_origin) != 1 or normalized_origin[0] not in CORS_ORIGINS:
                await websocket.close(code=4403, reason="WebSocket Origin not allowed")
                return
        ws_client = websocket.client
        ws_host = str(ws_client.host) if ws_client is not None else "unknown"
        ws_limit_key = f"ip:{ws_host[:240]}"
        ws_admission = rooms.admission_limiter.admit_ws(ws_limit_key)
        if not ws_admission.allowed:
            await websocket.close(code=4429, reason="WebSocket rate limit exceeded")
            return
        room = rooms.get_room(room_id)
        if room is None:
            await websocket.close(code=4404, reason="room not found")
            return
        mode = (mode or "spectate").lower()
        if mode not in {"spectate", "play", "god", "replay"}:
            await websocket.close(code=4400, reason="unknown mode")
            return
        if mode == "replay" and room.status not in TERMINAL_ROOM_STATUSES:
            await websocket.close(code=4409, reason="replay is available only after room termination")
            return
        capability_token, selected_subprotocol, malformed_capability = _websocket_capability(
            websocket,
            token,
        )
        if malformed_capability:
            await websocket.close(code=4403, reason="invalid capability transport")
            return
        if mode in {"god", "replay"} and not _valid_admin_token(room, capability_token):
            await websocket.close(code=4403, reason="admin token required")
            return
        if mode == "play":
            if seat not in room.human_seats or not _valid_seat_token(room, seat, capability_token):
                await websocket.close(code=4403, reason="seat token required")
                return
        elif seat is not None:
            await websocket.close(code=4400, reason="seat is only valid in play mode")
            return
        resume_cursor: int | None = None
        if since is not None:
            if len(since) > 20 or not re.fullmatch(r"0|[1-9][0-9]*", since):
                await websocket.close(code=4400, reason="invalid delivery cursor")
                return
            try:
                resume_cursor = int(since)
            except ValueError:
                await websocket.close(code=4400, reason="invalid delivery cursor")
                return
        try:
            cid = await rooms.connect(
                room,
                websocket,
                seat=seat,
                mode=mode,
                since=resume_cursor,
                capability_token=capability_token,
                websocket_subprotocol=selected_subprotocol,
            )
        except CapabilityAuthorizationError:
            await websocket.close(code=4403, reason="capability changed or revoked")
            return
        except RoomClientCapacityError:
            await websocket.close(code=4429, reason="room connection capacity reached")
            return
        except DeliveryHistoryGapError as err:
            await websocket.close(
                code=4409,
                reason=f"history gap; earliest={err.earliest}; current={err.current}",
            )
            return
        except FutureDeliveryCursorError:
            await websocket.close(code=4400, reason="future delivery cursor")
            return
        except InvalidDeliveryCursorError:
            await websocket.close(code=4400, reason="invalid delivery cursor")
            return
        try:
            while True:
                # 客户端消息:人类玩家操作 / 心跳 / 订阅过滤
                msg = await websocket.receive_text()
                ws_message_admission = rooms.admission_limiter.admit_ws(ws_limit_key)
                if not ws_message_admission.allowed:
                    await websocket.close(code=4429, reason="WebSocket message rate limit exceeded")
                    rooms.disconnect(room, cid)
                    return
                if msg == "ping":
                    rooms.send_client_text(room, cid, "pong")
                    continue
                import json as _json

                async def _reject_bad_human_payload(reason: str, request_id: str = "") -> None:
                    if mode == "play" and seat is not None:
                        rooms.send_client_payload(room, cid, {
                            "type": "human_action_rejected",
                            "seat": seat,
                            "request_id": request_id,
                            "reason": reason,
                        })

                try:
                    data = _json.loads(msg)
                except _json.JSONDecodeError:
                    await _reject_bad_human_payload("invalid_payload")
                    continue
                if not isinstance(data, dict):
                    await _reject_bad_human_payload("invalid_payload")
                    continue
                if data.get("type") == "human_action":
                    request_id = str(data.get("request_id") or "")
                    if mode == "play" and seat is not None:
                        await rooms.handle_human_action(room, seat, data)
                    else:
                        await _reject_bad_human_payload("no_pending_request", request_id)
        except WebSocketDisconnect:
            rooms.disconnect(room, cid)
        except Exception as err:  # noqa: BLE001
            logger.error("WS 异常 room=%s cid=%s error_type=%s", room_id, cid, type(err).__name__)
            rooms.disconnect(room, cid)

    # ------------------------------------------------------------------
    # 静态前端(React build 产物 in frontend/dist)
    # /assets/* 静态资源;/ 与所有未匹配路径回退 index.html(SPA)
    # ------------------------------------------------------------------
    if FRONTEND_DIR.exists():
        # A frontend build can replace ``dist`` atomically and briefly leave
        # the asset directory absent.  StaticFiles checks its directory at
        # construction time, so only mount it when the directory is present;
        # the SPA fallback remains available for an index-only build.
        assets_dir = FRONTEND_DIR / "assets"
        if assets_dir.is_dir():
            app.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir), check_dir=False),
                name="assets",
            )

        @app.get("/{path:path}")
        async def spa_fallback(path: str) -> FileResponse:
            # 优先匹配真实静态文件(vitefavicon/robots 等),否则回退 index.html
            candidate = _safe_frontend_file(path)
            if candidate is not None:
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIR / "index.html")
    else:
        @app.get("/{path:path}", response_class=HTMLResponse)
        async def frontend_not_built(path: str) -> HTMLResponse:
            return HTMLResponse(
                """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Werewolf MAS frontend not built</title>
    <style>
      body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #0b1020; color: #e5e7eb; font: 16px/1.6 system-ui, sans-serif; }
      main { max-width: 680px; padding: 32px; border: 1px solid rgba(148,163,184,.28); border-radius: 8px; background: rgba(15,23,42,.88); }
      h1 { margin: 0 0 12px; font-size: 24px; }
      code { color: #93c5fd; }
      pre { overflow: auto; padding: 14px 16px; border-radius: 8px; background: #020617; }
    </style>
  </head>
  <body>
    <main>
      <h1>前端尚未构建</h1>
      <p>当前 FastAPI 只负责服务 <code>frontend/dist</code> 中的生产构建产物。请选择一种方式启动:</p>
      <pre>cd frontend
npm install
npm run dev</pre>
      <p>然后访问 <code>http://localhost:5173</code>。或者先运行:</p>
      <pre>cd frontend
npm install
npm run build
cd ..
python -m src.api.server</pre>
      <p>构建完成后访问 <code>http://localhost:8000</code>。</p>
    </main>
  </body>
</html>
                """.strip(),
                status_code=503,
            )

    return app


# 模块级 app(供 uvicorn src.api.server:app 直接引用)
app = create_app()


def main() -> None:
    """CLI 入口:python -m src.api.server"""
    import uvicorn

    from ..config import HOST, PORT, LOG_LEVEL

    logging.basicConfig(
        level=LOG_LEVEL.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("启动 Werewolf MAS @ http://%s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL)


if __name__ == "__main__":
    main()
