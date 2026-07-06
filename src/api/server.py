"""FastAPI server: REST + WebSocket + static frontend.

承 ARCHITECTURE.md §6 + §7。同一进程服务:
- REST  /api/*         房间创建 / 启动 / 元信息 / 回放
- WS    /ws/{room_id}  实时事件流(信息隔离广播由 RoomManager 接管)
- STATIC /             前端 SPA(frontend/ 目录)

启动:python -m src.api.server
"""
from __future__ import annotations

import logging
import os
import hmac
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from ..config import DEFAULT_MODEL_CONFIG, providers_meta
from ..llm.models import ModelConfig
from .room_manager import Room, RoomManager

logger = logging.getLogger(__name__)

# 前端根目录(React build 产物)
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"
ROOM_TOKEN_HEADER = "X-Room-Token"


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


def _strip_internal_event_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if not str(key).startswith("_")
    }


def _header_token(token: str | None) -> str | None:
    return token.strip() if token and token.strip() else None


def _valid_admin_token(room: Room, token: str | None) -> bool:
    token = _header_token(token)
    return bool(token and hmac.compare_digest(token, room.admin_token))


def _valid_seat_token(room: Room, seat: int | None, token: str | None) -> bool:
    token = _header_token(token)
    if seat is None:
        return False
    expected = room.seat_tokens.get(seat)
    return bool(expected and token and hmac.compare_digest(token, expected))


def _require_admin_token(room: Room, token: str | None) -> None:
    if not _valid_admin_token(room, token):
        raise HTTPException(status_code=403, detail="缺少房间管理 token")


def _model_config_override_requires_key(
    default_config: dict[str, Any] | ModelConfig | None,
    override: dict[str, Any] | None,
) -> bool:
    """Return True when an override would send the backend default key elsewhere."""
    if not isinstance(override, dict):
        return False
    if override.get("api_key"):
        return False
    default = default_config if isinstance(default_config, ModelConfig) else ModelConfig(**(default_config or {}))
    if not default.api_key:
        return False

    provider_set = bool(str(override.get("provider") or "").strip())
    base_set = bool(str(override.get("api_base") or "").strip())
    provider = str(override.get("provider") or default.provider).lower()
    api_base = str(override.get("api_base") or default.api_base)
    provider_changed = provider_set and provider != default.provider
    base_changed = base_set and _normalize_api_base(api_base) != _normalize_api_base(default.api_base)
    return provider_changed or base_changed


def _normalize_api_base(value: str | None) -> str:
    return (value or "").strip().rstrip("/")


def _ensure_model_config_safe(default_config: dict[str, Any] | ModelConfig | None, override: dict[str, Any] | None) -> None:
    if _model_config_override_requires_key(default_config, override):
        raise HTTPException(
            status_code=400,
            detail="修改 provider/API Base 时必须显式提供该 endpoint 的 api_key,不能继承后端默认 key",
        )


def _ensure_known_model_config_fields(override: dict[str, Any] | None) -> None:
    if not isinstance(override, dict):
        return
    allowed = set(ModelConfig.model_fields)
    unknown = sorted(str(key) for key in override if key not in allowed)
    if unknown:
        raise HTTPException(status_code=400, detail=f"未知模型配置字段: {', '.join(unknown)}")


# =============================================================================
# Pydantic 请求体
# =============================================================================
class CreateRoomRequest(BaseModel):
    player_names: list[str] = Field(..., min_length=6, max_length=12)
    model_config_dict: dict[str, Any] | None = Field(default=None, alias="model_config")
    human_seats: list[int] = Field(default_factory=list)
    # 可选:角色板子(默认走 default_role_deck)
    deck: list[str] | None = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# =============================================================================
# FastAPI 应用工厂
# =============================================================================
def create_app(manager: RoomManager | None = None) -> FastAPI:
    rooms = manager or RoomManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await rooms.aclose()

    app = FastAPI(title="Werewolf MAS", version="0.1.0", lifespan=lifespan)
    app.state.room_manager = rooms
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # REST API
    # ------------------------------------------------------------------
    @app.get("/api/providers")
    async def get_providers() -> dict[str, Any]:
        return providers_meta()

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        """返回前端需要的默认模型配置,但绝不下发任何 key 片段。"""
        cfg = dict(DEFAULT_MODEL_CONFIG)
        cfg["api_key_configured"] = bool(cfg.get("api_key"))
        cfg["api_key"] = ""
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
        _ensure_model_config_safe(DEFAULT_MODEL_CONFIG, req.model_config_dict)
        try:
            cfg_dict = dict(DEFAULT_MODEL_CONFIG)  # 起手拿默认
            if isinstance(req.model_config_dict, dict):
                for k, v in req.model_config_dict.items():
                    # 空字符串视为未填,不覆盖默认(尤其 api_key)
                    if v in (None, ""):
                        continue
                    cfg_dict[k] = v
            cfg = ModelConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
            logger.info(
                "创建房间 llm_configured=%s provider=%s",
                bool(cfg.api_key and cfg.model and cfg.api_base),
                cfg.provider,
            )
            room = rooms.create_room(
                player_names=req.player_names,
                default_model_config=cfg,
                human_seats=set(req.human_seats),
            )
        except Exception as err:
            logger.exception("创建房间失败")
            raise HTTPException(status_code=400, detail=str(err))
        return {
            "room_id": room.id,
            "status": room.status,
            "end_reason": room.end_reason,
            "error": room.error,
            "human_seats": sorted(room.human_seats),
            "admin_token": room.admin_token,
            "seat_tokens": {str(seat): token for seat, token in sorted(room.seat_tokens.items())},
            "players": [
                {"seat": p.seat, "name": p.name, "alive": p.alive}
                for p in room.state.players
            ],
        }

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
            "players": _room_players(room, reveal_roles=room.status == "ended"),
            "winner": room.state.winner.value if room.state.winner else None,
        }

    @app.post("/api/rooms/{room_id}/start")
    async def start_room(room_id: str, x_room_token: str | None = Header(default=None, alias=ROOM_TOKEN_HEADER)) -> dict[str, Any]:
        room = rooms.get_room(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="房间不存在")
        _require_admin_token(room, x_room_token)
        try:
            await rooms.start_game(room)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))
        return {
            "room_id": room.id,
            "status": room.status,
            "end_reason": room.end_reason,
            "error": room.error,
            "human_seats": sorted(room.human_seats),
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
        _ensure_model_config_safe(room.default_config, body)
        try:
            rooms.set_seat_model_config(room, seat, body)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))
        return {"ok": True}

    @app.get("/api/rooms/{room_id}/replay")
    async def get_replay(room_id: str, x_room_token: str | None = Header(default=None, alias=ROOM_TOKEN_HEADER)) -> dict[str, Any]:
        room = rooms.get_room(room_id)
        if room is None:
            raise HTTPException(status_code=404, detail="房间不存在")
        _require_admin_token(room, x_room_token)
        if room.status != "ended":
            raise HTTPException(status_code=409, detail="replay is available only after game ended")
        analysis = next(
            (
                ev.get("analysis")
                for ev in reversed(room.event_history)
                if ev.get("type") == "analysis" and isinstance(ev.get("analysis"), dict)
            ),
            None,
        )
        return {
            "room_id": room.id,
            "status": room.status,
            "end_reason": room.end_reason,
            "error": room.error,
            "phase": room.state.phase,
            "day": room.state.day,
            "winner": room.state.winner.value if room.state.winner else None,
            "human_seats": sorted(room.human_seats),
            "events": [_strip_internal_event_fields(ev) for ev in room.event_history],
            "thinking": [_strip_internal_event_fields(t) for t in room.thinking_history],
            "analysis": analysis,
            "players": _room_players(room, reveal_roles=True),
        }

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
    ):
        room = rooms.get_room(room_id)
        if room is None:
            await websocket.close(code=4404, reason="room not found")
            return
        mode = (mode or "spectate").lower()
        if mode not in {"spectate", "play", "god", "replay"}:
            await websocket.close(code=4400, reason="unknown mode")
            return
        if mode == "replay" and room.status != "ended":
            await websocket.close(code=4409, reason="replay is available only after game ended")
            return
        if mode in {"god", "replay"} and not _valid_admin_token(room, token):
            await websocket.close(code=4403, reason="admin token required")
            return
        if mode == "play":
            if seat not in room.human_seats or not _valid_seat_token(room, seat, token):
                await websocket.close(code=4403, reason="seat token required")
                return
        elif seat is not None:
            await websocket.close(code=4400, reason="seat is only valid in play mode")
            return
        cid = await rooms.connect(room, websocket, seat=seat, mode=mode)
        try:
            while True:
                # 客户端消息:人类玩家操作 / 心跳 / 订阅过滤
                msg = await websocket.receive_text()
                if msg == "ping":
                    await websocket.send_text("pong")
                    continue
                try:
                    import json as _json
                    data = _json.loads(msg)
                    if data.get("type") == "human_action" and mode == "play" and seat is not None:
                        await rooms.handle_human_action(room, seat, data)
                except Exception:  # noqa: BLE001
                    pass
        except WebSocketDisconnect:
            rooms.disconnect(room, cid)
        except Exception:  # noqa: BLE001
            logger.exception("WS 异常 room=%s cid=%s", room_id, cid)
            rooms.disconnect(room, cid)

    # ------------------------------------------------------------------
    # 静态前端(React build 产物 in frontend/dist)
    # /assets/* 静态资源;/ 与所有未匹配路径回退 index.html(SPA)
    # ------------------------------------------------------------------
    if FRONTEND_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

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
