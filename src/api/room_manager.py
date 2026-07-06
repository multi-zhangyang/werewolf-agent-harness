"""房间管理器 —— 房间生命周期 + 游戏循环 + 信息隔离广播。

承 ARCHITECTURE.md §6:每个事件带 visibility+recipients,按 seat 过滤后下发。
spectate 只收公开事件与思考摘要;play 收该 seat 私有+公开;god 收全部(含隐藏推理)。
人类席位超时走 _legal_skip(透明不行动,非伪造,承 no-fallback-design)。
"""
from __future__ import annotations

import asyncio
import secrets
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from ..config import (
    AGENT_DECISION_TIMEOUT,
    AGENT_PHASE_DEADLINE,
    AGENT_PHASE_DEADLINE_BY_PHASE,
    LLM_CONCURRENCY,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
)
from ..agent.actor import AgentActor
from ..game.models import EventVisibility, GameState, Phase
from ..game.orchestrator import GameOrchestratorV2 as GameOrchestrator, build_actors
from ..game.roles import Role, default_role_deck
from ..game.rules import RulesEngine
from ..game.state import new_game
from ..llm.models import ModelConfig
from ..llm.router import LLMRouter

logger = logging.getLogger(__name__)


@dataclass
class Room:
    """一个游戏房间。"""

    id: str
    state: GameState
    orchestrator: GameOrchestrator | None = None
    actors: dict[str, AgentActor] = field(default_factory=dict)
    status: str = "waiting"  # waiting / running / ended / failed / timeout / cancelled
    end_reason: str | None = None
    error: str | None = None
    default_config: ModelConfig | None = None
    seat_configs: dict[int, ModelConfig] = field(default_factory=dict)
    human_seats: set[int] = field(default_factory=set)
    admin_token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    seat_tokens: dict[int, str] = field(default_factory=dict)
    # 连接的客户端:client_id -> (websocket, seat, mode)
    clients: dict[str, tuple[WebSocket, int | None, str]] = field(default_factory=dict)
    # 完整事件流(回放用,按时间序)
    event_history: list[dict[str, Any]] = field(default_factory=list)
    thinking_history: list[dict[str, Any]] = field(default_factory=list)
    task: asyncio.Task | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RoomManager:
    """全局房间管理。"""

    def __init__(self, router: LLMRouter | None = None, *, room_timeout: float | None = None) -> None:
        self.rooms: dict[str, Room] = {}
        self.router = router or LLMRouter(
            timeout=LLM_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
            concurrency=LLM_CONCURRENCY,
        )
        self.room_timeout = _room_timeout_from_env() if room_timeout is None else room_timeout

    # ------------------------------------------------------------------
    # 创建/查询
    # ------------------------------------------------------------------
    def create_room(
        self,
        *,
        player_names: list[str],
        default_model_config: ModelConfig | dict[str, Any] | None = None,
        roles: dict[str, str] | None = None,
        human_seats: set[int] | None = None,
    ) -> Room:
        import uuid

        room_id = uuid.uuid4().hex[:12]
        state = new_game(player_names)
        if isinstance(default_model_config, dict):
            default_model_config = ModelConfig(**default_model_config)
        room = Room(
            id=room_id,
            state=state,
            default_config=default_model_config,
            human_seats=human_seats or set(),
        )
        room.seat_tokens = {
            seat: secrets.token_urlsafe(24)
            for seat in sorted(room.human_seats)
        }
        self.rooms[room_id] = room
        return room

    def get_room(self, room_id: str) -> Room | None:
        return self.rooms.get(room_id)

    def set_seat_model_config(self, room: Room, seat: int, config: ModelConfig | dict) -> None:
        if room.status != "waiting":
            raise ValueError("只能在 waiting 阶段修改座位配置")
        cfg = config if isinstance(config, ModelConfig) else ModelConfig(**config)
        base = room.default_config or ModelConfig()
        base.merge(cfg)
        room.seat_configs[seat] = cfg

    # ------------------------------------------------------------------
    # 启动游戏
    # ------------------------------------------------------------------
    async def start_game(self, room: Room) -> None:
        """发牌 + 构建 actors + 启动编排器协程。"""
        async with room.lock:
            if room.status != "waiting":
                raise ValueError(f"房间状态 {room.status},无法开始")
            deck = default_role_deck(len(room.state.players))
            RulesEngine.deal_roles(room.state, deck=deck)
            base_cfg = room.default_config or ModelConfig()
            room.actors = build_actors(
                room.state,
                model_config=base_cfg,
                router=self.router,
                seat_configs=room.seat_configs,
                human_seats=room.human_seats,
            )
            room.orchestrator = GameOrchestrator(
                state=room.state,
                actors=room.actors,
                deck=deck,
                rng=random.Random(),
                on_event=self._make_event_broadcaster(room),
                on_thinking=self._make_thinking_broadcaster(room),
                verbose_thinking=True,  # 完整 reasoning 入 history;广播时按 mode 净化,god 专属完整推理
                decision_timeout=AGENT_DECISION_TIMEOUT,
                phase_deadline=AGENT_PHASE_DEADLINE,
                phase_deadlines=AGENT_PHASE_DEADLINE_BY_PHASE,
            )
            room.status = "running"
            room.end_reason = None
            room.error = None
            await self._broadcast_room_status(room)
            # 启动游戏循环(后台任务)
            room.task = asyncio.create_task(self._run_room(room))

    async def _run_room(self, room: Room) -> None:
        if room.orchestrator is None:
            room.status = "failed"
            room.end_reason = "missing_orchestrator"
            room.error = "orchestrator is not initialized"
            await self._broadcast(room, {
                "type": "game_error",
                "reason": room.end_reason,
                "message": room.error,
            })
            await self._broadcast_room_status(room)
            return
        try:
            if self.room_timeout and self.room_timeout > 0:
                await asyncio.wait_for(room.orchestrator.run(), timeout=self.room_timeout)
            else:
                await room.orchestrator.run()
        except asyncio.TimeoutError:
            room.status = "timeout"
            room.end_reason = "timeout"
            room.error = f"room exceeded {self.room_timeout:.0f}s timeout"
            logger.error("房间 %s 游戏循环超时: %s", room.id, room.error)
            await self._broadcast(room, {
                "type": "game_error",
                "reason": room.end_reason,
                "message": room.error,
            })
            await self._broadcast_room_status(room)
        except asyncio.CancelledError:
            room.status = "cancelled"
            room.end_reason = "cancelled"
            room.error = "room task was cancelled"
            await self._broadcast_room_status(room)
            raise
        except Exception as err:  # noqa: BLE001
            room.status = "failed"
            room.end_reason = "error"
            room.error = _public_room_error(err)
            logger.exception("房间 %s 游戏循环异常: %s", room.id, err)
            await self._broadcast(room, {
                "type": "game_error",
                "reason": room.end_reason,
                "message": room.error,
            })
            await self._broadcast_room_status(room)
        else:
            if room.state.phase == Phase.ENDED and room.state.winner:
                room.status = "ended"
                room.end_reason = "completed"
                room.error = None
            else:
                room.status = "failed"
                room.end_reason = "incomplete"
                room.error = "orchestrator returned before ended/winner"
                await self._broadcast(room, {
                    "type": "game_error",
                    "reason": room.end_reason,
                    "message": room.error,
                })
            await self._broadcast_room_status(room)

    async def _broadcast_room_status(self, room: Room) -> None:
        payload: dict[str, Any] = {
            "type": "room_status",
            "status": room.status,
            "reason": room.end_reason,
        }
        if room.error:
            payload["error"] = room.error
        await self._broadcast(room, payload)

    # ------------------------------------------------------------------
    # 信息隔离广播
    # ------------------------------------------------------------------
    def _make_event_broadcaster(self, room: Room):
        async def on_event(ev: dict[str, Any]) -> None:
            stored = dict(ev)
            stored.setdefault("_ts", time.monotonic())
            room.event_history.append(stored)
            await self._broadcast(room, ev)
        return on_event

    def _make_thinking_broadcaster(self, room: Room):
        async def on_thinking(t: dict[str, Any]) -> None:
            stored = dict(t)
            stored.setdefault("_ts", time.monotonic())
            room.thinking_history.append(stored)
            # 思考流只推给 god/spectate;_broadcast 会按 mode 移除非 god 的完整 reasoning。
            await self._broadcast(room, {"type": "agent_thinking", **t}, thinking=True)
        return on_thinking

    async def _broadcast(self, room: Room, payload: dict[str, Any], *, thinking: bool = False) -> None:
        """按客户端 mode 过滤后下发。信息隔离核心。"""
        dead_clients = []
        for cid, (ws, seat, mode) in room.clients.items():
            if not self._should_receive(room, payload, seat, mode, thinking):
                continue
            visible_payload = self._payload_for_client(payload, seat, mode, thinking)
            msg = json.dumps(visible_payload, ensure_ascii=False, default=str)
            try:
                await ws.send_text(msg)
            except Exception:  # noqa: BLE001
                dead_clients.append(cid)
        for cid in dead_clients:
            room.clients.pop(cid, None)

    def _payload_for_client(
        self,
        payload: dict[str, Any],
        seat: int | None,
        mode: str,
        thinking: bool,
    ) -> dict[str, Any]:
        """Return the mode-specific event payload after privacy filtering."""
        del seat
        if thinking and mode != "god":
            # The orchestrator stores complete reasoning in thinking_history,
            # but live non-god clients must not receive role strategy,
            # suspicion graphs, or role-specific action names.
            visible = {
                "type": "agent_thinking",
                "seat": payload.get("seat"),
                "summary": "AI 思考已记录,隐藏推理赛后由授权复盘查看。",
            }
            action = _public_action(payload.get("action"))
            if action:
                visible["action"] = action
            if "bid" in payload:
                visible["bid"] = payload.get("bid")
            return visible

        if mode in ("god", "replay"):
            return payload

        visible = dict(payload)
        visible.pop("visibility", None)
        visible.pop("recipients", None)

        etype = visible.get("type")
        if etype == "night_resolved":
            visible["deaths"] = [
                {"seat": d.get("seat"), "name": d.get("name")}
                for d in visible.get("deaths", [])
                if isinstance(d, dict)
            ]
        elif etype == "agent_decision_failed":
            phase = str(visible.get("phase") or "")
            sanitized: dict[str, Any] = {
                "type": "agent_decision_failed",
                "phase": phase,
                "reason": "AI 决策失败,已按规则跳过。",
            }
            # Seat identity is public for daytime/vote/last-words failures,
            # but night and wolf-caucus failures can reveal hidden roles.
            if phase in {"day", "voting", "pk", "last_words", "hunter"} and visible.get("seat") is not None:
                sanitized["seat"] = visible.get("seat")
            if bool(visible.get("timeout")):
                sanitized["timeout"] = True
            return sanitized
        return visible

    def _should_receive(self, room: Room, payload: dict[str, Any], seat: int | None, mode: str, thinking: bool) -> bool:
        """信息隔离裁决:该客户端能否收到此事件。"""
        etype = payload.get("type", "")

        # 思考摘要:god 实时看,spectate 看经整理的(这里都给),play 不给(保公平)
        if thinking:
            if mode == "god":
                return True
            # Spectators may see public-phase thinking summaries only. Night
            # actions are omitted because even a seat-level summary leaks roles.
            return mode == "spectate" and _public_action(payload.get("action")) is not None

        # 全知模式:看一切
        if mode == "god":
            return True

        # 回放/复盘:看一切,但只有赛后连接才允许进入 replay 模式。
        if mode in ("replay",):
            return room.status == "ended"

        recipients = payload.get("recipients") or []
        visibility = payload.get("visibility")
        if visibility == "private" or recipients:
            if mode != "play" or seat is None:
                return False
            player = next((p for p in room.state.players if p.seat == seat), None)
            recipient_set = {str(r) for r in recipients}
            return bool(
                player
                and (
                    player.id in recipient_set
                    or str(seat) in recipient_set
                    or payload.get("seat") == seat
                )
            )

        # 公开事件:所有人(spectate/play)都收
        public_types = {
            "phase_started", "night_resolved", "speech", "vote_cast", "vote_resolved",
            "last_words", "hunter_shot", "game_ended", "room_status", "game_error",
            "vote_incomplete", "analysis", "agent_decision_failed",
        }
        if etype in public_types:
            return True

        # 信任更新:上帝模式专属(不能泄露给普通玩家)
        if etype in ("trust_update", "reflections_update"):
            return mode == "god"

        # 人类操作请求:仅该 seat 的 play 模式收
        if etype == "human_action_request":
            return mode == "play" and seat == payload.get("seat")

        return False

    # ------------------------------------------------------------------
    # WebSocket 连接管理
    # ------------------------------------------------------------------
    async def connect(self, room: Room, ws: WebSocket, *, seat: int | None, mode: str) -> str:
        await ws.accept()
        import uuid

        cid = uuid.uuid4().hex[:8]
        room.clients[cid] = (ws, seat, mode)
        # 推送当前状态快照
        await ws.send_text(json.dumps({
            "type": "snapshot",
            "status": room.status,
            "view": self._view_for(room, seat, mode),
        }, ensure_ascii=False, default=str))
        # 推送历史事件(新连接者补看)。把 thinking 和普通事件按实际时间合并,
        # 让 god 视角仍能看到"思考摘要 + 内容"贴在同一消息附近。
        history: list[dict[str, Any]] = []
        for idx, ev in enumerate(room.event_history):
            history.append({"__kind": "event", "__idx": idx, **ev})
        for idx, item in enumerate(room.thinking_history):
            history.append({"__kind": "thinking", "__idx": idx, **item})
        history.sort(key=lambda item: (float(item.get("_ts") or 0), int(item.get("__idx") or 0)))
        for item in history:
            is_thinking = item.get("__kind") == "thinking"
            clean = {
                key: value
                for key, value in item.items()
                if key not in {"__kind", "__idx", "_ts"}
            }
            payload = {"type": "agent_thinking", **clean} if is_thinking else clean
            if self._should_receive(room, payload, seat, mode, thinking=is_thinking):
                try:
                    visible = self._payload_for_client(payload, seat, mode, thinking=is_thinking)
                    await ws.send_text(json.dumps(visible, ensure_ascii=False, default=str))
                except Exception:  # noqa: BLE001
                    break
        return cid

    def disconnect(self, room: Room, cid: str) -> None:
        room.clients.pop(cid, None)

    def _view_for(self, room: Room, seat: int | None, mode: str) -> dict[str, Any]:
        """根据 mode 生成视图快照。"""
        state = room.state
        # 提取 persona(所有模式可见 — 不影响游戏公平,只是 AI 风格标签)
        seat_to_persona: dict[str, str] = {}
        for pid, actor in room.actors.items():
            player = state.get_player(pid) if pid in [p.id for p in state.players] else None
            if player is None:
                continue
            seat_to_persona[str(player.seat)] = getattr(actor, "persona_name", "")
        if mode == "god":
            view = state.public_view() | {"god": True, "players_full": [
                {"seat": p.seat, "name": p.name, "role": p.role, "team": p.team, "alive": p.alive,
                 "persona": seat_to_persona.get(str(p.seat), "")}
                for p in state.players
            ]}
            view["hidden_state"] = {
                "witch_antidote": state.witch_antidote,
                "witch_poison": state.witch_poison,
                "last_guarded_seat": state.last_guarded_seat,
                "pending_hunter": list(state.pending_hunter),
            }
            # 信任网络(上帝模式可见)
            view["trust_network"] = self._trust_network_for(room)
            # LLM 统计
            view["llm_stats"] = self.router.stats.snapshot()
            return view
        # 给公开/私密视图也附加 persona 映射(非上帝模式也能看到发言风格)
        view = state.public_view() if mode != "play" or seat is None else state.private_view_for(
            next((p.id for p in state.players if p.seat == seat), "")
        )
        view["personas"] = seat_to_persona
        return view

    def _trust_network_for(self, room: Room) -> dict[str, Any]:
        """导出所有 agent 的信任网络(god mode 视图)。"""
        # actor 的 key 是 player.id
        seat_map = {p.id: p.seat for p in room.state.players}
        network: dict[str, dict[str, float]] = {}
        for pid, actor in room.actors.items():
            seat = seat_map.get(pid)
            if seat is None:
                continue
            # actor.memory.trust: {seat -> suspicion}
            network[str(seat)] = {str(k): round(v, 3) for k, v in actor.memory.trust.items()}
        # 反思洞察
        reflections: dict[str, list[str]] = {}
        for pid, actor in room.actors.items():
            seat = seat_map.get(pid)
            if seat is None:
                continue
            recent_refl = [r.text for r in actor.memory.reflections[-3:]]
            if recent_refl:
                reflections[str(seat)] = recent_refl
        return {"trust": network, "reflections": reflections}

    # ------------------------------------------------------------------
    # 人类玩家操作(play 模式)
    # ------------------------------------------------------------------
    async def handle_human_action(self, room: Room, seat: int, action: dict[str, Any]) -> None:
        """处理人类玩家的操作(发言/投票/夜间行动)。

        将操作放入对应 AgentActor 的人类队列,由编排器消费。
        """
        player = next((p for p in room.state.players if p.seat == seat), None)
        if player is None:
            logger.warning("人类操作 seat=%s 不存在", seat)
            return
        actor = room.actors.get(player.id)
        if actor is None or not actor.is_human:
            logger.warning("seat=%s 不是人类玩家", seat)
            return
        actor.human_queue.put_nowait(action)
        logger.info("人类操作已入队 seat=%s action=%s", seat, action.get("type"))

    async def aclose(self) -> None:
        pending: list[asyncio.Task] = []
        for room in self.rooms.values():
            if room.task and not room.task.done():
                room.task.cancel()
                pending.append(room.task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await self.router.aclose()


def _room_timeout_from_env() -> float | None:
    raw = os.getenv("WEREWOLF_ROOM_TIMEOUT", "900")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 900.0
    return value if value > 0 else None


def _public_room_error(err: BaseException) -> str:
    """Public room-level error message; raw provider details stay in logs."""
    return f"{type(err).__name__} during game loop"


def _public_action(action: Any) -> str | None:
    """Map internal action names to public-safe categories."""
    if not action:
        return None
    name = str(action)
    public = {
        "speak": "speak",
        "vote": "vote",
        "last_words": "last_words",
        "hunter_shot": "hunter_shot",
    }
    return public.get(name)
