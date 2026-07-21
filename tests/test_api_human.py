"""人机混合 API 集成测试 —— 验证人类玩家操作队列端到端。"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from src.api.room_manager import RoomManager
from src.api.server import create_app

TEST_MODEL_CONFIG = {
    "provider": "openai",
    "model": "unit-test-model",
    "api_base": "https://example.invalid/v1",
    "api_key": "unit-test-key",
}


def _admin_headers(token: str) -> dict[str, str]:
    return {"X-Room-Token": token}


@pytest.fixture
def manager() -> RoomManager:
    from src.llm.router import LLMRouter

    async def _fake_run_room(room):
        # 不运行真实编排器;测试手动驱动 human actor
        return

    mgr = RoomManager(router=LLMRouter())
    mgr._run_room = _fake_run_room  # type: ignore[method-assign]
    return mgr


@pytest.fixture
def client(manager: RoomManager):
    app = create_app(manager=manager)
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.mark.asyncio
async def test_human_seat_receives_action_request(client, manager):
    """人类座位在 play 模式下应收到 human_action_request,提交后编排器能消费。"""
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [1],
        "model_config": TEST_MODEL_CONFIG,
    })
    room_id = res.json()["room_id"]
    admin_token = res.json()["admin_token"]
    seat_token = res.json()["seat_tokens"]["1"]
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None and room.orchestrator is not None
    human_pid = next(p.id for p in room.state.players if p.seat == 1)
    actor = room.actors[human_pid]
    assert actor.is_human

    with client.websocket_connect(f"/ws/{room_id}?seat=1&mode=play&token={seat_token}") as ws:
        ws.receive_json()  # snapshot

        # 模拟编排器已有一个当前真人请求,再广播给前端。
        actor.current_human_request = {
            "request_id": "req-api-vote",
            "action_type": "vote",
            "day": 1,
            "phase": "voting",
            "allowed_target_seats": [2, 3, 4, 5, 6],
        }
        request = {
            "type": "human_action_request",
            "request_id": "req-api-vote",
            "seat": 1,
            "action_type": "vote",
            "context": {"phase": "voting", "day": 1},
            "timeout": 90,
        }
        await room.orchestrator.on_event(request)

        received = ws.receive_json()
        assert received["type"] == "human_action_request"
        assert received["request_id"] == "req-api-vote"
        assert received["seat"] == 1
        assert received["action_type"] == "vote"

        # 过期/错误 request_id 不能进入队列。
        ws.send_json({"type": "human_action", "request_id": "stale", "action": "vote", "target_seat": 2})
        rejected = ws.receive_json()
        assert rejected["type"] == "human_action_rejected"
        assert rejected["request_id"] == "stale"
        assert rejected["reason"] == "request_id_mismatch"
        await asyncio.sleep(0.05)
        assert actor.human_queue.empty()

        # 当前 request 也必须携带 day/phase 绑定。
        ws.send_json({"type": "human_action", "request_id": "req-api-vote", "action": "vote", "target_seat": 2})
        rejected = ws.receive_json()
        assert rejected["type"] == "human_action_rejected"
        assert rejected["request_id"] == "req-api-vote"
        assert rejected["reason"] == "phase_missing"
        await asyncio.sleep(0.05)
        assert actor.human_queue.empty()

        # 前端提交当前投票。
        ws.send_json({
            "type": "human_action",
            "request_id": "req-api-vote",
            "day": 1,
            "phase": "voting",
            "action": "vote",
            "target_seat": 2,
        })
        accepted = ws.receive_json()
        assert accepted["type"] == "human_action_accepted"
        assert accepted["request_id"] == "req-api-vote"

        # 给事件循环机会处理 WS 消息入队
        await asyncio.sleep(0.05)

        # 消费队列验证收到了操作
        assert not actor.human_queue.empty()
        action = actor.human_queue.get_nowait()
        assert action["request_id"] == "req-api-vote"
        assert action["action"] == "vote"
        assert action["target_seat"] == 2


@pytest.mark.asyncio
async def test_human_action_rejects_malformed_websocket_payloads(client, manager):
    """play WS 收到坏 JSON/非对象 payload 时要明确拒绝,不能静默吞掉。"""
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [1],
        "model_config": TEST_MODEL_CONFIG,
    })
    room_id = res.json()["room_id"]
    admin_token = res.json()["admin_token"]
    seat_token = res.json()["seat_tokens"]["1"]
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None
    human_pid = next(p.id for p in room.state.players if p.seat == 1)
    actor = room.actors[human_pid]

    with client.websocket_connect(f"/ws/{room_id}?seat=1&mode=play&token={seat_token}") as ws:
        ws.receive_json()  # snapshot

        ws.send_text("{bad json")
        rejected = ws.receive_json()
        assert rejected == {
            "type": "human_action_rejected",
            "seat": 1,
            "request_id": "",
            "reason": "invalid_payload",
        }
        await asyncio.sleep(0.05)
        assert actor.human_queue.empty()

        ws.send_json(["not", "an", "object"])
        rejected = ws.receive_json()
        assert rejected == {
            "type": "human_action_rejected",
            "seat": 1,
            "request_id": "",
            "reason": "invalid_payload",
        }
        await asyncio.sleep(0.05)
        assert actor.human_queue.empty()


@pytest.mark.asyncio
async def test_spectator_does_not_receive_human_action_request(client, manager):
    """观战者不应收到 human_action_request。"""
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [1],
        "model_config": TEST_MODEL_CONFIG,
    })
    room_id = res.json()["room_id"]
    admin_token = res.json()["admin_token"]
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None and room.orchestrator is not None

    with client.websocket_connect(f"/ws/{room_id}?mode=spectate") as ws:
        ws.receive_json()  # snapshot
        await room.orchestrator.on_event({
            "type": "human_action_request",
            "request_id": "req-spectator",
            "seat": 1,
            "action_type": "vote",
            "context": {},
            "timeout": 90,
        })
        with pytest.raises(Exception):
            ws.receive_json(timeout=0.2)
