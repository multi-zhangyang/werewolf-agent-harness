"""WebSocket 实时事件广播测试 —— 验证后端能持续向客户端推送事件。"""
from __future__ import annotations

import pytest
from starlette.websockets import WebSocketDisconnect

from src.api.room_manager import RoomManager
from src.api.server import create_app


@pytest.fixture
def manager() -> RoomManager:
    from src.llm.router import LLMRouter

    async def _fake_run_room(room):
        # 不运行真实编排器;测试手动触发事件
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


def _admin_headers(token: str) -> dict[str, str]:
    return {"X-Room-Token": token}


def _create_room(client, *, human_seats: list[int] | None = None):
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": human_seats or [],
    })
    assert res.status_code == 200
    body = res.json()
    return body["room_id"], body["admin_token"], body.get("seat_tokens", {})


@pytest.mark.asyncio
async def test_websocket_receives_events_after_snapshot(client, manager):
    """连接 spectate 后,后端推送的事件必须实时到达前端。"""
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None
    assert room.orchestrator is not None

    with client.websocket_connect(f"/ws/{room_id}?mode=spectate") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"

        event = {
            "type": "phase_started",
            "phase": "day",
            "day": 1,
            "message": "天亮了",
        }
        await room.orchestrator.on_event(event)

        received = ws.receive_json()
        assert received["type"] == "phase_started"
        assert received["phase"] == "day"


@pytest.mark.asyncio
async def test_websocket_spectator_does_not_see_private_events(client, manager):
    """观战者不应收到私有事件(如 seer_result)。"""
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None and room.orchestrator is not None

    with client.websocket_connect(f"/ws/{room_id}?mode=spectate") as ws:
        ws.receive_json()  # snapshot

        private_event = {
            "type": "seer_result",
            "phase": "night",
            "day": 1,
            "message": "查验结果",
            "visibility": "private",
            "recipients": [room.state.players[0].id],
        }
        await room.orchestrator.on_event(private_event)

        with pytest.raises(Exception):  # 会超时抛异常
            ws.receive_json(timeout=0.2)


@pytest.mark.asyncio
async def test_websocket_play_receives_only_own_private_events(client, manager):
    """play 模式应收到自己 recipients 的私有事件,其他座位不应收到。"""
    room_id, admin_token, seat_tokens = _create_room(client, human_seats=[1, 2])
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None and room.orchestrator is not None
    seat1_pid = room.state.players[0].id

    with (
        client.websocket_connect(f"/ws/{room_id}?mode=play&seat=1&token={seat_tokens['1']}") as seat1_ws,
        client.websocket_connect(f"/ws/{room_id}?mode=play&seat=2&token={seat_tokens['2']}") as seat2_ws,
    ):
        seat1_ws.receive_json()  # snapshot
        seat2_ws.receive_json()  # snapshot

        private_event = {
            "type": "seer_result",
            "phase": "night",
            "day": 1,
            "message": "查验结果",
            "visibility": "private",
            "recipients": [seat1_pid],
        }
        await room.orchestrator.on_event(private_event)

        received = seat1_ws.receive_json()
        assert received["type"] == "seer_result"
        assert received["message"] == "查验结果"
        with pytest.raises(Exception):
            seat2_ws.receive_json(timeout=0.2)


def test_websocket_replay_rejected_before_game_ended(client):
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/ws/{room_id}?mode=replay&token={admin_token}"):
            pass
    assert exc.value.code == 4409


def test_websocket_god_requires_admin_token(client):
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/ws/{room_id}?mode=god"):
            pass
    assert exc.value.code == 4403


def test_websocket_play_requires_matching_seat_token(client):
    room_id, admin_token, seat_tokens = _create_room(client, human_seats=[1, 2])
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/ws/{room_id}?mode=play&seat=1"):
            pass
    assert exc.value.code == 4403

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/ws/{room_id}?mode=play&seat=1&token={seat_tokens['2']}"):
            pass
    assert exc.value.code == 4403

    with client.websocket_connect(f"/ws/{room_id}?mode=play&seat=1&token={seat_tokens['1']}") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"
        assert msg["view"]["self"]["seat"] == 1


def test_websocket_spectate_rejects_seat_parameter(client):
    room_id, admin_token, _ = _create_room(client, human_seats=[1])
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/ws/{room_id}?mode=spectate&seat=1"):
            pass
    assert exc.value.code == 4400


def test_websocket_snapshot_hides_hidden_role_state_except_god(client, manager):
    """公开快照不泄漏技能资源;god 快照单独携带 hidden_state。"""
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None
    room.state.witch_antidote = False
    room.state.witch_poison = False
    room.state.last_guarded_seat = 4
    room.state.pending_hunter = [room.state.players[0].id]

    with client.websocket_connect(f"/ws/{room_id}?mode=spectate") as ws:
        msg = ws.receive_json()
        view = msg["view"]
        assert "witch_antidote" not in view
        assert "witch_poison" not in view
        assert "last_guarded_seat" not in view
        assert "pending_hunter" not in view
        assert "hidden_state" not in view

    with client.websocket_connect(f"/ws/{room_id}?mode=god&token={admin_token}") as ws:
        msg = ws.receive_json()
        hidden = msg["view"]["hidden_state"]
        assert hidden == {
            "witch_antidote": False,
            "witch_poison": False,
            "last_guarded_seat": 4,
            "pending_hunter": [room.state.players[0].id],
        }


@pytest.mark.asyncio
async def test_private_marker_overrides_public_allowlist(client, manager):
    """即使 type 是 speech,private 标记也必须优先于公开白名单。"""
    room_id, admin_token, seat_tokens = _create_room(client, human_seats=[1, 2])
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None and room.orchestrator is not None
    seat1_pid = room.state.players[0].id

    with (
        client.websocket_connect(f"/ws/{room_id}?mode=spectate") as spectate_ws,
        client.websocket_connect(f"/ws/{room_id}?mode=play&seat=1&token={seat_tokens['1']}") as seat1_ws,
        client.websocket_connect(f"/ws/{room_id}?mode=play&seat=2&token={seat_tokens['2']}") as seat2_ws,
    ):
        spectate_ws.receive_json()
        seat1_ws.receive_json()
        seat2_ws.receive_json()

        event = {
            "type": "speech",
            "phase": "day",
            "day": 1,
            "seat": 1,
            "name": "A",
            "text": "PRIVATE",
            "visibility": "private",
            "recipients": [seat1_pid],
        }
        await room.orchestrator.on_event(event)

        received = seat1_ws.receive_json()
        assert received["type"] == "speech"
        assert received["text"] == "PRIVATE"
        assert "visibility" not in received
        assert "recipients" not in received

        with pytest.raises(Exception):
            spectate_ws.receive_json(timeout=0.2)
        with pytest.raises(Exception):
            seat2_ws.receive_json(timeout=0.2)


@pytest.mark.asyncio
async def test_spectator_night_resolved_hides_death_reason_live_and_history(client, manager):
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None and room.orchestrator is not None
    event = {
        "type": "night_resolved",
        "day": 1,
        "deaths": [{"seat": 2, "name": "B", "reason": "wolf_kill"}],
    }

    with client.websocket_connect(f"/ws/{room_id}?mode=spectate") as ws:
        ws.receive_json()
        await room.orchestrator.on_event(event)
        live = ws.receive_json()
        assert live["deaths"] == [{"seat": 2, "name": "B"}]

    with client.websocket_connect(f"/ws/{room_id}?mode=spectate") as ws:
        ws.receive_json()
        history = ws.receive_json()
        assert history["type"] == "night_resolved"
        assert history["deaths"] == [{"seat": 2, "name": "B"}]


@pytest.mark.asyncio
async def test_spectator_agent_decision_failed_hides_raw_night_details(client, manager):
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None and room.orchestrator is not None
    event = {
        "type": "agent_decision_failed",
        "seat": 4,
        "phase": "night",
        "action": "seer_action",
        "reason": "seer_action provider leaked raw stack",
        "timeout": True,
    }

    with client.websocket_connect(f"/ws/{room_id}?mode=spectate") as ws:
        ws.receive_json()
        await room.orchestrator.on_event(event)
        msg = ws.receive_json()
        assert msg == {
            "type": "agent_decision_failed",
            "phase": "night",
            "reason": "AI 决策失败,已按规则跳过。",
            "timeout": True,
        }


@pytest.mark.asyncio
async def test_spectator_thinking_stream_hides_full_reasoning(client, manager):
    """观战只收整理摘要;完整 reasoning 只给 god 模式。"""
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None and room.orchestrator is not None

    with (
        client.websocket_connect(f"/ws/{room_id}?mode=spectate") as spectate_ws,
        client.websocket_connect(f"/ws/{room_id}?mode=god&token={admin_token}") as god_ws,
    ):
        spectate_ws.receive_json()  # snapshot
        god_ws.receive_json()  # snapshot

        thinking = {
            "seat": 3,
            "action": "speak",
            "summary": "我会观察公开发言。",
            "reasoning": "我是狼人,这轮要误导好人投4号。",
            "suspicion_top": [{"seat": 4, "suspicion": 0.7}],
        }
        await room.orchestrator.on_thinking(thinking)

        spectate_msg = spectate_ws.receive_json()
        god_msg = god_ws.receive_json()

        assert spectate_msg["type"] == "agent_thinking"
        assert spectate_msg["summary"] == "AI 思考已记录,隐藏推理赛后由授权复盘查看。"
        assert "reasoning" not in spectate_msg
        assert "suspicion_top" not in spectate_msg

        assert god_msg["type"] == "agent_thinking"
        assert god_msg["reasoning"] == "我是狼人,这轮要误导好人投4号。"


@pytest.mark.asyncio
async def test_spectator_does_not_receive_night_thinking_but_god_does(client, manager):
    """夜间思考即使脱敏也会泄露角色行动,只能给 god。"""
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None and room.orchestrator is not None

    with (
        client.websocket_connect(f"/ws/{room_id}?mode=spectate") as spectate_ws,
        client.websocket_connect(f"/ws/{room_id}?mode=god&token={admin_token}") as god_ws,
    ):
        spectate_ws.receive_json()  # snapshot
        god_ws.receive_json()  # snapshot

        thinking = {
            "seat": 5,
            "action": "night_kill",
            "summary": "夜间决策已记录。",
            "reasoning": "我是狼人,今晚准备刀2号。",
            "suspicion_top": [{"seat": 2, "suspicion": 0.8}],
        }
        await room.orchestrator.on_thinking(thinking)

        god_msg = god_ws.receive_json()
        assert god_msg["type"] == "agent_thinking"
        assert god_msg["action"] == "night_kill"
        assert god_msg["reasoning"] == "我是狼人,今晚准备刀2号。"

        with pytest.raises(Exception):
            spectate_ws.receive_json(timeout=0.2)
