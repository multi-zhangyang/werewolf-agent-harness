"""WebSocket 实时事件广播测试 —— 验证后端能持续向客户端推送事件。"""
from __future__ import annotations

import json

import pytest
from starlette.websockets import WebSocketDisconnect

from src.agent.information import build_observation
from src.api.room_manager import RoomManager
from src.api.server import create_app
TEST_MODEL_CONFIG = {
    "provider": "openai",
    "model": "unit-test-model",
    "api_base": "https://example.invalid/v1",
    "api_key": "unit-test-key",
}


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
        "model_config": TEST_MODEL_CONFIG,
    })
    assert res.status_code == 200
    body = res.json()
    return body["room_id"], body["admin_token"], body.get("seat_tokens", {})


def test_websocket_origin_must_match_exact_browser_allowlist(client):
    room_id, _, _ = _create_room(client)

    with pytest.raises(WebSocketDisconnect) as denied:
        with client.websocket_connect(
            f"/ws/{room_id}?mode=spectate",
            headers={"Origin": "https://attacker.invalid"},
        ):
            pass
    assert denied.value.code == 4403

    with client.websocket_connect(
        f"/ws/{room_id}?mode=spectate",
        headers={"Origin": "http://localhost:5173"},
    ) as websocket:
        assert websocket.receive_json()["type"] == "snapshot"


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
async def test_websocket_reconnect_since_delivers_only_missing_event(client, manager):
    room_id, _, _ = _create_room(client)
    room = manager.get_room(room_id)
    assert room is not None
    for day in (1, 2):
        await manager._broadcast(
            room,
            {"type": "phase_started", "phase": "day", "day": day},
            initial_replay=True,
        )

    with client.websocket_connect(f"/ws/{room_id}?mode=spectate") as first_ws:
        first_snapshot = first_ws.receive_json()
        first = first_ws.receive_json()
        second = first_ws.receive_json()
        assert first_snapshot["cursor"] == 2
        assert [first["delivery_seq"], second["delivery_seq"]] == [1, 2]

    await manager._broadcast(
        room,
        {"type": "phase_started", "phase": "day", "day": 3},
        initial_replay=True,
    )
    with client.websocket_connect(f"/ws/{room_id}?mode=spectate&since=2") as resumed_ws:
        resumed_snapshot = resumed_ws.receive_json()
        missing = resumed_ws.receive_json()
        assert resumed_snapshot["stream_id"] == first_snapshot["stream_id"]
        assert resumed_snapshot["resumed_from"] == 2
        assert resumed_snapshot["cursor"] == 3
        assert missing["delivery_seq"] == 3
        assert missing["day"] == 3


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


def test_websocket_replay_accepts_incomplete_terminal_room(client, manager):
    room_id, admin_token, _ = _create_room(client)
    room = manager.get_room(room_id)
    assert room is not None
    room.status = "incomplete"
    room.end_reason = "no_progress"

    with client.websocket_connect(
        f"/ws/{room_id}?mode=replay&token={admin_token}"
    ) as websocket:
        snapshot = websocket.receive_json()

    assert snapshot["type"] == "snapshot"
    assert snapshot["status"] == "incomplete"
    assert snapshot["view"]["god"] is True


def test_websocket_god_requires_admin_token(client):
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/ws/{room_id}?mode=god"):
            pass
    assert exc.value.code == 4403


def test_public_and_player_snapshots_hide_persona_strategic_priors(client):
    room_id, admin_token, seat_tokens = _create_room(client, human_seats=[1])
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))
    room = client.app.state.room_manager.get_room(room_id)
    assert room is not None
    for actor in room.actors.values():
        actor.persona_name = f"private-persona-{actor.seat}"

    with client.websocket_connect(f"/ws/{room_id}?mode=spectate") as spectator:
        public = spectator.receive_json()
        assert "personas" not in public["view"]
        assert all("persona" not in player for player in public["view"].get("players", []))

    with client.websocket_connect(
        f"/ws/{room_id}?mode=play&seat=1&token={seat_tokens['1']}"
    ) as player:
        private = player.receive_json()
        assert "personas" not in private["view"]
        assert all("persona" not in player for player in private["view"].get("players", []))

    with client.websocket_connect(f"/ws/{room_id}?mode=god&token={admin_token}") as god:
        full = god.receive_json()["view"]["players_full"]
        assert any(item.get("persona") == "private-persona-1" for item in full)


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


@pytest.mark.parametrize("cursor", ["-1", "+1", "1.5", "abc", ""])
def test_websocket_rejects_malformed_delivery_cursor(client, cursor):
    room_id, _, _ = _create_room(client)

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/ws/{room_id}?mode=spectate&since={cursor}"):
            pass
    assert exc.value.code == 4400


def test_websocket_rejects_future_delivery_cursor(client):
    room_id, _, _ = _create_room(client)

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/ws/{room_id}?mode=spectate&since=1") as ws:
            ws.receive_json()
    assert exc.value.code == 4400


def test_websocket_reports_explicit_retained_history_gap():
    import asyncio
    from fastapi.testclient import TestClient

    isolated_manager = RoomManager(ws_delivery_history_size=2)
    with TestClient(create_app(manager=isolated_manager)) as isolated_client:
        room_id, _, _ = _create_room(isolated_client)
        room = isolated_manager.get_room(room_id)
        assert room is not None
        for day in range(1, 5):
            asyncio.run(isolated_manager._broadcast(
                room,
                {"type": "phase_started", "phase": "day", "day": day},
                initial_replay=True,
            ))

        with pytest.raises(WebSocketDisconnect) as exc:
            with isolated_client.websocket_connect(
                f"/ws/{room_id}?mode=spectate&since=1"
            ) as ws:
                ws.receive_json()
        assert exc.value.code == 4409


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
async def test_play_public_event_strips_hidden_fields_like_spectate(client, manager):
    """公开事件即使发给 play 客户端,也不能保留隐藏身份/推理字段。"""
    room_id, admin_token, seat_tokens = _create_room(client, human_seats=[1])
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None and room.orchestrator is not None

    with (
        client.websocket_connect(f"/ws/{room_id}?mode=spectate") as spectate_ws,
        client.websocket_connect(f"/ws/{room_id}?mode=play&seat=1&token={seat_tokens['1']}") as play_ws,
    ):
        spectate_ws.receive_json()
        play_ws.receive_json()

        event = {
            "type": "speech",
            "phase": "day",
            "day": 1,
            "seat": 3,
            "name": "C",
            "text": "hello",
            "role": "werewolf",
            "team": "werewolves",
            "reasoning": "hidden",
            "thought": "secret",
            "teammates": [5],
        }
        await room.orchestrator.on_event(event)

        spectate_msg = spectate_ws.receive_json()
        play_msg = play_ws.receive_json()

        for msg in (spectate_msg, play_msg):
            assert msg["type"] == "speech"
            assert msg["text"] == "hello"
            assert "role" not in msg
            assert "team" not in msg
            assert "reasoning" not in msg
            assert "thought" not in msg
            assert "teammates" not in msg


@pytest.mark.asyncio
async def test_private_reasoning_uses_admin_trace_only_and_never_websocket(client, manager):
    room_id, admin_token, seat_tokens = _create_room(client, human_seats=[1])
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None and room.orchestrator is not None
    marker = "model-private-reasoning-ws-sentinel"

    with (
        client.websocket_connect(f"/ws/{room_id}?mode=spectate") as spectate_ws,
        client.websocket_connect(
            f"/ws/{room_id}?mode=play&seat=1&token={seat_tokens['1']}"
        ) as play_ws,
        client.websocket_connect(f"/ws/{room_id}?mode=god&token={admin_token}") as god_ws,
    ):
        for websocket in (spectate_ws, play_ws, god_ws):
            assert websocket.receive_json()["type"] == "snapshot"

        room.orchestrator.on_trace({
            "type": "agent_response",
            "seat": 2,
            "phase": "day",
            "envelope": {
                "decision": {
                    "action": "speak",
                    "speech": "public cover story",
                    "reasoning": marker,
                },
            },
        })

        trace = client.get(
            f"/api/rooms/{room_id}/trace",
            headers=_admin_headers(admin_token),
        )
        assert trace.status_code == 200
        decision = next(
            item for item in trace.json()["trace"]
            if item["kind"] == "decision" and item["payload"].get("envelope")
        )
        assert decision["payload"]["envelope"]["decision"]["reasoning"] == marker

        # Recording a decision trace does not create a game event on any live
        # stream, including the admin-authenticated god stream.
        for websocket in (spectate_ws, play_ws, god_ws):
            with pytest.raises(Exception):
                websocket.receive_json(timeout=0.2)

        await room.orchestrator.on_event({
            "type": "speech",
            "phase": "day",
            "day": 1,
            "seat": 2,
            "name": "B",
            "text": "public cover story",
            "reasoning": marker,
            "nested": {
                "thought": marker,
                "items": [{"private_reasoning": marker}],
            },
        })
        for websocket in (spectate_ws, play_ws, god_ws):
            delivered = websocket.receive_json()
            assert delivered["type"] == "speech"
            assert marker not in json.dumps(delivered, ensure_ascii=False)

    event_artifacts = {
        "event_history": room.event_history,
        "delivery_source_history": [item[1] for item in room.delivery_source_history],
        "delivery_streams": {
            key: [record.payload for record in stream.history]
            for key, stream in room.delivery_streams.items()
        },
    }
    assert marker not in json.dumps(event_artifacts, ensure_ascii=False, default=str)

    for actor in room.actors.values():
        memory = [
            {"text": item.text, "metadata": item.metadata}
            for item in actor.memory.observations
        ]
        assert marker not in json.dumps(memory, ensure_ascii=False, default=str)
    for player in room.state.players:
        observation = build_observation(room.state, player.id).model_dump()
        assert marker not in json.dumps(observation, ensure_ascii=False, default=str)

    public_room = client.get(f"/api/rooms/{room_id}")
    assert public_room.status_code == 200
    assert marker not in json.dumps(public_room.json(), ensure_ascii=False)


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
        assert msg["delivery_seq"] >= 1
        assert isinstance(msg["delivery_id"], str)
        assert {
            key: value
            for key, value in msg.items()
            if key not in {"delivery_seq", "delivery_id"}
        } == {
            "type": "agent_decision_failed",
            "phase": "night",
            "reason": "AI 决策失败,本请求未产生 DecisionEnvelope。",
            "agent_kind": "llm",
            "timeout": True,
        }


@pytest.mark.asyncio
async def test_live_analysis_remains_factual_without_synthetic_social_metrics(client, manager):
    """Final analysis is forwarded without adding inferred social scores."""
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    room = manager.get_room(room_id)
    assert room is not None and room.orchestrator is not None

    with client.websocket_connect(f"/ws/{room_id}?mode=god&token={admin_token}") as god_ws:
        god_ws.receive_json()

        await room.orchestrator.on_event({
            "type": "speech",
            "phase": "day",
            "day": 1,
            "seat": 1,
            "name": "A",
            "text": "我觉得2号需要解释一下。",
            "reply_to": 2,
            "accuses": [2],
        })
        god_ws.receive_json()

        await room.orchestrator.on_event({
            "type": "analysis",
            "analysis": {"winner": "village", "days": 1, "seats": []},
        })
        msg = god_ws.receive_json()

        assert msg["type"] == "analysis"
        assert msg["analysis"] == {"winner": "village", "days": 1, "seats": []}
        assert "social_metrics" not in msg["analysis"]
