"""FastAPI REST + WebSocket 集成测试。

避免真实 LLM 调用:只验证房间生命周期、信息隔离广播骨架、
WebSocket 快照与事件流能正常工作。
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import src.api.server as server_module
from src.api.server import create_app
from src.game.models import Phase


@pytest.fixture
def client():
    from src.api.room_manager import RoomManager

    async def _noop_run_room(room):
        # 不启动真实编排器,避免 LLM 调用;deal_roles 已在 start_game 中完成
        return

    manager = RoomManager()
    manager._run_room = _noop_run_room  # type: ignore[method-assign]
    app = create_app(manager=manager)
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


def test_providers(client):
    res = client.get("/api/providers")
    assert res.status_code == 200
    providers = res.json()
    assert {"openai", "openai_responses", "anthropic"}.issubset(providers)
    assert "Chat Completions" in providers["openai"]["label"]
    assert "Responses" in providers["openai_responses"]["label"]
    assert "Messages" in providers["anthropic"]["label"]
    serialized = json.dumps(providers)
    assert "api_key" not in serialized
    for vendor_name in ("Kimchi", "DeepSeek", "Moonshot", "vLLM", "minimax"):
        assert vendor_name not in serialized


@pytest.mark.parametrize("provider", ["openai", "openai_responses", "anthropic"])
def test_config_hides_api_key(client, provider):
    secret = "unit-test-secret-key-should-not-leak"
    with patch("src.api.server.DEFAULT_MODEL_CONFIG", {
        "provider": provider,
        "model": "model-a",
        "api_base": "https://example.invalid/v1",
        "api_key": secret,
        "temperature": 0.85,
        "max_tokens": 0,
        "use_json_format": False,
    }):
        res = client.get("/api/config")

    assert res.status_code == 200
    cfg = res.json()
    assert cfg["provider"] == provider
    assert cfg["api_key"] == ""
    assert cfg["api_key_configured"] is True
    assert secret not in json.dumps(cfg)
    assert secret[:6] not in json.dumps(cfg)


@pytest.mark.parametrize("provider", ["openai", "openai_responses", "anthropic"])
def test_create_room_accepts_each_standard_provider_with_explicit_key(client, provider):
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "model_config": {
            "provider": provider,
            "model": "model-a",
            "api_base": "https://example.invalid/v1",
            "api_key": "explicit-key",
        },
    })

    assert res.status_code == 200
    room = client.app.state.room_manager.get_room(res.json()["room_id"])
    assert room.default_config.provider == provider
    assert room.default_config.model == "model-a"
    assert room.default_config.api_base == "https://example.invalid/v1"
    assert room.default_config.api_key == "explicit-key"


def test_create_room_rejects_default_key_reuse_when_api_base_changes(client):
    with patch("src.api.server.DEFAULT_MODEL_CONFIG", {
        "provider": "openai",
        "model": "model-a",
        "api_base": "https://default.example/v1",
        "api_key": "unit-test-secret",
        "temperature": 0.85,
        "max_tokens": 0,
        "use_json_format": False,
    }):
        res = client.post("/api/rooms", json={
            "player_names": ["A", "B", "C", "D", "E", "F"],
            "model_config": {
                "model": "attacker-model",
                "api_base": "https://attacker.invalid/openai/v1",
            },
        })

    assert res.status_code == 400
    assert "api_key" in res.json()["detail"]


@pytest.mark.parametrize("provider", ["openai_responses", "anthropic"])
def test_create_room_rejects_default_key_reuse_when_provider_changes(client, provider):
    with patch("src.api.server.DEFAULT_MODEL_CONFIG", {
        "provider": "openai",
        "model": "model-a",
        "api_base": "https://default.example/v1",
        "api_key": "unit-test-secret",
        "temperature": 0.85,
        "max_tokens": 0,
        "use_json_format": False,
    }):
        res = client.post("/api/rooms", json={
            "player_names": ["A", "B", "C", "D", "E", "F"],
            "model_config": {
                "provider": provider,
                "model": "other-model",
            },
        })

    assert res.status_code == 400
    assert "api_key" in res.json()["detail"]


def test_create_room_accepts_api_base_change_with_explicit_key(client):
    with patch("src.api.server.DEFAULT_MODEL_CONFIG", {
        "provider": "openai",
        "model": "model-a",
        "api_base": "https://default.example/v1",
        "api_key": "unit-test-secret",
        "temperature": 0.85,
        "max_tokens": 0,
        "use_json_format": False,
    }):
        res = client.post("/api/rooms", json={
            "player_names": ["A", "B", "C", "D", "E", "F"],
            "model_config": {
                "model": "other-model",
                "api_base": "https://other.example/v1",
                "api_key": "explicit-key",
            },
        })

    assert res.status_code == 200
    room = client.app.state.room_manager.get_room(res.json()["room_id"])
    assert room.default_config.api_base == "https://other.example/v1"
    assert room.default_config.api_key == "explicit-key"


@pytest.mark.parametrize("provider", ["openai_responses", "anthropic"])
def test_create_room_accepts_provider_change_with_explicit_key(client, provider):
    with patch("src.api.server.DEFAULT_MODEL_CONFIG", {
        "provider": "openai",
        "model": "model-a",
        "api_base": "https://default.example/v1",
        "api_key": "unit-test-secret",
        "temperature": 0.85,
        "max_tokens": 0,
        "use_json_format": False,
    }):
        res = client.post("/api/rooms", json={
            "player_names": ["A", "B", "C", "D", "E", "F"],
            "model_config": {
                "provider": provider,
                "model": "other-model",
                "api_base": "https://other.example/v1",
                "api_key": "explicit-key",
            },
        })

    assert res.status_code == 200
    room = client.app.state.room_manager.get_room(res.json()["room_id"])
    assert room.default_config.provider == provider
    assert room.default_config.api_key == "explicit-key"


def test_spa_fallback_does_not_serve_files_outside_dist(monkeypatch, tmp_path):
    """Encoded traversal must not expose files outside frontend/dist."""
    from src.api.room_manager import RoomManager

    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("SPA INDEX", encoding="utf-8")
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("SENTINEL_SHOULD_NOT_LEAK", encoding="utf-8")
    monkeypatch.setattr(server_module, "FRONTEND_DIR", dist)

    app = create_app(manager=RoomManager())
    with TestClient(app) as c:
        res = c.get("/%2e%2e/sentinel.txt")

    assert res.status_code == 200
    assert "SPA INDEX" in res.text
    assert "SENTINEL_SHOULD_NOT_LEAK" not in res.text


def test_missing_frontend_dist_returns_explicit_setup_page(monkeypatch, tmp_path):
    """A clean checkout without frontend/dist must not serve raw Vite TSX entry."""
    from src.api.room_manager import RoomManager

    missing_dist = tmp_path / "frontend" / "dist"
    (missing_dist.parent).mkdir(parents=True)
    (missing_dist.parent / "index.html").write_text(
        '<script type="module" src="/src/main.tsx"></script>',
        encoding="utf-8",
    )
    monkeypatch.setattr(server_module, "FRONTEND_DIR", missing_dist)

    app = create_app(manager=RoomManager())
    with TestClient(app) as c:
        res = c.get("/")

    assert res.status_code == 503
    assert "前端尚未构建" in res.text
    assert "npm run dev" in res.text
    assert "npm run build" in res.text
    assert "/src/main.tsx" not in res.text


def test_create_and_get_room(client):
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [],
    })
    assert res.status_code == 200
    data = res.json()
    assert "room_id" in data
    assert data["admin_token"]
    assert len(data["players"]) == 6

    room_id = data["room_id"]
    res2 = client.get(f"/api/rooms/{room_id}")
    assert res2.status_code == 200
    assert res2.json()["room_id"] == room_id
    assert res2.json()["status"] == "waiting"


def test_room_endpoints_preserve_human_seats(client):
    """REST responses must expose declared human seats without leaking roles."""
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [5, 2],
    })
    assert res.status_code == 200
    room_id = res.json()["room_id"]
    admin_token = res.json()["admin_token"]
    assert res.json()["human_seats"] == [2, 5]

    waiting = client.get(f"/api/rooms/{room_id}")
    assert waiting.status_code == 200
    assert waiting.json()["human_seats"] == [2, 5]
    assert all("role" not in p for p in waiting.json()["players"])

    started = client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))
    assert started.status_code == 200
    assert started.json()["human_seats"] == [2, 5]

    running = client.get(f"/api/rooms/{room_id}")
    assert running.status_code == 200
    assert running.json()["human_seats"] == [2, 5]
    assert all("role" not in p for p in running.json()["players"])

    room = client.app.state.room_manager.get_room(room_id)
    room.status = "ended"
    replay = client.get(f"/api/rooms/{room_id}/replay", headers=_admin_headers(admin_token))
    assert replay.status_code == 200
    assert replay.json()["human_seats"] == [2, 5]


@pytest.mark.parametrize("human_seats", [[0], [7], [-1], [1, 99]])
def test_create_room_rejects_human_seats_outside_player_range(client, human_seats):
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": human_seats,
    })

    assert res.status_code == 400
    assert "human_seats" in res.json()["detail"]


def test_start_game_builds_human_actors_for_declared_seats(client):
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [1, 6],
    })
    assert res.status_code == 200
    room_id = res.json()["room_id"]
    admin_token = res.json()["admin_token"]

    started = client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))
    assert started.status_code == 200
    room = client.app.state.room_manager.get_room(room_id)
    human_actor_seats = sorted(actor.seat for actor in room.actors.values() if actor.is_human)
    assert human_actor_seats == [1, 6]


def test_create_room_rejects_unsupported_deck_field(client):
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [],
        "deck": ["werewolf", "werewolf", "seer", "witch", "villager", "villager"],
    })

    assert res.status_code == 400
    assert "角色板" in res.json()["detail"]


def test_create_room_accepts_temperature_zero(client):
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [],
        "model_config": {"temperature": 0},
    })
    assert res.status_code == 200
    room = client.app.state.room_manager.get_room(res.json()["room_id"])
    assert room.default_config.temperature == 0


def test_create_room_rejects_unknown_top_level_fields(client):
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [],
        "extra_body": {"x": 1},
    })

    assert res.status_code == 422


@pytest.mark.parametrize("field_name", ["extra_body", "thinking", "reasoning_effort", "top_k"])
def test_create_room_rejects_non_standard_model_config_fields(client, field_name):
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [],
        "model_config": {
            "provider": "openai",
            "model": "test-model",
            "api_base": "https://example.invalid/v1",
            "api_key": "explicit-key",
            field_name: {"enabled": True},
        },
    })

    assert res.status_code == 400
    assert field_name in res.json()["detail"]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("thinking", ""), ("top_k", None), ("reasoning_effort", "")],
)
def test_create_room_rejects_empty_non_standard_model_config_fields(client, field_name, value):
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [],
        "model_config": {field_name: value},
    })

    assert res.status_code == 400
    assert field_name in res.json()["detail"]


def test_start_game_needs_waiting_room(client):
    room_id, admin_token, _ = _create_room(client)
    res2 = client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))
    assert res2.status_code == 200
    assert res2.json()["status"] == "running"

    res3 = client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))
    assert res3.status_code == 400


def test_get_room_running_hides_roles(client):
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))

    res2 = client.get(f"/api/rooms/{room_id}")
    assert res2.status_code == 200
    assert res2.json()["status"] == "running"
    assert all("role" not in p for p in res2.json()["players"])


def test_set_seat_config_only_while_waiting(client):
    room_id, admin_token, _ = _create_room(client)
    res2 = client.post(f"/api/rooms/{room_id}/seats/1/model_config", json={
        "model": "test-model",
        "temperature": 0.5,
    }, headers=_admin_headers(admin_token))
    assert res2.status_code == 200

    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))
    res3 = client.post(f"/api/rooms/{room_id}/seats/2/model_config", json={
        "model": "other-model",
    }, headers=_admin_headers(admin_token))
    assert res3.status_code == 400


@pytest.mark.parametrize("field_name", ["extra_body", "thinking", "reasoning_effort", "top_k"])
def test_set_seat_config_rejects_non_standard_model_config_fields(client, field_name):
    room_id, admin_token, _ = _create_room(client)
    res = client.post(f"/api/rooms/{room_id}/seats/1/model_config", json={
        "model": "seat-model",
        field_name: {"enabled": True},
    }, headers=_admin_headers(admin_token))

    assert res.status_code == 400
    assert field_name in res.json()["detail"]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("thinking", ""), ("top_k", None), ("reasoning_effort", "")],
)
def test_set_seat_config_rejects_empty_non_standard_model_config_fields(client, field_name, value):
    room_id, admin_token, _ = _create_room(client)
    res = client.post(f"/api/rooms/{room_id}/seats/1/model_config", json={
        "model": "seat-model",
        field_name: value,
    }, headers=_admin_headers(admin_token))

    assert res.status_code == 400
    assert field_name in res.json()["detail"]


def test_set_seat_config_rejects_default_key_reuse_when_api_base_changes(client):
    room_id, admin_token, _ = _create_room(client)
    room = client.app.state.room_manager.get_room(room_id)
    room.default_config = server_module.ModelConfig(
        provider="openai",
        model="default-model",
        api_base="https://default.example/v1",
        api_key="unit-test-secret",
    )

    res = client.post(f"/api/rooms/{room_id}/seats/1/model_config", json={
        "model": "seat-model",
        "api_base": "https://attacker.invalid/v1",
    }, headers=_admin_headers(admin_token))

    assert res.status_code == 400
    assert "api_key" in res.json()["detail"]


@pytest.mark.parametrize("provider", ["openai_responses", "anthropic"])
def test_set_seat_config_rejects_default_key_reuse_when_provider_changes(client, provider):
    room_id, admin_token, _ = _create_room(client)
    room = client.app.state.room_manager.get_room(room_id)
    room.default_config = server_module.ModelConfig(
        provider="openai",
        model="default-model",
        api_base="https://default.example/v1",
        api_key="unit-test-secret",
    )

    res = client.post(f"/api/rooms/{room_id}/seats/1/model_config", json={
        "provider": provider,
        "model": "seat-model",
    }, headers=_admin_headers(admin_token))

    assert res.status_code == 400
    assert "api_key" in res.json()["detail"]


@pytest.mark.parametrize("provider", ["openai_responses", "anthropic"])
def test_set_seat_config_accepts_provider_change_with_explicit_key(client, provider):
    room_id, admin_token, _ = _create_room(client)
    room = client.app.state.room_manager.get_room(room_id)
    room.default_config = server_module.ModelConfig(
        provider="openai",
        model="default-model",
        api_base="https://default.example/v1",
        api_key="unit-test-secret",
    )

    res = client.post(f"/api/rooms/{room_id}/seats/1/model_config", json={
        "provider": provider,
        "model": "seat-model",
        "api_base": "https://seat.example/v1",
        "api_key": "seat-key",
    }, headers=_admin_headers(admin_token))

    assert res.status_code == 200
    cfg = room.seat_configs[1]
    assert cfg.provider == provider
    assert cfg.api_base == "https://seat.example/v1"
    assert cfg.api_key == "seat-key"


def test_start_room_requires_admin_token(client):
    room_id, admin_token, _ = _create_room(client)

    missing = client.post(f"/api/rooms/{room_id}/start")
    assert missing.status_code == 403

    wrong = client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers("wrong"))
    assert wrong.status_code == 403

    ok = client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))
    assert ok.status_code == 200


def test_websocket_snapshot(client):
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [],
    })
    room_id = res.json()["room_id"]
    with client.websocket_connect(f"/ws/{room_id}?mode=spectate") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"
        assert msg["status"] == "waiting"
        assert "view" in msg
        assert len(msg["view"]["players"]) == 6


def test_websocket_god_mode_sees_roles(client):
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))
    with client.websocket_connect(f"/ws/{room_id}?mode=god&token={admin_token}") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"
        # god 模式应看到完整角色
        full = msg["view"].get("players_full", [])
        assert len(full) == 6
        assert all(p.get("role") for p in full)


def test_replay_endpoint_rejects_running_game(client):
    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))
    res2 = client.get(f"/api/rooms/{room_id}/replay", headers=_admin_headers(admin_token))
    assert res2.status_code == 409


def test_replay_endpoint_exposes_latest_analysis(client):
    room_id, admin_token, _ = _create_room(client)
    # TestClient fixture exposes the in-memory manager through the app state by closure.
    # Use the public create/start path, then append the same event shape WS replay uses.
    app = client.app
    manager = app.state.room_manager
    room = manager.get_room(room_id)
    room.status = "ended"
    room.event_history.extend([
        {"type": "analysis", "analysis": {"winner": "village", "days": 2}},
        {"type": "room_status", "status": "ended"},
        {"type": "analysis", "analysis": {"winner": "werewolves", "days": 3}},
    ])

    res2 = client.get(f"/api/rooms/{room_id}/replay", headers=_admin_headers(admin_token))
    assert res2.status_code == 200
    body = res2.json()
    assert body["analysis"] == {"winner": "werewolves", "days": 3}
    assert body["events"][-1]["type"] == "analysis"
