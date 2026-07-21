"""FastAPI REST + WebSocket 集成测试。

避免真实 LLM 调用:只验证房间生命周期、信息隔离广播骨架、
WebSocket 快照与事件流能正常工作。
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import src.api.server as server_module
from src.api.server import create_app
from src.config import parse_cors_origins
from src.game.models import Phase

TEST_MODEL_CONFIG = {
    "provider": "openai",
    "model": "unit-test-model",
    "api_base": "https://example.invalid/v1",
    "api_key": "unit-test-key",
    "temperature": 0.85,
    "max_tokens": 0,
    "use_json_format": False,
}


@pytest.fixture
def client():
    from src.api.room_manager import RoomManager

    async def _noop_run_room(room):
        # 不启动真实编排器,避免 LLM 调用;deal_roles 已在 start_game 中完成
        return

    manager = RoomManager()
    manager._run_room = _noop_run_room  # type: ignore[method-assign]
    app = create_app(manager=manager)
    with patch("src.api.server.DEFAULT_MODEL_CONFIG", TEST_MODEL_CONFIG):
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


def test_health_and_readiness_are_bounded_and_distinct(client):
    health = client.get("/healthz")
    ready = client.get("/readyz")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "checks": {"room_manager": "ready", "router": "ready"},
    }
    serialized = json.dumps({"health": health.json(), "ready": ready.json()})
    assert "api_key" not in serialized
    assert "api_base" not in serialized
    assert "room_id" not in serialized


def test_readiness_fails_while_closing_but_liveness_stays_ok(client):
    manager = client.app.state.room_manager
    manager._closing = True
    try:
        ready = client.get("/readyz")
        health = client.get("/healthz")
    finally:
        manager._closing = False

    assert ready.status_code == 503
    assert ready.json() == {
        "status": "not_ready",
        "checks": {"room_manager": "closing", "router": "ready"},
    }
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}


def test_cors_preflight_allows_exact_local_origin_only(client):
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,x-room-token",
    }
    allowed = client.options("/api/rooms", headers=headers)
    denied = client.options(
        "/api/rooms",
        headers={**headers, "Origin": "https://attacker.invalid"},
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-credentials" not in allowed.headers
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


@pytest.mark.parametrize(
    "value",
    [
        "*",
        "https://*.example.com",
        "ftp://example.com",
        "https://user:password@example.com",
        "https://example.com/path",
        "https://example.com/",
        "https://example.com?tenant=1",
        "https://example.com,",
        "",
    ],
)
def test_cors_origin_parser_fails_closed(value):
    with pytest.raises(ValueError):
        parse_cors_origins(value)


def test_cors_origin_parser_normalizes_and_deduplicates_exact_origins():
    assert parse_cors_origins(
        "HTTP://LOCALHOST:5173, https://example.com:8443,HTTP://LOCALHOST:5173"
    ) == ("http://localhost:5173", "https://example.com:8443")


def test_create_room_capacity_maps_to_429_without_evicting_existing_room():
    from src.api.room_manager import RoomManager

    manager = RoomManager(max_rooms=1, terminal_room_ttl=3600)
    app = create_app(manager=manager)
    payload = {"player_names": ["A", "B", "C", "D", "E", "F"]}
    with patch("src.api.server.DEFAULT_MODEL_CONFIG", TEST_MODEL_CONFIG):
        with TestClient(app) as isolated_client:
            first = isolated_client.post("/api/rooms", json=payload)
            second = isolated_client.post("/api/rooms", json=payload)

    assert first.status_code == 200
    assert second.status_code == 429
    assert "容量" in second.json()["detail"]
    assert list(manager.rooms) == [first.json()["room_id"]]


def test_create_room_while_manager_is_closing_maps_to_503(client):
    manager = client.app.state.room_manager
    manager._closing = True
    try:
        response = client.post("/api/rooms", json={
            "player_names": ["A", "B", "C", "D", "E", "F"],
        })
    finally:
        manager._closing = False

    assert response.status_code == 503
    assert response.json() == {"detail": "服务暂不接受新房间"}


def test_explicit_room_cleanup_requires_admin_and_removes_idle_room(client):
    room_id, admin_token, _ = _create_room(client)

    missing = client.delete(f"/api/rooms/{room_id}")
    wrong = client.delete(
        f"/api/rooms/{room_id}",
        headers=_admin_headers("wrong-token"),
    )
    removed = client.delete(
        f"/api/rooms/{room_id}",
        headers=_admin_headers(admin_token),
    )

    assert missing.status_code == 403
    assert wrong.status_code == 403
    assert removed.status_code == 200
    assert removed.json() == {"room_id": room_id, "status": "deleted"}
    assert client.get(f"/api/rooms/{room_id}").status_code == 404


def test_explicit_room_cleanup_refuses_running_room(client):
    room_id, admin_token, _ = _create_room(client)
    started = client.post(
        f"/api/rooms/{room_id}/start",
        headers=_admin_headers(admin_token),
    )

    removed = client.delete(
        f"/api/rooms/{room_id}",
        headers=_admin_headers(admin_token),
    )

    assert started.status_code == 200
    assert removed.status_code == 409
    assert client.app.state.room_manager.get_room(room_id) is not None


def test_capability_rotation_and_seat_revocation_are_enforced_by_public_api(client):
    room_id, admin_token, seat_tokens = _create_room(client, human_seats=[1])
    old_seat_token = seat_tokens["1"]

    rotated_seat = client.post(
        f"/api/rooms/{room_id}/tokens/rotate",
        json={"seat": 1},
        headers=_admin_headers(admin_token),
    )
    assert rotated_seat.status_code == 200
    new_seat_token = rotated_seat.json()["token"]
    assert new_seat_token and new_seat_token != old_seat_token
    assert rotated_seat.json()["version"] == 2

    with pytest.raises(WebSocketDisconnect) as stale_seat:
        with client.websocket_connect(
            f"/ws/{room_id}?mode=play&seat=1&token={old_seat_token}"
        ):
            pass
    assert getattr(stale_seat.value, "code", None) == 4403
    with client.websocket_connect(
        f"/ws/{room_id}?mode=play&seat=1&token={new_seat_token}"
    ) as websocket:
        assert websocket.receive_json()["type"] == "snapshot"

    revoked = client.post(
        f"/api/rooms/{room_id}/tokens/revoke",
        json={"seat": 1},
        headers=_admin_headers(admin_token),
    )
    assert revoked.status_code == 200
    assert revoked.json()["status"] == "revoked"
    with pytest.raises(WebSocketDisconnect) as revoked_seat:
        with client.websocket_connect(
            f"/ws/{room_id}?mode=play&seat=1&token={new_seat_token}"
        ):
            pass
    assert getattr(revoked_seat.value, "code", None) == 4403

    rotated_admin = client.post(
        f"/api/rooms/{room_id}/tokens/rotate",
        headers=_admin_headers(admin_token),
    )
    assert rotated_admin.status_code == 200
    new_admin_token = rotated_admin.json()["token"]
    assert new_admin_token and new_admin_token != admin_token
    assert client.get(
        f"/api/rooms/{room_id}/trace",
        headers=_admin_headers(admin_token),
    ).status_code == 403
    assert client.get(
        f"/api/rooms/{room_id}/trace",
        headers=_admin_headers(new_admin_token),
    ).status_code == 200


def test_websocket_capability_can_use_subprotocol_without_query_token(client):
    room_id, admin_token, _ = _create_room(client)
    with client.websocket_connect(
        f"/ws/{room_id}?mode=god",
        subprotocols=[
            server_module.WS_PROTOCOL,
            f"{server_module.WS_CAPABILITY_PROTOCOL_PREFIX}{admin_token}",
        ],
    ) as websocket:
        assert websocket.accepted_subprotocol == server_module.WS_PROTOCOL
        assert websocket.receive_json()["type"] == "snapshot"


def test_capability_issuance_responses_are_not_cacheable(client):
    room_id, admin_token, _ = _create_room(client, human_seats=[1])
    created = client.post(
        "/api/rooms",
        json={"player_names": ["A", "B", "C", "D", "E", "F"]},
    )
    assert created.status_code == 200
    rotated_seat = client.post(
        f"/api/rooms/{room_id}/tokens/rotate",
        json={"seat": 1},
        headers=_admin_headers(admin_token),
    )
    rotated_admin = client.post(
        f"/api/rooms/{room_id}/tokens/rotate",
        headers=_admin_headers(admin_token),
    )
    current_admin = rotated_admin.json()["token"]
    revoked = client.post(
        f"/api/rooms/{room_id}/tokens/revoke",
        json={"seat": 1},
        headers=_admin_headers(current_admin),
    )
    for response in (created, rotated_seat, rotated_admin, revoked):
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["pragma"] == "no-cache"
        assert response.headers["referrer-policy"] == "no-referrer"


def test_capability_issuance_and_rotation_do_not_log_plaintext(client, caplog):
    caplog.set_level(20, logger=server_module.__name__)
    created = client.post(
        "/api/rooms",
        json={
            "player_names": ["A", "B", "C", "D", "E", "F"],
            "human_seats": [1],
        },
    )
    assert created.status_code == 200
    body = created.json()
    rotated = client.post(
        f"/api/rooms/{body['room_id']}/tokens/rotate",
        headers=_admin_headers(body["admin_token"]),
    )
    assert rotated.status_code == 200

    server_logs = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == server_module.__name__
    )
    for capability in (
        body["admin_token"],
        body["seat_tokens"]["1"],
        rotated.json()["token"],
    ):
        assert capability not in server_logs


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


def test_config_sanitizes_api_base_url(client):
    with patch("src.api.server.DEFAULT_MODEL_CONFIG", {
        "provider": "openai",
        "model": "model-a",
        "api_base": "https://user:pass@example.invalid:8443/v1/chat/completions?api_key=leak#frag",
        "api_key": "unit-test-secret",
        "temperature": 0.85,
        "max_tokens": 0,
        "use_json_format": False,
    }):
        res = client.get("/api/config")

    assert res.status_code == 200
    serialized = json.dumps(res.json())
    assert res.json()["api_base"] == "https://example.invalid:8443/v1/chat/completions"
    assert "user" not in serialized
    assert "pass" not in serialized
    assert "api_key=leak" not in serialized
    assert "chat/completions" in serialized
    assert "frag" not in serialized


def test_config_recursively_redacts_nested_secret_shaped_values(client):
    nested_secret = "sk-public-config-nested-secret-123456789"
    with patch("src.api.server.DEFAULT_MODEL_CONFIG", {
        "provider": "openai_responses",
        "model": "model-a",
        "api_base": "https://user:pass@example.invalid/v1?api_key=leak",
        "api_key": "configured-key",
        "temperature": 0.4,
        "max_tokens": 0,
        "use_json_format": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "decision",
                "description": f"Bearer {nested_secret}",
                "schema": {
                    "type": "object",
                    "properties": {
                        "authorization": {"const": nested_secret},
                        "note": {"description": nested_secret},
                    },
                },
            },
        },
        "reasoning": {
            "summary": f"https://example.invalid/signed/{nested_secret}?token=leak",
            "api_key": nested_secret,
        },
        "thinking": {"secret": nested_secret},
    }):
        res = client.get("/api/config")

    assert res.status_code == 200
    cfg = res.json()
    serialized = json.dumps(cfg)
    assert cfg["api_key"] == ""
    assert cfg["api_key_configured"] is True
    assert cfg["api_base"] == "https://example.invalid/v1"
    assert nested_secret not in serialized
    assert "user:pass" not in serialized
    assert "api_key=leak" not in serialized
    assert cfg["reasoning"]["api_key"] == "[redacted]"
    assert cfg["thinking"]["secret"] == "[redacted]"


def test_invalid_room_config_does_not_echo_or_log_api_key(client, caplog):
    secret = "sk-test-invalid-config-secret-123456789"

    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "model_config": {
            "provider": "openai",
            "model": "model-a",
            "api_base": "https://example.invalid/v1",
            "api_key": secret,
            "temperature": "not-a-number",
        },
    })

    assert res.status_code == 400
    assert secret not in res.text
    assert secret not in caplog.text


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


def test_create_room_clears_inherited_key_when_api_base_changes_without_explicit_key(client):
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

    assert res.status_code == 200
    room = client.app.state.room_manager.get_room(res.json()["room_id"])
    assert room.default_config.model == "attacker-model"
    assert room.default_config.api_base == "https://attacker.invalid/openai/v1"
    assert room.default_config.api_key == ""


@pytest.mark.parametrize("provider", ["openai_responses", "anthropic"])
def test_create_room_clears_inherited_key_when_provider_changes_without_explicit_key(client, provider):
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

    assert res.status_code == 200
    room = client.app.state.room_manager.get_room(res.json()["room_id"])
    assert room.default_config.provider == provider
    assert room.default_config.model == "other-model"
    assert room.default_config.api_key == ""


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


def test_partial_frontend_dist_does_not_break_api_startup(monkeypatch, tmp_path):
    """A build/deploy window with only index.html remains an explicit setup page."""
    from src.api.room_manager import RoomManager

    partial_dist = tmp_path / "frontend" / "dist"
    partial_dist.mkdir(parents=True)
    (partial_dist / "index.html").write_text("PARTIAL_INDEX", encoding="utf-8")
    monkeypatch.setattr(server_module, "FRONTEND_DIR", partial_dist)

    app = create_app(manager=RoomManager())
    with TestClient(app) as c:
        res = c.get("/")

    assert res.status_code == 200
    assert res.text == "PARTIAL_INDEX"


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
    terminal = client.get(f"/api/rooms/{room_id}")
    assert terminal.status_code == 200
    assert all("role" not in p and "team" not in p for p in terminal.json()["players"])
    replay = client.get(f"/api/rooms/{room_id}/replay", headers=_admin_headers(admin_token))
    assert replay.status_code == 200
    assert replay.json()["human_seats"] == [2, 5]
    assert all(player.get("role") for player in replay.json()["players"])


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


@pytest.mark.parametrize("field_name", ["extra_body", "reasoning_effort", "top_k"])
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
    [("top_k", None), ("reasoning_effort", "")],
)
def test_create_room_rejects_empty_non_standard_model_config_fields(client, field_name, value):
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [],
        "model_config": {field_name: value},
    })

    assert res.status_code == 400
    assert field_name in res.json()["detail"]


def test_create_room_accepts_standard_reasoning_and_thinking_fields(client):
    res = client.post("/api/rooms", json={
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "human_seats": [],
        "model_config": {
            "provider": "anthropic",
            "model": "test-model",
            "api_base": "https://example.invalid",
            "api_key": "explicit-key",
            "reasoning": {"effort": "high", "summary": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        },
    })

    assert res.status_code == 200
    room = client.app.state.room_manager.get_room(res.json()["room_id"])
    assert room.default_config.reasoning == {"effort": "high", "summary": "auto"}
    assert room.default_config.thinking == {"type": "enabled", "budget_tokens": 2048}


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


@pytest.mark.parametrize("field_name", ["extra_body", "reasoning_effort", "top_k"])
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
    [("top_k", None), ("reasoning_effort", "")],
)
def test_set_seat_config_rejects_empty_non_standard_model_config_fields(client, field_name, value):
    room_id, admin_token, _ = _create_room(client)
    res = client.post(f"/api/rooms/{room_id}/seats/1/model_config", json={
        "model": "seat-model",
        field_name: value,
    }, headers=_admin_headers(admin_token))

    assert res.status_code == 400
    assert field_name in res.json()["detail"]


def test_set_seat_config_accepts_standard_reasoning_and_thinking_fields(client):
    room_id, admin_token, _ = _create_room(client)
    res = client.post(f"/api/rooms/{room_id}/seats/1/model_config", json={
        "provider": "anthropic",
        "model": "seat-model",
        "reasoning": {"effort": "high", "summary": "auto"},
        "thinking": {"type": "enabled", "budget_tokens": 2048},
    }, headers=_admin_headers(admin_token))

    assert res.status_code == 200
    room = client.app.state.room_manager.get_room(room_id)
    cfg = room.seat_configs[1]
    assert cfg.reasoning == {"effort": "high", "summary": "auto"}
    assert cfg.thinking == {"type": "enabled", "budget_tokens": 2048}


def test_set_seat_config_allows_api_base_change_without_explicit_key(client):
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

    assert res.status_code == 200
    cfg = room.seat_configs[1]
    assert cfg.model == "seat-model"
    assert cfg.api_base == "https://attacker.invalid/v1"
    assert cfg.api_key == ""


@pytest.mark.parametrize("provider", ["openai_responses", "anthropic"])
def test_set_seat_config_allows_provider_change_without_explicit_key(client, provider):
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

    assert res.status_code == 200
    cfg = room.seat_configs[1]
    assert cfg.provider == provider
    assert cfg.model == "seat-model"
    assert cfg.api_key == ""


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


def test_set_seat_config_rejects_unknown_seat(client):
    room_id, admin_token, _ = _create_room(client)

    res = client.post(f"/api/rooms/{room_id}/seats/99/model_config", json={
        "model": "seat-model",
    }, headers=_admin_headers(admin_token))

    assert res.status_code == 400
    assert "座位不存在: 99" in res.json()["detail"]


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


@pytest.mark.parametrize(
    "terminal_status",
    ["ended", "incomplete", "failed", "timeout", "cancelled", "interrupted"],
)
def test_replay_endpoint_accepts_every_terminal_room_status(client, terminal_status):
    room_id, admin_token, _ = _create_room(client)
    room = client.app.state.room_manager.get_room(room_id)
    assert room is not None
    room.status = terminal_status
    room.end_reason = "test_terminal"

    replay = client.get(
        f"/api/rooms/{room_id}/replay",
        headers=_admin_headers(admin_token),
    )

    assert replay.status_code == 200
    assert replay.json()["status"] == terminal_status
    assert replay.json()["end_reason"] == "test_terminal"


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


def test_trace_and_replay_endpoints_require_admin_token_and_redact_structured_secrets(client):
    room_id, admin_token, seat_tokens = _create_room(client, human_seats=[1])
    _, other_room_admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))
    manager = client.app.state.room_manager
    room = manager.get_room(room_id)
    assert room is not None
    room.status = "ended"
    room.end_reason = "completed"
    credential = "unit-test-secret-should-not-leak"
    access_token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiJ9.signaturevalue123456"
    room.event_history.append({
        "type": "analysis",
        "message": f"provider returned {credential} at https://user:{credential}@example.invalid/v1?api_key={credential}",
        "analysis": {
            "winner": "village",
            "api_key": credential,
            "nested": {
                "authorization": f"Bearer {credential}",
                "access_token": access_token,
            },
        },
    })
    room.decision_trace.append({
        "type": "decision_consumed",
        "seat": 2,
        "phase": "day",
        "admin_token": credential,
        "reason": f"see https://user:{credential}@trace.invalid/private?token={credential}",
        "envelope": {
            "decision": {
                "reasoning": f"模型自报私有推理。Bearer {credential} JWT={access_token}",
            },
            "seat_token": credential,
        },
    })

    for endpoint in ("trace", "replay"):
        missing = client.get(f"/api/rooms/{room_id}/{endpoint}")
        wrong = client.get(f"/api/rooms/{room_id}/{endpoint}", headers=_admin_headers("wrong-token"))
        seat = client.get(
            f"/api/rooms/{room_id}/{endpoint}",
            headers=_admin_headers(seat_tokens["1"]),
        )
        other_room = client.get(
            f"/api/rooms/{room_id}/{endpoint}",
            headers=_admin_headers(other_room_admin_token),
        )
        assert missing.status_code == 403
        assert wrong.status_code == 403
        assert seat.status_code == 403
        assert other_room.status_code == 403

    trace = client.get(f"/api/rooms/{room_id}/trace", headers=_admin_headers(admin_token))
    replay = client.get(f"/api/rooms/{room_id}/replay", headers=_admin_headers(admin_token))

    assert trace.status_code == 200
    assert replay.status_code == 200
    for protected in (trace, replay):
        assert protected.headers["cache-control"] == "no-store"
        assert protected.headers["pragma"] == "no-cache"
        assert protected.headers["referrer-policy"] == "no-referrer"
        assert protected.headers["vary"].lower() == server_module.ROOM_TOKEN_HEADER.lower()
    replay_body = replay.json()
    assert replay_body["players"][0]["role"] is not None
    assert "thinking" not in replay_body
    decision_item = next(item for item in trace.json()["trace"] if item["kind"] == "decision" and item["payload"].get("envelope"))
    assert decision_item["payload"]["envelope"]["decision"]["reasoning"] == "模型自报私有推理。Bearer [redacted] JWT=[redacted]"
    assert decision_item["payload"]["envelope"]["seat_token"] == "[redacted]"
    assert replay_body["analysis"]["api_key"] == "[redacted]"
    assert replay_body["analysis"]["nested"]["authorization"] == "[redacted]"
    assert replay_body["analysis"]["nested"]["access_token"] == "[redacted]"
    assert replay_body["run_spec"]["run_id"] == room_id
    assert replay_body["core_run_spec"]["run_id"] == room_id
    assert replay_body["core_run_spec"]["environment"] == {
        "id": "werewolf.classic",
        "version": "1",
    }
    assert replay_body["transcript"]["run_id"] == room_id
    for removed in ("social_spec", "social_spec_issues", "interaction_graph", "social_metrics"):
        assert removed not in replay_body
        assert removed not in trace.json()
    serialized = json.dumps({"trace": trace.json(), "replay": replay_body}, ensure_ascii=False)
    assert credential not in serialized
    assert access_token not in serialized
    assert admin_token not in serialized


def test_trace_endpoint_includes_decision_trace_without_secrets(client):
    room_id, admin_token, _ = _create_room(client)
    manager = client.app.state.room_manager
    room = manager.get_room(room_id)
    secret = "unit-test-secret-should-not-leak"
    room.decision_trace.append({
        "_trace_seq": 7,
        "_ts": 123.0,
        "type": "decision_consumed",
        "day": 1,
        "phase": "voting",
        "seat": 2,
        "action": "vote",
        "llm_call": {
            "call_id": "llm-abc",
            "protocol": "openai",
            "provider": "openai",
            "model": "model-a",
            "base_fingerprint": "fp-base",
            "request_hash": "req-hash",
            "response_hash": "resp-hash",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
        "_internal_runtime_field": secret,
    })

    res = client.get(f"/api/rooms/{room_id}/trace", headers=_admin_headers(admin_token))

    assert res.status_code == 200
    body = res.json()
    assert body["decision_trace_count"] == 1
    decision_items = [item for item in body["trace"] if item["kind"] == "decision"]
    assert len(decision_items) == 1
    payload = decision_items[0]["payload"]
    assert payload["type"] == "decision_consumed"
    assert payload["llm_call"]["call_id"] == "llm-abc"
    serialized = json.dumps(body)
    assert secret not in serialized
    assert "api_key" not in serialized


def test_trace_endpoint_includes_unified_harness_transcript(client):
    import asyncio

    room_id, admin_token, _ = _create_room(client)
    client.post(f"/api/rooms/{room_id}/start", headers=_admin_headers(admin_token))
    room = client.app.state.room_manager.get_room(room_id)
    assert room is not None and room.orchestrator is not None
    assert room.run_spec is not None
    if room.run_spec.default_model is not None:
        room.run_spec.default_model.api_base = "https://user:unit-test-secret-should-not-leak@example.invalid/private/v1?api_key=unit-test-secret-should-not-leak"

    asyncio.run(room.orchestrator.on_event({
        "type": "phase_started",
        "phase": "day",
        "day": 1,
        "message": "天亮了",
        "provider_error": "failed at https://user:pass@example.invalid/private/tenant-42?signature=do-not-leak",
        "api_key": "unit-test-secret-should-not-leak",
    }))

    res = client.get(f"/api/rooms/{room_id}/trace", headers=_admin_headers(admin_token))

    assert res.status_code == 200
    body = res.json()
    transcript = body["transcript"]
    assert transcript["run_id"] == room_id
    assert transcript["counts_by_kind"]["event"] >= 1
    assert transcript["entries"][-1]["payload"]["api_key"] == "[redacted]"
    assert body["run_spec"]["run_id"] == room_id
    assert body["run_spec"]["environment_id"] == "werewolf.classic"
    assert body["core_run_spec"]["run_id"] == room_id
    assert body["core_run_spec"]["actors"]["human_actor_ids"] == []
    for removed in ("social_spec", "social_spec_issues", "interaction_graph", "social_metrics"):
        assert removed not in body
    assert "unit-test-secret-should-not-leak" not in json.dumps(body)
    assert body["run_spec"]["default_model"]["api_base"] == "https://example.invalid/private/v1"
    serialized = json.dumps(body)
    assert "https://example.invalid" in serialized
    assert "tenant-42" not in serialized
    assert "signature=do-not-leak" not in serialized
    assert "user:pass" not in serialized


def test_trace_endpoint_supports_lightweight_redacted_incremental_reads(client):
    room_id, admin_token, _ = _create_room(client)
    started = client.post(
        f"/api/rooms/{room_id}/start",
        headers=_admin_headers(admin_token),
    )
    assert started.status_code == 200

    manager = client.app.state.room_manager
    room = manager.get_room(room_id)
    assert room is not None and room.run_spec is not None and room.transcript is not None
    since = room.trace_seq
    secret = "incremental-secret-should-not-leak"

    manager._store_room_event(room, {  # noqa: SLF001 - endpoint contract fixture
        "type": "phase_started",
        "day": 1,
        "api_key": secret,
    })
    manager._make_trace_recorder(room)({  # noqa: SLF001 - endpoint contract fixture
        "kind": "agent_request",
        "request": {"request_id": "req-incremental"},
        "authorization": f"Bearer {secret}",
    })
    manager._store_room_event(room, {  # noqa: SLF001 - endpoint contract fixture
        "type": "speech",
        "seat": 2,
        "text": "same public delivery may be repeated",
    })

    full = client.get(
        f"/api/rooms/{room_id}/trace",
        headers=_admin_headers(admin_token),
    )
    incremental = client.get(
        f"/api/rooms/{room_id}/trace?since={since}",
        headers=_admin_headers(admin_token),
    )

    assert full.status_code == 200
    assert incremental.status_code == 200
    full_body = full.json()
    body = incremental.json()
    assert full_body["incremental"] is False
    assert full_body["since"] is None
    assert full_body["trace_seq"] == room.trace_seq
    assert full_body["run_spec"] is not None
    assert full_body["core_run_spec"] is not None
    assert full_body["transcript"] is not None
    assert body["incremental"] is True
    assert body["since"] == since
    assert body["trace_seq"] == room.trace_seq
    assert body["run_spec"] is None
    assert body["core_run_spec"] is None
    assert body["transcript"] is None

    sequences = [item["trace_seq"] for item in body["trace"]]
    assert sequences == sorted(sequences)
    assert sequences == list(range(since + 1, room.trace_seq + 1))
    assert [item["kind"] for item in body["trace"]] == ["event", "decision", "event"]
    full_increment = {
        item["trace_seq"]: item["payload"]
        for item in full_body["trace"]
        if item["trace_seq"] > since
    }
    assert {item["trace_seq"]: item["payload"] for item in body["trace"]} == full_increment
    serialized = json.dumps(body)
    assert secret not in serialized
    assert admin_token not in serialized


def test_trace_endpoint_validates_incremental_cursor_and_reports_rewind(client):
    room_id, admin_token, _ = _create_room(client)
    headers = _admin_headers(admin_token)

    zero = client.get(f"/api/rooms/{room_id}/trace?since=0", headers=headers)
    assert zero.status_code == 200
    assert zero.json()["since"] == 0
    assert zero.json()["incremental"] is True

    for invalid in ("-1", "01", "1.5", "+1", "1e3", "not-a-cursor", "9" * 21):
        response = client.get(
            f"/api/rooms/{room_id}/trace",
            params={"since": invalid},
            headers=headers,
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "invalid trace cursor"

    room = client.app.state.room_manager.get_room(room_id)
    future_cursor = room.trace_seq + 10
    rewind = client.get(
        f"/api/rooms/{room_id}/trace?since={future_cursor}",
        headers=headers,
    )
    assert rewind.status_code == 200
    assert rewind.json()["since"] == future_cursor
    assert rewind.json()["trace_seq"] < future_cursor
    assert rewind.json()["trace"] == []
    assert rewind.json()["run_spec"] is None
    assert rewind.json()["core_run_spec"] is None
    assert rewind.json()["transcript"] is None
