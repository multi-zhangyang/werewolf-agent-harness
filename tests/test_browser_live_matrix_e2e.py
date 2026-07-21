"""Opt-in browser journeys across the live REST/WS/React boundary.

The fixture exposes one human seat and marks all internal actors human, so it
never starts a provider loop. The browser still uses the real FastAPI
REST/WebSocket endpoints and the production React build; no browser network
routes are installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from src.api.room_manager import Room, RoomClient, RoomManager
from src.api import server as api_server
from src.game.models import Phase
from src.game.orchestrator import build_actors
from src.game.roles import Team, default_role_deck
from src.game.rules import RulesEngine
from src.llm.models import ModelConfig


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MARKER = "PUBLIC_MATRIX_EVENT"
PRIVATE_MARKER = "PRIVATE_MATRIX_EVENT"
GOD_TRACE_MARKER = "GOD_MATRIX_PRIVATE_REASONING"


class _LiveMatrixFixtureManager(RoomManager):
    """A real RoomManager with a provider-free live room."""

    def __init__(self) -> None:
        super().__init__(terminal_room_ttl=3600)
        self.accepted_actions: list[dict[str, Any]] = []
        self.capability_query_leaks: dict[str, bool] = {}
        self._action_lock = threading.RLock()

    async def _run_room(self, room: Room) -> None:
        return

    def create_room(self, **kwargs: Any) -> Room:
        room = super().create_room(**kwargs)
        RulesEngine.deal_roles(
            room.state,
            deck=default_role_deck(len(room.state.players)),
            seed=room.role_seed,
        )
        room.state.phase = Phase.DAY
        room.state.day = 1
        room.default_config = ModelConfig(
            provider="openai",
            model="browser-matrix-fixture",
            api_base="https://example.invalid/v1",
            api_key="",
        )
        # Keep all internal actors human even though only seat 1 is advertised
        # to the browser. This makes an accidental fixture task provider-free.
        room.actors = build_actors(
            room.state,
            model_config=room.default_config,
            router=self.router,
            human_seats={player.seat for player in room.state.players},
            rng=random.Random(room.actor_seed),
            budget_scope=self.provider_budget_ledger.room_scope(room.id),
        )
        for actor in room.actors.values():
            actor.persona_name = f"matrix-persona-seat-{actor.seat}"
        room.status = "running"
        room.end_reason = None
        room.error = None
        room.terminal_at = None
        self._install_god_trace(room)
        self._persist_room(room)
        return room

    def _install_god_trace(self, room: Room) -> None:
        record = self._make_trace_recorder(room)
        request_id = "req-browser-matrix-trace"
        record({
            "kind": "agent_request",
            "request": {
                "request_id": request_id,
                "seat": 1,
                "phase": "day",
                "action_kind": "speak",
                "legal_actions": [{
                    "action": "speak",
                    "target_seats": [],
                    "can_skip": True,
                }],
            },
        })
        record({
            "kind": "agent_response",
            "request_id": request_id,
            "envelope": {
                "model_call_id": "browser-matrix-call",
                "prompt_hash": "browser-matrix-prompt",
                "response_hash": "browser-matrix-response",
                "latency_seconds": 0.01,
                "parse_status": "tool_call",
                "decision": {
                    "action": "speak",
                    "speech": "fixture output",
                    "reasoning": GOD_TRACE_MARKER,
                },
            },
            "validation": {"valid": True, "issues": []},
        })

    def publish_visibility_probe(self, room: Room) -> None:
        player = next(item for item in room.state.players if item.seat == 1)

        async def publish() -> None:
            await self._emit_room_event(room, {
                "type": "speech",
                "phase": "day",
                "day": 1,
                "seat": 2,
                "name": room.state.players[1].name,
                "text": PUBLIC_MARKER,
            })
            await self._emit_room_event(room, {
                "type": "speech",
                "phase": "day",
                "day": 1,
                "seat": 1,
                "name": player.name,
                "text": PRIVATE_MARKER,
                "visibility": "private",
                "recipients": [player.id],
            })

        asyncio.run(publish())

    def issue_human_request(
        self,
        room: Room,
        *,
        action: str,
        phase: str,
        allowed_target_seats: list[int],
        requires_target: bool,
        can_skip: bool,
        request_id: str,
        day: int = 1,
    ) -> None:
        player = next(item for item in room.state.players if item.seat == 1)
        actor = room.actors[player.id]
        actor.current_human_request = {
            "request_id": request_id,
            "action_type": action,
            "accepted_actions": [action],
            "requires_target": requires_target,
            "can_skip": can_skip,
            "day": day,
            "phase": phase,
            "allowed_target_seats": list(allowed_target_seats),
        }
        room.state.phase = {
            "day": Phase.DAY,
            "voting": Phase.VOTING,
            "night": Phase.NIGHT,
            "pk": Phase.VOTING,
        }.get(phase, Phase.DAY)
        room.state.day = day
        context = {
            "phase": phase,
            "day": day,
            "requested_action": action,
            "allowed_target_seats": list(allowed_target_seats),
            "requires_target": requires_target,
            "can_skip": can_skip,
            "timeout": 120,
            "timeout_ms": 120_000,
        }

        async def publish() -> None:
            await self._emit_room_event(room, {
                "type": "phase_started",
                "phase": phase,
                "day": day,
                "message": f"fixture phase {phase}",
            })
            await self._emit_room_event(room, {
                "type": "human_action_request",
                "request_id": request_id,
                "seat": 1,
                "action_type": action,
                "context": context,
                "timeout": 120,
                "day": day,
                "phase": phase,
            })

        asyncio.run(publish())

    def supersede_human_request(
        self,
        room: Room,
        *,
        action: str,
        phase: str,
        request_id: str,
    ) -> None:
        player = next(item for item in room.state.players if item.seat == 1)
        room.actors[player.id].current_human_request = {
            "request_id": request_id,
            "action_type": action,
            "accepted_actions": [action],
            "requires_target": True,
            "can_skip": False,
            "day": 1,
            "phase": phase,
            "allowed_target_seats": [2],
        }

    async def handle_human_action(
        self,
        room: Room,
        seat: int,
        action: dict[str, Any],
    ) -> None:
        player = next((item for item in room.state.players if item.seat == seat), None)
        before = room.actors[player.id].human_queue.qsize() if player is not None else 0
        await super().handle_human_action(room, seat, action)
        after = room.actors[player.id].human_queue.qsize() if player is not None else before
        if after > before:
            with self._action_lock:
                self.accepted_actions.append(dict(action))

    async def connect(self, room: Room, websocket: Any, **kwargs: Any) -> str:
        capability = str(kwargs.get("capability_token") or "")
        query = str(websocket.url.query)
        with self._action_lock:
            self.capability_query_leaks[room.id] = bool(capability and capability in query)
        return await super().connect(room, websocket, **kwargs)

    def latest_accepted(self) -> dict[str, Any]:
        with self._action_lock:
            if not self.accepted_actions:
                raise AssertionError("the backend did not consume a human action")
            return dict(self.accepted_actions[-1])

    def accepted_count(self) -> int:
        with self._action_lock:
            return len(self.accepted_actions)

    def active_clients(self, room: Room, *, mode: str, seat: int | None = None) -> int:
        with room.delivery_lock:
            return sum(
                isinstance(connection, RoomClient)
                and connection.mode == mode
                and connection.seat == seat
                for connection in room.clients.values()
            )

    def force_close_client(
        self,
        room: Room,
        *,
        mode: str,
        seat: int | None,
        code: int,
        reason: str,
    ) -> None:
        with room.delivery_lock:
            candidates = [
                connection
                for connection in room.clients.values()
                if isinstance(connection, RoomClient)
                and connection.mode == mode
                and connection.seat == seat
            ]
        if not candidates:
            raise AssertionError(f"no live {mode} connection to close")
        connection = candidates[-1]

        def close_on_owner_loop() -> None:
            self._terminate_client_on_owner_loop(
                room,
                connection,
                code=code,
                reason=reason,
            )

        connection.loop.call_soon_threadsafe(close_on_owner_loop)

    def finish_room(self, room: Room) -> None:
        room.status = "ended"
        room.end_reason = "browser_matrix_terminal"
        room.state.phase = Phase.ENDED
        room.state.winner = Team.VILLAGE
        room.terminal_at = time.monotonic()

        async def publish() -> None:
            await self._emit_room_event(room, {
                "type": "game_ended",
                "phase": "ended",
                "day": room.state.day,
                "winner": "village",
                "message": "fixture game ended",
            })
            await self._emit_room_event(room, {
                "type": "room_status",
                "status": "ended",
                "reason": room.end_reason,
            })
            await self._emit_room_event(room, {
                "type": "analysis",
                "visibility": "admin",
                "analysis": {
                    "winner": "village",
                    "days": room.state.day,
                    "seats": [
                        {
                            "seat": player.seat,
                            "role": str(
                                getattr(player.role, "value", player.role)
                                or "villager"
                            ),
                            "team": (
                                "werewolves"
                                if getattr(player.role, "value", player.role) == "werewolf"
                                else "village"
                            ),
                        }
                        for player in room.state.players
                    ],
                },
            })

        asyncio.run(publish())


def _browser(session: str, *args: str, stdin: str | None = None) -> str:
    env = dict(os.environ)
    # A user-configured HTTP proxy must not intercept the local fixture.
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    completed = subprocess.run(
        ["agent-browser", "--session", session, *args],
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
        env=env,
        timeout=30,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"agent-browser command failed ({' '.join(args)}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _browser_eval(session: str, source: str) -> Any:
    raw = _browser(session, "eval", "--stdin", stdin=f"JSON.stringify({source})")
    decoded = json.loads(raw)
    return json.loads(decoded) if isinstance(decoded, str) else decoded


@contextmanager
def _serve_fixture(manager: RoomManager) -> Iterator[str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    origin = f"http://127.0.0.1:{port}"
    previous_origins = api_server.CORS_ORIGINS
    api_server.CORS_ORIGINS = tuple(dict.fromkeys((*previous_origins, origin)))
    server: uvicorn.Server | None = None
    thread: threading.Thread | None = None
    try:
        server = uvicorn.Server(uvicorn.Config(
            api_server.create_app(manager=manager),
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
        ))
        thread = threading.Thread(target=server.run, name="browser-matrix-server", daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not server.started:
            if not thread.is_alive():
                raise RuntimeError("browser matrix server terminated during startup")
            time.sleep(0.02)
        if not server.started:
            raise RuntimeError("browser matrix server did not become ready")
        yield f"http://127.0.0.1:{port}"
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None and thread.ident is not None:
            thread.join(timeout=10)
        api_server.CORS_ORIGINS = previous_origins
        if thread is not None and thread.is_alive():
            raise RuntimeError("browser matrix server did not stop")


def _require_browser_e2e() -> None:
    if os.environ.get("WEREWOLF_RUN_BROWSER_E2E") != "1":
        pytest.skip("set WEREWOLF_RUN_BROWSER_E2E=1 to run the real browser matrix")
    if shutil.which("agent-browser") is None:
        pytest.fail("agent-browser is required for WEREWOLF_RUN_BROWSER_E2E=1")
    if not (ROOT / "frontend" / "dist" / "index.html").is_file():
        pytest.fail("build frontend/dist before running browser E2E")


def _create_running_room(
    session: str,
    base_url: str,
    manager: _LiveMatrixFixtureManager,
) -> Room:
    _browser(session, "open", f"{base_url}/")
    _browser(session, "wait", "--text", "创建真实对局")
    _browser(session, "find", "role", "button", "click", "--name", "6人")
    _browser(session, "find", "first", 'button[aria-label="AI 座位"]', "click")
    _browser(session, "wait", "--text", "真人 1 席")
    _browser(session, "find", "role", "button", "click", "--name", "创建房间")
    _browser(session, "wait", "--text", "等待室")
    _browser(session, "wait", "--text", "进行中")
    rooms = list(manager.rooms.values())
    if not rooms:
        raise AssertionError("the real REST create-room boundary did not create a room")
    room = rooms[-1]
    if room.human_seats != {1}:
        raise AssertionError("browser fixture did not retain the advertised human seat")
    return room


def _enter_mode(session: str, mode: str) -> None:
    label = {
        "spectate": "观战",
        "play": "1号真人座位",
        "god": "上帝视角",
    }[mode]
    _browser(session, "find", "role", "button", "click", "--name", label)
    _browser(session, "wait", "--text", "Agent Harness Run")
    _browser(session, "wait", "--text", "WS connected")


def _select_visible_tab(session: str, label: str) -> None:
    _browser(
        session,
        "eval",
        "--stdin",
        stdin="""
(() => {
  const visibleTab = [...document.querySelectorAll('[role="tab"]')].find((item) => {
    const style = getComputedStyle(item);
    const rect = item.getBoundingClientRect();
    return item.textContent?.trim() === %s
      && style.display !== 'none' && style.visibility !== 'hidden'
      && rect.width > 0 && rect.height > 0
      && rect.right > 0 && rect.left < innerWidth
      && rect.bottom > 0 && rect.top < innerHeight;
  });
  if (!visibleTab) throw new Error("visible tab was not rendered");
  if (typeof PointerEvent === 'function') {
    visibleTab.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, cancelable: true, composed: true, button: 0}));
    visibleTab.dispatchEvent(new PointerEvent('pointerup', {bubbles: true, cancelable: true, composed: true, button: 0}));
  }
  visibleTab.dispatchEvent(new MouseEvent('mousedown', {bubbles: true, cancelable: true, view: window, button: 0}));
  visibleTab.dispatchEvent(new MouseEvent('mouseup', {bubbles: true, cancelable: true, view: window, button: 0}));
  visibleTab.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, view: window, button: 0}));
  return true;
})()
""" % json.dumps(label),
    )
    _browser(
        session,
        "wait",
        "--fn",
        """
(() => {
  return [...document.querySelectorAll('[role="tab"]')].some((item) => {
    const style = getComputedStyle(item);
    const rect = item.getBoundingClientRect();
    return item.textContent?.trim() === %s
      && item.getAttribute('aria-selected') === 'true'
      && style.display !== 'none' && style.visibility !== 'hidden'
      && rect.width > 0 && rect.height > 0
      && rect.right > 0 && rect.left < innerWidth
      && rect.bottom > 0 && rect.top < innerHeight;
  });
})()
""" % json.dumps(label),
    )


def _click_visible_button(session: str, label: str) -> None:
    """Click the button whose center is actually owned by the viewport UI."""
    _browser(
        session,
        "eval",
        "--stdin",
        stdin="""
(() => {
  const label = %s;
  const button = [...document.querySelectorAll('button')].find((item) => {
    const text = item.innerText.trim();
    const style = getComputedStyle(item);
    const rect = item.getBoundingClientRect();
    if (!text.includes(label) || style.display === 'none' || style.visibility === 'hidden'
      || rect.width <= 0 || rect.height <= 0
      || rect.right <= 0 || rect.left >= innerWidth
      || rect.bottom <= 0 || rect.top >= innerHeight) return false;
    const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
    return hit === item || item.contains(hit);
  });
  if (!button) throw new Error(`visible button not hit-testable: ${label}`);
  button.click();
  return true;
})()
""" % json.dumps(label),
    )


def _select_visible_seats_tab(session: str) -> None:
    _select_visible_tab(session, "座位")


def _visible_seat_labels(session: str, *, mobile: bool) -> list[str]:
    root_selector = '[role="dialog"]' if mobile else "aside"
    return _browser_eval(session, f"""(() => {{
  const root = document.querySelector({json.dumps(root_selector)});
  if (!root) return [];
  return [...root.querySelectorAll('[aria-label]')]
    .filter((item) => /^\\d+号 · /.test(item.getAttribute('aria-label') || ''))
    .map((item) => item.getAttribute('aria-label'));
}})()""")


def _surface(session: str) -> dict[str, Any]:
    return _browser_eval(session, """({
      url: location.href,
      local: Object.fromEntries(Object.entries(localStorage)),
      session: Object.fromEntries(Object.entries(sessionStorage)),
      localKeys: Object.keys(localStorage),
      sessionKeys: Object.keys(sessionStorage),
    })""")


def _assert_capability_boundary(session: str, room: Room) -> None:
    surface = _surface(session)
    if "token=" in str(surface["url"]).lower() or "capability" in str(surface["url"]).lower():
        raise AssertionError("a room capability was placed in the browser URL")
    serialized = json.dumps(surface, ensure_ascii=False)
    for token in (room.admin_token, *room.seat_tokens.values()):
        if token and token in serialized:
            raise AssertionError("a room capability was persisted in browser storage")
    if any("roomauth" in str(key).lower() for key in (*surface["localKeys"], *surface["sessionKeys"])):
        raise AssertionError("legacy roomAuth storage survived the browser journey")


def _assert_ws_capability_not_in_query(manager: _LiveMatrixFixtureManager, room: Room) -> None:
    if room.id not in manager.capability_query_leaks:
        raise AssertionError("the browser did not reach the real WebSocket connect boundary")
    if manager.capability_query_leaks.get(room.id, False):
        raise AssertionError("a room capability was placed in the WebSocket URL")


def _assert_mobile_layout(session: str) -> None:
    layout = _browser_eval(session, """({
      width: innerWidth,
      documentWidth: document.documentElement.scrollWidth,
      bodyWidth: document.body.scrollWidth,
      rootOverflowCount: (() => {
        const root = document.querySelector('#root');
        if (!root) return 0;
        const style = getComputedStyle(root);
        const rect = root.getBoundingClientRect();
        return style.display !== 'none' && style.visibility !== 'hidden'
          && rect.width > 0 && rect.height > 0
          && (rect.left < -1 || rect.right > innerWidth + 1)
          ? 1
          : 0;
      })(),
    })""")
    assert layout["width"] == 390
    assert layout["documentWidth"] <= 390
    assert layout["bodyWidth"] <= 390
    assert layout["rootOverflowCount"] == 0


def _assert_desktop_layout(session: str) -> None:
    layout = _browser_eval(session, """({
      width: innerWidth,
      documentWidth: document.documentElement.scrollWidth,
    })""")
    assert layout["width"] == 1280
    assert layout["documentWidth"] <= 1280


def _leave_game(session: str) -> None:
    _browser(session, "press", "Escape")
    _browser(session, "wait", "--fn", "!document.querySelector('[role=dialog]')")
    _browser(session, "find", "role", "button", "click", "--name", "返回房间")
    _browser(session, "wait", "--text", "等待室")
    _browser(session, "find", "role", "button", "click", "--name", "返回大厅")
    _browser(session, "wait", "--text", "创建真实对局")


def test_real_backend_live_audience_matrix_browser_journey(tmp_path: Path) -> None:
    """Spectator/player/God projections stay isolated on desktop and mobile."""
    _require_browser_e2e()
    manager = _LiveMatrixFixtureManager()
    with _serve_fixture(manager) as base_url:
        session = f"werewolf-browser-audience-{uuid.uuid4().hex[:8]}"
        try:
            for mode in ("spectate", "play", "god"):
                _browser(session, "set", "viewport", "1280", "900")
                room = _create_running_room(session, base_url, manager)
                manager.publish_visibility_probe(room)
                _enter_mode(session, mode)
                _browser(session, "wait", "--text", PUBLIC_MARKER)
                _select_visible_seats_tab(session)
                desktop_labels = _visible_seat_labels(session, mobile=False)
                assert len(desktop_labels) == 6
                body = _browser_eval(session, "document.body.innerText")
                if mode == "spectate":
                    assert all(label.endswith("身份隐藏") for label in desktop_labels)
                    assert PRIVATE_MARKER not in body
                    assert GOD_TRACE_MARKER not in body
                elif mode == "play":
                    assert sum(not label.endswith("身份隐藏") for label in desktop_labels) == 1
                    own_label = next(label for label in desktop_labels if label.startswith("1号 · "))
                    assert not own_label.endswith("身份隐藏")
                    assert PRIVATE_MARKER in body
                    assert GOD_TRACE_MARKER not in body
                else:
                    assert all(not label.endswith("身份隐藏") for label in desktop_labels)
                    _browser(session, "wait", "--text", "admin-only read projection")
                    _browser(
                        session,
                        "find",
                        "text",
                        "request_id=req-browser-matrix-trace",
                        "click",
                        "--exact",
                    )
                    _browser(session, "wait", "--text", GOD_TRACE_MARKER)
                    assert PRIVATE_MARKER in _browser_eval(session, "document.body.innerText")
                assert PUBLIC_MARKER in _browser_eval(session, "document.body.innerText")
                _assert_desktop_layout(session)
                _assert_capability_boundary(session, room)
                _assert_ws_capability_not_in_query(manager, room)
                _browser(
                    session,
                    "screenshot",
                    str(tmp_path / f"audience-{mode}-desktop.png"),
                )

                _browser(session, "set", "viewport", "390", "844")
                _browser(
                    session,
                    "find",
                    "role",
                    "button",
                    "click",
                    "--name",
                    "打开对局信息",
                )
                _select_visible_seats_tab(session)
                mobile_labels = _visible_seat_labels(session, mobile=True)
                assert len(mobile_labels) == 6
                if mode == "spectate":
                    assert all(label.endswith("身份隐藏") for label in mobile_labels)
                elif mode == "play":
                    assert sum(not label.endswith("身份隐藏") for label in mobile_labels) == 1
                else:
                    assert all(not label.endswith("身份隐藏") for label in mobile_labels)
                _assert_mobile_layout(session)
                _assert_capability_boundary(session, room)
                _assert_ws_capability_not_in_query(manager, room)
                _browser(
                    session,
                    "screenshot",
                    str(tmp_path / f"audience-{mode}-mobile.png"),
                )
                _leave_game(session)
                _browser(session, "set", "viewport", "1280", "900")
        finally:
            try:
                _browser(session, "close")
            except Exception:
                pass


def test_real_backend_human_action_schema_and_terminal_lifecycle(tmp_path: Path) -> None:
    """Target, speech, skip, stale, and terminal controls follow server schema."""
    _require_browser_e2e()
    manager = _LiveMatrixFixtureManager()
    with _serve_fixture(manager) as base_url:
        session = f"werewolf-browser-human-{uuid.uuid4().hex[:8]}"
        try:
            _browser(session, "set", "viewport", "1280", "900")
            room = _create_running_room(session, base_url, manager)
            _enter_mode(session, "play")

            target_request = "req-browser-target"
            manager.issue_human_request(
                room,
                action="vote",
                phase="voting",
                allowed_target_seats=[4, 2],
                requires_target=True,
                can_skip=False,
                request_id=target_request,
            )
            _browser(session, "wait", "--text", "轮到你了")
            _select_visible_tab(session, "行动")
            advertised = _browser_eval(session, """[...document.querySelectorAll('button')]
              .filter((item) => {
                const rect = item.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
              })
              .map((item) => item.innerText.trim())
              .map((text) => text.match(/(\\d+)号 · /)?.[1] || null)
              .filter((seat) => seat !== null)""")
            assert [int(seat) for seat in advertised] == [4, 2]
            target_name = room.state.players[3].name
            _click_visible_button(session, f"4号 · {target_name}")
            _browser(session, "find", "role", "button", "click", "--name", "确认")
            _browser(session, "wait", "--text", "暂无真人操作")
            _wait_until(lambda: manager.accepted_count() == 1)
            accepted = manager.latest_accepted()
            assert accepted["request_id"] == target_request
            assert accepted["action"] == "vote"
            assert accepted["target_seat"] == 4

            speech_request = "req-browser-speech"
            manager.issue_human_request(
                room,
                action="speak",
                phase="day",
                allowed_target_seats=[],
                requires_target=False,
                can_skip=True,
                request_id=speech_request,
            )
            _browser(session, "wait", "--text", "action=speak")
            _browser(
                session,
                "find",
                "placeholder",
                "输入本次 action 的公开文本",
                "fill",
                "exact public matrix speech",
            )
            _browser(session, "find", "role", "button", "click", "--name", "2 · 重要")
            _browser(session, "find", "role", "button", "click", "--name", "提交")
            _browser(session, "wait", "--text", "暂无真人操作")
            _wait_until(lambda: manager.accepted_count() == 2)
            accepted = manager.latest_accepted()
            assert accepted["request_id"] == speech_request
            assert accepted["action"] == "speak"
            assert accepted["speech"] == "exact public matrix speech"
            assert accepted["bid"] == 2

            skip_request = "req-browser-skip"
            manager.issue_human_request(
                room,
                action="poison",
                phase="night",
                allowed_target_seats=[3, 2],
                requires_target=True,
                can_skip=True,
                request_id=skip_request,
            )
            _browser(session, "wait", "--text", "使用毒药")
            _browser(session, "find", "role", "button", "click", "--name", "弃权")
            _browser(session, "wait", "--text", "暂无真人操作")
            _wait_until(lambda: manager.accepted_count() == 3)
            accepted = manager.latest_accepted()
            assert accepted["request_id"] == skip_request
            assert accepted["action"] == "skip"
            assert "target_seat" not in accepted
            _assert_desktop_layout(session)
            _assert_capability_boundary(session, room)
            _assert_ws_capability_not_in_query(manager, room)
            _browser(session, "screenshot", str(tmp_path / "human-schema-desktop.png"))

            _browser(session, "set", "viewport", "390", "844")
            stale_request = "req-browser-stale"
            manager.issue_human_request(
                room,
                action="vote",
                phase="voting",
                allowed_target_seats=[2],
                requires_target=True,
                can_skip=False,
                request_id=stale_request,
            )
            _browser(session, "wait", "--text", "行动")
            _browser(
                session,
                "find",
                "role",
                "button",
                "click",
                "--name",
                "打开待处理行动",
            )
            _browser(session, "wait", "--text", "轮到你了")
            manager.supersede_human_request(
                room,
                action="vote",
                phase="voting",
                request_id="req-browser-server-new",
            )
            stale_name = room.state.players[1].name
            _click_visible_button(session, f"2号 · {stale_name}")
            _click_visible_button(session, "确认")
            _browser(session, "wait", "--text", "暂无真人操作")
            _browser(session, "wait", "--text", "操作请求已过期")
            _browser(session, "press", "Escape")
            _browser(session, "wait", "--fn", "!document.querySelector('[role=dialog]')")
            assert manager.accepted_count() == 3

            terminal_request = "req-browser-terminal"
            manager.issue_human_request(
                room,
                action="vote",
                phase="voting",
                allowed_target_seats=[2],
                requires_target=True,
                can_skip=False,
                request_id=terminal_request,
            )
            _browser(session, "wait", "--text", "行动")
            _browser(
                session,
                "find",
                "role",
                "button",
                "click",
                "--name",
                "打开待处理行动",
            )
            _browser(session, "wait", "--text", "轮到你了")
            manager.finish_room(room)
            _browser(session, "wait", "--text", "暂无真人操作")
            _browser(session, "press", "Escape")
            _browser(session, "wait", "--text", "run completed · winner=village")
            _browser(session, "wait", "--fn", "!document.querySelector('[role=dialog]')")
            terminal_controls = _browser_eval(session, """[...document.querySelectorAll('button')]
              .map((item) => item.innerText.trim())
              .filter((text) => ['确认', '弃权', '提交', '明确弃权'].includes(text))""")
            assert terminal_controls == []
            _assert_mobile_layout(session)
            _assert_capability_boundary(session, room)
            _assert_ws_capability_not_in_query(manager, room)
            _browser(session, "screenshot", str(tmp_path / "human-terminal-mobile.png"))

            _browser(session, "find", "role", "button", "click", "--name", "返回房间")
            _browser(session, "wait", "--text", "复盘")
            _browser(session, "network", "requests", "--clear")
            _browser(session, "find", "role", "button", "click", "--name", "复盘")
            _browser(session, "wait", "--text", "Environment timeline")
            _browser(session, "wait", "--text", "Factual analysis")
            _assert_capability_boundary(session, room)
            replay_network = json.loads(
                _browser(session, "network", "requests", "--filter", "/ws/", "--json")
            )
            assert replay_network["success"] is True
            assert replay_network["data"]["requests"] == []
            _assert_mobile_layout(session)
            _browser(
                session,
                "screenshot",
                str(tmp_path / "terminal-to-replay-mobile.png"),
            )
        finally:
            try:
                _browser(session, "close")
            except Exception:
                pass


def _wait_until(predicate: Any, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition did not become true before browser matrix timeout")


def test_real_backend_retryable_ws_close_requires_manual_retry(tmp_path: Path) -> None:
    """Permanent 4410/4429 closes do not reconnect until the user retries."""
    _require_browser_e2e()
    manager = _LiveMatrixFixtureManager()
    with _serve_fixture(manager) as base_url:
        session = f"werewolf-browser-retry-{uuid.uuid4().hex[:8]}"
        try:
            _browser(session, "set", "viewport", "1280", "900")
            room = _create_running_room(session, base_url, manager)
            _enter_mode(session, "play")

            manager.force_close_client(
                room,
                mode="play",
                seat=1,
                code=4410,
                reason="client too slow",
            )
            _browser(session, "wait", "--text", "WebSocket 已关闭 (4410)")
            _browser(session, "wait", "--text", "手动重新连接")
            _wait_until(lambda: manager.active_clients(room, mode="play", seat=1) == 0)
            _browser(session, "find", "role", "button", "click", "--name", "手动重新连接")
            _browser(session, "wait", "--text", "WS connected")
            _wait_until(lambda: manager.active_clients(room, mode="play", seat=1) == 1)
            _browser(session, "screenshot", str(tmp_path / "retry-4410-recovered.png"))

            _browser(session, "set", "viewport", "390", "844")
            manager.force_close_client(
                room,
                mode="play",
                seat=1,
                code=4429,
                reason="room connection capacity reached",
            )
            _browser(session, "wait", "--text", "WebSocket 已关闭 (4429)")
            _browser(session, "wait", "--text", "手动重新连接")
            _wait_until(lambda: manager.active_clients(room, mode="play", seat=1) == 0)
            _browser(session, "screenshot", str(tmp_path / "retry-4429-mobile.png"))
            _browser(session, "find", "role", "button", "click", "--name", "手动重新连接")
            _browser(session, "wait", "--text", "WS connected")
            _wait_until(lambda: manager.active_clients(room, mode="play", seat=1) == 1)
            _assert_mobile_layout(session)
            _assert_capability_boundary(session, room)
            _assert_ws_capability_not_in_query(manager, room)
            _browser(session, "screenshot", str(tmp_path / "retry-4429-recovered.png"))
        finally:
            try:
                _browser(session, "close")
            except Exception:
                pass
