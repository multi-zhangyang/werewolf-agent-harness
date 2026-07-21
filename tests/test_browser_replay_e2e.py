from __future__ import annotations

import json
import os
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

from src.api.room_manager import Room, RoomManager
from src.api.server import create_app
from src.game.models import Phase
from src.game.roles import Team, default_role_deck
from src.game.rules import RulesEngine


ROOT = Path(__file__).resolve().parents[1]


class _EndedReplayFixtureManager(RoomManager):
    """Create a terminal room through the normal REST boundary for browser QA."""

    def create_room(self, **kwargs: Any) -> Room:
        room = super().create_room(**kwargs)
        RulesEngine.deal_roles(
            room.state,
            deck=default_role_deck(len(room.state.players)),
            seed=room.role_seed,
        )
        room.state.phase = Phase.ENDED
        room.state.winner = Team.VILLAGE

        self._store_room_event(room, {
            "type": "phase_started",
            "phase": "day",
            "day": 1,
            "message": "天亮了",
        })
        self._store_room_event(room, {
            "type": "speech",
            "phase": "day",
            "day": 1,
            "seat": 1,
            "name": room.state.players[0].name,
            "text": "这是后端 transcript 中的原始公开发言。",
        })
        self._store_room_event(room, {
            "type": "phase_started",
            "phase": "voting",
            "day": 1,
            "message": "进入投票",
        })
        self._store_room_event(room, {
            "type": "vote_cast",
            "phase": "voting",
            "day": 1,
            "seat": 2,
            "name": room.state.players[1].name,
            "target_seat": 1,
            "text": "2号投给1号",
        })

        request_id = "req-browser-e2e-vote"
        record_trace = self._make_trace_recorder(room)
        record_trace({
            "kind": "agent_request",
            "request": {
                "request_id": request_id,
                "seat": 2,
                "phase": "voting",
                "action_kind": "vote",
                "legal_actions": [{
                    "action": "vote",
                    "target_seats": [1],
                    "can_skip": False,
                }],
            },
        })
        record_trace({
            "type": "agent_turn_started",
            "visibility": "admin",
            "request_id": request_id,
            "session_id": "seat-2",
            "turn_id": "turn-1",
            "seat": 2,
            "phase": "voting",
            "day": 1,
            "tool_count": 4,
        })
        record_trace({
            "type": "model_generation",
            "visibility": "admin",
            "request_id": request_id,
            "step": 1,
            "reasoning": "仅授权 God/Admin 可见的 fixture reasoning。",
            "tool_call_count": 1,
        })
        record_trace({
            "type": "tool_call_requested",
            "visibility": "admin",
            "request_id": request_id,
            "step": 1,
            "call_id": "call-observe",
            "tool": "read_public_events",
            "arguments_hash": "fixture-arguments-hash",
        })
        record_trace({
            "type": "tool_result",
            "visibility": "admin",
            "request_id": request_id,
            "step": 1,
            "call_id": "call-observe",
            "tool": "read_public_events",
            "ok": True,
            "terminal": False,
            "output_hash": "fixture-output-hash",
        })
        record_trace({
            "type": "agent_action_submitted",
            "visibility": "admin",
            "request_id": request_id,
            "step": 2,
            "action": "vote",
            "target_seat": 1,
        })
        record_trace({
            "kind": "agent_response",
            "request_id": request_id,
            "envelope": {
                "model_call_id": "fixture-model-call",
                "prompt_hash": "fixture-prompt-hash",
                "response_hash": "fixture-response-hash",
                "latency_seconds": 0.25,
                "parse_status": "tool_call",
                "decision": {
                    "action": "vote",
                    "target_seat": 1,
                    "reasoning": "仅授权 God/Admin 可见的 fixture reasoning。",
                },
            },
            "validation": {"valid": True, "issues": []},
        })

        self._store_room_event(room, {
            "type": "game_ended",
            "phase": "ended",
            "day": 1,
            "winner": "village",
            "message": "村民阵营获胜",
        })
        self._store_room_event(room, {
            "type": "analysis",
            "phase": "ended",
            "day": 1,
            "analysis": {
                "winner": "village",
                "days": 1,
                "turn_policy": "sequential",
                "decision_count": 1,
                "decision_trace_metrics": {
                    "trace_row_count": 7,
                    "request_count": 1,
                    "response_count": 1,
                    "response_failure_count": 0,
                    "response_cancelled_count": 0,
                    "response_validation_failure_count": 0,
                    "terminal_response_count": 1,
                    "unpaired_request_count": 0,
                    "duplicate_terminal_count": 0,
                    "orphan_terminal_count": 0,
                    "consumed_decision_count": 1,
                    "rules_resolution_count": 1,
                },
                "parse_metrics": {
                    "decision_count": 1,
                    "parsed_model_decision_count": 1,
                    "clean_parse_count": 1,
                    "parse_recovered_count": 0,
                    "parse_recovered_rate": 0.0,
                    "parse_recovered_by_action": {},
                    "parse_recovered_by_phase": {},
                    "parse_method_counts": {"tool_call": 1},
                    "lossy_consumed_count": 0,
                    "missing_provenance_count": 0,
                    "not_applicable_count": 0,
                },
                "decision_failure_metrics": {
                    "failure_count": 0,
                    "timeout_count": 0,
                    "by_phase": {},
                    "by_action": {},
                    "by_seat": {},
                    "by_error_type": {},
                    "records": [],
                },
            },
        })
        room.status = "ended"
        room.end_reason = "browser_e2e_fixture_completed"
        room.terminal_at = time.monotonic()
        return room


def _browser(session: str, *args: str, stdin: str | None = None) -> str:
    env = dict(os.environ)
    local_hosts = "127.0.0.1,localhost"
    env["NO_PROXY"] = local_hosts
    env["no_proxy"] = local_hosts
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
    server = uvicorn.Server(uvicorn.Config(
        create_app(manager=manager),
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
    ))
    thread = threading.Thread(target=server.run, name="browser-e2e-server", daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not server.started:
        if not thread.is_alive():
            raise RuntimeError("browser E2E server terminated during startup")
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("browser E2E server did not become ready")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("browser E2E server did not stop")


def test_real_backend_replay_browser_journey(tmp_path: Path) -> None:
    if os.environ.get("WEREWOLF_RUN_BROWSER_E2E") != "1":
        pytest.skip("set WEREWOLF_RUN_BROWSER_E2E=1 to run the real browser journey")
    if shutil.which("agent-browser") is None:
        pytest.fail("agent-browser is required for WEREWOLF_RUN_BROWSER_E2E=1")
    if not (ROOT / "frontend" / "dist" / "index.html").is_file():
        pytest.fail("build frontend/dist before running browser E2E")

    manager = _EndedReplayFixtureManager(terminal_room_ttl=3600)
    session = f"werewolf-browser-e2e-{uuid.uuid4().hex[:10]}"
    try:
        with _serve_fixture(manager) as base_url:
            _browser(session, "set", "viewport", "1280", "900")
            _browser(session, "open", f"{base_url}/")
            _browser(session, "wait", "--text", "创建真实对局")
            _browser(session, "find", "role", "button", "click", "--name", "创建房间")
            _browser(session, "wait", "--text", "进入对局")

            assert len(manager.rooms) == 1
            room = next(iter(manager.rooms.values()))
            capability = room.admin_token
            _browser(session, "network", "requests", "--clear")
            _browser(session, "find", "role", "button", "click", "--name", "复盘")
            _browser(session, "wait", "--text", "Environment timeline")

            desktop = _browser_eval(session, """({
              width: innerWidth,
              documentWidth: document.documentElement.scrollWidth,
              hasTimeline: document.body.innerText.includes("Environment timeline"),
              hasAnalysis: document.body.innerText.includes("Factual analysis"),
              localStorage: Object.fromEntries(Object.entries(localStorage)),
            })""")
            assert desktop["width"] == 1280
            assert desktop["documentWidth"] <= desktop["width"]
            assert desktop["hasTimeline"] is True
            assert desktop["hasAnalysis"] is True
            persisted = json.dumps(desktop["localStorage"], ensure_ascii=False)
            assert capability not in persisted
            assert "roomAuth" not in persisted

            _browser(session, "eval", "--stdin", stdin="""
const values = ["voting", "vote_cast", "2"];
for (const [index, element] of [...document.querySelectorAll("select")].entries()) {
  const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value").set;
  setter.call(element, values[index]);
  element.dispatchEvent(new Event("change", { bubbles: true }));
}
true;
""")
            _browser(session, "wait", "--fn", "document.querySelectorAll('article').length === 1")
            filtered = _browser_eval(session, """({
              rows: [...document.querySelectorAll("article")].map((item) => item.innerText),
              filters: [...document.querySelectorAll("select")].map((item) => item.value),
            })""")
            assert filtered["filters"] == ["voting", "vote_cast", "2"]
            assert len(filtered["rows"]) == 1
            assert "2号投给1号" in filtered["rows"][0]

            _browser(session, "find", "text", "request_id=req-browser-e2e-vote", "click", "--exact")
            _browser(session, "wait", "--text", "decision_action=vote")
            trace = _browser_eval(session, """({
              action: document.body.innerText.includes("decision_action=vote"),
              target: document.body.innerText.includes("target_seat=1"),
              tool: document.body.innerText.includes("tool=read_public_events"),
              reasoning: document.body.innerText.includes("仅授权 God/Admin 可见的 fixture reasoning。"),
            })""")
            assert trace == {"action": True, "target": True, "tool": True, "reasoning": True}

            _browser(session, "screenshot", str(tmp_path / "replay-desktop.png"))
            _browser(session, "set", "viewport", "390", "844")
            mobile = _browser_eval(session, """({
              width: innerWidth,
              documentWidth: document.documentElement.scrollWidth,
              overflowCount: [...document.querySelectorAll("*")].filter((element) => {
                const rect = element.getBoundingClientRect();
                return rect.left < -1 || rect.right > innerWidth + 1;
              }).length,
              hasTrace: document.body.innerText.includes("ActionRequest / DecisionEnvelope"),
            })""")
            assert mobile == {
                "width": 390,
                "documentWidth": 390,
                "overflowCount": 0,
                "hasTrace": True,
            }
            _browser(session, "screenshot", str(tmp_path / "replay-mobile.png"))

            network_raw = _browser(session, "network", "requests", "--filter", "/ws/", "--json")
            network = json.loads(network_raw)
            assert network["success"] is True
            assert network["data"]["requests"] == []
            assert (tmp_path / "replay-desktop.png").stat().st_size > 0
            assert (tmp_path / "replay-mobile.png").stat().st_size > 0
    finally:
        try:
            _browser(session, "close")
        except Exception:
            pass
