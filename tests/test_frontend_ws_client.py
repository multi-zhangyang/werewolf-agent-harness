from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_frontend_websocket_reconnect_lifecycle_is_bounded(tmp_path: Path) -> None:
    """Exercise timer cancellation and permanent-close handling without a browser."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "frontend" / "node_modules" / "typescript").exists():
        pytest.skip("frontend dependencies are not installed")

    script = tmp_path / "check-ws-client.cjs"
    script.write_text(
        r"""
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = process.argv[2];
const ts = require(path.join(root, "frontend/node_modules/typescript"));
const source = fs.readFileSync(path.join(root, "frontend/src/lib/ws-client.ts"), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

let nextTimer = 1;
const timeouts = new Map();
const intervals = new Map();
const windowMock = {
  setTimeout(callback, delay) {
    const id = nextTimer++;
    timeouts.set(id, {
      delay,
      callback: () => {
        timeouts.delete(id);
        callback();
      },
    });
    return id;
  },
  clearTimeout(id) {
    timeouts.delete(id);
  },
  setInterval(callback, delay) {
    const id = nextTimer++;
    intervals.set(id, { callback, delay });
    return id;
  },
  clearInterval(id) {
    intervals.delete(id);
  },
};

class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static instances = [];

  constructor(url, protocols) {
    this.url = url;
    this.protocols = protocols;
    this.readyState = MockWebSocket.CONNECTING;
    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;
    this.closeCalls = [];
    MockWebSocket.instances.push(this);
  }

  close(code = 1000, reason = "") {
    this.closeCalls.push({ code, reason });
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) this.onclose({ code, reason, wasClean: true });
  }

  open() {
    this.readyState = MockWebSocket.OPEN;
    if (this.onopen) this.onopen();
  }

  message(data) {
    if (this.onmessage) this.onmessage({ data });
  }

  serverClose(code, reason, wasClean = false) {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) this.onclose({ code, reason, wasClean });
  }

  send() {}
}

const moduleObj = { exports: {} };
vm.runInNewContext(compiled, {
  exports: moduleObj.exports,
  module: moduleObj,
  require,
  console,
  window: windowMock,
  WebSocket: MockWebSocket,
  location: { protocol: "https:", host: "example.test" },
  URL,
  URLSearchParams,
  clearInterval: windowMock.clearInterval,
  JSON,
  Number,
  String,
  RegExp,
}, { filename: "ws-client.js" });

const { GameSocket } = moduleObj.exports;
const urlWithoutToken = moduleObj.exports.buildWsUrl("room", 2, "play", "secret-token");
assert(!urlWithoutToken.includes("secret-token"), "capability must not be placed in the WebSocket URL");
assert(urlWithoutToken.includes("mode=play") && urlWithoutToken.includes("seat=2"), "WS URL must retain mode and seat");
const closeEvents = [];
const handlers = {
  onEvent() {},
  onOpen() {},
  onClose(info) { closeEvents.push(info); },
  onError() {},
};

const authenticated = new GameSocket("wss://example.test/ws/room?mode=god", handlers, "secret-token");
authenticated.connect();
const authenticatedSocket = MockWebSocket.instances.at(-1);
assert(Array.isArray(authenticatedSocket.protocols), "WebSocket protocols must be supplied as an array");
assert(authenticatedSocket.protocols[0] === "werewolf.v1", "stable WebSocket protocol must be offered first");
assert(authenticatedSocket.protocols[1] === "werewolf.cap.secret-token", "capability must be carried in a subprotocol");
assert(!authenticatedSocket.url.includes("secret-token"), "authenticated WS URL must not contain the capability");
authenticated.close();

const baselineInstances = MockWebSocket.instances.length;
const transient = new GameSocket("ws://example.test/ws/room?mode=spectate", handlers);
transient.connect();
transient.connect();
assert(MockWebSocket.instances.length === baselineInstances + 1, "connect must not open a second CONNECTING socket");
MockWebSocket.instances.at(-1).serverClose(1006, "network lost");
assert(closeEvents.at(-1).code === 1006, "close code must be exposed");
assert(closeEvents.at(-1).reason === "network lost", "close reason must be exposed");
assert(closeEvents.at(-1).willReconnect === true, "transient network close should reconnect");
assert(timeouts.size === 1, "one transient close should schedule exactly one reconnect");
transient.close();
assert(timeouts.size === 0, "close must cancel the pending reconnect timer");
for (const { callback } of [...timeouts.values()]) callback();
assert(MockWebSocket.instances.length === baselineInstances + 1, "a cancelled reconnect must not open an orphan socket");

const permanent = new GameSocket("ws://example.test/ws/room?mode=god", handlers);
permanent.connect();
MockWebSocket.instances.at(-1).serverClose(4403, "admin token required");
assert(closeEvents.at(-1).code === 4403, "permanent close code must be exposed");
assert(closeEvents.at(-1).reason === "admin token required", "permanent close reason must be exposed");
assert(closeEvents.at(-1).willReconnect === false, "auth failure must terminate reconnects");
assert(timeouts.size === 0, "auth failure must not schedule a reconnect");

const replayModeError = new GameSocket("ws://example.test/ws/room?mode=replay", handlers);
replayModeError.connect();
MockWebSocket.instances.at(-1).serverClose(4409, "replay is available only after room termination");
assert(closeEvents.at(-1).willReconnect === false, "non-history 4409 must be permanent");
assert(timeouts.size === 0, "non-history 4409 must not spin");

const rateLimited = new GameSocket("ws://example.test/ws/room?mode=spectate", handlers);
rateLimited.connect();
MockWebSocket.instances.at(-1).serverClose(4429, "WebSocket rate limit exceeded");
assert(closeEvents.at(-1).willReconnect === false, "rate-limit close must not create a reconnect loop");
assert(timeouts.size === 0, "rate-limit close must require deliberate user retry");
const rateLimitedInstances = MockWebSocket.instances.length;
rateLimited.connect();
assert(MockWebSocket.instances.length === rateLimitedInstances + 1, "manual retry must open exactly one replacement socket");
rateLimited.connect();
assert(MockWebSocket.instances.length === rateLimitedInstances + 1, "manual retry must not duplicate a CONNECTING socket");
rateLimited.close();

const slowClient = new GameSocket("ws://example.test/ws/room?mode=spectate", handlers);
slowClient.connect();
MockWebSocket.instances.at(-1).serverClose(4410, "client too slow");
assert(closeEvents.at(-1).willReconnect === false, "backpressure close must not immediately overload the server again");
assert(timeouts.size === 0, "slow-client close must require deliberate user retry");
const slowClientInstances = MockWebSocket.instances.length;
slowClient.connect();
assert(MockWebSocket.instances.length === slowClientInstances + 1, "slow-client manual retry must open one socket");
slowClient.close();

const historyGap = new GameSocket("ws://example.test/ws/room?mode=spectate", handlers);
historyGap.connect();
MockWebSocket.instances.at(-1).serverClose(4409, "history gap; earliest=4; current=8");
assert(closeEvents.at(-1).willReconnect === true, "retained-history 4409 should request a fresh snapshot");
assert(timeouts.size === 1, "retained-history gap should schedule one reconnect");
historyGap.close();
assert(timeouts.size === 0, "closing a history-gap client must cancel its fresh-snapshot timer");

// A delayed callback from an old transport must not affect the replacement.
const staleEvents = [];
const staleErrors = [];
const stale = new GameSocket("ws://example.test/ws/room?mode=spectate", {
  onEvent(ev) { staleEvents.push(ev); },
  onOpen() {},
  onClose(info) { staleEvents.push({ type: "close", info }); },
  onError(message) { staleErrors.push(message); },
});
stale.connect();
const oldSocket = MockWebSocket.instances.at(-1);
oldSocket.serverClose(1006, "network lost");
assert(timeouts.size === 1, "stale-callback probe should have one reconnect timer");
const reconnectTimer = [...timeouts.values()][0];
reconnectTimer.callback();
const replacement = MockWebSocket.instances.at(-1);
assert(replacement !== oldSocket, "reconnect must allocate a replacement socket");
const beforeStaleCallbacks = staleEvents.length;
oldSocket.message(JSON.stringify({ type: "game_ended", winner: "village" }));
oldSocket.onerror && oldSocket.onerror();
oldSocket.onclose && oldSocket.onclose({ code: 1006, reason: "late", wasClean: false });
assert(staleEvents.length === beforeStaleCallbacks, "late callbacks from an old socket must be ignored");
assert(staleErrors.length === 0, "late old-socket error must not reach the reducer");
stale.close();

function snapshot(streamId, cursor, resumedFrom = null, replayFrom = 1) {
  return JSON.stringify({
    type: "snapshot",
    status: "running",
    stream_id: streamId,
    cursor,
    resumed_from: resumedFrom,
    replay_from: replayFrom,
    view: { phase: "day", day: 1, players: [] },
  });
}

// A sequence gap is recoverable: clear the cursor, close with the client
// resync code, and schedule exactly one fresh snapshot.
const gapErrors = [];
const gapCloses = [];
const gap = new GameSocket("ws://example.test/ws/room?mode=spectate", {
  onEvent() {},
  onOpen() {},
  onClose(info) { gapCloses.push(info); },
  onError(message) { gapErrors.push(message); },
});
gap.connect();
const gapSocket = MockWebSocket.instances.at(-1);
gapSocket.open();
gapSocket.message(snapshot("stream-gap", 0));
gapSocket.message(JSON.stringify({ type: "speech", day: 1, seat: 1, name: "A", text: "first", delivery_seq: 1, delivery_id: "stream-gap.1" }));
gapSocket.message(JSON.stringify({ type: "speech", day: 1, seat: 2, name: "B", text: "third", delivery_seq: 3, delivery_id: "stream-gap.3" }));
assert(gapErrors.at(-1).includes("delivery 存在缺口"), "delivery gap must be surfaced as a resync error");
assert(gapSocket.closeCalls.at(-1).code === 4001, "delivery gap must use recoverable resync close code");
assert(gapCloses.at(-1).willReconnect === true, "delivery gap must schedule a fresh reconnect");
assert(timeouts.size === 1, "delivery gap must schedule one reconnect timer");
gap.close();

// A restarted delivery stream during resume is also recoverable.
const streamChange = new GameSocket("ws://example.test/ws/room?mode=spectate", {
  onEvent() {},
  onOpen() {},
  onClose(info) { gapCloses.push(info); },
  onError(message) { gapErrors.push(message); },
});
streamChange.connect();
const firstStream = MockWebSocket.instances.at(-1);
firstStream.open();
firstStream.message(snapshot("stream-a", 0));
firstStream.message(JSON.stringify({ type: "speech", day: 1, seat: 1, name: "A", text: "first", delivery_seq: 1, delivery_id: "stream-a.1" }));
firstStream.serverClose(1006, "network lost");
assert(timeouts.size === 1, "stream-change probe should have one reconnect timer");
const streamTimer = [...timeouts.values()][0];
streamTimer.callback();
const secondStream = MockWebSocket.instances.at(-1);
secondStream.open();
secondStream.message(snapshot("stream-b", 1, 1, 2));
assert(secondStream.closeCalls.at(-1).code === 4001, "stream change must request recoverable resync");
assert(gapCloses.at(-1).willReconnect === true, "stream change must continue reconnecting");
streamChange.close();

// A duplicate sequence carrying a different id is a protocol conflict, not
// an idempotent replay, and must stop reconnect spinning.
const duplicateErrors = [];
const duplicate = new GameSocket("ws://example.test/ws/room?mode=spectate", {
  onEvent() {},
  onOpen() {},
  onClose(info) { gapCloses.push(info); },
  onError(message) { duplicateErrors.push(message); },
});
duplicate.connect();
const duplicateSocket = MockWebSocket.instances.at(-1);
duplicateSocket.open();
duplicateSocket.message(snapshot("stream-duplicate", 0));
duplicateSocket.message(JSON.stringify({ type: "speech", day: 1, seat: 1, name: "A", text: "first", delivery_seq: 1, delivery_id: "stream-duplicate.1" }));
duplicateSocket.message(JSON.stringify({ type: "speech", day: 1, seat: 1, name: "A", text: "tampered", delivery_seq: 1, delivery_id: "stream-duplicate.other" }));
assert(duplicateErrors.at(-1).includes("序号重复"), "same delivery sequence with another id must be rejected");
assert(duplicateSocket.closeCalls.at(-1).code === 4002, "conflicting duplicate must use permanent protocol close");
assert(gapCloses.at(-1).willReconnect === false, "conflicting duplicate must not reconnect forever");
duplicate.close();

// A half-open transport cannot wait forever: missing pong forces the same
// bounded fresh-snapshot recovery path.
const heartbeatErrors = [];
const heartbeatCloses = [];
const heartbeat = new GameSocket("ws://example.test/ws/room?mode=spectate", {
  onEvent() {},
  onOpen() {},
  onClose(info) { heartbeatCloses.push(info); },
  onError(message) { heartbeatErrors.push(message); },
});
heartbeat.connect();
const heartbeatSocket = MockWebSocket.instances.at(-1);
heartbeatSocket.open();
heartbeatSocket.message(snapshot("stream-heartbeat", 0));
assert(intervals.size === 1, "open connection must install one heartbeat interval");
const heartbeatInterval = [...intervals.values()][0];
assert(heartbeatInterval.delay === 25000, "heartbeat cadence must remain bounded");
heartbeatInterval.callback();
assert(timeouts.size === 1, "outstanding ping must arm one pong timeout");
const pongTimeout = [...timeouts.values()][0];
assert(pongTimeout.delay === 10000, "pong timeout must be bounded");
pongTimeout.callback();
assert(heartbeatErrors.at(-1).includes("心跳超时"), "missing pong must be surfaced");
assert(heartbeatSocket.closeCalls.at(-1).code === 4001, "missing pong must request recoverable resync");
assert(heartbeatCloses.at(-1).willReconnect === true, "missing pong must reconnect");
assert(intervals.size === 0, "heartbeat interval must be removed on close");
heartbeat.close();
""",
        encoding="utf-8",
    )
    subprocess.run(
        ["node", str(script), str(root)],
        check=True,
        cwd=root,
        text=True,
        capture_output=True,
    )


def test_frontend_views_preserve_distinct_delivery_rows() -> None:
    root = Path(__file__).resolve().parents[1]
    console = (root / "frontend" / "src" / "components" / "HarnessConsole.tsx").read_text(encoding="utf-8")
    game_view = (root / "frontend" / "src" / "views" / "GameView.tsx").read_text(encoding="utf-8")

    assert "const entries = state.log;" in console
    assert "dedupeEntries" not in console
    assert "dedupeLog" not in game_view
    assert "state.log.filter(isTimelineEntry)" in game_view
