from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_frontend_store_reducer_tracks_harness_events_without_synthetic_streams(tmp_path: Path) -> None:
    """Exercise the TypeScript reducer without adding a frontend test runner."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "frontend" / "node_modules" / "typescript").exists():
        pytest.skip("frontend dependencies are not installed")

    script = tmp_path / "check-store-reducer.cjs"
    script.write_text(
        r"""
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = process.argv[2];
const ts = require(path.join(root, "frontend/node_modules/typescript"));
const source = fs.readFileSync(path.join(root, "frontend/src/lib/store.ts"), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const moduleObj = { exports: {} };
vm.runInNewContext(compiled, {
  exports: moduleObj.exports,
  module: moduleObj,
  require,
  console,
  Date,
  Map,
  Set,
  Number,
  String,
  Object,
}, { filename: "store.js" });

const { makeInitial, reduce } = moduleObj.exports;
function assert(condition, message) {
  if (!condition) throw new Error(message);
}
function seat(state, n) {
  return state.seats.find((item) => item.seat === n);
}

let state = makeInitial();
state = reduce(state, { type: "__context__", mySeat: null, mode: "spectate" });
state = reduce(state, {
  type: "snapshot",
  status: "running",
  view: {
    phase: "day",
    day: 1,
    players: [
      { id: "p1", seat: 1, name: "A", alive: true },
      { id: "p2", seat: 2, name: "B", alive: true },
      { id: "p3", seat: 3, name: "C", alive: true },
      { id: "p4", seat: 4, name: "D", alive: true },
      { id: "p5", seat: 5, name: "E", alive: true },
      { id: "p6", seat: 6, name: "F", alive: true },
    ],
  },
});

const exactSpeech = "我是狼人。这是 agent 本次 decision 的原文。";
state = reduce(state, {
  type: "speech",
  day: 1,
  seat: 1,
  name: "A",
  text: exactSpeech,
  bid: 3,
  reply_to: 2,
  accuses: [2],
});
const speech = state.log.find((entry) => entry.kind === "speech");
assert(speech && speech.text === exactSpeech, "store must preserve exact agent speech");
assert(speech.bid === 3, "store must retain the real bid");
assert(speech.replyTo === 2, "store must retain public reply metadata");
assert(JSON.stringify(state).indexOf("streamId") === -1, "store must not synthesize stream state");
state = reduce(state, {
  type: "speech",
  day: 1,
  seat: 1,
  name: "A",
  text: exactSpeech,
  bid: 3,
  reply_to: 2,
  accuses: [2],
});
assert(
  state.log.filter((entry) => entry.kind === "speech" && entry.text === exactSpeech).length === 2,
  "identical public outputs remain separate delivery records",
);

state = reduce(state, {
  type: "vote_cast",
  day: 1,
  seat: 1,
  name: "A",
  target_seat: 2,
});
assert(state.votes[1] === 2, "vote should be recorded");
assert(seat(state, 1).votedTarget === 2, "agent row should expose its accepted vote");

state = reduce(state, { type: "__context__", mySeat: 1, mode: "play" });
state = reduce(state, {
  type: "human_action_request",
  request_id: "req-timeout",
  seat: 1,
  action_type: "speak",
  context: {},
  timeout: 60,
  day: 1,
  phase: "day",
});
assert(state.pendingHuman?.requestId === "req-timeout", "human request should become pending");
state = reduce(state, {
  type: "agent_decision_failed",
  request_id: "another-request",
  seat: 1,
  phase: "day",
  reason: "unrelated failure",
  agent_kind: "human",
});
assert(state.pendingHuman?.requestId === "req-timeout", "unrelated failure must not clear pending request");
state = reduce(state, {
  type: "agent_decision_failed",
  request_id: "req-timeout",
  seat: 1,
  phase: "day",
  reason: "真人决策超时或失败,本请求未产生 DecisionEnvelope。",
  agent_kind: "human",
  timeout: true,
});
assert(state.pendingHuman === undefined, "matching decision failure must clear stale human request");

state = reduce(state, {
  type: "human_action_request",
  request_id: "req-invalid",
  seat: 1,
  action_type: "speak",
  context: {},
  timeout: 60,
  day: 1,
  phase: "day",
});
state = reduce(state, {
  type: "decision_envelope_rejected",
  request_id: "req-invalid",
  seat: 1,
  phase: "day",
  action: "speak",
  reason: "DecisionEnvelope 未通过协议校验,本请求未被执行。",
  error_type: "DecisionEnvelopeRejected",
});
assert(state.pendingHuman === undefined, "matching envelope rejection must clear pending request");

state = reduce(state, {
  type: "human_action_request",
  request_id: "req-expired",
  seat: 1,
  action_type: "vote",
  context: {},
  timeout: 60,
  day: 1,
  phase: "voting",
});
state = reduce(state, {
  type: "human_action_expired",
  request_id: "req-expired",
  seat: 1,
  action_type: "vote",
  reason: "human_timeout",
  day: 1,
  phase: "voting",
});
assert(state.pendingHuman === undefined, "explicit expiry must clear pending human request");

state = reduce(state, {
  type: "human_action_request",
  request_id: "req-stale",
  seat: 1,
  action_type: "speak",
  context: {},
  timeout: 60,
  day: 2,
  phase: "day",
});
state = reduce(state, {
  type: "human_action_rejected",
  request_id: "req-stale",
  seat: 1,
  reason: "no_pending_request",
});
assert(state.pendingHuman === undefined, "terminal stale rejection must clear pending request");

state = reduce(state, {
  type: "human_action_request",
  request_id: "req-rules-rejected",
  seat: 1,
  action_type: "vote",
  context: {},
  timeout: 60,
  day: 2,
  phase: "voting",
});
state = reduce(state, {
  type: "action_rejected",
  request_id: "req-rules-rejected",
  seat: 1,
  phase: "voting",
  action: "vote",
  reason_code: "target_not_allowed",
  reason: "RulesEngine rejected the validated action.",
});
assert(state.pendingHuman === undefined, "matching rules rejection must clear pending human request");

// A fresh snapshot after a retained-history gap is authoritative. It must not
// leave an old human control, vote map, analysis, or death projection alive.
state = reduce(state, {
  type: "human_action_request",
  request_id: "req-before-gap",
  seat: 1,
  action_type: "vote",
  context: {},
  timeout: 60,
  day: 2,
  phase: "voting",
});
state = reduce(state, {
  type: "vote_cast",
  day: 2,
  seat: 1,
  name: "A",
  target_seat: 2,
});
state = reduce(state, {
  type: "analysis",
  analysis: {
    winner: null,
    days: 2,
    turn_policy: "test",
    seats: [],
    decision_count: 0,
    decision_trace_metrics: {},
    parse_metrics: {},
    decision_failure_metrics: {},
  },
});
state = reduce(state, {
  type: "snapshot",
  status: "running",
  history_gap: true,
  cursor: 3,
  replay_from: 2,
  stream_id: "fresh-stream",
  resumed_from: null,
  view: {
    phase: "day",
    day: 1,
    players: [
      { id: "p1", seat: 1, name: "A", alive: true },
      { id: "p2", seat: 2, name: "B", alive: true },
    ],
    votes: {},
  },
});
assert(state.pendingHuman === undefined, "history-gap snapshot must clear stale human request");
assert(Object.keys(state.votes).length === 0, "history-gap snapshot must clear stale votes");
assert(state.analysis === undefined, "history-gap snapshot must clear stale analysis");
assert(state.lastDeaths.length === 0, "history-gap snapshot must clear stale death projection");
assert(state.log.length === 0, "history-gap snapshot must rebuild the timeline");
assert(seat(state, 1).lastSpeech === undefined, "history-gap snapshot must clear stale seat speech");
assert(seat(state, 1).deathReason === undefined, "history-gap snapshot must clear stale seat death metadata");

// Long runs retain a bounded, deterministic client timeline.
for (let index = 0; index < 700; index += 1) {
  state = reduce(state, {
    type: "speech",
    day: 1,
    seat: 1,
    name: "A",
    text: `bounded-${index}`,
  });
}
assert(state.log.length === 500, "store timeline must be bounded to 500 records");
assert(state.log[0].text === "bounded-200", "bounded timeline must evict oldest records first");

state = reduce(state, {
  type: "room_cleanup_failed",
  stage: "provider_scope_close",
  error_type: "CleanupError",
  pending_task_count: 2,
  fatal: true,
});
assert(state.error.includes("provider_scope_close/CleanupError"), "cleanup failure should be retained as an error");
assert(state.log.some((entry) => entry.kind === "failed" && entry.text.includes("pending=2")), "cleanup failure should be visible in the log");

state = reduce(state, {
  type: "__close__",
  code: 4403,
  reason: "admin token required",
  willReconnect: false,
});
assert(state.connected === false, "permanent socket close should mark the store disconnected");
assert(state.error.includes("4403") && state.error.includes("admin token required"), "socket close code/reason should be retained");
assert(state.socketClose?.retryableByUser === false, "authorization failure must not expose a retry loop");

state = reduce(state, {
  type: "__close__",
  code: 4429,
  reason: "WebSocket rate limit exceeded",
  willReconnect: false,
});
assert(state.socketClose?.code === 4429, "rate-limit close metadata must be retained");
assert(state.socketClose?.retryableByUser === true, "rate-limit close must allow deliberate user retry");
state = reduce(state, { type: "__manual_reconnect__" });
assert(state.socketClose === undefined, "manual retry must consume the permanent close state");
assert(state.error.includes("重新连接"), "manual retry must expose its in-progress state");
state = reduce(state, { type: "__open__" });
assert(state.socketClose === undefined, "successful reconnect must clear permanent close metadata");
assert(state.error === undefined, "successful reconnect must clear the stale close error");

state = reduce(state, { type: "game_ended", winner: "village" });
assert(state.status === "ended", "game_ended should close the run");
assert(state.winner === "village", "winner should be retained");
state = reduce(state, {
  type: "room_status",
  status: "interrupted",
  reason: "process_restart",
  error: "room interrupted during process restart",
});
assert(state.status === "interrupted", "restart interruption must remain a terminal status");
assert(
  state.log.some((entry) => entry.text.includes("房间已中断")),
  "restart interruption should have a distinct factual log entry",
);

state = reduce(state, {
  type: "human_action_request",
  request_id: "req-game-error",
  seat: 1,
  action_type: "vote",
  context: {
    requested_action: "vote",
    requires_target: true,
    can_skip: false,
    allowed_target_seats: [2],
  },
  timeout: 60,
  day: 2,
  phase: "voting",
});
assert(state.pendingHuman?.requestId === "req-game-error", "fatal-path request should become pending");
state = reduce(state, {
  type: "game_error",
  message: "orchestrator failed",
  reason: "error",
});
assert(state.pendingHuman === undefined, "game_error must immediately close an impossible human request");
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
