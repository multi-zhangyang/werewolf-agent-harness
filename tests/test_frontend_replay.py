from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_frontend_replay_projection_is_immutable_and_filterable(tmp_path: Path) -> None:
    """Exercise the replay projection without starting a browser or a model."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "frontend" / "node_modules" / "typescript").exists():
        pytest.skip("frontend dependencies are not installed")

    script = tmp_path / "check-replay.cjs"
    script.write_text(
        r"""
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const root = process.argv[2];
const ts = require(path.join(root, "frontend/node_modules/typescript"));
const source = fs.readFileSync(path.join(root, "frontend/src/lib/replay.ts"), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const moduleObj = { exports: {} };
const localRequire = (name) => {
  if (name === "./store") {
    return { makeInitial: () => ({ connected: false, status: "waiting", phase: "setup", day: 0, winner: null, mode: "spectate", seats: [] }) };
  }
  return require(name);
};
vm.runInNewContext(compiled, {
  exports: moduleObj.exports,
  module: moduleObj,
  require: localRequire,
  Object,
  Array,
  Number,
  Set,
  JSON,
}, { filename: "replay.js" });

const {
  freezeReplayPayload,
  replayTimeline,
  replayDecisionTrace,
  filterReplayRows,
  replayFilterOptions,
} = moduleObj.exports;
function assert(condition, message) { if (!condition) throw new Error(message); }

const payload = {
  room_id: "replay-test",
  status: "ended",
  winner: "village",
  events: [],
  players: [],
  transcript: {
    entries: [
      { seq: 1, kind: "event", phase: "day", day: 1, seat: 1, payload: { type: "speech", text: "first" } },
      { seq: 2, kind: "event", phase: "voting", day: 1, seat: 2, payload: { type: "vote_cast", text: "second" } },
      { seq: 3, kind: "decision", phase: "voting", day: 1, seat: 2, payload: { kind: "agent_request", request: { request_id: "req-1", seat: 2 } } },
      { seq: 4, kind: "decision", phase: "voting", day: 1, seat: 2, payload: { kind: "agent_response", request_id: "req-1", envelope: { decision: { action: "vote" } }, validation: { valid: true, issues: [] } } },
    ],
  },
};
const frozen = freezeReplayPayload(payload);
assert(Object.isFrozen(frozen), "payload root must be frozen");
assert(Object.isFrozen(payload.transcript), "nested transcript must be frozen");
try { payload.transcript.entries.push({ seq: 99 }); } catch (_) {}
assert(payload.transcript.entries.length === 4, "frozen evidence must reject mutation");

const rows = replayTimeline(payload);
assert(rows.length === 2 && rows[0].seq === 1 && rows[1].kind === "vote_cast", "event timeline must exclude decision rows");
const filtered = filterReplayRows(rows, { phase: "voting", kind: "vote_cast", seat: "2" }, 1);
assert(filtered.length === 1 && filtered[0].text === "second", "playhead and filters must compose");
assert(replayFilterOptions(rows, "phase").join(",") === "day,voting", "phase options must be factual");
const trace = replayDecisionTrace(payload);
assert(trace.length === 2 && trace[0].kind === "decision", "decision entries must form protocol trace input");
""",
        encoding="utf-8",
    )
    subprocess.run(["node", str(script), str(root)], check=True, capture_output=True, text=True)
