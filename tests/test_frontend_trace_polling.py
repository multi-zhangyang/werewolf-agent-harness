from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_get_trace_forwards_incremental_cursor_and_abort_signal(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / "frontend" / "node_modules" / "typescript").exists():
        pytest.skip("frontend dependencies are not installed")

    script = tmp_path / "check-trace-api.cjs"
    script.write_text(
        r"""
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = process.argv[2];
const ts = require(path.join(root, "frontend/node_modules/typescript"));
const source = fs.readFileSync(path.join(root, "frontend/src/lib/api.ts"), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;

const requests = [];
const fetch = async (url, options = {}) => {
  requests.push({ url, options });
  return { ok: true, json: async () => ({ trace: [] }) };
};
const moduleObj = { exports: {} };
vm.runInNewContext(compiled, {
  exports: moduleObj.exports,
  module: moduleObj,
  require,
  fetch,
  AbortController,
  encodeURIComponent,
}, { filename: "api.js" });

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

(async () => {
  const controller = new AbortController();
  await moduleObj.exports.getTrace("room id", "admin-token", 7, controller.signal);
  assert(requests.length === 1, "getTrace should make exactly one request");
  assert(requests[0].url === "/api/rooms/room id/trace?since=7", "trace cursor must be forwarded");
  assert(requests[0].options.signal === controller.signal, "AbortSignal identity must be preserved");
  assert(requests[0].options.headers["X-Room-Token"] === "admin-token", "admin token header must be preserved");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
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


def test_game_view_uses_completion_driven_cancellable_trace_polling() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "frontend" / "src" / "views" / "GameView.tsx").read_text(
        encoding="utf-8",
    )

    assert 'const shouldPoll = mode === "god" && Boolean(adminToken);' in source
    assert "new AbortController()" in source
    assert "controller.signal" in source
    assert "traceRequestRef.current?.abort()" in source
    assert "window.setTimeout" in source
    assert "window.setInterval" not in source
    assert "mergeTraceItems(previous" in source
    assert "response.trace_seq < requestedCursor" in source
    assert "getTrace(roomId, token, null, controller.signal)" in source
