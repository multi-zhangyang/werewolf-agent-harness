from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_human_controls_follow_only_server_advertised_schema(tmp_path: Path) -> None:
    """Exercise the dependency-free ActionRequest projection without a browser."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "frontend" / "node_modules" / "typescript").exists():
        pytest.skip("frontend dependencies are not installed")

    script = tmp_path / "check-human-actions.cjs"
    script.write_text(
        r"""
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = process.argv[2];
const ts = require(path.join(root, "frontend/node_modules/typescript"));
const source = fs.readFileSync(path.join(root, "frontend/src/lib/human-actions.ts"), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const moduleObj = { exports: {} };
vm.runInNewContext(compiled, {
  exports: moduleObj.exports,
  module: moduleObj,
  require,
  Map,
  Set,
  Number,
  Object,
  Array,
}, { filename: "human-actions.js" });

const { humanActionControls, parseHumanActionSchema, resolveAction } = moduleObj.exports;
function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const state = {
  mySeat: 1,
  seats: [
    { seat: 1, name: "A", alive: true },
    { seat: 2, name: "B", alive: true },
    { seat: 3, name: "C", alive: false },
    { seat: 4, name: "D", alive: true },
  ],
};
const targetedActions = ["night_kill", "see", "save", "poison", "guard", "hunter_shot", "vote"];
for (const action of targetedActions) {
  const controls = humanActionControls(state, action, {
    requested_action: action,
    requires_target: true,
    can_skip: action === "save" || action === "poison" || action === "hunter_shot",
    allowed_target_seats: [4, 2],
  });
  assert(controls.ok, `${action} must accept its complete advertised schema`);
  assert(controls.schema.action === action, `${action} action identity must be exact`);
  assert(controls.schema.requiresTarget === true, `${action} must remain targeted`);
  assert(
    controls.targets.map((seat) => seat.seat).join(",") === "4,2",
    `${action} target controls must preserve only the advertised order`,
  );
}

for (const action of ["speak", "last_words"]) {
  const parsed = parseHumanActionSchema(action, {
    requested_action: action,
    requires_target: false,
    can_skip: true,
    allowed_target_seats: [],
  });
  assert(parsed.ok, `${action} must accept its complete advertised schema`);
  assert(parsed.schema.inputKind === "text", `${action} must render text input`);
  assert(parsed.schema.targetSeats.length === 0, `${action} must not infer targets`);
}

const emptyTargetSkip = parseHumanActionSchema("save", {
  requested_action: "save",
  requires_target: true,
  can_skip: true,
  allowed_target_seats: [],
});
assert(emptyTargetSkip.ok, "an explicitly skippable empty target set must remain executable");

const malformed = [
  ["missing requested action", "vote", { requires_target: true, can_skip: false, allowed_target_seats: [2] }],
  ["mismatched requested action", "vote", { requested_action: "see", requires_target: true, can_skip: false, allowed_target_seats: [2] }],
  ["unsupported action", "dance", { requested_action: "dance", requires_target: false, can_skip: true, allowed_target_seats: [] }],
  ["missing target requirement", "vote", { requested_action: "vote", can_skip: false, allowed_target_seats: [2] }],
  ["missing skip policy", "vote", { requested_action: "vote", requires_target: true, allowed_target_seats: [2] }],
  ["missing target set", "vote", { requested_action: "vote", requires_target: true, can_skip: false }],
  ["coerced string target", "vote", { requested_action: "vote", requires_target: true, can_skip: false, allowed_target_seats: ["2"] }],
  ["duplicate target", "vote", { requested_action: "vote", requires_target: true, can_skip: false, allowed_target_seats: [2, 2] }],
  ["target requirement mismatch", "vote", { requested_action: "vote", requires_target: false, can_skip: true, allowed_target_seats: [] }],
  ["text target mismatch", "speak", { requested_action: "speak", requires_target: true, can_skip: true, allowed_target_seats: [2] }],
  ["unexecutable target request", "save", { requested_action: "save", requires_target: true, can_skip: false, allowed_target_seats: [] }],
];
for (const [label, action, context] of malformed) {
  assert(!parseHumanActionSchema(action, context).ok, `${label} must fail closed`);
  assert(resolveAction(action, context) === "", `${label} must not infer a fallback action`);
}

const missingSnapshotSeat = humanActionControls(state, "vote", {
  requested_action: "vote",
  requires_target: true,
  can_skip: false,
  allowed_target_seats: [99],
});
assert(!missingSnapshotSeat.ok, "advertised seats absent from the snapshot must not render a guessed control");

const deadAdvertisedSeat = humanActionControls(state, "vote", {
  requested_action: "vote",
  requires_target: true,
  can_skip: false,
  allowed_target_seats: [3],
});
assert(
  deadAdvertisedSeat.ok && deadAdvertisedSeat.targets[0].seat === 3,
  "the client must not override server legality from locally inferred alive state",
);
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


def test_human_text_control_locks_until_server_terminal_event() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "frontend/src/components/HarnessConsole.tsx").read_text(
        encoding="utf-8",
    )

    assert "const [submittedAt, setSubmittedAt]" in source
    assert "if (submitted || expired || !state.connected) return;" in source
    assert "setSubmittedAt(Date.now());" in source
    assert 'entry.text.startsWith("真人操作被拒绝")' in source
    assert 'submitted ? "等待确认"' in source
    assert 'onHumanAction("skip", {})' in source
    assert "canSkip" in source
