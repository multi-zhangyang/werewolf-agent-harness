from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def test_frontend_protocol_trace_surfaces_bounded_safe_private_tool_reasoning(tmp_path: Path) -> None:
    """Execute the dependency-free TypeScript trace projection with realistic payloads."""
    root = Path(__file__).resolve().parents[1]
    if not (root / "frontend" / "node_modules" / "typescript").exists():
        pytest.skip("frontend dependencies are not installed")

    script = tmp_path / "check-protocol-trace.cjs"
    script.write_text(
        r"""
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = process.argv[2];
const ts = require(path.join(root, "frontend/node_modules/typescript"));
const source = fs.readFileSync(path.join(root, "frontend/src/lib/protocol-trace.ts"), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const moduleObj = { exports: {} };
vm.runInNewContext(compiled, {
  exports: moduleObj.exports,
  module: moduleObj,
  require,
  Map,
  Number,
  Object,
  Array,
}, { filename: "protocol-trace.js" });

const { protocolRecords } = moduleObj.exports;
function assert(condition, message) {
  if (!condition) throw new Error(message);
}
function decision(payload, traceSeq) {
  return { kind: "decision", idx: traceSeq, trace_seq: traceSeq, payload };
}
function publicEvent(payload, traceSeq) {
  return { kind: "event", idx: traceSeq, trace_seq: traceSeq, payload };
}
function request(requestId, traceSeq) {
  return decision({
    kind: "agent_request",
    request: {
      request_id: requestId,
      seat: traceSeq,
      phase: "day",
      action_kind: "speak",
      legal_actions: [{ action: "speak", target_seats: [], can_skip: false }],
    },
  }, traceSeq);
}

const rejectedValue = "REJECTED_PRIVATE_VALUE_MUST_NEVER_SURFACE";
const rawToolValue = "RAW_TOOL_OUTPUT_MUST_NEVER_SURFACE";
const privatePlan = "先伪装预言家施压，再根据反应决定是否切割队友";
const credential = "sk-trace-projection-secret";
const accessToken = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiJ9.signaturevalue123456";
const longModelContent = "x".repeat(13000);
const successfulAttempts = [
  {
    attempt: 1,
    status: "response_rejected",
    error_type: "ValidationError",
    validation_issues: [{
      path: "private_state.candidate_plans",
      code: "too_short",
      input: rejectedValue,
      message: rejectedValue,
    }],
    rejected_value: rejectedValue,
    llm_call: { api_key: credential, raw_response: rejectedValue },
  },
  {
    attempt: 2,
    status: "accepted",
    error_type: null,
    llm_call: { api_key: credential },
  },
];
const failedAttempts = Array.from({ length: 12 }, (_, index) => ({
  attempt: index + 1,
  status: "provider_failed",
  error_type: `ProviderFailure${index}`,
  validation_issues: Array.from({ length: 12 }, (__, issueIndex) => ({
    path: `private_state.beliefs.${issueIndex}`,
    code: "invalid",
    input: rejectedValue,
  })),
  llm_call: { authorization: credential, response_body: rejectedValue },
}));

const records = protocolRecords([
  request("request-success", 1),
  decision({
    type: "agent_turn_started",
    visibility: "admin",
    audience: "admin",
    request_id: "request-success",
    session_id: "request-success:agent-session",
    turn_id: "turn-1",
    seat: 1,
    state_version: 0,
    phase: "day",
    day: 1,
    tool_count: 11,
    context: { raw_prompt: rawToolValue, api_key: credential },
  }, 2),
  decision({
    type: "agent_history_compacted",
    visibility: "admin",
    audience: "admin",
    request_id: "request-success",
    session_id: "request-success:agent-session",
    turn_id: "turn-1",
    seat: 1,
    state_version: 0,
    step: 1,
    original_message_count: 3,
    model_message_count: 3,
    original_chars: 32000,
    model_chars: 32000,
    compacted_tool_groups: 0,
    limit_satisfied: false,
    model_history_hash: "bounded-history-hash",
  }, 2.5),
  decision({
    type: "model_generation",
    visibility: "admin",
    audience: "admin",
    request_id: "request-success",
    session_id: "request-success:agent-session",
    turn_id: "turn-1",
    seat: 1,
    state_version: 0,
    step: 1,
    call_id: "model-call-1",
    content: longModelContent,
    reasoning: `bounded admin reasoning ${credential} access_token=${accessToken}`,
    latency: 1.25,
    tool_call_count: 1,
    usage: {
      prompt_tokens: 120,
      completion_tokens: 30,
      arbitrary_private_counter: 99,
    },
    router_trace: { authorization: credential, raw_response: rawToolValue },
  }, 3),
  decision({
    type: "model_generation_failed",
    visibility: "admin",
    audience: "admin",
    request_id: "request-success",
    session_id: "request-success:agent-session",
    turn_id: "turn-1",
    seat: 1,
    state_version: 0,
    step: 1,
    response_attempt: 1,
    will_retry: true,
    error_type: "LLMResponseError",
    call_id: "model-call-failed",
    request_hash: "failed-request-hash",
    router_trace: { authorization: credential, raw_response: rawToolValue },
  }, 3.5),
  decision({
    type: "tool_call_requested",
    visibility: "admin",
    audience: "admin",
    request_id: "request-success",
    session_id: "request-success:agent-session",
    turn_id: "turn-1",
    seat: 1,
    state_version: 0,
    step: 1,
    call_id: "tool-call-1",
    tool: "speak",
    arguments_hash: "arguments-hash",
    arguments: {
      speech: privatePlan,
      api_key: credential,
      access_token: accessToken,
      [`${"a".repeat(170)}_private_key`]: accessToken,
    },
  }, 4),
  decision({
    type: "tool_result",
    visibility: "admin",
    audience: "admin",
    request_id: "request-success",
    session_id: "request-success:agent-session",
    turn_id: "turn-1",
    seat: 1,
    state_version: 1,
    call_id: "tool-call-1",
    tool: "speak",
    kind: "terminal",
    ok: true,
    terminal: true,
    latency: 0.005,
    output_hash: "output-hash",
    output: { private_note: rawToolValue, api_key: credential },
    error: null,
  }, 5),
  decision({
    type: "agent_action_submitted",
    visibility: "admin",
    audience: "admin",
    request_id: "request-success",
    session_id: "request-success:agent-session",
    turn_id: "turn-1",
    seat: 1,
    state_version: 1,
    call_id: "tool-call-1",
    tool: "speak",
    action: "speak",
    decision: {
      action: "speak",
      target_seat: 2,
      speech: rawToolValue,
      reasoning: rawToolValue,
    },
  }, 6),
  decision({
    kind: "agent_response",
    request_id: "request-success",
    seat: 1,
    envelope: {
      decision: {
        action: "speak",
        speech: `public output ${credential}`,
        reasoning: `admin reasoning access_token=${accessToken}`,
      },
      model_call_id: "call-2",
      parse_status: "ok",
    },
    validation: { valid: true, issues: [] },
  }, 7),
  decision({
    type: "decision_consumed",
    request_id: "request-success",
    seat: 1,
    llm_call: { actor_response_attempts: successfulAttempts },
  }, 8),
  // Public events and explicitly non-admin rows cannot impersonate private
  // AgentSession evidence even when their payload shape matches.
  publicEvent({
    type: "model_generation",
    visibility: "admin",
    audience: "admin",
    request_id: "request-success",
    content: rawToolValue,
  }, 9),
  decision({
    type: "model_generation",
    visibility: "public",
    audience: "public",
    request_id: "request-success",
    content: rawToolValue,
  }, 10),
  request("request-failed", 11),
  decision({
    kind: "agent_response_failed",
    request_id: "request-failed",
    seat: 4,
    phase: "day",
    action: "speak",
    failure: {
      error_type: "AgentDecisionError",
      timeout: false,
      reason: "AgentDecisionError during day/speak",
      llm_call_attempts: failedAttempts,
    },
  }, 12),
  ...Array.from({ length: 260 }, (_, index) => decision({
    type: "tool_result",
    visibility: "admin",
    audience: "admin",
    request_id: "request-failed",
    session_id: "request-failed:agent-session",
    turn_id: "turn-failed",
    seat: 4,
    state_version: index,
    call_id: `tool-failed-${index}`,
    tool: "read_public_events",
    kind: "read_only",
    ok: true,
    terminal: false,
    output_hash: `hash-${index}`,
  }, 13 + index)),
]);

assert(records.length === 2, "both terminal records must remain paired with requests");
const successful = records.find((record) => record.requestId === "request-success");
const failed = records.find((record) => record.requestId === "request-failed");
assert(successful && successful.attempts.length === 2, "successful retry history must be projected");
assert(successful.attempts[0].attempt === 1, "attempt number must be retained");
assert(successful.attempts[0].status === "response_rejected", "attempt status must be retained");
assert(successful.attempts[0].errorType === "ValidationError", "attempt error type must be retained");
assert(
  successful.attempts[0].validationIssues[0].path === "private_state.candidate_plans"
    && successful.attempts[0].validationIssues[0].code === "too_short",
  "validation path and code must be retained",
);
assert(successful.attempts[1].status === "accepted", "accepted retry must be visible");
assert(successful.toolLoop, "tool-loop events must be grouped under their ActionRequest");
assert(successful.toolLoop.events.length === 7, "only the seven authorized tool-loop rows must be retained");
assert(successful.toolLoop.generationCount === 1, "model generation count must be retained");
assert(successful.toolLoop.generationFailureCount === 1, "model generation failure count must be retained");
assert(successful.toolLoop.toolCallCount === 1, "tool call count must be retained");
assert(successful.toolLoop.toolResultCount === 1, "tool result count must be retained");
assert(successful.toolLoop.terminalActionCount === 1, "terminal submission count must be retained");
assert(successful.toolLoop.historyCompactionCount === 0, "zero compacted groups must not inflate compaction count");
assert(successful.toolLoop.historyLimitMissCount === 1, "soft history-window misses must remain visible");
assert(
  successful.toolLoop.events.map((event) => event.type).join(",")
    === "agent_turn_started,agent_history_compacted,model_generation,model_generation_failed,tool_call_requested,tool_result,agent_action_submitted",
  "tool-loop event order must follow trace sequence",
);
const generation = successful.toolLoop.events.find((event) => event.type === "model_generation");
const historyWindow = successful.toolLoop.events.find((event) => event.type === "agent_history_compacted");
const failedGeneration = successful.toolLoop.events.find((event) => event.type === "model_generation_failed");
const call = successful.toolLoop.events.find((event) => event.type === "tool_call_requested");
assert(
  failedGeneration.responseAttempt === 1
    && failedGeneration.willRetry === true
    && failedGeneration.errorCode === "LLMResponseError",
  "bounded model-generation failure provenance must be retained",
);
const result = successful.toolLoop.events.find((event) => event.type === "tool_result");
const submitted = successful.toolLoop.events.find((event) => event.type === "agent_action_submitted");
assert(generation.content.length === 12000, "model content must be bounded");
assert(generation.reasoning.includes("bounded admin reasoning"), "authorized bounded reasoning must remain visible");
assert(generation.reasoning.includes("[redacted]"), "credentials in model reasoning must be redacted");
assert(generation.usage.prompt_tokens === 120 && generation.usage.completion_tokens === 30, "known usage must remain visible");
assert(!("arbitrary_private_counter" in generation.usage), "arbitrary usage keys must not enter the projection");
assert(
  historyWindow.compactedToolGroups === 0
    && historyWindow.limitSatisfied === false
    && historyWindow.originalChars === 32000,
  "history-window miss provenance must remain visible",
);
assert(call.tool === "speak" && call.argumentsHash === "arguments-hash", "tool request metadata must remain visible");
assert(call.argumentsText.includes(privatePlan), "bounded private tool reasoning must remain visible to God");
assert(call.argumentsText.includes("[redacted]"), "sensitive argument fields must be redacted client-side");
assert(result.ok === true && result.terminal === true && result.outputHash === "output-hash", "tool result status must remain visible");
assert(submitted.action === "speak" && submitted.targetSeat === 2, "terminal action metadata must remain visible");
assert(successful.response.speech.includes("[redacted]"), "final public text credentials must be redacted");
assert(successful.response.reasoning.includes("[redacted]"), "final private reasoning credentials must be redacted");
assert(failed && failed.attempts.length === 8, "failed attempt projection must be bounded");
assert(failed.attempts[0].validationIssues.length === 8, "validation issue projection must be bounded");
assert(failed.toolLoop && failed.toolLoop.events.length === 256, "tool-loop event history must be bounded");
assert(failed.toolLoop.truncated === true, "bounded tool-loop history must expose truncation");
assert(failed.toolLoop.toolResultCount === 256, "bounded tool-loop counts must match retained rows");

const projected = JSON.stringify(records);
assert(!projected.includes(rejectedValue), "rejected values must not enter the UI projection");
assert(!projected.includes(credential), "credentials must not enter the UI projection");
assert(!projected.includes(accessToken), "access tokens and JWTs must not enter the UI projection");
assert(!projected.includes(rawToolValue), "raw tool arguments/results/decision text must not enter the UI projection");
assert(projected.includes(privatePlan), "authorized private structured reasoning must enter the God projection");
assert(!projected.includes("llm_call"), "nested model call payloads must not enter the UI projection");
assert(!projected.includes("message"), "validation messages must not enter the UI projection");
assert(!projected.includes("router_trace"), "raw router traces must not enter the UI projection");
assert(!projected.includes('"output":'), "raw tool output objects must not enter the UI projection");
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
