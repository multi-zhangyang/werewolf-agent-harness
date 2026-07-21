import type { RoomTraceItem } from "./api";

const MAX_ACTOR_RESPONSE_ATTEMPTS = 8;
const MAX_VALIDATION_ISSUES_PER_ATTEMPT = 8;
const MAX_TOOL_LOOP_EVENTS = 256;
const MAX_TOOL_TRACE_TEXT = 12_000;
const MAX_TOOL_TRACE_METADATA = 160;
const MAX_TOOL_TRACE_ERROR = 600;
const MAX_TOOL_ARGUMENT_TEXT = 12_000;
const MAX_TOOL_ARGUMENT_STRING = 2_000;
const MAX_TOOL_ARGUMENT_ITEMS = 32;
const MAX_TOOL_ARGUMENT_KEYS = 32;
const MAX_TOOL_ARGUMENT_DEPTH = 5;
const MAX_USAGE_VALUE = 1_000_000_000;
const REDACTED = "[redacted]";
const SENSITIVE_KEY_FRAGMENTS = [
  "api_key", "apikey", "authorization", "bearer", "secret", "password",
  "x_api_key", "x_room_token", "admin_token", "seat_token", "seat_tokens",
  "access_token", "refresh_token", "id_token", "session_token", "client_secret",
  "private_key", "cookie", "set_cookie",
];

export type AgentToolLoopEventType =
  | "agent_turn_started"
  | "agent_history_compacted"
  | "model_generation"
  | "model_generation_failed"
  | "tool_call_requested"
  | "tool_result"
  | "agent_action_submitted";

/**
 * Safe, bounded projection of one private AgentSession event.
 *
 * The server keeps a bounded, credential-redacted admin trace. The console
 * projects model text and tool arguments in bounded, recursively sanitized form because structured
 * tool arguments are often the model's only observable plan/reasoning output.
 * Arbitrary tool results and router payloads remain excluded.
 */
export interface AgentToolLoopEvent {
  type: AgentToolLoopEventType;
  sequence: number;
  requestId: string;
  sessionId?: string;
  turnId?: string;
  seat?: number;
  stateVersion?: number;
  step?: number;
  phase?: string;
  day?: number;
  toolCount?: number;
  callId?: string;
  tool?: string;
  kind?: string;
  ok?: boolean;
  terminal?: boolean;
  latencySeconds?: number;
  requestHash?: string;
  responseHash?: string;
  argumentsHash?: string;
  argumentsText?: string;
  outputHash?: string;
  errorCode?: string;
  errorMessage?: string;
  action?: string;
  targetSeat?: number;
  content?: string;
  reasoning?: string;
  toolCallCount?: number;
  responseAttempt?: number;
  willRetry?: boolean;
  usage?: Record<string, number>;
  originalMessageCount?: number;
  modelMessageCount?: number;
  originalChars?: number;
  modelChars?: number;
  compactedToolGroups?: number;
  limitSatisfied?: boolean;
  modelHistoryHash?: string;
}

export interface AgentToolLoopTrace {
  events: AgentToolLoopEvent[];
  truncated: boolean;
  generationCount: number;
  generationFailureCount: number;
  toolCallCount: number;
  toolResultCount: number;
  terminalActionCount: number;
  historyCompactionCount: number;
  historyLimitMissCount: number;
}

export interface ActorResponseAttempt {
  attempt?: number;
  status?: string;
  errorType?: string;
  validationIssues: { path: string; code: string }[];
}

export interface ProtocolRecord {
  requestId: string;
  sequence: number;
  seat?: number;
  phase?: string;
  action?: string;
  legalActions: { action: string; targetSeats: number[]; canSkip: boolean }[];
  attempts: ActorResponseAttempt[];
  toolLoop?: AgentToolLoopTrace;
  response?: {
    accepted: boolean;
    issues: string[];
    decisionAction?: string;
    targetSeat?: number;
    speech?: string;
    reasoning?: string;
    modelCallId?: string;
    promptHash?: string;
    responseHash?: string;
    latencySeconds?: number;
    parseStatus?: string;
  };
  failure?: {
    errorType?: string;
    timeout: boolean;
    timeoutSeconds?: number;
    reason?: string;
    envelopeProduced: boolean;
  };
}

export function protocolRecords(items: RoomTraceItem[]): ProtocolRecord[] {
  const records = new Map<string, ProtocolRecord>();
  const sorted = [...items].sort((a, b) => Number(a.trace_seq || 0) - Number(b.trace_seq || 0));
  for (const item of sorted) {
    // The REST endpoint contains both event and decision rows. Tool-loop
    // evidence is private decision-trace data; never infer it from a public
    // event payload even if a malicious event happens to use the same `type`.
    if (item.kind !== "decision") continue;
    const payload = asRecord(item.payload);
    const kind = asString(payload.kind);
    if (kind === "agent_request") {
      const request = asRecord(payload.request);
      const requestId = asString(request.request_id);
      if (!requestId) continue;
      const legalActions = asArray(request.legal_actions).map((value) => {
        const legal = asRecord(value);
        return {
          action: asString(legal.action) || "unknown",
          targetSeats: asArray(legal.target_seats).map(asNumber).filter((value): value is number => value !== undefined),
          canSkip: Boolean(legal.can_skip),
        };
      });
      const previous = records.get(requestId);
      records.set(requestId, {
        requestId,
        sequence: Math.min(previous?.sequence ?? traceSequence(item), traceSequence(item)),
        seat: asNumber(request.seat) ?? previous?.seat,
        phase: asString(request.phase) ?? previous?.phase,
        action: asString(request.action_kind) ?? previous?.action,
        legalActions,
        attempts: previous?.attempts ?? [],
        response: previous?.response,
        failure: previous?.failure,
        toolLoop: previous?.toolLoop,
      });
    } else if (kind === "agent_response") {
      const requestId = asString(payload.request_id);
      if (!requestId) continue;
      const current = records.get(requestId) || emptyRecord(item, payload, requestId);
      const envelope = asRecord(payload.envelope);
      const decision = asRecord(envelope.decision);
      const validation = asRecord(payload.validation);
      current.response = {
        accepted: validation.valid === true,
        issues: asArray(validation.issues).map((issue) => asString(asRecord(issue).code)).filter((value): value is string => Boolean(value)),
        decisionAction: asString(decision.action),
        targetSeat: asNumber(decision.target_seat),
        speech: asSafeTraceText(decision.speech, MAX_TOOL_TRACE_TEXT),
        reasoning: asSafeTraceText(decision.reasoning, MAX_TOOL_TRACE_TEXT),
        modelCallId: asString(envelope.model_call_id),
        promptHash: asString(envelope.prompt_hash),
        responseHash: asString(envelope.response_hash),
        latencySeconds: asNumber(envelope.latency_seconds),
        parseStatus: asString(envelope.parse_status),
      };
      records.set(requestId, current);
    } else if (kind === "agent_response_failed" || kind === "agent_response_cancelled") {
      const requestId = asString(payload.request_id);
      if (!requestId) continue;
      const current = records.get(requestId) || emptyRecord(item, payload, requestId);
      const failure: Record<string, unknown> = kind === "agent_response_cancelled"
        ? { error_type: "DecisionCancelled", reason: asString(asRecord(payload.cancellation).reason) }
        : asRecord(payload.failure);
      current.failure = {
        errorType: asString(failure.error_type),
        timeout: failure.timeout === true,
        timeoutSeconds: asNumber(failure.timeout_seconds),
        reason: asSafeTraceText(failure.reason, MAX_TOOL_TRACE_ERROR),
        envelopeProduced: false,
      };
      const attempts = actorResponseAttempts(failure.llm_call_attempts);
      if (attempts.length > 0) current.attempts = attempts;
      records.set(requestId, current);
    } else if (kind === "agent_response_validation_failed") {
      const requestId = asString(payload.request_id);
      if (!requestId) continue;
      const current = records.get(requestId) || emptyRecord(item, payload, requestId);
      const envelope = asRecord(payload.envelope);
      const decision = asRecord(envelope.decision);
      const failure = asRecord(payload.failure);
      current.response = {
        accepted: false,
        issues: [],
        decisionAction: asString(decision.action),
        targetSeat: asNumber(decision.target_seat),
        speech: asSafeTraceText(decision.speech, MAX_TOOL_TRACE_TEXT),
        reasoning: asSafeTraceText(decision.reasoning, MAX_TOOL_TRACE_TEXT),
        modelCallId: asString(envelope.model_call_id),
        promptHash: asString(envelope.prompt_hash),
        responseHash: asString(envelope.response_hash),
        latencySeconds: asNumber(envelope.latency_seconds),
        parseStatus: asString(envelope.parse_status),
      };
      current.failure = {
        errorType: asString(failure.error_type),
        timeout: false,
        reason: asSafeTraceText(failure.reason, MAX_TOOL_TRACE_ERROR),
        envelopeProduced: true,
      };
      const attempts = actorResponseAttempts(failure.llm_call_attempts);
      if (attempts.length > 0) current.attempts = attempts;
      records.set(requestId, current);
    } else if (isAgentToolLoopEvent(payload.type) && isAdminTracePayload(payload)) {
      const requestId = asString(payload.request_id);
      if (!requestId) continue;
      const current = records.get(requestId) || emptyRecord(item, payload, requestId);
      const event = parseAgentToolLoopEvent(item, payload, payload.type, requestId);
      if (!event) continue;
      const toolLoop = current.toolLoop || emptyToolLoopTrace();
      // Counts describe the bounded projection, so they cannot claim more
      // rows than the console can retain or render.
      if (toolLoop.events.length < MAX_TOOL_LOOP_EVENTS) {
        toolLoop.events.push(event);
        if (event.type === "model_generation") toolLoop.generationCount += 1;
        if (event.type === "model_generation_failed") toolLoop.generationFailureCount += 1;
        if (event.type === "tool_call_requested") toolLoop.toolCallCount += 1;
        if (event.type === "tool_result") toolLoop.toolResultCount += 1;
        if (event.type === "agent_action_submitted") toolLoop.terminalActionCount += 1;
        if (event.type === "agent_history_compacted") {
          if ((event.compactedToolGroups ?? 0) > 0) toolLoop.historyCompactionCount += 1;
          if (event.limitSatisfied === false) toolLoop.historyLimitMissCount += 1;
        }
      } else {
        toolLoop.truncated = true;
      }
      current.toolLoop = toolLoop;
      records.set(requestId, current);
    } else if (asString(payload.type) === "decision_consumed") {
      const requestId = asString(payload.request_id);
      if (!requestId) continue;
      const attempts = actorResponseAttempts(asRecord(payload.llm_call).actor_response_attempts);
      if (attempts.length === 0) continue;
      const current = records.get(requestId) || emptyRecord(item, payload, requestId);
      current.attempts = attempts;
      records.set(requestId, current);
    }
  }
  return [...records.values()].sort((a, b) => a.sequence - b.sequence);
}

function actorResponseAttempts(value: unknown): ActorResponseAttempt[] {
  return asArray(value).slice(0, MAX_ACTOR_RESPONSE_ATTEMPTS).map((rawAttempt) => {
    const attempt = asRecord(rawAttempt);
    return {
      attempt: asAttemptNumber(attempt.attempt),
      status: asBoundedMetadata(attempt.status, 64),
      errorType: asBoundedMetadata(attempt.error_type, 96),
      validationIssues: asArray(attempt.validation_issues)
        .slice(0, MAX_VALIDATION_ISSUES_PER_ATTEMPT)
        .map((rawIssue) => {
          const issue = asRecord(rawIssue);
          return {
            path: asBoundedMetadata(issue.path, 160) || "<root>",
            code: asBoundedMetadata(issue.code, 64) || "invalid",
          };
        }),
    };
  });
}

function emptyRecord(
  item: RoomTraceItem,
  payload: Record<string, unknown>,
  requestId: string,
): ProtocolRecord {
  return {
    requestId,
    sequence: traceSequence(item),
    seat: asNumber(payload.seat),
    phase: asString(payload.phase),
    action: asString(payload.action),
    legalActions: [],
    attempts: [],
  };
}

function emptyToolLoopTrace(): AgentToolLoopTrace {
  return {
    events: [],
    truncated: false,
    generationCount: 0,
    generationFailureCount: 0,
    toolCallCount: 0,
    toolResultCount: 0,
    terminalActionCount: 0,
    historyCompactionCount: 0,
    historyLimitMissCount: 0,
  };
}

function isAgentToolLoopEvent(value: unknown): value is AgentToolLoopEventType {
  return value === "agent_turn_started"
    || value === "agent_history_compacted"
    || value === "model_generation"
    || value === "model_generation_failed"
    || value === "tool_call_requested"
    || value === "tool_result"
    || value === "agent_action_submitted";
}

function isAdminTracePayload(payload: Record<string, unknown>): boolean {
  // Older admin artifacts did not carry visibility metadata on every row.
  // Explicitly public/private rows are rejected; missing markers remain
  // compatible with those older, already-authorized `/trace` responses.
  const visibility = asString(payload.visibility);
  const audience = asString(payload.audience);
  return (visibility === undefined || visibility === "admin")
    && (audience === undefined || audience === "admin");
}

function parseAgentToolLoopEvent(
  item: RoomTraceItem,
  payload: Record<string, unknown>,
  type: AgentToolLoopEventType,
  requestId: string,
): AgentToolLoopEvent | undefined {
  const event: AgentToolLoopEvent = {
    type,
    sequence: traceSequence(item),
    requestId,
    sessionId: asBoundedMetadata(payload.session_id, MAX_TOOL_TRACE_METADATA),
    turnId: asBoundedMetadata(payload.turn_id, MAX_TOOL_TRACE_METADATA),
    seat: asPositiveInteger(payload.seat),
    stateVersion: asBoundedInteger(payload.state_version),
    step: asBoundedInteger(payload.step),
    phase: asBoundedMetadata(payload.phase, 64),
    day: asBoundedInteger(payload.day),
    callId: asBoundedMetadata(payload.call_id, MAX_TOOL_TRACE_METADATA),
    tool: asBoundedMetadata(payload.tool, 96),
  };

  if (type === "agent_turn_started") {
    event.toolCount = asBoundedInteger(payload.tool_count);
    return event;
  }
  if (type === "agent_history_compacted") {
    event.originalMessageCount = asBoundedInteger(payload.original_message_count);
    event.modelMessageCount = asBoundedInteger(payload.model_message_count);
    event.originalChars = asBoundedInteger(payload.original_chars, MAX_USAGE_VALUE);
    event.modelChars = asBoundedInteger(payload.model_chars, MAX_USAGE_VALUE);
    event.compactedToolGroups = asBoundedInteger(payload.compacted_tool_groups);
    event.limitSatisfied = asBoolean(payload.limit_satisfied);
    event.modelHistoryHash = asBoundedMetadata(payload.model_history_hash, MAX_TOOL_TRACE_METADATA);
    return event;
  }
  if (type === "model_generation") {
    event.content = asSafeTraceText(payload.content, MAX_TOOL_TRACE_TEXT);
    event.reasoning = asSafeTraceText(payload.reasoning, MAX_TOOL_TRACE_TEXT);
    event.latencySeconds = asBoundedSeconds(payload.latency);
    event.toolCallCount = asBoundedInteger(payload.tool_call_count);
    event.usage = boundedUsage(payload.usage);
    event.callId = asBoundedMetadata(payload.call_id, MAX_TOOL_TRACE_METADATA);
    event.requestHash = asBoundedMetadata(payload.request_hash, MAX_TOOL_TRACE_METADATA);
    event.responseHash = asBoundedMetadata(payload.response_hash, MAX_TOOL_TRACE_METADATA);
    return event;
  }
  if (type === "model_generation_failed") {
    event.responseAttempt = asPositiveInteger(payload.response_attempt);
    event.willRetry = asBoolean(payload.will_retry);
    event.errorCode = asBoundedMetadata(payload.error_type, 96);
    event.requestHash = asBoundedMetadata(payload.request_hash, MAX_TOOL_TRACE_METADATA);
    return event;
  }
  if (type === "tool_call_requested") {
    event.argumentsHash = asBoundedMetadata(payload.arguments_hash, MAX_TOOL_TRACE_METADATA);
    event.argumentsText = boundedToolArguments(payload.arguments);
    return event;
  }
  if (type === "tool_result") {
    event.kind = asBoundedMetadata(payload.kind, 64);
    event.ok = asBoolean(payload.ok);
    event.terminal = asBoolean(payload.terminal);
    event.latencySeconds = asBoundedSeconds(payload.latency);
    event.outputHash = asBoundedMetadata(payload.output_hash, MAX_TOOL_TRACE_METADATA);
    const error = asRecord(payload.error);
    event.errorCode = asBoundedMetadata(error.code, 96)
      ?? asBoundedMetadata(payload.error_code, 96);
    event.errorMessage = asSafeTraceText(error.message, MAX_TOOL_TRACE_ERROR);
    // Do not retain `output` or `error.details`: those are arbitrary private
    // objects and may contain facts, notes, or provider material not needed to
    // prove the call chain.
    return event;
  }

  // agent_action_submitted deliberately keeps only the environment-bound
  // action/target, never the full Decision (which may contain speech,
  // reasoning, claims, or private messages).
  event.action = asBoundedMetadata(payload.action, 96);
  const decision = asRecord(payload.decision);
  event.targetSeat = asPositiveInteger(decision.target_seat);
  if (!event.action) event.action = asBoundedMetadata(decision.action, 96);
  return event;
}

function boundedUsage(value: unknown): Record<string, number> | undefined {
  const raw = asRecord(value);
  const usage: Record<string, number> = {};
  for (const key of [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
  ]) {
    const number = asBoundedInteger(raw[key], MAX_USAGE_VALUE);
    if (number !== undefined) usage[key] = number;
  }
  return Object.keys(usage).length > 0 ? usage : undefined;
}

function boundedToolArguments(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  const sanitized = sanitizeToolArgument(value, 0);
  let serialized: string;
  try {
    serialized = JSON.stringify(sanitized, null, 2);
  } catch {
    return undefined;
  }
  return serialized.length > MAX_TOOL_ARGUMENT_TEXT
    ? `${serialized.slice(0, MAX_TOOL_ARGUMENT_TEXT)}\n[truncated]`
    : serialized;
}

function sanitizeToolArgument(value: unknown, depth: number): unknown {
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string") return redactCredentialText(value).slice(0, MAX_TOOL_ARGUMENT_STRING);
  if (depth >= MAX_TOOL_ARGUMENT_DEPTH) return "[truncated:depth]";
  if (Array.isArray(value)) {
    const items = value
      .slice(0, MAX_TOOL_ARGUMENT_ITEMS)
      .map((item) => sanitizeToolArgument(item, depth + 1));
    return value.length > MAX_TOOL_ARGUMENT_ITEMS
      ? { items, omittedItems: value.length - MAX_TOOL_ARGUMENT_ITEMS }
      : items;
  }
  if (typeof value !== "object") return String(value).slice(0, MAX_TOOL_ARGUMENT_STRING);

  const result: Record<string, unknown> = {};
  const entries = Object.entries(value as Record<string, unknown>);
  for (const [rawKey, item] of entries.slice(0, MAX_TOOL_ARGUMENT_KEYS)) {
    const key = rawKey.slice(0, 160);
    // Inspect the complete key before truncating it for display; otherwise a
    // long prefix could hide a sensitive suffix such as `_access_token`.
    const normalized = rawKey.toLowerCase().replace(/-/g, "_");
    result[key] = SENSITIVE_KEY_FRAGMENTS.some((fragment) => normalized.includes(fragment))
      ? REDACTED
      : sanitizeToolArgument(item, depth + 1);
  }
  if (entries.length > MAX_TOOL_ARGUMENT_KEYS) {
    result._omittedKeyCount = entries.length - MAX_TOOL_ARGUMENT_KEYS;
  }
  return result;
}

function redactCredentialText(value: string): string {
  return value
    .replace(/\bBearer\s+[A-Za-z0-9._~+/=-]+/gi, `Bearer ${REDACTED}`)
    .replace(/\bsk-[A-Za-z0-9_-]{8,}\b/gi, REDACTED)
    .replace(
      /(^|[^A-Za-z0-9_])(["']?(?:access[_-]?token|refresh[_-]?token|id[_-]?token|session[_-]?token|client[_-]?secret|private[_-]?key)["']?\s*[:=]\s*["']?)[^\s,;"']{8,}/gi,
      `$1$2${REDACTED}`,
    )
    .replace(/\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b/g, REDACTED)
    .replace(
      /\b[A-Za-z0-9._~+/=-]*(?:api[_-]?key|secret|password)[A-Za-z0-9._~+/=-]{6,}\b/gi,
      REDACTED,
    );
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function asBoundedMetadata(value: unknown, maxLength: number): string | undefined {
  if (typeof value !== "string") return undefined;
  const cleaned = redactCredentialText(value).replace(/[\x00-\x1f\x7f]/g, " ").trim();
  return cleaned ? cleaned.slice(0, maxLength) : undefined;
}

function asSafeTraceText(value: unknown, maxLength: number): string | undefined {
  if (typeof value !== "string") return undefined;
  return asBoundedText(redactCredentialText(value), maxLength);
}

function asBoundedText(value: unknown, maxLength: number): string | undefined {
  if (typeof value !== "string") return undefined;
  const cleaned = value
    .replace(/\r\n?/g, "\n")
    .replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, " ")
    .trim();
  return cleaned ? cleaned.slice(0, maxLength) : undefined;
}

function asBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function asBoundedInteger(value: unknown, max = 999_999): number | undefined {
  const parsed = asNumber(value);
  return parsed !== undefined && Number.isInteger(parsed) && parsed >= 0 && parsed <= max
    ? parsed
    : undefined;
}

function asPositiveInteger(value: unknown): number | undefined {
  const parsed = asBoundedInteger(value, 999);
  return parsed !== undefined && parsed >= 1 ? parsed : undefined;
}

function asBoundedSeconds(value: unknown): number | undefined {
  const parsed = asNumber(value);
  return parsed !== undefined && parsed >= 0 && parsed <= 86_400 ? parsed : undefined;
}

function traceSequence(item: RoomTraceItem): number {
  const parsed = asNumber(item.trace_seq) ?? asNumber(item.idx);
  return parsed !== undefined && parsed >= 0 ? parsed : 0;
}

function asNumber(value: unknown): number | undefined {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function asAttemptNumber(value: unknown): number | undefined {
  const parsed = asNumber(value);
  return parsed !== undefined && Number.isInteger(parsed) && parsed >= 1 && parsed <= 999
    ? parsed
    : undefined;
}
