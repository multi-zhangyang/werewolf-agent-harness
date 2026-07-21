import type { GameState } from "./store";
import { makeInitial } from "./store";
import type { ReplayPayload, ReplayTranscriptEntry, RoomTraceItem } from "./api";

export interface ReplayTimelineRow {
  seq: number;
  kind: string;
  phase?: string;
  day?: number;
  seat?: number;
  name?: string;
  text: string;
  payload: Record<string, unknown>;
}

export interface ReplayFilters {
  phase: string;
  kind: string;
  seat: string;
}

/**
 * Replay is an observation surface. Freeze the server response before any UI
 * projection so playhead/filter state can never mutate the evidence payload.
 */
export function freezeReplayPayload<T>(value: T): Readonly<T> {
  return deepFreeze(value) as Readonly<T>;
}

function deepFreeze(value: unknown): unknown {
  if (value === null || typeof value !== "object" || Object.isFrozen(value)) return value;
  if (Array.isArray(value)) {
    for (const item of value) deepFreeze(item);
  } else {
    for (const item of Object.values(value)) deepFreeze(item);
  }
  return Object.freeze(value);
}

export function replayTimeline(payload: ReplayPayload): ReplayTimelineRow[] {
  const transcriptRows = payload.transcript?.entries || [];
  const eventRows = transcriptRows
    .filter((entry) => entry.kind === "event")
    .map((entry, index) => timelineRowFromTranscript(entry, index));
  if (eventRows.length > 0) return eventRows;

  return (payload.events || []).map((raw, index) => timelineRowFromPayload(raw, index + 1));
}

export function replayDecisionTrace(payload: ReplayPayload): RoomTraceItem[] {
  const entries = payload.transcript?.entries || [];
  return entries
    .filter((entry) => entry.kind === "decision")
    .map((entry) => {
      const payloadValue = asRecord(entry.payload);
      const traceSeq = asNumber(payloadValue._trace_seq) ?? entry.seq;
      return {
        kind: "decision" as const,
        idx: entry.seq,
        trace_seq: traceSeq,
        ts: entry.ts_monotonic ?? null,
        payload: payloadValue,
      };
    });
}

export function filterReplayRows(
  rows: ReplayTimelineRow[],
  filters: ReplayFilters,
  playheadIndex: number,
): ReplayTimelineRow[] {
  const playheadSeq = rows[Math.max(0, Math.min(playheadIndex, rows.length - 1))]?.seq ?? Number.MAX_SAFE_INTEGER;
  return rows.filter((row) => {
    if (row.seq > playheadSeq) return false;
    if (filters.phase !== "all" && (row.phase || "unknown") !== filters.phase) return false;
    if (filters.kind !== "all" && row.kind !== filters.kind) return false;
    if (filters.seat !== "all" && String(row.seat ?? "unknown") !== filters.seat) return false;
    return true;
  });
}

export function replayFilterOptions(rows: ReplayTimelineRow[], field: "phase" | "kind" | "seat"): string[] {
  const values = new Set<string>();
  for (const row of rows) {
    const value = field === "phase"
      ? row.phase || "unknown"
      : field === "kind"
        ? row.kind
        : row.seat == null ? "unknown" : String(row.seat);
    values.add(value);
  }
  return [...values].sort((a, b) => a.localeCompare(b, "zh-CN", { numeric: true }));
}

export function replaySummaryState(payload: ReplayPayload): GameState {
  const state = makeInitial();
  state.connected = true;
  state.status = payload.status || "ended";
  state.phase = payload.phase || "ended";
  state.day = payload.day ?? 0;
  state.winner = payload.winner ?? null;
  state.mode = "replay";
  state.analysis = payload.analysis ?? undefined;
  state.seats = (payload.players || []).map((player) => ({
    seat: player.seat,
    name: player.name,
    alive: player.alive,
    role: player.role,
    team: player.team,
    isSpeaking: false,
  }));
  return state;
}

function timelineRowFromTranscript(entry: ReplayTranscriptEntry, index: number): ReplayTimelineRow {
  return timelineRowFromPayload(entry.payload, entry.seq || index + 1, entry);
}

function timelineRowFromPayload(
  raw: Record<string, unknown>,
  seq: number,
  entry?: ReplayTranscriptEntry,
): ReplayTimelineRow {
  const payload = asRecord(raw);
  const kind = asString(payload.type) || "event";
  const phase = asString(entry?.phase) || asString(payload.phase);
  const day = asNumber(entry?.day) ?? asNumber(payload.day);
  const seat = asNumber(entry?.seat) ?? asNumber(payload.seat);
  const text = asString(payload.text) ?? asString(payload.message) ?? kind;
  return {
    seq,
    kind,
    phase,
    day,
    seat,
    name: asString(payload.name),
    text,
    payload,
  };
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
