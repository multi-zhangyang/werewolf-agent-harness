// REST API client —— 真实后端 /api/* 调用,无 mock。
import type { RoomInfo, ProviderMeta, ModelConfigDTO, HarnessSeeds, GameAnalysis } from "./types";

const BASE = "/api";

async function jget(url: string, signal?: AbortSignal): Promise<any> {
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

async function jpost(url: string, body?: any): Promise<any> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export interface CreateRoomPayload {
  player_names: string[];
  model_config?: Partial<ModelConfigDTO>;
  human_seats?: number[];
  experiment_seed?: number;
}

export interface RoomAuth {
  admin_token?: string;
  seat_tokens?: Record<string, string>;
}

export interface CreateRoomResponse extends RoomAuth, HarnessSeeds {
  room_id: string;
  status: string;
  players: any[];
  human_seats?: number[];
}

export interface RoomTraceItem {
  kind: "event" | "decision";
  idx: number;
  trace_seq?: number | null;
  ts?: number | null;
  payload: Record<string, unknown>;
}

export interface RoomTrace extends HarnessSeeds {
  room_id: string;
  status: string;
  end_reason?: string | null;
  error?: string | null;
  phase?: string;
  day?: number;
  winner?: string | null;
  event_count: number;
  decision_trace_count?: number;
  trace_seq?: number;
  since?: number | null;
  incremental?: boolean;
  trace: RoomTraceItem[];
  run_spec?: Record<string, unknown> | null;
  core_run_spec?: Record<string, unknown> | null;
  transcript?: Record<string, unknown> | null;
}

export interface ReplayTranscriptEntry {
  schema_version?: string;
  run_id?: string;
  seq: number;
  kind: string;
  ts_monotonic?: number;
  day?: number | null;
  phase?: string | null;
  seat?: number | null;
  visibility?: string | null;
  source_idx?: number | null;
  payload_hash?: string;
  payload: Record<string, unknown>;
}

export interface ReplayTranscript {
  schema_version?: string;
  run_id?: string;
  metadata?: Record<string, unknown>;
  counts_by_kind?: Record<string, number>;
  stable_digest?: string;
  entries: ReplayTranscriptEntry[];
}

export interface ReplayPlayer {
  seat: number;
  name: string;
  alive: boolean;
  role?: string;
  team?: string;
}

export interface ReplayPayload extends HarnessSeeds {
  room_id: string;
  status: string;
  end_reason?: string | null;
  error?: string | null;
  phase?: string;
  day?: number;
  winner?: string | null;
  human_seats?: number[];
  events: Record<string, unknown>[];
  analysis?: GameAnalysis | null;
  players: ReplayPlayer[];
  run_spec?: Record<string, unknown> | null;
  core_run_spec?: Record<string, unknown> | null;
  transcript?: ReplayTranscript | null;
}

function authHeaders(token?: string): Record<string, string> {
  return token ? { "X-Room-Token": token } : {};
}

async function jpostAuth(url: string, body: any | undefined, token?: string): Promise<any> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

async function jgetAuth(url: string, token?: string, signal?: AbortSignal): Promise<any> {
  const r = await fetch(url, { headers: authHeaders(token), signal });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export async function createRoom(payload: CreateRoomPayload): Promise<CreateRoomResponse> {
  return jpost(`${BASE}/rooms`, payload);
}

export async function getRoom(roomId: string): Promise<RoomInfo> {
  return jget(`${BASE}/rooms/${roomId}`);
}

export async function startRoom(roomId: string, adminToken?: string): Promise<{ room_id: string; status: string }> {
  return jpostAuth(`${BASE}/rooms/${roomId}/start`, undefined, adminToken);
}

export async function getReplay(
  roomId: string,
  adminToken?: string,
  signal?: AbortSignal,
): Promise<ReplayPayload> {
  return jgetAuth(`${BASE}/rooms/${roomId}/replay`, adminToken, signal);
}

export async function getTrace(
  roomId: string,
  adminToken?: string,
  since?: number | null,
  signal?: AbortSignal,
): Promise<RoomTrace> {
  const query = since == null ? "" : `?since=${encodeURIComponent(String(since))}`;
  return jgetAuth(`${BASE}/rooms/${roomId}/trace${query}`, adminToken, signal);
}

export async function getProviders(): Promise<Record<string, ProviderMeta>> {
  return jget(`${BASE}/providers`);
}

export async function getConfig(): Promise<Partial<ModelConfigDTO>> {
  return jget(`${BASE}/config`);
}

export async function setSeatModelConfig(roomId: string, seat: number, cfg: Partial<ModelConfigDTO>, adminToken?: string): Promise<{ ok: boolean }> {
  return jpostAuth(`${BASE}/rooms/${roomId}/seats/${seat}/model_config`, cfg, adminToken);
}
