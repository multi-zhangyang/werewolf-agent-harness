// REST API client —— 真实后端 /api/* 调用,无 mock。
import type { RoomInfo, ProviderMeta, ModelConfigDTO } from "./types";

const BASE = "/api";

async function jget(url: string): Promise<any> {
  const r = await fetch(url);
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
}

export interface RoomAuth {
  admin_token?: string;
  seat_tokens?: Record<string, string>;
}

export interface CreateRoomResponse extends RoomAuth {
  room_id: string;
  status: string;
  players: any[];
  human_seats?: number[];
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

async function jgetAuth(url: string, token?: string): Promise<any> {
  const r = await fetch(url, { headers: authHeaders(token) });
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

export async function getReplay(roomId: string, adminToken?: string): Promise<any> {
  return jgetAuth(`${BASE}/rooms/${roomId}/replay`, adminToken);
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
