// WebSocket client —— 连真实后端 ws://host/ws/{room_id}?seat=&mode=,无 mock。
// 断线自动重连(指数退避)。心跳 ping/pong。
import type { GameEvent } from "./types";

export interface WSHandlers {
  onEvent: (ev: GameEvent) => void;
  onOpen: () => void;
  onClose: () => void;
  onError: (e: string) => void;
}

export class GameSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: WSHandlers;
  private reconnectDelay = 1000;
  private shouldReconnect = true;
  private pingTimer: number | null = null;

  constructor(url: string, handlers: WSHandlers) {
    this.url = url;
    this.handlers = handlers;
  }

  connect() {
    this.shouldReconnect = true;
    this._open();
  }

  private _open() {
    try {
      this.ws = new WebSocket(this.url);
    } catch (e) {
      this.handlers.onError(String(e));
      this._scheduleReconnect();
      return;
    }
    this.ws.onopen = () => {
      this.reconnectDelay = 1000;
      this.handlers.onOpen();
      this._startPing();
    };
    this.ws.onmessage = (msg) => {
      if (msg.data === "pong") return;
      try {
        const ev = JSON.parse(msg.data) as GameEvent;
        this.handlers.onEvent(ev);
      } catch (e) {
        console.error("WS 解析失败:", e, msg.data);
      }
    };
    this.ws.onclose = () => {
      this.handlers.onClose();
      this._stopPing();
      if (this.shouldReconnect) this._scheduleReconnect();
    };
    this.ws.onerror = () => {
      this.handlers.onError("WebSocket 连接错误");
    };
  }

  private _scheduleReconnect() {
    if (!this.shouldReconnect) return;
    const delay = Math.min(this.reconnectDelay, 10000);
    window.setTimeout(() => this._open(), delay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, 10000);
  }

  private _startPing() {
    this._stopPing();
    this.pingTimer = window.setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send("ping");
      }
    }, 25000);
  }

  private _stopPing() {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  send(payload: object) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }

  close() {
    this.shouldReconnect = false;
    this._stopPing();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export function buildWsUrl(roomId: string, seat: number | null, mode: string, token?: string): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams();
  params.set("mode", mode);
  if (seat !== null && seat !== undefined) params.set("seat", String(seat));
  if (token) params.set("token", token);
  return `${proto}//${location.host}/ws/${roomId}?${params.toString()}`;
}
