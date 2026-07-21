// WebSocket client —— 连真实后端 ws://host/ws/{room_id}?seat=&mode=,无 mock。
// 断线自动重连(指数退避)。心跳 ping/pong。
import type { GameEvent } from "./types";

export interface WSHandlers {
  onEvent: (ev: GameEvent) => void;
  onOpen: () => void;
  onClose: (info: WSCloseInfo) => void;
  onError: (e: string) => void;
}

export interface WSCloseInfo {
  code: number;
  reason: string;
  wasClean: boolean;
  willReconnect: boolean;
}

export class GameSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: WSHandlers;
  private reconnectDelay = 1000;
  private shouldReconnect = true;
  private reconnectTimer: number | null = null;
  private pingTimer: number | null = null;
  private pongTimer: number | null = null;
  private lastDeliverySeq = 0;
  private lastDeliveryId: string | null = null;
  private recentDeliveryIds = new Map<string, number>();
  private recentDeliverySequences = new Map<number, string>();
  private streamId: string | null = null;
  private openingResumeCursor: number | null = null;
  private capabilityToken: string | null;
  private socketGeneration = 0;

  constructor(url: string, handlers: WSHandlers, capabilityToken?: string) {
    this.url = url;
    this.handlers = handlers;
    this.capabilityToken = capabilityToken?.trim() || null;
  }

  connect() {
    this.shouldReconnect = true;
    this._clearReconnectTimer();
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      return;
    }
    this._open();
  }

  private _open() {
    if (!this.shouldReconnect) return;
    this.reconnectTimer = null;
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
      return;
    }
    const generation = ++this.socketGeneration;
    let ws: WebSocket;
    try {
      this.openingResumeCursor = this.lastDeliverySeq > 0 ? this.lastDeliverySeq : null;
      const protocols = ["werewolf.v1"];
      if (this.capabilityToken) protocols.push(`werewolf.cap.${this.capabilityToken}`);
      ws = new WebSocket(this._urlForOpen(), protocols);
      this.ws = ws;
    } catch (e) {
      this.handlers.onError(String(e));
      this._scheduleReconnect();
      return;
    }
    ws.onopen = () => {
      if (!this._isCurrentSocket(ws, generation)) return;
      this.handlers.onOpen();
      this._startPing();
    };
    ws.onmessage = (msg) => {
      if (!this._isCurrentSocket(ws, generation)) return;
      if (msg.data === "pong") {
        this._clearPongTimer();
        return;
      }
      try {
        const ev = JSON.parse(msg.data) as GameEvent;
        if (ev.type === "snapshot") {
          if (!this._acceptSnapshot(ev)) return;
        } else if (!this._acceptDelivery(ev)) {
          return;
        }
        this.handlers.onEvent(ev);
      } catch (e) {
        console.error("WS 解析失败:", e, msg.data);
      }
    };
    ws.onclose = (event) => {
      // Every callback is bound to the socket generation that created it. A
      // late close from an older connection must not tear down a replacement,
      // clear its heartbeat, or reset the reducer's connectivity state.
      if (!this._isCurrentSocket(ws, generation)) return;
      this.ws = null;
      this._stopPing();
      const willReconnect = this._reconnectForClose(event);
      this.handlers.onClose({
        code: event.code,
        reason: event.reason || "",
        wasClean: event.wasClean,
        willReconnect,
      });
      if (willReconnect) this._scheduleReconnect();
    };
    ws.onerror = () => {
      if (!this._isCurrentSocket(ws, generation)) return;
      this.handlers.onError("WebSocket 连接错误");
    };
  }

  private _isCurrentSocket(ws: WebSocket, generation: number): boolean {
    return this.ws === ws && this.socketGeneration === generation;
  }

  private _reconnectForClose(event: CloseEvent): boolean {
    if (!this.shouldReconnect) return false;
    // 4409 is overloaded by the server: a retained-history gap is recoverable
    // with a fresh cursor, while replay-before-termination is a terminal mode
    // error and must not spin forever.
    if (event.code === 4409) {
      if (/^history gap\b/i.test(event.reason || "")) {
        this._resetDeliveryCursor();
        return true;
      }
      this.shouldReconnect = false;
      this._clearReconnectTimer();
      return false;
    }
    // These indicate an invalid request, revoked capability, missing room, or
    // a client protocol violation. Retrying cannot repair the current URL or
    // credentials; expose the close to the UI and wait for user navigation.
    if ([1000, 1008, 4002, 4400, 4403, 4404, 4410, 4429].includes(event.code)) {
      this.shouldReconnect = false;
      this._clearReconnectTimer();
      return false;
    }
    return true;
  }

  private _urlForOpen(): string {
    const url = new URL(this.url);
    if (this.openingResumeCursor !== null) {
      url.searchParams.set("since", String(this.openingResumeCursor));
    } else {
      url.searchParams.delete("since");
    }
    return url.toString();
  }

  private _acceptSnapshot(ev: Extract<GameEvent, { type: "snapshot" }>): boolean {
    if (
      typeof ev.stream_id !== "string"
      || !ev.stream_id
      || !Number.isSafeInteger(ev.cursor)
      || (ev.cursor ?? -1) < 0
      || !Number.isSafeInteger(ev.replay_from)
      || (ev.replay_from ?? 0) < 1
    ) {
      return this._protocolFailure("WebSocket snapshot cursor 无效");
    }

    if (this.streamId !== null && this.streamId !== ev.stream_id && this.openingResumeCursor !== null) {
      this._resetDeliveryCursor();
      return this._recoverableProtocolFailure("WebSocket delivery stream 已更换，正在重新同步");
    }

    const resumedFrom = ev.resumed_from;
    if (resumedFrom !== null && resumedFrom !== undefined) {
      if (
        !Number.isSafeInteger(resumedFrom)
        || resumedFrom < 0
        || resumedFrom !== this.openingResumeCursor
        || resumedFrom > (ev.cursor ?? -1)
      ) {
        return this._protocolFailure("WebSocket resume cursor 与请求不匹配");
      }
      this.lastDeliverySeq = resumedFrom;
    } else {
      const baseline = (ev.replay_from ?? 1) - 1;
      if (baseline > (ev.cursor ?? -1)) {
        return this._protocolFailure("WebSocket replay 起点超出快照 cursor");
      }
      this.lastDeliverySeq = baseline;
      this.lastDeliveryId = null;
      this.recentDeliveryIds.clear();
      this.recentDeliverySequences.clear();
    }
    this.streamId = ev.stream_id;
    // Reset transport backoff only after the application-level snapshot is
    // valid. A TCP/WebSocket open followed by an immediate protocol failure is
    // not a healthy connection and must continue backing off.
    this.reconnectDelay = 1000;
    return true;
  }

  private _acceptDelivery(ev: GameEvent): boolean {
    if (ev.delivery_seq === undefined && ev.delivery_id === undefined) return true;
    if (
      !Number.isSafeInteger(ev.delivery_seq)
      || (ev.delivery_seq ?? 0) <= 0
      || typeof ev.delivery_id !== "string"
      || !ev.delivery_id
    ) {
      return this._protocolFailure("WebSocket delivery metadata 无效");
    }
    const seq = ev.delivery_seq as number;
    const knownSeq = this.recentDeliveryIds.get(ev.delivery_id);
    if (knownSeq !== undefined && knownSeq !== seq) {
      return this._protocolFailure("WebSocket delivery id 重复但序号不同");
    }
    const knownId = this.recentDeliverySequences.get(seq);
    if (knownId !== undefined && knownId !== ev.delivery_id) {
      return this._protocolFailure("WebSocket delivery 序号重复但 id 不同");
    }
    if (seq <= this.lastDeliverySeq) {
      // Replayed/live overlap is harmless: the stable sequence is the
      // idempotency key and the reducer sees the event at most once.
      if (seq === this.lastDeliverySeq && ev.delivery_id !== this.lastDeliveryId) {
        return this._protocolFailure("WebSocket delivery 序号重复但 id 不同");
      }
      return false;
    }
    if (seq !== this.lastDeliverySeq + 1) {
      const expected = this.lastDeliverySeq + 1;
      this._resetDeliveryCursor();
      return this._recoverableProtocolFailure(
        `WebSocket delivery 存在缺口: expected=${expected}, received=${seq}`,
      );
    }
    this.lastDeliverySeq = seq;
    this.lastDeliveryId = ev.delivery_id;
    this.recentDeliveryIds.set(ev.delivery_id, seq);
    this.recentDeliverySequences.set(seq, ev.delivery_id);
    while (this.recentDeliveryIds.size > 512) {
      const oldest = this.recentDeliveryIds.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      this.recentDeliveryIds.delete(oldest);
    }
    while (this.recentDeliverySequences.size > 512) {
      const oldest = this.recentDeliverySequences.keys().next().value as number | undefined;
      if (oldest === undefined) break;
      this.recentDeliverySequences.delete(oldest);
    }
    return true;
  }

  private _protocolFailure(message: string): false {
    this.handlers.onError(message);
    const ws = this.ws;
    if (ws && ws.readyState !== WebSocket.CLOSED) {
      try {
        ws.close(4002, "delivery protocol error");
      } catch {
        // A browser can race a transport close with a protocol failure. The
        // error has already been surfaced; do not let the callback throw.
      }
    }
    return false;
  }

  private _recoverableProtocolFailure(message: string): false {
    this.handlers.onError(message);
    this._resetDeliveryCursor();
    const ws = this.ws;
    if (ws && ws.readyState !== WebSocket.CLOSED) {
      try {
        // 4001 is a client-side resync request. It is intentionally outside
        // the server's permanent-error set, so onclose schedules one fresh
        // snapshot without retaining a now-invalid cursor.
        ws.close(4001, "delivery resync");
        return false;
      } catch {
        // Fall through to a timer if close itself races the transport.
      }
    }
    this._scheduleReconnect();
    return false;
  }

  private _resetDeliveryCursor() {
    this.lastDeliverySeq = 0;
    this.lastDeliveryId = null;
    this.recentDeliveryIds.clear();
    this.recentDeliverySequences.clear();
    this.streamId = null;
    this.openingResumeCursor = null;
  }

  private _scheduleReconnect() {
    if (!this.shouldReconnect || this.reconnectTimer !== null) return;
    const delay = Math.min(this.reconnectDelay, 10000);
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      if (this.shouldReconnect) this._open();
    }, delay);
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, 10000);
  }

  private _clearReconnectTimer() {
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private _startPing() {
    this._stopPing();
    this.pingTimer = window.setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send("ping");
        this._armPongTimer();
      }
    }, 25000);
  }

  private _armPongTimer() {
    this._clearPongTimer();
    this.pongTimer = window.setTimeout(() => {
      this.pongTimer = null;
      this._recoverableProtocolFailure("WebSocket 心跳超时，正在重新连接");
    }, 10000);
  }

  private _clearPongTimer() {
    if (this.pongTimer !== null) {
      window.clearTimeout(this.pongTimer);
      this.pongTimer = null;
    }
  }

  private _stopPing() {
    if (this.pingTimer !== null) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
    this._clearPongTimer();
  }

  send(payload: object): boolean {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
      return true;
    }
    return false;
  }

  close() {
    this.shouldReconnect = false;
    this._clearReconnectTimer();
    this._stopPing();
    const ws = this.ws;
    // Invalidate callbacks before asking the browser to close. Some WebSocket
    // implementations deliver onclose synchronously, and that close is a
    // caller-intentional teardown rather than a connection fault.
    this.socketGeneration += 1;
    this.ws = null;
    if (ws) {
      try {
        ws.close();
      } catch {
        // Closing an already-closed browser socket is harmless.
      }
    }
    this.capabilityToken = null;
  }
}

export function buildWsUrl(roomId: string, seat: number | null, mode: string, _token?: string): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams();
  params.set("mode", mode);
  if (seat !== null && seat !== undefined) params.set("seat", String(seat));
  return `${proto}//${location.host}/ws/${roomId}?${params.toString()}`;
}
