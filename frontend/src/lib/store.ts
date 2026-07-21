// 游戏状态 reducer —— 把真实后端 WS 事件流归约成前端 store。
// 零 mock:所有状态来自真实事件。snapshot 提供初始投影,后续事件增量更新。
import type { GameEvent, SnapshotView, DeathRecord, Claim, GameAnalysis } from "./types";

// 内部 meta 事件(WS 连接生命周期 + store 重置),非后端事件
export type MetaEvent =
  | { type: "__open__" }
  | { type: "__close__"; code?: number; reason?: string; willReconnect?: boolean }
  | { type: "__socket_error__"; message: string }
  | { type: "__manual_reconnect__" }
  | { type: "__reset__" }
  | { type: "__context__"; mySeat: number | null; mode: string };

export type StoreEvent = GameEvent | MetaEvent;

export interface LogEntry {
  id: number;
  kind: string; // phase / speech / vote / death / hunter / last_words / failed / system / night_resolved
  day: number;
  text: string;
  seat?: number;
  targetSeat?: number;
  claim?: Claim;
  ts: number;
  bid?: number;
  // Public relationship metadata returned by the same agent decision.
  replyTo?: number | null;
  accuses?: number[];
  pk?: boolean;
}

export interface SeatState {
  seat: number;
  name: string;
  alive: boolean;
  role?: string;
  team?: string;
  persona?: string;
  isSpeaking: boolean;
  votedTarget?: number; // 本轮投票目标
  lastSpeech?: string;
  deathReason?: string;
  deathDay?: number;
}

export interface GameState {
  connected: boolean;
  socketClose?: {
    code: number;
    reason: string;
    retryableByUser: boolean;
  };
  status: string; // waiting / running / ended
  phase: string;
  day: number;
  winner: string | null;
  seats: SeatState[];
  mySeat: number | null;
  mode: string;
  // 日志流(按时间序)
  log: LogEntry[];
  // 投票(本轮 voterSeat -> targetSeat)
  votes: Record<number, number>;
  // 当前人类操作请求(play 模式)
  pendingHuman?: {
    requestId: string;
    actionType: string;
    context: Record<string, unknown>;
    deadline: number;
    timeoutMs: number;
    day?: number;
    phase?: string;
  };
  // god/admin factual provider counters
  llmStats?: Record<string, number>;
  analysis?: GameAnalysis;
  error?: string;
  // UI 提示
  lastDeaths: DeathRecord[];
  speakingSeat: number | null;
}

let logId = 0;

export function makeInitial(): GameState {
  return {
    connected: false,
    status: "waiting",
    phase: "setup",
    day: 0,
    winner: null,
    seats: [],
    mySeat: null,
    mode: "spectate",
    log: [],
    votes: {},
    lastDeaths: [],
    speakingSeat: null,
  };
}

function seatsFromView(view: SnapshotView, prev: SeatState[]): SeatState[] {
  const full = view.players_full;
  const pub = view.players || [];
  const personas = view.personas || {};
  const map = new Map<number, SeatState>();
  for (const s of prev) map.set(s.seat, s);
  const out: SeatState[] = [];
  // god 模式有 players_full(含 role/team/persona)
  if (full && full.length) {
    for (const p of full) {
      const old = map.get(p.seat);
      out.push({
        seat: p.seat,
        name: p.name,
        alive: p.alive,
        role: p.role,
        team: p.team,
        persona: p.persona || personas[String(p.seat)] || old?.persona,
        isSpeaking: false,
        votedTarget: old?.votedTarget,
        lastSpeech: old?.lastSpeech,
        deathReason: old?.deathReason,
        deathDay: old?.deathDay,
      });
    }
  } else {
    for (const p of pub) {
      const old = map.get(p.seat);
      const self = view.self;
      const isSelf = self && self.seat === p.seat;
      out.push({
        seat: p.seat,
        name: p.name,
        alive: p.alive,
        role: isSelf ? self.role : old?.role,
        team: isSelf ? self.team : old?.team,
        persona: personas[String(p.seat)] || old?.persona,
        isSpeaking: false,
        votedTarget: old?.votedTarget,
        lastSpeech: old?.lastSpeech,
      });
    }
  }
  return out;
}

function pushLog(state: GameState, e: Omit<LogEntry, "id" | "ts">) {
  state.log.push({ ...e, id: ++logId, ts: Date.now() });
  // 限制日志长度避免无限增长
  if (state.log.length > 500) state.log = state.log.slice(-500);
}

function setSeat(seats: SeatState[], seat: number, patch: Partial<SeatState>) {
  const s = seats.find((x) => x.seat === seat);
  if (s) Object.assign(s, patch);
}

function labelPhase(phase: string): string {
  const labels: Record<string, string> = {
    night: "夜间",
    day: "白天",
    voting: "投票",
    pk: "PK",
    last_words: "遗言",
    hunter: "猎人",
  };
  return labels[phase] || "公开";
}

function humanRejectReasonLabel(reason: string): string {
  const labels: Record<string, string> = {
    invalid_payload: "提交内容格式错误",
    no_pending_request: "当前没有等待处理的操作",
    request_id_mismatch: "操作请求已过期",
    phase_missing: "缺少阶段绑定",
    day_missing: "缺少天数绑定",
    phase_mismatch: "阶段已变化",
    day_mismatch: "天数已变化",
    day_invalid: "天数格式错误",
    action_type_mismatch: "操作类型不匹配",
    target_required: "这一步必须选择目标",
    target_invalid: "目标格式错误",
    target_not_allowed: "目标不在可选范围内",
    bid_invalid: "发言优先级格式错误",
    bid_out_of_range: "发言优先级超出范围",
  };
  return labels[reason] || reason || "未知原因";
}

export function reduce(state: GameState, ev: StoreEvent): GameState {
  // meta 事件
  if (ev.type === "__open__") {
    return { ...state, connected: true, socketClose: undefined, error: undefined };
  }
  if (ev.type === "__close__") {
    const permanentError = ev.willReconnect === false && ev.code !== undefined && ev.code !== 1000
      ? `WebSocket 已关闭 (${ev.code})${ev.reason ? `: ${ev.reason.slice(0, 240)}` : ""}`
      : undefined;
    return {
      ...state,
      connected: false,
      socketClose: ev.willReconnect === false && ev.code !== undefined
        ? {
            code: ev.code,
            reason: ev.reason || "",
            retryableByUser: ev.code === 4410 || ev.code === 4429,
          }
        : undefined,
      error: permanentError || state.error,
      // A terminal socket cannot accept this request later. Transient closes
      // retain it across cursor resume, while permanent closes remove a dead
      // control instead of leaving the UI in an impossible actionable state.
      pendingHuman: ev.willReconnect === false ? undefined : state.pendingHuman,
    };
  }
  if (ev.type === "__socket_error__") {
    return { ...state, connected: false, error: ev.message };
  }
  if (ev.type === "__manual_reconnect__") {
    return {
      ...state,
      connected: false,
      socketClose: undefined,
      error: "正在重新连接…",
    };
  }
  if (ev.type === "__reset__") return makeInitial();
  if (ev.type === "__context__") return { ...state, mySeat: ev.mySeat, mode: ev.mode };
  // 浅拷贝顶层 + seats(嵌套需深拷贝以触发 React 重渲染)
  const s: GameState = {
    ...state,
    seats: state.seats.map((x) => ({ ...x })),
    log: state.log.map((entry) => ({ ...entry })),
  };
  switch (ev.type) {
    case "snapshot": {
      const freshSnapshot = ev.resumed_from == null || ev.history_gap === true;
      s.status = ev.status;
      s.phase = ev.view.phase || s.phase;
      s.day = ev.view.day ?? s.day;
      if (ev.view.winner !== undefined) {
        s.winner = ev.view.winner ?? null;
      } else if (freshSnapshot) {
        s.winner = null;
      }
      // A cursor resume replays only events after the reducer's last accepted
      // sequence, so existing timeline state must remain. Fresh snapshots (or
      // an explicit retained-history gap) rebuild from the available replay.
      if (freshSnapshot) {
        s.log = [];
        s.lastDeaths = [];
        s.speakingSeat = null;
        s.pendingHuman = undefined;
        s.votes = {};
        s.analysis = undefined;
        s.llmStats = undefined;
      }
      if (ev.view.self?.seat != null) s.mySeat = ev.view.self.seat;
      s.seats = seatsFromView(ev.view, freshSnapshot ? [] : state.seats);
      if (ev.view.llm_stats) s.llmStats = ev.view.llm_stats;
      // 已有 votes:后端 votes 是 {voter_player_id: target_player_id},转成 {voter_seat: target_seat}
      if (ev.view.votes) {
        const players = ev.view.players || [];
        const idToSeat = new Map<string, number>();
        for (const p of players) idToSeat.set(p.id, p.seat);
        const v: Record<number, number> = {};
        for (const [voterId, targetId] of Object.entries(ev.view.votes)) {
          const vseat = idToSeat.get(String(voterId));
          const tseat = idToSeat.get(String(targetId));
          if (vseat != null && tseat != null) v[vseat] = tseat;
        }
        s.votes = v;
        // 同步座位的 votedTarget
        for (const seat of s.seats) seat.votedTarget = v[seat.seat];
      } else if (freshSnapshot) {
        for (const seat of s.seats) seat.votedTarget = undefined;
      }
      break;
    }
    case "phase_started": {
      s.phase = ev.phase;
      s.day = ev.day;
      s.pendingHuman = undefined;
      if (s.status === "waiting") s.status = "running";
      s.speakingSeat = null;
      for (const seat of s.seats) seat.isSpeaking = false;
      // 进入新投票/PK/夜晚时清空本轮票型,避免旧票型误导用户。
      if (ev.phase === "voting" || ev.phase === "pk" || ev.phase === "night") {
        s.votes = {};
        for (const seat of s.seats) seat.votedTarget = undefined;
      }
      if (ev.phase === "night") s.lastDeaths = [];
      pushLog(s, { kind: "phase", day: ev.day, text: ev.message || `进入 ${ev.phase} 阶段` });
      break;
    }
    case "night_resolved": {
      s.lastDeaths = ev.deaths || [];
      const deaths = ev.deaths || [];
      pushLog(s, {
        kind: "night_resolved",
        day: ev.day,
        text: ev.message || (deaths.length ? `昨夜 ${deaths.length} 名玩家死亡` : "昨夜平安夜,无人死亡"),
      });
      for (const d of ev.deaths || []) {
        setSeat(s.seats, d.seat, { alive: false, deathReason: d.reason, deathDay: ev.day });
        pushLog(s, { kind: "death", day: ev.day, seat: d.seat, text: `${d.seat}号 ${d.name} 死亡${d.reason ? `(${d.reason})` : ""}` });
      }
      break;
    }
    case "speech": {
      setSeat(s.seats, ev.seat, { lastSpeech: ev.text, isSpeaking: false });
      for (const seat of s.seats) seat.isSpeaking = false;
      if (s.speakingSeat === ev.seat) s.speakingSeat = null;
      pushLog(s, {
        kind: "speech", day: ev.day, seat: ev.seat, text: ev.text, claim: ev.claim,
        replyTo: ev.reply_to ?? undefined, accuses: ev.accuses, bid: ev.bid, pk: ev.pk,
      });
      break;
    }
    case "vote_cast": {
      s.votes = { ...s.votes, [ev.seat]: ev.target_seat };
      setSeat(s.seats, ev.seat, { votedTarget: ev.target_seat });
      const tgt = s.seats.find((x) => x.seat === ev.target_seat);
      pushLog(s, {
        kind: "vote", day: ev.day, seat: ev.seat, targetSeat: ev.target_seat,
        text: `${ev.seat}号 → ${ev.target_seat}号${tgt ? `(${tgt.name})` : ""}`,
      });
      break;
    }
    case "vote_rejected": {
      pushLog(s, {
        kind: "failed",
        day: ev.day,
        seat: ev.seat,
        targetSeat: ev.target_seat ?? undefined,
        text: `${ev.seat}号投票无效: ${ev.reason}`,
      });
      break;
    }
    case "action_rejected": {
      pushLog(s, {
        kind: "failed",
        day: s.day,
        seat: ev.seat,
        text: `${ev.seat ?? "?"}号 ${ev.phase}/${ev.action} 被规则拒绝: ${ev.reason_code}`,
      });
      if (
        s.pendingHuman
        && s.mySeat === ev.seat
        && s.pendingHuman.requestId === ev.request_id
      ) {
        s.pendingHuman = undefined;
      }
      break;
    }
    case "vote_resolved": {
      if (ev.exiled_seat != null && !ev.no_exile) {
        setSeat(s.seats, ev.exiled_seat, {
          alive: false,
          deathReason: "exiled",
          deathDay: ev.day,
        });
        const exiled = s.seats.find((seat) => seat.seat === ev.exiled_seat);
        s.lastDeaths = [{
          seat: ev.exiled_seat,
          name: exiled?.name || `${ev.exiled_seat}号`,
          reason: "exiled",
        }];
      }
      pushLog(s, {
        kind: "vote_resolved",
        day: ev.day,
        targetSeat: ev.exiled_seat ?? undefined,
        text: ev.message || (ev.no_exile ? "无人被放逐" : "投票结算完成"),
      });
      break;
    }
    case "vote_incomplete": {
      pushLog(s, { kind: "vote_incomplete", day: ev.day, text: `投票不完整(${ev.cast}/${ev.needed}),未投视为弃票` });
      break;
    }
    case "last_words": {
      setSeat(s.seats, ev.seat, { lastSpeech: ev.text });
      pushLog(s, { kind: "last_words", day: ev.day, seat: ev.seat, text: ev.text });
      break;
    }
    case "last_words_skipped": {
      pushLog(s, {
        kind: "last_words_skipped",
        day: ev.day,
        seat: ev.seat,
        text: `${ev.seat}号放弃遗言（environment resolution: ${ev.skip_reason}）`,
      });
      break;
    }
    case "hunter_shot": {
      if (ev.target_seat !== null) {
        setSeat(s.seats, ev.target_seat, { alive: false, deathReason: "hunter_shot", deathDay: ev.day });
        pushLog(s, { kind: "hunter", day: ev.day, seat: ev.seat, targetSeat: ev.target_seat ?? undefined, text: `${ev.seat}号猎人开枪带走 ${ev.target_seat}号` });
      } else {
        const text = ev.resolution_reason === "decision_failed"
          ? `${ev.seat}号猎人请求失败，未产生开枪 Decision`
          : ev.resolution_reason === "rules_rejected"
            ? `${ev.seat}号猎人目标被规则拒绝，未执行开枪`
            : `${ev.seat}号猎人明确选择不开枪`;
        pushLog(s, { kind: "hunter", day: ev.day, seat: ev.seat, text });
      }
      break;
    }
    case "agent_decision_failed": {
      const reason = ev.reason.slice(0, 120);
      const text = ev.seat != null
        ? `${ev.seat}号(${ev.phase})决策失败: ${reason}`
        : `${labelPhase(ev.phase)}决策失败: ${reason}`;
      pushLog(s, { kind: "failed", day: s.day, seat: ev.seat, text });
      if (
        s.pendingHuman
        && s.mySeat === ev.seat
        && (!ev.request_id || s.pendingHuman.requestId === ev.request_id)
        && (!s.pendingHuman.phase || s.pendingHuman.phase === ev.phase)
      ) {
        s.pendingHuman = undefined;
      }
      break;
    }
    case "decision_envelope_rejected": {
      const text = ev.seat != null
        ? `${ev.seat}号(${ev.phase}) DecisionEnvelope 被协议拒绝`
        : `${labelPhase(ev.phase)} DecisionEnvelope 被协议拒绝`;
      pushLog(s, { kind: "failed", day: s.day, seat: ev.seat, text });
      if (
        s.pendingHuman
        && s.mySeat === ev.seat
        && s.pendingHuman.requestId === ev.request_id
        && (!s.pendingHuman.phase || s.pendingHuman.phase === ev.phase)
      ) {
        s.pendingHuman = undefined;
      }
      break;
    }
    case "decision_validation_failed": {
      const reason = ev.reason.slice(0, 120);
      const text = ev.seat != null
        ? `${ev.seat}号(${ev.phase}) DecisionEnvelope 的 Harness 校验器失败: ${reason}`
        : `${labelPhase(ev.phase)} DecisionEnvelope 的 Harness 校验器失败: ${reason}`;
      pushLog(s, { kind: "failed", day: s.day, seat: ev.seat, text });
      if (s.pendingHuman?.requestId === ev.request_id) {
        s.pendingHuman = undefined;
      }
      break;
    }
    case "human_action_request": {
      if (s.mySeat === ev.seat) {
        const timeoutMs = Math.max(1, Number(ev.timeout || 0) * 1000);
        s.pendingHuman = {
          requestId: ev.request_id,
          actionType: ev.action_type,
          context: ev.context,
          deadline: Date.now() + timeoutMs,
          timeoutMs,
          day: ev.day,
          phase: ev.phase,
        };
      }
      break;
    }
    case "human_action_expired": {
      if (s.mySeat === ev.seat && s.pendingHuman?.requestId === ev.request_id) {
        s.pendingHuman = undefined;
        pushLog(s, {
          kind: "failed",
          day: ev.day ?? s.day,
          seat: ev.seat,
          text: `真人操作已过期: ${ev.reason}`,
        });
      }
      break;
    }
    case "human_action_accepted": {
      if (s.mySeat === ev.seat && s.pendingHuman?.requestId === ev.request_id) {
        s.pendingHuman = undefined;
      }
      break;
    }
    case "human_action_rejected": {
      if (s.mySeat === ev.seat) {
        pushLog(s, {
          kind: "failed",
          day: s.day,
          seat: ev.seat,
          text: `真人操作被拒绝: ${humanRejectReasonLabel(ev.reason)}`,
        });
        const staleReasons = new Set([
          "no_pending_request",
          "request_id_mismatch",
          "phase_mismatch",
          "day_mismatch",
        ]);
        if (
          s.pendingHuman
          && staleReasons.has(ev.reason)
          && (!ev.request_id || ev.request_id === s.pendingHuman.requestId)
        ) {
          s.pendingHuman = undefined;
        }
      }
      break;
    }
    case "game_ended": {
      s.winner = ev.winner;
      s.status = "ended";
      s.pendingHuman = undefined;
      pushLog(s, { kind: "system", day: s.day, text: `游戏结束 — ${ev.winner === "werewolves" ? "狼人阵营" : ev.winner === "village" ? "好人阵营" : ev.winner} 获胜` });
      break;
    }
    case "analysis": {
      s.analysis = ev.analysis;
      for (const a of ev.analysis.seats || []) {
        setSeat(s.seats, a.seat, {
          role: a.role,
          team: a.team,
          alive: a.alive,
          deathReason: a.death_reason ?? undefined,
          deathDay: a.death_day ?? undefined,
        });
      }
      break;
    }
    case "room_status": {
      s.status = ev.status;
      if (ev.status !== "running") s.pendingHuman = undefined;
      if (ev.error) {
        s.error = ev.error;
      }
      if (ev.status === "failed" || ev.status === "timeout" || ev.status === "cancelled" || ev.status === "interrupted") {
        const label = ev.status === "timeout"
          ? "房间超时"
          : ev.status === "cancelled"
            ? "房间已取消"
            : ev.status === "interrupted"
              ? "房间已中断"
              : "房间异常";
        pushLog(s, { kind: "system", day: s.day, text: `${label}${ev.error ? `: ${ev.error}` : ""}` });
      } else if (ev.status === "incomplete") {
        pushLog(s, {
          kind: "system",
          day: s.day,
          text: `对局未完成${ev.reason ? `: ${ev.reason}` : ""}`,
        });
      }
      break;
    }
    case "room_cleanup_failed": {
      const detail = [ev.stage, ev.error_type].filter(Boolean).join("/");
      const pending = ev.pending_task_count ? `, pending=${ev.pending_task_count}` : "";
      const message = `房间资源清理失败${detail ? `: ${detail}` : ""}${pending}`;
      s.error = message;
      pushLog(s, { kind: "failed", day: s.day, text: message });
      break;
    }
    case "game_error": {
      s.error = ev.message;
      s.pendingHuman = undefined;
      pushLog(s, { kind: "system", day: s.day, text: `错误: ${ev.message}` });
      break;
    }
  }
  return s;
}
