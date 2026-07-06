// 游戏状态 reducer —— 把真实后端 WS 事件流归约成前端 store。
// 零 mock:所有状态来自真实事件。snapshot 提供初始投影,后续事件增量更新。
import type { GameEvent, SnapshotView, DeathRecord, Claim, GameAnalysis } from "./types";

// 内部 meta 事件(WS 连接生命周期 + store 重置),非后端事件
export type MetaEvent =
  | { type: "__open__" }
  | { type: "__close__" }
  | { type: "__reset__" }
  | { type: "__context__"; mySeat: number | null; mode: string };

export type StoreEvent = GameEvent | MetaEvent;

export interface LogEntry {
  id: number;
  kind: string; // phase / speech / vote / death / hunter / last_words / thinking / failed / system
  day: number;
  text: string;
  seat?: number;
  targetSeat?: number;
  claim?: Claim;
  ts: number;
  // 思考流专用:god 可见完整 reasoning;spectate 只收到后端净化后的 summary。
  action?: string;
  reasoning?: string;
  suspicionTop?: { seat: number; suspicion: number }[];
  bid?: number;
  // 方向A/B/C 对话元数据(god/spectate 可见,渲染反驳箭头/指控/态度网络/欺骗标签)
  replyTo?: number | null;
  accuses?: number[];
  attitudes?: Record<string, string>;
  deception?: string | null;
  objectiveSummary?: string;
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
  trust?: Record<string, number>; // 该 seat 对他人的怀疑度(god 可见)
  reflections?: string[];
  deathReason?: string;
  deathDay?: number;
}

export interface GameState {
  connected: boolean;
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
  pendingHuman?: { actionType: string; context: any; deadline: number };
  // god 全知
  trustNetwork?: Record<string, Record<string, number>>;
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
        trust: old?.trust,
        reflections: old?.reflections,
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
        trust: old?.trust,
        reflections: old?.reflections,
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

export function reduce(state: GameState, ev: StoreEvent): GameState {
  // meta 事件
  if (ev.type === "__open__") return { ...state, connected: true, error: undefined };
  if (ev.type === "__close__") return { ...state, connected: false };
  if (ev.type === "__reset__") return makeInitial();
  if (ev.type === "__context__") return { ...state, mySeat: ev.mySeat, mode: ev.mode };
  // 浅拷贝顶层 + seats(嵌套需深拷贝以触发 React 重渲染)
  const s: GameState = { ...state, seats: state.seats.map((x) => ({ ...x })) };
  switch (ev.type) {
    case "snapshot": {
      s.status = ev.status;
      s.phase = ev.view.phase || s.phase;
      s.day = ev.view.day ?? s.day;
      s.winner = ev.view.winner ?? s.winner;
      if (ev.view.self?.seat != null) s.mySeat = ev.view.self.seat;
      s.seats = seatsFromView(ev.view, state.seats);
      if (ev.view.trust_network) s.trustNetwork = ev.view.trust_network.trust;
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
      }
      break;
    }
    case "phase_started": {
      s.phase = ev.phase;
      s.day = ev.day;
      if (s.status === "waiting") s.status = "running";
      // 进入新阶段时清空本轮投票与发言高亮
      if (ev.phase === "voting" || ev.phase === "night") {
        s.votes = {};
        for (const seat of s.seats) seat.votedTarget = undefined;
      }
      pushLog(s, { kind: "phase", day: ev.day, text: ev.message || `进入 ${ev.phase} 阶段` });
      break;
    }
    case "night_resolved": {
      s.lastDeaths = ev.deaths || [];
      for (const d of ev.deaths || []) {
        setSeat(s.seats, d.seat, { alive: false, deathReason: d.reason, deathDay: ev.day });
        pushLog(s, { kind: "death", day: ev.day, seat: d.seat, text: `${d.seat}号 ${d.name} 死亡${d.reason ? `(${d.reason})` : ""}` });
      }
      if (!(ev.deaths || []).length) pushLog(s, { kind: "system", day: ev.day, text: "昨夜平安夜,无人死亡" });
      break;
    }
    case "speech": {
      setSeat(s.seats, ev.seat, { lastSpeech: ev.text, isSpeaking: true });
      // 其他座位取消高亮
      for (const seat of s.seats) if (seat.seat !== ev.seat) seat.isSpeaking = false;
      s.speakingSeat = ev.seat;
      pushLog(s, {
        kind: "speech", day: ev.day, seat: ev.seat, text: ev.text, claim: ev.claim,
        replyTo: ev.reply_to ?? undefined, accuses: ev.accuses,
        attitudes: ev.attitudes, deception: ev.deception,
      });
      break;
    }
    case "vote_cast": {
      s.votes = { ...s.votes, [ev.seat]: ev.target_seat };
      setSeat(s.seats, ev.seat, { votedTarget: ev.target_seat });
      const tgt = s.seats.find((x) => x.seat === ev.target_seat);
      pushLog(s, {
        kind: "vote", day: ev.day, seat: ev.seat, targetSeat: ev.target_seat,
        objectiveSummary: ev.objective_summary,
        text: `${ev.seat}号 → ${ev.target_seat}号${tgt ? `(${tgt.name})` : ""}`,
      });
      break;
    }
    case "vote_resolved": {
      pushLog(s, { kind: "system", day: ev.day, text: ev.message || "投票结算完成" });
      break;
    }
    case "vote_incomplete": {
      pushLog(s, { kind: "system", day: ev.day, text: `投票不完整(${ev.cast}/${ev.needed}),跳过本轮` });
      break;
    }
    case "last_words": {
      setSeat(s.seats, ev.seat, { lastSpeech: ev.text });
      pushLog(s, { kind: "last_words", day: ev.day, seat: ev.seat, text: ev.text });
      break;
    }
    case "wolf_caucus": {
      // god 模式可见狼队白天党团私聊提案(信息隔离:_should_receive 仅 god/replay 收)
      pushLog(s, { kind: "caucus", day: ev.day, seat: ev.seat, targetSeat: ev.target_seat ?? undefined, text: ev.text });
      break;
    }
    case "wolf_caucus_consensus": {
      pushLog(s, { kind: "caucus_consensus", day: ev.day, targetSeat: ev.target_seat ?? undefined, text: ev.text });
      break;
    }
    case "hunter_shot": {
      if (ev.target_seat !== null) {
        setSeat(s.seats, ev.target_seat, { alive: false, deathReason: "hunter_shot", deathDay: ev.day });
        pushLog(s, { kind: "hunter", day: ev.day, seat: ev.seat, targetSeat: ev.target_seat ?? undefined, text: `${ev.seat}号猎人开枪带走 ${ev.target_seat}号` });
      } else {
        pushLog(s, { kind: "hunter", day: ev.day, seat: ev.seat, text: `${ev.seat}号猎人未开枪` });
      }
      break;
    }
    case "trust_update": {
      setSeat(s.seats, ev.seat, { trust: ev.trust });
      if (s.trustNetwork) s.trustNetwork = { ...s.trustNetwork, [String(ev.seat)]: ev.trust };
      break;
    }
    case "reflections_update": {
      for (const [seat, refls] of Object.entries(ev.reflections)) {
        setSeat(s.seats, Number(seat), { reflections: refls });
      }
      break;
    }
    case "agent_thinking": {
      // god 模式会带完整 thought;观战模式只带 summary。
      pushLog(s, {
        kind: "thinking",
        day: s.day,
        seat: ev.seat,
        text: ev.summary || "(思考)",
        action: ev.action,
        reasoning: ev.reasoning,
        suspicionTop: ev.suspicion_top,
        bid: ev.bid,
      });
      break;
    }
    case "agent_decision_failed": {
      pushLog(s, { kind: "failed", day: s.day, seat: ev.seat, text: `${ev.seat}号(${ev.phase})决策失败: ${ev.reason.slice(0, 120)}` });
      break;
    }
    case "human_action_request": {
      if (s.mySeat === ev.seat) {
        s.pendingHuman = { actionType: ev.action_type, context: ev.context, deadline: Date.now() + ev.timeout * 1000 };
      }
      break;
    }
    case "game_ended": {
      s.winner = ev.winner;
      s.status = "ended";
      pushLog(s, { kind: "system", day: s.day, text: `游戏结束 — ${ev.winner === "werewolves" ? "狼人阵营" : ev.winner === "village" ? "好人阵营" : ev.winner} 获胜` });
      break;
    }
    case "analysis": {
      s.analysis = ev.analysis;
      for (const a of ev.analysis.seats || []) {
        setSeat(s.seats, a.seat, { role: a.role, team: a.team, alive: a.alive, deathReason: a.death_reason, deathDay: a.death_day });
      }
      break;
    }
    case "room_status": {
      s.status = ev.status;
      if (ev.error) {
        s.error = ev.error;
      }
      if (ev.status === "failed" || ev.status === "timeout" || ev.status === "cancelled") {
        const label = ev.status === "timeout" ? "房间超时" : ev.status === "cancelled" ? "房间已取消" : "房间异常";
        pushLog(s, { kind: "system", day: s.day, text: `${label}${ev.error ? `: ${ev.error}` : ""}` });
      }
      break;
    }
    case "game_error": {
      s.error = ev.message;
      pushLog(s, { kind: "system", day: s.day, text: `错误: ${ev.message}` });
      break;
    }
  }
  return s;
}
