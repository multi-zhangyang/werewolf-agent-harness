// 后端契约类型 —— 严格对应 src/api/room_manager.py + orchestrator.py emit 的事件与视图。
// 真实对接,绝不 mock。所有字段来自真实后端 WS/REST。

// ===== 视图 =====
export interface PlayerPublicView {
  id: string;
  name: string;
  seat: number;
  alive: boolean;
}

export interface PlayerPrivateView extends PlayerPublicView {
  role: string;
  team: string;
}

// god 模式 players_full
export interface GodPlayer extends PlayerPublicView {
  role: string;
  team: string;
  persona: string;
}

export type RoomStatus = "waiting" | "running" | "ended" | "failed" | "timeout" | "cancelled";

// /api/rooms/{id} 返回
export interface RoomInfo {
  room_id: string;
  status: RoomStatus;
  end_reason?: string | null;
  error?: string | null;
  phase?: string;
  day?: number;
  human_seats?: number[];
  players: { seat: number; name: string; alive: boolean; role?: string; team?: string }[];
  winner?: string | null;
}

export interface ProviderMeta {
  label: string;
  hint: string;
  default_api_base: string;
  default_model: string;
}

export interface ModelConfigDTO {
  provider: string;
  model: string;
  api_base: string;
  api_key: string;
  api_key_configured?: boolean;
  temperature: number;
  max_tokens: number;
  use_json_format: boolean;
}

// ===== WS 事件 =====
export type GameEvent =
  | { type: "snapshot"; status: string; view: SnapshotView }
  | { type: "phase_started"; phase: string; day: number; message?: string }
  | { type: "night_resolved"; day: number; deaths: DeathRecord[]; message?: string }
  | { type: "speech"; day: number; seat: number; name: string; text: string; bid?: number; claim?: Claim; pk?: boolean;
      reply_to?: number | null; accuses?: number[]; attitudes?: Record<string, string>; deception?: string | null }
  | { type: "vote_cast"; day: number; seat: number; name: string; target_seat: number; objective_summary?: string }
  | { type: "vote_resolved"; day: number; message?: string; votes?: Record<string, number> }
  | { type: "vote_incomplete"; day: number; cast: number; needed: number }
  | { type: "last_words"; day: number; seat: number; name: string; text: string }
  | { type: "wolf_caucus"; day: number; seat: number; target_seat?: number | null; strategy?: string; text: string }
  | { type: "wolf_caucus_consensus"; day: number; target_seat?: number | null; strategy?: string; text: string }
  | { type: "hunter_shot"; day: number; seat: number; name: string; target_seat: number | null }
  | { type: "trust_update"; seat: number; trust: Record<string, number>; phase: string }
  | { type: "reflections_update"; reflections: Record<string, string[]> }
  | { type: "agent_thinking"; seat: number; action: string; summary: string; reasoning?: string; suspicion_top?: { seat: number; suspicion: number }[]; bid?: number }
  | { type: "agent_decision_failed"; seat: number; phase: string; reason: string; action?: string; error_type?: string; timeout?: boolean; timeout_seconds?: number }
  | { type: "human_action_request"; seat: number; action_type: string; context: any; timeout: number; day?: number; phase?: string }
  | { type: "game_ended"; winner: string | null }
  | { type: "analysis"; analysis: GameAnalysis }
  | { type: "room_status"; status: RoomStatus; reason?: string | null; error?: string | null }
  | { type: "game_error"; message: string; reason?: string | null };

export interface DeathRecord {
  seat: number;
  name: string;
  reason?: string;
}

export interface Claim {
  role?: string;
  checked_seat?: number;
  result?: string;
  [k: string]: any;
}

export interface SnapshotView {
  id?: string;
  phase?: string;
  day?: number;
  players?: PlayerPublicView[];
  events?: any[];
  votes?: Record<string, number>;
  winner?: string | null;
  self?: PlayerPrivateView;
  role_state?: {
    witch_antidote?: boolean;
    witch_poison?: boolean;
    last_guarded_seat?: number | null;
    pending_hunter?: boolean;
  };
  god?: boolean;
  hidden_state?: {
    witch_antidote?: boolean;
    witch_poison?: boolean;
    last_guarded_seat?: number | null;
    pending_hunter?: string[];
  };
  players_full?: GodPlayer[];
  trust_network?: { trust: Record<string, Record<string, number>>; reflections: Record<string, string[]> };
  llm_stats?: Record<string, number>;
  personas?: Record<string, string>;
}

export interface QualityScore {
  seat: number;
  role: string;
  RI: number; // 角色推断
  SJ: number; // 战略判断
  DR: number; // 欺骗推理(好人识谎/狼人伪装)
  PS: number; // 劝说发言
  CT: number; // 反事实权衡
  highlight: string;
}

export interface GameQuality {
  scores: QualityScore[];
  game_quality: number;
  game_summary: string;
}

export interface GameAnalysis {
  winner: string | null;
  days: number;
  seats: {
    seat: number;
    name: string;
    role: string;
    team: string;
    alive: boolean;
    death_reason?: string;
    death_day?: number;
  }[];
  agent_summaries: {
    seat: number;
    role: string;
    persona: string;
    trust_final: Record<string, number>;
    claims: Record<string, any>;
  }[];
  quality?: GameQuality; // 五维对局质量评分(Beyond Survival WereAlign),LLM 事后评分,可能缺失
  parse_metrics?: ParseMetrics; // 决策解析失败/有损解析统计,赛后 analysis 输出
  decision_failure_metrics?: DecisionFailureMetrics; // 真实决策失败/超时统计,不伪造补决策
  dialogue_metrics?: DialogueMetrics; // 方向A/B/C 对话对抗量化指标(客观统计)
  debate_process_metrics?: DebateProcessMetrics; // turn_policy ablation 的辩论过程指标
  deception_audit?: DeceptionAudit; // 狼人欺骗声明与赛后审计对齐指标
  collusion_audit?: CollusionAudit; // 狼人合谋协同审计指标
  objective_metrics?: ObjectiveMetrics; // 赛后确定性轨迹指标(用真值复算,不进入 live prompt)
  posterior_metrics?: PosteriorMetrics; // EvidenceGraph 后验轨迹指标(赛后分析,不进入 live prompt)
  posterior_trace?: PosteriorTraceEntry[]; // 紧凑后验快照轨迹,仅复盘/研究用
}

export interface ParseMetrics {
  decision_count: number;
  parse_failed_count: number;
  parse_failed_rate: number;
  parse_failed_by_action: Record<string, number>;
}

export interface DecisionFailureMetrics {
  failure_count: number;
  timeout_count: number;
  by_phase: Record<string, number>;
  by_action: Record<string, number>;
  by_seat: Record<string, number>;
  by_error_type?: Record<string, number>;
  records?: {
    day?: number;
    phase?: string;
    seat?: number;
    action?: string;
    error_type?: string;
    reason?: string;
    timeout?: boolean;
    timeout_seconds?: number;
  }[];
}

export interface DialogueMetrics {
  speech_count: number;
  reply_rate: number;
  accuse_rate: number;
  attitude_rate: number;
  support_edges: number;
  oppose_edges: number;
  wolf_coordination: number;
  wolf_seats: number[];
  wolf_deception_count?: number;
  wolf_deception_dist?: Record<string, number>;
}

export interface DebateProcessMetrics {
  turn_policy?: string;
  caucus_enabled?: number;
  uses_bid_order?: number;
  uses_reply_priority?: number;
  speech_count?: number;
  speaker_count?: number;
  speaker_concentration?: number | null;
  bid_entropy?: number | null;
  avg_bid?: number | null;
  reply_count?: number;
  avg_reply_latency?: number | null;
  claim_count?: number;
  claim_challenged_count?: number;
  claim_challenged_rate?: number | null;
  accuse_target_count?: number;
  top_accuse_target_share?: number | null;
  support_loop_count?: number;
  opposition_loop_count?: number;
}

export interface DeceptionAudit {
  wolf_speech_count: number;
  declared_deception_count: number;
  audited_deception_count: number;
  declared_vs_audited_agreement: number | null;
  deception_success_rate: number | null;
  successful_misdirection_count?: number;
  target_good_audit_count?: number;
  misdirection_shift_coverage?: number | null;
  unauditable_misdirection_count?: number;
  avg_good_target_suspicion_gain: number | null;
  detected_deception_count?: number;
  peer_detection_opportunity_count?: number;
  peer_detection_rate?: number | null;
  avg_speaker_suspicion_gain?: number | null;
  listener_shift_sample_count?: number;
  evidence_linked_count?: number;
  listener_susceptibility_by_seat?: Record<string, DeceptionListenerSusceptibility>;
  villager_false_positive_rate: number | null;
  villager_false_positive_count?: number;
  good_accuse_count?: number;
  declared_by_type?: Record<string, number>;
  audited_by_type: Record<string, number>;
  records?: DeceptionAuditRecord[];
}

export interface DeceptionListenerSusceptibility {
  misdirection_samples: number;
  avg_good_target_suspicion_gain: number | null;
  misdirected_rate: number | null;
  detection_samples: number;
  avg_speaker_suspicion_gain: number | null;
  peer_detection_rate: number | null;
}

export interface DeceptionListenerShift {
  viewer_seat: number;
  target_good_suspicion_gain: number | null;
  speaker_suspicion_gain: number | null;
  misdirected: boolean;
  detected_speaker: boolean;
}

export interface DeceptionPeerDetection {
  detected: boolean;
  detector_seats: number[];
  avg_speaker_suspicion_gain: number | null;
}

export interface DeceptionAuditRecord {
  day: number;
  seat: number;
  declared: string | null;
  audited_types: string[];
  target_good_seats: number[];
  target_wolf_seats: number[];
  avg_good_target_suspicion_gain: number | null;
  successful_misdirection: boolean;
  evidence_ids: string[];
  posterior_delta_ids: string[];
  evidence_source_types: Record<string, number>;
  listener_shifts: DeceptionListenerShift[];
  peer_detection: DeceptionPeerDetection;
}

export interface CollusionAudit {
  wolf_speech_count?: number | null;
  wolf_pair_count?: number | null;
  active_wolf_pair_count?: number | null;
  wolf_to_wolf_support_count?: number | null;
  mutual_support_pair_count?: number | null;
  shared_good_target_count?: number | null;
  shared_good_target_speaker_coverage?: number | null;
  narrative_overlap_pair_count?: number | null;
  avg_narrative_overlap?: number | null;
  coordinated_pressure_count?: number | null;
  avg_shared_target_suspicion_gain?: number | null;
  avg_colluder_suspicion_gain?: number | null;
  evidence_linked_count?: number | null;
  pair_listener_shift_sample_count?: number | null;
  avg_pair_target_suspicion_gain?: number | null;
  pair_target_misdirected_rate?: number | null;
  windowed_relay_count?: number | null;
  avg_windowed_relay_latency?: number | null;
  avg_relay_target_suspicion_gain?: number | null;
  relay_target_misdirected_rate?: number | null;
  deception_linked_pair_count?: number | null;
  pair_listener_susceptibility_by_pair?: Record<string, CollusionPairSusceptibility>;
  records?: CollusionAuditRecord[];
}

export interface CollusionPairSusceptibility {
  wolf_seats?: number[];
  active_days?: number[];
  shared_good_target_count?: number | null;
  wolf_to_wolf_support_count?: number | null;
  mutual_support_pair_count?: number | null;
  narrative_overlap_pair_count?: number | null;
  coordinated_pressure_count?: number | null;
  target_shift_sample_count?: number | null;
  avg_target_suspicion_gain?: number | null;
  target_misdirected_rate?: number | null;
  colluder_shift_sample_count?: number | null;
  avg_colluder_suspicion_gain?: number | null;
  avg_narrative_overlap?: number | null;
  windowed_relay_count?: number | null;
  avg_windowed_relay_latency?: number | null;
  avg_relay_target_suspicion_gain?: number | null;
  relay_target_misdirected_rate?: number | null;
  evidence_linked_count?: number | null;
  deception_record_count?: number | null;
  successful_deception_record_count?: number | null;
  peer_detected_deception_record_count?: number | null;
  audited_deception_types?: Record<string, number>;
  evidence_ids?: string[];
  posterior_delta_ids?: string[];
}

export interface CollusionAuditRecord {
  type?: string;
  day?: number | null;
  target_good_seat?: number | null;
  wolf_seats?: number[];
  pair?: number[];
  lead_wolf_seat?: number | null;
  follow_wolf_seat?: number | null;
  relay_latency?: number | null;
  shared_good_targets?: number[];
  follower_supports_lead?: boolean;
  avg_target_suspicion_gain?: number | null;
  speaker_seat?: number | null;
  target_seat?: number | null;
  target_good_seats?: number[];
  wolf_to_wolf_support_count?: number | null;
  shared_good_target_count?: number | null;
  narrative_overlap?: number | null;
  coordinated_pressure?: boolean;
  evidence_ids?: string[];
  posterior_delta_ids?: string[];
  [k: string]: unknown;
}

export interface ObjectiveMetrics {
  vote_count: number;
  good_vote_count: number;
  wolf_vote_count: number;
  vote_accuracy_good: number | null;
  vote_accuracy_wolf: number | null;
  accuse_count: number;
  good_accuse_count: number;
  wolf_accuse_count: number;
  accuse_precision_good: number | null;
  accuse_precision_wolf: number | null;
  attitude_vote_consistency: number | null;
  attitude_vote_count: number;
  accuse_to_vote_conversion: number | null;
  osr_summary_rate: number | null;
  ct_marker_rate: number | null;
  seer_claim_follow_rate: number | null;
  seer_claim_follow_vote_count: number;
}

export interface PosteriorMetrics {
  snapshot_count: number;
  speech_snapshot_count: number;
  avg_speech_posterior_shift: number | null;
  good_final_wolf_suspicion_gap: number | null;
  good_final_top_suspect_accuracy: number | null;
  herding_index: number | null;
  herding_event_count?: number | null;
  correct_herding_rate?: number | null;
  wrong_herding_rate?: number | null;
  final_brier_score: number | null;
  final_log_loss: number | null;
  good_final_brier_score: number | null;
  good_final_log_loss: number | null;
  constrained_final_brier_score?: number | null;
  constrained_final_log_loss?: number | null;
  constrained_good_final_brier_score?: number | null;
  constrained_good_final_log_loss?: number | null;
  constrained_calibration_ece?: number | null;
  constrained_calibration_bins?: CalibrationBin[];
  calibration_ece: number | null;
  calibration_bins: CalibrationBin[];
}

export interface CalibrationBin {
  range?: [number, number];
  avg_prediction?: number | null;
  wolf_rate?: number | null;
  count: number;
  bin?: number;
  bin_start?: number;
  bin_end?: number;
  lower?: number;
  upper?: number;
  confidence?: number;
  accuracy?: number;
  total?: number;
  [k: string]: number | string | [number, number] | null | undefined;
}

export interface PosteriorTraceEntry {
  day: number;
  phase: string;
  trigger: string;
  source_seat?: number | null;
  viewer_seat: number;
  posterior: Record<string, number>;
  constrained_posterior?: Record<string, number>;
  legal_worlds?: LegalWorlds;
  evidence_items?: EvidenceItem[];
  posterior_deltas?: PosteriorDelta[];
  top_suspects: { seat: number; werewolf_suspicion: number }[];
}

export interface LegalWorlds {
  wolf_count: number;
  known_wolves: number[];
  known_villagers: number[];
  world_count: number;
  is_contradictory: boolean;
  is_truncated: boolean;
  worlds?: { wolf_seats: number[] }[];
  [k: string]: unknown;
}

export interface EvidenceItem {
  evidence_id: string;
  type: string;
  visibility: "public" | "private" | string;
  provenance: string;
  confidence?: number | null;
  day?: number | null;
  phase?: string;
  source_seat?: number | null;
  target_seat?: number | null;
  payload?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface PosteriorDelta {
  target_seat: number | null;
  delta: number;
  after: number;
  evidence_id: string;
  source_type: string;
  reason?: string;
}
