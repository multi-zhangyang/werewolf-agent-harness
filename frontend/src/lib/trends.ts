import type { GameAnalysis, QualityScore } from "./types";

export type TrendDim = "RI" | "SJ" | "DR" | "PS" | "CT";

export interface GameTrendEntry {
  id: string;
  ts: number;
  winner: string | null;
  days: number;
  seat_count: number;
  game_quality: number | null;
  RI: number | null;
  SJ: number | null;
  DR: number | null;
  PS: number | null;
  CT: number | null;
  speech_count: number | null;
  reply_rate: number | null;
  accuse_rate: number | null;
  attitude_rate: number | null;
  support_edges: number | null;
  oppose_edges: number | null;
  wolf_coordination: number | null;
  wolf_deception_count: number | null;
  turn_policy: string | null;
  speaker_concentration: number | null;
  bid_entropy: number | null;
  claim_challenged_rate: number | null;
  top_accuse_target_share: number | null;
  wolf_speech_count: number | null;
  declared_deception_count: number | null;
  audited_deception_count: number | null;
  declared_vs_audited_agreement: number | null;
  deception_success_rate: number | null;
  misdirection_shift_coverage: number | null;
  unauditable_misdirection_count: number | null;
  avg_good_target_suspicion_gain: number | null;
  detected_deception_count: number | null;
  peer_detection_opportunity_count: number | null;
  peer_detection_rate: number | null;
  avg_speaker_suspicion_gain: number | null;
  listener_shift_sample_count: number | null;
  evidence_linked_count: number | null;
  villager_false_positive_rate: number | null;
  audited_by_type: Record<string, number> | null;
  collusion_active_wolf_pair_count: number | null;
  collusion_wolf_to_wolf_support_count: number | null;
  collusion_mutual_support_pair_count: number | null;
  collusion_shared_good_target_count: number | null;
  collusion_avg_narrative_overlap: number | null;
  collusion_coordinated_pressure_count: number | null;
  collusion_pair_listener_shift_sample_count: number | null;
  collusion_avg_pair_target_suspicion_gain: number | null;
  collusion_pair_target_misdirected_rate: number | null;
  collusion_windowed_relay_count: number | null;
  collusion_avg_windowed_relay_latency: number | null;
  collusion_avg_relay_target_suspicion_gain: number | null;
  collusion_relay_target_misdirected_rate: number | null;
  collusion_deception_linked_pair_count: number | null;
  decision_count: number | null;
  parse_failed_count: number | null;
  parse_failed_rate: number | null;
  vote_accuracy_good: number | null;
  vote_accuracy_wolf: number | null;
  accuse_precision_good: number | null;
  accuse_precision_wolf: number | null;
  attitude_vote_consistency: number | null;
  accuse_to_vote_conversion: number | null;
  osr_summary_rate: number | null;
  ct_marker_rate: number | null;
  snapshot_count: number | null;
  speech_snapshot_count: number | null;
  avg_speech_posterior_shift: number | null;
  good_final_wolf_suspicion_gap: number | null;
  good_final_top_suspect_accuracy: number | null;
  herding_index: number | null;
  herding_event_count: number | null;
  correct_herding_rate: number | null;
  wrong_herding_rate: number | null;
  final_brier_score: number | null;
  final_log_loss: number | null;
  good_final_brier_score: number | null;
  good_final_log_loss: number | null;
  constrained_final_brier_score: number | null;
  constrained_final_log_loss: number | null;
  constrained_good_final_brier_score: number | null;
  constrained_good_final_log_loss: number | null;
  constrained_calibration_ece: number | null;
  calibration_ece: number | null;
}

const STORAGE_KEY = "werewolf.mas.gameTrends.v1";
const MAX_ENTRIES = 40;
const DIMS: TrendDim[] = ["RI", "SJ", "DR", "PS", "CT"];

export function analysisToTrendEntry(analysis: GameAnalysis, id: string): GameTrendEntry {
  const scores = analysis.quality?.scores || [];
  const dm = analysis.dialogue_metrics;
  const debate = analysis.debate_process_metrics;
  const audit = analysis.deception_audit;
  const collusion = analysis.collusion_audit;
  const parseMetrics = analysis.parse_metrics;
  const om = analysis.objective_metrics;
  const pm = analysis.posterior_metrics;
  return {
    id,
    ts: Date.now(),
    winner: analysis.winner,
    days: Number(analysis.days) || 0,
    seat_count: analysis.seats?.length || scores.length || 0,
    game_quality: finite01(analysis.quality?.game_quality),
    RI: avgScore(scores, "RI"),
    SJ: avgScore(scores, "SJ"),
    DR: avgScore(scores, "DR"),
    PS: avgScore(scores, "PS"),
    CT: avgScore(scores, "CT"),
    speech_count: finite(dm?.speech_count),
    reply_rate: finite01(dm?.reply_rate),
    accuse_rate: finite01(dm?.accuse_rate),
    attitude_rate: finite01(dm?.attitude_rate),
    support_edges: finite(dm?.support_edges),
    oppose_edges: finite(dm?.oppose_edges),
    wolf_coordination: finite(dm?.wolf_coordination),
    wolf_deception_count: finite(dm?.wolf_deception_count),
    turn_policy: typeof debate?.turn_policy === "string" ? debate.turn_policy : null,
    speaker_concentration: finite01(debate?.speaker_concentration),
    bid_entropy: finite01(debate?.bid_entropy),
    claim_challenged_rate: finite01(debate?.claim_challenged_rate),
    top_accuse_target_share: finite01(debate?.top_accuse_target_share),
    wolf_speech_count: finite(audit?.wolf_speech_count),
    declared_deception_count: finite(audit?.declared_deception_count),
    audited_deception_count: finite(audit?.audited_deception_count),
    declared_vs_audited_agreement: finite01(audit?.declared_vs_audited_agreement),
    deception_success_rate: finite01(audit?.deception_success_rate),
    misdirection_shift_coverage: finite01(audit?.misdirection_shift_coverage),
    unauditable_misdirection_count: finite(audit?.unauditable_misdirection_count),
    avg_good_target_suspicion_gain: finite(audit?.avg_good_target_suspicion_gain),
    detected_deception_count: finite(audit?.detected_deception_count),
    peer_detection_opportunity_count: finite(audit?.peer_detection_opportunity_count),
    peer_detection_rate: finite01(audit?.peer_detection_rate),
    avg_speaker_suspicion_gain: finite(audit?.avg_speaker_suspicion_gain),
    listener_shift_sample_count: finite(audit?.listener_shift_sample_count),
    evidence_linked_count: finite(audit?.evidence_linked_count),
    villager_false_positive_rate: finite01(audit?.villager_false_positive_rate),
    audited_by_type: finiteRecord(audit?.audited_by_type),
    collusion_active_wolf_pair_count: finite(collusion?.active_wolf_pair_count),
    collusion_wolf_to_wolf_support_count: finite(collusion?.wolf_to_wolf_support_count),
    collusion_mutual_support_pair_count: finite(collusion?.mutual_support_pair_count),
    collusion_shared_good_target_count: finite(collusion?.shared_good_target_count),
    collusion_avg_narrative_overlap: finite01(collusion?.avg_narrative_overlap),
    collusion_coordinated_pressure_count: finite(collusion?.coordinated_pressure_count),
    collusion_pair_listener_shift_sample_count: finite(collusion?.pair_listener_shift_sample_count),
    collusion_avg_pair_target_suspicion_gain: finite(collusion?.avg_pair_target_suspicion_gain),
    collusion_pair_target_misdirected_rate: finite01(collusion?.pair_target_misdirected_rate),
    collusion_windowed_relay_count: finite(collusion?.windowed_relay_count),
    collusion_avg_windowed_relay_latency: finite(collusion?.avg_windowed_relay_latency),
    collusion_avg_relay_target_suspicion_gain: finite(collusion?.avg_relay_target_suspicion_gain),
    collusion_relay_target_misdirected_rate: finite01(collusion?.relay_target_misdirected_rate),
    collusion_deception_linked_pair_count: finite(collusion?.deception_linked_pair_count),
    decision_count: finite(parseMetrics?.decision_count),
    parse_failed_count: finite(parseMetrics?.parse_failed_count),
    parse_failed_rate: finite01(parseMetrics?.parse_failed_rate),
    vote_accuracy_good: finite01(om?.vote_accuracy_good),
    vote_accuracy_wolf: finite01(om?.vote_accuracy_wolf),
    accuse_precision_good: finite01(om?.accuse_precision_good),
    accuse_precision_wolf: finite01(om?.accuse_precision_wolf),
    attitude_vote_consistency: finite01(om?.attitude_vote_consistency),
    accuse_to_vote_conversion: finite01(om?.accuse_to_vote_conversion),
    osr_summary_rate: finite01(om?.osr_summary_rate),
    ct_marker_rate: finite01(om?.ct_marker_rate),
    snapshot_count: finite(pm?.snapshot_count),
    speech_snapshot_count: finite(pm?.speech_snapshot_count),
    avg_speech_posterior_shift: finite01(pm?.avg_speech_posterior_shift),
    good_final_wolf_suspicion_gap: finite(pm?.good_final_wolf_suspicion_gap),
    good_final_top_suspect_accuracy: finite01(pm?.good_final_top_suspect_accuracy),
    herding_index: finite01(pm?.herding_index),
    herding_event_count: finite(pm?.herding_event_count),
    correct_herding_rate: finite01(pm?.correct_herding_rate),
    wrong_herding_rate: finite01(pm?.wrong_herding_rate),
    final_brier_score: finite(pm?.final_brier_score),
    final_log_loss: finite(pm?.final_log_loss),
    good_final_brier_score: finite(pm?.good_final_brier_score),
    good_final_log_loss: finite(pm?.good_final_log_loss),
    constrained_final_brier_score: finite(pm?.constrained_final_brier_score),
    constrained_final_log_loss: finite(pm?.constrained_final_log_loss),
    constrained_good_final_brier_score: finite(pm?.constrained_good_final_brier_score),
    constrained_good_final_log_loss: finite(pm?.constrained_good_final_log_loss),
    constrained_calibration_ece: finite01(pm?.constrained_calibration_ece),
    calibration_ece: finite01(pm?.calibration_ece),
  };
}

export function loadGameTrends(): GameTrendEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map(normalizeEntry)
      .filter((e): e is GameTrendEntry => e !== null)
      .sort((a, b) => a.ts - b.ts)
      .slice(-MAX_ENTRIES);
  } catch {
    return [];
  }
}

export function upsertGameTrend(entry: GameTrendEntry): GameTrendEntry[] {
  const current = loadGameTrends();
  const prior = current.find((e) => e.id === entry.id);
  const stable = { ...entry, ts: prior?.ts ?? entry.ts };
  const next = current
    .filter((e) => e.id !== entry.id)
    .concat(stable)
    .sort((a, b) => a.ts - b.ts)
    .slice(-MAX_ENTRIES);
  save(next);
  return next;
}

export function clearGameTrends() {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // localStorage can be unavailable in restricted browser contexts.
  }
}

export function averageTrend(entries: GameTrendEntry[], key: keyof GameTrendEntry): number | null {
  const nums = entries.map((e) => e[key]).filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  if (!nums.length) return null;
  return nums.reduce((sum, v) => sum + v, 0) / nums.length;
}

export function trendDims(): TrendDim[] {
  return DIMS.slice();
}

function avgScore(scores: QualityScore[], key: TrendDim): number | null {
  const nums = scores.map((s) => finite01(s[key])).filter((v): v is number => v !== null);
  if (!nums.length) return null;
  return nums.reduce((sum, v) => sum + v, 0) / nums.length;
}

function normalizeEntry(raw: unknown): GameTrendEntry | null {
  if (!raw || typeof raw !== "object") return null;
  const obj = raw as Record<string, unknown>;
  if (typeof obj.id !== "string" || !obj.id) return null;
  return {
    id: obj.id,
    ts: finite(obj.ts) || Date.now(),
    winner: typeof obj.winner === "string" ? obj.winner : null,
    days: finite(obj.days) || 0,
    seat_count: finite(obj.seat_count) || 0,
    game_quality: finite01(obj.game_quality),
    RI: finite01(obj.RI),
    SJ: finite01(obj.SJ),
    DR: finite01(obj.DR),
    PS: finite01(obj.PS),
    CT: finite01(obj.CT),
    speech_count: finite(obj.speech_count),
    reply_rate: finite01(obj.reply_rate),
    accuse_rate: finite01(obj.accuse_rate),
    attitude_rate: finite01(obj.attitude_rate),
    support_edges: finite(obj.support_edges),
    oppose_edges: finite(obj.oppose_edges),
    wolf_coordination: finite(obj.wolf_coordination),
    wolf_deception_count: finite(obj.wolf_deception_count),
    turn_policy: typeof obj.turn_policy === "string" ? obj.turn_policy : null,
    speaker_concentration: finite01(obj.speaker_concentration),
    bid_entropy: finite01(obj.bid_entropy),
    claim_challenged_rate: finite01(obj.claim_challenged_rate),
    top_accuse_target_share: finite01(obj.top_accuse_target_share),
    wolf_speech_count: finite(obj.wolf_speech_count),
    declared_deception_count: finite(obj.declared_deception_count),
    audited_deception_count: finite(obj.audited_deception_count),
    declared_vs_audited_agreement: finite01(obj.declared_vs_audited_agreement),
    deception_success_rate: finite01(obj.deception_success_rate),
    misdirection_shift_coverage: finite01(obj.misdirection_shift_coverage),
    unauditable_misdirection_count: finite(obj.unauditable_misdirection_count),
    avg_good_target_suspicion_gain: finite(obj.avg_good_target_suspicion_gain),
    detected_deception_count: finite(obj.detected_deception_count),
    peer_detection_opportunity_count: finite(obj.peer_detection_opportunity_count),
    peer_detection_rate: finite01(obj.peer_detection_rate),
    avg_speaker_suspicion_gain: finite(obj.avg_speaker_suspicion_gain),
    listener_shift_sample_count: finite(obj.listener_shift_sample_count),
    evidence_linked_count: finite(obj.evidence_linked_count),
    villager_false_positive_rate: finite01(obj.villager_false_positive_rate),
    audited_by_type: finiteRecord(obj.audited_by_type),
    collusion_active_wolf_pair_count: finite(obj.collusion_active_wolf_pair_count),
    collusion_wolf_to_wolf_support_count: finite(obj.collusion_wolf_to_wolf_support_count),
    collusion_mutual_support_pair_count: finite(obj.collusion_mutual_support_pair_count),
    collusion_shared_good_target_count: finite(obj.collusion_shared_good_target_count),
    collusion_avg_narrative_overlap: finite01(obj.collusion_avg_narrative_overlap),
    collusion_coordinated_pressure_count: finite(obj.collusion_coordinated_pressure_count),
    collusion_pair_listener_shift_sample_count: finite(obj.collusion_pair_listener_shift_sample_count),
    collusion_avg_pair_target_suspicion_gain: finite(obj.collusion_avg_pair_target_suspicion_gain),
    collusion_pair_target_misdirected_rate: finite01(obj.collusion_pair_target_misdirected_rate),
    collusion_windowed_relay_count: finite(obj.collusion_windowed_relay_count),
    collusion_avg_windowed_relay_latency: finite(obj.collusion_avg_windowed_relay_latency),
    collusion_avg_relay_target_suspicion_gain: finite(obj.collusion_avg_relay_target_suspicion_gain),
    collusion_relay_target_misdirected_rate: finite01(obj.collusion_relay_target_misdirected_rate),
    collusion_deception_linked_pair_count: finite(obj.collusion_deception_linked_pair_count),
    decision_count: finite(obj.decision_count),
    parse_failed_count: finite(obj.parse_failed_count),
    parse_failed_rate: finite01(obj.parse_failed_rate),
    vote_accuracy_good: finite01(obj.vote_accuracy_good),
    vote_accuracy_wolf: finite01(obj.vote_accuracy_wolf),
    accuse_precision_good: finite01(obj.accuse_precision_good),
    accuse_precision_wolf: finite01(obj.accuse_precision_wolf),
    attitude_vote_consistency: finite01(obj.attitude_vote_consistency),
    accuse_to_vote_conversion: finite01(obj.accuse_to_vote_conversion),
    osr_summary_rate: finite01(obj.osr_summary_rate),
    ct_marker_rate: finite01(obj.ct_marker_rate),
    snapshot_count: finite(obj.snapshot_count),
    speech_snapshot_count: finite(obj.speech_snapshot_count),
    avg_speech_posterior_shift: finite01(obj.avg_speech_posterior_shift),
    good_final_wolf_suspicion_gap: finite(obj.good_final_wolf_suspicion_gap),
    good_final_top_suspect_accuracy: finite01(obj.good_final_top_suspect_accuracy),
    herding_index: finite01(obj.herding_index),
    herding_event_count: finite(obj.herding_event_count),
    correct_herding_rate: finite01(obj.correct_herding_rate),
    wrong_herding_rate: finite01(obj.wrong_herding_rate),
    final_brier_score: finite(obj.final_brier_score),
    final_log_loss: finite(obj.final_log_loss),
    good_final_brier_score: finite(obj.good_final_brier_score),
    good_final_log_loss: finite(obj.good_final_log_loss),
    constrained_final_brier_score: finite(obj.constrained_final_brier_score),
    constrained_final_log_loss: finite(obj.constrained_final_log_loss),
    constrained_good_final_brier_score: finite(obj.constrained_good_final_brier_score),
    constrained_good_final_log_loss: finite(obj.constrained_good_final_log_loss),
    constrained_calibration_ece: finite01(obj.constrained_calibration_ece),
    calibration_ece: finite01(obj.calibration_ece),
  };
}

function save(entries: GameTrendEntry[]) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Keep the UI usable even if the browser refuses persistence.
  }
}

function finite(v: unknown): number | null {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function finite01(v: unknown): number | null {
  const n = finite(v);
  if (n === null) return null;
  return Math.max(0, Math.min(1, n));
}

function finiteRecord(v: unknown): Record<string, number> | null {
  if (!v || typeof v !== "object" || Array.isArray(v)) return null;
  const entries = Object.entries(v as Record<string, unknown>)
    .flatMap(([key, value]) => {
      const n = finite(value);
      return n === null ? [] : [[key, n] as [string, number]];
    });
  if (!entries.length) return null;
  return Object.fromEntries(entries);
}
