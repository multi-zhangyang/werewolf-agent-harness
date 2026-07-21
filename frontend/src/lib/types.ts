// Browser contracts for factual REST/WS projections from the harness runtime.

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

export interface GodPlayer extends PlayerPublicView {
  role: string;
  team: string;
  persona: string;
}

export type RoomStatus = "waiting" | "running" | "ended" | "incomplete" | "failed" | "timeout" | "cancelled" | "interrupted";

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

export interface HarnessSeeds {
  experiment_seed?: number | null;
  role_seed?: number | null;
  actor_seed?: number | null;
  orchestrator_seed?: number | null;
}

export interface DeathRecord {
  seat: number;
  name: string;
  reason?: string;
}

export interface Claim {
  role?: string;
  checked_seat?: number;
  result?: string;
}

export interface SnapshotView {
  id?: string;
  phase?: string;
  day?: number;
  players?: PlayerPublicView[];
  events?: unknown[];
  votes?: Record<string, string>;
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
  llm_stats?: Record<string, number>;
  personas?: Record<string, string>;
}

export interface ParseMetrics {
  decision_count: number;
  parsed_model_decision_count: number;
  clean_parse_count: number;
  parse_recovered_count: number;
  parse_recovered_rate: number | null;
  parse_recovered_by_action: Record<string, number>;
  parse_recovered_by_phase: Record<string, number>;
  parse_method_counts: Record<string, number>;
  lossy_consumed_count: number;
  missing_provenance_count: number;
  not_applicable_count: number;
}

export interface DecisionFailureRecord {
  request_id?: string;
  day?: number;
  phase?: string;
  seat?: number;
  action?: string;
  error_type?: string;
  terminal_kind?: "no_envelope" | "envelope_rejected" | "validation_failure";
  reason?: string;
  timeout?: boolean;
  timeout_seconds?: number;
}

export interface DecisionFailureMetrics {
  failure_count: number;
  timeout_count: number;
  by_phase: Record<string, number>;
  by_action: Record<string, number>;
  by_seat: Record<string, number>;
  by_error_type: Record<string, number>;
  records: DecisionFailureRecord[];
}

export interface AgentTurnSeatMetrics {
  seat: number;
  request_count: number;
  finished_request_count: number;
  telemetry_request_count: number;
  duplicate_finished_count: number;
  invalid_telemetry_count: number;
  missing_finished_count: number;
  generation_attempts: number;
  model_generations: number;
  generation_failures: number;
  response_retries: number;
  tool_calls: number;
  tool_successes: number;
  tool_failures: number;
  model_latency_seconds: number;
  tool_latency_seconds: number;
  elapsed_seconds: number;
  total_tokens: number;
  token_usage_complete_count: number;
  token_usage_incomplete_count: number;
  budget_failure_count: number;
  budget_failure_by_code: Record<string, number>;
  max_generation_attempts_per_request: number;
  max_model_generations_per_request: number;
  max_response_retries_per_request: number;
  max_tool_calls_per_request: number;
  max_total_tokens_per_request: number;
  max_model_latency_seconds_per_request: number;
  max_tool_latency_seconds_per_request: number;
  max_elapsed_seconds_per_request: number;
  generation_attempts_per_request: number | null;
  model_generations_per_request: number | null;
  response_retries_per_request: number | null;
  tool_calls_per_request: number | null;
  tool_failures_per_request: number | null;
  model_latency_seconds_per_request: number | null;
  tool_latency_seconds_per_request: number | null;
  elapsed_seconds_per_request: number | null;
  total_tokens_per_request: number | null;
}

export interface AgentTurnFairnessFact {
  minimum: number;
  maximum: number;
  spread: number;
  max_to_min_ratio: number | null;
  minimum_seats: number[];
  maximum_seats: number[];
}

export interface DecisionTraceMetrics {
  trace_row_count: number;
  request_count: number;
  response_count: number;
  response_failure_count: number;
  response_cancelled_count: number;
  response_validation_failure_count: number;
  terminal_response_count: number;
  unpaired_request_count: number;
  duplicate_terminal_count: number;
  orphan_terminal_count: number;
  consumed_decision_count: number;
  rules_resolution_count: number;
  model_generation_count?: number;
  model_generation_failure_count?: number;
  tool_call_count?: number;
  tool_result_count?: number;
  tool_success_count?: number;
  tool_failure_count?: number;
  tool_failure_by_code?: Record<string, number>;
  tool_failure_by_tool?: Record<string, number>;
  terminal_tool_result_count?: number;
  terminal_tool_failure_count?: number;
  requests_with_tool_failures?: number;
  max_model_generations_per_request?: number;
  max_tool_calls_per_request?: number;
  history_compaction_count?: number;
  requests_with_history_compaction?: number;
  max_compacted_tool_groups?: number;
  max_history_chars_before_compaction?: number;
  max_model_history_chars_after_compaction?: number;
  history_compaction_limit_unsatisfied_count?: number;
  max_unsatisfied_model_history_chars?: number;
  agent_turn_finished_count?: number;
  unique_agent_turn_finished_count?: number;
  duplicate_agent_turn_finished_count?: number;
  orphan_agent_turn_finished_count?: number;
  requests_with_agent_turn_finished?: number;
  requests_without_agent_turn_finished?: number;
  ambiguous_agent_turn_finished_request_count?: number;
  agent_turn_telemetry_request_count?: number;
  invalid_agent_turn_telemetry_count?: number;
  agent_turn_telemetry_identity_mismatch_count?: number;
  agent_turn_generation_attempts?: number;
  agent_turn_model_generations?: number;
  agent_turn_generation_failures?: number;
  agent_turn_response_retries?: number;
  agent_turn_tool_calls?: number;
  agent_turn_tool_successes?: number;
  agent_turn_tool_failures?: number;
  agent_turn_model_latency_seconds?: number;
  agent_turn_tool_latency_seconds?: number;
  agent_turn_elapsed_seconds?: number;
  agent_turn_total_tokens?: number;
  agent_turn_token_usage_complete_count?: number;
  agent_turn_token_usage_incomplete_count?: number;
  agent_turn_token_usage_unavailable_count?: number;
  agent_turn_token_usage_complete?: boolean | null;
  agent_turn_budget_failure_count?: number;
  agent_turn_budget_failure_by_code?: Record<string, number>;
  max_agent_turn_generation_attempts_per_request?: number;
  max_agent_turn_model_generations_per_request?: number;
  max_agent_turn_response_retries_per_request?: number;
  max_agent_turn_tool_calls_per_request?: number;
  max_agent_turn_total_tokens_per_request?: number;
  max_agent_turn_model_latency_seconds_per_request?: number;
  max_agent_turn_tool_latency_seconds_per_request?: number;
  max_agent_turn_elapsed_seconds_per_request?: number;
  agent_turn_by_seat?: AgentTurnSeatMetrics[];
  agent_turn_seat_fairness_facts?: Record<string, AgentTurnFairnessFact>;
}

export interface AgentStrategySeatMetrics {
  seat: number;
  private_state_revision: number;
  belief_count: number;
  belief_brier: number | null;
  public_commitment_count: number;
  structured_claim_count: number;
  false_role_claim_count: number;
  false_seer_result_count: number;
  role_claim_switch_count: number;
  seer_result_contradiction_count: number;
}

export interface AgentStrategyMetrics {
  schema_version: "werewolf.agent-strategy-metrics.v1";
  private_state_seat_count: number;
  belief_observation_count: number;
  belief_brier: number | null;
  structured_claim_count: number;
  false_role_claim_count: number;
  false_seer_result_count: number;
  seer_result_contradiction_count: number;
  wolf_council_message_count: number;
  wolf_final_vote_count: number;
  wolf_final_vote_target_count: number;
  wolf_final_vote_agreement: boolean | null;
  seats: AgentStrategySeatMetrics[];
}

export interface GameAnalysis {
  winner: string | null;
  days: number;
  turn_policy: string;
  seats: {
    seat: number;
    name: string;
    role: string;
    team: string;
    alive: boolean;
    death_reason?: string | null;
    death_day?: number | null;
  }[];
  decision_count: number;
  decision_trace_metrics: DecisionTraceMetrics;
  parse_metrics: ParseMetrics;
  decision_failure_metrics: DecisionFailureMetrics;
  agent_strategy_metrics?: AgentStrategyMetrics;
}

export interface DeliveryMetadata {
  delivery_seq?: number;
  delivery_id?: string;
}

export type GameEvent = (
  | {
      type: "snapshot";
      status: string;
      view: SnapshotView;
      stream_id?: string;
      cursor?: number;
      resumed_from?: number | null;
      replay_from?: number;
      history_gap?: boolean;
    }
  | { type: "phase_started"; phase: string; day: number; message?: string }
  | { type: "night_resolved"; day: number; deaths: DeathRecord[]; message?: string }
  | { type: "speech"; day: number; seat: number; name: string; text: string; bid?: number; claim?: Claim; pk?: boolean; reply_to?: number | null; accuses?: number[] }
  | { type: "vote_cast"; day: number; seat: number; name: string; target_seat: number }
  | { type: "vote_rejected"; request_id?: string; day: number; seat: number; name: string; target_seat?: number | null; reason: string; allowed_seats?: number[] }
  | { type: "vote_resolved"; day: number; message?: string; votes?: Record<string, number>; exiled_seat?: number | null; tied_seats?: number[]; no_exile?: boolean }
  | { type: "vote_incomplete"; day: number; cast: number; needed: number }
  | { type: "last_words"; day: number; seat: number; name: string; text: string }
  | { type: "last_words_skipped"; day: number; seat: number; name: string; skip_reason: string }
  | { type: "hunter_shot"; request_id?: string; day: number; seat: number; name: string; target_seat: number | null; skip_reason?: string | null; resolution_reason?: "decision_failed" | "rules_rejected" }
  | { type: "action_rejected"; request_id: string; seat?: number; phase: string; action: string; reason_code: string; reason: string }
  | { type: "agent_decision_failed"; request_id?: string; seat?: number; phase: string; reason: string; action?: string; error_type?: string; agent_kind?: "llm" | "human"; timeout?: boolean; timeout_seconds?: number }
  | { type: "decision_envelope_rejected"; request_id: string; seat?: number; phase: string; reason: string; action?: string; error_type: "DecisionEnvelopeRejected"; agent_kind?: "llm" | "human" }
  | { type: "decision_validation_failed"; request_id: string; seat?: number; phase: string; reason: string; action?: string; error_type: "DecisionValidatorError"; agent_kind?: "llm" | "human" }
  | { type: "human_action_request"; request_id: string; seat: number; action_type: string; context: Record<string, unknown>; timeout: number; day?: number; phase?: string }
  | { type: "human_action_expired"; request_id: string; seat: number; action_type: string; reason: string; day?: number; phase?: string }
  | { type: "human_action_accepted"; request_id: string; seat: number }
  | { type: "human_action_rejected"; request_id?: string; seat: number; reason: string }
  | { type: "game_ended"; winner: string | null }
  | { type: "analysis"; analysis: GameAnalysis }
  | { type: "room_status"; status: RoomStatus; reason?: string | null; error?: string | null }
  | { type: "room_cleanup_failed"; stage?: string; error_type?: string; timeout?: boolean; pending_task_count?: number; fatal?: boolean; room_id?: string }
  | { type: "game_error"; message: string; reason?: string | null }
) & DeliveryMetadata;
