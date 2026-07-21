import type { GameState } from "./store";

const TARGET_ACTIONS = new Set([
  "night_kill",
  "see",
  "save",
  "poison",
  "guard",
  "hunter_shot",
  "vote",
] as const);

const TEXT_ACTIONS = new Set(["speak", "last_words"] as const);

export type HumanActionName =
  | "night_kill"
  | "see"
  | "save"
  | "poison"
  | "guard"
  | "hunter_shot"
  | "vote"
  | "speak"
  | "last_words";

export interface HumanActionSchema {
  action: HumanActionName;
  requiresTarget: boolean;
  canSkip: boolean;
  targetSeats: number[];
  inputKind: "target" | "text";
}

export type HumanActionControls =
  | { ok: true; schema: HumanActionSchema; targets: GameState["seats"] }
  | { ok: false; reason: string };

export const ACTION_LABEL: Record<string, string> = {
  speak: "发言",
  vote: "投票放逐",
  night_action: "夜间行动",
  night_kill: "夜间击杀",
  see: "查验身份",
  save: "使用解药",
  poison: "使用毒药",
  guard: "守护玩家",
  hunter_shot: "猎人开枪",
  last_words: "遗言",
  skip: "弃权",
};

export function humanActionControls(
  state: GameState,
  actionType: string,
  context: unknown,
): HumanActionControls {
  const parsed = parseHumanActionSchema(actionType, context);
  if (!parsed.ok) return parsed;

  const bySeat = new Map(state.seats.map((seat) => [seat.seat, seat]));
  const targets = parsed.schema.targetSeats.map((seat) => bySeat.get(seat));
  if (targets.some((seat) => seat === undefined)) {
    return { ok: false, reason: "advertised_target_missing_from_snapshot" };
  }
  return {
    ok: true,
    schema: parsed.schema,
    targets: targets as GameState["seats"],
  };
}

export function parseHumanActionSchema(
  actionType: string,
  context: unknown,
): { ok: true; schema: HumanActionSchema } | { ok: false; reason: string } {
  if (!isRecord(context)) return { ok: false, reason: "context_not_object" };
  if (typeof context.requested_action !== "string" || !context.requested_action) {
    return { ok: false, reason: "requested_action_missing" };
  }
  if (context.requested_action !== actionType) {
    return { ok: false, reason: "requested_action_mismatch" };
  }
  if (!isHumanActionName(context.requested_action)) {
    return { ok: false, reason: "unsupported_action" };
  }
  if (typeof context.requires_target !== "boolean") {
    return { ok: false, reason: "requires_target_missing" };
  }
  if (typeof context.can_skip !== "boolean") {
    return { ok: false, reason: "can_skip_missing" };
  }
  if (!Array.isArray(context.allowed_target_seats)) {
    return { ok: false, reason: "allowed_targets_missing" };
  }

  const targetSeats = context.allowed_target_seats;
  if (targetSeats.some((seat) => typeof seat !== "number" || !Number.isInteger(seat) || seat < 1)) {
    return { ok: false, reason: "allowed_target_invalid" };
  }
  if (new Set(targetSeats).size !== targetSeats.length) {
    return { ok: false, reason: "allowed_target_duplicate" };
  }

  const action = context.requested_action;
  const inputKind = isTextAction(action) ? "text" : "target";
  if (context.requires_target !== (inputKind === "target")) {
    return { ok: false, reason: "target_requirement_mismatch" };
  }
  if (!context.requires_target && targetSeats.length > 0) {
    return { ok: false, reason: "unexpected_allowed_targets" };
  }
  if (context.requires_target && targetSeats.length === 0 && !context.can_skip) {
    return { ok: false, reason: "request_has_no_terminal_choice" };
  }

  return {
    ok: true,
    schema: {
      action,
      requiresTarget: context.requires_target,
      canSkip: context.can_skip,
      targetSeats: [...targetSeats],
      inputKind,
    },
  };
}

export function resolveAction(actionType: string, context: unknown): string {
  const parsed = parseHumanActionSchema(actionType, context);
  return parsed.ok ? parsed.schema.action : "";
}

export function isTextAction(action: string): action is "speak" | "last_words" {
  return TEXT_ACTIONS.has(action as "speak" | "last_words");
}

export function actionHint(action: string): string {
  return {
    night_kill: "选择今晚的击杀目标。",
    see: "选择你要查验的玩家。",
    save: "确认是否救下今晚被击杀的玩家。",
    poison: "选择你要毒杀的玩家，或弃权保留毒药。",
    guard: "选择你要守护的玩家。",
    hunter_shot: "选择你要带走的玩家。",
    vote: "选择你要放逐的玩家。",
    speak: "轮到你公开发言。",
    last_words: "留下你的遗言。",
  }[action] || "提交你的操作。";
}

function isHumanActionName(action: string): action is HumanActionName {
  return TARGET_ACTIONS.has(action as never) || TEXT_ACTIONS.has(action as never);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
