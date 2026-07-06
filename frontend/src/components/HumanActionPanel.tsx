import { useEffect, useMemo, useState } from "react";
import { AlertCircle, CheckCircle2, Clock3, Send, SkipForward, Target } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { GameState } from "../lib/store";
import { RoleAvatar } from "./RoleAvatar";

const ACTION_LABEL: Record<string, string> = {
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

export function HumanActionPanel({
  state,
  onSubmit,
  className,
  showTextEditor = true,
}: {
  state: GameState;
  onSubmit: (action: string, data: Record<string, any>) => void;
  className?: string;
  showTextEditor?: boolean;
}) {
  const req = state.pendingHuman;
  const [text, setText] = useState("");
  const [target, setTarget] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const actionType = req?.actionType || "";
  const action = resolveAction(actionType, req?.context);
  const textAction = action === "speak" || action === "last_words";
  const targetAction = needsTarget(action);
  const targets = useMemo(() => humanTargets(state, action, req?.context), [state, action, req?.context]);
  const remainingMs = Math.max(0, (req?.deadline || now) - now);
  const timeoutRaw = Number(req?.context?.timeout_ms || req?.context?.timeout || 60_000);
  const timeoutMs = Math.max(1, timeoutRaw < 1000 ? timeoutRaw * 1000 : timeoutRaw);
  const progress = req ? Math.max(0, Math.min(100, (remainingMs / timeoutMs) * 100)) : 0;

  useEffect(() => {
    if (!req) {
      setText("");
      setTarget(null);
      return;
    }
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [req]);

  useEffect(() => {
    setText("");
    setTarget(null);
  }, [req?.actionType, req?.deadline]);

  if (!req) {
    return (
      <Card className={cn("bg-card/95", className)} size="sm">
        <CardContent className="space-y-2 px-3">
          <div className="flex items-center gap-2 text-sm font-medium">
            <CheckCircle2 className="size-4 text-emerald-500" />
            暂无真人操作
          </div>
          <p className="text-sm leading-6 text-muted-foreground">真实事件会在轮到你的座位时请求操作。</p>
        </CardContent>
      </Card>
    );
  }

  const submit = () => {
    const data: Record<string, any> = {};
    if (textAction) data.speech = text.trim();
    if (targetAction) data.target_seat = target;
    onSubmit(action, data);
  };
  const disabled = (targetAction && target === null) || (textAction && showTextEditor && !text.trim());

  return (
    <Card className={cn("border-primary/35 bg-primary/5", className)} size="sm">
      <CardHeader className="px-3">
        <CardTitle className="flex flex-wrap items-center justify-between gap-2">
          <span className="flex items-center gap-2">
            <AlertCircle className="size-4" />
            轮到你了
          </span>
          <Badge>{ACTION_LABEL[action] || ACTION_LABEL[actionType] || actionType}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 px-3">
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <Clock3 className="size-3.5" />
              剩余 {Math.ceil(remainingMs / 1000)} 秒
            </span>
            <span>第 {state.day || 0} 天 · {phaseLabel(state.phase)}</span>
          </div>
          <Progress value={progress} />
        </div>

        <p className="text-sm leading-6 text-muted-foreground">{actionHint(action, showTextEditor)}</p>

        {textAction && showTextEditor && (
          <Textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder={action === "last_words" ? "留下遗言" : "输入你的发言"}
            rows={4}
            className="text-base leading-7"
          />
        )}

        {targetAction && (
          <div className="grid gap-2">
            {targets.map((seat) => {
              const selected = target === seat.seat;
              return (
                <Button
                  key={seat.seat}
                  type="button"
                  variant={selected ? "default" : "outline"}
                  className="h-auto justify-start gap-3 px-3 py-2.5 text-left"
                  onClick={() => setTarget(seat.seat)}
                >
                  <RoleAvatar role={seat.role} team={seat.team} seat={seat.seat} alive={seat.alive} reveal={seat.seat === state.mySeat || state.mode === "god" || state.status === "ended"} size="sm" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate font-medium">{seat.seat}号 · {seat.name}</span>
                    <span className="block text-xs text-muted-foreground">{seat.alive ? "存活" : "出局"}</span>
                  </span>
                  {selected && <Target className="size-4" />}
                </Button>
              );
            })}
            {targets.length === 0 && (
              <Alert>
                <AlertDescription>当前没有可选目标，可以选择弃权。</AlertDescription>
              </Alert>
            )}
          </div>
        )}

        <div className="flex flex-wrap justify-end gap-2">
          <Button variant="ghost" onClick={() => onSubmit("skip", {})} className="gap-2">
            <SkipForward className="size-4" />
            弃权
          </Button>
          {(showTextEditor || targetAction) && (
            <Button onClick={submit} disabled={disabled} className="gap-2">
              <Send className="size-4" />
              确认
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function resolveAction(actionType: string, context: any): string {
  const requested = typeof context?.requested_action === "string" ? context.requested_action : "";
  if (requested) return requested;
  if (actionType !== "night_action") return actionType;
  const role = String(context?.role || "");
  if (role === "werewolf") return "night_kill";
  if (role === "seer") return "see";
  if (role === "guard") return "guard";
  if (role === "witch") return "poison";
  if (role === "hunter") return "hunter_shot";
  return "skip";
}

function needsTarget(action: string): boolean {
  return ["night_kill", "see", "save", "poison", "guard", "hunter_shot", "vote"].includes(action);
}

function humanTargets(state: GameState, action: string, context: any): GameState["seats"] {
  if (action === "save") {
    const killedSeat = Number(context?.killed_seat);
    const target = state.seats.find((seat) => seat.seat === killedSeat && seat.alive);
    return target ? [target] : [];
  }
  if (needsTarget(action)) return state.seats.filter((seat) => seat.alive && seat.seat !== state.mySeat);
  return [];
}

function actionHint(action: string, showTextEditor: boolean): string {
  if ((action === "speak" || action === "last_words") && !showTextEditor) {
    return "在证词流底部输入并发送，发言会进入真实对局。";
  }
  return {
    night_kill: "选择今晚的击杀目标。",
    see: "选择你要查验的玩家。",
    save: "确认是否救下今晚被击杀的玩家。",
    poison: "选择你要毒杀的玩家，或弃权保留毒药。",
    guard: "选择你要守护的玩家。",
    hunter_shot: "选择你要带走的玩家。",
    vote: "选择你要放逐的玩家。",
    speak: "输入你的发言，参与白天讨论。",
    last_words: "留下你的遗言。",
  }[action] || "提交你的操作。";
}

function phaseLabel(phase: string): string {
  return {
    setup: "准备",
    night: "夜晚",
    day: "白天讨论",
    voting: "投票",
    pk: "PK",
    ended: "结束",
  }[phase] || phase;
}
