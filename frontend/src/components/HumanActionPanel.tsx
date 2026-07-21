import { useEffect, useMemo, useState } from "react";
import { AlertCircle, CheckCircle2, Clock3, Send, SkipForward, Target } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { ACTION_LABEL, actionHint, humanActionControls } from "@/lib/human-actions";
import type { GameState } from "@/lib/store";
import { RoleAvatar } from "./RoleAvatar";

export function HumanActionPanel({
  state,
  onSubmit,
  className,
}: {
  state: GameState;
  onSubmit: (action: string, data: Record<string, any>) => boolean;
  className?: string;
}) {
  const req = state.pendingHuman;
  const [target, setTarget] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const [submitStatus, setSubmitStatus] = useState<"ready" | "submitted" | "error">("ready");
  const [submittedAt, setSubmittedAt] = useState<number | null>(null);
  const controls = useMemo(
    () => req ? humanActionControls(state, req.actionType, req.context) : null,
    [req, state.seats],
  );
  const schema = controls?.ok ? controls.schema : null;
  const action = schema?.action || "";
  const targets = controls?.ok ? controls.targets : [];
  const targetAction = schema?.requiresTarget === true;
  const textAction = schema?.inputKind === "text";
  const canSkip = schema?.canSkip === true;
  const schemaError = controls && !controls.ok ? controls.reason : null;
  const targetKey = schema?.targetSeats.join(",") || "";
  const remainingMs = Math.max(0, (req?.deadline || now) - now);
  const timeoutRaw = Number(req?.timeoutMs || req?.context?.timeout_ms || req?.context?.timeout || 60_000);
  const timeoutMs = Math.max(1, timeoutRaw < 1000 ? timeoutRaw * 1000 : timeoutRaw);
  const progress = req ? Math.max(0, Math.min(100, (remainingMs / timeoutMs) * 100)) : 0;
  const expired = Boolean(req && remainingMs <= 0);
  const disconnected = !state.connected;
  const submitted = submitStatus === "submitted";

  useEffect(() => {
    if (!req) {
      setTarget(null);
      return;
    }
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [req]);

  useEffect(() => {
    setTarget(null);
    setSubmitStatus("ready");
    setSubmittedAt(null);
  }, [req?.requestId, req?.actionType, req?.deadline]);

  useEffect(() => {
    if (target !== null && (!schema || !schema.targetSeats.includes(target))) {
      setTarget(null);
    }
  }, [schema, target, targetKey]);

  useEffect(() => {
    if (submitStatus === "submitted" && !state.connected) {
      setSubmitStatus("ready");
      setSubmittedAt(null);
    }
  }, [state.connected, submitStatus]);

  useEffect(() => {
    if (submitStatus !== "submitted" || submittedAt == null) return;
    const rejected = [...state.log]
      .reverse()
      .find((entry) => (
        entry.kind === "failed"
        && entry.seat === state.mySeat
        && entry.ts >= submittedAt
        && entry.text.startsWith("真人操作被拒绝")
      ));
    if (rejected) {
      setSubmitStatus("ready");
      setSubmittedAt(null);
    }
  }, [state.log, state.mySeat, submitStatus, submittedAt]);

  if (!req) {
    return (
      <Card size="sm" className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle2 className="size-4 text-emerald-500" />
            暂无真人操作
          </CardTitle>
          <CardDescription>真实事件会在轮到你的座位时请求操作。</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const submitAction = (nextAction: string, data: Record<string, any>) => {
    if (expired || disconnected || submitted) return;
    setSubmitStatus("submitted");
    setSubmittedAt(Date.now());
    const ok = onSubmit(nextAction, data);
    if (!ok) {
      setSubmitStatus("error");
      setSubmittedAt(null);
    }
  };

  const submit = () => {
    if (targetAction) submitAction(action, { target_seat: target });
  };

  return (
    <Card size="sm" className={cn("border-primary/30", className)}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertCircle className="size-4" />
          轮到你了
        </CardTitle>
        <CardDescription>
          {schemaError
            ? "environment 返回的操作 schema 不完整，客户端已禁止提交。"
            : textAction
              ? "在 Harness Console 底部输入本次公开输出。"
              : actionHint(action)}
        </CardDescription>
        <CardAction>
          <Badge>{ACTION_LABEL[action] || req.actionType}</Badge>
        </CardAction>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <Clock3 className="size-3.5" />
              {expired ? "等待系统处理" : `剩余 ${Math.ceil(remainingMs / 1000)} 秒`}
            </span>
            <span>第 {state.day || 0} 天 · {phaseLabel(state.phase)}</span>
          </div>
          <Progress value={progress} />
        </div>

        {expired && (
          <Alert>
            <Clock3 className="size-4" />
            <AlertTitle>操作时间已到</AlertTitle>
            <AlertDescription>等待后端结算本次超时结果。</AlertDescription>
          </Alert>
        )}
        {disconnected && !expired && (
          <Alert variant="destructive">
            <AlertCircle className="size-4" />
            <AlertTitle>连接中断</AlertTitle>
            <AlertDescription>正在重连，连接恢复后再提交本次操作。</AlertDescription>
          </Alert>
        )}

        {schemaError && (
          <Alert variant="destructive">
            <AlertCircle className="size-4" />
            <AlertTitle>ActionRequest schema 无法安全渲染</AlertTitle>
            <AlertDescription>
              请求已 fail-closed（{schemaError}），等待 environment 终结或替换该请求。
            </AlertDescription>
          </Alert>
        )}

        {submitted && (
          <Alert>
            <CheckCircle2 className="size-4 text-emerald-500" />
            <AlertTitle>已提交，等待确认</AlertTitle>
            <AlertDescription>后端确认前会锁定本次选择，避免重复提交。</AlertDescription>
          </Alert>
        )}
        {submitStatus === "error" && !submitted && (
          <Alert variant="destructive">
            <AlertCircle className="size-4" />
            <AlertTitle>提交失败</AlertTitle>
            <AlertDescription>请检查连接状态后重新提交。</AlertDescription>
          </Alert>
        )}

        {textAction && (
          <Alert>
            <CheckCircle2 className="size-4 text-emerald-500" />
            <AlertTitle>公开输出在 Console 底部提交</AlertTitle>
            <AlertDescription>提交内容会作为本次 Decision 的公开文本原样进入 transcript。</AlertDescription>
          </Alert>
        )}

        {targetAction && (
          <ScrollArea className="h-[42svh] max-h-80" viewportClassName="pr-3">
            <div className="grid gap-2">
              {targets.map((seat) => {
                const selected = target === seat.seat;
                return (
                  <Button
                    key={seat.seat}
                    type="button"
                    variant={selected ? "default" : "outline"}
                    className="h-auto justify-start gap-3 px-3 py-2.5 text-left"
                    disabled={expired || disconnected || submitted}
                    onClick={() => {
                      setTarget(seat.seat);
                      if (submitStatus === "error") setSubmitStatus("ready");
                    }}
                  >
                    <RoleAvatar
                      role={seat.role}
                      team={seat.team}
                      seat={seat.seat}
                      alive={seat.alive}
                      reveal={seat.seat === state.mySeat || state.mode === "god" || state.status === "ended"}
                      size="default"
                    />
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
                  <AlertDescription>
                    {canSkip
                      ? "当前没有可选目标，可以显式弃权。"
                      : "当前请求没有合法目标且不允许弃权，environment 将记录失败。"}
                  </AlertDescription>
                </Alert>
              )}
            </div>
          </ScrollArea>
        )}
      </CardContent>

      <CardFooter className="justify-end gap-2">
        {canSkip && (
          <Button
            variant="ghost"
            onClick={() => submitAction("skip", {})}
            disabled={expired || disconnected || submitted}
            className="gap-2"
          >
            <SkipForward className="size-4" />
            {submitted ? "等待确认" : "弃权"}
          </Button>
        )}
        {targetAction && (
          <Button onClick={submit} disabled={expired || disconnected || submitted || target === null} className="gap-2">
            <Send className="size-4" />
            {submitted ? "等待确认" : "确认"}
          </Button>
        )}
      </CardFooter>
    </Card>
  );
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
