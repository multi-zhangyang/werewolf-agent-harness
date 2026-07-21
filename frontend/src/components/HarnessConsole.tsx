import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  CircleDot,
  Crosshair,
  Gavel,
  Network,
  SendHorizontal,
  TerminalSquare,
  UserRound,
  Vote,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { humanActionControls } from "@/lib/human-actions";
import {
  protocolRecords,
  type ActorResponseAttempt,
  type AgentToolLoopEvent,
  type AgentToolLoopTrace,
} from "@/lib/protocol-trace";
import type { RoomTraceItem } from "@/lib/api";
import type { GameState, LogEntry } from "@/lib/store";
import { RoleAvatar, roleLabel } from "./RoleAvatar";

export function HarnessConsole({
  state,
  roomId,
  mySeat,
  isPlay,
  isGod,
  protocolTrace,
  onHumanAction,
  onOpenActions,
  className,
}: {
  state: GameState;
  roomId: string;
  mySeat: number | null;
  isPlay: boolean;
  isGod: boolean;
  protocolTrace?: RoomTraceItem[];
  onHumanAction: (action: string, data: Record<string, any>) => boolean;
  onOpenActions?: () => void;
  className?: string;
}) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const entries = state.log;
  const pendingControls = useMemo(
    () => state.pendingHuman
      ? humanActionControls(state, state.pendingHuman.actionType, state.pendingHuman.context)
      : null,
    [state.pendingHuman, state.seats],
  );
  const pendingAction = pendingControls?.ok ? pendingControls.schema.action : "";
  const canType = Boolean(
    isPlay
    && state.pendingHuman
    && state.mySeat === mySeat
    && pendingControls?.ok
    && pendingControls.schema.inputKind === "text",
  );
  const needsActionPanel = Boolean(
    isPlay
    && state.pendingHuman
    && state.mySeat === mySeat
    && (!pendingControls?.ok || pendingControls.schema.inputKind !== "text"),
  );

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    viewport.scrollTop = viewport.scrollHeight;
  }, [entries.length]);

  return (
    <section className={cn("flex min-h-0 min-w-0 flex-1 flex-col bg-background", className)}>
      <HarnessRunHeader state={state} roomId={roomId} entryCount={entries.length} />

      <div ref={viewportRef} className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
        <div className="mx-auto w-full max-w-6xl space-y-3 px-3 py-4 sm:px-5 lg:px-7">
          {isGod && <ProtocolTracePanel items={protocolTrace || []} />}
          {entries.length === 0 ? (
            <EmptyRun />
          ) : (
            entries.map((entry, index) => (
              <HarnessEntry
                key={entry.id}
                entry={entry}
                sequence={index + 1}
                state={state}
                mySeat={mySeat}
                isGod={isGod}
              />
            ))
          )}
        </div>
      </div>

      <div className="shrink-0 border-t bg-card/90 px-3 py-3 backdrop-blur sm:px-5">
        <div className="mx-auto w-full max-w-5xl">
          {canType ? (
            <HumanTextAction
              state={state}
              action={pendingAction}
              canSkip={pendingControls?.ok === true && pendingControls.schema.canSkip}
              onHumanAction={onHumanAction}
            />
          ) : needsActionPanel ? (
            <Alert>
              <Crosshair className="size-4" />
              <AlertTitle>ActionRequest 等待输入</AlertTitle>
              <AlertDescription className="flex flex-wrap items-center justify-between gap-3">
                <span>当前动作需要从 environment 提供的合法目标中选择。</span>
                <Button size="sm" onClick={onOpenActions}>打开动作面板</Button>
              </AlertDescription>
            </Alert>
          ) : (
            <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-muted-foreground">
              <span>{runStatusText(state)}</span>
              <span className="flex flex-wrap gap-1.5">
                <Badge variant="outline">{state.connected ? "WS connected" : "WS reconnecting"}</Badge>
                <Badge variant="outline">{state.seats.filter((seat) => seat.alive).length}/{state.seats.length} active</Badge>
              </span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export function ProtocolTracePanel({ items }: { items: RoomTraceItem[] }) {
  const records = useMemo(() => protocolRecords(items).reverse(), [items]);
  return (
    <Card size="sm" className="overflow-hidden border-primary/25">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <TerminalSquare className="size-4" />
          ActionRequest / DecisionEnvelope
        </CardTitle>
        <CardDescription>
          admin-only read projection · requests={records.length} · no scripted replay
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        {records.length === 0 ? (
          <p className="text-sm text-muted-foreground">等待 agent protocol trace。</p>
        ) : records.map((record) => (
          <Collapsible key={record.requestId}>
            <CollapsibleTrigger asChild>
              <Button variant="outline" className="h-auto w-full justify-start gap-2 px-3 py-2 text-left">
                <span className={cn(
                  "size-2 shrink-0 rounded-full",
                  record.failure
                    ? "bg-destructive"
                    : !record.response
                      ? "bg-amber-500"
                      : record.response.accepted
                        ? "bg-emerald-500"
                        : "bg-destructive",
                )} />
                <span className="min-w-0 flex-1">
                  <span className="flex flex-wrap items-center gap-1.5">
                    <code className="text-[11px]">#{record.sequence}</code>
                    <Badge variant="outline">seat={record.seat ?? "?"}</Badge>
                    <Badge variant="outline">{record.phase || "?"}/{record.action || "?"}</Badge>
                    <Badge variant={record.failure || record.response?.accepted === false ? "destructive" : "outline"}>
                      {record.failure
                        ? "failed"
                        : record.response
                          ? (record.response.accepted ? "accepted" : "rejected")
                          : "pending"}
                    </Badge>
                    {record.attempts.length > 0 && (
                      <Badge
                        variant={record.attempts.some((attempt) => attempt.status !== "accepted") ? "secondary" : "outline"}
                      >
                        attempts={record.attempts.length}
                      </Badge>
                    )}
                    {record.toolLoop && record.toolLoop.events.length > 0 && (
                      <Badge variant="secondary">
                        tool_events={record.toolLoop.events.length}
                      </Badge>
                    )}
                  </span>
                  <code className="mt-1 block truncate text-[11px] text-muted-foreground">request_id={record.requestId}</code>
                </span>
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-2 space-y-3 rounded-md border bg-muted/20 p-3">
              <div className="flex flex-wrap gap-1.5">
                {record.legalActions.map((legal, index) => (
                  <Badge key={`${legal.action}-${index}`} variant="outline">
                    legal={legal.action} targets=[{legal.targetSeats.join(",")}] skip={String(legal.canSkip)}
                  </Badge>
                ))}
              </div>
              <ActorResponseAttempts attempts={record.attempts} />
              <AgentToolLoopPanel trace={record.toolLoop} />
              {record.response && (
                <div className="space-y-2 text-xs text-muted-foreground">
                  <div className="grid gap-x-4 gap-y-1 sm:grid-cols-2">
                    <TraceValue label="model_call_id" value={record.response.modelCallId} />
                    <TraceValue label="parse_status" value={record.response.parseStatus} />
                    <TraceValue label="latency_seconds" value={record.response.latencySeconds} />
                    <TraceValue label="decision_action" value={record.response.decisionAction} />
                    <TraceValue label="target_seat" value={record.response.targetSeat} />
                    <TraceValue label="prompt_hash" value={record.response.promptHash} />
                    <TraceValue label="response_hash" value={record.response.responseHash} />
                  </div>
                  {record.response.issues.length > 0 && (
                    <p className="text-destructive">validation_issues=[{record.response.issues.join(", ")}]</p>
                  )}
                  {record.response.speech && (
                    <div>
                      <p className="mb-1 font-medium text-foreground">exact public output</p>
                      <p className="whitespace-pre-wrap break-words rounded-md border bg-background p-2 text-sm leading-6 text-foreground">
                        {record.response.speech}
                      </p>
                    </div>
                  )}
                  {record.response.reasoning && (
                    <div>
                      <p className="mb-1 font-medium text-foreground">private reasoning</p>
                      <pre className="max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-md bg-background p-2 text-xs leading-5">
                        {record.response.reasoning}
                      </pre>
                    </div>
                  )}
                </div>
              )}
              {record.failure ? (
                <div className="space-y-2 rounded-md border border-destructive/40 bg-destructive/5 p-3 text-xs">
                  <p className="font-medium text-destructive">
                    {record.failure.envelopeProduced
                      ? "DecisionEnvelope produced; Harness validator failed"
                      : "DecisionEnvelope not produced"}
                  </p>
                  <div className="grid gap-x-4 gap-y-1 text-muted-foreground sm:grid-cols-2">
                    <TraceValue label="failure.error_type" value={record.failure.errorType} />
                    <TraceValue label="failure.timeout" value={String(record.failure.timeout)} />
                    <TraceValue label="failure.timeout_seconds" value={record.failure.timeoutSeconds} />
                  </div>
                  {record.failure.reason && (
                    <p className="break-words text-muted-foreground">
                      failure.reason=<code className="text-foreground">{record.failure.reason}</code>
                    </p>
                  )}
                </div>
              ) : !record.response ? (
                <p className="text-xs text-muted-foreground">等待该 ActionRequest 的终态。</p>
              ) : null}
            </CollapsibleContent>
          </Collapsible>
        ))}
      </CardContent>
    </Card>
  );
}

function AgentToolLoopPanel({ trace }: { trace?: AgentToolLoopTrace }) {
  if (!trace || trace.events.length === 0) return null;
  return (
    <div className="space-y-2 rounded-md border border-primary/20 bg-background/70 p-2 text-xs">
      <div className="flex min-w-0 flex-wrap items-center gap-1.5">
        <Activity className="size-3.5 shrink-0 text-primary" />
        <span className="font-medium text-foreground">agent tool loop</span>
        <Badge variant="outline">generations={trace.generationCount}</Badge>
        {trace.generationFailureCount > 0 && (
          <Badge variant="destructive">generation_failures={trace.generationFailureCount}</Badge>
        )}
        <Badge variant="outline">calls={trace.toolCallCount}</Badge>
        <Badge variant="outline">results={trace.toolResultCount}</Badge>
        <Badge variant={trace.terminalActionCount ? "outline" : "secondary"}>
          terminal={trace.terminalActionCount}
        </Badge>
        {trace.historyCompactionCount > 0 && (
          <Badge variant="outline">history_compactions={trace.historyCompactionCount}</Badge>
        )}
        {trace.historyLimitMissCount > 0 && (
          <Badge variant="secondary">history_limit_misses={trace.historyLimitMissCount}</Badge>
        )}
        {trace.truncated && <Badge variant="secondary">events_truncated</Badge>}
      </div>
      <ol className="max-h-[32rem] space-y-1.5 overflow-y-auto pr-1">
        {trace.events.map((event, index) => (
          <AgentToolLoopEventRow key={`${event.sequence}-${event.type}-${event.callId || index}`} event={event} />
        ))}
      </ol>
    </div>
  );
}

function AgentToolLoopEventRow({ event }: { event: AgentToolLoopEvent }) {
  const failed = event.type === "model_generation_failed"
    || (event.type === "tool_result" && event.ok === false);
  const terminal = event.type === "tool_result" && event.terminal === true;
  const historyLimitMiss = event.type === "agent_history_compacted"
    && event.limitSatisfied === false;
  return (
    <li className={cn(
      "min-w-0 rounded border bg-muted/20 px-2 py-1.5",
      failed && "border-destructive/40 bg-destructive/5",
      terminal && "border-emerald-500/40 bg-emerald-500/5",
    )}>
      <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
        {failed ? (
          <AlertTriangle className="size-3.5 shrink-0 text-destructive" />
        ) : historyLimitMiss ? (
          <AlertTriangle className="size-3.5 shrink-0 text-amber-600" />
        ) : terminal ? (
          <CheckCircle2 className="size-3.5 shrink-0 text-emerald-600" />
        ) : (
          <CircleDot className="size-3.5 shrink-0 text-muted-foreground" />
        )}
        <Badge variant={failed ? "destructive" : terminal ? "outline" : "secondary"} className="h-5 px-1.5 text-[10px]">
          {toolLoopEventLabel(event.type)}
        </Badge>
        {event.step !== undefined && <code>step={event.step}</code>}
        {event.tool && <code className="max-w-full break-all text-foreground">tool={event.tool}</code>}
        {event.kind && <code className="text-muted-foreground">kind={event.kind}</code>}
        {event.ok !== undefined && <code className={event.ok ? "text-emerald-700" : "text-destructive"}>ok={String(event.ok)}</code>}
        {event.terminal !== undefined && <code>terminal={String(event.terminal)}</code>}
      </div>

      <div className="mt-1 flex min-w-0 flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
        {event.type === "agent_turn_started" && (
          <>
            <TraceValue label="session_id" value={event.sessionId} />
            <TraceValue label="turn_id" value={event.turnId} />
            <TraceValue label="phase" value={event.phase} />
            <TraceValue label="day" value={event.day} />
          </>
        )}
        <TraceValue label="trace_seq" value={event.sequence} />
        <TraceValue label="state_version" value={event.stateVersion} />
        <TraceValue label="call_id" value={event.callId} />
        <TraceValue label="latency" value={event.latencySeconds} />
        <TraceValue label="tool_count" value={event.toolCount} />
        <TraceValue label="tool_calls" value={event.toolCallCount} />
        <TraceValue label="response_attempt" value={event.responseAttempt} />
        <TraceValue label="will_retry" value={event.willRetry === undefined ? undefined : String(event.willRetry)} />
        <TraceValue label="target_seat" value={event.targetSeat} />
        <TraceValue label="action" value={event.action} />
        <TraceValue label="arguments_hash" value={event.argumentsHash} />
        <TraceValue label="output_hash" value={event.outputHash} />
        <TraceValue label="request_hash" value={event.requestHash} />
        <TraceValue label="response_hash" value={event.responseHash} />
        <TraceValue label="history_messages_before" value={event.originalMessageCount} />
        <TraceValue label="history_messages_after" value={event.modelMessageCount} />
        <TraceValue label="history_chars_before" value={event.originalChars} />
        <TraceValue label="history_chars_after" value={event.modelChars} />
        <TraceValue label="compacted_groups" value={event.compactedToolGroups} />
        <TraceValue
          label="history_limit_satisfied"
          value={event.limitSatisfied === undefined ? undefined : String(event.limitSatisfied)}
        />
        <TraceValue label="model_history_hash" value={event.modelHistoryHash} />
      </div>

      {event.usage && (
        <div className="mt-1 flex min-w-0 flex-wrap gap-1">
          {Object.entries(event.usage).map(([key, value]) => (
            <Badge key={key} variant="outline" className="h-5 px-1.5 font-mono text-[10px]">
              {key}={value}
            </Badge>
          ))}
        </div>
      )}

      {event.type === "model_generation" && (
        <>
          {event.content && <TraceText label="model content" value={event.content} />}
          {event.reasoning && <TraceText label="private reasoning" value={event.reasoning} />}
        </>
      )}
      {event.type === "tool_call_requested" && event.argumentsText && (
        <TraceText label="private tool arguments" value={event.argumentsText} />
      )}
      {failed && (event.errorCode || event.errorMessage) && (
        <div className="mt-1 min-w-0 break-words text-destructive">
          {event.errorCode && <code>error={event.errorCode}</code>}
          {event.errorMessage && <span className="ml-2 text-muted-foreground">{event.errorMessage}</span>}
        </div>
      )}
    </li>
  );
}

function toolLoopEventLabel(type: AgentToolLoopEvent["type"]): string {
  switch (type) {
    case "agent_turn_started": return "turn started";
    case "agent_history_compacted": return "history window";
    case "model_generation": return "model generation";
    case "model_generation_failed": return "model generation failed";
    case "tool_call_requested": return "tool call";
    case "tool_result": return "tool result";
    case "agent_action_submitted": return "action submitted";
  }
}

function TraceText({ label, value }: { label: string; value: string }) {
  return (
    <div className="mt-1 min-w-0">
      <p className="mb-1 font-medium text-foreground">{label}</p>
      <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words rounded-md bg-background p-2 text-[11px] leading-5 text-foreground">
        {value}
      </pre>
    </div>
  );
}

function ActorResponseAttempts({ attempts }: { attempts: ActorResponseAttempt[] }) {
  if (attempts.length === 0) return null;
  return (
    <div className="space-y-1.5 rounded-md border bg-background/70 p-2 text-xs">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="font-medium text-foreground">actor response attempts</span>
        <code className="text-[11px] text-muted-foreground">count={attempts.length}</code>
      </div>
      <ol className="space-y-1">
        {attempts.map((attempt, index) => (
          <li key={`${attempt.attempt ?? "unknown"}-${index}`} className="min-w-0 rounded border bg-muted/20 px-2 py-1.5">
            <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
              <code>attempt={attempt.attempt ?? "?"}</code>
              <Badge
                variant={attempt.status === "accepted" ? "outline" : "destructive"}
                className="h-4 px-1.5 text-[10px]"
              >
                status={attempt.status || "unknown"}
              </Badge>
              <code className="break-all text-muted-foreground">
                error_type={attempt.errorType || "-"}
              </code>
            </div>
            {attempt.validationIssues.length > 0 && (
              <div className="mt-1 flex min-w-0 flex-wrap gap-1 text-destructive">
                {attempt.validationIssues.map((issue, issueIndex) => (
                  <code
                    key={`${issue.path}-${issue.code}-${issueIndex}`}
                    className="max-w-full break-all rounded bg-destructive/10 px-1.5 py-0.5"
                  >
                    path={issue.path} code={issue.code}
                  </code>
                ))}
              </div>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

function TraceValue({ label, value }: { label: string; value: string | number | undefined }) {
  if (value === undefined || value === "") return null;
  return (
    <div className="min-w-0">
      <span>{label}=</span>
      <code className="break-all text-foreground">{String(value)}</code>
    </div>
  );
}

function HarnessRunHeader({ state, roomId, entryCount }: { state: GameState; roomId: string; entryCount: number }) {
  const failures = state.log.filter((entry) => entry.kind === "failed").length;
  return (
    <div className="shrink-0 border-b bg-card/70 px-3 py-3 sm:px-5">
      <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center gap-2">
        <div className="mr-auto flex min-w-0 items-center gap-2">
          <span className="grid size-8 shrink-0 place-items-center rounded-md border bg-background">
            <Network className="size-4" />
          </span>
          <span className="min-w-0">
            <span className="block truncate text-sm font-semibold">Agent Harness Run</span>
            <span className="block truncate font-mono text-[11px] text-muted-foreground">run_id={roomId}</span>
          </span>
        </div>
        <Badge variant="outline">status={state.status}</Badge>
        <Badge variant="outline">phase={state.phase}</Badge>
        <Badge variant="outline">day={state.day}</Badge>
        <Badge variant="outline">agents={state.seats.length}</Badge>
        <Badge variant="outline">events={entryCount}</Badge>
        <Badge variant={failures ? "destructive" : "outline"}>failures={failures}</Badge>
      </div>
    </div>
  );
}

function EmptyRun() {
  return (
    <Card className="mx-auto mt-10 max-w-2xl border-dashed">
      <CardHeader className="text-center">
        <div className="mx-auto grid size-12 place-items-center rounded-xl border bg-muted/40">
          <TerminalSquare className="size-5" />
        </div>
        <CardTitle>等待 environment 事件</CardTitle>
        <CardDescription>
          开始运行后，这里按 transcript 顺序展示规则阶段、agent 决策、公开输出、投票和失败。
        </CardDescription>
      </CardHeader>
    </Card>
  );
}

function HarnessEntry({
  entry,
  sequence,
  state,
  mySeat,
  isGod,
}: {
  entry: LogEntry;
  sequence: number;
  state: GameState;
  mySeat: number | null;
  isGod: boolean;
}) {
  const seat = entry.seat != null
    ? state.seats.find((candidate) => candidate.seat === entry.seat)
    : undefined;
  const isAgent = entry.seat != null && ["speech", "vote", "last_words", "hunter"].includes(entry.kind);
  const failed = entry.kind === "failed";
  const reveal = Boolean(seat?.role && (isGod || seat.seat === mySeat || !seat.alive || state.status === "ended"));
  const Icon = entryIcon(entry.kind);

  return (
    <Card size="sm" className={cn("overflow-hidden", failed && "border-destructive/50 bg-destructive/5")}>
      <CardContent className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)] gap-3 p-3 sm:p-4">
        {isAgent ? (
          <RoleAvatar
            role={seat?.role}
            team={seat?.team}
            seat={entry.seat}
            alive={seat?.alive ?? true}
            reveal={reveal}
          />
        ) : (
          <span className={cn(
            "grid size-9 place-items-center rounded-md border bg-muted/40",
            failed && "border-destructive/40 text-destructive",
          )}>
            <Icon className="size-4" />
          </span>
        )}

        <div className="min-w-0 space-y-2">
          <div className="flex min-w-0 flex-wrap items-center gap-1.5">
            <span className="font-mono text-[11px] text-muted-foreground">#{String(sequence).padStart(3, "0")}</span>
            <Badge variant="outline">D{entry.day}</Badge>
            <Badge variant={isAgent ? "secondary" : "outline"}>{entryKindLabel(entry.kind)}</Badge>
            <Badge variant="outline">source={isAgent ? `agent:${entry.seat}` : "environment"}</Badge>
            {seat && <span className="truncate text-sm font-medium">{seat.name}</span>}
            {seat?.seat === mySeat && <Badge>you</Badge>}
            {reveal && seat?.role && <Badge variant="outline">{roleLabel(seat.role)}</Badge>}
            {entry.targetSeat != null && <Badge variant="outline">target={entry.targetSeat}</Badge>}
          </div>

          {entry.text && (
            <p className="whitespace-pre-wrap break-words text-sm leading-6 text-foreground [overflow-wrap:anywhere]">
              {entry.text}
            </p>
          )}

          <EntrySignals entry={entry} />

        </div>
      </CardContent>
    </Card>
  );
}

function EntrySignals({ entry }: { entry: LogEntry }) {
  const items: ReactNode[] = [];
  if (entry.replyTo != null) items.push(<Badge key="reply" variant="outline">reply_to={entry.replyTo}</Badge>);
  if (entry.accuses?.length) items.push(<Badge key="accuse" variant="outline">accuses=[{entry.accuses.join(",")}]</Badge>);
  if (entry.claim) {
    items.push(
      <Badge key="claim" variant="outline">
        claim={entry.claim.role}:{entry.claim.checked_seat}:{entry.claim.result}
      </Badge>,
    );
  }
  if (entry.bid != null) items.push(<Badge key="bid" variant="outline">bid={entry.bid}</Badge>);
  if (!items.length) return null;
  return <div className="flex flex-wrap gap-1.5">{items}</div>;
}

function HumanTextAction({
  state,
  action,
  canSkip,
  onHumanAction,
}: {
  state: GameState;
  action: string;
  canSkip: boolean;
  onHumanAction: (action: string, data: Record<string, any>) => boolean;
}) {
  const [text, setText] = useState("");
  const [bid, setBid] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [now, setNow] = useState(Date.now());
  const [submittedAt, setSubmittedAt] = useState<number | null>(null);
  const requestId = state.pendingHuman?.requestId || "";
  const requiresBid = action === "speak";
  const expired = Boolean(state.pendingHuman && state.pendingHuman.deadline <= now);
  const submitted = submittedAt !== null;

  useEffect(() => {
    setText("");
    setBid(null);
    setError("");
    setSubmittedAt(null);
    setNow(Date.now());
  }, [requestId]);

  useEffect(() => {
    if (!state.pendingHuman) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [requestId, state.pendingHuman]);

  useEffect(() => {
    if (submitted && !state.connected) setSubmittedAt(null);
  }, [state.connected, submitted]);

  useEffect(() => {
    if (submittedAt === null) return;
    const rejected = [...state.log]
      .reverse()
      .find((entry) => (
        entry.kind === "failed"
        && entry.seat === state.mySeat
        && entry.ts >= submittedAt
        && entry.text.startsWith("真人操作被拒绝")
      ));
    if (rejected) {
      setSubmittedAt(null);
      setError("提交被 environment 拒绝，请修正后重试");
    }
  }, [state.log, state.mySeat, submittedAt]);

  const submit = () => {
    if (submitted || expired || !state.connected) return;
    const speech = text.trim();
    if (!speech) {
      setError("公开文本不能为空");
      return;
    }
    if (requiresBid && bid === null) {
      setError("请选择本次发言优先级");
      return;
    }
    setSubmittedAt(Date.now());
    if (!onHumanAction(action, { speech, ...(requiresBid ? { bid } : {}) })) {
      setSubmittedAt(null);
      setError("WebSocket 未接受该动作");
      return;
    }
    setError("");
  };

  const skip = () => {
    if (submitted || expired || !state.connected || !canSkip) return;
    setSubmittedAt(Date.now());
    if (!onHumanAction("skip", {})) {
      setSubmittedAt(null);
      setError("WebSocket 未接受该动作");
      return;
    }
    setError("");
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <Badge variant="outline">ActionRequest</Badge>
        <code>action={action}</code>
        <code>request_id={requestId.slice(0, 8)}</code>
      </div>
      {requiresBid && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-muted-foreground">发言优先级</span>
          {[
            { value: 1, label: "一般" },
            { value: 2, label: "重要" },
            { value: 3, label: "紧急" },
            { value: 4, label: "直接回应" },
          ].map(({ value, label }) => (
            <Button
              key={value}
              type="button"
              size="sm"
              variant={bid === value ? "default" : "outline"}
              onClick={() => {
                setBid(value);
                setError("");
              }}
              disabled={submitted || expired || !state.connected}
            >
              {value} · {label}
            </Button>
          ))}
        </div>
      )}
      <div className="flex items-end gap-2">
        <Textarea
          value={text}
          onChange={(event) => setText(event.currentTarget.value)}
          disabled={submitted || expired || !state.connected}
          placeholder="输入本次 action 的公开文本"
          className="min-h-16 resize-none"
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
        />
        <Button
          onClick={submit}
          disabled={submitted || expired || !state.connected || !text.trim() || (requiresBid && bid === null)}
          className="shrink-0 gap-1.5"
        >
          <SendHorizontal className="size-4" />
          {submitted ? "等待确认" : expired ? "已超时" : "提交"}
        </Button>
      </div>
      {canSkip && (
        <div className="flex justify-end">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={skip}
            disabled={submitted || expired || !state.connected}
          >
            {submitted ? "等待确认" : "明确弃权"}
          </Button>
        </div>
      )}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

function entryIcon(kind: string) {
  if (kind === "failed") return AlertTriangle;
  if (kind === "vote" || kind === "vote_resolved") return Vote;
  if (kind === "phase") return Activity;
  if (kind === "hunter") return Crosshair;
  if (kind === "speech" || kind === "last_words") return UserRound;
  if (kind === "last_words_skipped") return CircleDot;
  if (kind === "system") return CheckCircle2;
  if (kind === "night_resolved" || kind === "death") return Gavel;
  return CircleDot;
}

function entryKindLabel(kind: string): string {
  return {
    phase: "phase transition",
    speech: "public output",
    vote: "vote decision",
    vote_resolved: "vote resolution",
    vote_incomplete: "incomplete vote",
    last_words: "last words",
    last_words_skipped: "last words skipped",
    night_resolved: "night resolution",
    death: "state change",
    hunter: "hunter decision",
    failed: "decision failure",
    system: "run event",
  }[kind] || kind;
}

function runStatusText(state: GameState): string {
  if (state.status === "ended") return `run completed · winner=${state.winner || "unknown"}`;
  if (state.status === "incomplete") return "run incomplete · inspect termination reason";
  if (state.status === "failed" || state.status === "timeout") return `run ${state.status} · inspect failure events`;
  if (state.phase === "night") return "environment 正在收集夜间合法动作";
  if (state.phase === "day") return "environment 正在调度公开发言动作";
  if (state.phase === "voting" || state.phase === "pk") return "environment 正在收集投票动作";
  return "run waiting for the next environment transition";
}
