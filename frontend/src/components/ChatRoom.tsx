import { type ReactNode, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Brain,
  Circle,
  Crosshair,
  Gavel,
  MessageSquare,
  Reply,
  Send,
  Sparkles,
  Target,
  ThumbsDown,
  ThumbsUp,
  Vote,
} from "lucide-react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { GameState, LogEntry } from "../lib/store";
import { RoleAvatar, roleLabel } from "./RoleAvatar";

type ChatItem =
  | { id: string; type: "system"; entry: LogEntry }
  | { id: string; type: "message"; entry: LogEntry; thinkings?: LogEntry[] };

type PendingThinking = {
  index: number;
  entries: LogEntry[];
};

export function ChatRoom({
  state,
  mySeat,
  isPlay,
  isGod = false,
  onHumanAction,
  className,
}: {
  state: GameState;
  mySeat: number | null;
  isPlay: boolean;
  isGod?: boolean;
  onHumanAction: (action: string, data: Record<string, any>) => void;
  className?: string;
}) {
  const [input, setInput] = useState("");
  const viewportRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);
  const didInitialScrollRef = useRef(false);
  const pendingAction = state.pendingHuman ? resolveAction(state.pendingHuman.actionType, state.pendingHuman.context) : "";
  const canType = Boolean(isPlay && state.pendingHuman && state.mySeat === mySeat && ["speak", "last_words"].includes(pendingAction));
  const items = buildChatItems(state.log);
  const scrollKey = items
    .map((item) => {
      if (item.type === "system") return `${item.id}:${item.entry.text.length}`;
      const thinkingSize = (item.thinkings || []).reduce((sum, entry) => sum + entry.text.length + (entry.reasoning?.length || 0), 0);
      return `${item.id}:${item.entry.text.length}:${item.entry.reasoning?.length || 0}:${thinkingSize}`;
    })
    .join("|");
  const thinkingCount = state.log.filter((entry) => entry.kind === "thinking").length;

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    if (didInitialScrollRef.current && !stickToBottomRef.current) return;
    const behavior: ScrollBehavior = didInitialScrollRef.current ? "smooth" : "auto";
    const scroll = (nextBehavior: ScrollBehavior) => {
      viewport.scrollTo({ top: viewport.scrollHeight, behavior: nextBehavior });
      didInitialScrollRef.current = true;
      stickToBottomRef.current = true;
    };
    const frame = window.requestAnimationFrame(() => scroll(behavior));
    const followup = window.setTimeout(() => scroll("auto"), 100);
    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(followup);
    };
  }, [scrollKey]);

  const trackScroll = () => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const distance = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
    stickToBottomRef.current = distance < 120;
  };

  const send = () => {
    const text = input.trim();
    if (!canType || !text || !state.pendingHuman) return;
    onHumanAction(pendingAction, { speech: text });
    setInput("");
    stickToBottomRef.current = true;
  };

  return (
    <section className={cn("flex min-h-0 min-w-0 flex-col", className)}>
      <Card className="flex h-full min-h-0 flex-col bg-card/95 py-0 shadow-sm">
        <CardHeader className="shrink-0 border-b px-3 py-2.5 sm:px-4">
          <CardTitle className="flex flex-wrap items-center justify-between gap-3">
            <span className="flex min-w-0 items-center gap-2">
              <MessageSquare className="size-4 text-muted-foreground" />
              <span>证词流</span>
            </span>
            <span className="flex flex-wrap items-center gap-2 text-xs font-normal text-muted-foreground">
              <Badge variant="outline">{items.length} 条</Badge>
              {thinkingCount > 0 && <Badge variant="outline">{thinkingCount} 次思考</Badge>}
              <Badge variant={state.connected ? "outline" : "destructive"} className="gap-1.5">
                <Circle className={cn("size-2 fill-current", state.connected ? "text-emerald-500" : "text-current")} />
                {state.connected ? "实时" : "重连中"}
              </Badge>
            </span>
          </CardTitle>
        </CardHeader>

        <CardContent className="min-h-0 flex-1 px-0 pb-0">
          <ScrollArea
            className="h-full"
            viewportRef={viewportRef}
            viewportClassName="scroll-smooth"
            onViewportScroll={trackScroll}
          >
            <div className="mx-auto flex w-full max-w-5xl flex-col gap-4 px-3 py-4 sm:px-5">
              {items.length === 0 && <EmptyTranscript />}
              {items.map((item) => (
                <TranscriptItem key={item.id} item={item} state={state} mySeat={mySeat} isGod={isGod} />
              ))}
            </div>
          </ScrollArea>
        </CardContent>

        <CardFooter className="shrink-0 border-t bg-card px-3 py-2 sm:px-4">
          <div className="mx-auto flex w-full max-w-5xl gap-2">
            <Textarea
              value={input}
              disabled={!canType}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  send();
                }
              }}
              placeholder={canType ? "轮到你发言。Enter 发送，Shift+Enter 换行" : placeholder(state)}
              rows={1}
              className="max-h-28 min-h-11 resize-none text-base leading-7"
            />
            {canType && (
              <Button onClick={send} disabled={!input.trim()} className="mt-auto gap-2">
                <Send className="size-4" />
                发送
              </Button>
            )}
          </div>
        </CardFooter>
      </Card>
    </section>
  );
}

function TranscriptItem({
  item,
  state,
  mySeat,
  isGod,
}: {
  item: ChatItem;
  state: GameState;
  mySeat: number | null;
  isGod: boolean;
}) {
  if (item.type === "system") return <SystemEvent entry={item.entry} />;
  return <PlayerMessage item={item} state={state} mySeat={mySeat} isGod={isGod} />;
}

function SystemEvent({ entry }: { entry: LogEntry }) {
  const Icon = entry.kind === "failed" ? AlertTriangle : entry.kind === "vote_resolved" ? Vote : Gavel;
  return (
    <Alert className="mx-auto max-w-[min(680px,100%)] border-border/80 bg-muted/35 py-2 shadow-none">
      <Icon className="size-4" />
      <AlertDescription className="min-w-0 text-sm leading-6">
        <div className="mb-1 flex flex-wrap items-center gap-1.5">
          <Badge variant="outline">D{entry.day}</Badge>
          <Badge variant="outline">{kindLabel(entry.kind)}</Badge>
          {entry.seat != null && <Badge variant="outline">{entry.seat}号</Badge>}
          {entry.targetSeat != null && <Badge variant="outline">目标 {entry.targetSeat}号</Badge>}
        </div>
        <span className="text-foreground">{entry.text}</span>
      </AlertDescription>
    </Alert>
  );
}

function PlayerMessage({
  item,
  state,
  mySeat,
  isGod,
}: {
  item: Extract<ChatItem, { type: "message" }>;
  state: GameState;
  mySeat: number | null;
  isGod: boolean;
}) {
  const entry = item.entry;
  const seat = entry.seat ? state.seats.find((candidate) => candidate.seat === entry.seat) : null;
  const reveal = Boolean(seat?.role && (isGod || seat.seat === mySeat || !seat.alive || state.status === "ended"));
  const thinkings = item.thinkings || (entry.kind === "thinking" ? [entry] : []);
  const thinkingText = formatThinkingText(thinkings);
  const pendingOnly = entry.kind === "thinking";
  const action = entry.action || thinkings[thinkings.length - 1]?.action;
  const actionText = action ? actionLabel(action) : kindLabel(entry.kind);
  const finalText = pendingOnly ? "" : entry.text;

  return (
    <article className="flex min-w-0 gap-3 animate-in fade-in-0 slide-in-from-bottom-1 duration-200">
      <div className="shrink-0 pt-1">
        <RoleAvatar
          role={seat?.role}
          team={seat?.team}
          seat={entry.seat}
          alive={seat?.alive ?? true}
          reveal={reveal}
          size="lg"
        />
      </div>
      <Card size="sm" className="min-w-0 flex-1 gap-0 rounded-lg border-border/80 bg-background/85 py-0 shadow-none">
        <CardContent className="px-3 py-3 sm:px-4 sm:py-3.5">
          <div className="mb-2 grid min-w-0 gap-2 sm:flex sm:items-start sm:justify-between">
            <div className="min-w-0">
              <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
                <span className="min-w-0 break-words text-sm font-semibold sm:text-base">{entry.seat}号 · {seat?.name || "玩家"}</span>
                <Badge variant="secondary">{actionText}</Badge>
                {seat?.seat === mySeat && <Badge>你</Badge>}
                {pendingOnly && (
                  <Badge variant="outline" className="gap-1.5">
                    <Sparkles className="size-3" />
                    已记录思考
                  </Badge>
                )}
              </div>
              {entry.claim && <p className="mt-1 text-sm leading-6 text-muted-foreground">{claimLabel(entry.claim)}</p>}
            </div>
            <div className="flex flex-wrap gap-1.5 text-xs text-muted-foreground sm:shrink-0 sm:justify-end">
              <Badge variant="outline">D{entry.day}</Badge>
              {reveal && seat?.role && <Badge variant="outline">{roleLabel(seat.role)}</Badge>}
            </div>
          </div>

          {thinkingText && (
            <Accordion type="single" collapsible defaultValue={pendingOnly ? "reasoning" : undefined} className="mb-3">
              <AccordionItem value="reasoning" className="border-l border-muted-foreground/25 pl-3">
                <AccordionTrigger className="py-1 text-xs text-muted-foreground hover:no-underline">
                  <span className="flex items-center gap-2">
                    <Brain className="size-4" />
                    思考摘要
                  </span>
                </AccordionTrigger>
                <AccordionContent className="pb-2 text-sm leading-6 text-muted-foreground">
                  <p className="whitespace-pre-wrap break-words">{thinkingText}</p>
                  <ThinkingSignals thinkings={thinkings} />
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          )}

          {finalText ? (
            <div className="whitespace-pre-wrap break-words text-[16px] leading-8 text-foreground sm:text-[17px]">
              {finalText}
            </div>
          ) : (
            <div className="rounded-md bg-muted/30 px-3 py-2.5 text-sm leading-6 text-muted-foreground">
              {thinkingOnlyText(action)}
            </div>
          )}

          <MessageSignals entry={entry} />
        </CardContent>
      </Card>
    </article>
  );
}

function ThinkingSignals({ thinkings }: { thinkings: LogEntry[] }) {
  const latest = thinkings[thinkings.length - 1];
  if (!latest?.suspicionTop?.length && latest?.bid == null) return null;
  return (
    <>
      <Separator className="my-3" />
      <div className="flex flex-wrap gap-2">
        {latest.bid != null && <Badge variant="outline">发言优先级 {latest.bid}</Badge>}
        {(latest.suspicionTop || []).slice(0, 3).map((item) => (
          <Badge key={`${item.seat}-${item.suspicion}`} variant="outline">
            怀疑 {item.seat}号 {Math.round(item.suspicion * 100)}%
          </Badge>
        ))}
      </div>
    </>
  );
}

function SignalBadge({ children }: { children: ReactNode }) {
  return (
    <Badge variant="outline" className="h-6 bg-card px-2 font-normal">
      {children}
    </Badge>
  );
}

function MessageSignals({ entry }: { entry: LogEntry }) {
  const attitudes = Object.entries(entry.attitudes || {});
  const accuses = entry.accuses || [];
  const supports = attitudes.filter(([, stance]) => stance === "support");
  const opposes = attitudes.filter(([, stance]) => stance === "oppose");
  const observes = attitudes.filter(([, stance]) => stance !== "support" && stance !== "oppose");
  const hasAny = entry.replyTo || accuses.length || supports.length || opposes.length || observes.length || entry.deception || entry.objectiveSummary || entry.targetSeat != null;
  if (!hasAny) return null;

  return (
    <div className="mt-4 space-y-2 rounded-lg border bg-muted/20 px-3 py-2">
      {entry.targetSeat != null && (
        <SignalLine icon={<Crosshair className="size-3.5" />} label="目标">
          <SignalBadge>{entry.targetSeat}号</SignalBadge>
        </SignalLine>
      )}
      {entry.replyTo && (
        <SignalLine icon={<Reply className="size-3.5" />} label="回应">
          <SignalBadge>{entry.replyTo}号</SignalBadge>
        </SignalLine>
      )}
      {accuses.length > 0 && (
        <SignalLine icon={<Target className="size-3.5" />} label="指控">
          {accuses.map((seat) => <SignalBadge key={seat}>{seat}号</SignalBadge>)}
        </SignalLine>
      )}
      {supports.length > 0 && (
        <SignalLine icon={<ThumbsUp className="size-3.5" />} label="支持">
          {supports.map(([seat]) => <SignalBadge key={seat}>{seat}号</SignalBadge>)}
        </SignalLine>
      )}
      {opposes.length > 0 && (
        <SignalLine icon={<ThumbsDown className="size-3.5" />} label="反对">
          {opposes.map(([seat]) => <SignalBadge key={seat}>{seat}号</SignalBadge>)}
        </SignalLine>
      )}
      {observes.length > 0 && (
        <SignalLine icon={<MessageSquare className="size-3.5" />} label="观察">
          {observes.map(([seat, stance]) => <SignalBadge key={`${seat}-${stance}`}>{seat}号 · {stance}</SignalBadge>)}
        </SignalLine>
      )}
      {entry.deception && (
        <SignalLine icon={<Sparkles className="size-3.5" />} label="策略">
          <SignalBadge>{entry.deception}</SignalBadge>
        </SignalLine>
      )}
      {entry.objectiveSummary && (
        <p className="text-xs leading-5 text-muted-foreground">{entry.objectiveSummary}</p>
      )}
    </div>
  );
}

function SignalLine({ icon, label, children }: { icon: ReactNode; label: string; children: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
      <span className="inline-flex min-w-10 items-center gap-1 font-medium">
        {icon}
        {label}
      </span>
      <span className="flex flex-wrap gap-1.5">{children}</span>
    </div>
  );
}

function EmptyTranscript() {
  return (
    <Alert>
      <MessageSquare className="size-4" />
      <AlertDescription className="leading-6">
        等待真实事件进入。开始房间后，AI 发言、投票、死亡和阶段变化都会显示在这里。
      </AlertDescription>
    </Alert>
  );
}

function buildChatItems(log: LogEntry[]): ChatItem[] {
  const items: (ChatItem | null)[] = [];
  const pendingBySeat = new Map<number, PendingThinking[]>();
  const seen = new Set<string>();
  for (const entry of log) {
    const fingerprint = entryFingerprint(entry);
    if (seen.has(fingerprint)) continue;
    seen.add(fingerprint);

    if (entry.kind === "thinking") {
      const existing = findPendingThinking(pendingBySeat, entry);
      if (existing) {
        existing.entries.push(entry);
        const item = items[existing.index];
        if (item?.type === "message") {
          item.id = `thinking-${existing.entries.map((thinking) => thinking.id).join("-")}`;
          item.entry = entry;
          item.thinkings = [...existing.entries];
        }
        continue;
      }
      const item: ChatItem = { id: `thinking-${entry.id}`, type: "message", entry, thinkings: [entry] };
      const index = items.length;
      items.push(item);
      if (entry.seat != null) {
        const pending = pendingBySeat.get(entry.seat) || [];
        pending.push({ index, entries: [entry] });
        pendingBySeat.set(entry.seat, pending);
      }
      continue;
    }

    if (isContentEntry(entry)) {
      const thinking = takeMatchingThinking(pendingBySeat, entry);
      if (thinking) {
        items[thinking.index] = null;
        items.push({
          id: `message-${thinking.entries.map((candidate) => candidate.id).join("-")}-${entry.id}`,
          type: "message",
          entry,
          thinkings: thinking.entries,
        });
        continue;
      }
      items.push({ id: `message-${entry.id}`, type: "message", entry });
      continue;
    }

    items.push({ id: `system-${entry.id}`, type: "system", entry });
  }
  return items.filter((item): item is ChatItem => Boolean(item));
}

function isContentEntry(entry: LogEntry): boolean {
  if (entry.seat == null) return false;
  return ["speech", "last_words", "vote", "caucus", "hunter"].includes(entry.kind);
}

function findPendingThinking(pendingBySeat: Map<number, PendingThinking[]>, entry: LogEntry): PendingThinking | undefined {
  if (entry.seat == null) return undefined;
  const pending = pendingBySeat.get(entry.seat);
  if (!pending?.length) return undefined;
  return pending.find((candidate) => isSameThinkingGroup(candidate.entries[candidate.entries.length - 1], entry));
}

function takeMatchingThinking(pendingBySeat: Map<number, PendingThinking[]>, entry: LogEntry): PendingThinking | undefined {
  if (entry.seat == null) return undefined;
  const pending = pendingBySeat.get(entry.seat);
  if (!pending?.length) return undefined;
  for (let i = pending.length - 1; i >= 0; i--) {
    const candidate = pending[i];
    if (!candidate.entries.some((thinking) => isCompatibleThinking(thinking, entry))) continue;
    pending.splice(i, 1);
    if (!pending.length) pendingBySeat.delete(entry.seat);
    return candidate;
  }
  return undefined;
}

function isSameThinkingGroup(a: LogEntry, b: LogEntry): boolean {
  return a.seat === b.seat && a.day === b.day && (a.action || "") === (b.action || "");
}

function isCompatibleThinking(thinking: LogEntry, entry: LogEntry): boolean {
  if (thinking.seat !== entry.seat) return false;
  if (thinking.day !== entry.day) return false;
  if (!thinking.action) return isContentEntry(entry);
  if (thinking.action === "speak") return entry.kind === "speech";
  if (thinking.action === "last_words") return entry.kind === "last_words";
  if (thinking.action === "vote") return entry.kind === "vote";
  if (thinking.action === "wolf_caucus") return entry.kind === "caucus";
  if (thinking.action === "hunter_shot") return entry.kind === "hunter";
  if (["night_action", "night_kill", "see", "save", "poison", "guard"].includes(thinking.action)) {
    return ["death", "hunter", "vote_resolved", "system"].includes(entry.kind);
  }
  return false;
}

function formatThinkingText(thinkings: LogEntry[]): string {
  const seen = new Set<string>();
  const chunks: string[] = [];
  for (const thinking of thinkings) {
    const text = normalizeTextBlock(thinking.reasoning || thinking.text);
    if (!text || seen.has(text)) continue;
    seen.add(text);
    chunks.push(text);
  }
  return chunks.join("\n\n");
}

function entryFingerprint(entry: LogEntry): string {
  return [
    entry.kind,
    entry.day,
    entry.seat ?? "",
    entry.targetSeat ?? "",
    entry.action ?? "",
    normalizeText(entry.text),
    normalizeText(entry.reasoning || ""),
    stableStringify(entry.claim),
    stableStringify(entry.replyTo),
    stableStringify(entry.accuses),
    stableStringify(entry.attitudes),
    stableStringify(entry.suspicionTop),
    stableStringify(entry.bid),
    normalizeText(entry.objectiveSummary || ""),
  ].join("\u001f");
}

function normalizeText(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function normalizeTextBlock(text: string): string {
  return text.replace(/\n{3,}/g, "\n\n").trim();
}

function stableStringify(value: unknown): string {
  if (value == null) return "";
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${key}:${stableStringify(record[key])}`).join(",")}}`;
  }
  return String(value);
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

function placeholder(state: GameState): string {
  if (state.status === "ended") return "对局已结束";
  if (state.phase === "night") return "夜晚行动中";
  if (state.phase === "voting" || state.phase === "pk") return "投票阶段，请在行动面板选择目标";
  return "观战中，等待轮到你的座位发言";
}

function kindLabel(kind: string): string {
  return {
    phase: "阶段",
    system: "系统",
    vote: "投票",
    vote_resolved: "票决",
    vote_incomplete: "缺票",
    death: "死亡",
    hunter: "猎枪",
    failed: "失败",
    speech: "发言",
    thinking: "思考",
    last_words: "遗言",
    caucus: "狼队密谈",
    caucus_consensus: "狼队共识",
  }[kind] || kind;
}

function actionLabel(action: string): string {
  return {
    speak: "发言",
    vote: "投票",
    night_action: "夜间行动",
    night_kill: "击杀",
    wolf_caucus: "狼队密谈",
    see: "查验",
    save: "救人",
    poison: "毒人",
    guard: "守护",
    hunter_shot: "猎枪",
    last_words: "遗言",
    reflect: "复盘思考",
  }[action] || action;
}

function thinkingOnlyText(action?: string): string {
  if (!action) return "真实决策已记录，等待后续事件。";
  if (["night_action", "night_kill", "see", "save", "poison", "guard"].includes(action)) {
    return "夜间决策已提交，公开结算后显示结果。";
  }
  if (action === "wolf_caucus") return "狼队私聊提案已记录，仅上帝视角可见。";
  if (action === "vote") return "投票决策已记录，等待票型公开。";
  if (action === "hunter_shot") return "猎人开枪决策已记录，等待公开结果。";
  if (action === "last_words") return "遗言决策已记录，等待公开内容。";
  return "真实决策已记录，等待后续事件。";
}

function claimLabel(claim: any): string {
  if (claim.role === "seer") return `声称预言家 · 查验 ${claim.checked_seat ?? "?"}号`;
  if (claim.role) return `声称 ${roleLabel(claim.role)}`;
  return "有身份声明";
}
