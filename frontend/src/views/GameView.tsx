import { useEffect, useMemo, useState, type ElementType } from "react";
import {
  Activity,
  ArrowLeft,
  BarChart3,
  Clipboard,
  Loader2,
  Moon,
  Play,
  Radio,
  ShieldAlert,
  Skull,
  Sparkles,
  Sun,
  Trophy,
  Users,
  Vote,
  Wifi,
  WifiOff,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import type { GameState, LogEntry, SeatState } from "../lib/store";
import { startRoom } from "../lib/api";
import { ChatRoom } from "../components/ChatRoom";
import { HumanActionPanel } from "../components/HumanActionPanel";
import { RoleAvatar, roleLabel } from "../components/RoleAvatar";

const PHASE_LABEL: Record<string, string> = {
  setup: "准备",
  night: "夜晚",
  day: "白天讨论",
  voting: "投票",
  pk: "PK",
  ended: "结束",
};

const PHASE_ICON: Record<string, ElementType> = {
  setup: Sparkles,
  night: Moon,
  day: Sun,
  voting: Vote,
  pk: ShieldAlert,
  ended: Trophy,
};

export function GameView({
  state,
  roomId,
  seat,
  mode,
  adminToken,
  onHumanAction,
  onBack,
}: {
  state: GameState;
  roomId: string;
  seat: number | null;
  mode: string;
  adminToken?: string;
  onHumanAction: (action: string, data: Record<string, any>) => void;
  onBack: () => void;
}) {
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState("");
  const isGod = mode === "god" || mode === "replay";
  const isPlay = mode === "play";
  const mobileDefaultTab = state.pendingHuman ? "action" : "chat";
  const [mobileTab, setMobileTab] = useState(mobileDefaultTab);

  useEffect(() => {
    if (state.pendingHuman) setMobileTab("action");
  }, [state.pendingHuman]);

  const handleStartGame = async () => {
    setStarting(true);
    setStartError("");
    try {
      await startRoom(roomId, adminToken);
    } catch (error: any) {
      setStartError(String(error.message || error));
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="grid h-[calc(100svh-64px)] min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-2 overflow-hidden">
      <div className="space-y-2">
        <GameHeader
          state={state}
          roomId={roomId}
          mode={mode}
          canStart={Boolean(adminToken)}
          starting={starting}
          onStartGame={handleStartGame}
          onBack={onBack}
        />
        {(startError || state.error) && (
          <Alert variant="destructive">
            <AlertDescription>{startError || state.error}</AlertDescription>
          </Alert>
        )}
      </div>

      <div className="min-h-0">
        <div className="hidden h-full min-h-0 grid-cols-[280px_minmax(0,1fr)_320px] gap-3 xl:grid 2xl:grid-cols-[300px_minmax(0,1fr)_340px]">
          <SeatRail state={state} mySeat={seat} isGod={isGod} />
          <ChatRoom state={state} mySeat={seat} isPlay={isPlay} isGod={isGod} onHumanAction={onHumanAction} className="min-h-0" />
          <ContextRail state={state} roomId={roomId} mySeat={seat} mode={mode} isGod={isGod} isPlay={isPlay} onHumanAction={onHumanAction} />
        </div>

        <Tabs value={mobileTab} onValueChange={setMobileTab} className="flex h-full min-h-0 flex-col xl:hidden">
          <TabsList className="grid w-full shrink-0 grid-cols-3">
            <TabsTrigger value="chat" className="gap-1.5">
              <Radio className="size-4" />
              证词
            </TabsTrigger>
            <TabsTrigger value="seats" className="gap-1.5">
              <Users className="size-4" />
              座位
            </TabsTrigger>
            <TabsTrigger value="action" className="gap-1.5">
              <Activity className="size-4" />
              行动
            </TabsTrigger>
          </TabsList>
          <TabsContent value="chat" className="min-h-0 w-full flex-1 flex-col data-[state=active]:flex data-[state=inactive]:hidden">
            <ChatRoom state={state} mySeat={seat} isPlay={isPlay} isGod={isGod} onHumanAction={onHumanAction} className="h-full min-h-0" />
          </TabsContent>
          <TabsContent value="seats" className="min-h-0 w-full flex-1 flex-col data-[state=active]:flex data-[state=inactive]:hidden">
            <SeatRail state={state} mySeat={seat} isGod={isGod} />
          </TabsContent>
          <TabsContent value="action" className="min-h-0 w-full flex-1 flex-col data-[state=active]:flex data-[state=inactive]:hidden">
            <ContextRail state={state} roomId={roomId} mySeat={seat} mode={mode} isGod={isGod} isPlay={isPlay} onHumanAction={onHumanAction} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

function GameHeader({
  state,
  roomId,
  mode,
  canStart,
  starting,
  onStartGame,
  onBack,
}: {
  state: GameState;
  roomId: string;
  mode: string;
  canStart: boolean;
  starting: boolean;
  onStartGame: () => void;
  onBack: () => void;
}) {
  const PhaseIcon = PHASE_ICON[state.phase] || Activity;
  const alive = state.seats.filter((candidate) => candidate.alive).length;
  const title = winnerLabel(state.winner) || `${PHASE_LABEL[state.phase] || state.phase} · 第 ${state.day || 0} 天`;

  return (
    <Card className="shrink-0 bg-card/95 py-0 shadow-sm">
      <CardContent className="grid gap-2 px-3 py-2 md:flex md:min-h-12 md:items-center md:gap-3 md:px-4">
        <div className="flex min-w-0 items-center gap-2 md:flex-1">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-lg border bg-muted">
            <PhaseIcon className="size-4" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 items-center gap-2">
              <span className="min-w-0 text-base font-semibold leading-6 md:truncate">{title}</span>
              <Badge variant="outline" className="hidden md:inline-flex">{modeLabel(mode)}</Badge>
            </div>
            <p className="hidden truncate text-sm text-muted-foreground lg:block">{headline(state)}</p>
          </div>
        </div>

        <div className="flex min-w-0 flex-wrap items-center justify-start gap-1.5 md:justify-end">
          <Badge variant="outline" className="hidden max-w-[142px] font-mono sm:inline-flex md:max-w-[180px]">
            <span className="truncate">{roomId}</span>
          </Badge>
          <Button
            size="icon-sm"
            variant="ghost"
            onClick={() => navigator.clipboard?.writeText(roomId)}
            aria-label="复制房间号"
            className="hidden sm:inline-flex"
          >
            <Clipboard className="size-4" />
          </Button>
          <Badge variant="outline" className="md:hidden">{modeLabel(mode)}</Badge>
          <Badge variant={state.connected ? "default" : "destructive"} className="gap-1.5">
            {state.connected ? <Wifi className="size-3" /> : <WifiOff className="size-3" />}
            <span className="hidden sm:inline">{state.connected ? "实时连接" : "重连中"}</span>
            <span className="sm:hidden">{state.connected ? "实时" : "重连"}</span>
          </Badge>
          <Badge variant="outline">存活 {alive}/{state.seats.length || 0}</Badge>
          {state.status === "waiting" && state.phase === "setup" && canStart && (
            <Button size="sm" onClick={onStartGame} disabled={starting} className="gap-2">
              {starting ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
              {starting ? "启动中" : "开始真实对局"}
            </Button>
          )}
          <Button variant="outline" size="sm" onClick={onBack} className="gap-2">
            <ArrowLeft className="size-4" />
            <span className="hidden sm:inline">离开</span>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function SeatRail({ state, mySeat, isGod }: { state: GameState; mySeat: number | null; isGod: boolean }) {
  const seats = useMemo(() => [...state.seats].sort((a, b) => a.seat - b.seat), [state.seats]);
  const alive = seats.filter((candidate) => candidate.alive).length;
  return (
    <Card className="flex h-full min-h-0 w-full min-w-0 flex-col bg-card/95 py-0 shadow-sm">
      <CardHeader className="shrink-0 border-b px-3 py-2.5">
        <CardTitle className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-2">
            <Users className="size-4 text-muted-foreground" />
            座位
          </span>
          <Badge variant="outline">{alive}/{seats.length || 0}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 px-0 pb-0">
        <ScrollArea className="h-full">
          <div className="space-y-2 p-3">
            {seats.length === 0 && (
              <Alert>
                <AlertDescription>等待房间快照。</AlertDescription>
              </Alert>
            )}
            {seats.map((candidate) => (
              <SeatRow key={candidate.seat} seat={candidate} state={state} mySeat={mySeat} isGod={isGod} />
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

function SeatRow({
  seat,
  state,
  mySeat,
  isGod,
}: {
  seat: SeatState;
  state: GameState;
  mySeat: number | null;
  isGod: boolean;
}) {
  const reveal = Boolean((isGod || seat.seat === mySeat || !seat.alive || state.status === "ended") && seat.role);
  const active = state.speakingSeat === seat.seat || seat.isSpeaking;
  const incomingVotes = incomingVoteCount(state.votes, seat.seat);
  return (
    <Card
      size="sm"
      className={cn(
        "bg-background/65 py-0 shadow-none transition-colors",
        active && "border-primary/50 bg-primary/10 ring-1 ring-primary/30",
        seat.seat === mySeat && "border-foreground/25",
        !seat.alive && "bg-muted/40 opacity-75",
      )}
    >
      <CardContent className="flex items-center gap-3 px-3 py-2.5">
        <RoleAvatar role={seat.role} team={seat.team} seat={seat.seat} alive={seat.alive} reveal={reveal} />
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center gap-1.5">
            <span className="truncate font-medium">{seat.seat}号 · {seat.name}</span>
            {seat.seat === mySeat && <Badge>你</Badge>}
          </div>
          <div className="mt-1 flex flex-wrap gap-1.5">
            <Badge variant={reveal ? "secondary" : "outline"}>{reveal ? roleLabel(seat.role) : "身份隐藏"}</Badge>
            {active && <Badge>发言中</Badge>}
            {!seat.alive && <Badge variant="destructive">出局</Badge>}
            {seat.votedTarget !== undefined && <Badge variant="outline">投 {seat.votedTarget}</Badge>}
            {incomingVotes > 0 && <Badge variant="outline">被投 {incomingVotes}</Badge>}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function ContextRail({
  state,
  roomId,
  mySeat,
  mode,
  isGod,
  isPlay,
  onHumanAction,
}: {
  state: GameState;
  roomId: string;
  mySeat: number | null;
  mode: string;
  isGod: boolean;
  isPlay: boolean;
  onHumanAction: (action: string, data: Record<string, any>) => void;
}) {
  return (
    <Card className="flex h-full min-h-0 w-full min-w-0 flex-col bg-card/95 py-0 shadow-sm">
      <CardHeader className="shrink-0 border-b px-3 py-2.5">
        <CardTitle className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-2">
            <Activity className="size-4 text-muted-foreground" />
            当前局势
          </span>
          <Badge variant="outline">{modeLabel(mode)}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="min-h-0 flex-1 px-0 pb-0">
        <ScrollArea className="h-full">
          <div className="space-y-3 p-3">
            <PhasePanel state={state} />
            {isPlay && <HumanActionPanel state={state} onSubmit={onHumanAction} showTextEditor={false} />}
            {!isPlay && state.pendingHuman && (
              <Alert>
                <AlertDescription>真人座位正在操作。观战模式只显示公开事件。</AlertDescription>
              </Alert>
            )}
            <VotePanel state={state} />
            <FocusPanel state={state} mySeat={mySeat} isGod={isGod} />
            {(isGod || mode === "replay") && <ResearchPanel state={state} roomId={roomId} />}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

function PhasePanel({ state }: { state: GameState }) {
  const PhaseIcon = PHASE_ICON[state.phase] || Activity;
  const alive = state.seats.filter((seat) => seat.alive).length;
  return (
    <Card size="sm" className="bg-background/70 shadow-none">
      <CardContent className="space-y-3 px-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 font-medium">
              <PhaseIcon className="size-4" />
              {winnerLabel(state.winner) || PHASE_LABEL[state.phase] || state.phase}
            </div>
            <p className="mt-1 text-sm leading-6 text-muted-foreground">{headline(state)}</p>
          </div>
          <Badge variant="outline">第 {state.day || 0} 天</Badge>
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>阶段推进</span>
            <span>{alive} 名存活</span>
          </div>
          <Progress value={phaseProgress(state)} />
        </div>
      </CardContent>
    </Card>
  );
}

function VotePanel({ state }: { state: GameState }) {
  const alive = state.seats.filter((seat) => seat.alive).length;
  const cast = Object.keys(state.votes).length;
  const tally = voteTally(state);
  return (
    <Card size="sm" className="bg-background/70 shadow-none">
      <CardHeader className="px-3">
        <CardTitle className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-2">
            <Vote className="size-4" />
            投票态势
          </span>
          <Badge variant="outline">{cast}/{Math.max(alive, cast)}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 px-3">
        {tally.length === 0 ? (
          <p className="text-sm leading-6 text-muted-foreground">还没有公开投票。</p>
        ) : (
          tally.map((item) => {
            const target = state.seats.find((candidate) => candidate.seat === item.target);
            const value = Math.max(8, (item.count / Math.max(cast, 1)) * 100);
            return (
              <div key={item.target} className="space-y-1.5">
                <div className="flex items-center justify-between gap-2 text-sm">
                  <span className="truncate">{item.target}号{target ? ` · ${target.name}` : ""}</span>
                  <span className="font-medium">{item.count}票</span>
                </div>
                <Progress value={value} />
              </div>
            );
          })
        )}
      </CardContent>
    </Card>
  );
}

function FocusPanel({ state, mySeat, isGod }: { state: GameState; mySeat: number | null; isGod: boolean }) {
  const speaker = state.speakingSeat ? state.seats.find((seat) => seat.seat === state.speakingSeat) : undefined;
  const latest = dedupeLog(state.log).slice(-1)[0];
  return (
    <Card size="sm" className="bg-background/70 shadow-none">
      <CardHeader className="px-3">
        <CardTitle className="flex items-center gap-2">
          <Radio className="size-4" />
          现场焦点
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 px-3 text-sm leading-6">
        {speaker ? (
          <div className="flex items-center gap-2">
            <RoleAvatar role={speaker.role} team={speaker.team} seat={speaker.seat} alive={speaker.alive} reveal={isGod || speaker.seat === mySeat || !speaker.alive || state.status === "ended"} size="sm" />
            <span className="min-w-0 truncate font-medium">{speaker.seat}号 · {speaker.name}</span>
            <Badge>发言中</Badge>
          </div>
        ) : (
          <p className="text-muted-foreground">等待下一位玩家行动。</p>
        )}
        <DeathBadges state={state} />
        {latest && (
          <>
            <Separator />
            <div>
              <div className="mb-1 flex flex-wrap gap-1.5">
                <Badge variant="outline">最新事件</Badge>
                <Badge variant="outline">D{latest.day}</Badge>
                <Badge variant="outline">{kindLabel(latest.kind)}</Badge>
              </div>
              <p className="whitespace-pre-wrap break-words text-muted-foreground">{latest.text}</p>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function DeathBadges({ state }: { state: GameState }) {
  if (!state.lastDeaths.length) return <p className="text-muted-foreground">本轮暂无死亡信息。</p>;
  return (
    <div className="flex flex-wrap gap-2">
      {state.lastDeaths.map((death) => (
        <Badge key={`${death.seat}-${death.reason || "death"}`} variant="destructive" className="gap-1.5">
          <Skull className="size-3" />
          {death.seat}号死亡
        </Badge>
      ))}
    </div>
  );
}

function ResearchPanel({ state, roomId }: { state: GameState; roomId: string }) {
  const calls = Number(state.llmStats?.calls || 0);
  const tokens = Number(state.llmStats?.total_tokens_in || 0) + Number(state.llmStats?.total_tokens_out || 0);
  const summary = state.analysis?.quality?.game_summary;
  return (
    <Card size="sm" className="bg-background/70 shadow-none">
      <CardHeader className="px-3">
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="size-4" />
          复盘与指标
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 px-3 text-sm leading-6 text-muted-foreground">
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline" className="max-w-full font-mono">
            <span className="truncate">{roomId}</span>
          </Badge>
          {calls > 0 && <Badge variant="outline">模型请求 {formatCompact(calls)}</Badge>}
          {tokens > 0 && <Badge variant="outline">Token {formatCompact(tokens)}</Badge>}
          {state.analysis?.quality?.game_quality != null && (
            <Badge variant="outline">质量 {Math.round((state.analysis.quality.game_quality || 0) * 100)}/100</Badge>
          )}
        </div>
        <p>{summary || "上帝/复盘模式会展示后端下发的真实统计，不生成假指标。"}</p>
      </CardContent>
    </Card>
  );
}

function voteTally(state: GameState): { target: number; count: number }[] {
  const counts = new Map<number, number>();
  for (const target of Object.values(state.votes)) counts.set(target, (counts.get(target) || 0) + 1);
  return [...counts.entries()]
    .map(([target, count]) => ({ target, count }))
    .sort((a, b) => b.count - a.count || a.target - b.target);
}

function incomingVoteCount(votes: Record<number, number>, targetSeat: number): number {
  return Object.values(votes).filter((target) => target === targetSeat).length;
}

function phaseProgress(state: GameState): number {
  if (state.status === "ended" || state.phase === "ended") return 100;
  if (state.phase === "setup") return 8;
  if (state.phase === "night") return 28;
  if (state.phase === "day") return 55;
  if (state.phase === "voting") return 78;
  if (state.phase === "pk") return 88;
  return Math.min(96, 20 + Math.max(0, state.day) * 12);
}

function dedupeLog(log: LogEntry[]): LogEntry[] {
  const seen = new Set<string>();
  const out: LogEntry[] = [];
  for (const entry of log) {
    const key = [entry.kind, entry.day, entry.seat ?? "", entry.targetSeat ?? "", entry.text].join("\u001f");
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(entry);
  }
  return out;
}

function headline(state: GameState): string {
  if (state.winner) return `${winnerLabel(state.winner)}，可进入复盘查看真实分析。`;
  if (state.pendingHuman) return "真人座位需要操作。";
  if (state.speakingSeat) return `${state.speakingSeat}号正在发言。`;
  if (state.phase === "night") return "夜间行动处理中，等待公开结算。";
  if (state.phase === "voting" || state.phase === "pk") return "投票阶段，关注票型变化。";
  if (state.phase === "day") return "白天讨论阶段，关注证词与回应。";
  return "等待真实事件推进。";
}

function winnerLabel(winner: string | null): string {
  if (winner === "village") return "好人阵营获胜";
  if (winner === "werewolves") return "狼人阵营获胜";
  return winner ? `${winner} 获胜` : "";
}

function modeLabel(mode: string): string {
  return {
    spectate: "观战",
    play: "人机对局",
    god: "上帝视角",
    replay: "复盘",
  }[mode] || mode;
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

function formatCompact(value: number): string {
  return new Intl.NumberFormat("zh-CN", { notation: "compact" }).format(value);
}
