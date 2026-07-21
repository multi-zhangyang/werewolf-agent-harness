import { useEffect, useMemo, useRef, useState, type ElementType } from "react";
import {
  Activity,
  ArrowLeft,
  BarChart3,
  Clipboard,
  Loader2,
  Menu,
  Moon,
  Play,
  Radio,
  RefreshCw,
  ShieldAlert,
  Skull,
  Sparkles,
  Sun,
  Trophy,
  Vote,
  Wifi,
  WifiOff,
} from "lucide-react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { GameState, LogEntry, SeatState } from "../lib/store";
import type { AgentStrategyMetrics } from "../lib/types";
import { getTrace, startRoom, type RoomTraceItem } from "../lib/api";
import { HarnessConsole } from "../components/HarnessConsole";
import { HumanActionPanel } from "../components/HumanActionPanel";
import { RoleAvatar, roleLabel } from "../components/RoleAvatar";

type GamePanelTab = "seats" | "phase" | "votes" | "trace" | "action";

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
  onReconnect,
  onBack,
}: {
  state: GameState;
  roomId: string;
  seat: number | null;
  mode: string;
  adminToken?: string;
  onHumanAction: (action: string, data: Record<string, any>) => boolean;
  onReconnect: () => void;
  onBack: () => void;
}) {
  const isGod = mode === "god" || mode === "replay";
  const isPlay = mode === "play";
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState("");
  const [protocolTrace, setProtocolTrace] = useState<RoomTraceItem[]>([]);
  const [panelTab, setPanelTab] = useState<GamePanelTab>(() => (isGod ? "trace" : "seats"));
  const [mobileSheetOpen, setMobileSheetOpen] = useState(false);
  const traceAutoOpened = useRef(false);
  const traceContextRef = useRef<string | null>(null);
  const traceCursorRef = useRef<number | null>(null);
  const traceRequestRef = useRef<AbortController | null>(null);
  const traceTimerRef = useRef<number | null>(null);
  const roomStatusRef = useRef(state.status);
  roomStatusRef.current = state.status;
  const hasPendingHuman = Boolean(isPlay && state.pendingHuman && state.mySeat === seat);
  const hasTraceData = Boolean(isGod && (state.analysis || protocolTrace.length || state.log.some(hasTraceMetadata)));

  useEffect(() => {
    if (hasPendingHuman) setPanelTab("action");
  }, [hasPendingHuman, state.pendingHuman?.deadline]);

  useEffect(() => {
    if (!isGod && panelTab === "trace") setPanelTab("seats");
    // The initial God view already opens on Trace. Mark that one-time
    // automatic choice as consumed so an explicit user switch to Seats is
    // not immediately overridden on every render.
    if (isGod && panelTab === "trace") traceAutoOpened.current = true;
    if (isGod && hasTraceData && !hasPendingHuman && !traceAutoOpened.current && panelTab === "seats") {
      traceAutoOpened.current = true;
      setPanelTab("trace");
    }
  }, [hasTraceData, hasPendingHuman, isGod, panelTab]);

  useEffect(() => {
    const contextKey = `${roomId}\u0000${mode}\u0000${adminToken || ""}`;
    if (traceContextRef.current !== contextKey) {
      traceContextRef.current = contextKey;
      traceCursorRef.current = null;
      setProtocolTrace([]);
    }

    const shouldPoll = mode === "god" && Boolean(adminToken);
    if (!shouldPoll) return;

    const token = adminToken as string;
    let cancelled = false;
    let inFlight = false;

    const schedule = () => {
      if (
        cancelled
        || traceTimerRef.current !== null
        || roomStatusRef.current !== "running"
      ) return;
      traceTimerRef.current = window.setTimeout(() => {
        traceTimerRef.current = null;
        void load();
      }, 1500);
    };

    const load = async () => {
      if (cancelled || inFlight) return;
      inFlight = true;
      const controller = new AbortController();
      traceRequestRef.current = controller;
      try {
        const requestedCursor = traceCursorRef.current;
        let response = await getTrace(
          roomId,
          token,
          requestedCursor,
          controller.signal,
        );
        let replaceTrace = false;
        if (
          requestedCursor !== null
          && isTraceSequence(response.trace_seq)
          && response.trace_seq < requestedCursor
        ) {
          response = await getTrace(roomId, token, null, controller.signal);
          replaceTrace = true;
        }
        if (cancelled) return;

        const incoming = response.trace.filter((item) => item.kind === "decision");
        setProtocolTrace((previous) => mergeTraceItems(replaceTrace ? [] : previous, incoming));

        if (isTraceSequence(response.trace_seq)) {
          traceCursorRef.current = replaceTrace
            ? response.trace_seq
            : Math.max(traceCursorRef.current ?? 0, response.trace_seq);
        } else {
          const latestIncoming = response.trace.reduce(
            (latest, item) => isTraceSequence(item.trace_seq)
              ? Math.max(latest, item.trace_seq)
              : latest,
            replaceTrace ? 0 : traceCursorRef.current ?? 0,
          );
          if (replaceTrace || latestIncoming > (traceCursorRef.current ?? 0)) {
            traceCursorRef.current = latestIncoming;
          }
        }
      } catch (error) {
        // Aborts are expected when the room/mode changes or the view unmounts.
        // Other transient failures leave already collected evidence visible.
        if (isAbortError(error)) return;
      } finally {
        inFlight = false;
        if (traceRequestRef.current === controller) traceRequestRef.current = null;
        schedule();
      }
    };

    void load();

    return () => {
      cancelled = true;
      traceRequestRef.current?.abort();
      traceRequestRef.current = null;
      if (traceTimerRef.current !== null) {
        window.clearTimeout(traceTimerRef.current);
        traceTimerRef.current = null;
      }
    };
  }, [adminToken, mode, roomId, state.status]);

  useEffect(() => {
    if (!mobileSheetOpen) return;
    const desktopQuery = window.matchMedia("(min-width: 1280px)");
    const closeOnDesktop = () => {
      if (desktopQuery.matches) setMobileSheetOpen(false);
    };
    closeOnDesktop();
    desktopQuery.addEventListener("change", closeOnDesktop);
    return () => desktopQuery.removeEventListener("change", closeOnDesktop);
  }, [mobileSheetOpen]);

  const openActions = () => {
    setPanelTab("action");
    setMobileSheetOpen(true);
  };

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
    <TooltipProvider>
      <div className="flex h-full min-h-0 flex-col overflow-hidden bg-background">
        <GameTopBar
          state={state}
          roomId={roomId}
          mySeat={seat}
          mode={mode}
          isPlay={isPlay}
          canStart={Boolean(adminToken)}
          starting={starting}
          panelTab={panelTab}
          mobileSheetOpen={mobileSheetOpen}
          onPanelTabChange={setPanelTab}
          onMobileSheetOpenChange={setMobileSheetOpen}
          hasPendingHuman={hasPendingHuman}
          onStartGame={handleStartGame}
          onHumanAction={onHumanAction}
          onBack={onBack}
        />

        {(startError || state.error) && (
          <Alert variant="destructive" className="m-3 shrink-0">
            <AlertDescription className="flex flex-wrap items-center gap-2 break-words [overflow-wrap:anywhere]">
              <span className="min-w-0 flex-1">{startError || state.error}</span>
              {state.socketClose?.retryableByUser && !state.connected && (
                <Button size="sm" variant="outline" onClick={onReconnect}>
                  <RefreshCw className="size-4" />
                  手动重新连接
                </Button>
              )}
            </AlertDescription>
          </Alert>
        )}

        <div className="grid min-h-0 flex-1 grid-cols-1 xl:grid-cols-[minmax(0,1fr)_360px] 2xl:grid-cols-[minmax(0,1fr)_400px]">
          <HarnessConsole
            state={state}
            roomId={roomId}
            mySeat={seat}
            isPlay={isPlay}
            isGod={isGod}
            protocolTrace={protocolTrace}
            onHumanAction={onHumanAction}
            onOpenActions={openActions}
            className="min-h-0"
          />

          <aside className="hidden min-h-0 min-w-0 overflow-x-hidden border-l bg-muted/20 xl:flex">
          <GameSidebar
            state={state}
            roomId={roomId}
            mySeat={seat}
            isGod={isGod}
            isPlay={isPlay}
            tab={panelTab}
            onTabChange={setPanelTab}
              onHumanAction={onHumanAction}
            />
          </aside>
        </div>
      </div>
    </TooltipProvider>
  );
}

function GameTopBar({
  state,
  roomId,
  mySeat,
  mode,
  isPlay,
  canStart,
  starting,
  panelTab,
  mobileSheetOpen,
  onPanelTabChange,
  onMobileSheetOpenChange,
  hasPendingHuman,
  onStartGame,
  onHumanAction,
  onBack,
}: {
  state: GameState;
  roomId: string;
  mySeat: number | null;
  mode: string;
  isPlay: boolean;
  canStart: boolean;
  starting: boolean;
  panelTab: GamePanelTab;
  mobileSheetOpen: boolean;
  onPanelTabChange: (tab: GamePanelTab) => void;
  onMobileSheetOpenChange: (open: boolean) => void;
  hasPendingHuman: boolean;
  onStartGame: () => void;
  onHumanAction: (action: string, data: Record<string, any>) => boolean;
  onBack: () => void;
}) {
  const PhaseIcon = PHASE_ICON[state.phase] || Activity;
  const alive = state.seats.filter((candidate) => candidate.alive).length;
  const title = winnerLabel(state.winner) || `${PHASE_LABEL[state.phase] || state.phase} · 第 ${state.day || 0} 天`;

  return (
    <header className="flex min-h-14 shrink-0 items-center gap-2 border-b bg-background px-3 sm:px-4">
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg border bg-muted">
          <PhaseIcon className="size-4" />
        </div>
        <div className="min-w-0">
          <div className="truncate text-sm font-medium sm:text-base">{title}</div>
          <div className="hidden truncate text-xs text-muted-foreground sm:block">{headline(state)}</div>
        </div>
      </div>

      <div className="hidden items-center gap-1.5 md:flex">
        <Badge variant={state.connected ? "outline" : "destructive"} className="gap-1.5">
          {state.connected ? <Wifi className="size-3" /> : <WifiOff className="size-3" />}
          {state.connected
            ? "实时"
            : state.socketClose?.retryableByUser
              ? "已断开"
              : "重连"}
        </Badge>
        <Badge variant="outline">{modeLabel(mode)}</Badge>
        <Badge variant="outline">存活 {alive}/{state.seats.length || 0}</Badge>
        <Badge variant="outline" className="min-w-0 max-w-40 shrink font-mono">
          <span className="block min-w-0 max-w-full truncate">{roomId}</span>
        </Badge>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button size="icon-sm" variant="ghost" onClick={() => navigator.clipboard?.writeText(roomId)} aria-label="复制房间号">
              <Clipboard className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>复制房间号</TooltipContent>
        </Tooltip>
      </div>

      {state.status === "waiting" && state.phase === "setup" && canStart && (
        <StartGameConfirmButton starting={starting} onStartGame={onStartGame} compact />
      )}

      <MobileGameSheet
        state={state}
        roomId={roomId}
        mode={mode}
        mySeat={mySeat}
        isGod={mode === "god" || mode === "replay"}
        isPlay={isPlay}
        tab={panelTab}
        open={mobileSheetOpen}
        hasPendingHuman={hasPendingHuman}
        onTabChange={onPanelTabChange}
        onOpenChange={onMobileSheetOpenChange}
        onHumanAction={onHumanAction}
      />

      <Tooltip>
        <TooltipTrigger asChild>
          <Button variant="ghost" size="icon-sm" onClick={onBack} aria-label="返回房间">
            <ArrowLeft className="size-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>返回房间</TooltipContent>
      </Tooltip>
    </header>
  );
}

function MobileGameSheet({
  state,
  roomId,
  mode,
  mySeat,
  isGod,
  isPlay,
  tab,
  open,
  hasPendingHuman,
  onTabChange,
  onOpenChange,
  onHumanAction,
}: {
  state: GameState;
  roomId: string;
  mode: string;
  mySeat: number | null;
  isGod: boolean;
  isPlay: boolean;
  tab: GamePanelTab;
  open: boolean;
  hasPendingHuman: boolean;
  onTabChange: (tab: GamePanelTab) => void;
  onOpenChange: (open: boolean) => void;
  onHumanAction: (action: string, data: Record<string, any>) => boolean;
}) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetTrigger asChild>
        <Button
          variant={hasPendingHuman ? "default" : "outline"}
          size={hasPendingHuman ? "sm" : "icon-sm"}
          className="gap-1.5 xl:hidden"
          aria-label={hasPendingHuman ? "打开待处理行动" : "打开对局信息"}
        >
          <Menu className="size-4" />
          {hasPendingHuman && <span>行动</span>}
        </Button>
      </SheetTrigger>
      <SheetContent
        side="right"
        className="min-h-0 max-w-[92vw] overflow-hidden data-[side=right]:w-[92vw] sm:data-[side=right]:max-w-md"
      >
        <SheetHeader className="min-w-0 max-w-full shrink-0 pr-14">
          <SheetTitle className="min-w-0 truncate">{hasPendingHuman ? "轮到你行动" : "对局信息"}</SheetTitle>
          <SheetDescription className="min-w-0 break-words [overflow-wrap:anywhere]">{modeLabel(mode)} · {roomId}</SheetDescription>
        </SheetHeader>
        <ScrollArea className="min-h-0 min-w-0 flex-1">
          <div className="min-w-0 max-w-full overflow-x-hidden px-4 pb-[calc(1rem+env(safe-area-inset-bottom))]">
            <Tabs value={tab} onValueChange={(value) => onTabChange(value as GamePanelTab)} className="min-h-0 min-w-0">
              <TabsList className={cn("grid w-full min-w-0", isGod ? "grid-cols-5" : "grid-cols-4")}>
                <TabsTrigger value="seats" className="min-w-0 px-1">座位</TabsTrigger>
                <TabsTrigger value="phase" className="min-w-0 px-1">局势</TabsTrigger>
                <TabsTrigger value="votes" className="min-w-0 px-1">票型</TabsTrigger>
                {isGod && <TabsTrigger value="trace" className="min-w-0 px-1">Trace</TabsTrigger>}
                <TabsTrigger value="action" className="min-w-0 px-1">
                  行动
                  {hasPendingHuman && <span className="size-1.5 shrink-0 rounded-full bg-destructive" aria-hidden />}
                </TabsTrigger>
              </TabsList>
              <TabsContent value="seats" className="mt-4 min-w-0">
                <SeatList state={state} mySeat={mySeat} isGod={isGod} />
              </TabsContent>
              <TabsContent value="phase" className="mt-4 min-w-0 space-y-4">
                <PhaseOverview state={state} />
                <RecentEvents state={state} />
              </TabsContent>
              {isGod && (
                <TabsContent value="trace" className="mt-4 min-w-0">
                  <TracePanel state={state} roomId={roomId} />
                </TabsContent>
              )}
              <TabsContent value="votes" className="mt-4 min-w-0">
                <VoteOverview state={state} />
              </TabsContent>
              <TabsContent value="action" className="mt-4 min-w-0">
                {isPlay ? (
                  <HumanActionPanel state={state} onSubmit={onHumanAction} />
                ) : (
                  <Alert>
                    <AlertDescription>观战和上帝视角不会提交真人操作。</AlertDescription>
                  </Alert>
                )}
              </TabsContent>
            </Tabs>
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}

function StartGameConfirmButton({
  starting,
  onStartGame,
  compact = false,
}: {
  starting: boolean;
  onStartGame: () => void;
  compact?: boolean;
}) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button size="sm" disabled={starting} className="gap-2">
          {starting ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
          <span className={compact ? "hidden sm:inline" : undefined}>
            {starting ? "启动中" : "开始真实对局"}
          </span>
          {compact && <span className="sm:hidden">开始</span>}
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>开始真实对局</DialogTitle>
          <DialogDescription>
            这会立即启动后端游戏循环，并对 AI 座位发起真实模型调用。
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="outline">取消</Button>
          </DialogClose>
          <DialogClose asChild>
            <Button onClick={onStartGame} disabled={starting} className="gap-2">
              {starting ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
              确认开始
            </Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function GameSidebar({
  state,
  roomId,
  mySeat,
  isGod,
  isPlay,
  tab,
  onTabChange,
  onHumanAction,
}: {
  state: GameState;
  roomId: string;
  mySeat: number | null;
  isGod: boolean;
  isPlay: boolean;
  tab: GamePanelTab;
  onTabChange: (tab: GamePanelTab) => void;
  onHumanAction: (action: string, data: Record<string, any>) => boolean;
}) {
  const hasPendingHuman = Boolean(isPlay && state.pendingHuman);
  return (
    <div className="flex min-h-0 min-w-0 w-full flex-col overflow-x-hidden">
      <Tabs value={tab} onValueChange={(value) => onTabChange(value as GamePanelTab)} className="min-h-0 min-w-0 flex-1 gap-0">
        <div className="min-w-0 border-b p-3">
          <TabsList className={cn("grid w-full min-w-0", isGod ? "grid-cols-5" : "grid-cols-4")}>
            <TabsTrigger value="seats" className="min-w-0 px-1">座位</TabsTrigger>
            <TabsTrigger value="phase" className="min-w-0 px-1">局势</TabsTrigger>
            <TabsTrigger value="votes" className="min-w-0 px-1">票型</TabsTrigger>
            {isGod && <TabsTrigger value="trace" className="min-w-0 px-1">Trace</TabsTrigger>}
            <TabsTrigger value="action" className="min-w-0 px-1">
              行动
              {hasPendingHuman && <span className="size-1.5 shrink-0 rounded-full bg-destructive" aria-hidden />}
            </TabsTrigger>
          </TabsList>
        </div>
        <ScrollArea className="min-h-0 min-w-0 flex-1">
          <div className="min-w-0 max-w-full overflow-x-hidden p-3">
            <TabsContent value="seats" className="min-w-0">
              <SeatList state={state} mySeat={mySeat} isGod={isGod} />
            </TabsContent>
            <TabsContent value="phase" className="min-w-0 space-y-4">
              <PhaseOverview state={state} />
              <RecentEvents state={state} />
            </TabsContent>
            {isGod && (
              <TabsContent value="trace" className="min-w-0">
                <TracePanel state={state} roomId={roomId} />
              </TabsContent>
            )}
            <TabsContent value="votes" className="min-w-0">
              <VoteOverview state={state} />
            </TabsContent>
            <TabsContent value="action" className="min-w-0 space-y-4">
              {isPlay ? (
                <HumanActionPanel state={state} onSubmit={onHumanAction} />
              ) : (
                <Alert>
                  <AlertDescription>观战和上帝视角不会提交真人操作。</AlertDescription>
                </Alert>
              )}
            </TabsContent>
          </div>
        </ScrollArea>
      </Tabs>
    </div>
  );
}

function SeatList({ state, mySeat, isGod }: { state: GameState; mySeat: number | null; isGod: boolean }) {
  const seats = useMemo(() => [...state.seats].sort((a, b) => a.seat - b.seat), [state.seats]);
  if (!seats.length) {
    return (
      <Alert>
        <AlertDescription>等待房间快照。</AlertDescription>
      </Alert>
    );
  }
  return (
    <div className="space-y-2">
      {seats.map((candidate) => (
        <SeatRow key={candidate.seat} seat={candidate} state={state} mySeat={mySeat} isGod={isGod} />
      ))}
    </div>
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
        "flex-row items-center justify-start gap-3 px-3 py-2.5 text-left",
        active && "border-primary/50 bg-primary/10",
        seat.seat === mySeat && "border-foreground/25",
        !seat.alive && "opacity-70",
      )}
    >
      <RoleAvatar role={seat.role} team={seat.team} seat={seat.seat} alive={seat.alive} reveal={reveal} />
      <span className="min-w-0 flex-1">
        <span className="flex min-w-0 items-center gap-1.5">
          <span className="truncate font-medium">{seat.seat}号 · {seat.name}</span>
          {seat.seat === mySeat && <Badge>你</Badge>}
        </span>
        <span className="mt-1 flex flex-wrap gap-1.5">
          <Badge variant={reveal ? "secondary" : "outline"}>{reveal ? roleLabel(seat.role) : "身份隐藏"}</Badge>
          {active && <Badge>发言中</Badge>}
          {!seat.alive && <Badge variant="destructive">出局</Badge>}
          {seat.votedTarget !== undefined && <Badge variant="outline">投 {seat.votedTarget}</Badge>}
          {incomingVotes > 0 && <Badge variant="outline">被投 {incomingVotes}</Badge>}
        </span>
      </span>
    </Card>
  );
}

function PhaseOverview({ state }: { state: GameState }) {
  const PhaseIcon = PHASE_ICON[state.phase] || Activity;
  const alive = state.seats.filter((seat) => seat.alive).length;
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <PhaseIcon className="size-4" />
          {winnerLabel(state.winner) || PHASE_LABEL[state.phase] || state.phase}
        </CardTitle>
        <CardDescription>{headline(state)}</CardDescription>
        <CardAction>
          <Badge variant="outline">第 {state.day || 0} 天</Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>阶段推进</span>
            <span>{alive} 名存活</span>
          </div>
          <Progress value={phaseProgress(state)} />
        </div>
        <DeathBadges state={state} />
      </CardContent>
    </Card>
  );
}

function VoteOverview({ state }: { state: GameState }) {
  const alive = state.seats.filter((seat) => seat.alive).length;
  const cast = Object.keys(state.votes).length;
  const tally = voteTally(state);
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Vote className="size-4" />
          投票态势
        </CardTitle>
        <CardDescription>{tally.length === 0 ? "还没有公开投票。" : "当前公开票型统计。"}</CardDescription>
        <CardAction>
          <Badge variant="outline">{cast}/{Math.max(alive, cast)}</Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="space-y-3">
        {tally.length === 0 ? (
          <p className="text-sm leading-6 text-muted-foreground">等待真实投票事件。</p>
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

function RecentEvents({ state }: { state: GameState }) {
  const latest = state.log.filter(isTimelineEntry).slice(-4).reverse();
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Radio className="size-4" />
          最近事件
        </CardTitle>
        <CardDescription>最新公开事件和状态变化。</CardDescription>
      </CardHeader>
      <CardContent>
        {latest.length === 0 ? (
          <p className="text-sm leading-6 text-muted-foreground">等待真实事件。</p>
        ) : (
          <div className="space-y-3">
            {latest.map((entry, index) => (
              <div key={entry.id}>
                {index > 0 && <Separator className="mb-3" />}
                <div className="mb-1 flex flex-wrap gap-1.5">
                  <Badge variant="outline">D{entry.day}</Badge>
                  <Badge variant="outline">{kindLabel(entry.kind)}</Badge>
                  {entry.seat != null && <Badge variant="outline">{entry.seat}号</Badge>}
                </div>
                <p className="line-clamp-3 break-words text-sm leading-6 text-muted-foreground">{entry.text}</p>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function DeathBadges({ state }: { state: GameState }) {
  if (!state.lastDeaths.length) return <p className="text-sm text-muted-foreground">暂无近期死亡或放逐信息。</p>;
  return (
    <div className="flex flex-wrap gap-2">
      {state.lastDeaths.map((death) => (
        <Badge key={`${death.seat}-${death.reason || "death"}`} variant="destructive" className="gap-1.5">
          <Skull className="size-3" />
          {death.seat}号{death.reason === "exiled" ? "放逐" : "死亡"}
        </Badge>
      ))}
    </div>
  );
}

function TracePanel({ state, roomId }: { state: GameState; roomId: string }) {
  const calls = Number(state.llmStats?.calls || 0);
  const successes = Number(state.llmStats?.successes || 0);
  const failures = Number(state.llmStats?.failures || 0);
  const retries = Number(state.llmStats?.retries || 0);
  const tokensIn = Number(state.llmStats?.total_tokens_in || 0);
  const tokensOut = Number(state.llmStats?.total_tokens_out || 0);
  const traceEntries = state.log
    .filter((entry) => hasTraceMetadata(entry) || entry.kind === "failed")
    .slice(-40)
    .reverse();
  return (
    <Card size="sm" className="min-w-0 max-w-full overflow-hidden">
      <CardHeader className="min-w-0">
        <CardTitle className="flex min-w-0 items-center gap-2">
          <BarChart3 className="size-4" />
          <span className="min-w-0 truncate">Harness trace</span>
        </CardTitle>
        <CardDescription className="min-w-0 break-words [overflow-wrap:anywhere]">
          只显示真实运行计数、决策记录和失败，不展示推测性评分。
        </CardDescription>
      </CardHeader>
      <CardContent className="min-w-0 max-w-full space-y-3 overflow-x-hidden">
        <div className="flex min-w-0 max-w-full flex-wrap gap-2">
          <Badge variant="outline" className="min-w-0 max-w-full shrink font-mono">
            <span className="block min-w-0 max-w-full truncate">{roomId}</span>
          </Badge>
          <Badge variant="outline">calls={formatCompact(calls)}</Badge>
          <Badge variant="outline">success={formatCompact(successes)}</Badge>
          <Badge variant={failures ? "destructive" : "outline"}>provider_failures={formatCompact(failures)}</Badge>
          <Badge variant="outline">retries={formatCompact(retries)}</Badge>
          <Badge variant="outline">tokens_in={formatCompact(tokensIn)}</Badge>
          <Badge variant="outline">tokens_out={formatCompact(tokensOut)}</Badge>
        </div>
        <div className="rounded-md border bg-background/55 p-3 text-xs leading-5 text-muted-foreground">
          <div>run_status={state.status}</div>
          <div>phase={state.phase} day={state.day}</div>
          <div>transcript_events={state.log.length}</div>
          <div>decision_failures={state.log.filter((entry) => entry.kind === "failed").length}</div>
          <div>winner={state.winner || "pending"}</div>
        </div>

        {state.analysis?.agent_strategy_metrics && (
          <>
            <Separator />
            <StrategyMetricsPanel metrics={state.analysis.agent_strategy_metrics} />
          </>
        )}

        <Separator />
        <div className="min-w-0 max-w-full space-y-2 overflow-x-hidden">
          <div className="flex min-w-0 items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-2 text-sm font-medium">
              <Sparkles className="size-4" />
              <span className="truncate">真实 decision trace</span>
            </div>
            <Badge variant="outline">最近 {traceEntries.length}/40</Badge>
          </div>
          {traceEntries.length === 0 ? (
            <p className="text-sm leading-6 text-muted-foreground">等待真实 Decision 与失败 trace。</p>
          ) : (
            <Accordion
              type="multiple"
              defaultValue={traceEntries.filter((entry) => entry.kind === "failed").slice(0, 2).map(traceEntryValue)}
              className="min-w-0 max-w-full gap-2"
            >
              {traceEntries.map((entry) => (
                <AccordionItem
                  key={entry.id}
                  value={traceEntryValue(entry)}
                  className="min-w-0 overflow-hidden rounded-md border bg-background/55 px-2"
                >
                  <AccordionTrigger className="min-w-0 gap-2 py-2 hover:no-underline">
                    <TraceEntryHeader entry={entry} label={kindLabel(entry.kind)} />
                  </AccordionTrigger>
                  <AccordionContent className="min-w-0 space-y-2 pb-3">
                    <TraceLogEntry entry={entry} separated={false} showHeader={false} />
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function StrategyMetricsPanel({ metrics }: { metrics: AgentStrategyMetrics }) {
  return (
    <div className="min-w-0 max-w-full space-y-3 overflow-hidden">
      <div className="flex min-w-0 items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2 text-sm font-medium">
          <Activity className="size-4 shrink-0" />
          <span className="truncate">Agent strategy metrics</span>
        </div>
        <Badge variant="outline">seats={metrics.private_state_seat_count}</Badge>
      </div>
      <div className="flex flex-wrap gap-2">
        <Badge variant="outline">beliefs={metrics.belief_observation_count}</Badge>
        <Badge variant="outline">belief_brier={formatNullableMetric(metrics.belief_brier)}</Badge>
        <Badge variant="outline">claims={metrics.structured_claim_count}</Badge>
        <Badge variant={metrics.false_role_claim_count ? "destructive" : "outline"}>
          false_roles={metrics.false_role_claim_count}
        </Badge>
        <Badge variant={metrics.false_seer_result_count ? "destructive" : "outline"}>
          false_seer_results={metrics.false_seer_result_count}
        </Badge>
        <Badge variant={metrics.seer_result_contradiction_count ? "destructive" : "outline"}>
          seer_contradictions={metrics.seer_result_contradiction_count}
        </Badge>
        <Badge variant="outline">wolf_council={metrics.wolf_council_message_count}</Badge>
        <Badge variant="outline">wolf_votes={metrics.wolf_final_vote_count}</Badge>
        <Badge variant="outline">wolf_targets={metrics.wolf_final_vote_target_count}</Badge>
        <Badge variant="outline">
          wolf_agreement={metrics.wolf_final_vote_agreement === null ? "n/a" : String(metrics.wolf_final_vote_agreement)}
        </Badge>
      </div>
      {metrics.seats.length > 0 && (
        <div className="max-w-full overflow-x-auto rounded-md border bg-background/55">
          <table className="w-full min-w-[760px] border-collapse text-left text-[11px]">
            <thead className="border-b text-muted-foreground">
              <tr>
                <th className="px-2 py-2 font-medium">seat</th>
                <th className="px-2 py-2 font-medium">revision</th>
                <th className="px-2 py-2 font-medium">beliefs</th>
                <th className="px-2 py-2 font-medium">brier</th>
                <th className="px-2 py-2 font-medium">commitments</th>
                <th className="px-2 py-2 font-medium">claims</th>
                <th className="px-2 py-2 font-medium">false role</th>
                <th className="px-2 py-2 font-medium">false seer</th>
                <th className="px-2 py-2 font-medium">switches</th>
                <th className="px-2 py-2 font-medium">contradictions</th>
              </tr>
            </thead>
            <tbody>
              {metrics.seats.map((row) => (
                <tr key={row.seat} className="border-b last:border-b-0">
                  <td className="px-2 py-2 font-medium text-foreground">{row.seat}</td>
                  <td className="px-2 py-2">{row.private_state_revision}</td>
                  <td className="px-2 py-2">{row.belief_count}</td>
                  <td className="px-2 py-2">{formatNullableMetric(row.belief_brier)}</td>
                  <td className="px-2 py-2">{row.public_commitment_count}</td>
                  <td className="px-2 py-2">{row.structured_claim_count}</td>
                  <td className="px-2 py-2">{row.false_role_claim_count}</td>
                  <td className="px-2 py-2">{row.false_seer_result_count}</td>
                  <td className="px-2 py-2">{row.role_claim_switch_count}</td>
                  <td className="px-2 py-2">{row.seer_result_contradiction_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function formatNullableMetric(value: number | null): string {
  return value === null ? "n/a" : String(value);
}

function traceEntryValue(entry: LogEntry): string {
  return `trace-${entry.id}`;
}

function TraceLogEntry({
  entry,
  separated,
  showHeader = true,
}: {
  entry: LogEntry;
  separated: boolean;
  showHeader?: boolean;
}) {
  const text = entry.text;
  return (
    <div className="min-w-0 max-w-full space-y-2 overflow-x-hidden">
      {separated && <Separator />}
      {showHeader && <TraceEntryHeader entry={entry} label={kindLabel(entry.kind)} />}
      {text && (
        <p className="whitespace-pre-wrap break-words text-sm leading-6 text-muted-foreground [overflow-wrap:anywhere]">
          {text}
        </p>
      )}
      <TraceBadges entry={entry} />
    </div>
  );
}

function TraceEntryHeader({ entry, label }: { entry: LogEntry; label: string }) {
  return (
    <div className="flex min-w-0 max-w-full flex-wrap items-center gap-1.5">
      <Badge variant="outline" className="h-auto min-h-5 overflow-visible whitespace-normal">D{entry.day}</Badge>
      <Badge variant="outline" className="h-auto min-h-5 max-w-full overflow-visible whitespace-normal break-words [overflow-wrap:anywhere]">{label}</Badge>
      {entry.seat != null && <Badge variant="outline" className="h-auto min-h-5 overflow-visible whitespace-normal">{entry.seat}号</Badge>}
      {entry.targetSeat != null && <Badge variant="outline" className="h-auto min-h-5 overflow-visible whitespace-normal">目标 {entry.targetSeat}号</Badge>}
    </div>
  );
}

function TraceBadges({ entry }: { entry: LogEntry }) {
  const badges = [
    entry.bid != null ? ["发言优先级", String(entry.bid)] : null,
    entry.accuses?.length ? ["指控", entry.accuses.map((seat) => `${seat}号`).join("、")] : null,
    entry.replyTo != null ? ["回应", `${entry.replyTo}号`] : null,
    entry.claim ? ["公开声明", `${entry.claim.role || "unknown"}:${entry.claim.checked_seat || "?"}:${entry.claim.result || "?"}`] : null,
  ].filter((item): item is [string, string] => Boolean(item));
  if (!badges.length) return null;
  return (
    <div className="flex min-w-0 max-w-full flex-wrap gap-1.5">
      {badges.map(([label, value]) => (
        <Badge
          key={`${label}-${value}`}
          variant="outline"
          className="h-auto min-h-6 w-auto min-w-0 max-w-full shrink whitespace-normal break-words leading-5 [overflow-wrap:anywhere]"
        >
          <span className="shrink-0 font-medium">{label}</span>
          <span className="ml-1 min-w-0 font-normal text-muted-foreground">{value}</span>
        </Badge>
      ))}
    </div>
  );
}

function hasTraceMetadata(entry: LogEntry): boolean {
  return Boolean(
    entry.bid != null
    || entry.accuses?.length
    || entry.replyTo != null
    || entry.claim
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

function isTimelineEntry(entry: LogEntry): boolean {
  return [
    "phase",
    "night_resolved",
    "death",
    "speech",
    "vote",
    "vote_resolved",
    "vote_incomplete",
    "last_words",
    "hunter",
    "failed",
    "system",
  ].includes(entry.kind);
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
    night_resolved: "夜间结算",
    vote: "投票",
    vote_resolved: "票决",
    vote_incomplete: "缺票",
    death: "死亡",
    hunter: "猎枪",
    failed: "失败",
    speech: "发言",
    last_words: "遗言",
  }[kind] || kind;
}

function formatCompact(value: number): string {
  return new Intl.NumberFormat("zh-CN", { notation: "compact" }).format(value);
}

function isTraceSequence(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function mergeTraceItems(previous: RoomTraceItem[], incoming: RoomTraceItem[]): RoomTraceItem[] {
  const bySequence = new Map<number, RoomTraceItem>();
  const unsequenced = new Map<string, RoomTraceItem>();

  for (const item of [...previous, ...incoming]) {
    if (isTraceSequence(item.trace_seq)) {
      if (!bySequence.has(item.trace_seq)) bySequence.set(item.trace_seq, item);
      continue;
    }
    const key = `${item.kind}:${item.idx}`;
    if (!unsequenced.has(key)) unsequenced.set(key, item);
  }

  return [...bySequence.values(), ...unsequenced.values()].sort((left, right) => {
    const leftSequence = isTraceSequence(left.trace_seq) ? left.trace_seq : Number.MAX_SAFE_INTEGER;
    const rightSequence = isTraceSequence(right.trace_seq) ? right.trace_seq : Number.MAX_SAFE_INTEGER;
    return leftSequence - rightSequence
      || Number(left.ts || 0) - Number(right.ts || 0)
      || left.idx - right.idx;
  });
}

function isAbortError(error: unknown): boolean {
  return Boolean(
    error
    && typeof error === "object"
    && "name" in error
    && error.name === "AbortError",
  );
}
