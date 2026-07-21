import { useEffect, useMemo, useState, type ReactNode } from "react";
import { ChevronLeft, ChevronRight, Film, LockKeyhole, Pause, Play, RotateCcw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import {
  filterReplayRows,
  replayDecisionTrace,
  replayFilterOptions,
  replayTimeline,
  type ReplayFilters,
  type ReplayTimelineRow,
} from "@/lib/replay";
import type { ReplayPayload } from "@/lib/api";
import { ProtocolTracePanel } from "./HarnessConsole";

export function ReplayConsole({
  roomId,
  payload,
  loading,
  error,
  onBack,
}: {
  roomId: string;
  payload: ReplayPayload | null;
  loading: boolean;
  error?: string;
  onBack: () => void;
}) {
  if (loading) return <ReplayShell roomId={roomId}><ReplayMessage title="正在加载复盘" text="读取已提交的 transcript，不会重新执行 Agent。" /></ReplayShell>;
  if (error) return <ReplayShell roomId={roomId}><ReplayMessage title="复盘不可用" text={error} destructive onBack={onBack} /></ReplayShell>;
  if (!payload) return <ReplayShell roomId={roomId}><ReplayMessage title="没有复盘数据" text="服务没有返回可验证的 ended-run payload。" destructive onBack={onBack} /></ReplayShell>;
  return <ReplayLoaded roomId={roomId} payload={payload} onBack={onBack} />;
}

function ReplayLoaded({ roomId, payload, onBack }: { roomId: string; payload: ReplayPayload; onBack: () => void }) {
  const rows = useMemo(() => replayTimeline(payload), [payload]);
  const decisionTrace = useMemo(() => replayDecisionTrace(payload), [payload]);
  const [playhead, setPlayhead] = useState(Math.max(0, rows.length - 1));
  const [playing, setPlaying] = useState(false);
  const [filters, setFilters] = useState<ReplayFilters>({ phase: "all", kind: "all", seat: "all" });
  const visibleRows = useMemo(() => filterReplayRows(rows, filters, playhead), [filters, playhead, rows]);
  const current = rows[playhead];
  const phases = replayFilterOptions(rows, "phase");
  const kinds = replayFilterOptions(rows, "kind");
  const seats = replayFilterOptions(rows, "seat");

  useEffect(() => {
    if (!playing || rows.length < 2) return;
    const timer = window.setInterval(() => {
      setPlayhead((value) => {
        if (value >= rows.length - 1) {
          setPlaying(false);
          return value;
        }
        return value + 1;
      });
    }, 700);
    return () => window.clearInterval(timer);
  }, [playing, rows.length]);

  const move = (delta: number) => {
    setPlaying(false);
    setPlayhead((value) => Math.max(0, Math.min(Math.max(0, rows.length - 1), value + delta)));
  };
  const reset = () => {
    setPlaying(false);
    setPlayhead(Math.max(0, rows.length - 1));
    setFilters({ phase: "all", kind: "all", seat: "all" });
  };

  return (
    <ReplayShell roomId={roomId} onBack={onBack}>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-7xl space-y-3 px-3 py-4 sm:px-5 lg:px-7">
          <Card size="sm" className="border-primary/25">
            <CardHeader className="gap-2">
              <div className="flex flex-wrap items-center gap-2">
                <CardTitle className="flex items-center gap-2 text-base"><Film className="size-4" />事实复盘</CardTitle>
                <Badge variant="outline">status={payload.status}</Badge>
                <Badge variant="outline">winner={payload.winner || "unknown"}</Badge>
                <Badge variant="outline">events={rows.length}</Badge>
                <Badge variant="outline">decisions={decisionTrace.length}</Badge>
              </div>
              <CardDescription>
                这是 immutable transcript 的观察投影；不重新运行 Agent、不改写公开输出、不改变当前对局状态。
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" variant="outline" onClick={() => move(-1)} disabled={!rows.length || playhead <= 0} aria-label="上一事件"><ChevronLeft className="size-4" />上一事件</Button>
                <Button size="sm" variant={playing ? "default" : "outline"} onClick={() => setPlaying((value) => !value)} disabled={!rows.length}>
                  {playing ? <Pause className="size-4" /> : <Play className="size-4" />}{playing ? "暂停" : "播放"}
                </Button>
                <Button size="sm" variant="outline" onClick={() => move(1)} disabled={!rows.length || playhead >= rows.length - 1} aria-label="下一事件">下一事件<ChevronRight className="size-4" /></Button>
                <Button size="sm" variant="ghost" onClick={reset}><RotateCcw className="size-4" />重置</Button>
                <span className="text-xs text-muted-foreground">playhead={rows.length ? `${playhead + 1}/${rows.length}` : "0/0"}</span>
              </div>
              <input
                aria-label="复盘时间轴"
                type="range"
                min={0}
                max={Math.max(0, rows.length - 1)}
                value={rows.length ? playhead : 0}
                onChange={(event) => { setPlaying(false); setPlayhead(Number(event.currentTarget.value)); }}
                disabled={!rows.length}
                className="w-full accent-primary"
              />
              <div className="grid gap-2 sm:grid-cols-3">
                <ReplayFilter label="phase" value={filters.phase} options={phases} onChange={(value) => setFilters((old) => ({ ...old, phase: value }))} />
                <ReplayFilter label="kind" value={filters.kind} options={kinds} onChange={(value) => setFilters((old) => ({ ...old, kind: value }))} />
                <ReplayFilter label="seat" value={filters.seat} options={seats} onChange={(value) => setFilters((old) => ({ ...old, seat: value }))} />
              </div>
              {current && <CurrentReplayRow row={current} />}
            </CardContent>
          </Card>

          <div className="grid min-w-0 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(320px,420px)]">
            <Card size="sm" className="min-w-0 overflow-hidden">
              <CardHeader>
                <CardTitle className="text-sm">Environment timeline</CardTitle>
                <CardDescription>仅显示 playhead 之前的事实事件；筛选只改变投影，不删除原始证据。</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {visibleRows.length === 0 ? <p className="text-sm text-muted-foreground">没有匹配的 timeline row。</p> : visibleRows.map((row) => <ReplayRow key={`${row.seq}-${row.kind}`} row={row} active={current?.seq === row.seq} />)}
              </CardContent>
            </Card>
            <div className="min-w-0 space-y-3">
              <FactualAnalysis payload={payload} />
              <ProtocolTracePanel items={decisionTrace} />
            </div>
          </div>
        </div>
      </div>
    </ReplayShell>
  );
}

function ReplayShell({ roomId, onBack, children }: { roomId: string; onBack?: () => void; children: ReactNode }) {
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-background">
      <header className="flex min-h-14 shrink-0 items-center gap-3 border-b bg-card/80 px-3 sm:px-5">
        <div className="flex min-w-0 flex-1 items-center gap-2"><LockKeyhole className="size-4 shrink-0 text-primary" /><span className="truncate text-sm font-semibold">Replay Console</span><code className="hidden truncate text-xs text-muted-foreground sm:block">run_id={roomId}</code></div>
        {onBack && <Button size="sm" variant="outline" onClick={onBack}>返回房间</Button>}
      </header>
      {children}
    </div>
  );
}

function ReplayMessage({ title, text, destructive, onBack }: { title: string; text: string; destructive?: boolean; onBack?: () => void }) {
  return <Card className={cn("mx-auto mt-10 w-[min(100%-1.5rem,42rem)]", destructive && "border-destructive/50")}><CardHeader><CardTitle>{title}</CardTitle><CardDescription className="break-words">{text}</CardDescription></CardHeader>{onBack && <CardContent><Button onClick={onBack}>返回房间</Button></CardContent>}</Card>;
}

function ReplayFilter({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return <label className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground"><span>{label}</span><select value={value} onChange={(event) => onChange(event.currentTarget.value)} className="h-8 min-w-0 flex-1 rounded-md border bg-background px-2 text-sm text-foreground"><option value="all">all</option>{options.map((option) => <option key={option} value={option}>{option}</option>)}</select></label>;
}

function CurrentReplayRow({ row }: { row: ReplayTimelineRow }) {
  return <div className="rounded-md border border-primary/30 bg-primary/5 p-2 text-xs"><div className="mb-1 flex flex-wrap gap-1.5"><Badge>current #{row.seq}</Badge><Badge variant="outline">{row.kind}</Badge>{row.phase && <Badge variant="outline">phase={row.phase}</Badge>}{row.seat != null && <Badge variant="outline">seat={row.seat}</Badge>}</div><p className="whitespace-pre-wrap break-words leading-5">{row.text}</p></div>;
}

function ReplayRow({ row, active }: { row: ReplayTimelineRow; active: boolean }) {
  return <article className={cn("rounded-md border p-2 text-xs", active ? "border-primary/50 bg-primary/5" : "bg-card/40")}><div className="flex flex-wrap items-center gap-1.5"><code>#{row.seq}</code><Badge variant="outline">{row.kind}</Badge>{row.phase && <Badge variant="outline">{row.phase}</Badge>}{row.day != null && <Badge variant="outline">D{row.day}</Badge>}{row.seat != null && <Badge variant="outline">seat={row.seat}</Badge>}{row.name && <span className="font-medium">{row.name}</span>}</div><p className="mt-1 whitespace-pre-wrap break-words leading-5 text-foreground">{row.text}</p></article>;
}

function FactualAnalysis({ payload }: { payload: ReplayPayload }) {
  const analysis = payload.analysis;
  return <Card size="sm"><CardHeader><CardTitle className="text-sm">Factual analysis</CardTitle><CardDescription>来自结束时 analysis payload；不是模型自评或因果结论。</CardDescription></CardHeader><CardContent className="space-y-2 text-xs">{analysis ? <><div className="grid gap-1 sm:grid-cols-2"><span>winner={analysis.winner || "unknown"}</span><span>days={analysis.days}</span><span>decisions={analysis.decision_count}</span><span>turn_policy={analysis.turn_policy}</span></div><Separator /><pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted/30 p-2">{JSON.stringify({ decision_trace_metrics: analysis.decision_trace_metrics, parse_metrics: analysis.parse_metrics, decision_failure_metrics: analysis.decision_failure_metrics }, null, 2)}</pre></> : <p className="text-muted-foreground">结束 payload 没有 analysis。</p>}</CardContent></Card>;
}
