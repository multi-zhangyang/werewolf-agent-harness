import { useCallback, useEffect, useState, type ReactNode } from "react";
import {
  ArrowLeft,
  BarChart3,
  Clipboard,
  Eye,
  Loader2,
  Play,
  Search,
  Settings,
  UserRound,
  Users,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { getProviders, getRoom, setSeatModelConfig, startRoom } from "../lib/api";
import type { RoomAuth } from "../lib/api";
import type { ProviderMeta, RoomInfo } from "../lib/types";
import { ModelConfigModal } from "../components/ModelConfigModal";

export function RoomView({
  roomId,
  auth,
  onEnter,
  onBack,
}: {
  roomId: string;
  auth?: RoomAuth;
  onEnter: (roomId: string, seat: number | null, mode: string) => void;
  onBack: () => void;
}) {
  const [room, setRoom] = useState<RoomInfo | null>(null);
  const [providers, setProviders] = useState<Record<string, ProviderMeta>>({});
  const [err, setErr] = useState("");
  const [starting, setStarting] = useState(false);
  const [modalSeat, setModalSeat] = useState<number | null>(null);
  const [seatConfigured, setSeatConfigured] = useState<Record<number, { provider: string; model: string }>>({});

  const refresh = useCallback(async () => {
    try {
      setRoom(await getRoom(roomId));
    } catch (error: any) {
      setErr(String(error.message || error));
    }
  }, [roomId]);

  useEffect(() => {
    getProviders().then(setProviders).catch(() => {});
    refresh();
    const timer = window.setInterval(refresh, 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const startGame = async () => {
    setStarting(true);
    setErr("");
    try {
      await startRoom(roomId, auth?.admin_token);
      await refresh();
    } catch (error: any) {
      setErr(String(error.message || error));
    } finally {
      setStarting(false);
    }
  };

  const saveSeatCfg = async (seat: number, cfg: any) => {
    try {
      await setSeatModelConfig(roomId, seat, cfg, auth?.admin_token);
      setSeatConfigured((prev) => ({ ...prev, [seat]: { provider: cfg.provider || "openai", model: cfg.model || "" } }));
      setModalSeat(null);
    } catch (error: any) {
      setErr(String(error.message || error));
    }
  };

  const players = room?.players || [];
  const humanSeats = room?.human_seats || [];
  const firstHumanSeat = humanSeats[0] ?? null;
  const terminal = ["ended", "failed", "timeout", "cancelled"].includes(room?.status || "");
  const running = room?.status === "running" || terminal;
  const waiting = room?.status === "waiting";
  const firstHumanSeatToken = firstHumanSeat == null ? "" : auth?.seat_tokens?.[String(firstHumanSeat)];

  return (
    <div className="grid min-h-[calc(100svh-72px)] gap-4 pb-[calc(5rem+env(safe-area-inset-bottom))] lg:grid-cols-[minmax(0,1fr)_380px] lg:pb-0 xl:grid-cols-[minmax(0,1fr)_420px]">
      <Card className="min-h-0 bg-card/95 shadow-sm">
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <CardTitle className="flex items-center gap-2 text-xl">
                <Users className="size-5" />
                等待室
              </CardTitle>
              <CardDescription className="mt-1">确认座位、启动真实模型对局，或选择进入方式。</CardDescription>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant="outline" className="max-w-[180px] font-mono">
                <span className="truncate">{roomId}</span>
              </Badge>
              <Button size="icon-sm" variant="ghost" onClick={() => navigator.clipboard?.writeText(roomId)} aria-label="复制房间号">
                <Clipboard className="size-4" />
              </Button>
              <Badge variant={room?.status === "running" ? "default" : terminal ? "destructive" : "outline"}>{statusLabel(room?.status)}</Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent className="min-h-0 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <Button variant="ghost" onClick={onBack} className="gap-2">
              <ArrowLeft className="size-4" />
              返回大厅
            </Button>
            {waiting && (
              <Button onClick={startGame} disabled={starting || !auth?.admin_token} className="gap-2">
                {starting ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
                {starting ? "启动中..." : "开始真实对局"}
              </Button>
            )}
          </div>

          {(err || room?.error) && (
            <Alert variant="destructive">
              <AlertDescription>{err || room?.error}</AlertDescription>
            </Alert>
          )}

          <ScrollArea className="h-[clamp(220px,calc(100svh-360px),600px)] rounded-lg border bg-background/45">
            <div className="grid gap-2 p-3 sm:grid-cols-2 xl:grid-cols-3">
              {players.map((player) => {
                const configured = seatConfigured[player.seat];
                const human = humanSeats.includes(player.seat);
                return (
                  <Card key={player.seat} size="sm" className="bg-card/90 shadow-none">
                    <CardContent className="flex items-center gap-3 px-3">
                      <Badge variant={human ? "default" : "outline"} className="w-12 justify-center">{player.seat}号</Badge>
                      <div className="min-w-0 flex-1">
                        <div className="truncate font-medium">{player.name}</div>
                        <div className="truncate text-xs text-muted-foreground">
                          {human ? "真人座位" : "AI 座位"} · {configured?.model || "默认模型"}
                        </div>
                      </div>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button size="icon-sm" variant="ghost" aria-label="配置座位模型" onClick={() => setModalSeat(player.seat)}>
                            <Settings className="size-4" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>配置座位模型</TooltipContent>
                      </Tooltip>
                    </CardContent>
                  </Card>
                );
              })}
              {!players.length && (
                <Alert className="sm:col-span-2 xl:col-span-3">
                  <AlertDescription>正在读取房间状态。</AlertDescription>
                </Alert>
              )}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      <aside className="space-y-4">
        <Card className="bg-card/95 shadow-sm">
          <CardHeader>
            <CardTitle>进入对局</CardTitle>
            <CardDescription>不同模式只使用后端授权的真实视图。</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {waiting && (
              <Alert>
                <AlertDescription className="leading-6">
                  {auth?.admin_token ? "房主可以直接启动真实模型对局。" : "缺少房主 token，不能启动房间。"}
                </AlertDescription>
              </Alert>
            )}

            <div className="grid gap-2">
              <ModeButton
                icon={<Eye className="size-4" />}
                title="观战"
                desc="只看后端公开事件"
                onClick={() => onEnter(roomId, null, "spectate")}
                disabled={!running && !waiting}
              />
              <ModeButton
                icon={<UserRound className="size-4" />}
                title="扮演真人座位"
                desc={firstHumanSeat ? `进入 ${firstHumanSeat} 号座位` : "创建时没有选择真人座位"}
                onClick={() => onEnter(roomId, firstHumanSeat, "play")}
                disabled={!running || firstHumanSeat == null || !firstHumanSeatToken}
              />
              <ModeButton
                icon={<Search className="size-4" />}
                title="上帝视角"
                desc="需要房主 token，可见全量授权信息"
                onClick={() => onEnter(roomId, null, "god")}
                disabled={!running || !auth?.admin_token}
              />
              <ModeButton
                icon={<BarChart3 className="size-4" />}
                title="复盘"
                desc="对局结束后查看分析"
                onClick={() => onEnter(roomId, null, "replay")}
                disabled={room?.status !== "ended" || !auth?.admin_token}
              />
            </div>

            <Separator />

            <div className="grid gap-2 text-sm">
              <InfoLine label="房间状态" value={statusLabel(room?.status)} />
              <InfoLine label="玩家人数" value={`${players.length || 0} 人`} />
              <InfoLine label="真人座位" value={humanSeats.length ? humanSeats.map((seat) => `${seat}号`).join(" / ") : "无"} />
            </div>
          </CardContent>
        </Card>
      </aside>

      {modalSeat !== null && (
        <ModelConfigModal
          seat={modalSeat}
          providers={providers}
          onClose={() => setModalSeat(null)}
          onSave={(cfg) => saveSeatCfg(modalSeat, cfg)}
        />
      )}

      <div className="fixed inset-x-0 bottom-0 z-40 border-t bg-card/95 px-3 pt-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))] shadow-lg backdrop-blur lg:hidden">
        {waiting ? (
          <Button className="w-full gap-2" onClick={startGame} disabled={starting || !auth?.admin_token}>
            {starting ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
            {starting ? "启动中..." : "开始真实对局"}
          </Button>
        ) : (
          <Button className="w-full gap-2" onClick={() => onEnter(roomId, null, auth?.admin_token ? "god" : "spectate")}>
            {auth?.admin_token ? <Search className="size-4" /> : <Eye className="size-4" />}
            {auth?.admin_token ? "进入上帝视角" : "进入观战"}
          </Button>
        )}
      </div>
    </div>
  );
}

function ModeButton({
  icon,
  title,
  desc,
  onClick,
  disabled,
}: {
  icon: ReactNode;
  title: string;
  desc: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <Button variant="outline" className="h-auto justify-start gap-3 px-3 py-3 text-left" onClick={onClick} disabled={disabled}>
      <span className="text-muted-foreground">{icon}</span>
      <span className="min-w-0">
        <span className="block font-medium">{title}</span>
        <span className="block whitespace-normal text-xs text-muted-foreground">{desc}</span>
      </span>
    </Button>
  );
}

function InfoLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border bg-background/60 px-3 py-2">
      <span className="text-muted-foreground">{label}</span>
      <span className="min-w-0 truncate text-right font-medium">{value}</span>
    </div>
  );
}

function statusLabel(status?: string): string {
  return {
    waiting: "等待中",
    running: "进行中",
    ended: "已结束",
    failed: "异常",
    timeout: "超时",
    cancelled: "已取消",
  }[status || ""] || status || "读取中";
}
