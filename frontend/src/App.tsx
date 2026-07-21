// 顶层 App —— 视图状态机:lobby → room(等待) → game(观战/人机/上帝/复盘)
// 真实对接:所有数据来自后端 REST/WS,无 mock。
import { useCallback, useEffect, useReducer, useState } from "react";
import { Moon, Swords, Users } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { LobbyView } from "./views/LobbyView";
import { RoomView } from "./views/RoomView";
import { GameView } from "./views/GameView";
import { reduce, makeInitial } from "./lib/store";
import { GameSocket, buildWsUrl } from "./lib/ws-client";
import { getReplay, type ReplayPayload, type RoomAuth } from "./lib/api";
import { ReplayConsole } from "./components/ReplayConsole";

type Screen =
  | { name: "lobby" }
  | { name: "room"; roomId: string }
  | { name: "game"; roomId: string; seat: number | null; mode: string; token?: string; adminToken?: string };

const SCREEN_STORAGE_KEY = "werewolf.mas.screen.v1";

export default function App() {
  const [roomAuth, setRoomAuth] = useState<Record<string, RoomAuth>>(() => {
    clearLegacyRoomAuth();
    return {};
  });
  const [screen, setScreen] = useState<Screen>(() => readStoredScreen());
  const [state, dispatch] = useReducer(reduce, undefined, makeInitial);
  const [socket, setSocket] = useState<GameSocket | null>(null);
  const [replayPayload, setReplayPayload] = useState<ReplayPayload | null>(null);
  const [replayLoading, setReplayLoading] = useState(false);
  const [replayError, setReplayError] = useState("");

  useEffect(() => {
    writeStoredScreen(screen);
  }, [screen]);

  useEffect(() => {
    if (screen.name !== "game") return;
    dispatch({ type: "__reset__" });
    dispatch({ type: "__context__", mySeat: screen.seat, mode: screen.mode });
    setReplayPayload(null);
    setReplayError("");
  }, [screen]);

  useEffect(() => {
    if (screen.name !== "game" || screen.mode !== "replay") {
      setReplayLoading(false);
      return;
    }
    if (!screen.adminToken) {
      setReplayPayload(null);
      setReplayLoading(false);
      setReplayError("缺少房间管理 capability，不能读取 admin replay。");
      return;
    }
    const controller = new AbortController();
    setReplayLoading(true);
    setReplayError("");
    void getReplay(screen.roomId, screen.adminToken, controller.signal)
      .then((payload) => {
        if (!controller.signal.aborted) setReplayPayload(payload);
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setReplayPayload(null);
          setReplayError(error instanceof Error ? error.message : String(error));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setReplayLoading(false);
      });
    return () => controller.abort();
  }, [screen]);

  // 进入游戏屏幕时连接 WS
  useEffect(() => {
    if (screen.name !== "game") return;
    if (screen.mode === "replay") {
      setSocket(null);
      return;
    }
    const url = buildWsUrl(screen.roomId, screen.seat, screen.mode);
    const sk = new GameSocket(url, {
      onEvent: (ev) => dispatch(ev),
      onOpen: () => dispatch({ type: "__open__" }),
      onClose: (info) => dispatch({
        type: "__close__",
        code: info.code,
        reason: info.reason,
        willReconnect: info.willReconnect,
      }),
      onError: (message) => dispatch({ type: "__socket_error__", message }),
    }, screen.token);
    sk.connect();
    setSocket(sk);
    return () => {
      sk.close();
      setSocket(null);
    };
  }, [screen]);

  // 房间状态轮询(等待→running 自动跳游戏)。简化:RoomView 自行处理。

  const enterRoom = useCallback((roomId: string, auth?: RoomAuth) => {
    if (auth) setRoomAuth((prev) => ({ ...prev, [roomId]: auth }));
    setScreen({ name: "room", roomId });
  }, []);

  const enterGame = useCallback((roomId: string, seat: number | null, mode: string) => {
    const auth = roomAuth[roomId];
    const token =
      mode === "play" && seat != null ? auth?.seat_tokens?.[String(seat)] :
      mode === "god" || mode === "replay" ? auth?.admin_token :
      undefined;
    // Keep the admin capability out of spectator/player screens. The browser
    // may hold it in the room owner flow, but only God/Replay surfaces need to
    // pass it into a game component or trace request.
    const adminToken = mode === "god" || mode === "replay" ? auth?.admin_token : undefined;
    setScreen({ name: "game", roomId, seat, mode, token, adminToken });
  }, [roomAuth]);

  const backToLobby = useCallback(() => {
    // Capabilities are intentionally scoped to the current in-memory room.
    setRoomAuth({});
    setScreen({ name: "lobby" });
  }, []);

  const backToRoom = useCallback(() => {
    setScreen((current) => (
      current.name === "game"
        ? { name: "room", roomId: current.roomId }
        : current
    ));
  }, []);

  const sendHumanAction = useCallback(
    (action: string, data: Record<string, any>): boolean => {
      if (!socket || !state.pendingHuman) return false;
      const sent = socket.send({
        type: "human_action",
        request_id: state.pendingHuman.requestId,
        day: state.pendingHuman.day ?? state.day,
        phase: state.pendingHuman.phase ?? state.phase,
        action,
        ...data,
      });
      if (!sent) return false;
      return true;
    },
    [socket, state.day, state.pendingHuman, state.phase],
  );

  const reconnectSocket = useCallback(() => {
    if (!socket || state.connected || !state.socketClose?.retryableByUser) return;
    dispatch({ type: "__manual_reconnect__" });
    socket.connect();
  }, [socket, state.connected, state.socketClose?.retryableByUser]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      {screen.name !== "game" && <TopBar screen={screen} onHome={backToLobby} />}
      <main
        className={
          screen.name === "game"
            ? "h-[100svh] min-h-0 w-full overflow-hidden px-0 py-0"
            : "mx-auto min-h-[calc(100svh-48px)] w-full max-w-[1680px] px-2 py-2 sm:px-3 lg:px-4"
        }
      >
        {screen.name === "lobby" && <LobbyView onCreated={enterRoom} />}
        {screen.name === "room" && (
          <RoomView roomId={screen.roomId} auth={roomAuth[screen.roomId]} onEnter={enterGame} onBack={backToLobby} />
        )}
        {screen.name === "game" && (
          screen.mode === "replay" ? (
            <ReplayConsole
              roomId={screen.roomId}
              payload={replayPayload}
              loading={replayLoading}
              error={replayError}
              onBack={backToRoom}
            />
          ) : (
            <GameView
              state={state}
              roomId={screen.roomId}
              seat={screen.seat}
              mode={screen.mode}
              adminToken={screen.adminToken}
              onHumanAction={sendHumanAction}
              onReconnect={reconnectSocket}
              onBack={backToRoom}
            />
          )
        )}
      </main>
    </div>
  );
}

function TopBar({ screen, onHome }: { screen: Screen; onHome: () => void }) {
  return (
    <header className="sticky top-0 z-40 flex h-12 items-center gap-3 border-b bg-card/95 px-3 shadow-sm backdrop-blur sm:px-4 lg:px-6">
      <Button className="gap-2 px-2 text-sm font-semibold sm:text-base" variant="ghost" onClick={onHome}>
        <Swords className="size-4" />
        狼烟
      </Button>
      <div className="hidden items-center gap-2 text-sm text-muted-foreground sm:flex">
        {screen.name === "lobby" && (
          <>
            <Users className="size-4" />
            创建房间
          </>
        )}
        {screen.name === "room" && (
          <>
            <Moon className="size-4" />
            等待开局
          </>
        )}
        {screen.name === "game" && (
          <>
            <Moon className="size-4" />
            真实对局
          </>
        )}
      </div>
      <div className="ml-auto flex items-center gap-2">
        {screen.name === "game" && (
          <Badge variant="outline">{labelMode(screen.mode)}</Badge>
        )}
      </div>
    </header>
  );
}

function labelMode(mode: string): string {
  return { spectate: "观战", play: "人机对局", god: "上帝视角", replay: "复盘" }[mode] || mode;
}

function clearLegacyRoomAuth(): void {
  if (typeof window === "undefined") return;
  try {
    // Older builds persisted plaintext capabilities. Remove them without
    // reading or migrating the values into React state.
    window.localStorage.removeItem("werewolf.mas.roomAuth.v1");
  } catch {
    // Storage may be unavailable in restricted browser contexts.
  }
}

function readStoredScreen(): Screen {
  if (typeof window === "undefined") return { name: "lobby" };
  try {
    const raw = JSON.parse(window.localStorage.getItem(SCREEN_STORAGE_KEY) || "null");
    if (!raw || typeof raw !== "object") return { name: "lobby" };
    if (raw.name === "room" && typeof raw.roomId === "string" && raw.roomId) {
      return { name: "room", roomId: raw.roomId };
    }
    if (raw.name === "game" && typeof raw.roomId === "string" && raw.roomId && typeof raw.mode === "string") {
      // A persisted screen never carries a capability. Re-enter the neutral
      // room screen and require a fresh in-memory authorization for play/god.
      return { name: "room", roomId: raw.roomId };
    }
  } catch {
    return { name: "lobby" };
  }
  return { name: "lobby" };
}

function writeStoredScreen(screen: Screen): void {
  try {
    const persisted =
      screen.name === "game"
        ? { name: "room", roomId: screen.roomId }
        : screen;
    window.localStorage.setItem(SCREEN_STORAGE_KEY, JSON.stringify(persisted));
  } catch {
    // Local storage can be unavailable in restricted browser contexts.
  }
}
