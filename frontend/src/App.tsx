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
import type { RoomAuth } from "./lib/api";

type Screen =
  | { name: "lobby" }
  | { name: "room"; roomId: string }
  | { name: "game"; roomId: string; seat: number | null; mode: string; token?: string; adminToken?: string };

export default function App() {
  const [screen, setScreen] = useState<Screen>({ name: "lobby" });
  const [state, dispatch] = useReducer(reduce, undefined, makeInitial);
  const [socket, setSocket] = useState<GameSocket | null>(null);
  const [roomAuth, setRoomAuth] = useState<Record<string, RoomAuth>>({});

  // 进入游戏屏幕时连接 WS
  useEffect(() => {
    if (screen.name !== "game") return;
    const url = buildWsUrl(screen.roomId, screen.seat, screen.mode, screen.token);
    const sk = new GameSocket(url, {
      onEvent: (ev) => dispatch(ev),
      onOpen: () => dispatch({ type: "__open__" }),
      onClose: () => dispatch({ type: "__close__" }),
      onError: () => {},
    });
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
    dispatch({ type: "__reset__" });
    dispatch({ type: "__context__", mySeat: seat, mode });
    setScreen({ name: "game", roomId, seat, mode, token, adminToken: auth?.admin_token });
  }, [roomAuth]);

  const backToLobby = useCallback(() => setScreen({ name: "lobby" }), []);

  const sendHumanAction = useCallback(
    (action: string, data: Record<string, any>) => {
      if (socket) socket.send({ type: "human_action", action, ...data });
    },
    [socket],
  );

  return (
    <div className="min-h-screen bg-background text-foreground">
      <TopBar screen={screen} onHome={backToLobby} />
      <main className="mx-auto min-h-[calc(100svh-48px)] w-full max-w-[1680px] px-2 py-2 sm:px-3 lg:px-4">
        {screen.name === "lobby" && <LobbyView onCreated={enterRoom} />}
        {screen.name === "room" && (
          <RoomView roomId={screen.roomId} auth={roomAuth[screen.roomId]} onEnter={enterGame} onBack={backToLobby} />
        )}
        {screen.name === "game" && (
          <GameView
            state={state}
            roomId={screen.roomId}
            seat={screen.seat}
            mode={screen.mode}
            adminToken={screen.adminToken}
            onHumanAction={sendHumanAction}
            onBack={backToLobby}
          />
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
