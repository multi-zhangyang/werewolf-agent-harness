"""WebSocket 观战客户端示例。

用法:
    source .venv/bin/activate
    python tests/ws_client.py <room_id> [--mode spectate|god|play|replay]
        [--seat SEAT] [--token ROOM_TOKEN]

先启动服务并创建房间,获取 room_id 后使用本脚本实时查看事件流。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from urllib.parse import urlencode

import websockets


async def main() -> None:
    parser = argparse.ArgumentParser(description="Werewolf MAS WebSocket 客户端")
    parser.add_argument("room_id", help="房间 ID")
    parser.add_argument("--mode", default="spectate", choices=["spectate", "god", "play", "replay"])
    parser.add_argument("--seat", type=int, default=None, help="play 模式下指定座位")
    parser.add_argument("--token", default=None, help="god/replay 的 admin token 或 play 的 seat token")
    parser.add_argument("--host", default="localhost:8000")
    args = parser.parse_args()

    params: dict[str, str] = {"mode": args.mode}
    if args.seat is not None:
        params["seat"] = str(args.seat)
    if args.token:
        params["token"] = args.token
    url = f"ws://{args.host}/ws/{args.room_id}?{urlencode(params)}"

    print(f"连接 ws://{args.host}/ws/{args.room_id}?mode={args.mode} (token 不回显)")
    async with websockets.connect(url) as ws:
        async for msg in ws:
            data = json.loads(msg)
            t = data.get("type")
            if t == "snapshot":
                print("[snapshot] status:", data.get("status"))
                view = data.get("view", {})
                players = view.get("players", [])
                print(f"  玩家: {len(players)} 人, 阶段: {view.get('phase')}, day: {view.get('day')}")
            elif t in ("speech", "vote_cast", "vote_resolved", "night_resolved", "phase_started", "game_ended"):
                print(f"[{t}] {data.get('message') or data}")
            else:
                print(f"[{t}] {data}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已退出")
        sys.exit(0)
