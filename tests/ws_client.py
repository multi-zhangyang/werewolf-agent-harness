"""WebSocket 观战客户端示例。

用法:
    source .venv/bin/activate
    python tests/ws_client.py <room_id> [--mode spectate|god|play] [--seat SEAT]

先启动服务并创建房间,获取 room_id 后使用本脚本实时查看事件流。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import websockets


async def main() -> None:
    parser = argparse.ArgumentParser(description="Werewolf MAS WebSocket 客户端")
    parser.add_argument("room_id", help="房间 ID")
    parser.add_argument("--mode", default="spectate", choices=["spectate", "god", "play"])
    parser.add_argument("--seat", type=int, default=None, help="play 模式下指定座位")
    parser.add_argument("--host", default="localhost:8000")
    args = parser.parse_args()

    params = f"mode={args.mode}"
    if args.seat is not None:
        params += f"&seat={args.seat}"
    url = f"ws://{args.host}/ws/{args.room_id}?{params}"

    print(f"连接 {url}")
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
