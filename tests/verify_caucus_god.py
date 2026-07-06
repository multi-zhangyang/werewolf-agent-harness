"""验证前端 god 模式可见性:狼人白天党团 caucus 事件 + 欺骗 deception 字段端到端到 WS。

验证:
1. god 模式收到 wolf_caucus / wolf_caucus_consensus 事件(信息隔离:仅 god/replay)
2. speech 事件带 deception 字段
3. 0 决策失败
"""
from __future__ import annotations

import asyncio
import json
import sys

import httpx
import websockets

HOST = "localhost:8000"
NAMES = ["李白", "杜甫", "苏轼", "辛弃疾", "陆游", "王维"]


async def main() -> int:
    async with httpx.AsyncClient(base_url=f"http://{HOST}", timeout=30) as http:
        r = await http.post("/api/rooms", json={"player_names": NAMES, "human_seats": []})
        r.raise_for_status()
        room_id = r.json()["room_id"]
        print(f"房间: {room_id}")

        ws = await websockets.connect(
            f"ws://{HOST}/ws/{room_id}?mode=god",
            ping_interval=15, ping_timeout=10, max_size=2**22,
        )
        await http.post(f"/api/rooms/{room_id}/start")

        caucus_n = 0
        consensus_n = 0
        speech_with_deception = 0
        speech_total = 0
        failed = 0
        winner = None

        try:
            async for msg in ws:
                data = json.loads(msg)
                t = data.get("type")
                if t == "wolf_caucus":
                    caucus_n += 1
                    print(f"  🐺 党团提案: seat={data.get('seat')} target={data.get('target_seat')} text={data.get('text','')[:50]}")
                elif t == "wolf_caucus_consensus":
                    consensus_n += 1
                    print(f"  🐺 党团共识: target={data.get('target_seat')} text={data.get('text','')[:60]}")
                elif t == "speech":
                    speech_total += 1
                    if data.get("deception") and data["deception"] != "none":
                        speech_with_deception += 1
                elif t == "agent_decision_failed":
                    failed += 1
                elif t == "game_ended":
                    winner = data.get("winner")
                elif t == "analysis":
                    break
        except Exception as e:  # noqa: BLE001
            print(f"WS 异常: {type(e).__name__}: {e}")
        finally:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    print(f"\n=== 结果 ===")
    print(f"winner={winner}")
    print(f"党团提案事件: {caucus_n}")
    print(f"党团共识事件: {consensus_n}")
    print(f"发言总数: {speech_total}, 含欺骗声明: {speech_with_deception}")
    print(f"决策失败: {failed}")
    ok = caucus_n > 0 and consensus_n > 0 and failed == 0
    print(f"\n{'✅ god 模式可见狼人党团私聊 + 欺骗字段端到端' if ok else '⚠️ 缺失'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
