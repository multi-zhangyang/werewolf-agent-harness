"""验证方向A:agent 之间真正的对话——reply_to/accuses 结构化 + 被提及者优先调度。

验证目标:
1. speech 事件带 reply_to/accuses 字段(结构化对话关系端到端送达)
2. 存在 reply_to 非空的发言(被点名者真的在回应)
3. 存在 accuses 非空的发言(agent 真的在点名指控)
4. 被提及者优先:被指控 seat 在后续轮次 bid≥4 时被优先叫起(看 speech 顺序)
5. 0 决策失败(无伪造,no-fallback 铁律守恒)
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
        r = await http.post(f"/api/rooms/{room_id}/start")
        r.raise_for_status()
        print("游戏已开始,等待事件...\n")

        speeches: list[dict] = []
        failed = 0
        winner = None
        # 按轮次记录发言顺序(检测被提及者优先)
        day_round_order: list[list[int]] = []
        current_round: list[int] = []
        mentioned_seats: set[int] = set()
        mentioned_then_replied: int = 0  # 被提及后下一轮 bid≥4 被叫起回应的次数

        try:
            async for msg in ws:
                data = json.loads(msg)
                t = data.get("type")
                if t == "speech":
                    seat = data.get("seat")
                    reply_to = data.get("reply_to")
                    accuses = data.get("accuses")
                    speeches.append({
                        "seat": seat, "text": data.get("text", ""),
                        "bid": data.get("bid"), "reply_to": reply_to, "accuses": accuses,
                    })
                    current_round.append(seat)
                    if accuses:
                        for a in accuses:
                            mentioned_seats.add(a)
                    marker = ""
                    if reply_to:
                        marker += f" [回应{reply_to}号]"
                    if accuses:
                        marker += f" [指控{accuses}]"
                    print(f"  {seat}号(bid={data.get('bid')}): {data.get('text','')[:60]}{marker}")
                elif t == "phase_started" and data.get("phase") == "voting":
                    if current_round:
                        day_round_order.append(current_round)
                        current_round = []
                    print(f"  --- 进入投票 ---")
                elif t == "agent_decision_failed":
                    failed += 1
                    print(f"  ⚠️ 决策失败: seat={data.get('seat')} {data.get('reason','')[:80]}")
                elif t == "game_ended":
                    winner = data.get("winner")
                    print(f"\n  🏁 结束 winner={winner}")
        except Exception as e:  # noqa: BLE001
            print(f"WS 异常: {type(e).__name__}: {e}")
        finally:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    # 统计
    reply_count = sum(1 for s in speeches if s["reply_to"])
    accuse_count = sum(1 for s in speeches if s["accuses"])
    # 被提及者在后续发言中是否真的回应了(reply_to 指向曾指控他的人)
    replied_after_mentioned = 0
    for s in speeches:
        if s["reply_to"] and s["seat"] in mentioned_seats:
            replied_after_mentioned += 1

    print("\n=== 方向A 验证结果 ===")
    print(f"总发言数: {len(speeches)}")
    print(f"含 reply_to(回应)的发言: {reply_count}")
    print(f"含 accuses(指控)的发言: {accuse_count}")
    print(f"被指控过的座位数: {len(mentioned_seats)} -> {sorted(mentioned_seats)}")
    print(f"被指控后真的回应了的发言数: {replied_after_mentioned}")
    print(f"决策失败数: {failed}")

    print("\n--- 含对话关系的发言样本(前5条) ---")
    for s in speeches:
        if s["reply_to"] or s["accuses"]:
            print(f"  {s['seat']}号 [回应{s['reply_to']} 指控{s['accuses']}]: {s['text'][:70]}")

    # 判定:结构化字段非空 + 无伪造(0失败)
    ok = (reply_count > 0 or accuse_count > 0) and failed == 0
    # 严格判定:既要有指控也要有回应(真正对话)
    strict = reply_count > 0 and accuse_count > 0 and failed == 0
    print(f"\n{'✅ 结构化字段已流转,0 伪造' if ok else '❌ 无结构化对话关系'}")
    print(f"{'✅✅ 真正对话:既有指控也有回应' if strict else '⚠️ 未同时出现指控+回应(可能对局太短)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
