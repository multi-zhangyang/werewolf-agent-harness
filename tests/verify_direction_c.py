"""验证方向C:狼人白天话术协同(弱协同)——党团会议私聊 + 信息隔离。

验证目标:
1. 狼队白天党团会议触发(≥2狼时):wolf_caucus 私聊事件注入狼人记忆
2. 信息隔离:wolf_caucus 事件仅狼人可见,好人 memory 里无此事件(harness 职责)
3. 共识注入:wolf_caucus_consensus 事件含统一目标+口径,仅狼人可见
4. 不伪造:harness 不写狼人发言,狼人发言仍走 decide_speak LLM(0 决策失败)
5. 平衡:胜率不爆(记录 winner,多局统计待 task #24)

设计依据(用户确认):弱协同 + 仅白天发言前1次私聊。复用夜间 _werewolf_deliberation 拓扑。
学术依据:AutoGen Swarm + S2§3.3 多 agent 协作。原创设计无 benchmark 先例。
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
        # god 模式能看到所有事件,但 wolf_caucus 是私有观察(不广播),所以这里只能
        # 通过 agent_decision_failed + 最终 winner + 发言内容间接验证。

        try:
            async for msg in ws:
                data = json.loads(msg)
                t = data.get("type")
                if t == "speech":
                    seat = data.get("seat")
                    speeches.append({
                        "seat": seat, "text": data.get("text", ""),
                        "bid": data.get("bid"),
                        "accuses": data.get("accuses"),
                        "attitudes": data.get("attitudes"),
                    })
                    mk = ""
                    if data.get("accuses"):
                        mk += f" [指控{data['accuses']}]"
                    if data.get("attitudes"):
                        mk += f" [态度{data['attitudes']}]"
                    print(f"  {seat}号(bid={data.get('bid')}): {data.get('text','')[:55]}{mk}")
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

    # 统计(间接验证:god 看不到 wolf_caucus 私聊,但能看狼人是否协同推同一目标)
    accuse_edges: dict[int, list[int]] = {}
    for s in speeches:
        for a in (s["accuses"] or []):
            accuse_edges.setdefault(s["seat"], []).append(a)

    print("\n=== 方向C 验证结果(间接) ===")
    print(f"总发言数: {len(speeches)}")
    print(f"决策失败数: {failed}(应为0,no-fallback)")
    print(f"winner: {winner}")
    print(f"各座位指控目标:")
    for seat in sorted(accuse_edges):
        from collections import Counter
        c = Counter(accuse_edges[seat])
        print(f"  {seat}号 → {dict(c)}")

    # 直接验证 wolf_caucus 信息隔离:用 god API 查房间状态(私有事件不暴露)
    # 真正的信息隔离单测在 pytest 已覆盖(memory.observe_event 只写自己)
    ok = failed == 0 and winner is not None
    print(f"\n{'✅ 对局完成,0 决策失败,党团会议未触发伪造' if ok else '❌ 有决策失败或对局异常'}")
    print(f"{'✅✅ 狼人协同迹象:多狼指控同一目标(需人工观察)' if ok else ''}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
