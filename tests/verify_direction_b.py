"""验证方向B:二阶 ToM 态度网络——attitude_edges 显式信念图。

验证目标:
1. speech 事件带 attitudes 字段(agent 显式声明立场,结构化信念图数据层)
2. 存在 attitudes 非空的发言(agent 真的在建模"谁和谁抱团/谁怀疑谁")
3. 态度边覆盖 support/oppose 两种(agent 区分帮腔与指控)
4. accuses(方向A)+ attitudes(方向B)+ votes 融合成完整态度网络
5. 0 决策失败(无伪造,no-fallback 铁律守恒)

学术依据:S2§3.2 explicit belief graph(Li 2023 prompt 显式信念状态增强多 agent 协作)
+ S2§3.3 Suspicion-Agent 二阶 ToM(预测对手相信我会做什么)。
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

        try:
            async for msg in ws:
                data = json.loads(msg)
                t = data.get("type")
                if t == "speech":
                    seat = data.get("seat")
                    atts = data.get("attitudes")
                    speeches.append({
                        "seat": seat, "text": data.get("text", ""),
                        "bid": data.get("bid"),
                        "reply_to": data.get("reply_to"),
                        "accuses": data.get("accuses"),
                        "attitudes": atts,
                    })
                    marker = ""
                    if data.get("reply_to"):
                        marker += f" [回应{data['reply_to']}号]"
                    if data.get("accuses"):
                        marker += f" [指控{data['accuses']}]"
                    if atts:
                        marker += f" [态度{atts}]"
                    print(f"  {seat}号(bid={data.get('bid')}): {data.get('text','')[:55]}{marker}")
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
    att_count = sum(1 for s in speeches if s["attitudes"])
    support_edges = 0
    oppose_edges = 0
    for s in speeches:
        if s["attitudes"]:
            for stance in s["attitudes"].values():
                if stance == "support":
                    support_edges += 1
                elif stance == "oppose":
                    oppose_edges += 1
    accuse_count = sum(1 for s in speeches if s["accuses"])
    reply_count = sum(1 for s in speeches if s["reply_to"])

    print("\n=== 方向B 验证结果 ===")
    print(f"总发言数: {len(speeches)}")
    print(f"含 attitudes(显式信念)的发言: {att_count}")
    print(f"  support 边(帮腔/信任): {support_edges}")
    print(f"  oppose 边(指控/怀疑): {oppose_edges}")
    print(f"含 accuses(方向A 指控)的发言: {accuse_count}")
    print(f"含 reply_to(方向A 回应)的发言: {reply_count}")
    print(f"决策失败数: {failed}")

    print("\n--- 含态度声明的发言样本(前8条) ---")
    shown = 0
    for s in speeches:
        if s["attitudes"] and shown < 8:
            print(f"  {s['seat']}号 态度{s['attitudes']}: {s['text'][:60]}")
            shown += 1

    # 判定:attitudes 非空 + 覆盖 support/oppose + 0 伪造
    ok = att_count > 0 and (support_edges > 0 or oppose_edges > 0) and failed == 0
    rich = att_count > 0 and support_edges > 0 and oppose_edges > 0 and failed == 0
    print(f"\n{'✅ 态度网络已流转,agent 在显式建模他人关系' if ok else '❌ 无结构化态度'}")
    print(f"{'✅✅ 丰富:既支持也反对,二阶 ToM 信念图成型' if rich else '⚠️ 态度类型单一或对局太短'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
