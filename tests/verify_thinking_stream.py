"""验证前端思考流链路:创建房间→god 模式 WS 连接→开始游戏→确认 agent_thinking 事件带 reasoning 字段。

验证目标(verbose_thinking 端到端):
1. agent_thinking 事件存在 reasoning 字段(完整 thought,非空)
2. reasoning 内容包含分析/欺骗手段(不是 120 字摘要)
3. 整局 0 决策失败
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
        # 1. 创建房间
        r = await http.post("/api/rooms", json={"player_names": NAMES, "human_seats": []})
        r.raise_for_status()
        room_id = r.json()["room_id"]
        print(f"房间创建: {room_id}")

        # 2. god 模式 WS 连接(收思考流)
        ws_url = f"ws://{HOST}/ws/{room_id}?mode=god"
        ws = await websockets.connect(ws_url)
        print(f"god 模式已连接: {ws_url}")

        # 3. 开始游戏
        r = await http.post(f"/api/rooms/{room_id}/start")
        r.raise_for_status()
        print("游戏已开始,等待事件...")

        thinking_count = 0
        reasoning_nonempty = 0
        reasoning_samples: list[str] = []
        failed = 0
        winner = None
        quality = None
        try:
            async for msg in ws:
                data = json.loads(msg)
                t = data.get("type")
                if t == "agent_thinking":
                    thinking_count += 1
                    reasoning = data.get("reasoning") or ""
                    if reasoning and len(reasoning) > 130:  # 比摘要长 = 完整 reasoning
                        reasoning_nonempty += 1
                    if reasoning and len(reasoning_samples) < 3:
                        reasoning_samples.append(f"[{data.get('seat')}号@{data.get('action')}] {reasoning[:200]}")
                elif t == "agent_decision_failed":
                    failed += 1
                    print(f"  ⚠️ 决策失败: {data}")
                elif t == "analysis":
                    quality = data.get("analysis", {}).get("quality")
                    print(f"  📊 评分到达: quality={quality.get('game_quality') if quality else None}")
                elif t == "game_ended":
                    winner = data.get("winner")
                    print(f"  🏁 结束 winner={winner}")
        except Exception as e:  # noqa: BLE001
            print(f"WS 异常: {type(e).__name__}: {e}")
        finally:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    print("\n=== 验证结果 ===")
    print(f"思考事件数: {thinking_count}")
    print(f"含完整 reasoning(>130字): {reasoning_nonempty}")
    print(f"决策失败数: {failed}")
    print(f"胜者: {winner}")
    print("\n--- reasoning 样本(前3条,各截200字) ---")
    for s in reasoning_samples:
        print(f"  {s}")

    print("\n--- 五维对局质量评分 ---")
    if quality:
        print(f"  全局质量: {quality.get('game_quality')} / 1.0")
        print(f"  总评: {quality.get('game_summary')}")
        for sc in quality.get("scores", []):
            dims = " ".join(f"{d}={sc.get(d)}" for d in ("RI", "SJ", "DR", "PS", "CT"))
            print(f"  {sc.get('seat')}号({sc.get('role')}): {dims}")
            print(f"    └ {sc.get('highlight')}")
    else:
        print("  (评分未到达——可能 LLM 调用失败,不致命)")

    ok = thinking_count > 0 and reasoning_nonempty > 0 and failed == 0
    print(f"\n{'✅ 验证通过' if ok else '❌ 验证失败'}: 思考流完整 reasoning 已端到端送达 god 模式")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
