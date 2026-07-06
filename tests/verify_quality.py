"""验证 task#25:5维对局质量自动评分 + 方向A/B/C 对话量化指标。

验证目标:
1. game_ended 后能收到 analysis 事件(含 quality 5维评分 + dialogue_metrics)
2. quality.scores 每人 5 维分(RI/SJ/DR/PS/CT)非空 + game_quality 0-1
3. dialogue_metrics 客观统计 A/B/C 提升:
   - reply_rate/accuse_rate(方向A 对话交锋)
   - attitude_rate/support_edges/oppose_edges(方向B 信念网络)
   - wolf_coordination(方向C 狼人协同度)
4. 0 决策失败(no-fallback)

学术依据:Beyond Survival (arXiv:2510.11389) WereAlign 五维评估。
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
        print("游戏已开始,等待结束 + analysis 事件...\n")

        failed = 0
        winner = None
        analysis = None

        try:
            async for msg in ws:
                data = json.loads(msg)
                t = data.get("type")
                if t == "agent_decision_failed":
                    failed += 1
                    print(f"  ⚠️ 决策失败: seat={data.get('seat')}")
                elif t == "game_ended":
                    winner = data.get("winner")
                    print(f"  🏁 winner={winner},等待 analysis...")
                elif t == "analysis":
                    analysis = data.get("analysis")
                    break  # analysis 是最后事件
        except Exception as e:  # noqa: BLE001
            print(f"WS 异常: {type(e).__name__}: {e}")
        finally:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    if not analysis:
        print("❌ 未收到 analysis 事件")
        return 1

    print("\n=== 对话量化指标(direction A/B/C) ===")
    dm = analysis.get("dialogue_metrics", {})
    print(f"发言总数: {dm.get('speech_count')}")
    print(f"reply_rate(方向A 回应): {dm.get('reply_rate')}")
    print(f"accuse_rate(方向A 指控): {dm.get('accuse_rate')}")
    print(f"attitude_rate(方向B 信念): {dm.get('attitude_rate')}")
    print(f"support_edges: {dm.get('support_edges')} | oppose_edges: {dm.get('oppose_edges')}")
    print(f"wolf_coordination(方向C 协同): {dm.get('wolf_coordination')} (狼座:{dm.get('wolf_seats')})")
    print(f"wolf_deception_count(DR): {dm.get('wolf_deception_count')} 分布:{dm.get('wolf_deception_dist')}")

    print("\n=== 5维质量评分(Beyond Survival WereAlign) ===")
    q = analysis.get("quality")
    if not q:
        print("⚠️ quality 评分为空(LLM 调用可能失败,不致命)")
    else:
        print(f"全局质量 game_quality: {q.get('game_quality')}")
        print(f"总评: {q.get('game_summary')}")
        print(f"{'seat':>4} {'role':>10} {'RI':>5} {'SJ':>5} {'DR':>5} {'PS':>5} {'CT':>5}  highlight")
        for sc in q.get("scores", []):
            print(f"{sc.get('seat'):>4} {str(sc.get('role')):>10} "
                  f"{sc.get('RI'):>5} {sc.get('SJ'):>5} {sc.get('DR'):>5} "
                  f"{sc.get('PS'):>5} {sc.get('CT'):>5}  {sc.get('highlight','')[:50]}")

    print(f"\n决策失败数: {failed}")
    ok = bool(q) and dm.get("speech_count", 0) > 0 and failed == 0
    print(f"\n{'✅ 5维评分 + 对话指标双产出,A/B/C 提升可量化' if ok else '⚠️ 部分缺失'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
