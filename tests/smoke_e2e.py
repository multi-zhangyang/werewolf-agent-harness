"""端到端真实对局 smoke test —— 跑一局完整 AI 狼人杀验证编排器。"""
import asyncio
import sys
import logging
import os
from typing import Any

sys.path.insert(0, ".")

from src.game.rules import RulesEngine
from src.game.roles import Role, default_role_deck
from src.game.state import new_game
from src.game.orchestrator import GameOrchestratorV2, build_actors
from src.llm.router import LLMRouter
from src.llm.models import ModelConfig
from src.config import DEFAULT_MODEL_CONFIG, LLM_MAX_RETRIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("smoke")


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _quality_is_complete(quality: Any) -> bool:
    if not isinstance(quality, dict):
        return False
    game_quality = quality.get("game_quality")
    if not isinstance(game_quality, (int, float)) or not 0 <= float(game_quality) <= 1:
        return False
    scores = quality.get("scores")
    if not isinstance(scores, list) or not scores:
        return False
    for score in scores:
        if not isinstance(score, dict):
            return False
        for dim in ("RI", "SJ", "DR", "PS", "CT"):
            value = score.get(dim)
            if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                return False
    return True


async def main() -> int:
    names = ["李白", "杜甫", "苏轼", "辛弃疾", "陆游", "王维"]
    state = new_game(names)
    deck = default_role_deck(len(names))
    RulesEngine.deal_roles(state, deck=deck)
    print("角色分配:", [(p.seat, p.role) for p in state.players])

    cfg = ModelConfig(**DEFAULT_MODEL_CONFIG)
    cfg.use_json_format = False
    router = LLMRouter(timeout=120, max_retries=LLM_MAX_RETRIES, concurrency=6)
    actors = build_actors(state, model_config=cfg, router=router)

    events_log: list[dict] = []
    thinking_log: list[dict] = []

    async def on_event(ev):
        events_log.append(ev)
        et = ev.get("type", "")
        if et in ("phase_started", "night_resolved", "speech", "vote_cast", "vote_resolved",
                  "last_words", "hunter_shot", "game_ended", "agent_decision_failed", "analysis"):
            if et == "speech":
                seat = ev.get("seat")
                bid = ev.get("bid")
                claim = ev.get("claim")
                claim_str = f" [claim={claim}]" if claim else ""
                print(f"  [{seat}号 bid={bid}]{claim_str} {ev.get('text','')}")
            elif et == "agent_decision_failed":
                print(f"  ⚠️ 决策失败 seat={ev.get('seat')} phase={ev.get('phase')}: {ev.get('reason','')}")
            elif et == "game_ended":
                print(f"  🏁 游戏结束 winner={ev.get('winner')}")
            elif et == "phase_started":
                print(f"\n=== {ev.get('phase')} D{ev.get('day')} ===")
            elif et == "night_resolved":
                print(f"  🌅 夜晚结算: {ev.get('message','')}")
            elif et == "vote_cast":
                print(f"  🗳️ {ev.get('voter_seat')}号 → {ev.get('target_seat')}号")
            elif et == "vote_resolved":
                print(f"  📊 {ev.get('message','')}")
            elif et == "last_words":
                print(f"  💀 遗言[{ev.get('seat')}号]: {ev.get('text','')}")
            elif et == "hunter_shot":
                print(f"  🔫 猎人{ev.get('seat')}号开枪→{ev.get('target_seat')}号")

    async def on_thinking(t):
        thinking_log.append(t)
        # 完整打印每个 agent 的思考过程(分析/欺骗算计/手段)——这是多 agent 对抗的核心可观察层
        seat = t.get("seat")
        action = t.get("action", "")
        summary = t.get("summary") or ""
        reasoning = t.get("reasoning") or ""
        bid = t.get("bid")
        top = t.get("suspicion_top") or []
        top_str = ", ".join(f"{x.get('seat')}号({x.get('suspicion')})" for x in top[:3]) if top else ""
        bid_str = f" bid={bid}" if bid is not None else ""
        # verbose 模式下 reasoning 即完整 thought;否则用 summary
        body = reasoning or summary
        print(f"  💭 [{seat}号 @{action}{bid_str}] {body}")
        if top_str:
            print(f"       怀疑: {top_str}")

    orch = GameOrchestratorV2(
        state=state, actors=actors, deck=deck,
        on_event=on_event, on_thinking=on_thinking,
        max_speak_rounds=3,
        verbose_thinking=True,  # 暴露完整推理(分析/欺骗算计/手段),供研究观察多 agent 对抗
    )

    ok = True
    timeout = float(os.getenv("WEREWOLF_SMOKE_TIMEOUT", "900"))
    try:
        await asyncio.wait_for(orch.run(), timeout=timeout)
    except asyncio.TimeoutError:
        ok = False
        print(f"\n⏱️ 对局超时({timeout:.0f}s)")
    except Exception as e:
        ok = False
        print(f"\n❌ 对局异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await router.aclose()

    print(f"\n=== 统计 ===")
    print(f"事件总数: {len(events_log)}")
    print(f"思考摘要数: {len(thinking_log)}")
    print(f"LLM 调用统计: {router.stats.snapshot()}")
    print(f"最终阶段: {state.phase}, 胜者: {state.winner}")
    print(f"存活: {[(p.seat, p.role, p.alive) for p in state.players]}")
    for a in actors.values():
        top = sorted(a.memory.trust.items(), key=lambda kv: kv[1], reverse=True)[:2]
        print(f"  {a.seat}号({a.role.value}/{a.persona_name}) 怀疑最高: {top}")
    failed_events = [ev for ev in events_log if ev.get("type") == "agent_decision_failed"]
    if failed_events:
        ok = False
        print(f"agent_decision_failed: {len(failed_events)}")
    if state.phase != "ended" or not state.winner:
        ok = False
        print("smoke 未完成到 ended/winner")
    if _truthy_env("WEREWOLF_SMOKE_REQUIRE_QUALITY"):
        analysis_events = [ev for ev in events_log if ev.get("type") == "analysis"]
        analysis = analysis_events[-1].get("analysis", {}) if analysis_events else {}
        quality = analysis.get("quality") if isinstance(analysis, dict) else None
        if not _quality_is_complete(quality):
            ok = False
            print("smoke quality judge 未成功或 quality 结构不完整")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
