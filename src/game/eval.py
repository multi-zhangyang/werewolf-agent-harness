"""评测与平衡性分析。

1. 蒙特卡洛胜率估计(纯随机/盲信预言家/真实 agent)。
2. 逐决策评分:发言/投票/技能/时机/影响力。
3. 赛后复盘:谁是关键手、谁误判、谁带节奏。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..game.models import GameState, Phase
from ..game.roles import Role, Team


class GameEvaluator:
    """赛后复盘与逐决策评分。"""

    @staticmethod
    def final_summary(state: GameState) -> dict[str, Any]:
        seats = [
            {"seat": p.seat, "name": p.name, "role": p.role, "team": p.team,
             "alive": p.alive, "death_reason": p.death_reason, "death_day": p.death_day}
            for p in state.players
        ]
        return {
            "winner": state.winner.value if state.winner else None,
            "days": state.day,
            "seats": seats,
        }

    @staticmethod
    def key_moments(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """提取关键节点:首夜死亡、首次放逐、平票、关键投票逆转。"""
        moments = []
        for ev in events:
            et = ev.get("type")
            if et == "night_resolved" and ev.get("day") == 1:
                moments.append({"type": "first_night", "message": ev.get("message")})
            elif et == "player_exiled" and not any(m.get("type") == "first_exile" for m in moments):
                moments.append({"type": "first_exile", "seat": ev.get("seat"), "message": ev.get("message")})
            elif et == "vote_tied":
                moments.append({"type": "tie", "day": ev.get("day"), "message": ev.get("message")})
            elif et == "hunter_shot":
                moments.append({"type": "hunter_turn", "seat": ev.get("seat"), "target": ev.get("target_seat")})
        return moments

    @staticmethod
    def vote_quality(state: GameState, events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        """逐投票评分:投给狼 +1,投给好人 -0.5,投给队友(狼互投) -1。"""
        role_map = {p.seat: p.role for p in state.players}
        scores: dict[int, dict[str, Any]] = {}
        for ev in events:
            if ev.get("type") != "vote_cast":
                continue
            seat = ev.get("seat")
            target = ev.get("target_seat")
            if seat is None or target is None:
                continue
            voter_role = role_map.get(seat)
            target_role = role_map.get(target)
            if voter_role is None or target_role is None:
                continue
            voter_team = Role(voter_role)
            target_team = Role(target_role)
            score = 0.0
            if voter_team == Role.WEREWOLF:
                if target_team == Role.WEREWOLF:
                    score = -1.0
                else:
                    score = 1.0
            else:
                if target_team == Role.WEREWOLF:
                    score = 1.0
                else:
                    score = -0.5
            entry = scores.setdefault(seat, {"votes": 0, "score": 0.0, "correct": 0})
            entry["votes"] += 1
            entry["score"] += score
            if score > 0:
                entry["correct"] += 1
        return scores

    @staticmethod
    def balance_check(wins: Counter) -> dict[str, Any]:
        total = sum(wins.values())
        if total == 0:
            return {"total": 0, "rates": {}, "balanced": False}
        rates = {k: round(v / total, 3) for k, v in wins.items()}
        village_rate = rates.get("village", 0.0)
        wolf_rate = rates.get("werewolves", 0.0)
        balanced = 0.4 <= village_rate <= 0.6 and 0.4 <= wolf_rate <= 0.6
        return {"total": total, "rates": rates, "balanced": balanced}
