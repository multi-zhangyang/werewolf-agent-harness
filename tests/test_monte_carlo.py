"""蒙特卡洛基线测试(纯随机/盲信预言家)作为 pytest 可发现用例。

这些测试运行较快,验证规则引擎不会崩溃并给出合理胜率区间。
"""
from __future__ import annotations

import random
from collections import Counter

import pytest

from src.game.models import NightAction, NightActionType, Vote
from src.game.roles import Role, default_role_deck
from src.game.rules import RulesEngine
from src.game.state import new_game


NAMES = ["A", "B", "C", "D", "E", "F"]


def _simulate_random(seed: int) -> str:
    rng = random.Random(seed)
    state = new_game(NAMES)
    deck = default_role_deck(6)
    RulesEngine.deal_roles(state, deck=deck, seed=seed)
    for _ in range(20):
        if state.phase.value == "ended":
            break
        if state.phase.value == "night":
            wolves = [p for p in state.living_players() if p.role.value == "werewolf"]
            victims = [p for p in state.living_players() if p.role.value != "werewolf"]
            if wolves and victims:
                target = rng.choice(victims)
                RulesEngine.submit_night_action(
                    state,
                    NightAction(
                        actor_id=wolves[0].id,
                        action=NightActionType.KILL,
                        target_id=target.id,
                    ),
                )
            RulesEngine.resolve_night(state)
        elif state.phase.value in ("day", "voting"):
            living = state.living_players()
            if len(living) <= 2:
                break
            if state.phase.value == "day":
                RulesEngine.start_vote(state)
            votes = {}
            for p in living:
                others = [x for x in living if x.id != p.id]
                votes[p.id] = rng.choice(others).id
            for voter, target in votes.items():
                RulesEngine.submit_vote(state, Vote(voter_id=voter, target_id=target))
            RulesEngine.resolve_vote(state)
    return state.winner.value if state.winner else "draw"


def _simulate_seer_reveal(seed: int) -> str:
    rng = random.Random(seed)
    state = new_game(NAMES)
    deck = default_role_deck(6)
    RulesEngine.deal_roles(state, deck=deck, seed=seed)
    seer_results: dict[int, str] = {}

    for _ in range(20):
        if state.phase.value == "ended":
            break
        if state.phase.value == "night":
            wolves = [p for p in state.living_players() if p.role.value == "werewolf"]
            victims = [p for p in state.living_players() if p.role.value != "werewolf"]
            seer = next((p for p in state.living_players() if p.role.value == "seer"), None)
            if seer and victims:
                checked = set(seer_results.keys())
                candidates = [
                    p for p in state.living_players()
                    if p.seat != seer.seat and p.seat not in checked
                ]
                if candidates:
                    target = rng.choice(candidates)
                    team = "werewolves" if target.role.value == "werewolf" else "village"
                    seer_results[target.seat] = team
            if wolves and victims:
                target = rng.choice(victims)
                RulesEngine.submit_night_action(
                    state,
                    NightAction(
                        actor_id=wolves[0].id,
                        action=NightActionType.KILL,
                        target_id=target.id,
                    ),
                )
            RulesEngine.resolve_night(state)
        elif state.phase.value in ("day", "voting"):
            living = state.living_players()
            if len(living) <= 2:
                break
            if state.phase.value == "day":
                RulesEngine.start_vote(state)
            known_wolves = [
                seat for seat, team in seer_results.items()
                if team == "werewolves" and any(p.seat == seat and p.alive for p in living)
            ]
            for p in living:
                if known_wolves:
                    target = next(x for x in living if x.seat == known_wolves[0])
                else:
                    others = [x for x in living if x.id != p.id]
                    target = rng.choice(others)
                RulesEngine.submit_vote(state, Vote(voter_id=p.id, target_id=target.id))
            RulesEngine.resolve_vote(state)
    return state.winner.value if state.winner else "draw"


@pytest.mark.parametrize("simulator", [_simulate_random, _simulate_seer_reveal])
def test_monte_carlo_no_crash(simulator):
    """跑 50 局确保规则引擎不崩溃。"""
    results = Counter(simulator(i) for i in range(50))
    assert sum(results.values()) == 50
    assert "draw" not in results or results["draw"] < 5


def test_random_baseline_vs_seer_reveal():
    """盲信预言家基线应显著高于随机基线,验证信息价值梯度。"""
    n = 200
    random_wins = Counter(_simulate_random(i) for i in range(n))
    seer_wins = Counter(_simulate_seer_reveal(i) for i in range(n))
    random_rate = random_wins.get("village", 0) / n
    seer_rate = seer_wins.get("village", 0) / n
    assert seer_rate > random_rate + 0.05
