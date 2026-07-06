"""盲信预言家基线:每晚随机查验,白天公开所有查验结果并盲信。"""
import random
from collections import Counter

import sys
sys.path.insert(0, ".")

from src.game.rules import RulesEngine
from src.game.roles import default_role_deck, Role
from src.game.state import new_game
from src.game.models import NightAction, NightActionType, Vote


def simulate_one(seed: int) -> str:
    rng = random.Random(seed)
    names = ["A", "B", "C", "D", "E", "F"]
    state = new_game(names)
    deck = default_role_deck(len(names))
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
                # 预言家随机查验一个未查过的活人
                checked = set(seer_results.keys())
                candidates = [p for p in state.living_players() if p.seat != seer.seat and p.seat not in checked]
                if candidates:
                    target = rng.choice(candidates)
                    team = "werewolves" if target.role.value == "werewolf" else "village"
                    seer_results[target.seat] = team
            if wolves and victims:
                target = rng.choice(victims)
                RulesEngine.submit_night_action(state, NightAction(actor_id=wolves[0].id, action=NightActionType.KILL, target_id=target.id))
            RulesEngine.resolve_night(state)
        elif state.phase.value in ("day", "voting"):
            living = state.living_players()
            if len(living) <= 2:
                break
            if state.phase.value == "day":
                RulesEngine.start_vote(state)
            # 盲信查验:如果有查出的狼,全体投他;否则随机
            known_wolves = [seat for seat, team in seer_results.items() if team == "werewolves"
                            and any(p.seat == seat and p.alive for p in living)]
            votes = {}
            for p in living:
                if known_wolves:
                    target = next(x for x in living if x.seat == known_wolves[0])
                else:
                    others = [x for x in living if x.id != p.id]
                    target = rng.choice(others)
                votes[p.id] = target.id
            for voter, target in votes.items():
                RulesEngine.submit_vote(state, Vote(voter_id=voter, target_id=target))
            RulesEngine.resolve_vote(state)
    return state.winner.value if state.winner else "draw"


wins = Counter()
total = 500
for i in range(total):
    wins[simulate_one(i)] += 1
print("盲信预言家基线:", dict(wins))
print(f"村民胜率: {wins.get('village',0)/total:.1%}")
