"""蒙特卡洛平衡性测试(纯随机基线)。"""
import asyncio
import random
from collections import Counter

import sys
sys.path.insert(0, ".")

from src.game.rules import RulesEngine
from src.game.roles import default_role_deck
from src.game.state import new_game
from src.game.models import NightAction, NightActionType, Vote


def simulate_one(names: list[str], seed: int) -> str:
    rng = random.Random(seed)
    state = new_game(names)
    deck = default_role_deck(len(names))
    RulesEngine.deal_roles(state, deck=deck, seed=seed)

    for _ in range(20):
        if state.phase.value == "ended":
            break
        if state.phase.value == "night":
            wolves = [p for p in state.living_players() if p.role.value == "werewolf"]
            victims = [p for p in state.living_players() if p.role.value != "werewolf"]
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
            votes = {}
            for p in living:
                others = [x for x in living if x.id != p.id]
                votes[p.id] = rng.choice(others).id
            for voter, target in votes.items():
                RulesEngine.submit_vote(state, Vote(voter_id=voter, target_id=target))
            RulesEngine.resolve_vote(state)
    return state.winner.value if state.winner else "draw"


async def main():
    names = ["A", "B", "C", "D", "E", "F"]
    wins = Counter()
    total = 500
    for i in range(total):
        wins[simulate_one(names, i)] += 1
    print("随机基线:", dict(wins))
    village_rate = wins.get("village", 0) / total
    wolf_rate = wins.get("werewolves", 0) / total
    print(f"村民胜率: {village_rate:.1%}, 狼人胜率: {wolf_rate:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
