"""RulesEngine 单元测试。"""
from __future__ import annotations

import random

import pytest

from src.game.models import NightAction, NightActionType, Vote
from src.game.roles import Role, Team, default_role_deck
from src.game.rules import RulesEngine, RulesError
from src.game.state import new_game


@pytest.fixture
def six_player_game():
    names = ["A", "B", "C", "D", "E", "F"]
    state = new_game(names)
    deck = default_role_deck(6)
    RulesEngine.deal_roles(state, deck=deck, seed=42)
    return state


def test_role_deck_size():
    for n in range(6, 13):
        deck = default_role_deck(n)
        assert len(deck) == n
        assert deck.count(Role.WEREWOLF) >= 2


def test_deal_roles_requires_setup(six_player_game):
    state = six_player_game
    with pytest.raises(RulesError):
        RulesEngine.deal_roles(state, deck=default_role_deck(6), seed=42)


def test_night_kill_flow(six_player_game):
    state = six_player_game
    wolves = [p for p in state.players if p.role == Role.WEREWOLF]
    victims = [p for p in state.players if p.role != Role.WEREWOLF]
    assert wolves and victims

    RulesEngine.submit_night_action(
        state, NightAction(actor_id=wolves[0].id, action=NightActionType.KILL, target_id=victims[0].id)
    )
    RulesEngine.resolve_night(state)
    assert state.phase.value == "day"
    assert not victims[0].alive
    assert victims[0].death_reason.value == "wolf_kill"


def test_guard_blocks_kill(six_player_game):
    state = six_player_game
    wolves = [p for p in state.players if p.role == Role.WEREWOLF]
    guard = next((p for p in state.players if p.role == Role.GUARD), None)
    victim = next((p for p in state.players if p.role == Role.VILLAGER), None)
    if guard is None or victim is None:
        pytest.skip("need guard and villager")

    RulesEngine.submit_night_action(
        state, NightAction(actor_id=wolves[0].id, action=NightActionType.KILL, target_id=victim.id)
    )
    RulesEngine.submit_night_action(
        state, NightAction(actor_id=guard.id, action=NightActionType.GUARD, target_id=victim.id)
    )
    RulesEngine.resolve_night(state)
    assert victim.alive


def test_guard_and_save_same_target_dies(six_player_game):
    """同守同救:双救反死。"""
    state = six_player_game
    wolves = [p for p in state.players if p.role == Role.WEREWOLF]
    guard = next((p for p in state.players if p.role == Role.GUARD), None)
    witch = next((p for p in state.players if p.role == Role.WITCH), None)
    victim = next((p for p in state.players if p.role == Role.VILLAGER), None)
    if None in (guard, witch, victim):
        pytest.skip("need guard, witch and villager")

    RulesEngine.submit_night_action(
        state, NightAction(actor_id=wolves[0].id, action=NightActionType.KILL, target_id=victim.id)
    )
    RulesEngine.submit_night_action(
        state, NightAction(actor_id=guard.id, action=NightActionType.GUARD, target_id=victim.id)
    )
    RulesEngine.submit_night_action(
        state, NightAction(actor_id=witch.id, action=NightActionType.SAVE, target_id=victim.id)
    )
    RulesEngine.resolve_night(state)
    assert not victim.alive


def test_witch_poison_blocks_hunter_shot(six_player_game):
    state = six_player_game
    hunter = next((p for p in state.players if p.role == Role.HUNTER), None)
    witch = next((p for p in state.players if p.role == Role.WITCH), None)
    if hunter is None or witch is None:
        pytest.skip("need hunter and witch")

    RulesEngine.submit_night_action(
        state, NightAction(actor_id=witch.id, action=NightActionType.POISON, target_id=hunter.id)
    )
    RulesEngine.resolve_night(state)
    assert not hunter.alive
    assert hunter.death_reason.value == "poisoned"
    assert hunter.id not in state.pending_hunter


def test_vote_exile(six_player_game):
    state = six_player_game
    # 伪造进入投票阶段
    state.phase = state.phase.__class__("voting")
    living = state.living_players()
    target = living[0]
    for p in living[1:]:
        RulesEngine.submit_vote(state, Vote(voter_id=p.id, target_id=target.id))
    RulesEngine.submit_vote(state, Vote(voter_id=target.id, target_id=living[1].id))
    RulesEngine.resolve_vote(state, allow_pk=False)
    assert not target.alive
    assert target.death_reason.value == "exiled"


def test_vote_tie_no_exile(six_player_game):
    state = six_player_game
    state.phase = state.phase.__class__("voting")
    living = state.living_players()
    assert len(living) == 6
    a, b, c, d, e, f = living
    # 制造 a/b 3:3 平票
    RulesEngine.submit_vote(state, Vote(voter_id=a.id, target_id=b.id))
    RulesEngine.submit_vote(state, Vote(voter_id=b.id, target_id=a.id))
    RulesEngine.submit_vote(state, Vote(voter_id=c.id, target_id=a.id))
    RulesEngine.submit_vote(state, Vote(voter_id=d.id, target_id=b.id))
    RulesEngine.submit_vote(state, Vote(voter_id=e.id, target_id=a.id))
    RulesEngine.submit_vote(state, Vote(voter_id=f.id, target_id=b.id))
    RulesEngine.resolve_vote(state, allow_pk=False)
    assert a.alive and b.alive


def test_winner_wolves_outnumber(six_player_game):
    state = six_player_game
    wolves = [p for p in state.players if p.role == Role.WEREWOLF]
    others = [p for p in state.players if p.role != Role.WEREWOLF]
    # 只剩狼人和一个平民
    for p in others[1:]:
        p.alive = False
    winner = RulesEngine.check_winner(state)
    assert winner == Team.WEREWOLVES


def test_winner_village_all_wolves_dead(six_player_game):
    state = six_player_game
    for p in state.players:
        if p.role == Role.WEREWOLF:
            p.alive = False
    assert RulesEngine.check_winner(state) == Team.VILLAGE
