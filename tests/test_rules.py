"""RulesEngine 单元测试。"""
from __future__ import annotations

from collections import Counter
from itertools import permutations
import random

import pytest

from src.game.models import NightAction, NightActionType, Vote
from src.game.roles import (
    CLASSIC_RULESET_ID,
    Role,
    Team,
    default_role_deck,
    validate_role_deck,
)
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
    expected = {
        6: {Role.WEREWOLF: 2, Role.SEER: 1, Role.VILLAGER: 3},
        7: {Role.WEREWOLF: 2, Role.SEER: 1, Role.WITCH: 1, Role.VILLAGER: 3},
        8: {
            Role.WEREWOLF: 2, Role.SEER: 1, Role.WITCH: 1,
            Role.HUNTER: 1, Role.VILLAGER: 3,
        },
        9: {
            Role.WEREWOLF: 3, Role.SEER: 1, Role.WITCH: 1,
            Role.GUARD: 1, Role.HUNTER: 1, Role.VILLAGER: 2,
        },
        10: {
            Role.WEREWOLF: 3, Role.SEER: 1, Role.WITCH: 1,
            Role.GUARD: 1, Role.HUNTER: 1, Role.VILLAGER: 3,
        },
        11: {
            Role.WEREWOLF: 3, Role.SEER: 1, Role.WITCH: 1,
            Role.GUARD: 1, Role.HUNTER: 1, Role.VILLAGER: 4,
        },
        12: {
            Role.WEREWOLF: 4, Role.SEER: 1, Role.WITCH: 1,
            Role.GUARD: 1, Role.HUNTER: 1, Role.VILLAGER: 4,
        },
    }
    for n in range(6, 13):
        deck = default_role_deck(n)
        assert len(deck) == n
        assert deck.count(Role.WEREWOLF) >= 2
        assert Role.DOCTOR not in deck
        assert Counter(deck) == Counter(expected[n])


def test_classic_v1_accepts_every_implemented_role_capability():
    deck = validate_role_deck(
        [
            Role.WEREWOLF,
            Role.WEREWOLF,
            Role.WEREWOLF,
            Role.WEREWOLF,
            Role.SEER,
            Role.DOCTOR,
            Role.WITCH,
            Role.GUARD,
            Role.HUNTER,
            Role.VILLAGER,
            Role.VILLAGER,
            Role.VILLAGER,
        ],
        player_count=12,
        ruleset_id=CLASSIC_RULESET_ID,
    )

    assert set(deck) == set(Role)


@pytest.mark.parametrize(
    ("deck", "message"),
    [
        (
            [Role.WEREWOLF, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER],
            "size must match player count",
        ),
        ([Role.VILLAGER] * 6, "at least one werewolf"),
        ([Role.WEREWOLF] * 6, "at least one non-werewolf"),
        (
            [
                Role.WEREWOLF,
                Role.SEER,
                Role.SEER,
                Role.VILLAGER,
                Role.VILLAGER,
                Role.VILLAGER,
            ],
            "at most one of each power role",
        ),
        (
            ["werewolf", "villager", "villager", "villager", "villager", "cupid"],
            "unknown role",
        ),
    ],
)
def test_deal_roles_rejects_unplayable_or_unimplemented_decks_before_mutation(deck, message):
    state = new_game(["A", "B", "C", "D", "E", "F"])

    with pytest.raises(RulesError, match=message):
        RulesEngine.deal_roles(state, deck=deck, seed=1)

    assert state.phase.value == "setup"
    assert all(player.role is None for player in state.players)


def test_deal_roles_rejects_unknown_ruleset_before_mutation():
    state = new_game(["A", "B", "C", "D", "E", "F"])

    with pytest.raises(RulesError, match="unsupported Werewolf ruleset"):
        RulesEngine.deal_roles(
            state,
            deck=default_role_deck(6),
            seed=1,
            ruleset_id="classic.v2",
        )

    assert state.phase.value == "setup"
    assert all(player.role is None for player in state.players)


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


@pytest.mark.parametrize(
    ("target_id", "message"),
    [
        ("", "target is required"),
        ("   ", "target is required"),
        ("missing-player", "target must be a known player"),
    ],
)
def test_submit_night_action_rejects_missing_or_unknown_target_without_mutation(
    six_player_game,
    target_id,
    message,
):
    actor = next(player for player in six_player_game.players if player.role == Role.SEER)
    event_count = len(six_player_game.events)

    with pytest.raises(RulesError, match=message):
        RulesEngine.submit_night_action(
            six_player_game,
            NightAction(
                actor_id=actor.id,
                action=NightActionType.SEE,
                target_id=target_id,
            ),
        )

    assert six_player_game.night_actions == []
    assert len(six_player_game.events) == event_count


@pytest.mark.parametrize(
    ("role", "action"),
    [
        (Role.SEER, NightActionType.SEE),
        (Role.WITCH, NightActionType.POISON),
    ],
)
def test_night_actions_match_advertised_self_target_restrictions(role, action):
    state = new_game(["A", "B", "C", "D", "E", "F", "G"])
    RulesEngine.deal_roles(state, deck=default_role_deck(7), seed=42)
    actor = next(player for player in state.players if player.role == role)
    event_count = len(state.events)

    with pytest.raises(RulesError, match="cannot target the acting player"):
        RulesEngine.submit_night_action(
            state,
            NightAction(actor_id=actor.id, action=action, target_id=actor.id),
        )

    assert state.night_actions == []
    assert len(state.events) == event_count


def test_wolf_council_message_preserves_exact_nonblank_text(six_player_game):
    wolf = next(player for player in six_player_game.players if player.role == Role.WEREWOLF)
    target = next(player for player in six_player_game.players if player.role != Role.WEREWOLF)
    text = "  先考虑这个目标，再看队友意见。\n"

    event = RulesEngine.record_wolf_council_message(
        six_player_game,
        actor_id=wolf.id,
        target_id=target.id,
        message=text,
    )

    assert event.message == text
    assert six_player_game.events[-1].message == text


def test_wolf_council_unknown_target_is_a_rules_error_without_mutation(six_player_game):
    wolf = next(player for player in six_player_game.players if player.role == Role.WEREWOLF)
    event_count = len(six_player_game.events)

    with pytest.raises(RulesError, match="player must be known"):
        RulesEngine.record_wolf_council_message(
            six_player_game,
            actor_id=wolf.id,
            target_id="missing-player",
            message="proposal",
        )

    assert len(six_player_game.events) == event_count


def test_last_words_low_level_boundary_rejects_unknown_empty_and_duplicate_without_mutation(six_player_game):
    dead = six_player_game.players[0]
    dead.alive = False
    event_count = len(six_player_game.events)

    with pytest.raises(RulesError, match="player must be known"):
        RulesEngine.queue_last_words(six_player_game, "missing-player", reason="exiled")
    with pytest.raises(RulesError, match="text must not be empty"):
        RulesEngine.record_last_words(six_player_game, dead.id, " \n")

    RulesEngine.queue_last_words(six_player_game, dead.id, reason="exiled")
    with pytest.raises(RulesError, match="already queued"):
        RulesEngine.queue_last_words(six_player_game, dead.id, reason="exiled")

    assert len(six_player_game.events) == event_count
    assert len(six_player_game.last_words_queue) == 1


def test_hunter_unknown_target_does_not_consume_pending_shot():
    state = new_game(["A", "B", "C", "D", "E", "F", "G", "H"])
    RulesEngine.deal_roles(state, deck=default_role_deck(8), seed=42)
    hunter = next(player for player in state.players if player.role == Role.HUNTER)
    hunter.alive = False
    state.pending_hunter = [hunter.id]
    event_count = len(state.events)

    with pytest.raises(RulesError, match="known player"):
        RulesEngine.hunter_shoot(state, hunter.id, "missing-player")

    assert state.pending_hunter == [hunter.id]
    assert len(state.events) == event_count


def test_vote_unknown_target_is_a_rules_error_without_mutation(six_player_game):
    six_player_game.phase = six_player_game.phase.__class__("voting")
    voter = six_player_game.living_players()[0]
    event_count = len(six_player_game.events)

    with pytest.raises(RulesError, match="player must be known"):
        RulesEngine.submit_vote(
            six_player_game,
            Vote(voter_id=voter.id, target_id="missing-player"),
        )

    assert six_player_game.votes == {}
    assert len(six_player_game.events) == event_count


def test_night_resolution_is_invariant_to_action_submission_order():
    """Distinct actors' simultaneous choices must not inherit submission order."""
    baseline = new_game([f"P{seat}" for seat in range(1, 10)])
    RulesEngine.deal_roles(
        baseline,
        deck=[
            Role.WEREWOLF,
            Role.WEREWOLF,
            Role.SEER,
            Role.DOCTOR,
            Role.WITCH,
            Role.GUARD,
            Role.HUNTER,
            Role.VILLAGER,
            Role.VILLAGER,
        ],
        seed=41,
    )
    wolves = sorted(
        (player for player in baseline.players if player.role == Role.WEREWOLF),
        key=lambda player: player.seat,
    )
    villagers = sorted(
        (player for player in baseline.players if player.role == Role.VILLAGER),
        key=lambda player: player.seat,
    )
    seer = next(player for player in baseline.players if player.role == Role.SEER)
    doctor = next(player for player in baseline.players if player.role == Role.DOCTOR)
    witch = next(player for player in baseline.players if player.role == Role.WITCH)
    guard = next(player for player in baseline.players if player.role == Role.GUARD)
    lower_target, higher_target = villagers

    actions = (
        NightAction(
            actor_id=wolves[0].id,
            action=NightActionType.KILL,
            target_id=lower_target.id,
        ),
        NightAction(
            actor_id=wolves[1].id,
            action=NightActionType.KILL,
            target_id=higher_target.id,
        ),
        NightAction(
            actor_id=seer.id,
            action=NightActionType.SEE,
            target_id=wolves[0].id,
        ),
        NightAction(
            actor_id=doctor.id,
            action=NightActionType.SAVE,
            target_id=lower_target.id,
        ),
        NightAction(
            actor_id=witch.id,
            action=NightActionType.POISON,
            target_id=higher_target.id,
        ),
        NightAction(
            actor_id=guard.id,
            action=NightActionType.GUARD,
            target_id=lower_target.id,
        ),
    )

    expected_snapshot = None
    for submission_order in permutations(actions):
        state = baseline.model_copy(deep=True)
        for action in submission_order:
            RulesEngine.submit_night_action(state, action)
        RulesEngine.resolve_night(state)

        wolf_kill_ids = [
            item["id"] for item in state.night_deaths if item["reason"] == "wolf_kill"
        ]
        assert wolf_kill_ids == [lower_target.id]
        assert state.get_player(higher_target.id).death_reason == "poisoned"

        seer_results = [event for event in state.events if event.type == "seer_result"]
        snapshot = {
            "players": [
                (player.seat, player.alive, player.death_reason, player.death_day)
                for player in sorted(state.players, key=lambda item: item.seat)
            ],
            "night_deaths": state.night_deaths,
            "last_guarded_seat": state.last_guarded_seat,
            "pending_hunter": state.pending_hunter,
            "seer_results": [
                (event.recipients, event.payload)
                for event in seer_results
            ],
            "phase": state.phase,
        }
        if expected_snapshot is None:
            expected_snapshot = snapshot
        assert snapshot == expected_snapshot

    assert expected_snapshot is not None
    assert expected_snapshot["last_guarded_seat"] == lower_target.seat
    assert expected_snapshot["seer_results"] == [
        ([seer.id], {
            "target_id": wolves[0].id,
            "target_seat": wolves[0].seat,
            "team": Team.WEREWOLVES,
        })
    ]


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


def test_doctor_save_prevents_wolf_kill_without_consuming_witch_antidote():
    state = new_game(["A", "B", "C", "D", "E", "F"])
    RulesEngine.deal_roles(
        state,
        deck=[
            Role.WEREWOLF,
            Role.WEREWOLF,
            Role.SEER,
            Role.DOCTOR,
            Role.VILLAGER,
            Role.VILLAGER,
        ],
        seed=17,
    )
    wolf = next(player for player in state.players if player.role == Role.WEREWOLF)
    doctor = next(player for player in state.players if player.role == Role.DOCTOR)
    victim = next(
        player for player in state.players
        if player.role not in {Role.WEREWOLF, Role.DOCTOR}
    )

    RulesEngine.submit_night_action(
        state,
        NightAction(actor_id=wolf.id, action=NightActionType.KILL, target_id=victim.id),
    )
    RulesEngine.submit_night_action(
        state,
        NightAction(actor_id=doctor.id, action=NightActionType.SAVE, target_id=victim.id),
    )

    assert state.witch_antidote is True
    RulesEngine.resolve_night(state)

    assert victim.alive is True
    assert state.night_deaths == []
    assert state.witch_antidote is True


def test_doctor_save_does_not_block_witch_poison():
    state = new_game(["A", "B", "C", "D", "E", "F"])
    RulesEngine.deal_roles(
        state,
        deck=[
            Role.WEREWOLF,
            Role.WEREWOLF,
            Role.SEER,
            Role.DOCTOR,
            Role.WITCH,
            Role.VILLAGER,
        ],
        seed=23,
    )
    doctor = next(player for player in state.players if player.role == Role.DOCTOR)
    witch = next(player for player in state.players if player.role == Role.WITCH)
    victim = next(
        player for player in state.players
        if player.role not in {Role.WEREWOLF, Role.DOCTOR, Role.WITCH}
    )

    RulesEngine.submit_night_action(
        state,
        NightAction(actor_id=doctor.id, action=NightActionType.SAVE, target_id=victim.id),
    )
    RulesEngine.submit_night_action(
        state,
        NightAction(actor_id=witch.id, action=NightActionType.POISON, target_id=victim.id),
    )
    RulesEngine.resolve_night(state)

    assert victim.alive is False
    assert victim.death_reason.value == "poisoned"


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


def test_incomplete_vote_does_not_exile_from_partial_tally(six_player_game):
    state = six_player_game
    state.phase = state.phase.__class__("voting")
    living = state.living_players()
    voter, target = living[0], living[1]

    RulesEngine.submit_vote(state, Vote(voter_id=voter.id, target_id=target.id))
    RulesEngine.resolve_vote(state, allow_pk=False, require_all=False)

    assert target.alive
    assert state.events[-1].type == "vote_tied"
    assert state.events[-1].payload["missing_player_ids"]


def test_submit_vote_rejects_self_vote_without_mutating_state(six_player_game):
    state = six_player_game
    state.phase = state.phase.__class__("voting")
    voter = state.living_players()[0]
    event_count = len(state.events)

    with pytest.raises(RulesError, match="cannot vote for themselves"):
        RulesEngine.submit_vote(
            state,
            Vote(voter_id=voter.id, target_id=voter.id),
        )

    assert state.votes == {}
    assert len(state.events) == event_count


def test_incomplete_vote_exiles_when_effective_votes_have_majority(six_player_game):
    state = six_player_game
    state.phase = state.phase.__class__("voting")
    living = state.living_players()
    target = living[0]

    for voter in living[1:5]:
        RulesEngine.submit_vote(state, Vote(voter_id=voter.id, target_id=target.id))
    RulesEngine.submit_vote(state, Vote(voter_id=target.id, target_id=living[1].id))
    RulesEngine.resolve_vote(state, allow_pk=False, require_all=False)

    assert not target.alive
    assert state.events[-1].type == "player_exiled"


def test_pk_vote_rejects_non_candidate_target(six_player_game):
    state = six_player_game
    state.phase = state.phase.__class__("voting")
    living = state.living_players()
    state.pk_candidates = [living[0].id, living[1].id]

    with pytest.raises(RulesError):
        RulesEngine.submit_vote(state, Vote(voter_id=living[2].id, target_id=living[3].id))


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
