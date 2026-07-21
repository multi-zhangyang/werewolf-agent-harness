"""Executable classic.v1 evidence matrix for every advertised role capability."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import pytest

from src.agent.memory import AgentMemory
from src.agent.schemas import AgentAction, Decision
from src.game.models import DeathReason, NightAction, NightActionType, Phase
from src.game.orchestrator import GameOrchestratorV2
from src.game.roles import (
    CLASSIC_RULESET_ID,
    CLASSIC_V1_IMPLEMENTED_ROLES,
    Role,
    validate_role_deck,
)
from src.game.rules import RulesEngine
from src.game.state import new_game
from src.harness.agent_protocol import ActionRequest, DecisionEnvelope


FULL_CAPABILITY_DECK = [
    Role.WEREWOLF,
    Role.WEREWOLF,
    Role.SEER,
    Role.DOCTOR,
    Role.WITCH,
    Role.GUARD,
    Role.HUNTER,
    Role.VILLAGER,
    Role.VILLAGER,
]


@dataclass(frozen=True)
class RoleScenario:
    key: str
    role: Role
    request_action: str
    decision_action: AgentAction
    rule_action: str
    can_skip: bool
    success_memory_kinds: tuple[str, ...]


ROLE_SCENARIOS = (
    RoleScenario(
        key="werewolf.kill",
        role=Role.WEREWOLF,
        request_action="night_kill",
        decision_action=AgentAction.NIGHT_KILL,
        rule_action="kill",
        can_skip=False,
        success_memory_kinds=("wolf_kill_chosen",),
    ),
    RoleScenario(
        key="seer.inspect",
        role=Role.SEER,
        request_action="see",
        decision_action=AgentAction.SEE,
        rule_action="see",
        can_skip=False,
        success_memory_kinds=("seer_action", "seer_result"),
    ),
    RoleScenario(
        key="doctor.protect",
        role=Role.DOCTOR,
        request_action="save",
        decision_action=AgentAction.SAVE,
        rule_action="save",
        can_skip=True,
        success_memory_kinds=("doctor_protect_target",),
    ),
    RoleScenario(
        key="witch.save",
        role=Role.WITCH,
        request_action="save",
        decision_action=AgentAction.SAVE,
        rule_action="save",
        can_skip=True,
        success_memory_kinds=("witch_kill_preview", "witch_save_used"),
    ),
    RoleScenario(
        key="witch.poison",
        role=Role.WITCH,
        request_action="poison",
        decision_action=AgentAction.POISON,
        rule_action="poison",
        can_skip=True,
        success_memory_kinds=("witch_poison_used",),
    ),
    RoleScenario(
        key="guard.protect",
        role=Role.GUARD,
        request_action="guard",
        decision_action=AgentAction.GUARD,
        rule_action="guard",
        can_skip=False,
        success_memory_kinds=("guard_target",),
    ),
    RoleScenario(
        key="hunter.shoot",
        role=Role.HUNTER,
        request_action="hunter_shot",
        decision_action=AgentAction.NIGHT_KILL,
        rule_action="hunter_shot",
        can_skip=True,
        success_memory_kinds=("hunter_shot",),
    ),
    RoleScenario(
        key="villager.vote",
        role=Role.VILLAGER,
        request_action="vote",
        decision_action=AgentAction.VOTE,
        rule_action="vote",
        can_skip=False,
        success_memory_kinds=("vote",),
    ),
)


class MatrixActor:
    """Test-local AgentProtocol implementation with explicit per-request plans."""

    def __init__(self, *, seat: int, name: str, role: Role) -> None:
        self.seat = seat
        self.name = name
        self.role = role
        self.persona_name = "role-matrix"
        self.memory = AgentMemory(seat=seat, role=role.value)
        self.rng = random.Random(10_000 + seat)
        self.requests: list[ActionRequest] = []
        self.plans: dict[str, Decision | BaseException] = {}
        self.on_human_request = None

    async def decide(self, request: ActionRequest) -> DecisionEnvelope:
        self.requests.append(request)
        planned = self.plans.get(request.action_kind)
        if isinstance(planned, BaseException):
            raise planned
        if planned is None:
            if request.action_kind == "wolf_council":
                planned = Decision(
                    action=AgentAction.WOLF_COUNCIL,
                    target_seat=request.legal_actions[0].target_seats[0],
                    team_message=f"seat {self.seat} council proposal",
                )
            elif request.action_kind == "last_words":
                planned = Decision(
                    action=AgentAction.SKIP,
                    skip_reason="test_no_last_words",
                )
            else:
                raise RuntimeError(
                    f"role matrix has no plan for {self.role.value}/{request.action_kind}"
                )
        return DecisionEnvelope(
            request_id=request.request_id,
            seat=self.seat,
            decision=planned,
            parse_status="not_applicable",
            metadata={"agent_kind": "test"},
        )

    def observe_event(self, *args: Any, **metadata: Any) -> None:
        if len(args) >= 4:
            day, phase, kind, text = args[:4]
            self.memory.observe(day, phase, kind, text, **metadata)

    def record_claim(self, seat: int, day: int, claim: dict[str, Any]) -> None:
        self.memory.record_claim(seat, day, claim)


@dataclass
class MatrixHarness:
    orchestrator: GameOrchestratorV2
    events: list[dict[str, Any]]

    @property
    def state(self):
        return self.orchestrator.state

    @property
    def actors(self) -> dict[str, MatrixActor]:
        return self.orchestrator.actors  # type: ignore[return-value]


def _build_matrix_harness() -> MatrixHarness:
    state = new_game([f"P{seat}" for seat in range(1, 10)], game_id="role-matrix")
    deck = validate_role_deck(
        FULL_CAPABILITY_DECK,
        player_count=9,
        ruleset_id=CLASSIC_RULESET_ID,
    )
    RulesEngine.deal_roles(
        state,
        deck=deck,
        seed=1_337,
        ruleset_id=CLASSIC_RULESET_ID,
    )
    actors = {
        player.id: MatrixActor(
            seat=player.seat,
            name=player.name,
            role=Role(player.role),
        )
        for player in state.players
    }
    events: list[dict[str, Any]] = []

    async def capture_event(payload: dict[str, Any]) -> None:
        events.append(dict(payload))

    orchestrator = GameOrchestratorV2(
        state=state,
        actors=actors,  # type: ignore[arg-type]
        deck=deck,
        rng=random.Random(2_026),
        on_event=capture_event,
        on_trace=None,
        internal_events=True,
        decision_timeout=1,
        phase_deadline=0,
        max_speak_rounds=1,
    )
    return MatrixHarness(orchestrator=orchestrator, events=events)


def _player_for(harness: MatrixHarness, role: Role):
    return next(player for player in harness.state.players if player.role == role)


def _players_for(harness: MatrixHarness, role: Role):
    return [player for player in harness.state.players if player.role == role]


def _ordinary_villager(harness: MatrixHarness):
    return min(
        (player for player in harness.state.players if player.role == Role.VILLAGER),
        key=lambda player: player.seat,
    )


def _success_target(harness: MatrixHarness, scenario: RoleScenario):
    if scenario.key == "seer.inspect":
        return _player_for(harness, Role.WEREWOLF)
    if scenario.key in {"witch.poison", "hunter.shoot", "villager.vote"}:
        return _player_for(harness, Role.WEREWOLF)
    return _ordinary_villager(harness)


def _actor_for(harness: MatrixHarness, scenario: RoleScenario) -> MatrixActor:
    player = _player_for(harness, scenario.role)
    return harness.actors[player.id]


def _submit_wolf_kill_precondition(harness: MatrixHarness, target_id: str) -> None:
    wolf = _player_for(harness, Role.WEREWOLF)
    RulesEngine.submit_night_action(
        harness.state,
        NightAction(
            actor_id=wolf.id,
            action=NightActionType.KILL,
            target_id=target_id,
        ),
    )


async def _execute_success(
    harness: MatrixHarness,
    scenario: RoleScenario,
) -> tuple[MatrixActor, Any, ActionRequest]:
    orchestrator = harness.orchestrator
    await orchestrator._notify_role_assigned()
    actor = _actor_for(harness, scenario)
    target = _success_target(harness, scenario)
    actor.plans[scenario.request_action] = Decision(
        action=scenario.decision_action,
        target_seat=target.seat,
    )

    if scenario.key == "werewolf.kill":
        for wolf in _players_for(harness, Role.WEREWOLF):
            harness.actors[wolf.id].plans["night_kill"] = Decision(
                action=AgentAction.NIGHT_KILL,
                target_seat=target.seat,
            )
        await orchestrator._collect_werewolf_kill_proposals()
        RulesEngine.resolve_night(harness.state)
        await orchestrator._push_night_results_to_memory()
    elif scenario.key == "seer.inspect":
        await orchestrator._night_role_actions(Role.SEER, [NightActionType.SEE])
        RulesEngine.resolve_night(harness.state)
        await orchestrator._push_night_results_to_memory()
    elif scenario.key == "doctor.protect":
        _submit_wolf_kill_precondition(harness, target.id)
        await orchestrator._night_role_actions(Role.DOCTOR, [NightActionType.SAVE])
        RulesEngine.resolve_night(harness.state)
        await orchestrator._push_night_results_to_memory()
    elif scenario.key == "witch.save":
        _submit_wolf_kill_precondition(harness, target.id)
        await orchestrator._witch_save_phase()
        RulesEngine.resolve_night(harness.state)
        await orchestrator._push_night_results_to_memory()
    elif scenario.key == "witch.poison":
        await orchestrator._witch_poison_phase()
        RulesEngine.resolve_night(harness.state)
        await orchestrator._push_night_results_to_memory()
    elif scenario.key == "guard.protect":
        _submit_wolf_kill_precondition(harness, target.id)
        await orchestrator._night_role_actions(Role.GUARD, [NightActionType.GUARD])
        RulesEngine.resolve_night(harness.state)
        await orchestrator._push_night_results_to_memory()
    elif scenario.key == "hunter.shoot":
        hunter = _player_for(harness, Role.HUNTER)
        hunter.alive = False
        hunter.death_reason = DeathReason.EXILED
        harness.state.phase = Phase.DAY
        harness.state.pending_hunter = [hunter.id]
        await orchestrator._process_deaths_and_hunter()
    elif scenario.key == "villager.vote":
        harness.state.phase = Phase.VOTING
        primary_target = target
        alternate_wolf = next(
            wolf
            for wolf in _players_for(harness, Role.WEREWOLF)
            if wolf.id != primary_target.id
        )
        for player in harness.state.living_players():
            vote_target = alternate_wolf if player.id == primary_target.id else primary_target
            harness.actors[player.id].plans["vote"] = Decision(
                action=AgentAction.VOTE,
                target_seat=vote_target.seat,
            )
        await orchestrator._run_voting()
    else:  # pragma: no cover - the fixed matrix is exhaustive
        raise AssertionError(f"unhandled role scenario: {scenario.key}")

    request = next(
        item for item in actor.requests if item.action_kind == scenario.request_action
    )
    return actor, target, request


def _invalid_target(harness: MatrixHarness, scenario: RoleScenario, actor: MatrixActor):
    if scenario.key == "werewolf.kill":
        return next(
            player
            for player in _players_for(harness, Role.WEREWOLF)
            if player.seat != actor.seat
        )
    if scenario.key in {"seer.inspect", "witch.poison", "villager.vote"}:
        return harness.state.get_player(
            next(player.id for player in harness.state.players if player.seat == actor.seat)
        )
    if scenario.key == "witch.save":
        return _player_for(harness, Role.WITCH)
    if scenario.key == "guard.protect":
        return _player_for(harness, Role.GUARD)
    if scenario.key == "hunter.shoot":
        target = _ordinary_villager(harness)
        target.alive = False
        target.death_reason = DeathReason.EXILED
        return target
    return None


async def _execute_rejection(
    harness: MatrixHarness,
    scenario: RoleScenario,
) -> tuple[MatrixActor, int, ActionRequest]:
    orchestrator = harness.orchestrator
    await orchestrator._notify_role_assigned()
    actor = _actor_for(harness, scenario)
    actor_player = next(player for player in harness.state.players if player.seat == actor.seat)
    invalid_player = _invalid_target(harness, scenario, actor)
    invalid_seat = invalid_player.seat if invalid_player is not None else 999

    if scenario.key == "werewolf.kill":
        wolves = sorted(_players_for(harness, Role.WEREWOLF), key=lambda item: item.seat)
        for index, wolf in enumerate(wolves):
            teammate = wolves[1 - index]
            harness.actors[wolf.id].plans["night_kill"] = Decision(
                action=AgentAction.NIGHT_KILL,
                target_seat=teammate.seat,
            )
        await orchestrator._collect_werewolf_kill_proposals()
    elif scenario.key == "seer.inspect":
        actor.plans["see"] = Decision(action=AgentAction.SEE, target_seat=invalid_seat)
        await orchestrator._night_role_actions(Role.SEER, [NightActionType.SEE])
    elif scenario.key == "doctor.protect":
        actor.plans["save"] = Decision(action=AgentAction.SAVE, target_seat=invalid_seat)
        await orchestrator._night_role_actions(Role.DOCTOR, [NightActionType.SAVE])
    elif scenario.key == "witch.save":
        killed = _ordinary_villager(harness)
        _submit_wolf_kill_precondition(harness, killed.id)
        actor.plans["save"] = Decision(action=AgentAction.SAVE, target_seat=invalid_seat)
        await orchestrator._witch_save_phase()
    elif scenario.key == "witch.poison":
        actor.plans["poison"] = Decision(
            action=AgentAction.POISON,
            target_seat=invalid_seat,
        )
        await orchestrator._witch_poison_phase()
    elif scenario.key == "guard.protect":
        harness.state.last_guarded_seat = invalid_seat
        actor.plans["guard"] = Decision(
            action=AgentAction.GUARD,
            target_seat=invalid_seat,
        )
        await orchestrator._night_role_actions(Role.GUARD, [NightActionType.GUARD])
    elif scenario.key == "hunter.shoot":
        actor_player.alive = False
        actor_player.death_reason = DeathReason.EXILED
        harness.state.phase = Phase.DAY
        harness.state.pending_hunter = [actor_player.id]
        actor.plans["hunter_shot"] = Decision(
            action=AgentAction.NIGHT_KILL,
            target_seat=invalid_seat,
        )
        await orchestrator._process_deaths_and_hunter()
    elif scenario.key == "villager.vote":
        harness.state.phase = Phase.VOTING
        actor.plans["vote"] = Decision(
            action=AgentAction.VOTE,
            target_seat=invalid_seat,
        )
        wolves = _players_for(harness, Role.WEREWOLF)
        target = wolves[0]
        alternate = wolves[1]
        for player in harness.state.living_players():
            if player.id == actor_player.id:
                continue
            vote_target = alternate if player.id == target.id else target
            harness.actors[player.id].plans["vote"] = Decision(
                action=AgentAction.VOTE,
                target_seat=vote_target.seat,
            )
        await orchestrator._run_voting()
    else:  # pragma: no cover - the fixed matrix is exhaustive
        raise AssertionError(f"unhandled role scenario: {scenario.key}")

    for event in list(orchestrator._failed_events):
        await orchestrator._emit(event)
    orchestrator._failed_events.clear()
    request = next(
        item for item in actor.requests if item.action_kind == scenario.request_action
    )
    return actor, invalid_seat, request


def _trace_rows(orchestrator: GameOrchestratorV2, request_id: str):
    return [
        row
        for row in orchestrator._decision_trace
        if row.get("request_id") == request_id
        or row.get("request", {}).get("request_id") == request_id
    ]


def _assert_request_observation_and_legal_scope(
    harness: MatrixHarness,
    scenario: RoleScenario,
    actor: MatrixActor,
    request: ActionRequest,
    target_seat: int,
) -> None:
    observation = request.observation
    legal = request.legal_actions[0]
    assert request.run_id == harness.state.id
    assert request.seat == actor.seat
    assert request.action_kind == scenario.request_action
    assert observation["my_seat"] == actor.seat
    assert observation["my_role"] == scenario.role.value
    assert observation["available_actions"] == [scenario.request_action]
    assert observation["candidate_targets"] == legal.target_seats
    assert all("role" not in seat for seat in observation["seats"])
    assert all(
        actor_player_id in event.get("recipients", [])
        for event in observation["private_events"]
        for actor_player_id in [
            next(
                player.id for player in harness.state.players if player.seat == actor.seat
            )
        ]
    )
    if scenario.role == Role.WEREWOLF:
        teammate_seats = {
            player.seat
            for player in harness.state.players
            if player.role == Role.WEREWOLF and player.seat != actor.seat
        }
        assert {item["seat"] for item in observation["my_teammates"]} == teammate_seats
    else:
        assert observation["my_teammates"] == []
    assert legal.action == scenario.decision_action.value
    assert legal.target_required is True
    assert legal.can_skip is scenario.can_skip
    assert target_seat in legal.target_seats


@pytest.mark.parametrize("scenario", ROLE_SCENARIOS, ids=lambda item: item.key)
@pytest.mark.asyncio
async def test_classic_v1_role_capability_success_matrix(scenario: RoleScenario) -> None:
    harness = _build_matrix_harness()
    actor, target, request = await _execute_success(harness, scenario)

    _assert_request_observation_and_legal_scope(
        harness,
        scenario,
        actor,
        request,
        target.seat,
    )
    if scenario.key == "witch.save":
        assert request.private_context == {"killed_seat": target.seat}
        assert request.legal_actions[0].target_seats == [target.seat]
    elif scenario.key == "doctor.protect":
        assert request.private_context == {}
        assert set(request.legal_actions[0].target_seats) == {
            player.seat for player in harness.state.players
        }

    rows = _trace_rows(harness.orchestrator, request.request_id)
    assert [row.get("kind") for row in rows if row.get("kind")] == [
        "agent_request",
        "agent_response",
    ]
    response = next(row for row in rows if row.get("kind") == "agent_response")
    assert response["validation"]["valid"] is True
    assert any(row.get("type") == "decision_consumed" for row in rows)
    accepted_rules = [
        row
        for row in rows
        if row.get("type") == "rules_result"
        and row.get("rules", {}).get("status") == "accepted"
    ]
    assert any(
        row["rules"]["action"] == scenario.rule_action for row in accepted_rules
    )
    assert any(row.get("target_seat") == target.seat for row in accepted_rules)

    memory_kinds = {item.kind for item in actor.memory.observations}
    assert "role_assigned" in memory_kinds
    assert set(scenario.success_memory_kinds) <= memory_kinds
    assert not any(
        event.get("type") in {
            "agent_decision_failed",
            "decision_envelope_rejected",
            "action_rejected",
        }
        and event.get("request_id") == request.request_id
        for event in harness.events
    )

    if scenario.key == "werewolf.kill":
        assert target.alive is False
        assert target.death_reason == DeathReason.WOLF_KILL
    elif scenario.key in {"doctor.protect", "witch.save", "guard.protect"}:
        assert target.alive is True
        assert not harness.state.night_deaths
    elif scenario.key == "seer.inspect":
        result = next(
            event
            for event in harness.state.events
            if event.type == "seer_result" and event.recipients
        )
        actor_id = next(
            player.id for player in harness.state.players if player.seat == actor.seat
        )
        assert result.recipients == [actor_id]
        assert result.payload["target_seat"] == target.seat
        assert result.payload["team"] == "werewolves"
    elif scenario.key == "witch.poison":
        assert target.alive is False
        assert target.death_reason == DeathReason.POISONED
        assert harness.state.witch_poison is False
    elif scenario.key == "hunter.shoot":
        assert target.alive is False
        assert target.death_reason == DeathReason.HUNTER_SHOT
        assert not harness.state.pending_hunter
    elif scenario.key == "villager.vote":
        assert target.alive is False
        assert target.death_reason == DeathReason.EXILED


@pytest.mark.parametrize("scenario", ROLE_SCENARIOS, ids=lambda item: item.key)
@pytest.mark.asyncio
async def test_classic_v1_role_capability_rejection_matrix(scenario: RoleScenario) -> None:
    harness = _build_matrix_harness()
    actor, invalid_seat, request = await _execute_rejection(harness, scenario)

    observation = request.observation
    legal = request.legal_actions[0]
    assert observation["my_role"] == scenario.role.value
    assert observation["candidate_targets"] == legal.target_seats
    assert invalid_seat not in legal.target_seats
    assert legal.target_required is True

    rows = _trace_rows(harness.orchestrator, request.request_id)
    assert [row.get("kind") for row in rows if row.get("kind")] == [
        "agent_request",
        "agent_response",
    ]
    response = next(row for row in rows if row.get("kind") == "agent_response")
    assert response["validation"]["valid"] is False
    assert "target_seat_not_legal" in {
        issue["code"] for issue in response["validation"]["issues"]
    }
    assert response["envelope"]["decision"]["action"] == scenario.decision_action.value
    assert response["envelope"]["decision"]["target_seat"] == invalid_seat
    assert not any(row.get("kind") == "agent_response_failed" for row in rows)
    assert not any(row.get("type") == "decision_consumed" for row in rows)
    assert not any(row.get("type") == "rules_result" for row in rows)

    rejected = [
        event
        for event in harness.events
        if event.get("type") == "decision_envelope_rejected"
        and event.get("request_id") == request.request_id
    ]
    assert len(rejected) == 1
    assert rejected[0]["seat"] == actor.seat
    assert rejected[0]["action"] in {
        scenario.request_action,
        f"{scenario.role.value}_action",
        "werewolf_kill_proposal",
        "werewolf_final_kill_vote",
        "witch_save",
        "witch_poison",
    }
    assert not any(
        event.get("type") == "action_rejected"
        and event.get("request_id") == request.request_id
        for event in harness.events
    )

    memory_kinds = {item.kind for item in actor.memory.observations}
    assert "role_assigned" in memory_kinds
    retained_observations = {"witch_kill_preview"} if scenario.key == "witch.save" else set()
    for success_kind in (
        set(scenario.success_memory_kinds) - {"role_assigned"} - retained_observations
    ):
        if success_kind == "vote":
            # Accepted votes by other living seats are public observations and
            # may still be in this memory. The rejected actor's own vote must
            # never be recorded as accepted action evidence.
            assert not any(
                item.kind == "vote"
                and item.metadata.get("voter_seat") == actor.seat
                for item in actor.memory.observations
            )
        else:
            assert success_kind not in memory_kinds
    if scenario.key == "witch.save":
        assert "witch_kill_preview" in memory_kinds
        assert "witch_save_used" not in memory_kinds
        assert harness.state.witch_antidote is True
    if scenario.key == "hunter.shoot":
        shot = next(
            event for event in harness.events if event.get("type") == "hunter_shot"
        )
        assert shot["target_seat"] is None
        assert shot["resolution_reason"] == "decision_failed"
        assert "skip_reason" not in shot
    if scenario.key == "villager.vote":
        assert not any(
            event.get("type") == "vote_cast" and event.get("seat") == actor.seat
            for event in harness.events
        )


def test_matrix_executes_every_classic_v1_advertised_role() -> None:
    assert {scenario.role for scenario in ROLE_SCENARIOS} == set(
        CLASSIC_V1_IMPLEMENTED_ROLES
    )
    assert {scenario.key for scenario in ROLE_SCENARIOS if scenario.role == Role.WITCH} == {
        "witch.save",
        "witch.poison",
    }
    assert validate_role_deck(
        FULL_CAPABILITY_DECK,
        player_count=9,
        ruleset_id=CLASSIC_RULESET_ID,
    ) == FULL_CAPABILITY_DECK
