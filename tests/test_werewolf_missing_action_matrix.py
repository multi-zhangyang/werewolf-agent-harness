"""Executable G-006 matrix for missing Werewolf actions.

Every row below drives the production ``GameOrchestratorV2`` request path and
therefore the shared ``DecisionRuntime``.  The failing test actors intentionally
produce *no* ``DecisionEnvelope``: a deadline is cancelled by the runtime and a
provider fault is raised at the Agent boundary.  The assertions then inspect
the environment-owned resolution, rather than accepting a synthetic target,
speech, vote, or SKIP.

The doubles in this module are test-only.  Production code must never use a
scripted/replay Agent implementation.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
import random
from typing import Any

import pytest

from src.agent.memory import AgentMemory
from src.agent.schemas import AgentAction, Decision
from src.game.models import DeathReason, NightAction, NightActionType, Phase
from src.game.orchestrator import GameOrchestratorV2
from src.game.roles import CLASSIC_RULESET_ID, Role, validate_role_deck
from src.game.rules import RulesEngine
from src.game.state import new_game
from src.harness.agent_protocol import ActionRequest, DecisionEnvelope
from src.harness.decision_runtime import DecisionRuntime
from src.llm.router import LLMError


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


class FailureMode(StrEnum):
    TIMEOUT = "timeout"
    PROVIDER_FAILURE = "provider_failure"


@dataclass(frozen=True)
class MissingActionScenario:
    key: str
    role: Role
    request_action: str
    # The orchestrator may use a more specific public label for failure events.
    failure_action: str
    fail_all_role_actors: bool = False


SCENARIOS = (
    MissingActionScenario(
        key="night_kill",
        role=Role.WEREWOLF,
        request_action="night_kill",
        failure_action="werewolf_final_kill_vote",
        fail_all_role_actors=True,
    ),
    MissingActionScenario(
        key="see",
        role=Role.SEER,
        request_action="see",
        failure_action="seer_action",
    ),
    MissingActionScenario(
        key="guard",
        role=Role.GUARD,
        request_action="guard",
        failure_action="guard_action",
    ),
    MissingActionScenario(
        key="doctor_save",
        role=Role.DOCTOR,
        request_action="save",
        failure_action="doctor_action",
    ),
    MissingActionScenario(
        key="witch_save",
        role=Role.WITCH,
        request_action="save",
        failure_action="witch_save",
    ),
    MissingActionScenario(
        key="witch_poison",
        role=Role.WITCH,
        request_action="poison",
        failure_action="witch_poison",
    ),
    MissingActionScenario(
        key="hunter_shot",
        role=Role.HUNTER,
        request_action="hunter_shot",
        failure_action="hunter_shot",
    ),
    MissingActionScenario(
        key="speak",
        role=Role.VILLAGER,
        request_action="speak",
        failure_action="speak",
    ),
    MissingActionScenario(
        key="vote",
        role=Role.VILLAGER,
        request_action="vote",
        failure_action="vote",
    ),
    MissingActionScenario(
        key="last_words",
        role=Role.VILLAGER,
        request_action="last_words",
        failure_action="last_words",
    ),
)


class MissingActionActor:
    """Small test-only AgentProtocol implementation.

    ``fail`` actors never return an envelope for the configured action.  Other
    actors return the first environment-advertised legal intent so the tested
    action is exercised in the real orchestrator path without fallback logic.
    """

    def __init__(
        self,
        *,
        seat: int,
        name: str,
        role: Role,
        mode: FailureMode | None = None,
        fail_action: str | None = None,
    ) -> None:
        self.seat = seat
        self.name = name
        self.role = role
        self.mode = mode
        self.fail_action = fail_action
        self.persona_name = "g006-test"
        self.memory = AgentMemory(seat=seat, role=role.value)
        self.rng = random.Random(80_000 + seat)
        self.requests: list[ActionRequest] = []
        self.on_human_request = None

    async def decide(self, request: ActionRequest) -> DecisionEnvelope:
        self.requests.append(request)
        if self.mode is not None and request.action_kind == self.fail_action:
            if self.mode == FailureMode.TIMEOUT:
                # DecisionRuntime owns cancellation and records the terminal
                # failure.  This coroutine cooperates with that cancellation.
                await asyncio.sleep(30)
                raise AssertionError("a timed-out decision must be cancelled")
            fault = LLMError("opaque provider transport failure")
            raise fault

        legal = request.legal_actions[0]
        target = legal.target_seats[0] if legal.target_seats else None
        action = legal.action
        if action in {"night_kill", "kill"}:
            decision = Decision(action=AgentAction.NIGHT_KILL, target_seat=target)
        elif action == "wolf_council":
            decision = Decision(
                action=AgentAction.WOLF_COUNCIL,
                target_seat=target,
                team_message=f"seat {self.seat} council proposal",
            )
        elif action == "see":
            decision = Decision(action=AgentAction.SEE, target_seat=target)
        elif action == "save":
            if target is None and legal.can_skip:
                decision = Decision(action=AgentAction.SKIP, skip_reason="no_legal_target")
            else:
                decision = Decision(action=AgentAction.SAVE, target_seat=target)
        elif action == "poison":
            if target is None and legal.can_skip:
                decision = Decision(action=AgentAction.SKIP, skip_reason="no_legal_target")
            else:
                decision = Decision(action=AgentAction.POISON, target_seat=target)
        elif action == "guard":
            decision = Decision(action=AgentAction.GUARD, target_seat=target)
        elif action == "hunter_shot":
            if target is None and legal.can_skip:
                decision = Decision(action=AgentAction.SKIP, skip_reason="declined_shot")
            else:
                # The protocol intentionally maps the hunter action to the
                # common NIGHT_KILL intent; the environment owns the rule.
                decision = Decision(action=AgentAction.NIGHT_KILL, target_seat=target)
        elif action == "speak":
            decision = Decision(
                action=AgentAction.SPEAK,
                speech=f"test speech from seat {self.seat}",
                bid=1,
            )
        elif action == "vote":
            if target is None:
                decision = Decision(action=AgentAction.SKIP, skip_reason="no_vote_target")
            else:
                decision = Decision(action=AgentAction.VOTE, target_seat=target)
        elif action == "last_words":
            decision = Decision(
                action=AgentAction.LAST_WORDS,
                speech=f"test last words from seat {self.seat}",
            )
        else:  # pragma: no cover - request action space is fixed by the matrix
            raise AssertionError(f"unexpected test action {action!r}")
        return DecisionEnvelope(
            request_id=request.request_id,
            seat=self.seat,
            decision=decision,
            parse_status="not_applicable",
            metadata={"agent_kind": "test"},
        )

    def observe_event(self, day: int, phase: str, kind: str, text: str, **meta: Any) -> None:
        self.memory.observe(day, phase, kind, text, **meta)

    def record_claim(self, seat: int, day: int, claim: dict[str, Any]) -> None:
        self.memory.record_claim(seat, day, claim)


@dataclass
class MatrixHarness:
    orchestrator: GameOrchestratorV2
    events: list[dict[str, Any]]

    @property
    def state(self):
        return self.orchestrator.state


def _players(harness: MatrixHarness, role: Role):
    return [player for player in harness.state.players if Role(player.role) == role]


def _players_in_state(state: Any, role: Role):
    return [player for player in state.players if Role(player.role) == role]


def _player(harness: MatrixHarness, role: Role):
    return min(_players(harness, role), key=lambda item: item.seat)


def _build_harness(
    scenario: MissingActionScenario,
    mode: FailureMode,
) -> tuple[MatrixHarness, list[str]]:
    state = new_game([f"P{seat}" for seat in range(1, 10)], game_id=f"g006-{scenario.key}-{mode}")
    deck = validate_role_deck(
        FULL_CAPABILITY_DECK,
        player_count=9,
        ruleset_id=CLASSIC_RULESET_ID,
    )
    RulesEngine.deal_roles(state, deck=deck, seed=1_337, ruleset_id=CLASSIC_RULESET_ID)

    role_players = _players_in_state(state, scenario.role)
    failed_ids = (
        [p.id for p in role_players]
        if scenario.fail_all_role_actors
        else [min(role_players, key=lambda item: item.seat).id]
    )
    failed_id_set = set(failed_ids)
    actors: dict[str, MissingActionActor] = {}
    for player in state.players:
        role = Role(player.role)
        should_fail = player.id in failed_id_set
        actors[player.id] = MissingActionActor(
            seat=player.seat,
            name=player.name,
            role=role,
            mode=mode if should_fail else None,
            fail_action=scenario.request_action if should_fail else None,
        )

    events: list[dict[str, Any]] = []

    async def capture(payload: dict[str, Any]) -> None:
        events.append(dict(payload))

    orchestrator = GameOrchestratorV2(
        state=state,
        actors=actors,  # type: ignore[arg-type]
        deck=deck,
        rng=random.Random(2_026),
        on_event=capture,
        internal_events=True,
        # Keep the intended 30-second sleeper well beyond the deadline while
        # leaving enough scheduler headroom for nine concurrent immediate
        # actors under a full CI suite. Even 100 ms occasionally timed out an
        # unrelated voter on a loaded host; the intended actor sleeps 30 s.
        decision_timeout=0.5,
        phase_deadline=0,
        max_speak_rounds=1,
    )
    return MatrixHarness(orchestrator=orchestrator, events=events), failed_ids


def _actor(harness: MatrixHarness, player_id: str) -> MissingActionActor:
    return harness.orchestrator.actors[player_id]  # type: ignore[return-value]


def _request_rows(harness: MatrixHarness, request_id: str) -> list[dict[str, Any]]:
    return [
        row
        for row in harness.orchestrator._decision_trace
        if row.get("request_id") == request_id
        or row.get("request", {}).get("request_id") == request_id
    ]


def _failure_event(harness: MatrixHarness, request_id: str) -> dict[str, Any]:
    matches = [
        event
        for event in harness.events
        if event.get("request_id") == request_id
        and event.get("type") == "agent_decision_failed"
    ]
    assert len(matches) == 1, matches
    assert matches[0]["type"] == "agent_decision_failed"
    return matches[0]


def _assert_failed_request(
    harness: MatrixHarness,
    actor: MissingActionActor,
    *,
    request_action: str,
    failure_action: str,
    mode: FailureMode,
) -> ActionRequest:
    requests = [item for item in actor.requests if item.action_kind == request_action]
    assert len(requests) == 1
    request = requests[0]
    rows = _request_rows(harness, request.request_id)
    assert sum(row.get("kind") == "agent_request" for row in rows) == 1
    terminals = [row for row in rows if row.get("kind", "").startswith("agent_response")]
    assert [row["kind"] for row in terminals] == ["agent_response_failed"]
    assert "envelope" not in terminals[0]
    assert not any(
        row.get("type") == "decision_consumed" and row.get("request_id") == request.request_id
        for row in rows
    )
    assert not any(
        row.get("type") == "rules_result" and row.get("request_id") == request.request_id
        for row in rows
    )
    event = _failure_event(harness, request.request_id)
    assert event["request_id"] == request.request_id
    assert event["action"] == failure_action
    assert event["error_type"] == ("DecisionTimeout" if mode == FailureMode.TIMEOUT else "LLMError")
    if mode == FailureMode.TIMEOUT:
        assert event.get("timeout") is True
    # A failure event carries boundary facts only.  It cannot smuggle a
    # replacement target, public text, or explicit SKIP payload downstream.
    assert not any(field in event for field in ("target_seat", "speech", "skip_reason"))
    assert request.action_kind == request_action
    assert request.run_id == harness.state.id
    assert request.legal_actions[0].target_required is (request_action not in {"speak", "last_words"})
    return request


async def _flush_failures(harness: MatrixHarness) -> None:
    """Night helpers defer failure events until after night resolution."""
    for event in list(harness.orchestrator._failed_events):
        await harness.orchestrator._emit(event)
    harness.orchestrator._failed_events.clear()


def _submit_kill(harness: MatrixHarness, target_id: str) -> None:
    wolf = _player(harness, Role.WEREWOLF)
    RulesEngine.submit_night_action(
        harness.state,
        NightAction(actor_id=wolf.id, action=NightActionType.KILL, target_id=target_id),
    )


def _assert_no_consumed_for(harness: MatrixHarness, request_ids: set[str]) -> None:
    assert not any(
        row.get("type") == "decision_consumed" and row.get("request_id") in request_ids
        for row in harness.orchestrator._decision_trace
    )


@pytest.mark.parametrize("mode", list(FailureMode), ids=lambda item: item.value)
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item.key)
@pytest.mark.asyncio
async def test_missing_action_resolution_matrix(
    scenario: MissingActionScenario,
    mode: FailureMode,
) -> None:
    """Timeout/provider failure never fabricates an Agent action.

    The per-scenario assertions below state the explicit no-action rule owned
    by the environment.  They deliberately do not call ``Decision`` or
    ``SKIP`` on behalf of the failed actor.
    """
    harness, failed_ids = _build_harness(scenario, mode)
    orch = harness.orchestrator
    assert isinstance(orch._decision_runtime, DecisionRuntime)
    await orch._notify_role_assigned()

    if scenario.key == "night_kill":
        await orch._collect_werewolf_kill_proposals()
        request_ids = {
            item.request_id
            for pid in failed_ids
            for item in _actor(harness, pid).requests
            if item.action_kind == "night_kill"
        }
        assert len(request_ids) == len(failed_ids)
        RulesEngine.resolve_night(harness.state)
        await _flush_failures(harness)
        assert not harness.state.night_deaths
        assert harness.state.night_kill_target is None
        assert not any(event.get("type") == "night_action_submitted" for event in harness.events)
        for pid in failed_ids:
            _assert_failed_request(
                harness,
                _actor(harness, pid),
                request_action="night_kill",
                failure_action=scenario.failure_action,
                mode=mode,
            )
        _assert_no_consumed_for(harness, request_ids)

    elif scenario.key == "see":
        await orch._night_role_actions(Role.SEER, [NightActionType.SEE])
        actor = _actor(harness, failed_ids[0])
        RulesEngine.resolve_night(harness.state)
        await _flush_failures(harness)
        assert not any(event.type == "seer_result" for event in harness.state.events)
        assert not any(event.get("type") == "night_action_submitted" for event in harness.events)
        _assert_failed_request(
            harness,
            actor,
            request_action="see",
            failure_action=scenario.failure_action,
            mode=mode,
        )

    elif scenario.key in {"guard", "doctor_save", "witch_save"}:
        target = _player(harness, Role.VILLAGER)
        _submit_kill(harness, target.id)
        if scenario.key == "guard":
            await orch._night_role_actions(Role.GUARD, [NightActionType.GUARD])
            expected_action = "guard_action"
        elif scenario.key == "doctor_save":
            await orch._night_role_actions(Role.DOCTOR, [NightActionType.SAVE])
            expected_action = "doctor_action"
        else:
            await orch._witch_save_phase()
            expected_action = "witch_save"
        actor = _actor(harness, failed_ids[0])
        RulesEngine.resolve_night(harness.state)
        await _flush_failures(harness)
        assert target.alive is False
        assert target.death_reason == DeathReason.WOLF_KILL
        assert not any(
            event.get("type") == "night_action_submitted"
            and event.get("seat") == actor.seat
            for event in harness.events
        )
        if scenario.key == "guard":
            assert harness.state.last_guarded_seat is None
        if scenario.key == "witch_save":
            assert harness.state.witch_antidote is True
        _assert_failed_request(
            harness,
            actor,
            request_action="save" if scenario.key != "guard" else "guard",
            failure_action=expected_action,
            mode=mode,
        )

    elif scenario.key == "witch_poison":
        target = _player(harness, Role.VILLAGER)
        await orch._witch_poison_phase()
        actor = _actor(harness, failed_ids[0])
        RulesEngine.resolve_night(harness.state)
        await _flush_failures(harness)
        assert target.alive is True
        assert harness.state.witch_poison is True
        assert not harness.state.night_deaths
        assert not any(
            event.get("type") == "night_action_submitted" and event.get("seat") == actor.seat
            for event in harness.events
        )
        _assert_failed_request(harness, actor, request_action="poison", failure_action="witch_poison", mode=mode)

    elif scenario.key == "hunter_shot":
        hunter = _player(harness, Role.HUNTER)
        hunter.alive = False
        hunter.death_reason = DeathReason.EXILED
        harness.state.phase = Phase.DAY
        harness.state.pending_hunter = [hunter.id]
        await orch._process_deaths_and_hunter()
        actor = _actor(harness, failed_ids[0])
        _assert_failed_request(harness, actor, request_action="hunter_shot", failure_action="hunter_shot", mode=mode)
        assert harness.state.pending_hunter == []
        shots = [
            event
            for event in harness.events
            if event.get("type") == "hunter_shot" and event.get("seat") == hunter.seat
        ]
        assert len(shots) == 1
        assert shots[0]["target_seat"] is None
        assert shots[0]["resolution_reason"] == "decision_failed"
        assert len([p for p in harness.state.players if not p.alive]) == 1

    elif scenario.key == "speak":
        harness.state.phase = Phase.DAY
        await orch._run_day()
        actor = _actor(harness, failed_ids[0])
        _assert_failed_request(harness, actor, request_action="speak", failure_action="speak", mode=mode)
        assert not any(
            event.get("type") == "speech" and event.get("seat") == actor.seat
            for event in harness.events
        )
        assert not any(
            row.get("type") == "rules_result"
            and row.get("request_id") == actor.requests[-1].request_id
            and row.get("rules", {}).get("status") == "skipped"
            for row in orch._decision_trace
        )

    elif scenario.key == "vote":
        harness.state.phase = Phase.VOTING
        await orch._run_voting()
        actor = _actor(harness, failed_ids[0])
        request = _assert_failed_request(harness, actor, request_action="vote", failure_action="vote", mode=mode)
        assert not any(event.get("type") == "vote_cast" and event.get("seat") == actor.seat for event in harness.events)
        incomplete = [event for event in harness.events if event.get("type") == "vote_incomplete"]
        assert incomplete and incomplete[-1]["cast"] == len(harness.state.players) - 1
        resolved = [event for event in harness.events if event.get("type") == "vote_resolved"]
        assert resolved
        assert all(str(voter_id) != next(pid for pid in failed_ids) for voter_id in resolved[-1]["votes"])
        assert request.legal_actions[0].target_seats

    elif scenario.key == "last_words":
        victim = _player(harness, Role.VILLAGER)
        victim.alive = False
        victim.death_reason = DeathReason.EXILED
        harness.state.phase = Phase.DAY
        RulesEngine.queue_last_words(harness.state, victim.id, reason="exiled")
        await orch._process_last_words_queue()
        actor = _actor(harness, failed_ids[0])
        _assert_failed_request(harness, actor, request_action="last_words", failure_action="last_words", mode=mode)
        assert harness.state.last_words_queue == []
        assert not any(
            event.get("type") in {"last_words", "last_words_skipped"}
            and event.get("seat") == victim.seat
            for event in harness.events
        )

    else:  # pragma: no cover - SCENARIOS is exhaustive
        raise AssertionError(scenario.key)


@pytest.mark.parametrize("mode", list(FailureMode), ids=lambda item: item.value)
@pytest.mark.asyncio
async def test_partial_wolf_failure_uses_only_the_surviving_real_proposal(
    mode: FailureMode,
) -> None:
    """A missing wolf proposal is omitted, never replaced by a target.

    This supplements the all-wolves-absent rows above: plurality still has a
    valid environment input when one teammate answers, and the resulting death
    must be exactly that teammate's advertised target.
    """
    scenario = next(item for item in SCENARIOS if item.key == "night_kill")
    scenario = MissingActionScenario(
        key=scenario.key,
        role=scenario.role,
        request_action=scenario.request_action,
        failure_action=scenario.failure_action,
    )
    harness, failed_ids = _build_harness(scenario, mode)
    orch = harness.orchestrator
    await orch._notify_role_assigned()
    wolves = _players(harness, Role.WEREWOLF)
    assert len(wolves) == 2
    failed_id = failed_ids[0]
    surviving_id = next(player.id for player in wolves if player.id != failed_id)
    surviving_actor = _actor(harness, surviving_id)
    await orch._collect_werewolf_kill_proposals()
    surviving_request = next(
        item for item in surviving_actor.requests if item.action_kind == "night_kill"
    )
    expected_target_seat = surviving_request.legal_actions[0].target_seats[0]
    expected_target = next(
        player for player in harness.state.players if player.seat == expected_target_seat
    )
    RulesEngine.resolve_night(harness.state)
    await _flush_failures(harness)

    failed_actor = _actor(harness, failed_id)
    _assert_failed_request(
        harness,
        failed_actor,
        request_action="night_kill",
            failure_action="werewolf_final_kill_vote",
        mode=mode,
    )
    assert expected_target.alive is False
    assert expected_target.death_reason == DeathReason.WOLF_KILL
    assert [item["id"] for item in harness.state.night_deaths] == [expected_target.id]
    assert any(
        row.get("type") == "decision_consumed"
        and row.get("request_id") == surviving_request.request_id
        for row in orch._decision_trace
    )
