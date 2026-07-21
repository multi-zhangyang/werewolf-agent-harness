"""编排器集成测试 —— 用 Mock Actor 避免真实 LLM 调用。"""
from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

import pytest

import src.game.orchestrator as orchestrator_module
from src.agent.actor import AgentDecisionError
from src.agent.information import build_observation
from src.agent.memory import AgentMemory
from src.agent.prompts import last_words_instruction
from src.agent.schemas import AgentAction, Decision
from src.game.models import GameState, NightActionType, Phase
from src.game.orchestrator import GameOrchestratorV2
from src.game.roles import Role, Team
from src.game.rules import RulesEngine, RulesError
from src.game.state import new_game
from src.harness.agent_protocol import ActionRequest, DecisionEnvelope
from src.llm.models import ModelConfig


class MockActor:
    """返回预定决策的 Actor 桩。"""

    def __init__(self, seat: int, name: str, role: Role):
        self.seat = seat
        self.name = name
        self.role = role
        self.memory = AgentMemory(seat=seat, role=role.value)
        self.persona_name = "mock"
        self.calls: list[tuple[str, Any]] = []
        self.night_kwargs: list[dict[str, Any]] = []
        self.speak_kwargs: list[dict[str, Any]] = []
        self.vote_kwargs: list[dict[str, Any]] = []
        self.requests: list[ActionRequest] = []
        self._state_provider = None

    async def decide(self, request: ActionRequest) -> DecisionEnvelope:
        """Test-local AgentProtocol implementation.

        The legacy-shaped helpers below are deliberately confined to this test
        double so individual tests can replace one behavior without adding a
        production compatibility branch.
        """
        self.requests.append(request)
        if self._state_provider is None:
            raise RuntimeError("MockActor state provider is not configured")
        state = self._state_provider()
        player_id = next(player.id for player in state.players if player.seat == self.seat)
        observation = request.observation
        if request.action_kind == "speak":
            decision = await self.decide_speak(
                state,
                player_id,
                today_speeches=observation.get("today_speeches", []),
            )
        elif request.action_kind == "vote":
            legal_seats = request.legal_actions[0].target_seats if request.legal_actions else []
            pk_candidates = [
                player.id for player in state.players if player.seat in legal_seats
            ] if observation.get("in_pk") else None
            decision = await self.decide_vote(
                state,
                player_id,
                today_speeches=observation.get("today_speeches", []),
                pk_candidates=pk_candidates,
            )
        elif request.action_kind == "last_words":
            decision = await self.decide_last_words(
                state,
                player_id,
                str(request.private_context.get("reason") or ""),
            )
        else:
            decision = await self.decide_night_action(
                state,
                player_id,
                requested_action=request.action_kind,
                human_context=request.private_context,
            )
        return DecisionEnvelope(
            request_id=request.request_id,
            seat=self.seat,
            decision=decision,
            metadata={"agent_kind": "test"},
        )

    async def decide_night_action(self, state: GameState, player_id: str, **kw) -> Decision:
        self.calls.append(("night", player_id))
        self.night_kwargs.append(kw)
        # 狼人稳定刀第一个非狼村民; 预言家查验第一个狼; 其他跳过
        targets = {p.seat: p.id for p in state.living_players()}
        my_team = {p.id for p in state.players if p.role == Role.WEREWOLF}
        if self.role == Role.WEREWOLF:
            for seat, pid in sorted(targets.items()):
                if pid not in my_team:
                    return Decision(action=AgentAction.NIGHT_KILL, target_seat=seat)
        if self.role == Role.SEER:
            wolf = next((p for p in state.players if p.role == Role.WEREWOLF and p.alive), None)
            if wolf:
                return Decision(action=AgentAction.SEE, target_seat=wolf.seat)
        if self.role == Role.GUARD:
            legal = kw.get("human_context") or {}
            excluded = legal.get("last_guarded_seat")
            target = next((p for p in state.living_players() if p.seat != excluded), None)
            if target:
                return Decision(action=AgentAction.GUARD, target_seat=target.seat)
        return Decision(action=AgentAction.SKIP)

    async def decide_speak(self, state: GameState, player_id: str, **kw) -> Decision:
        self.calls.append(("speak", player_id))
        self.speak_kwargs.append(kw)
        return Decision(action=AgentAction.SPEAK, speech=f"我是{self.seat}号", bid=1)

    async def decide_vote(self, state: GameState, player_id: str, **kw) -> Decision:
        self.calls.append(("vote", player_id))
        self.vote_kwargs.append(kw)
        # 好人投第一个存活的狼; 狼人投第一个非狼
        allowed_ids = set(kw.get("pk_candidates") or [])
        candidates = [
            p for p in state.living_players()
            if p.id != player_id and (not allowed_ids or p.id in allowed_ids)
        ]
        wolves = {p.id for p in state.players if p.role == Role.WEREWOLF and p.alive}
        if self.role == Role.WEREWOLF:
            for p in candidates:
                if p.id not in wolves:
                    return Decision(
                        action=AgentAction.VOTE,
                        target_seat=p.seat,
                        reasoning="如果投好人则狼人收益更高。",
                    )
        for p in candidates:
            if p.id in wolves:
                return Decision(
                    action=AgentAction.VOTE,
                    target_seat=p.seat,
                    reasoning="如果投到狼人则好人收益更高。",
                )
        if candidates:
            return Decision(action=AgentAction.VOTE, target_seat=candidates[0].seat)
        return Decision(action=AgentAction.SKIP)

    async def decide_last_words(self, state: GameState, player_id: str, reason: str, **kw) -> Decision:
        return Decision(action=AgentAction.LAST_WORDS, speech="遗言")

    def observe_event(self, *args, **kw) -> None:
        if len(args) >= 4:
            day, phase, kind, text = args[:4]
            self.memory.observe(day, phase, kind, text, **kw)

    def record_claim(self, seat: int, day: int, claim: dict[str, Any]) -> None:
        self.memory.record_claim(seat, day, claim)

def _build_orchestrator(deck: list[Role] | None = None, **orch_kwargs):
    names = ["P1", "P2", "P3", "P4", "P5", "P6"]
    state = new_game(names)
    deck = deck or [Role.WEREWOLF, Role.WEREWOLF, Role.SEER,
                    Role.VILLAGER, Role.VILLAGER, Role.VILLAGER]
    RulesEngine.deal_roles(state, deck=deck, seed=1)
    actors = {p.id: MockActor(p.seat, p.name, p.role) for p in state.players}
    events: list[dict] = []
    max_speak_rounds = orch_kwargs.pop("max_speak_rounds", 1)
    orch = GameOrchestratorV2(
        state=state,
        actors=actors,
        deck=deck,
        on_event=lambda ev: events.append(ev),
        max_speak_rounds=max_speak_rounds,
        **orch_kwargs,
    )
    for actor in actors.values():
        actor._state_provider = lambda orch=orch: orch.state
    return orch, events


def _build_actor_binding_fixture() -> tuple[GameState, dict[str, MockActor]]:
    state = new_game(["P1", "P2", "P3", "P4", "P5", "P6"])
    RulesEngine.deal_roles(
        state,
        deck=[
            Role.WEREWOLF,
            Role.WEREWOLF,
            Role.SEER,
            Role.VILLAGER,
            Role.VILLAGER,
            Role.VILLAGER,
        ],
        seed=1,
    )
    actors = {
        player.id: MockActor(player.seat, player.name, Role(player.role))
        for player in state.players
    }
    return state, actors


@pytest.mark.parametrize("key_error", ["missing", "extra"])
def test_orchestrator_rejects_actor_keys_that_do_not_exactly_cover_players(key_error: str):
    state, actors = _build_actor_binding_fixture()
    if key_error == "missing":
        actors.pop(state.players[0].id)
    else:
        actors["not-a-player"] = MockActor(99, "extra", Role.VILLAGER)

    with pytest.raises(ValueError, match="exactly cover state players"):
        GameOrchestratorV2(state=state, actors=actors)


def test_orchestrator_rejects_one_actor_reused_for_multiple_players():
    state, actors = _build_actor_binding_fixture()
    first, second = state.players[:2]
    actors[second.id] = actors[first.id]

    with pytest.raises(ValueError, match="actor object must be unique"):
        GameOrchestratorV2(state=state, actors=actors)


def test_orchestrator_rejects_one_memory_reused_for_multiple_actors():
    state, actors = _build_actor_binding_fixture()
    first, second = state.players[:2]
    actors[second.id].memory = actors[first.id].memory

    with pytest.raises(ValueError, match="memory object must be unique"):
        GameOrchestratorV2(state=state, actors=actors)


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("actor_seat", "actor.seat"),
        ("actor_name", "actor.name"),
        ("actor_role", "actor.role"),
        ("memory_seat", "memory.seat"),
        ("memory_role", "memory.role"),
    ],
)
def test_orchestrator_rejects_actor_identity_field_mismatch(
    mismatch: str,
    message: str,
):
    state, actors = _build_actor_binding_fixture()
    player = state.players[0]
    actor = actors[player.id]
    if mismatch == "actor_seat":
        actor.seat = player.seat + 100
    elif mismatch == "actor_name":
        actor.name = f"{player.name}-wrong"
    elif mismatch == "actor_role":
        actor.role = (
            Role.VILLAGER if Role(player.role) != Role.VILLAGER else Role.WEREWOLF
        )
    elif mismatch == "memory_seat":
        actor.memory.seat = player.seat + 100
    elif mismatch == "memory_role":
        actor.memory.role = (
            Role.VILLAGER.value
            if Role(player.role) != Role.VILLAGER
            else Role.WEREWOLF.value
        )
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(f"unknown mismatch {mismatch}")

    with pytest.raises(ValueError, match=message):
        GameOrchestratorV2(state=state, actors=actors)


@pytest.mark.asyncio
async def test_request_rejects_actor_bound_to_a_different_player():
    orch, _events = _build_orchestrator()
    first, second = orch.state.players[:2]

    with pytest.raises(ValueError, match="not the actor bound to player"):
        await orch._request_agent_decision(
            orch.actors[first.id],
            second.id,
            action_kind="speak",
            phase="day",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["seat", "role"])
async def test_request_rejects_projected_observation_identity_mismatch(
    monkeypatch,
    mismatch: str,
):
    orch, _events = _build_orchestrator()
    player = orch.state.players[0]
    original_build_observation = orchestrator_module.build_observation

    def mismatched_observation(*args, **kwargs):
        observation = original_build_observation(*args, **kwargs)
        if mismatch == "seat":
            observation.my_seat = player.seat + 100
        else:
            observation.my_role = (
                Role.VILLAGER.value
                if Role(player.role) != Role.VILLAGER
                else Role.WEREWOLF.value
            )
        return observation

    monkeypatch.setattr(orchestrator_module, "build_observation", mismatched_observation)

    with pytest.raises(ValueError, match="observation identity mismatch"):
        await orch._request_agent_decision(
            orch.actors[player.id],
            player.id,
            action_kind="speak",
            phase="day",
        )


@pytest.mark.asyncio
async def test_candidate_order_uses_only_the_bound_actor_rng_and_preserves_legality():
    seed = 1729
    orch, _events = _build_orchestrator()
    player = next(player for player in orch.state.players if player.seat == 3)
    actor = orch.actors[player.id]
    actor.rng = random.Random(seed)
    legal_set = {
        candidate.seat
        for candidate in orch.state.living_players()
        if candidate.seat != actor.seat
    }
    oracle = random.Random(seed)
    expected_orders: list[list[int]] = []
    for _ in range(2):
        order = sorted(legal_set)
        oracle.shuffle(order)
        expected_orders.append(order)

    for _ in range(2):
        await orch._request_agent_decision(
            actor,
            player.id,
            action_kind="vote",
            phase="voting",
        )

    requests = [request for request in actor.requests if request.action_kind == "vote"][-2:]
    actual_orders = [request.legal_actions[0].target_seats for request in requests]
    assert actual_orders == expected_orders
    assert actual_orders[0] != sorted(legal_set)
    for request in requests:
        assert set(request.legal_actions[0].target_seats) == legal_set
        assert request.observation["candidate_targets"] == request.legal_actions[0].target_seats
        assert request.observation["vote_targets"] == request.legal_actions[0].target_seats


@pytest.mark.asyncio
async def test_another_actor_request_does_not_advance_this_actors_target_rng():
    async def target_order(*, interleave_other_actor: bool) -> list[int]:
        orch, _events = _build_orchestrator()
        chosen = next(player for player in orch.state.players if player.seat == 3)
        other = next(player for player in orch.state.players if player.seat == 4)
        orch.actors[chosen.id].rng = random.Random(1729)
        orch.actors[other.id].rng = random.Random(9876)
        if interleave_other_actor:
            await orch._request_agent_decision(
                orch.actors[other.id],
                other.id,
                action_kind="vote",
                phase="voting",
            )
        await orch._request_agent_decision(
            orch.actors[chosen.id],
            chosen.id,
            action_kind="vote",
            phase="voting",
        )
        return orch.actors[chosen.id].requests[-1].legal_actions[0].target_seats

    assert await target_order(interleave_other_actor=True) == await target_order(
        interleave_other_actor=False
    )


@pytest.mark.asyncio
async def test_orchestrator_rejects_unexecutable_phase_instead_of_busy_looping():
    orch, _events = _build_orchestrator()
    orch.state.phase = Phase.SETUP

    with pytest.raises(RuntimeError, match="cannot execute phase setup"):
        await orch.run()


@pytest.mark.asyncio
async def test_day_speech_timeout_is_transparent_failure_without_fake_speech():
    """单次发言墙钟超时应透明失败,不能广播迟到的假/兜底发言。"""
    # The injected actor sleeps for one second. Keep enough scheduler
    # headroom that immediate actors are not misclassified under full-suite
    # load while retaining a sharply bounded timeout test.
    orch, events = _build_orchestrator(decision_timeout=0.1)
    orch.state.phase = Phase.DAY
    pid, actor = next(iter(orch.actors.items()))

    async def slow_speak(*_args, **_kwargs):
        await asyncio.sleep(1)
        return Decision(action=AgentAction.SPEAK, speech="这句超时后不应出现", bid=5)

    actor.decide_speak = slow_speak  # type: ignore[method-assign]

    await orch._run_day()

    failed = [ev for ev in events if ev.get("type") == "agent_decision_failed"]
    assert any(ev.get("seat") == actor.seat and ev.get("phase") == "day" for ev in failed)
    assert any("timeout" in ev.get("reason", "") for ev in failed)
    assert not any(
        ev.get("type") == "speech"
        and ev.get("seat") == actor.seat
        and ev.get("text") == "这句超时后不应出现"
        for ev in events
    )
    assert ("speak", pid) not in actor.calls


@pytest.mark.asyncio
async def test_day_phase_deadline_marks_not_started_seats_without_fake_speech():
    """共享 phase deadline 耗尽后,未开始 seat 透明失败,不执行 actor 发言。"""
    orch, events = _build_orchestrator(
        decision_timeout=10,
        phase_deadlines={"day": 0.02},
        turn_policy="fixed_round_robin",
        max_consecutive_decision_failures=99,
    )
    orch.state.phase = Phase.DAY
    living = [pid for pid in orch.actors if orch.state.get_player(pid).alive]
    first_pid = living[0]
    first_actor = orch.actors[first_pid]
    started: list[int] = []

    async def slow_first_speak(*_args, **_kwargs):
        started.append(first_actor.seat)
        await asyncio.sleep(1)
        return Decision(action=AgentAction.SPEAK, speech="这句不应出现", bid=5)

    first_actor.decide_speak = slow_first_speak  # type: ignore[method-assign]

    await orch._run_day()

    day_failures = [
        ev for ev in events
        if ev.get("type") == "agent_decision_failed" and ev.get("phase") == "day"
    ]
    assert len(day_failures) == len(living)
    assert all(ev["action"] == "speak" for ev in day_failures)
    assert all(ev["error_type"] == "PhaseDeadlineExceeded" for ev in day_failures)
    assert any("during decision" in ev["reason"] for ev in day_failures)
    not_started = [ev for ev in day_failures if "before decision start" in ev["reason"]]
    assert len(not_started) == len(living) - 1
    assert started == [first_actor.seat]
    for pid in living[1:]:
        assert ("speak", pid) not in orch.actors[pid].calls
    assert not any(ev.get("type") == "speech" for ev in events)
    metrics = orch._decision_failure_metrics()
    assert metrics["timeout_count"] >= len(living)
    assert metrics["by_error_type"]["PhaseDeadlineExceeded"] == len(living)


@pytest.mark.asyncio
async def test_vote_timeout_emits_failure_and_does_not_submit_fake_vote():
    """并发投票超时只产生失败事件和不完整投票,不会替 agent 投票。"""
    # Voting is concurrent; sub-500 ms test budgets can time out an unrelated
    # immediate voter on a loaded full-suite host.
    orch, events = _build_orchestrator(decision_timeout=0.5)
    orch.state.phase = Phase.DAY
    pid, actor = next(iter(orch.actors.items()))

    async def slow_vote(*_args, **_kwargs):
        await asyncio.sleep(1)
        target = next(p for p in orch.state.living_players() if p.id != pid)
        return Decision(action=AgentAction.VOTE, target_seat=target.seat)

    actor.decide_vote = slow_vote  # type: ignore[method-assign]
    orch.state = RulesEngine.start_vote(orch.state)

    await orch._run_voting(today_speeches=[{"seat": 2, "text": "公开发言"}])

    failed = [ev for ev in events if ev.get("type") == "agent_decision_failed"]
    assert any(ev.get("seat") == actor.seat and ev.get("phase") == "voting" for ev in failed)
    assert any("timeout" in ev.get("reason", "") for ev in failed)
    assert not any(
        ev.get("type") == "vote_cast" and ev.get("seat") == actor.seat
        for ev in events
    )
    incomplete = [ev for ev in events if ev.get("type") == "vote_incomplete"]
    assert incomplete[-1]["cast"] == len(orch.state.players) - 1
    assert incomplete[-1]["needed"] == len(orch.state.players)
    metrics = orch._decision_failure_metrics()
    assert metrics["timeout_count"] >= 1
    assert metrics["by_phase"]["voting"] == 1


@pytest.mark.asyncio
async def test_accepted_public_votes_are_broadcast_to_each_living_memory():
    """每张通过规则校验的票都应成为所有在场座位的公开观察。"""
    orch, events = _build_orchestrator()
    orch.state.phase = Phase.DAY
    orch.state = RulesEngine.start_vote(orch.state)

    # Make the vote shape deterministic while keeping every target legal.
    for pid, actor in orch.actors.items():
        async def decide_vote(_state, _player_id, *, _pid=pid, **_kwargs):
            voter_seat = orch.state.get_player(_pid).seat
            target_seat = 2 if voter_seat != 2 else 1
            return Decision(action=AgentAction.VOTE, target_seat=target_seat)

        actor.decide_vote = decide_vote  # type: ignore[method-assign]

    await orch._run_voting(today_speeches=[])

    accepted = [event for event in events if event.get("type") == "vote_cast"]
    expected = {
        (int(event["seat"]), int(event["target_seat"]), False)
        for event in accepted
    }
    assert expected
    assert len(accepted) == len(orch.state.players)

    for actor in orch.actors.values():
        observed = [item for item in actor.memory.observations if item.kind == "vote"]
        actual = {
            (
                int(item.metadata["voter_seat"]),
                int(item.metadata["target_seat"]),
                bool(item.metadata["pk"]),
            )
            for item in observed
        }
        assert actual == expected
        assert {
            (
                int(item["voter_seat"]),
                int(item["target_seat"]),
                bool(item["pk"]),
            )
            for item in actor.memory.public_vote_ledger
        } == expected
        assert all(
            item.text == f"{item.metadata['voter_seat']}号投了{item.metadata['target_seat']}号"
            for item in observed
        )


@pytest.mark.asyncio
async def test_pk_public_vote_memory_marks_only_pk_round_votes():
    """PK 票保留结构化 pk 标记，方便下一轮区分普通票与重投。"""
    orch, _events = _build_orchestrator()
    orch.state.phase = Phase.DAY
    orch.state = RulesEngine.start_vote(orch.state)
    candidates = [player for player in orch.state.living_players() if player.seat in {1, 2}]
    candidate_ids = [player.id for player in candidates]
    orch.state.pk_candidates = list(candidate_ids)

    for pid, actor in orch.actors.items():
        async def decide_vote(_state, _player_id, *, _pid=pid, **_kwargs):
            voter_seat = orch.state.get_player(_pid).seat
            target_seat = 1 if voter_seat != 1 else 2
            return Decision(action=AgentAction.VOTE, target_seat=target_seat)

        actor.decide_vote = decide_vote  # type: ignore[method-assign]

    await orch._run_voting(pk_candidates=candidate_ids, today_speeches=[])

    for actor in orch.actors.values():
        observed = [item for item in actor.memory.observations if item.kind == "vote"]
        assert observed
        assert all(item.metadata.get("pk") is True for item in observed)
        assert all(item["pk"] is True for item in actor.memory.public_vote_ledger)


@pytest.mark.asyncio
async def test_rejected_protocol_vote_is_not_written_as_public_vote_memory():
    """非法 envelope 不应伪造自己的公开投票承诺。"""
    orch, _events = _build_orchestrator()
    orch.state.phase = Phase.DAY
    orch.state = RulesEngine.start_vote(orch.state)
    living = list(orch.actors)
    invalid_pid = living[-1]
    target_pid = living[0]

    for pid, actor in orch.actors.items():
        async def decide_vote(_state, _player_id, *, _pid=pid, **_kwargs):
            if _pid == invalid_pid:
                return Decision(action=AgentAction.VOTE, target_seat=999)
            target_seat = orch.state.get_player(target_pid).seat
            if _pid == target_pid:
                target_seat = orch.state.get_player(living[1]).seat
            return Decision(action=AgentAction.VOTE, target_seat=target_seat)

        actor.decide_vote = decide_vote  # type: ignore[method-assign]

    await orch._run_voting(today_speeches=[])

    invalid_actor_votes = [
        item
        for item in orch.actors[invalid_pid].memory.public_vote_ledger
        if item.get("voter_seat") == orch.actors[invalid_pid].seat
    ]
    assert invalid_actor_votes == []


def test_agent_decision_error_public_reason_is_sanitized():
    """Actor/provider 原始错误不应进入公开 agent_decision_failed.reason。"""
    orch, _events = _build_orchestrator()
    actor = next(iter(orch.actors.values()))
    err = AgentDecisionError("SENTINEL_RAW_PROVIDER_DETAIL with hidden response")

    event = orch._agent_decision_failure_event(
        actor,
        phase="day",
        action="speak",
        err=err,
    )

    assert event["type"] == "agent_decision_failed"
    assert event["error_type"] == "AgentDecisionError"
    assert event["reason"] == "AgentDecisionError during day/speak"
    assert "SENTINEL_RAW_PROVIDER_DETAIL" not in event["reason"]


@pytest.mark.asyncio
async def test_validator_failure_is_distinct_public_terminal_and_harness_attributed():
    orch, events = _build_orchestrator()
    actor = next(iter(orch.actors.values()))
    err = AgentDecisionError("private validator implementation detail")
    setattr(err, "error_type", "DecisionValidatorError")
    setattr(err, "request_id", "validator-request-1")

    event = orch._agent_decision_failure_event(
        actor,
        phase="voting",
        action="vote",
        err=err,
    )
    await orch._emit(event)

    assert event["type"] == "decision_validation_failed"
    assert event["error_type"] == "DecisionValidatorError"
    assert event["request_id"] == "validator-request-1"
    assert "private validator implementation detail" not in event["reason"]
    assert events[-1] == event
    failure = orch._decision_failure_metrics()["records"][-1]
    assert failure["terminal_kind"] == "validation_failure"
    assert failure["request_id"] == "validator-request-1"


def test_validator_failure_trace_is_counted_as_one_paired_terminal():
    orch, _events = _build_orchestrator()
    orch._append_trace({
        "kind": "agent_request",
        "request": {"request_id": "validator-request-1"},
    })
    orch._append_trace({
        "kind": "agent_response_validation_failed",
        "request_id": "validator-request-1",
    })

    metrics = orch._decision_trace_metrics()

    assert metrics["request_count"] == 1
    assert metrics["response_validation_failure_count"] == 1
    assert metrics["terminal_response_count"] == 1
    assert metrics["unpaired_request_count"] == 0
    assert metrics["duplicate_terminal_count"] == 0
    assert metrics["orphan_terminal_count"] == 0


def test_decision_trace_metrics_expose_tool_loop_cost_without_private_payloads():
    orch, _events = _build_orchestrator()
    orch._append_trace({
        "type": "model_generation",
        "request_id": "tool-request-1",
        "reasoning": "private model reasoning",
    })
    orch._append_trace({"type": "model_generation", "request_id": "tool-request-1"})
    orch._append_trace({"type": "model_generation_failed", "request_id": "tool-request-2"})
    orch._append_trace({
        "type": "agent_history_compacted",
        "request_id": "tool-request-1",
        "compacted_tool_groups": 3,
        "original_chars": 48_000,
        "model_chars": 19_000,
        "limit_satisfied": True,
        "private_summary": "must not appear in metrics",
    })
    # A soft history-window miss is still evidence even when there was no old
    # complete group that could be compacted. It must not inflate the actual
    # compaction counters.
    orch._append_trace({
        "type": "agent_history_compacted",
        "request_id": "tool-request-2",
        "compacted_tool_groups": 0,
        "original_chars": 25_000,
        "model_chars": 25_000,
        "limit_satisfied": False,
    })
    orch._append_trace({
        "type": "tool_call_requested",
        "request_id": "tool-request-1",
        "tool": "update_beliefs",
        "arguments": {"private": "must not appear in metrics"},
    })
    orch._append_trace({
        "type": "tool_result",
        "request_id": "tool-request-1",
        "tool": "update_beliefs",
        "ok": False,
        "terminal": False,
        "error": {
            "code": "invalid_cognition_update",
            "message": "private tool detail",
        },
    })
    orch._append_trace({
        "type": "tool_call_requested",
        "request_id": "tool-request-2",
        "tool": "vote",
    })
    orch._append_trace({
        "type": "tool_result",
        "request_id": "tool-request-2",
        "tool": "vote",
        "ok": True,
        "terminal": True,
    })

    metrics = orch._decision_trace_metrics()

    assert metrics["model_generation_count"] == 2
    assert metrics["model_generation_failure_count"] == 1
    assert metrics["tool_call_count"] == metrics["tool_result_count"] == 2
    assert metrics["tool_success_count"] == metrics["tool_failure_count"] == 1
    assert metrics["tool_failure_by_code"] == {"invalid_cognition_update": 1}
    assert metrics["tool_failure_by_tool"] == {"update_beliefs": 1}
    assert metrics["requests_with_tool_failures"] == 1
    assert metrics["terminal_tool_result_count"] == 1
    assert metrics["terminal_tool_failure_count"] == 0
    assert metrics["max_model_generations_per_request"] == 2
    assert metrics["max_tool_calls_per_request"] == 1
    assert metrics["history_compaction_count"] == 1
    assert metrics["requests_with_history_compaction"] == 1
    assert metrics["max_compacted_tool_groups"] == 3
    assert metrics["max_history_chars_before_compaction"] == 48_000
    assert metrics["max_model_history_chars_after_compaction"] == 19_000
    assert metrics["history_compaction_limit_unsatisfied_count"] == 1
    assert metrics["max_unsatisfied_model_history_chars"] == 25_000
    serialized = str(metrics)
    assert "private model reasoning" not in serialized
    assert "private tool detail" not in serialized
    assert "must not appear in metrics" not in serialized


def test_decision_trace_metrics_roll_up_unique_finished_turns_and_seat_fairness():
    orch, _events = _build_orchestrator()

    def request(request_id: str, seat: int) -> dict[str, object]:
        return {
            "kind": "agent_request",
            "request": {"request_id": request_id, "seat": seat},
        }

    def finished(
        request_id: str,
        seat: int,
        *,
        generation_attempts: int,
        model_generations: int,
        generation_failures: int,
        response_retries: int,
        tool_calls: int,
        tool_successes: int,
        tool_failures: int,
        model_latency_seconds: float,
        tool_latency_seconds: float,
        total_tokens: int,
        token_usage_complete: bool,
        elapsed_seconds: float = 0.0,
        budget_exhausted: str | None = None,
    ) -> dict[str, object]:
        return {
            "type": "agent_turn_finished",
            "request_id": request_id,
            "seat": seat,
            "private_reasoning": "SENTINEL_FINISHED_PRIVATE_CONTENT",
            "telemetry": {
                "request_id": request_id,
                "seat": seat,
                "generation_attempts": generation_attempts,
                "model_generations": model_generations,
                "generation_failures": generation_failures,
                "response_retries": response_retries,
                "tool_calls": tool_calls,
                "tool_successes": tool_successes,
                "tool_failures": tool_failures,
                "model_latency_seconds": model_latency_seconds,
                "tool_latency_seconds": tool_latency_seconds,
                "elapsed_seconds": elapsed_seconds,
                "total_tokens": total_tokens,
                "token_usage_complete": token_usage_complete,
                "budget_exhausted": budget_exhausted,
                "usage": {"private_provider_field": "SENTINEL_PRIVATE_USAGE"},
                "limits": {"private_policy": "SENTINEL_PRIVATE_LIMIT"},
            },
        }

    for row in (
        request("r1", 1),
        request("r2", 2),
        request("duplicate", 2),
        request("missing", 3),
        request("identity-mismatch", 3),
        finished(
            "r1",
            1,
            generation_attempts=2,
            model_generations=1,
            generation_failures=1,
            response_retries=1,
            tool_calls=2,
            tool_successes=1,
            tool_failures=1,
            model_latency_seconds=1.25,
            tool_latency_seconds=0.1,
            total_tokens=10,
            token_usage_complete=True,
            budget_exhausted="max_model_generations",
        ),
        finished(
            "r2",
            2,
            generation_attempts=1,
            model_generations=1,
            generation_failures=0,
            response_retries=0,
            tool_calls=1,
            tool_successes=1,
            tool_failures=0,
            model_latency_seconds=0.75,
            tool_latency_seconds=0.05,
            total_tokens=7,
            token_usage_complete=False,
        ),
        # Neither copy contributes to costs: a duplicate closure is ambiguous.
        finished(
            "duplicate",
            2,
            generation_attempts=99,
            model_generations=99,
            generation_failures=0,
            response_retries=0,
            tool_calls=99,
            tool_successes=99,
            tool_failures=0,
            model_latency_seconds=99.0,
            tool_latency_seconds=99.0,
            total_tokens=99_999,
            token_usage_complete=True,
        ),
        finished(
            "duplicate",
            2,
            generation_attempts=99,
            model_generations=99,
            generation_failures=0,
            response_retries=0,
            tool_calls=99,
            tool_successes=99,
            tool_failures=0,
            model_latency_seconds=99.0,
            tool_latency_seconds=99.0,
            total_tokens=99_999,
            token_usage_complete=True,
        ),
        # This row has no originating ActionRequest and cannot affect a seat.
        finished(
            "orphan",
            1,
            generation_attempts=88,
            model_generations=88,
            generation_failures=0,
            response_retries=0,
            tool_calls=88,
            tool_successes=88,
            tool_failures=0,
            model_latency_seconds=88.0,
            tool_latency_seconds=88.0,
            total_tokens=88_888,
            token_usage_complete=True,
        ),
        # The request is for seat 3, so a top-level seat 2 closure is rejected.
        finished(
            "identity-mismatch",
            2,
            generation_attempts=1,
            model_generations=1,
            generation_failures=0,
            response_retries=0,
            tool_calls=1,
            tool_successes=1,
            tool_failures=0,
            model_latency_seconds=1.0,
            tool_latency_seconds=1.0,
            total_tokens=100,
            token_usage_complete=True,
        ),
    ):
        orch._append_trace(row)  # type: ignore[arg-type]

    metrics = orch._decision_trace_metrics()

    assert metrics["agent_turn_finished_count"] == 6
    assert metrics["unique_agent_turn_finished_count"] == 5
    assert metrics["duplicate_agent_turn_finished_count"] == 1
    assert metrics["orphan_agent_turn_finished_count"] == 1
    assert metrics["requests_with_agent_turn_finished"] == 4
    assert metrics["requests_without_agent_turn_finished"] == 1
    assert metrics["ambiguous_agent_turn_finished_request_count"] == 1
    assert metrics["agent_turn_telemetry_request_count"] == 2
    assert metrics["invalid_agent_turn_telemetry_count"] == 1
    assert metrics["agent_turn_telemetry_identity_mismatch_count"] == 1

    assert metrics["agent_turn_generation_attempts"] == 3
    assert metrics["agent_turn_model_generations"] == 2
    assert metrics["agent_turn_generation_failures"] == 1
    assert metrics["agent_turn_response_retries"] == 1
    assert metrics["agent_turn_tool_calls"] == 3
    assert metrics["agent_turn_tool_successes"] == 2
    assert metrics["agent_turn_tool_failures"] == 1
    assert metrics["agent_turn_model_latency_seconds"] == 2.0
    assert metrics["agent_turn_tool_latency_seconds"] == 0.15
    assert metrics["agent_turn_elapsed_seconds"] == 0.0
    assert metrics["agent_turn_total_tokens"] == 17
    assert metrics["agent_turn_token_usage_complete_count"] == 1
    assert metrics["agent_turn_token_usage_incomplete_count"] == 1
    assert metrics["agent_turn_token_usage_unavailable_count"] == 3
    assert metrics["agent_turn_token_usage_complete"] is False
    assert metrics["agent_turn_budget_failure_count"] == 1
    assert metrics["agent_turn_budget_failure_by_code"] == {
        "max_model_generations": 1,
    }

    assert metrics["max_agent_turn_generation_attempts_per_request"] == 2
    assert metrics["max_agent_turn_model_generations_per_request"] == 1
    assert metrics["max_agent_turn_response_retries_per_request"] == 1
    assert metrics["max_agent_turn_tool_calls_per_request"] == 2
    assert metrics["max_agent_turn_total_tokens_per_request"] == 10
    # Existing event-level metrics remain separate and retain their semantics.
    assert metrics["model_generation_count"] == 0
    assert metrics["tool_call_count"] == 0

    by_seat = {row["seat"]: row for row in metrics["agent_turn_by_seat"]}
    assert by_seat[1]["generation_attempts"] == 2
    assert by_seat[1]["budget_failure_by_code"] == {"max_model_generations": 1}
    assert by_seat[2]["request_count"] == 2
    assert by_seat[2]["finished_request_count"] == 2
    assert by_seat[2]["duplicate_finished_count"] == 1
    assert by_seat[2]["telemetry_request_count"] == 1
    assert by_seat[3]["request_count"] == 2
    assert by_seat[3]["missing_finished_count"] == 1
    assert by_seat[3]["invalid_telemetry_count"] == 1
    assert by_seat[3]["telemetry_request_count"] == 0

    fairness = metrics["agent_turn_seat_fairness_facts"]
    assert fairness["request_count"] == {
        "minimum": 1,
        "maximum": 2,
        "spread": 1,
        "max_to_min_ratio": 2.0,
        "minimum_seats": [1],
        "maximum_seats": [2, 3],
    }
    assert fairness["generation_attempts_per_request"]["maximum_seats"] == [1]
    assert fairness["generation_attempts_per_request"]["minimum_seats"] == [2]

    serialized = str(metrics)
    assert "SENTINEL_FINISHED_PRIVATE_CONTENT" not in serialized
    assert "SENTINEL_PRIVATE_USAGE" not in serialized
    assert "SENTINEL_PRIVATE_LIMIT" not in serialized


def test_agent_turn_rollup_rejects_inconsistent_counters_and_unknown_budget_code():
    orch, _events = _build_orchestrator()
    secret_code = "sk-unknown-budget-code-123456789"
    orch._append_trace({
        "kind": "agent_request",
        "request": {"request_id": "bad-telemetry", "seat": 1},
    })
    orch._append_trace({
        "type": "agent_turn_finished",
        "request_id": "bad-telemetry",
        "seat": 1,
        "telemetry": {
            "request_id": "bad-telemetry",
            "seat": 1,
            "generation_attempts": 2,
            "model_generations": 1,
            "generation_failures": 0,
            "response_retries": 0,
            "tool_calls": 1,
            "tool_successes": 0,
            "tool_failures": 0,
            "model_latency_seconds": 1.0,
            "tool_latency_seconds": 0.1,
            "elapsed_seconds": 1.2,
            "total_tokens": 20,
            "token_usage_complete": True,
            "budget_exhausted": secret_code,
        },
    })

    metrics = orch._decision_trace_metrics()

    assert metrics["invalid_agent_turn_telemetry_count"] == 1
    assert metrics["agent_turn_telemetry_request_count"] == 0
    assert metrics["agent_turn_generation_attempts"] == 0
    assert metrics["agent_turn_total_tokens"] == 0
    assert metrics["agent_turn_token_usage_unavailable_count"] == 1
    assert metrics["agent_turn_token_usage_complete"] is False
    assert metrics["agent_turn_budget_failure_by_code"] == {}
    assert secret_code not in str(metrics)


def test_consumed_decision_trace_records_bounded_private_belief_checkpoint():
    orch, _events = _build_orchestrator()
    actor = next(iter(orch.actors.values()))

    class PrivateStateStub:
        @staticmethod
        def snapshot():
            return {
                "owner_seat": actor.seat,
                "owner_role": "werewolf",
                "revision": 4,
                "selected_plan": "must stay private",
                "deception_plan": "must stay private",
                "beliefs": {
                    "2": {
                        "wolf_probability": 0.75,
                        "likely_role": "werewolf",
                        "confidence": 0.8,
                        "evidence": ["free-form evidence must stay private"],
                        "updated_day": 2,
                        "updated_phase": "voting",
                    },
                },
            }

    actor.private_state = PrivateStateStub()
    envelope = DecisionEnvelope(
        request_id="belief-checkpoint-request",
        seat=actor.seat,
        decision=Decision(
            action=AgentAction.VOTE,
            target_seat=2,
            reasoning="admin reasoning remains on the response trace",
        ),
    )

    orch._record_decision_trace(actor, envelope, phase="voting")

    row = orch._decision_trace[-1]
    checkpoint = row["belief_state_after"]
    assert checkpoint == {
        "schema_version": "werewolf.agent-belief-trace.v1",
        "owner_seat": actor.seat,
        "revision": 4,
        "beliefs": {
            "2": {
                "wolf_probability": 0.75,
                "likely_role": "werewolf",
                "confidence": 0.8,
                "updated_day": 2,
                "updated_phase": "voting",
            },
        },
    }
    serialized = json.dumps(checkpoint, ensure_ascii=False)
    assert "selected_plan" not in serialized
    assert "deception_plan" not in serialized
    assert "free-form evidence" not in serialized


@pytest.mark.asyncio
async def test_hunter_decision_failure_is_emitted_and_not_silently_swallowed():
    orch, events = _build_orchestrator(
        deck=[Role.HUNTER, Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.VILLAGER, Role.VILLAGER]
    )
    hunter = next(p for p in orch.state.players if p.role == Role.HUNTER)
    orch.state.pending_hunter = [hunter.id]
    actor = orch.actors[hunter.id]

    async def fail_hunter_decision(*_args, **_kwargs):
        raise AgentDecisionError("real hunter call exhausted")

    actor.decide_night_action = fail_hunter_decision  # type: ignore[method-assign]

    await orch._process_deaths_and_hunter()

    assert orch.state.pending_hunter == []
    failed = [ev for ev in events if ev.get("type") == "agent_decision_failed"]
    assert failed[-1]["seat"] == actor.seat
    assert failed[-1]["action"] == "hunter_shot"
    shots = [ev for ev in events if ev.get("type") == "hunter_shot"]
    assert shots[-1]["seat"] == actor.seat
    assert shots[-1]["target_seat"] is None
    assert shots[-1]["resolution_reason"] == "decision_failed"
    assert "skip_reason" not in shots[-1]
    assert shots[-1]["request_id"] == failed[-1]["request_id"]


@pytest.mark.asyncio
async def test_hunter_invalid_target_is_rejected_at_protocol_boundary_without_failing_game():
    orch, events = _build_orchestrator(
        deck=[Role.HUNTER, Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.VILLAGER, Role.VILLAGER]
    )
    hunter = next(p for p in orch.state.players if p.role == Role.HUNTER)
    dead_target = next(p for p in orch.state.players if p.id != hunter.id)
    dead_target.alive = False
    orch.state.pending_hunter = [hunter.id]
    actor = orch.actors[hunter.id]

    async def shoot_dead_target(*_args, **_kwargs):
        return Decision(action=AgentAction.NIGHT_KILL, target_seat=dead_target.seat)

    actor.decide_night_action = shoot_dead_target  # type: ignore[method-assign]

    await orch._process_deaths_and_hunter()

    assert orch.state.pending_hunter == []
    assert dead_target.death_reason is None
    rejected = [ev for ev in events if ev.get("type") == "decision_envelope_rejected"]
    assert rejected[-1]["seat"] == actor.seat
    assert rejected[-1]["phase"] == "hunter"
    assert rejected[-1]["action"] == "hunter_shot"
    shots = [ev for ev in events if ev.get("type") == "hunter_shot"]
    assert shots[-1]["seat"] == actor.seat
    assert shots[-1]["target_seat"] is None
    assert shots[-1]["resolution_reason"] == "decision_failed"
    assert "skip_reason" not in shots[-1]
    assert shots[-1]["request_id"] == rejected[-1]["request_id"]
    response = next(
        item for item in reversed(orch._decision_trace)
        if item.get("kind") == "agent_response" and item.get("seat") == actor.seat
    )
    assert response["validation"]["valid"] is False
    assert response["validation"]["issues"][0]["code"] == "target_seat_not_legal"


@pytest.mark.asyncio
async def test_night_role_action_plain_exception_emits_failure_without_action():
    """普通运行时异常也必须透明审计,不能伪装成无行动。"""
    orch, events = _build_orchestrator()
    seer_player = next(p for p in orch.state.players if p.role == Role.SEER)
    actor = orch.actors[seer_player.id]

    async def fail_night_action(*_args, **_kwargs):
        raise RuntimeError("provider exploded with private detail")

    actor.decide_night_action = fail_night_action  # type: ignore[method-assign]

    await orch._night_role_actions(Role.SEER, [])
    for ev in orch._failed_events:
        await orch._emit(ev)
    orch._failed_events.clear()

    failed = [ev for ev in events if ev.get("type") == "agent_decision_failed"]
    assert failed[-1]["seat"] == actor.seat
    assert failed[-1]["phase"] == "night"
    assert failed[-1]["action"] == "seer_action"
    assert failed[-1]["error_type"] == "RuntimeError"
    assert "private detail" not in failed[-1]["reason"]
    assert not orch.state.night_actions
    metrics = orch._decision_failure_metrics()
    assert metrics["by_action"]["seer_action"] == 1
    assert metrics["by_error_type"]["RuntimeError"] == 1


@pytest.mark.asyncio
async def test_rules_rejection_is_not_mislabeled_as_agent_response_failure(monkeypatch):
    orch, events = _build_orchestrator(internal_events=True)
    seer = next(player for player in orch.state.players if player.role == Role.SEER)

    def reject_valid_action(*_args, **_kwargs):
        raise RulesError("controlled rules rejection")

    monkeypatch.setattr(RulesEngine, "submit_night_action", reject_valid_action)

    await orch._night_role_actions(Role.SEER, [NightActionType.SEE])
    for event in orch._failed_events:
        await orch._emit(event)
    orch._failed_events.clear()

    assert not any(event.get("type") == "agent_decision_failed" for event in events)
    assert not any(event.get("type") == "decision_envelope_rejected" for event in events)
    rejected = [event for event in events if event.get("type") == "action_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["seat"] == seer.seat
    assert rejected[0]["reason_code"] == "rules_rejected"
    assert rejected[0]["visibility"] == "private"
    assert rejected[0]["recipients"] == [seer.id]
    response = next(row for row in orch._decision_trace if row.get("kind") == "agent_response")
    consumed = next(row for row in orch._decision_trace if row.get("type") == "decision_consumed")
    rules = next(row for row in orch._decision_trace if row.get("type") == "rules_result")
    assert response["validation"]["valid"] is True
    assert response["request_id"] == rejected[0]["request_id"]
    assert consumed["request_id"] == response["request_id"]
    assert rules["request_id"] == response["request_id"]
    assert rules["rules"]["status"] == "rejected"
    assert orch._decision_failure_metrics()["failure_count"] == 0

    # A RulesEngine rejection is an environment outcome, not a fabricated
    # action.  Persist only bounded, seat-private feedback so the next request
    # can recover without exposing the engine's raw exception to other seats.
    rejected_state_events = [
        event for event in orch.state.events if event.type == "action_rejected"
    ]
    assert len(rejected_state_events) == 1
    rejected_state_event = rejected_state_events[0]
    assert rejected_state_event.visibility == "private"
    assert rejected_state_event.recipients == [seer.id]
    assert rejected_state_event.payload == {
        "request_id": response["request_id"],
        "request_phase": "night",
        "action": "see",
        "reason_code": "rules_rejected",
        "committed": False,
    }
    assert "controlled rules rejection" not in rejected_state_event.model_dump_json()
    assert not orch.state.night_actions

    seer_observation = build_observation(orch.state, seer.id)
    assert any(
        event.get("type") == "action_rejected"
        and event.get("payload", {}).get("request_id") == response["request_id"]
        for event in seer_observation.private_events
    )
    other = next(player for player in orch.state.players if player.id != seer.id)
    other_observation = build_observation(orch.state, other.id)
    assert not any(
        event.get("type") == "action_rejected"
        and event.get("payload", {}).get("request_id") == response["request_id"]
        for event in other_observation.private_events
    )
    assert not any(
        event.type == "action_rejected"
        for event in orch.state.events
        if event.visibility == "public"
    )

    # The feedback survives into a subsequent ActionRequest for the same seat.
    await orch._request_agent_decision(
        orch.actors[seer.id],
        seer.id,
        action_kind="see",
        phase="night",
    )
    next_request = orch.actors[seer.id].requests[-1]
    assert any(
        event.get("type") == "action_rejected"
        and event.get("payload", {}).get("committed") is False
        for event in next_request.observation["private_events"]
    )


@pytest.mark.asyncio
async def test_wolf_council_rules_rejection_is_private_before_final_vote(monkeypatch):
    """A rejected team message is visible only to its sender's next request."""
    orch, _events = _build_orchestrator(internal_events=True)
    wolves = sorted(
        (player for player in orch.state.players if player.role == Role.WEREWOLF),
        key=lambda player: player.seat,
    )
    target = next(player for player in orch.state.players if player.role != Role.WEREWOLF)

    for wolf in wolves:
        actor = orch.actors[wolf.id]

        async def council_then_vote(state, player_id, *, _actor=actor, **kwargs):
            requested = kwargs.get("requested_action")
            if requested == "wolf_council":
                return Decision(
                    action=AgentAction.WOLF_COUNCIL,
                    target_seat=target.seat,
                    team_message=f"建议考虑{target.seat}号",
                )
            return Decision(action=AgentAction.NIGHT_KILL, target_seat=target.seat)

        actor.decide_night_action = council_then_vote  # type: ignore[method-assign]

    def reject_council(*_args, **_kwargs):
        raise RulesError("raw council rejection detail")

    monkeypatch.setattr(RulesEngine, "record_wolf_council_message", reject_council)

    await orch._collect_werewolf_kill_proposals()

    rejection_events = [
        event for event in orch.state.events if event.type == "action_rejected"
    ]
    assert len(rejection_events) == len(wolves)
    assert all(event.visibility == "private" for event in rejection_events)
    assert {recipient for event in rejection_events for recipient in event.recipients} == {
        wolf.id for wolf in wolves
    }
    assert all("raw council rejection detail" not in event.model_dump_json() for event in rejection_events)

    for wolf in wolves:
        council_request = next(
            request for request in orch.actors[wolf.id].requests
            if request.action_kind == "wolf_council"
        )
        final_request = next(
            request for request in orch.actors[wolf.id].requests
            if request.action_kind == "night_kill"
        )
        own_feedback = [
            event for event in final_request.observation["private_events"]
            if event.get("type") == "action_rejected"
        ]
        assert len(own_feedback) == 1
        assert own_feedback[0]["payload"]["request_id"] == council_request.request_id
        assert own_feedback[0]["payload"]["committed"] is False

    assert not any(event.type == "wolf_council_message" for event in orch.state.events)


@pytest.mark.asyncio
async def test_unexpected_rules_runtime_bug_is_not_swallowed(monkeypatch):
    orch, _events = _build_orchestrator(internal_events=True)

    def explode(*_args, **_kwargs):
        raise RuntimeError("unexpected engine bug")

    monkeypatch.setattr(RulesEngine, "submit_night_action", explode)

    with pytest.raises(RuntimeError, match="unexpected engine bug"):
        await orch._night_role_actions(Role.SEER, [NightActionType.SEE])


@pytest.mark.asyncio
async def test_night_protocol_passes_requested_action_names_to_actors():
    """编排器必须告诉 actor 当前请求的真实 action 名,前端不能暗猜接口。"""
    orch, _events = _build_orchestrator(
        deck=[
            Role.WEREWOLF,
            Role.WEREWOLF,
            Role.SEER,
            Role.GUARD,
            Role.WITCH,
            Role.VILLAGER,
        ]
    )
    seer = next(p for p in orch.state.players if p.role == Role.SEER)
    guard = next(p for p in orch.state.players if p.role == Role.GUARD)
    witch = next(p for p in orch.state.players if p.role == Role.WITCH)
    wolves = [p for p in orch.state.players if p.role == Role.WEREWOLF]
    first_good = next(p for p in orch.state.living_players() if p.role != Role.WEREWOLF)

    await orch._night_role_actions(Role.SEER, [NightActionType.SEE])
    await orch._night_role_actions(Role.GUARD, [NightActionType.GUARD])
    await orch._collect_werewolf_kill_proposals()
    await orch._witch_save_phase()
    await orch._witch_poison_phase()

    assert orch.actors[seer.id].night_kwargs[-1]["requested_action"] == "see"
    assert orch.actors[guard.id].night_kwargs[-1]["requested_action"] == "guard"
    assert [orch.actors[p.id].night_kwargs[-1]["requested_action"] for p in wolves] == [
        "night_kill",
        "night_kill",
    ]
    assert orch.actors[witch.id].night_kwargs[-2]["requested_action"] == "save"
    assert orch.actors[witch.id].night_kwargs[-2]["human_context"] == {"killed_seat": first_good.seat}
    assert orch.actors[witch.id].night_kwargs[-1]["requested_action"] == "poison"


@pytest.mark.asyncio
async def test_seeded_wolf_tie_break_is_independent_of_actor_registration_order():
    async def chosen_target_seat(*, reverse_actors: bool) -> int:
        orch, _events = _build_orchestrator(rng=random.Random(97), internal_events=True)
        wolves = sorted(
            (player for player in orch.state.players if player.role == Role.WEREWOLF),
            key=lambda player: player.seat,
        )
        targets = sorted(
            (player for player in orch.state.players if player.role != Role.WEREWOLF),
            key=lambda player: player.seat,
        )[:2]

        for wolf, target in zip(wolves, targets, strict=True):
            actor = orch.actors[wolf.id]

            async def propose_target(*_args, target_seat=target.seat, **_kwargs):
                return Decision(action=AgentAction.NIGHT_KILL, target_seat=target_seat)

            actor.decide_night_action = propose_target  # type: ignore[method-assign]

        if reverse_actors:
            orch.actors = dict(reversed(list(orch.actors.items())))

        await orch._collect_werewolf_kill_proposals()
        assert len(orch.state.night_actions) == 1
        return orch.state.get_player(orch.state.night_actions[0].target_id).seat

    forward = await chosen_target_seat(reverse_actors=False)
    reversed_order = await chosen_target_seat(reverse_actors=True)

    assert forward == reversed_order


@pytest.mark.asyncio
async def test_custom_doctor_deck_executes_save_through_protocol_rules_trace_and_memory():
    deck = [
        Role.WEREWOLF,
        Role.WEREWOLF,
        Role.SEER,
        Role.DOCTOR,
        Role.VILLAGER,
        Role.VILLAGER,
    ]
    orch, events = _build_orchestrator(deck=deck, internal_events=True)
    doctor = next(player for player in orch.state.players if player.role == Role.DOCTOR)
    doctor_actor = orch.actors[doctor.id]
    victim = min(
        (player for player in orch.state.living_players() if player.role != Role.WEREWOLF),
        key=lambda player: player.seat,
    )

    async def protect_wolf_target(state, player_id, **kwargs):
        doctor_actor.calls.append(("night", player_id))
        doctor_actor.night_kwargs.append(kwargs)
        return Decision(action=AgentAction.SAVE, target_seat=victim.seat)

    doctor_actor.decide_night_action = protect_wolf_target  # type: ignore[method-assign]

    await orch._run_night()

    request = next(item for item in doctor_actor.requests if item.action_kind == "save")
    assert request.phase == "night"
    assert request.private_context == {}
    assert request.legal_actions[0].action == "save"
    assert request.legal_actions[0].can_skip is True
    assert set(request.legal_actions[0].target_seats) == {1, 2, 3, 4, 5, 6}
    assert request.observation["candidate_targets"] == request.legal_actions[0].target_seats
    assert doctor_actor.night_kwargs[-1]["requested_action"] == "save"
    assert doctor_actor.night_kwargs[-1]["human_context"] == {}

    assert victim.alive is True
    assert orch.state.night_deaths == []
    assert orch.state.witch_antidote is True
    doctor_events = [
        event for event in orch.state.events
        if event.type == "night_action_submitted" and doctor.id in event.recipients
    ]
    assert len(doctor_events) == 1
    assert doctor_events[0].payload == {"action": "save", "target_id": victim.id}
    assert any(
        event.get("type") == "night_action_submitted"
        and event.get("seat") == doctor.seat
        and event.get("target_seat") == victim.seat
        for event in events
    )

    response = next(
        row for row in orch._decision_trace
        if row.get("kind") == "agent_response" and row.get("request_id") == request.request_id
    )
    consumed = next(
        row for row in orch._decision_trace
        if row.get("type") == "decision_consumed" and row.get("request_id") == request.request_id
    )
    rules = next(
        row for row in orch._decision_trace
        if row.get("type") == "rules_result" and row.get("request_id") == request.request_id
    )
    assert response["validation"]["valid"] is True
    assert consumed["action"] == "save"
    assert consumed["target_seat"] == victim.seat
    assert rules["rules"] == {"status": "accepted", "action": "save", "reason": None}
    assert rules["target_seat"] == victim.seat

    doctor_memory = [
        item for item in doctor_actor.memory.observations
        if item.kind == "doctor_protect_target"
    ]
    assert len(doctor_memory) == 1
    assert doctor_memory[0].metadata["target_seat"] == victim.seat


@pytest.mark.asyncio
async def test_doctor_skip_is_traced_without_submitting_or_remembering_save():
    deck = [
        Role.WEREWOLF,
        Role.WEREWOLF,
        Role.SEER,
        Role.DOCTOR,
        Role.VILLAGER,
        Role.VILLAGER,
    ]
    orch, _events = _build_orchestrator(deck=deck, internal_events=True)
    doctor = next(player for player in orch.state.players if player.role == Role.DOCTOR)
    actor = orch.actors[doctor.id]

    async def decline_protection(state, player_id, **kwargs):
        actor.calls.append(("night", player_id))
        actor.night_kwargs.append(kwargs)
        return Decision(action=AgentAction.SKIP, skip_reason="doctor_declined")

    actor.decide_night_action = decline_protection  # type: ignore[method-assign]

    await orch._night_role_actions(Role.DOCTOR, [NightActionType.SAVE])

    request = actor.requests[-1]
    assert not orch.state.night_actions
    assert not any(event.type == "night_action_submitted" for event in orch.state.events)
    assert not any(item.kind == "doctor_protect_target" for item in actor.memory.observations)
    assert not orch._failed_events
    response = next(
        row for row in orch._decision_trace
        if row.get("kind") == "agent_response" and row.get("request_id") == request.request_id
    )
    rules = next(
        row for row in orch._decision_trace
        if row.get("type") == "rules_result" and row.get("request_id") == request.request_id
    )
    assert response["validation"]["valid"] is True
    assert response["envelope"]["decision"]["action"] == "skip"
    assert rules["action"] == "skip"
    assert rules["target_seat"] is None
    assert rules["rules"] == {
        "status": "skipped",
        "action": "save",
        "reason": "doctor_declined",
    }


@pytest.mark.asyncio
async def test_doctor_failure_has_no_envelope_action_rules_result_or_memory():
    deck = [
        Role.WEREWOLF,
        Role.WEREWOLF,
        Role.SEER,
        Role.DOCTOR,
        Role.VILLAGER,
        Role.VILLAGER,
    ]
    orch, _events = _build_orchestrator(deck=deck, internal_events=True)
    doctor = next(player for player in orch.state.players if player.role == Role.DOCTOR)
    actor = orch.actors[doctor.id]

    async def fail_protection(*_args, **_kwargs):
        raise RuntimeError("private provider detail")

    actor.decide_night_action = fail_protection  # type: ignore[method-assign]

    await orch._night_role_actions(Role.DOCTOR, [NightActionType.SAVE])

    request = actor.requests[-1]
    assert not orch.state.night_actions
    assert not any(event.type == "night_action_submitted" for event in orch.state.events)
    assert not any(item.kind == "doctor_protect_target" for item in actor.memory.observations)
    assert len(orch._failed_events) == 1
    assert orch._failed_events[0]["type"] == "agent_decision_failed"
    assert orch._failed_events[0]["action"] == "doctor_action"
    assert orch._failed_events[0]["request_id"] == request.request_id
    assert not any(
        row.get("kind") == "agent_response" and row.get("request_id") == request.request_id
        for row in orch._decision_trace
    )
    assert any(
        row.get("kind") == "agent_response_failed" and row.get("request_id") == request.request_id
        for row in orch._decision_trace
    )
    assert not any(
        row.get("type") in {"decision_consumed", "rules_result"}
        and row.get("request_id") == request.request_id
        for row in orch._decision_trace
    )


@pytest.mark.asyncio
async def test_real_night_path_emits_private_rule_events_for_history_and_replay():
    """真实夜间流程必须把规则层私有事件转成 on_event,不能只留在 state.events。"""
    orch, events = _build_orchestrator(
        deck=[
            Role.WEREWOLF,
            Role.WEREWOLF,
            Role.SEER,
            Role.GUARD,
            Role.WITCH,
            Role.VILLAGER,
        ],
        turn_policy="fixed_round_robin",
        internal_events=True,
    )

    await orch._emit_new_rule_events({"role_assigned"})
    await orch._run_night()

    role_events = [event for event in events if event.get("type") == "role_assigned"]
    action_events = [event for event in events if event.get("type") == "night_action_submitted"]
    seer_events = [event for event in events if event.get("type") == "seer_result"]
    seer = next(player for player in orch.state.players if player.role == Role.SEER)

    assert len(role_events) == len(orch.state.players)
    assert all(event["visibility"] == "private" and event["recipients"] for event in role_events)
    assert action_events, "accepted night decisions should emit private action confirmations"
    assert all(event["visibility"] == "private" and event["recipients"] for event in action_events)
    assert len(seer_events) == 1
    assert seer_events[0]["visibility"] == "private"
    assert seer_events[0]["recipients"] == [seer.id]
    assert seer_events[0]["seat"] == seer.seat
    assert seer_events[0]["target_seat"] is not None


@pytest.mark.asyncio
async def test_default_event_callback_projects_public_events_and_hides_restricted_events():
    """直接使用编排器时默认 on_event 只能收到公开投影,私有/狼队事件必须隐藏。"""
    orch, events = _build_orchestrator()

    await orch._emit({"type": "role_assigned", "visibility": "private", "recipients": ["p1"], "role": "seer"})
    await orch._emit({"type": "seer_result", "visibility": "private", "recipients": ["p1"], "target_seat": 2})
    await orch._emit({"type": "team_note", "visibility": "wolf_team", "channel": "wolf_team", "text": "hidden"})
    await orch._emit({
        "type": "speech",
        "day": 1,
        "seat": 1,
        "name": "P1",
        "text": "公开发言",
        "_internal_note": "hidden",
        "reasoning": "hidden",
    })

    assert [event["type"] for event in events] == ["speech"]
    assert events[0]["text"] == "公开发言"
    assert "_internal_note" not in events[0]
    assert "reasoning" not in events[0]


@pytest.mark.asyncio
async def test_action_request_advertises_effective_decision_deadline():
    orch, _events = _build_orchestrator(decision_timeout=0.05)
    orch.state.phase = Phase.DAY
    pid, actor = next(iter(orch.actors.items()))

    await orch._request_agent_decision(
        actor,
        pid,
        action_kind="speak",
        phase="day",
        phase_deadline=time.monotonic() + 10.0,
    )

    request = actor.requests[-1]
    remaining = request.seconds_remaining()
    assert remaining is not None
    assert 0 < remaining <= 0.06


def test_human_decision_failure_records_agent_kind_for_accurate_projection():
    orch, _events = _build_orchestrator()
    actor = next(iter(orch.actors.values()))
    actor.is_human = True
    err = AgentDecisionError("day/speak decision timeout")
    setattr(err, "timeout", True)
    setattr(err, "timeout_seconds", 1.0)

    event = orch._agent_decision_failure_event(
        actor,
        phase="day",
        action="speak",
        err=err,
    )

    assert event["agent_kind"] == "human"
    assert event["timeout"] is True


@pytest.mark.asyncio
async def test_hunter_protocol_passes_requested_action_name_to_actor():
    """猎人开枪也必须传 hunter_shot,不能复用模糊 night_action 让前端猜。"""
    orch, _events = _build_orchestrator(
        deck=[Role.HUNTER, Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.VILLAGER, Role.VILLAGER]
    )
    hunter = next(p for p in orch.state.players if p.role == Role.HUNTER)
    orch.state.pending_hunter = [hunter.id]

    await orch._process_deaths_and_hunter()

    assert orch.actors[hunter.id].night_kwargs[-1]["requested_action"] == "hunter_shot"


@pytest.mark.asyncio
async def test_bid_order_plain_exception_emits_failure_and_does_not_schedule():
    """bid/speak 并发异常不应被排进发言队列或访问 .bid 崩溃。"""
    orch, events = _build_orchestrator(max_speak_rounds=2)
    orch.state.phase = Phase.DAY
    living = [pid for pid in orch.actors if orch.state.get_player(pid).alive]
    failing_pid = living[0]
    actor = orch.actors[failing_pid]

    async def fail_speak(*_args, **_kwargs):
        raise RuntimeError("bid path hidden request")

    actor.decide_speak = fail_speak  # type: ignore[method-assign]

    scheduled = await orch._collect_scheduled_speech_decisions(living, [], set())

    assert failing_pid not in {pid for pid, _decision in scheduled}
    assert not hasattr(actor, "_pending_speak_decision")
    failed = [ev for ev in events if ev.get("type") == "agent_decision_failed"]
    assert failed[-1]["seat"] == actor.seat
    assert failed[-1]["phase"] == "day"
    assert failed[-1]["action"] == "bid_speak"
    assert failed[-1]["error_type"] == "RuntimeError"
    assert "hidden request" not in failed[-1]["reason"]


@pytest.mark.asyncio
async def test_vote_plain_exception_emits_failure_and_vote_incomplete():
    """普通投票异常应有 seat 级失败审计,并保持不完整投票结算。"""
    orch, events = _build_orchestrator()
    orch.state.phase = Phase.DAY
    orch.state = RulesEngine.start_vote(orch.state)
    pid, actor = next(iter(orch.actors.items()))

    async def fail_vote(*_args, **_kwargs):
        raise RuntimeError("vote provider private text")

    actor.decide_vote = fail_vote  # type: ignore[method-assign]

    await orch._run_voting(today_speeches=[{"seat": 2, "text": "公开发言"}])

    failed = [ev for ev in events if ev.get("type") == "agent_decision_failed"]
    assert failed[-1]["seat"] == actor.seat
    assert failed[-1]["phase"] == "voting"
    assert failed[-1]["action"] == "vote"
    assert failed[-1]["error_type"] == "RuntimeError"
    assert "private text" not in failed[-1]["reason"]
    assert not any(ev.get("type") == "vote_cast" and ev.get("seat") == actor.seat for ev in events)
    incomplete = [ev for ev in events if ev.get("type") == "vote_incomplete"]
    assert incomplete[-1]["cast"] == len(orch.state.players) - 1
    assert incomplete[-1]["needed"] == len(orch.state.players)


@pytest.mark.asyncio
async def test_invalid_protocol_vote_is_omitted_while_valid_majority_can_resolve():
    orch, events = _build_orchestrator()
    orch.state.phase = Phase.DAY
    orch.state = RulesEngine.start_vote(orch.state)
    living = list(orch.actors)
    target_id = living[0]
    invalid_pid = living[-1]
    alternate_id = living[1]

    for pid, actor in orch.actors.items():
        async def decide_vote(_state, _player_id, *, _pid=pid, **_kw):
            if _pid == invalid_pid:
                return Decision(action=AgentAction.VOTE, target_seat=999)
            if _pid == target_id:
                return Decision(
                    action=AgentAction.VOTE,
                    target_seat=orch.state.get_player(alternate_id).seat,
                )
            return Decision(
                action=AgentAction.VOTE,
                target_seat=orch.state.get_player(target_id).seat,
            )

        actor.decide_vote = decide_vote  # type: ignore[method-assign]

    await orch._run_voting(today_speeches=[{"seat": 2, "text": "公开发言"}])

    rejected = [ev for ev in events if ev.get("type") == "decision_envelope_rejected"]
    assert rejected[-1]["seat"] == orch.actors[invalid_pid].seat
    assert rejected[-1]["phase"] == "voting"
    incomplete = [ev for ev in events if ev.get("type") == "vote_incomplete"]
    assert incomplete[-1]["cast"] == len(living) - 1
    assert not orch.state.get_player(target_id).alive
    resolved = [ev for ev in events if ev.get("type") == "vote_resolved"]
    assert resolved[-1]["exiled_seat"] == orch.state.get_player(target_id).seat
    response = next(
        item for item in reversed(orch._decision_trace)
        if item.get("kind") == "agent_response"
        and item.get("seat") == orch.actors[invalid_pid].seat
    )
    assert response["validation"]["valid"] is False
    assert response["validation"]["issues"][0]["code"] == "target_seat_not_legal"
    assert response["envelope"]["decision"]["target_seat"] == 999
    assert response["request_id"] == rejected[-1]["request_id"]


@pytest.mark.asyncio
async def test_orchestrator_runs_to_end():
    orch, events = _build_orchestrator(internal_events=True)
    final = await orch.run()
    assert final.phase == Phase.ENDED
    assert final.winner is not None
    assert any(ev["type"] == "analysis" for ev in events)


@pytest.mark.asyncio
async def test_consecutive_agent_failures_end_incomplete_with_paired_request_trace():
    orch, events = _build_orchestrator(
        internal_events=True,
        max_consecutive_decision_failures=2,
        max_consecutive_no_progress_rounds=99,
        max_game_rounds=99,
    )

    async def provider_failed(*_args, **_kwargs):
        raise RuntimeError("private provider outage detail")

    for actor in orch.actors.values():
        actor.decide_night_action = provider_failed  # type: ignore[method-assign]

    final = await orch.run()

    assert final.phase == Phase.ENDED
    assert final.winner is None
    assert orch.termination_status == "incomplete"
    assert orch.termination_reason == "consecutive_decision_failures"
    assert not final.night_actions
    assert not any(event.get("type") == "night_resolved" for event in events)
    ended = next(event for event in events if event.get("type") == "game_ended")
    assert ended["winner"] is None
    assert ended["status"] == "incomplete"
    assert ended["reason"] == "consecutive_decision_failures"
    analysis = next(event["analysis"] for event in events if event.get("type") == "analysis")
    assert analysis["termination"]["reason"] == "consecutive_decision_failures"

    trace = orch._decision_trace_metrics()
    assert trace["request_count"] >= 2
    assert trace["terminal_response_count"] == trace["request_count"]
    assert trace["unpaired_request_count"] == 0
    assert trace["duplicate_terminal_count"] == 0


@pytest.mark.asyncio
async def test_no_progress_round_streak_resets_when_a_later_round_recovers():
    orch, events = _build_orchestrator(
        max_consecutive_decision_failures=99,
        max_consecutive_no_progress_rounds=2,
        max_game_rounds=10,
    )
    orch.state.day = 2

    assert await orch._complete_progress_round() is False
    assert orch._consecutive_no_progress_rounds == 1

    orch._round_had_valid_vote = True
    orch.state.day = 3
    assert await orch._complete_progress_round() is False
    assert orch._consecutive_no_progress_rounds == 0
    assert orch.termination_status == "running"
    assert not any(event.get("type") == "game_ended" for event in events)
    assert [row["progress"] for row in orch._progress_round_history] == [False, True]


@pytest.mark.asyncio
async def test_no_progress_and_max_round_guards_have_distinct_terminal_reasons():
    no_progress, no_progress_events = _build_orchestrator(
        max_consecutive_decision_failures=99,
        max_consecutive_no_progress_rounds=2,
        max_game_rounds=10,
    )
    no_progress.state.day = 2
    assert await no_progress._complete_progress_round() is False
    no_progress.state.day = 3
    assert await no_progress._complete_progress_round() is True
    assert no_progress.termination_reason == "consecutive_no_progress_rounds"
    assert no_progress.state.winner is None
    assert no_progress_events[-1]["reason"] == "consecutive_no_progress_rounds"

    max_rounds, max_round_events = _build_orchestrator(
        max_consecutive_decision_failures=99,
        max_consecutive_no_progress_rounds=99,
        max_game_rounds=2,
    )
    max_rounds.state.day = 2
    max_rounds._round_had_valid_vote = True
    assert await max_rounds._complete_progress_round() is False
    max_rounds.state.day = 3
    max_rounds._round_had_valid_vote = True
    assert await max_rounds._complete_progress_round() is True
    assert max_rounds.termination_reason == "max_game_rounds"
    assert max_rounds.state.winner is None
    assert max_round_events[-1]["reason"] == "max_game_rounds"






@pytest.mark.asyncio
async def test_village_wins_when_villagers_vote_wolves():
    """好人白天稳定投狼、狼人夜间刀民 → 村民应获胜。"""
    orch, events = _build_orchestrator()
    final = await orch.run()
    assert final.winner == Team.VILLAGE


@pytest.mark.asyncio
async def test_vote_resolved_message_matches_actual_event():
    """regression: vote_resolved 不应误用历史 player_exiled 消息。"""
    orch, events = _build_orchestrator()
    await orch.run()
    vote_resolved = [ev for ev in events if ev["type"] == "vote_resolved"]
    # 从真实状态事件中提取对应放逐/平票消息
    state_messages = {ev.message for ev in orch.state.events
                      if ev.type in ("player_exiled", "vote_tied", "vote_tied_pk")}
    for vr in vote_resolved:
        assert vr["message"] in state_messages


@pytest.mark.asyncio
async def test_pk_limit_does_not_emit_extra_speeches_without_vote():
    orch, events = _build_orchestrator()
    orch.state.phase = Phase.VOTING
    living = orch.state.living_players()
    orch.state.pk_candidates = [living[0].id, living[1].id]

    await orch._run_pk(pk_round=2, max_pk_rounds=2)

    assert not orch.state.pk_candidates
    assert orch.state.phase == Phase.NIGHT
    assert not any(ev.get("type") == "phase_started" and ev.get("phase") == "pk" for ev in events)
    assert not any(ev.get("type") == "speech" and ev.get("pk") for ev in events)
    resolved = [ev for ev in events if ev.get("type") == "vote_resolved"]
    assert resolved[-1]["no_exile"] is True
    assert "PK 2 轮仍平票" in resolved[-1]["message"]


@pytest.mark.asyncio
async def test_pk_limit_still_counts_toward_max_game_rounds():
    orch, events = _build_orchestrator(
        max_consecutive_decision_failures=99,
        max_consecutive_no_progress_rounds=99,
        max_game_rounds=1,
    )
    orch.state.phase = Phase.VOTING
    orch.state.day = 1
    living = orch.state.living_players()
    orch.state.pk_candidates = [living[0].id, living[1].id]

    await orch._run_pk(pk_round=2, max_pk_rounds=2)

    assert orch.state.phase == Phase.ENDED
    assert orch.state.winner is None
    assert orch.termination_status == "incomplete"
    assert orch.termination_reason == "max_game_rounds"
    ended = next(event for event in events if event.get("type") == "game_ended")
    assert ended["reason"] == "max_game_rounds"


@pytest.mark.asyncio
async def test_game_ended_emitted_once_before_analysis():
    """结束事件只广播一次;赛后 analysis 仍作为复盘事件到达。"""
    orch, events = _build_orchestrator(internal_events=True)
    await orch.run()

    ended_indices = [i for i, ev in enumerate(events) if ev["type"] == "game_ended"]
    analysis_indices = [i for i, ev in enumerate(events) if ev["type"] == "analysis"]
    assert len(ended_indices) == 1
    assert len(analysis_indices) == 1
    assert ended_indices[0] < analysis_indices[0]




def test_public_speech_memory_records_only_public_fields():
    """公开发言进入所有存活 agent 记忆,但不写私有 reasoning 或额外标签。"""
    orch, _events = _build_orchestrator()
    speech = {
        "seat": 2,
        "name": "P2",
        "text": "我怀疑3号这轮在带节奏。",
        "reply_to": 1,
        "accuses": [3],
        "reasoning": "我是狼所以编的",
        "claim": {"role": "seer"},
        "day": 1,
    }

    orch._record_public_speech_memory(speech)

    alive_count = sum(1 for p in orch.state.players if p.alive)
    recorded = 0
    for actor in orch.actors.values():
        player = orch.state.get_player(next(pid for pid, a in orch.actors.items() if a is actor))
        if not player.alive:
            continue
        obs = actor.memory.observations[-1]
        recorded += 1
        assert obs.kind == "speech"
        assert "我怀疑3号" in obs.text
        assert "我是狼所以编的" not in obs.text
        assert "reasoning" not in obs.metadata
        assert "claim" not in obs.metadata
        assert obs.metadata["speaker_seat"] == 2
        assert obs.metadata["accuses"] == [3]
    assert recorded == alive_count


@pytest.mark.asyncio
async def test_vote_decisions_receive_public_day_speeches():
    """投票 prompt 应拿到当天公开发言证据链,不再空列表投票。"""
    orch, _events = _build_orchestrator()
    await orch.run()

    vote_kwargs = [kw for actor in orch.actors.values() for kw in actor.vote_kwargs]
    assert vote_kwargs
    assert any(kw.get("today_speeches") for kw in vote_kwargs)
    assert all(
        any("seat" in speech and "text" in speech for speech in kw["today_speeches"])
        for kw in vote_kwargs
        if kw.get("today_speeches")
    )


@pytest.mark.asyncio
async def test_live_today_speeches_do_not_leak_hidden_metadata():
    """后续上下文和实时 speech 事件只带公开字段,不泄漏私有 reasoning。"""
    orch, _events = _build_orchestrator()
    events: list[dict[str, Any]] = []
    orch.on_event = lambda ev: events.append(ev)
    for _pid, actor in orch.actors.items():
        def make_speak(mock_actor):
            async def _speak(state, player_id, **kw):
                mock_actor.speak_kwargs.append(kw)
                return Decision(
                    action=AgentAction.SPEAK,
                    speech=f"{mock_actor.seat}号公开发言",
                    bid=1,
                    accuses=[3],
                    reasoning="hidden wolf reasoning",
                    claim={"role": "seer", "checked_seat": 3, "result": "wolf"},
                )
            return _speak

        actor.decide_speak = make_speak(actor)

    await orch.run()

    contexts = [
        speech
        for actor in orch.actors.values()
        for kw in actor.speak_kwargs + actor.vote_kwargs
        for speech in kw.get("today_speeches", [])
    ]
    assert contexts
    for speech in contexts:
        assert "reasoning" not in speech
        assert speech.get("claim") == {"role": "seer", "checked_seat": 3, "result": "wolf"}

    speech_events = [ev for ev in events if ev["type"] == "speech"]
    assert speech_events
    assert all("reasoning" not in ev for ev in speech_events)


@pytest.mark.asyncio
async def test_public_projection_sanitizes_nested_claim_fields():
    events: list[dict[str, Any]] = []
    orch, _events = _build_orchestrator()
    orch.on_event = lambda ev: events.append(ev)
    orch.internal_events = False

    await orch._emit({
        "type": "speech",
        "day": 1,
        "seat": 4,
        "name": "P4",
        "text": "我跳预言家,查2号是狼。",
        "claim": {
            "role": "seer",
            "checked_seat": "2",
            "result": "wolf",
            "reasoning": "SECRET_REASONING",
            "team": "werewolves",
            "teammates": [3],
        },
    })

    assert events == [
        {
            "type": "speech",
            "day": 1,
            "seat": 4,
            "name": "P4",
            "text": "我跳预言家,查2号是狼。",
            "claim": {"role": "seer", "checked_seat": 2, "result": "wolf"},
        }
    ]
    serialized = json.dumps(events, ensure_ascii=False)
    assert "SECRET_REASONING" not in serialized
    assert "teammates" not in serialized
    assert "werewolves" not in serialized


@pytest.mark.asyncio
async def test_public_speech_is_exactly_the_agent_decision_without_rewrite_or_censorship():
    orch, events = _build_orchestrator(turn_policy="fixed_round_robin")
    orch.state.phase = Phase.DAY
    pid, actor = next(iter(orch.actors.items()))
    public_text = "我是狼人。你们可以把这句当自爆，也可以当诈身份。"

    async def decide_speak(*_args, **_kwargs):
        return Decision(
            action=AgentAction.SPEAK,
            speech=public_text,
            bid=3,
            reasoning="这段私有理由不能广播",
        )

    actor.decide_speak = decide_speak  # type: ignore[method-assign]
    await orch._run_day()

    speech = next(ev for ev in events if ev.get("type") == "speech" and ev.get("seat") == actor.seat)
    assert speech["text"] == public_text
    assert "这段私有理由不能广播" not in json.dumps(events, ensure_ascii=False)
    assert not any(str(ev.get("type", "")).startswith("speech_stream_") for ev in events)


@pytest.mark.asyncio
async def test_speech_skip_is_a_resolution_not_a_synthetic_public_utterance():
    orch, events = _build_orchestrator(turn_policy="fixed_round_robin")
    orch.state.phase = Phase.DAY
    _pid, actor = next(iter(orch.actors.items()))

    async def skip_speech(*_args, **_kwargs):
        return Decision(action=AgentAction.SKIP, skip_reason="agent_declined")

    actor.decide_speak = skip_speech  # type: ignore[method-assign]
    await orch._run_day()

    assert not any(
        ev.get("type") == "speech" and ev.get("seat") == actor.seat
        for ev in events
    )
    assert not any(
        ev.get("text") in {"(沉默)", "(倾听)"}
        for ev in events
    )
    skipped = [
        item for item in orch._decision_trace
        if item.get("type") == "rules_result"
        and item.get("seat") == actor.seat
        and item.get("action") == "skip"
    ]
    assert any(item["rules"]["status"] == "skipped" for item in skipped)


@pytest.mark.asyncio
async def test_last_words_skip_has_no_fabricated_quote():
    orch, events = _build_orchestrator()
    pid, actor = next(iter(orch.actors.items()))
    RulesEngine.queue_last_words(orch.state, pid, reason="exiled")

    async def skip_last_words(*_args, **_kwargs):
        return Decision(action=AgentAction.SKIP, skip_reason="agent_declined")

    actor.decide_last_words = skip_last_words  # type: ignore[method-assign]
    await orch._process_deaths_and_hunter()

    assert not any(
        ev.get("type") == "last_words" and ev.get("seat") == actor.seat
        for ev in events
    )
    skipped = next(
        ev for ev in events
        if ev.get("type") == "last_words_skipped" and ev.get("seat") == actor.seat
    )
    assert skipped["skip_reason"] == "agent_declined"
    assert "text" not in skipped
    assert not orch.state.last_words_queue


@pytest.mark.asyncio
async def test_accepted_last_words_enter_every_living_agents_memory_exactly():
    orch, events = _build_orchestrator()
    pid, actor = next(iter(orch.actors.items()))
    speaker = orch.state.get_player(pid)
    speaker.alive = False
    exact_text = "  我最后坚持：查验与投票必须分开看。\n"
    RulesEngine.queue_last_words(orch.state, pid, reason="exiled")

    async def exact_last_words(*_args, **_kwargs):
        return Decision(action=AgentAction.LAST_WORDS, speech=exact_text)

    actor.decide_last_words = exact_last_words  # type: ignore[method-assign]
    await orch._process_last_words_queue()

    public_event = next(
        event
        for event in events
        if event.get("type") == "last_words" and event.get("seat") == speaker.seat
    )
    assert public_event["text"] == exact_text
    for observer_pid, observer in orch.actors.items():
        remembered = [item for item in observer.memory.observations if item.kind == "last_words"]
        if orch.state.get_player(observer_pid).alive:
            assert len(remembered) == 1
            assert remembered[0].text == f"{speaker.seat}号遗言:{exact_text}"
            assert remembered[0].metadata == {"speaker_seat": speaker.seat}
        else:
            assert remembered == []


def test_last_words_instruction_maps_death_reason_without_defaulting_to_wolf_kill():
    assert "被放逐" in last_words_instruction("exiled")
    assert "被猎人带走" in last_words_instruction("hunter_shot")
    assert "中毒死亡" in last_words_instruction("poisoned")
    assert "昨夜死亡" in last_words_instruction("wolf_kill")
    assert "被狼人杀害" not in last_words_instruction("hunter_shot")
    assert "被狼人杀害" not in last_words_instruction("poisoned")




def test_parse_metrics_preserve_tool_protocol_status_and_trace_parse_priority():
    orch, _events = _build_orchestrator()
    actor = next(iter(orch.actors.values()))

    tool_decision = Decision(action=AgentAction.SKIP, skip_reason="tool_selected_skip")
    tool_decision.llm_call_trace = {
        "call_id": "tool-model-call",
        "actor_response_attempt_count": 1,
    }
    orch._record_consumed_decision(
        actor,
        DecisionEnvelope(
            request_id="tool-loop-request",
            seat=actor.seat,
            decision=tool_decision,
            model_call_id="tool-model-call",
            parse_status="not_applicable",
            metadata={"agent_kind": "llm", "runtime": "tool_loop"},
        ),
        phase="day",
    )

    parsed_decision = Decision(action=AgentAction.SKIP, skip_reason="parsed_skip")
    parsed_decision.llm_call_trace = {
        "call_id": "parsed-model-call",
        "parse": {"method": "fenced_json", "recovered": True, "lossy": False},
    }
    orch._record_consumed_decision(
        actor,
        DecisionEnvelope(
            request_id="parsed-request",
            seat=actor.seat,
            decision=parsed_decision,
            model_call_id="parsed-model-call",
            parse_status="not_applicable",
        ),
        phase="voting",
    )

    metrics = orch._parse_metrics()
    assert metrics["decision_count"] == 2
    assert metrics["parsed_model_decision_count"] == 1
    assert metrics["parse_recovered_count"] == 1
    assert metrics["parse_method_counts"] == {"fenced_json": 1}
    assert metrics["not_applicable_count"] == 1
    assert metrics["missing_provenance_count"] == 0
    assert orch._consumed_decisions[0]["model_call_id"] == "tool-model-call"


@pytest.mark.asyncio
async def test_analysis_includes_parse_metrics_from_consumed_response_provenance():
    """赛后只统计真实 response provenance，不在 Decision 上伪造失败位。"""
    orch, events = _build_orchestrator(internal_events=True)
    actor = next(a for a in orch.actors.values() if a.role == Role.WEREWOLF)
    original_speak = actor.decide_speak
    original_vote = actor.decide_vote
    marked = {"speak": False, "vote": False}

    def recovered(decision: Decision, method: str) -> Decision:
        return decision.model_copy(update={
            "llm_call_trace": {
                "call_id": f"recovered-{method}",
                "parse": {"method": method, "recovered": True, "lossy": False},
            }
        })

    async def recovered_speak(state, player_id, **kw):
        decision = await original_speak(state, player_id, **kw)
        if not marked["speak"]:
            marked["speak"] = True
            return recovered(decision, "fenced_json")
        return decision

    async def recovered_vote(state, player_id, **kw):
        decision = await original_vote(state, player_id, **kw)
        if not marked["vote"]:
            marked["vote"] = True
            return recovered(decision, "literal")
        return decision

    actor.decide_speak = recovered_speak
    actor.decide_vote = recovered_vote

    await orch.run()

    analysis = next(ev for ev in events if ev["type"] == "analysis")["analysis"]
    metrics = analysis["parse_metrics"]
    trace_metrics = analysis["decision_trace_metrics"]

    assert metrics["decision_count"] > 0
    assert analysis["decision_count"] == metrics["decision_count"]
    assert trace_metrics["consumed_decision_count"] == analysis["decision_count"]
    assert trace_metrics["trace_row_count"] > analysis["decision_count"]
    assert trace_metrics["terminal_response_count"] == (
        trace_metrics["response_count"]
        + trace_metrics["response_failure_count"]
        + trace_metrics["response_cancelled_count"]
        + trace_metrics["response_validation_failure_count"]
    )
    assert trace_metrics["terminal_response_count"] == trace_metrics["request_count"]
    assert trace_metrics["unpaired_request_count"] == 0
    assert trace_metrics["duplicate_terminal_count"] == 0
    assert trace_metrics["orphan_terminal_count"] == 0
    assert metrics["parsed_model_decision_count"] == 2
    assert metrics["parse_recovered_count"] == 2
    assert metrics["parse_recovered_rate"] == 1.0
    assert metrics["parse_recovered_by_action"]["speak"] == 1
    assert metrics["parse_recovered_by_action"]["vote"] == 1
    assert metrics["parse_recovered_by_phase"]["day"] == 1
    assert metrics["parse_recovered_by_phase"]["voting"] == 1
    assert metrics["parse_method_counts"] == {"fenced_json": 1, "literal": 1}
    assert metrics["lossy_consumed_count"] == 0
    assert metrics["missing_provenance_count"] == metrics["decision_count"] - 2
    assert all("parse_failed" not in ev for ev in events)










@pytest.mark.asyncio
async def test_speak_order_uses_llm_bid_in_second_round():
    """第二轮发言顺序应由 LLM 生成的 bid 决定(高 bid 先发言)。"""
    orch, events = _build_orchestrator(turn_policy="bid_reply")
    # 给每个 seat 分配固定 bid: seat 3 最高,其余依次
    bids = {1: 1, 2: 2, 3: 4, 4: 1, 5: 3, 6: 1}
    # 每个 seat 的发言计数器:每轮给不同内容,便于区分轮次
    counters = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for pid, actor in orch.actors.items():
        seat = actor.seat
        orig = actor.decide_speak

        def make_speak(original, seat_bid, seat_num):
            async def _speak(state, player_id, **kw):
                dec = await original(state, player_id, **kw)
                dec.bid = seat_bid
                counters[seat_num] += 1
                # 第 n 次发言带不同序号,便于断言顺序
                dec.speech = f"{seat_num}号第{counters[seat_num]}次发言"
                return dec
            return _speak

        actor.decide_speak = make_speak(orig, bids[seat], seat)

    orch.max_speak_rounds = 2
    await orch.run()

    speeches = [ev for ev in events if ev["type"] == "speech"]
    # 按天分组
    by_day: dict[int, list[dict]] = {}
    for ev in speeches:
        by_day.setdefault(ev["day"], []).append(ev)

    found = False
    for day, day_speeches in by_day.items():
        # 若某天有两轮发言,第二轮应为 bid 降序
        if len(day_speeches) >= 4:
            half = len(day_speeches) // 2
            second_round = day_speeches[half:]
            expected_order = sorted(second_round, key=lambda e: -bids[e["seat"]])
            assert [e["seat"] for e in second_round] == [e["seat"] for e in expected_order], (
                f"day {day} 第二轮顺序 {second_round} 不符合 bid 降序"
            )
            found = True
            break
    assert found, "未找到含两轮发言的白天"


@pytest.mark.asyncio
async def test_fixed_round_robin_policy_uses_fixed_order():
    """fixed_round_robin 每轮按 seat 顺序发言。"""
    orch, events = _build_orchestrator(turn_policy="fixed_round_robin")
    orch.state.phase = Phase.DAY
    orch.state.day = 1
    orch.max_speak_rounds = 2
    async def _no_vote(**kw):
        return None

    orch._run_voting = _no_vote

    counters = {seat: 0 for seat in range(1, 7)}
    unique_words = {
        (1, 1): "aaaaaaaa",
        (1, 2): "bbbbbbbb",
        (1, 3): "cccccccc",
        (1, 4): "dddddddd",
        (1, 5): "eeeeeeee",
        (1, 6): "ffffffff",
        (2, 1): "gggggggg",
        (2, 2): "hhhhhhhh",
        (2, 3): "iiiiiiii",
        (2, 4): "jjjjjjjj",
        (2, 5): "kkkkkkkk",
        (2, 6): "llllllll",
    }
    for actor in orch.actors.values():
        async def _speak(state, player_id, *, _actor=actor, **kw):
            counters[_actor.seat] += 1
            return Decision(
                    action=AgentAction.SPEAK,
                    speech=unique_words[(counters[_actor.seat], _actor.seat)],
                    bid=1,
                )

        actor.decide_speak = _speak

    await orch._run_day()

    speeches = [ev for ev in events if ev["type"] == "speech"]
    assert [ev["seat"] for ev in speeches] == [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]
