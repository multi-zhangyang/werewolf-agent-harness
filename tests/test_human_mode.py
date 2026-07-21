"""Human AgentProtocol input, validation, explicit SKIP, and timeout tests."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from src.agent.actor import AgentActor, AgentDecisionError
from src.agent.schemas import AgentAction, AgentObservation, Decision
from src.game.models import GameState
from src.game.roles import Role, role_team
from src.game.state import new_game
from src.harness.agent_protocol import ActionRequest, LegalAction
from src.harness.decision_runtime import DecisionRuntime
from src.llm.models import ModelConfig
from src.llm.router import LLMRouter


def _make_state() -> GameState:
    state = new_game(["A", "B", "C", "D", "E", "F"])
    for p in state.players:
        p.role = Role.VILLAGER
    return state


def _make_actor(is_human: bool = True, *, role: Role = Role.VILLAGER) -> AgentActor:
    return AgentActor(
        seat=1,
        name="A",
        role=role,
        model_config=ModelConfig(),
        router=LLMRouter(),
        is_human=is_human,
    )


def _request(
    actor: AgentActor,
    state: GameState,
    action_kind: str,
    *,
    target_seats: list[int] | None = None,
    private_context: dict | None = None,
) -> ActionRequest:
    target_actions = {"night_kill", "see", "save", "poison", "guard", "hunter_shot", "vote"}
    if target_seats is None:
        target_seats = (
            [player.seat for player in state.players if player.alive and player.seat != actor.seat]
            if action_kind in target_actions
            else []
        )
    observation = AgentObservation(
        my_seat=actor.seat,
        my_role=actor.role.value,
        my_team=role_team(actor.role).value,
        seats=[
            {"id": player.id, "seat": player.seat, "name": player.name, "alive": player.alive}
            for player in state.players
        ],
        alive_seats=[player.seat for player in state.players if player.alive],
        phase=state.phase.value,
        day=state.day,
        available_actions=[action_kind],
        candidate_targets=target_seats,
        vote_targets=target_seats if action_kind == "vote" else [],
    )
    protocol_action = "night_kill" if action_kind == "hunter_shot" else action_kind
    return ActionRequest(
        request_id=f"human-{action_kind}",
        run_id=state.id,
        seat=actor.seat,
        phase=state.phase.value,
        day=state.day,
        action_kind=action_kind,
        observation=observation.model_dump(),
        legal_actions=[LegalAction(
            action=protocol_action,
            target_seats=target_seats,
            target_required=action_kind in target_actions,
            can_skip=True,
        )],
        private_context=private_context or {},
    )


async def _submit_current_request(actor: AgentActor, payload: dict) -> None:
    for _ in range(100):
        current = actor.current_human_request
        if current:
            action = {
                "request_id": current["request_id"],
                "day": current["day"],
                "phase": current["phase"],
                **payload,
            }
            accepted, reason = actor.enqueue_human_action(action)
            assert accepted, reason
            return
        await asyncio.sleep(0.005)
    raise AssertionError("human request was not created")


@pytest.mark.asyncio
async def test_human_vote_action():
    actor = _make_actor()
    state = _make_state()

    async def feed():
        await asyncio.sleep(0.05)
        await _submit_current_request(actor, {"action": "vote", "target_seat": 2})

    asyncio.create_task(feed())
    dec = (await actor.decide(_request(actor, state, "vote"))).decision
    assert dec.action == AgentAction.VOTE
    assert dec.target_seat == 2


@pytest.mark.asyncio
async def test_human_speak_action():
    actor = _make_actor()
    state = _make_state()

    async def feed():
        await asyncio.sleep(0.05)
        await _submit_current_request(actor, {"action": "speak", "speech": "我是好人", "bid": 3})

    asyncio.create_task(feed())
    dec = (await actor.decide(_request(actor, state, "speak"))).decision
    assert dec.action == AgentAction.SPEAK
    assert dec.speech == "我是好人"
    assert dec.bid == 3


@pytest.mark.asyncio
async def test_human_action_timeout_produces_no_decision_envelope():
    actor = _make_actor()
    state = _make_state()
    events: list[dict] = []

    async def capture(payload: dict) -> None:
        events.append(payload)

    actor.on_human_request = capture

    with patch("src.config.HUMAN_TIMEOUT", 0.1):
        with pytest.raises(AgentDecisionError) as raised:
            await actor.decide(_request(actor, state, "vote"))
    assert getattr(raised.value, "error_type") == "HumanDecisionTimeout"
    assert getattr(raised.value, "timeout") is True
    assert [event["type"] for event in events] == [
        "human_action_request",
        "human_action_expired",
    ]
    assert events[1]["request_id"] == events[0]["request_id"]
    assert events[1]["reason"] == "human_timeout"


@pytest.mark.asyncio
async def test_human_timeout_has_failed_protocol_terminal_not_skip_envelope():
    actor = _make_actor()
    state = _make_state()
    lifecycle: list[dict] = []
    trace: list[dict] = []

    async def capture(payload: dict) -> None:
        lifecycle.append(payload)

    actor.on_human_request = capture
    timeout = 0.05
    request = _request(actor, state, "vote").model_copy(update={
        "deadline_monotonic": time.monotonic() + timeout,
        "metadata": {
            "deadline_source": "decision",
            "effective_timeout_seconds": timeout,
        },
    })

    with pytest.raises(AgentDecisionError) as raised:
        await DecisionRuntime(on_trace=trace.append).execute(actor, request)

    assert getattr(raised.value, "request_id") == request.request_id
    assert getattr(raised.value, "error_type") == "HumanDecisionTimeout"
    assert [event["type"] for event in lifecycle] == [
        "human_action_request",
        "human_action_expired",
    ]
    assert [row["kind"] for row in trace] == [
        "agent_request",
        "agent_response_failed",
    ]
    assert trace[1]["failure"]["error_type"] == "HumanDecisionTimeout"
    assert all(row.get("kind") != "agent_response" for row in trace)


@pytest.mark.asyncio
async def test_human_skip_action():
    actor = _make_actor()
    state = _make_state()

    async def feed():
        await asyncio.sleep(0.01)
        await _submit_current_request(actor, {"action": "skip"})

    asyncio.create_task(feed())
    dec = (await actor.decide(_request(actor, state, "vote"))).decision
    assert dec.action == AgentAction.SKIP
    assert dec.skip_reason == "human_skip"


@pytest.mark.asyncio
async def test_human_targeted_request_with_empty_target_set_only_allows_skip():
    actor = _make_actor(role=Role.WITCH)
    state = _make_state()
    lifecycle: list[dict] = []

    async def capture(payload: dict) -> None:
        lifecycle.append(payload)

    actor.on_human_request = capture
    request = _request(actor, state, "save", target_seats=[])
    task = asyncio.create_task(actor.decide(request))
    for _ in range(100):
        if actor.current_human_request:
            break
        await asyncio.sleep(0.005)

    assert actor.current_human_request is not None
    assert actor.current_human_request["requires_target"] is True
    assert actor.current_human_request["allowed_target_seats"] == []
    accepted, reason = actor.enqueue_human_action({
        "request_id": request.request_id,
        "day": request.day,
        "phase": request.phase,
        "action": "save",
        "target_seat": 2,
    })
    assert not accepted
    assert reason == "target_not_allowed"

    accepted, reason = actor.enqueue_human_action({
        "request_id": request.request_id,
        "day": request.day,
        "phase": request.phase,
        "action": "skip",
    })
    assert accepted, reason
    envelope = await task
    assert envelope.decision.action == AgentAction.SKIP
    assert lifecycle[0]["context"]["requires_target"] is True
    assert lifecycle[0]["context"]["allowed_target_seats"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_action", "expected"),
    [
        ("night_kill", AgentAction.NIGHT_KILL),
        ("see", AgentAction.SEE),
        ("save", AgentAction.SAVE),
        ("poison", AgentAction.POISON),
        ("guard", AgentAction.GUARD),
        ("hunter_shot", AgentAction.NIGHT_KILL),
    ],
)
async def test_human_requested_night_action_preserves_explicit_action_and_target(
    requested_action: str,
    expected: AgentAction,
):
    """Human response explicitly binds the action and target to the request."""
    actor = _make_actor(role=Role.WITCH)
    state = _make_state()
    human_context = {"killed_seat": 2} if requested_action == "save" else None

    async def feed():
        await asyncio.sleep(0.01)
        await _submit_current_request(actor, {"action": requested_action, "target_seat": 2})

    asyncio.create_task(feed())
    dec = (
        await actor.decide(_request(
            actor,
            state,
            requested_action,
            target_seats=[2],
            private_context=human_context,
        ))
    ).decision

    assert dec.action == expected
    assert dec.target_seat == 2


@pytest.mark.asyncio
async def test_human_rejects_stale_or_malformed_request_before_queueing():
    actor = _make_actor()
    state = _make_state()

    async def feed():
        await asyncio.sleep(0.01)
        current = actor.current_human_request
        assert current is not None
        accepted, reason = actor.enqueue_human_action({
            "request_id": current["request_id"],
            "day": current["day"],
            "phase": current["phase"],
            "action": "speak",
            "speech": "我是好人",
        })
        assert not accepted
        assert reason == "bid_required"
        accepted, reason = actor.enqueue_human_action({
            "request_id": "stale",
            "day": current["day"],
            "phase": current["phase"],
            "action": "vote",
            "target_seat": 2,
        })
        assert not accepted
        assert reason == "request_id_mismatch"
        accepted, reason = actor.enqueue_human_action({
            "request_id": current["request_id"],
            "day": current["day"],
            "phase": current["phase"],
            "action": "speak",
            "speech": "错阶段发言",
        })
        assert not accepted
        assert reason == "action_type_mismatch"
        accepted, reason = actor.enqueue_human_action({
            "request_id": current["request_id"],
            "day": current["day"],
            "phase": current["phase"],
            "action": "vote",
            "target_seat": "abc",
        })
        assert not accepted
        assert reason == "target_invalid"
        accepted, reason = actor.enqueue_human_action({
            "request_id": current["request_id"],
            "day": current["day"],
            "phase": current["phase"],
            "action": "vote",
            "target_seat": "2.9",
        })
        assert not accepted
        assert reason == "target_invalid"
        accepted, reason = actor.enqueue_human_action({
            "request_id": current["request_id"],
            "day": current["day"],
            "phase": current["phase"],
            "action": "vote",
        })
        assert not accepted
        assert reason == "target_required"

    asyncio.create_task(feed())
    with patch("src.config.HUMAN_TIMEOUT", 0.1):
        with pytest.raises(AgentDecisionError) as raised:
            await actor.decide(_request(actor, state, "vote"))

    assert getattr(raised.value, "error_type") == "HumanDecisionTimeout"


@pytest.mark.asyncio
async def test_human_speak_rejects_malformed_bid_before_queueing():
    actor = _make_actor()
    state = _make_state()

    async def feed():
        await asyncio.sleep(0.01)
        current = actor.current_human_request
        assert current is not None
        accepted, reason = actor.enqueue_human_action({
            "request_id": current["request_id"],
            "day": current["day"],
            "phase": current["phase"],
            "action": "speak",
            "speech": "我是好人",
            "bid": "high",
        })
        assert not accepted
        assert reason == "bid_invalid"
        accepted, reason = actor.enqueue_human_action({
            "request_id": current["request_id"],
            "day": current["day"],
            "phase": current["phase"],
            "action": "speak",
            "speech": "我是好人",
            "bid": 9,
        })
        assert not accepted
        assert reason == "bid_out_of_range"
        accepted, reason = actor.enqueue_human_action({
            "request_id": current["request_id"],
            "day": current["day"],
            "phase": current["phase"],
            "action": "speak",
            "speech": "我是好人",
            "bid": 0,
        })
        assert not accepted
        assert reason == "bid_zero_requires_skip"

    asyncio.create_task(feed())
    with patch("src.config.HUMAN_TIMEOUT", 0.1):
        with pytest.raises(AgentDecisionError) as raised:
            await actor.decide(_request(actor, state, "speak"))

    assert getattr(raised.value, "error_type") == "HumanDecisionTimeout"


@pytest.mark.asyncio
async def test_ai_actor_ignores_human_queue():
    actor = _make_actor(is_human=False)
    state = _make_state()
    # 即使队列有数据,AI actor 也不应读取
    actor.human_queue.put_nowait({"action": "vote", "target_seat": 2})
    # AI actor 会尝试 LLM 调用,模型配置为空会失败,最终抛 AgentDecisionError
    with pytest.raises(Exception):
        await actor.decide(_request(actor, state, "vote"))
