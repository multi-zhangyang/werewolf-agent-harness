"""人机混合模式测试 —— 验证人类玩家操作队列与超时跳过。"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from src.agent.actor import AgentActor
from src.agent.schemas import AgentAction, Decision
from src.game.models import GameState
from src.game.roles import Role
from src.game.state import new_game
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


@pytest.mark.asyncio
async def test_human_vote_action():
    actor = _make_actor()
    state = _make_state()

    async def feed():
        await asyncio.sleep(0.05)
        actor.human_queue.put_nowait({"action": "vote", "target_seat": 2})

    asyncio.create_task(feed())
    dec = await actor.decide_vote(state, state.players[0].id)
    assert dec.action == AgentAction.VOTE
    assert dec.target_id == state.players[1].id  # seat 2


@pytest.mark.asyncio
async def test_human_speak_action():
    actor = _make_actor()
    state = _make_state()

    async def feed():
        await asyncio.sleep(0.05)
        actor.human_queue.put_nowait({"action": "speak", "speech": "我是好人", "bid": 3})

    asyncio.create_task(feed())
    dec = await actor.decide_speak(state, state.players[0].id)
    assert dec.action == AgentAction.SPEAK
    assert dec.speech == "我是好人"
    assert dec.bid == 3


@pytest.mark.asyncio
async def test_human_action_timeout_becomes_skip():
    actor = _make_actor()
    state = _make_state()

    with patch("src.config.HUMAN_TIMEOUT", 0.1):
        dec = await actor.decide_vote(state, state.players[0].id, today_speeches=[])
    assert dec.action == AgentAction.SKIP
    assert dec.skip_reason == "human_timeout"


@pytest.mark.asyncio
async def test_human_skip_action():
    actor = _make_actor()
    state = _make_state()

    async def feed():
        await asyncio.sleep(0.01)
        actor.human_queue.put_nowait({"action": "skip"})

    asyncio.create_task(feed())
    dec = await actor.decide_vote(state, state.players[0].id, today_speeches=[])
    assert dec.action == AgentAction.SKIP
    assert dec.skip_reason == "human_skip"


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
async def test_human_requested_night_action_infers_action_without_duplicate_field(
    requested_action: str,
    expected: AgentAction,
):
    """前端只提交目标时,后端应按 requested_action 解析,不能要求暗传 action。"""
    actor = _make_actor(role=Role.WITCH)
    state = _make_state()

    async def feed():
        await asyncio.sleep(0.01)
        actor.human_queue.put_nowait({"target_seat": 2})

    asyncio.create_task(feed())
    dec = await actor.decide_night_action(
        state,
        state.players[0].id,
        requested_action=requested_action,
    )

    assert dec.action == expected
    assert dec.target_id == state.players[1].id


@pytest.mark.asyncio
async def test_human_invalid_target_returns_skip_not_exception():
    actor = _make_actor()
    state = _make_state()

    async def feed():
        await asyncio.sleep(0.01)
        actor.human_queue.put_nowait({"action": "vote", "target_seat": "abc"})

    asyncio.create_task(feed())
    dec = await actor.decide_vote(state, state.players[0].id, today_speeches=[])

    assert dec.action == AgentAction.SKIP
    assert dec.skip_reason == "human_action_invalid_target"


@pytest.mark.asyncio
async def test_ai_actor_ignores_human_queue():
    actor = _make_actor(is_human=False)
    state = _make_state()
    # 即使队列有数据,AI actor 也不应读取
    actor.human_queue.put_nowait({"action": "vote", "target_seat": 2})
    # AI actor 会尝试 LLM 调用,模型配置为空会失败,最终抛 AgentDecisionError
    with pytest.raises(Exception):
        await actor.decide_vote(state, state.players[0].id, today_speeches=[])
