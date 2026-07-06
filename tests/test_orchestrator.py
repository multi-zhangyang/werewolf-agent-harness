"""编排器集成测试 —— 用 Mock Actor 避免真实 LLM 调用。"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from src.agent.actor import AgentDecisionError
from src.agent.memory import AgentMemory
from src.agent.schemas import AgentAction, Decision
from src.game.models import GameState, NightActionType, Phase
from src.game.orchestrator import GameOrchestratorV2
from src.game.roles import Role, Team
from src.game.rules import RulesEngine
from src.game.state import new_game
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

    async def decide_night_action(self, state: GameState, player_id: str, **kw) -> Decision:
        self.calls.append(("night", player_id))
        self.night_kwargs.append(kw)
        # 狼人稳定刀第一个非狼村民; 预言家查验第一个狼; 其他跳过
        targets = {p.seat: p.id for p in state.living_players()}
        my_team = {p.id for p in state.players if p.role == Role.WEREWOLF}
        if self.role == Role.WEREWOLF:
            for seat, pid in sorted(targets.items()):
                if pid not in my_team:
                    return Decision(action=AgentAction.NIGHT_KILL, target_id=pid)
        if self.role == Role.SEER:
            wolf = next((p for p in state.players if p.role == Role.WEREWOLF and p.alive), None)
            if wolf:
                return Decision(action=AgentAction.SEE, target_id=wolf.id)
        return Decision(action=AgentAction.SKIP)

    async def decide_speak(self, state: GameState, player_id: str, **kw) -> Decision:
        self.calls.append(("speak", player_id))
        self.speak_kwargs.append(kw)
        return Decision(action=AgentAction.SPEAK, speech=f"我是{self.seat}号", bid=1)

    async def decide_wolf_caucus(self, state: GameState, player_id: str, **kw) -> dict | None:
        # 方向C:狼队党团会议——提议推第一个存活好人
        if self.role != Role.WEREWOLF:
            return None
        my_team = {p.id for p in state.players if p.role == Role.WEREWOLF}
        for p in state.living_players():
            if p.id not in my_team:
                return {"target_seat": p.seat, "strategy": "集体指控他", "reasoning": "mock"}
        return None

    async def decide_vote(self, state: GameState, player_id: str, **kw) -> Decision:
        self.calls.append(("vote", player_id))
        self.vote_kwargs.append(kw)
        # 好人投第一个存活的狼; 狼人投第一个非狼
        wolves = {p.id for p in state.players if p.role == Role.WEREWOLF and p.alive}
        if self.role == Role.WEREWOLF:
            for p in state.living_players():
                if p.id not in wolves:
                    return Decision(
                        action=AgentAction.VOTE,
                        target_id=p.id,
                        objective_summary=f"{self.seat}号客观摘要",
                        reasoning="如果投好人则狼人收益更高。",
                    )
        for p in state.living_players():
            if p.id in wolves:
                return Decision(
                    action=AgentAction.VOTE,
                    target_id=p.id,
                    objective_summary=f"{self.seat}号客观摘要",
                    reasoning="如果投到狼人则好人收益更高。",
                )
        return Decision(action=AgentAction.SKIP)

    async def decide_last_words(self, state: GameState, player_id: str, reason: str, **kw) -> Decision:
        return Decision(action=AgentAction.LAST_WORDS, speech="遗言")

    async def reflect(self, state: GameState, player_id: str, **kw) -> None:
        return None

    def observe_event(self, *args, **kw) -> None:
        if len(args) >= 4:
            day, phase, kind, text = args[:4]
            self.memory.observe(day, phase, kind, text, **kw)

    def record_claim(self, seat: int, day: int, claim: dict[str, Any]) -> None:
        self.memory.record_claim(seat, day, claim)

    def apply_suspicion(self, suspicion: dict[int, float]) -> None:
        pass

    def thinking_summary(self, decision: Decision) -> Any:
        return type("T", (), {"model_dump": lambda: {"seat": self.seat}})()


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
    return orch, events


@pytest.mark.asyncio
async def test_day_speech_timeout_is_transparent_failure_without_fake_speech():
    """单次发言墙钟超时应透明失败,不能广播迟到的假/兜底发言。"""
    orch, events = _build_orchestrator(decision_timeout=0.01)
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
    orch, events = _build_orchestrator(decision_timeout=0.01)
    orch.state.phase = Phase.DAY
    pid, actor = next(iter(orch.actors.items()))

    async def slow_vote(*_args, **_kwargs):
        await asyncio.sleep(1)
        target = next(p for p in orch.state.living_players() if p.id != pid)
        return Decision(action=AgentAction.VOTE, target_id=target.id)

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
    assert shots[-1]["skip_reason"] == "hunter_decision_failed"


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
    await orch._werewolf_deliberation()
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
async def test_wolf_day_caucus_plain_exception_emits_failure_immediately():
    """白天党团会议失败不能缓存在夜晚失败队列里。"""
    orch, events = _build_orchestrator()
    wolves = [(pid, actor) for pid, actor in orch.actors.items() if actor.role == Role.WEREWOLF]
    pid, actor = wolves[0]
    other_actor = wolves[1][1]

    async def fail_caucus(*_args, **_kwargs):
        raise ValueError("wolf caucus provider private text")

    actor.decide_wolf_caucus = fail_caucus  # type: ignore[method-assign]

    await orch._werewolf_day_caucus()

    failed = [ev for ev in events if ev.get("type") == "agent_decision_failed"]
    assert failed[-1]["seat"] == actor.seat
    assert failed[-1]["phase"] == "day"
    assert failed[-1]["action"] == "wolf_caucus"
    assert failed[-1]["error_type"] == "ValueError"
    assert "private text" not in failed[-1]["reason"]
    assert orch._failed_events == []
    consensus = [ev for ev in events if ev.get("type") == "wolf_caucus_consensus"]
    assert consensus
    assert any(
        obs.kind == "wolf_caucus_consensus"
        for obs in other_actor.memory.observations
    )


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

    order = await orch._bid_order_by_llm(living, [], set())

    assert failing_pid not in order
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
async def test_reflection_plain_exception_emits_failure_without_hiding_others():
    """反思失败不影响其他反思更新,但必须进入 no-fallback 审计。"""
    orch, events = _build_orchestrator()
    pid, actor = next(iter(orch.actors.items()))

    async def fail_reflect(*_args, **_kwargs):
        raise RuntimeError("reflect provider private text")

    async def ok_reflect(self, state, player_id, **kw):
        self.memory.reflect(state.day, "night", f"{self.seat}号复盘")

    actor.reflect = fail_reflect  # type: ignore[method-assign]
    for other in orch.actors.values():
        if other is not actor:
            other.reflect = ok_reflect.__get__(other, type(other))  # type: ignore[method-assign]

    await orch._reflect_all()

    failed = [ev for ev in events if ev.get("type") == "agent_decision_failed"]
    assert failed[-1]["seat"] == actor.seat
    assert failed[-1]["phase"] == "reflection"
    assert failed[-1]["action"] == "reflect"
    assert failed[-1]["error_type"] == "RuntimeError"
    assert "private text" not in failed[-1]["reason"]
    reflections = [ev for ev in events if ev.get("type") == "reflections_update"]
    assert reflections
    assert str(actor.seat) not in reflections[-1]["reflections"]
    assert len(reflections[-1]["reflections"]) == len(orch.actors) - 1


@pytest.mark.asyncio
async def test_orchestrator_runs_to_end():
    orch, events = _build_orchestrator()
    final = await orch.run()
    assert final.phase == Phase.ENDED
    assert final.winner is not None
    assert any(ev["type"] == "analysis" for ev in events)


@pytest.mark.asyncio
async def test_analysis_includes_quality_when_judge_succeeds(monkeypatch):
    """赛后 quality judge 成功时应进入 analysis,且 judge temperature 固定为 0。"""
    calls: list[dict[str, Any]] = []

    async def fake_score_game_quality(**kwargs):
        calls.append(kwargs)
        return {
            "scores": [
                {
                    "seat": 1,
                    "role": "villager",
                    "RI": 0.8,
                    "SJ": 0.7,
                    "DR": 0.6,
                    "PS": 0.5,
                    "CT": 0.4,
                    "highlight": "mock",
                }
            ],
            "game_quality": 0.66,
            "game_summary": "mock quality",
        }

    monkeypatch.setattr("src.agent.quality.score_game_quality", fake_score_game_quality)
    orch, events = _build_orchestrator()
    cfg = ModelConfig(
        provider="openai",
        model="judge-model",
        api_base="https://example.invalid/v1",
        api_key="unit-test-key",
        temperature=0.93,
    )
    router = object()
    for actor in orch.actors.values():
        actor.model_config = cfg
        actor.router = router

    await orch.run()

    analysis = next(ev for ev in events if ev["type"] == "analysis")["analysis"]
    assert analysis["quality"]["game_quality"] == 0.66
    assert analysis["quality"]["scores"][0]["RI"] == 0.8
    assert calls
    assert calls[0]["router"] is router
    assert calls[0]["config"].model == "judge-model"
    assert calls[0]["config"].temperature == 0.0
    assert cfg.temperature == 0.93


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
async def test_game_ended_emitted_once_before_analysis():
    """结束事件只广播一次;赛后 analysis 仍作为复盘事件到达。"""
    orch, events = _build_orchestrator()
    await orch.run()

    ended_indices = [i for i, ev in enumerate(events) if ev["type"] == "game_ended"]
    analysis_indices = [i for i, ev in enumerate(events) if ev["type"] == "analysis"]
    assert len(ended_indices) == 1
    assert len(analysis_indices) == 1
    assert ended_indices[0] < analysis_indices[0]


@pytest.mark.asyncio
async def test_vote_log_matches_vote_cast_events_without_duplicates():
    """每次成功投票只进入一条 analysis vote log,并保留 OSR 摘要。"""
    orch, events = _build_orchestrator()
    await orch.run()

    vote_casts = [ev for ev in events if ev["type"] == "vote_cast"]
    assert vote_casts
    assert len(orch._vote_log) == len(vote_casts)
    assert all(v.get("objective_summary") for v in orch._vote_log)


def test_public_speech_memory_records_only_public_fields():
    """公开发言进入所有存活 agent 记忆,但不写狼人 deception/私有 reasoning。"""
    orch, _events = _build_orchestrator()
    speech = {
        "seat": 2,
        "name": "P2",
        "text": "我怀疑3号这轮在带节奏。",
        "reply_to": 1,
        "accuses": [3],
        "attitudes": {3: "oppose"},
        "deception": "fabrication",
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
        assert "fabrication" not in obs.text
        assert "我是狼所以编的" not in obs.text
        assert "deception" not in obs.metadata
        assert "reasoning" not in obs.metadata
        assert "claim" not in obs.metadata
        assert obs.metadata["speaker_seat"] == 2
        assert obs.metadata["accuses"] == [3]
        assert obs.metadata["attitudes"] == {3: "oppose"}
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
    """后续上下文和实时 speech 事件只带公开字段,不泄漏狼人 deception。"""
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
                    deception="fabrication",
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
        assert "deception" not in speech
        assert "reasoning" not in speech
        assert speech.get("claim") == {"role": "seer", "checked_seat": 3, "result": "wolf"}

    speech_events = [ev for ev in events if ev["type"] == "speech"]
    assert speech_events
    assert all("deception" not in ev for ev in speech_events)

    # Internal analysis still keeps wolf deception intent for research metrics.
    assert any(s.get("deception") == "fabrication" for s in orch._speech_log)
    analysis = next(ev for ev in events if ev["type"] == "analysis")["analysis"]
    assert analysis["dialogue_metrics"]["wolf_deception_count"] > 0


@pytest.mark.asyncio
async def test_analysis_includes_objective_metrics_and_clamped_coordination():
    orch, events = _build_orchestrator()
    await orch.run()

    analysis_event = next(ev for ev in events if ev["type"] == "analysis")
    analysis = analysis_event["analysis"]
    objective = analysis["objective_metrics"]
    dialogue = analysis["dialogue_metrics"]

    assert objective["vote_count"] == len([ev for ev in events if ev["type"] == "vote_cast"])
    assert objective["vote_accuracy_good"] is None or 0.0 <= objective["vote_accuracy_good"] <= 1.0
    assert objective["vote_accuracy_wolf"] is None or 0.0 <= objective["vote_accuracy_wolf"] <= 1.0
    assert objective["osr_summary_rate"] == 1.0
    assert objective["ct_marker_rate"] == 1.0
    assert 0.0 <= dialogue["wolf_coordination"] <= 1.0


@pytest.mark.asyncio
async def test_analysis_includes_parse_metrics_for_lossy_decisions():
    """赛后统计有损 JSON 恢复决策,但不把解析质量塞进 live 事件。"""
    orch, events = _build_orchestrator()
    actor = next(a for a in orch.actors.values() if a.role == Role.WEREWOLF)
    original_speak = actor.decide_speak
    original_vote = actor.decide_vote
    marked = {"speak": False, "vote": False}

    async def lossy_speak(state, player_id, **kw):
        decision = await original_speak(state, player_id, **kw)
        if not marked["speak"]:
            marked["speak"] = True
            return decision.model_copy(update={"parse_failed": True})
        return decision

    async def lossy_vote(state, player_id, **kw):
        decision = await original_vote(state, player_id, **kw)
        if not marked["vote"]:
            marked["vote"] = True
            return decision.model_copy(update={"parse_failed": True})
        return decision

    actor.decide_speak = lossy_speak
    actor.decide_vote = lossy_vote

    await orch.run()

    analysis = next(ev for ev in events if ev["type"] == "analysis")["analysis"]
    metrics = analysis["parse_metrics"]

    assert metrics["decision_count"] > 0
    assert metrics["parse_failed_count"] == 2
    assert metrics["parse_failed_rate"] == pytest.approx(2 / metrics["decision_count"])
    assert metrics["parse_failed_by_action"]["speak"] == 1
    assert metrics["parse_failed_by_action"]["vote"] == 1
    assert metrics["parse_failed_by_phase"]["day"] == 1
    assert metrics["parse_failed_by_phase"]["voting"] == 1
    assert all("parse_failed" not in ev for ev in events if ev["type"] in {"speech", "vote_cast"})


def test_deception_audit_compares_declared_intent_with_listener_shift():
    """欺骗审计独立于狼人自报,并用后验变化估计误导效果。"""
    orch, _events = _build_orchestrator()
    wolf = next(p.seat for p in orch.state.players if p.role == Role.WEREWOLF)
    seer = next(p.seat for p in orch.state.players if p.role == Role.SEER)
    good_targets = [p.seat for p in orch.state.players if p.role != Role.WEREWOLF and p.seat != seer]
    good_speaker, good_target = good_targets[:2]

    orch._speech_log = [
        {
            "seat": good_speaker,
            "day": 1,
            "text": "我先怀疑一个好人测试误指控。",
            "accuses": [good_target],
            "attitudes": {good_target: "oppose"},
        },
        {
            "seat": wolf,
            "day": 1,
            "text": f"{good_target}号这轮很像狼,大家别被他带偏。",
            "deception": "misdirection",
            "accuses": [good_target],
            "attitudes": {good_target: "oppose"},
        },
        {
            "seat": wolf,
            "day": 1,
            "text": f"{seer}号这个预言家跳得太急,查验细节不对。",
            "deception": "distortion",
            "accuses": [seer],
            "attitudes": {seer: "oppose"},
        },
    ]
    orch._posterior_log = [
        {
            "day": 1,
            "trigger": "speech",
            "source_seat": good_speaker,
            "viewer_seat": seer,
            "posterior": {str(good_target): 0.2, str(wolf): 0.3},
            "top_suspects": [{"seat": wolf}],
        },
        {
            "day": 1,
            "trigger": "speech",
            "source_seat": wolf,
            "viewer_seat": seer,
            "posterior": {str(good_target): 0.55, str(wolf): 0.2},
            "evidence_items": [
                {
                    "evidence_id": f"accuse:1:{wolf}:{good_target}:oppose",
                    "type": "accuse",
                    "visibility": "public",
                    "provenance": "today_speech",
                    "day": 1,
                    "source_seat": wolf,
                    "target_seat": good_target,
                    "confidence": 0.55,
                },
            ],
            "posterior_deltas": [
                {
                    "evidence_id": f"accuse:1:{wolf}:{good_target}:oppose",
                    "target_seat": good_target,
                    "delta": 0.35,
                    "after": 0.55,
                    "source_type": "accuse",
                },
            ],
            "top_suspects": [{"seat": good_target}],
        },
        {
            "day": 1,
            "trigger": "speech",
            "source_seat": wolf,
            "viewer_seat": seer,
            "posterior": {str(good_target): 0.50, str(wolf): 0.25},
            "evidence_items": [
                {
                    "evidence_id": f"accuse:1:{wolf}:{seer}:oppose",
                    "type": "accuse",
                    "visibility": "public",
                    "provenance": "today_speech",
                    "day": 1,
                    "source_seat": wolf,
                    "target_seat": seer,
                    "confidence": 0.55,
                },
            ],
            "posterior_deltas": [
                {
                    "evidence_id": f"accuse:1:{wolf}:{seer}:oppose",
                    "target_seat": wolf,
                    "delta": 0.05,
                    "after": 0.25,
                    "source_type": "accuse",
                },
            ],
            "top_suspects": [{"seat": good_target}],
        },
    ]

    audit = orch._deception_audit()

    assert audit["wolf_speech_count"] == 2
    assert audit["declared_deception_count"] == 2
    assert audit["audited_deception_count"] == 2
    assert audit["audited_by_type"]["misdirection"] == 2
    assert audit["audited_by_type"]["distortion"] == 1
    assert audit["declared_vs_audited_agreement"] == 1.0
    assert audit["deception_success_rate"] == 1.0
    assert audit["misdirection_shift_coverage"] == 0.5
    assert audit["unauditable_misdirection_count"] == 1
    assert audit["avg_good_target_suspicion_gain"] == pytest.approx(0.35)
    assert audit["peer_detection_rate"] == 0.5
    assert audit["detected_deception_count"] == 1
    assert audit["peer_detection_opportunity_count"] == 2
    assert audit["evidence_linked_count"] == 2
    assert audit["listener_susceptibility_by_seat"][str(seer)]["misdirected_rate"] == 1.0
    assert audit["listener_susceptibility_by_seat"][str(seer)]["peer_detection_rate"] == 0.5
    assert audit["villager_false_positive_rate"] == 1.0
    successful = next(record for record in audit["records"] if record["successful_misdirection"])
    assert successful["avg_good_target_suspicion_gain"] == pytest.approx(0.35)
    assert successful["evidence_ids"] == [f"accuse:1:{wolf}:{good_target}:oppose"]
    assert successful["posterior_delta_ids"] == [f"accuse:1:{wolf}:{good_target}:oppose"]
    assert successful["evidence_source_types"] == {"accuse": 1}
    assert successful["listener_shifts"] == [
        {
            "viewer_seat": seer,
            "target_good_suspicion_gain": 0.35,
            "speaker_suspicion_gain": -0.1,
            "misdirected": True,
            "detected_speaker": False,
        }
    ]
    assert any(
        record["peer_detection"]["detected"]
        and record["peer_detection"]["detector_seats"] == [seer]
        for record in audit["records"]
    )

    dumped = json.dumps(audit, ensure_ascii=False)
    assert "查验细节" not in dumped
    assert "reasoning" not in dumped
    assert "wolf_caucus" not in dumped


def test_deception_audit_does_not_attribute_vote_shift_to_pk_speech():
    """PK 发言的 before 应使用 vote 后 baseline,不把投票变化算作发言误导。"""
    orch, _events = _build_orchestrator()
    wolf = next(p.seat for p in orch.state.players if p.role == Role.WEREWOLF)
    seer = next(p.seat for p in orch.state.players if p.role == Role.SEER)
    good_target = next(
        p.seat for p in orch.state.players
        if p.role != Role.WEREWOLF and p.seat != seer
    )
    good_speaker = next(
        p.seat for p in orch.state.players
        if p.role != Role.WEREWOLF and p.seat not in {seer, good_target}
    )

    orch._speech_log = [
        {
            "seat": wolf,
            "day": 1,
            "text": f"PK里我继续认为{good_target}号像狼。",
            "deception": "misdirection",
            "accuses": [good_target],
            "attitudes": {good_target: "oppose"},
        },
    ]
    orch._posterior_log = [
        {
            "day": 1,
            "trigger": "speech",
            "source_seat": good_speaker,
            "viewer_seat": seer,
            "posterior": {str(good_target): 0.2, str(wolf): 0.3},
            "top_suspects": [{"seat": wolf}],
        },
        {
            "day": 1,
            "trigger": "vote",
            "source_seat": good_speaker,
            "viewer_seat": seer,
            "posterior": {str(good_target): 0.8, str(wolf): 0.3},
            "top_suspects": [{"seat": good_target}],
        },
        {
            "day": 1,
            "trigger": "pk_speech",
            "source_seat": wolf,
            "viewer_seat": seer,
            "posterior": {str(good_target): 0.8, str(wolf): 0.3},
            "evidence_items": [
                {
                    "evidence_id": f"accuse:pk:{wolf}:{good_target}:oppose",
                    "type": "accuse",
                    "visibility": "public",
                    "provenance": "today_speech",
                    "day": 1,
                    "source_seat": wolf,
                    "target_seat": good_target,
                    "confidence": 0.55,
                },
            ],
            "posterior_deltas": [
                {
                    "evidence_id": f"accuse:pk:{wolf}:{good_target}:oppose",
                    "target_seat": good_target,
                    "delta": 0.0,
                    "after": 0.8,
                    "source_type": "accuse",
                },
            ],
            "top_suspects": [{"seat": good_target}],
        },
    ]

    groups = orch._posterior_speech_shift_groups()
    pk_shift = groups[(1, wolf)][0][0]
    assert pk_shift["previous_trigger"] == "vote"
    assert pk_shift["before"][str(good_target)] == pytest.approx(0.8)

    audit = orch._deception_audit()
    record = audit["records"][0]
    assert audit["deception_success_rate"] == 0.0
    assert audit["misdirection_shift_coverage"] == 1.0
    assert audit["unauditable_misdirection_count"] == 0
    assert audit["avg_good_target_suspicion_gain"] == pytest.approx(0.0)
    assert record["avg_good_target_suspicion_gain"] == pytest.approx(0.0)
    assert record["successful_misdirection"] is False
    assert record["listener_shifts"][0]["target_good_suspicion_gain"] == pytest.approx(0.0)

    metrics = orch._posterior_metrics()
    assert metrics["avg_speech_posterior_shift"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_analysis_includes_posterior_trace_and_metrics():
    """赛后输出可复算的后验轨迹指标,但不携带隐藏发言元数据。"""
    orch, events = _build_orchestrator()
    await orch.run()

    analysis = next(ev for ev in events if ev["type"] == "analysis")["analysis"]
    metrics = analysis["posterior_metrics"]
    trace = analysis["posterior_trace"]

    assert metrics["snapshot_count"] == len(trace)
    assert analysis["posterior_trace_total_count"] == len(trace)
    assert analysis["posterior_trace_truncated"] is False
    assert analysis["posterior_trace_dropped_count"] == 0
    assert metrics["speech_snapshot_count"] > 0
    assert metrics["avg_speech_posterior_shift"] is None or 0.0 <= metrics["avg_speech_posterior_shift"] <= 1.0
    assert metrics["good_final_wolf_suspicion_gap"] is None or -1.0 <= metrics["good_final_wolf_suspicion_gap"] <= 1.0
    assert metrics["good_final_top_suspect_accuracy"] is None or 0.0 <= metrics["good_final_top_suspect_accuracy"] <= 1.0
    assert metrics["herding_index"] is None or 0.0 <= metrics["herding_index"] <= 1.0
    assert metrics["herding_event_count"] >= 0
    assert metrics["correct_herding_rate"] is None or 0.0 <= metrics["correct_herding_rate"] <= 1.0
    assert metrics["wrong_herding_rate"] is None or 0.0 <= metrics["wrong_herding_rate"] <= 1.0
    assert metrics["final_brier_score"] is None or 0.0 <= metrics["final_brier_score"] <= 1.0
    assert metrics["good_final_brier_score"] is None or 0.0 <= metrics["good_final_brier_score"] <= 1.0
    assert metrics["final_log_loss"] is None or metrics["final_log_loss"] >= 0.0
    assert metrics["good_final_log_loss"] is None or metrics["good_final_log_loss"] >= 0.0
    assert metrics["constrained_final_brier_score"] is None or 0.0 <= metrics["constrained_final_brier_score"] <= 1.0
    assert metrics["constrained_good_final_brier_score"] is None or 0.0 <= metrics["constrained_good_final_brier_score"] <= 1.0
    assert metrics["constrained_final_log_loss"] is None or metrics["constrained_final_log_loss"] >= 0.0
    assert metrics["constrained_good_final_log_loss"] is None or metrics["constrained_good_final_log_loss"] >= 0.0
    assert metrics["constrained_calibration_ece"] is None or 0.0 <= metrics["constrained_calibration_ece"] <= 1.0
    assert metrics["calibration_ece"] is None or 0.0 <= metrics["calibration_ece"] <= 1.0
    assert isinstance(metrics["constrained_calibration_bins"], list)
    assert len(metrics["constrained_calibration_bins"]) == 5
    assert all("range" in b and "count" in b for b in metrics["constrained_calibration_bins"])
    assert isinstance(metrics["calibration_bins"], list)
    assert len(metrics["calibration_bins"]) == 5
    assert all("range" in b and "count" in b for b in metrics["calibration_bins"])
    assert any(s["trigger"] == "speech" for s in trace)
    assert any(s["trigger"] == "vote" for s in trace)
    assert all("viewer_seat" in s and "posterior" in s for s in trace)
    assert all("constrained_posterior" in s for s in trace)
    assert all(s["constrained_posterior"] for s in trace)
    assert all("posterior_deltas" in s for s in trace)
    deltas = [d for s in trace for d in s["posterior_deltas"]]
    assert deltas
    assert all("evidence_id" in d and "delta" in d and "target_seat" in d for d in deltas)
    assert all("evidence_items" in s for s in trace)
    assert all("legal_worlds" in s for s in trace)
    assert all("world_count" in s["legal_worlds"] and "wolf_count" in s["legal_worlds"] for s in trace)
    assert any(s["evidence_items"] for s in trace)
    for snap in trace:
        item_ids = {item["evidence_id"] for item in snap["evidence_items"]}
        assert {delta["evidence_id"] for delta in snap["posterior_deltas"]} <= item_ids
        assert all(
            {"evidence_id", "type", "visibility", "provenance", "confidence"} <= set(item)
            for item in snap["evidence_items"]
        )

    dumped = json.dumps(trace, ensure_ascii=False)
    assert "deception" not in dumped
    assert "reasoning" not in dumped


def test_posterior_shift_does_not_cross_day_boundary():
    """发言位移只比较同一天连续 speech 快照,不把夜间信息归因给次日发言。"""
    orch, _events = _build_orchestrator()
    orch._posterior_log = [
        {
            "day": 1,
            "trigger": "speech",
            "viewer_seat": 3,
            "posterior": {"1": 0.1, "2": 0.2},
            "top_suspects": [{"seat": 1}],
        },
        {
            "day": 2,
            "trigger": "speech",
            "viewer_seat": 3,
            "posterior": {"1": 0.9, "2": 0.1},
            "top_suspects": [{"seat": 1}],
        },
        {
            "day": 2,
            "trigger": "speech",
            "viewer_seat": 3,
            "posterior": {"1": 0.7, "2": 0.3},
            "top_suspects": [{"seat": 1}],
        },
    ]

    metrics = orch._posterior_metrics()

    assert metrics["speech_snapshot_count"] == 3
    assert metrics["avg_speech_posterior_shift"] == 0.2


@pytest.mark.asyncio
async def test_speak_order_uses_llm_bid_in_second_round():
    """第二轮发言顺序应由 LLM 生成的 bid 决定(高 bid 先发言)。"""
    orch, events = _build_orchestrator()
    # 给每个 seat 分配固定 bid: seat 3 最高,其余依次
    bids = {1: 1, 2: 2, 3: 4, 4: 1, 5: 3, 6: 1}
    # 每个 seat 的发言计数器:每轮给不同内容,避免被去重过滤
    counters = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for pid, actor in orch.actors.items():
        seat = actor.seat
        orig = actor.decide_speak

        def make_speak(original, seat_bid, seat_num):
            async def _speak(state, player_id, **kw):
                dec = await original(state, player_id, **kw)
                dec.bid = seat_bid
                counters[seat_num] += 1
                # 第 n 次发言带不同序号,确保不被去重判定为重复
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
async def test_bid_only_policy_ignores_mentioned_priority():
    """bid_only ablation 应只按 bid 排序,不让被点名优先盖过更高 bid。"""
    orch, _events = _build_orchestrator(turn_policy="bid_only")
    living_pids = [pid for pid, actor in orch.actors.items() if actor.seat in {1, 2}]

    for pid, actor in orch.actors.items():
        if actor.seat not in {1, 2}:
            continue

        async def _speak(state, player_id, *, _seat=actor.seat, **kw):
            bid = 4
            return Decision(action=AgentAction.SPEAK, speech=f"{_seat}号", bid=bid)

        actor.decide_speak = _speak

    tie_seed_speeches = [{"seat": 9, "text": f"seed{i}"} for i in range(6)]

    bid_only = await orch._bid_order_by_llm(
        living_pids,
        today_speeches=tie_seed_speeches,
        spoke_this_round=set(),
        mentioned_seats={1},
        use_reply_priority=False,
    )
    bid_reply = await orch._bid_order_by_llm(
        living_pids,
        today_speeches=tie_seed_speeches,
        spoke_this_round=set(),
        mentioned_seats={1},
        use_reply_priority=True,
    )

    assert [orch.actors[pid].seat for pid in bid_only[:2]] == [2, 1]
    assert [orch.actors[pid].seat for pid in bid_reply[:2]] == [1, 2]


@pytest.mark.asyncio
async def test_fixed_round_robin_policy_uses_fixed_order_without_caucus():
    """fixed_round_robin ablation 不启用 day caucus,且每轮按 seat 顺序发言。"""
    orch, events = _build_orchestrator(turn_policy="fixed_round_robin")
    orch.state.phase = Phase.DAY
    orch.state.day = 1
    orch.max_speak_rounds = 2
    caucus_calls = 0

    async def _caucus():
        nonlocal caucus_calls
        caucus_calls += 1

    async def _no_vote(**kw):
        return None

    orch._werewolf_day_caucus = _caucus
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
                bid=0,
            )

        actor.decide_speak = _speak

    await orch._run_day()

    speeches = [ev for ev in events if ev["type"] == "speech"]
    assert caucus_calls == 0
    assert [ev["seat"] for ev in speeches] == [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]


def test_debate_process_metrics_summarize_public_debate_shape():
    """debate_process_metrics 只用公开 speech 字段,输出可复算过程形态。"""
    orch, _events = _build_orchestrator(turn_policy="bid_reply")
    orch._speech_log = [
        {"seat": 1, "day": 1, "bid": 1, "text": "我跳预言家", "claim": {"role": "seer"}},
        {"seat": 2, "day": 1, "bid": 4, "text": "我反对1", "reply_to": 1, "accuses": [1], "attitudes": {1: "oppose"}},
        {"seat": 3, "day": 1, "bid": 2, "text": "支持2", "attitudes": {2: "support"}},
        {"seat": 2, "day": 1, "bid": 3, "text": "反对3", "accuses": [3], "attitudes": {3: "oppose"}},
        {"seat": 3, "day": 1, "bid": 2, "text": "反对2", "accuses": [2], "attitudes": {2: "oppose"}},
    ]

    metrics = orch._debate_process_metrics()

    assert metrics["turn_policy"] == "bid_reply"
    assert metrics["caucus_enabled"] == 0
    assert metrics["uses_bid_order"] == 1
    assert metrics["uses_reply_priority"] == 1
    assert metrics["speech_count"] == 5
    assert metrics["speaker_count"] == 3
    assert metrics["speaker_concentration"] == 0.4
    assert metrics["bid_entropy"] is not None and 0.0 <= metrics["bid_entropy"] <= 1.0
    assert metrics["avg_bid"] == pytest.approx(2.4)
    assert metrics["reply_count"] == 1
    assert metrics["avg_reply_latency"] == 1.0
    assert metrics["claim_count"] == 1
    assert metrics["claim_challenged_count"] == 1
    assert metrics["claim_challenged_rate"] == 1.0
    assert metrics["accuse_target_count"] == 3
    assert metrics["top_accuse_target_share"] == pytest.approx(1 / 3, abs=0.001)
    assert metrics["support_loop_count"] == 0
    assert metrics["opposition_loop_count"] == 1


def test_collusion_audit_detects_public_wolf_alignment_without_raw_text():
    """collusion_audit 只输出结构化合谋信号,不保存原文/推理/狼队私聊。"""
    orch, _events = _build_orchestrator()
    wolf_a, wolf_b = sorted(p.seat for p in orch.state.players if p.role == Role.WEREWOLF)
    target_good = next(p.seat for p in orch.state.players if p.role != Role.WEREWOLF and p.seat != 3)
    shared_text = f"{target_good}号跳太急不要信{target_good}号这查杀太巧"
    orch._speech_log = [
        {
            "seat": wolf_a,
            "day": 1,
            "bid": 3,
            "text": shared_text,
            "accuses": [target_good],
            "attitudes": {wolf_b: "support", target_good: "oppose"},
            "deception": "misdirection",
            "reasoning": "我是狼所以保队友",
            "wolf_caucus": f"私聊推{target_good}号",
        },
        {
            "seat": wolf_b,
            "day": 1,
            "bid": 3,
            "text": shared_text,
            "accuses": [target_good],
            "attitudes": {wolf_a: "support", target_good: "oppose"},
            "deception": "misdirection",
            "reasoning": f"我是狼所以帮{wolf_a}号",
        },
        {
            "seat": 3,
            "day": 1,
            "bid": 1,
            "text": "我更相信查验信息",
            "accuses": [wolf_a],
            "attitudes": {target_good: "support", wolf_a: "oppose"},
        },
    ]
    orch._posterior_log = [
        {
            "day": 1,
            "phase": "day",
            "trigger": "phase_started",
            "viewer_seat": 3,
            "posterior": {str(wolf_a): 0.30, str(wolf_b): 0.30, str(target_good): 0.20},
        },
        {
            "day": 1,
            "phase": "day",
            "trigger": "speech",
            "source_seat": wolf_a,
            "viewer_seat": 3,
            "posterior": {str(wolf_a): 0.36, str(wolf_b): 0.30, str(target_good): 0.37},
            "evidence_items": [
                {
                    "evidence_id": "ev-public-1",
                    "type": "accuse",
                    "visibility": "public",
                    "day": 1,
                    "source_seat": wolf_a,
                    "target_seat": target_good,
                },
                {
                    "evidence_id": "ev-private-ignored",
                    "type": "seer_result",
                    "visibility": "private",
                    "day": 1,
                    "source_seat": wolf_a,
                    "target_seat": target_good,
                },
            ],
            "posterior_deltas": [
                {"evidence_id": "ev-public-1", "target_seat": target_good, "delta": 0.17},
                {"evidence_id": "ev-private-ignored", "target_seat": target_good, "delta": 0.99},
            ],
        },
        {
            "day": 1,
            "phase": "day",
            "trigger": "speech",
            "source_seat": wolf_b,
            "viewer_seat": 3,
            "posterior": {str(wolf_a): 0.36, str(wolf_b): 0.35, str(target_good): 0.51},
            "evidence_items": [
                {
                    "evidence_id": "ev-public-2",
                    "type": "attitude",
                    "visibility": "public",
                    "day": 1,
                    "source_seat": wolf_b,
                    "target_seat": target_good,
                },
            ],
            "posterior_deltas": [
                {"evidence_id": "ev-public-2", "target_seat": target_good, "delta": 0.14},
            ],
        },
    ]

    audit = orch._collusion_audit()

    assert audit["wolf_speech_count"] == 2
    assert audit["wolf_pair_count"] == 1
    assert audit["active_wolf_pair_count"] == 1
    assert audit["wolf_to_wolf_support_count"] == 2
    assert audit["mutual_support_pair_count"] == 1
    assert audit["shared_good_target_count"] == 1
    assert audit["shared_good_target_speaker_coverage"] == 1.0
    assert audit["narrative_overlap_pair_count"] == 1
    assert audit["avg_narrative_overlap"] == 1.0
    assert audit["coordinated_pressure_count"] == 3
    assert audit["avg_shared_target_suspicion_gain"] == pytest.approx(0.155, abs=0.001)
    assert audit["avg_colluder_suspicion_gain"] == pytest.approx(0.055, abs=0.001)
    assert audit["evidence_linked_count"] >= 1
    assert audit["pair_listener_shift_sample_count"] == 2
    assert audit["avg_pair_target_suspicion_gain"] == pytest.approx(0.155, abs=0.001)
    assert audit["pair_target_misdirected_rate"] == 1.0
    assert audit["windowed_relay_count"] == 1
    assert audit["avg_windowed_relay_latency"] == 1.0
    assert audit["avg_relay_target_suspicion_gain"] == pytest.approx(0.14, abs=0.001)
    assert audit["relay_target_misdirected_rate"] == 1.0
    assert audit["deception_linked_pair_count"] == 1

    shared = next(record for record in audit["records"] if record["type"] == "shared_good_target")
    assert shared["day"] == 1
    assert shared["target_good_seat"] == target_good
    assert shared["wolf_seats"] == [wolf_a, wolf_b]
    assert shared["evidence_ids"] == ["ev-public-1", "ev-public-2"]
    assert "ev-private-ignored" not in shared["evidence_ids"]
    relay = next(record for record in audit["records"] if record["type"] == "windowed_relay")
    assert relay["day"] == 1
    assert relay["wolf_seats"] == [wolf_a, wolf_b]
    assert relay["lead_wolf_seat"] == wolf_a
    assert relay["follow_wolf_seat"] == wolf_b
    assert relay["relay_latency"] == 1
    assert relay["shared_good_targets"] == [target_good]
    assert relay["follower_supports_lead"] is True
    assert relay["avg_target_suspicion_gain"] == pytest.approx(0.14, abs=0.001)
    pair_key = f"{wolf_a}-{wolf_b}"
    pair = audit["pair_listener_susceptibility_by_pair"][pair_key]
    assert pair["wolf_seats"] == [wolf_a, wolf_b]
    assert pair["active_days"] == [1]
    assert pair["shared_good_target_count"] == 1
    assert pair["wolf_to_wolf_support_count"] == 2
    assert pair["mutual_support_pair_count"] == 1
    assert pair["narrative_overlap_pair_count"] == 1
    assert pair["coordinated_pressure_count"] == 3
    assert pair["target_shift_sample_count"] == 2
    assert pair["target_misdirected_rate"] == 1.0
    assert pair["avg_target_suspicion_gain"] == pytest.approx(0.155, abs=0.001)
    assert pair["colluder_shift_sample_count"] == 2
    assert pair["avg_colluder_suspicion_gain"] == pytest.approx(0.055, abs=0.001)
    assert pair["windowed_relay_count"] == 1
    assert pair["avg_windowed_relay_latency"] == 1.0
    assert pair["avg_relay_target_suspicion_gain"] == pytest.approx(0.14, abs=0.001)
    assert pair["relay_target_misdirected_rate"] == 1.0
    assert pair["deception_record_count"] == 2
    assert pair["successful_deception_record_count"] == 2
    assert pair["audited_deception_types"]["misdirection"] == 2
    assert pair["evidence_ids"] == ["ev-public-1", "ev-public-2"]
    assert "ev-private-ignored" not in pair["evidence_ids"]

    serialized = json.dumps(audit["records"], ensure_ascii=False)
    serialized += json.dumps(audit["pair_listener_susceptibility_by_pair"], ensure_ascii=False)
    assert f"{target_good}号跳太急" not in serialized
    assert "我是狼" not in serialized
    assert f"私聊推{target_good}号" not in serialized
