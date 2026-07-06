"""Agent 执行器 —— 决策编排 + 清洗校验 + 异步反思。

承 no-fallback-design 铁律(ARCHITECTURE.md §3.1, §9):
- 每个 AI 决策必须来自真实 LLM 调用,绝不伪造。
- 失败走深度重试(解析失败标记后由上层编排器重试);彻底失败抛 AgentDecisionError。
- 清洗:目标非法时按真实意图就近修正(选最近合法目标)或落入合法 SKIP,但带 skip_reason 透明审计。
- reasoning 私有保存(上帝/复盘可见,不广播)。
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable

from ..game.models import GameState
from ..game.roles import Role
from ..llm.models import ModelConfig
from ..llm.router import LLMError, LLMRouter
from .memory import AgentMemory
from .prompts import (
    assign_persona,
    build_messages,
    last_words_instruction,
    night_action_instruction,
    parse_suspicion,
    reflection_instruction,
    render_observation,
    role_prompt,
    speak_instruction,
    vote_instruction,
)
from .schemas import AgentAction, AgentThinking, Decision
from .information import attach_today_speeches, build_observation

logger = logging.getLogger(__name__)

DECISION_MAX_ATTEMPTS = 5
REFLECTION_MAX_ATTEMPTS = 2
RETRY_BASE_DELAY_SECONDS = 0.05
RETRY_MAX_DELAY_SECONDS = 0.8


class AgentDecisionError(RuntimeError):
    """Agent 决策彻底失败(真实 LLM 重试耗尽)。由编排器决定 _legal_skip。"""


class AgentActor:
    """单个 agent 的执行器。

    持有 memory + persona + model_config。每次 decide() 产生一个 Decision。
    所有 decide_* 方法都是真实 LLM 调用。
    """

    def __init__(
        self,
        *,
        seat: int,
        name: str,
        role: Role,
        model_config: ModelConfig,
        router: LLMRouter,
        rng: random.Random | None = None,
        is_human: bool = False,
        on_human_request: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self.seat = seat
        self.name = name
        self.role = role
        self.model_config = model_config
        self.router = router
        self.is_human = is_human
        self.on_human_request = on_human_request
        self.memory = AgentMemory(seat=seat, role=role.value)
        self.rng = rng or random.Random(seat * 104729)
        persona_name, persona_desc = assign_persona(seat, self.rng)
        self.persona_name = persona_name
        self.persona_desc = persona_desc
        # 人类玩家操作队列(人机混合模式)
        self.human_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    # ------------------------------------------------------------------
    # 记忆维护(编排器在状态推进时调用)
    # ------------------------------------------------------------------
    def observe_event(self, day: int, phase: str, kind: str, text: str, **meta: Any) -> None:
        self.memory.observe(day, phase, kind, text, **meta)

    def record_claim(self, seat: int, day: int, claim: dict[str, Any]) -> None:
        self.memory.record_claim(seat, day, claim)

    def apply_suspicion(self, suspicion: dict[int, float]) -> None:
        self.memory.update_trust(suspicion)

    def set_trust(self, seat: int, value: float) -> None:
        self.memory.set_trust(seat, value)

    # ------------------------------------------------------------------
    # 人类玩家操作(人机混合)
    # ------------------------------------------------------------------
    async def _wait_human_action(
        self,
        action_type: str,
        context: dict[str, Any],
        *,
        state: GameState,
        timeout: float | None = None,
    ) -> Decision:
        """向人类玩家请求操作并在队列等待结果。

        超时后返回透明 SKIP,承 no-fallback-design(不伪造决策)。
        """
        from ..config import HUMAN_TIMEOUT

        timeout = timeout or HUMAN_TIMEOUT
        request_payload = {
            "type": "human_action_request",
            "seat": self.seat,
            "action_type": action_type,
            "context": context,
            "timeout": timeout,
        }
        if self.on_human_request:
            try:
                await self.on_human_request(request_payload)
            except Exception:  # noqa: BLE001
                pass
        try:
            data = await asyncio.wait_for(self.human_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return Decision(
                action=AgentAction.SKIP,
                skip_reason="human_timeout",
                reasoning=f"人类玩家{self.seat}号在{timeout}秒内未操作,自动跳过。",
            )
        return self._parse_human_action(data, action_type, state)

    def _parse_human_action(self, data: dict[str, Any], action_type: str, state: GameState) -> Decision:
        """把前端 human_action 消息解析为 Decision,并将 target_seat 转为 player_id。"""
        if not isinstance(data, dict):
            return Decision(action=AgentAction.SKIP, skip_reason="human_action_invalid")

        user_action = str(data.get("action", "")).lower()
        target_seat = data.get("target_seat")
        speech = data.get("speech")
        bid = data.get("bid")

        if user_action == "skip":
            return Decision(action=AgentAction.SKIP, skip_reason="human_skip")

        action_map = {
            "night_kill": AgentAction.NIGHT_KILL,
            "kill": AgentAction.NIGHT_KILL,
            "hunter_shot": AgentAction.NIGHT_KILL,
            "see": AgentAction.SEE,
            "save": AgentAction.SAVE,
            "poison": AgentAction.POISON,
            "guard": AgentAction.GUARD,
            "speak": AgentAction.SPEAK,
            "vote": AgentAction.VOTE,
            "last_words": AgentAction.LAST_WORDS,
        }
        mapped = action_map.get(user_action)
        if mapped is None:
            # 如果没传 action,按请求类型推断
            mapped = action_map.get(action_type)
        if mapped is None:
            return Decision(action=AgentAction.SKIP, skip_reason="human_action_unknown")

        target_id = None
        if target_seat is not None:
            try:
                seat = int(float(str(target_seat).replace("号", "").strip()))
            except (ValueError, TypeError):
                return Decision(action=AgentAction.SKIP, skip_reason="human_action_invalid_target")
            for player in state.players:
                if player.seat == seat and player.alive:
                    target_id = player.id
                    break

        return Decision(
            action=mapped,
            target_id=target_id,
            speech=str(speech) if speech is not None else None,
            bid=int(bid) if bid is not None else None,
            reasoning="人类玩家操作",
        )

    # ------------------------------------------------------------------
    # 决策入口
    # ------------------------------------------------------------------
    async def decide_night_action(
        self,
        state: GameState,
        player_id: str,
        *,
        today_speeches: list[dict] | None = None,
        requested_action: str | None = None,
        human_context: dict[str, Any] | None = None,
        max_attempts: int = DECISION_MAX_ATTEMPTS,
    ) -> Decision:
        """夜间行动决策。"""
        if self.is_human:
            ctx = {
                "phase": state.phase.value,
                "day": state.day,
                "role": self.role.value,
                "requested_action": requested_action,
            }
            if human_context:
                ctx.update(human_context)
            return await self._wait_human_action(requested_action or "night_action", ctx, state=state)
        obs = build_observation(
            state,
            player_id,
            rng=self.rng,
            available_actions=[requested_action] if requested_action else None,
        )
        if today_speeches:
            attach_today_speeches(obs, today_speeches)

        extras = self._role_extras()
        if self.role == Role.WEREWOLF:
            role_text = role_prompt(self.role.value, teammates=obs.my_teammates, extras=extras)
        else:
            role_text = role_prompt(self.role.value, extras=extras)

        instruction = night_action_instruction(obs, self.role.value, requested_action=requested_action)
        system, messages, _ = build_messages(
            persona_name=self.persona_name,
            persona_desc=self.persona_desc,
            role_text=role_text,
            observation_text=render_observation(obs, self.memory.render_for_prompt()),
            action_instruction=instruction,
        )

        raw = await self._call_with_retry(messages, system, max_attempts=max_attempts)
        return self._sanitize_night(raw, obs, requested_action=requested_action)

    async def decide_speak(
        self,
        state: GameState,
        player_id: str,
        *,
        today_speeches: list[dict] | None = None,
        max_attempts: int = DECISION_MAX_ATTEMPTS,
    ) -> Decision:
        """白天发言决策(含竞价 bid)。"""
        if self.is_human:
            ctx = {"phase": state.phase.value, "day": state.day, "today_speeches": today_speeches or []}
            return await self._wait_human_action("speak", ctx, state=state)
        obs = build_observation(state, player_id, rng=self.rng)
        if today_speeches:
            attach_today_speeches(obs, today_speeches)

        extras = self._role_extras()
        if self.role == Role.WEREWOLF:
            role_text = role_prompt(self.role.value, teammates=obs.my_teammates, extras=extras)
        else:
            role_text = role_prompt(self.role.value, extras=extras)

        instruction = speak_instruction(obs)
        system, messages, _ = build_messages(
            persona_name=self.persona_name,
            persona_desc=self.persona_desc,
            role_text=role_text,
            observation_text=render_observation(obs, self.memory.render_for_prompt()),
            action_instruction=instruction,
        )

        raw = await self._call_with_retry(messages, system, max_attempts=max_attempts)
        decision = self._sanitize_speak(raw, obs)
        # 记录公开声称
        if decision.claim:
            self.record_claim(self.seat, state.day, decision.claim)
        return decision

    async def decide_wolf_caucus(
        self,
        state: GameState,
        player_id: str,
        *,
        max_attempts: int = DECISION_MAX_ATTEMPTS,
    ) -> dict[str, Any] | None:
        """狼队白天党团会议(方向C):白天发言前狼队私聊商定推人目标+口径。

        复用夜间 _werewolf_deliberation 的私聊拓扑(仅狼人可见的信息隔离通道)。
        返回 {target_seat, strategy, reasoning} 供 orchestrator 聚合共识。
        这是狼人的"主张",不直接落地——orchestrator 聚合后作为私有观察注入
        每个狼人记忆,狼人发言时自主决定是否照做(harness 不写发言,守 no-fallback)。
        """
        if self.is_human or self.role != Role.WEREWOLF:
            return None
        obs = build_observation(state, player_id, rng=self.rng)
        extras = self._role_extras()
        role_text = role_prompt(self.role.value, teammates=obs.my_teammates, extras=extras)
        alive_good = [s for s in obs.alive_seats if s != obs.my_seat
                      and not any(t.get("seat") == s for t in obs.my_teammates)]
        instruction = (
            f"现在是白天发言前的狼队党团会议(仅狼人可见的私聊,好人听不到)。\n"
            f"你是{obs.my_seat}号(狼人),队友:{[t.get('seat') for t in obs.my_teammates]}。\n"
            f"存活好人座位:{alive_good or '(无,已胜)'}\n"
            f"请和队友商定今天白天统一推谁出局(target_seat),以及统一口径(strategy,如'集体指控他行为异常'/'分散别跟太紧避免暴露抱团')。\n"
            f"返回 JSON:\n"
            f'{{"thought":"你的党团会议发言(私聊,写清为什么推这个目标/怎么配合,越细越好)",'
            f'"target_seat":目标座位号(int,填一个好人),'
            f'"strategy":"统一口径(一句话,供队友白天各自执行)"}}'
        )
        system, messages, _ = build_messages(
            persona_name=self.persona_name,
            persona_desc=self.persona_desc,
            role_text=role_text,
            observation_text=render_observation(obs, self.memory.render_for_prompt()),
            action_instruction=instruction,
        )
        raw = await self._call_with_retry(messages, system, max_attempts=max_attempts)
        target_seat = self._extract_int(raw, "target_seat")
        strategy = self._extract_str(raw, "strategy") or ""
        reasoning = self._extract_str(raw, "thought") or ""
        # 过滤非法目标(必须是活的好人,不能是自己/队友)
        valid = set(alive_good)
        if target_seat not in valid:
            target_seat = None
        if not target_seat and not strategy:
            return None
        return {"target_seat": target_seat, "strategy": strategy, "reasoning": reasoning}

    async def decide_vote(
        self,
        state: GameState,
        player_id: str,
        *,
        today_speeches: list[dict] | None = None,
        pk_candidates: list[str] | None = None,
        max_attempts: int = DECISION_MAX_ATTEMPTS,
    ) -> Decision:
        """投票决策。PK 时 pk_candidates(player_id 列表)限制投票目标(且不可投自己)。"""
        if self.is_human:
            ctx = {"phase": state.phase.value, "day": state.day, "today_speeches": today_speeches or [],
                   "pk_candidates": pk_candidates}
            return await self._wait_human_action("vote", ctx, state=state)
        # PK 候选:从 player_id 列表转成座位列表
        pk_seats: list[int] | None = None
        if pk_candidates:
            pk_seats = [state.get_player(pid).seat for pid in pk_candidates]
        obs = build_observation(state, player_id, rng=self.rng, vote_targets=pk_seats, in_pk=bool(pk_seats))
        if today_speeches:
            attach_today_speeches(obs, today_speeches)

        extras = self._role_extras()
        if self.role == Role.WEREWOLF:
            role_text = role_prompt(self.role.value, teammates=obs.my_teammates, extras=extras)
        else:
            role_text = role_prompt(self.role.value, extras=extras)

        instruction = vote_instruction(obs)
        system, messages, _ = build_messages(
            persona_name=self.persona_name,
            persona_desc=self.persona_desc,
            role_text=role_text,
            observation_text=render_observation(obs, self.memory.render_for_prompt()),
            action_instruction=instruction,
        )

        raw = await self._call_with_retry(messages, system, max_attempts=max_attempts)
        return self._sanitize_vote(raw, obs)

    async def decide_last_words(
        self,
        state: GameState,
        player_id: str,
        reason: str,
        *,
        max_attempts: int = DECISION_MAX_ATTEMPTS,
    ) -> Decision:
        """遗言。"""
        if self.is_human:
            return await self._wait_human_action("last_words", {"reason": reason, "day": state.day}, state=state)
        obs = build_observation(state, player_id, rng=self.rng)
        extras = self._role_extras()
        role_text = role_prompt(
            self.role.value,
            teammates=obs.my_teammates if self.role == Role.WEREWOLF else None,
            extras=extras,
        )
        instruction = last_words_instruction(reason)
        system, messages, _ = build_messages(
            persona_name=self.persona_name,
            persona_desc=self.persona_desc,
            role_text=role_text,
            observation_text=render_observation(obs, self.memory.render_for_prompt()),
            action_instruction=instruction,
        )
        raw = await self._call_with_retry(messages, system, max_attempts=max_attempts)
        return self._sanitize_last_words(raw)

    async def reflect(
        self,
        state: GameState,
        player_id: str,
        *,
        max_attempts: int = REFLECTION_MAX_ATTEMPTS,
    ) -> str | None:
        """轮末反思(异步,不阻塞游戏推进)。失败返回 None,不致命。"""
        obs = build_observation(state, player_id, rng=self.rng)
        instruction = reflection_instruction(state.phase, state.day)
        system, messages, _ = build_messages(
            persona_name=self.persona_name,
            persona_desc=self.persona_desc,
            role_text=role_prompt(self.role.value, extras=self._role_extras()),
            observation_text=render_observation(obs, self.memory.render_for_prompt()),
            action_instruction=instruction,
        )
        try:
            raw = await self._call_with_retry(messages, system, max_attempts=max_attempts)
        except AgentDecisionError:
            return None
        insight = self._extract_str(raw, "insight") or ""
        suspicion = parse_suspicion(raw.get("suspicion"), obs.alive_seats, self.seat)
        if insight:
            self.memory.reflect(state.day, state.phase, insight)
        if suspicion:
            self.apply_suspicion(suspicion)
        return insight or None

    # ------------------------------------------------------------------
    # LLM 调用 + 重试
    # ------------------------------------------------------------------
    async def _call_with_retry(self, messages: list[dict], system: str, *, max_attempts: int) -> dict[str, Any]:
        """真实 LLM 调用,解析失败重试(承 no-fallback-design)。

        - LLMError(网络/网关)由 router 内部已重试,这里捕获后继续重试(深度重试)。
        - 解析失败(JSON 拿到但字段缺失)在这里重试 max_attempts 次。
        - 彻底失败抛 AgentDecisionError,绝不返回伪造 dict。
        """
        last_err: Exception | None = None
        attempts = max(1, max_attempts)
        for attempt in range(attempts):
            try:
                allow_lossy = attempt == attempts - 1
                raw = await self.router.complete_json(
                    messages,
                    self.model_config,
                    system=system,
                    allow_lossy=allow_lossy,
                    include_parse_metadata=allow_lossy,
                )
                if not isinstance(raw, dict):
                    raise ValueError(f"LLM 返回非对象: {type(raw)}")
                if raw.get("_parse_lossy"):
                    logger.warning(
                        "agent %s(%s) 使用有损 JSON 恢复结果 attempt=%d/%d method=%s",
                        self.seat,
                        self.role.value,
                        attempt + 1,
                        attempts,
                        raw.get("_parse_method"),
                    )
                return raw
            except LLMError as err:
                last_err = err
                logger.warning(
                    "agent %s(%s) LLM调用失败 attempt=%d/%d: %s",
                    self.seat, self.role.value, attempt + 1, attempts, err,
                )
                if attempt < attempts - 1:
                    await self._sleep_before_retry(attempt)
                    continue
                break
            except (ValueError, KeyError, TypeError) as err:
                last_err = err
                logger.warning(
                    "agent %s(%s) 解析失败 attempt=%d/%d: %s",
                    self.seat, self.role.value, attempt + 1, attempts, err,
                )
                if attempt < attempts - 1:
                    await self._sleep_before_retry(attempt)
                    continue
                break
        raise AgentDecisionError(
            f"agent {self.seat}({self.role.value}) 决策彻底失败({max_attempts}次重试): {last_err}"
        )

    async def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(RETRY_MAX_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
        jitter = self.rng.uniform(0.0, delay * 0.25)
        await asyncio.sleep(delay + jitter)

    # ------------------------------------------------------------------
    # 清洗校验(承 no-fallback-design:就近修正或合法 SKIP,非伪造)
    # ------------------------------------------------------------------
    def _sanitize_night(self, raw: dict, obs, *, requested_action: str | None = None) -> Decision:
        thought = self._extract_str(raw, "thought") or ""
        parse_failed = self._parse_lossy(raw)
        suspicion = parse_suspicion(raw.get("suspicion"), obs.alive_seats, self.seat)
        if suspicion:
            self.apply_suspicion(suspicion)

        target_seat = self._extract_int(raw, "target_seat")
        # 女巫特殊处理
        if self.role == Role.WITCH:
            return self._sanitize_witch(raw, obs, thought, suspicion, requested_action=requested_action)

        action = self._night_action_for_request(requested_action)
        if action is None:
            action_map = {
                Role.WEREWOLF: AgentAction.NIGHT_KILL,
                Role.SEER: AgentAction.SEE,
                Role.GUARD: AgentAction.GUARD,
                Role.DOCTOR: AgentAction.SAVE,
            }
            action = action_map.get(self.role, AgentAction.SKIP)

        target_id = self._resolve_target(target_seat, obs)
        if action != AgentAction.SKIP and target_id is None:
            # 目标非法:真实意图无法落地,落入合法不行动(透明标记)
            return Decision(
                action=AgentAction.SKIP,
                reasoning=thought,
                suspicion=suspicion if suspicion else None,
                skip_reason="night_target_unresolved",
                parse_failed=parse_failed,
            )
        return Decision(
            action=action,
            target_id=target_id,
            reasoning=thought,
            suspicion=suspicion if suspicion else None,
            parse_failed=parse_failed,
        )

    def _sanitize_witch(
        self,
        raw: dict,
        obs,
        thought: str,
        suspicion: dict[int, float],
        *,
        requested_action: str | None = None,
    ) -> Decision:
        """女巫:可能救人(save)+ 毒人(poison),但引擎逐个处理。优先毒人(更主动)。

        编排器会分两次询问女巫(先救后毒),这里根据 raw 决定本步动作。
        实际编排:女巫夜间被问两次——一次决定救,一次决定毒。此处用 use_save/use_poison。
        """
        # 默认本步为 save(编排器先问救)
        parse_failed = self._parse_lossy(raw)
        use_save = bool(raw.get("use_save", False))
        use_poison = bool(raw.get("use_poison", False))
        # 引擎通过 available_actions 告知当前问的是哪步;这里简化:若 use_poison 且有 poison_target 返回毒
        poison_seat = self._extract_int(raw, "poison_target")
        save_seat = self._extract_int(raw, "save_target") or self._extract_int(raw, "target_seat")

        if requested_action == "save":
            if use_poison and poison_seat and not (use_save or save_seat):
                return Decision(
                    action=AgentAction.SKIP,
                    reasoning=thought,
                    suspicion=suspicion if suspicion else None,
                    skip_reason="requested_action_mismatch",
                    parse_failed=parse_failed,
                )
            if use_save or save_seat is not None:
                target_id = self._resolve_target(save_seat, obs, allow_self=True)
                if target_id:
                    return Decision(
                        action=AgentAction.SAVE,
                        target_id=target_id,
                        reasoning=thought,
                        suspicion=suspicion if suspicion else None,
                        parse_failed=parse_failed,
                    )
            return Decision(
                action=AgentAction.SKIP,
                reasoning=thought,
                suspicion=suspicion if suspicion else None,
                skip_reason="witch_save_skipped",
                parse_failed=parse_failed,
            )

        if requested_action == "poison":
            poison_target = poison_seat or self._extract_int(raw, "target_seat")
            if use_save and save_seat is not None and not (use_poison or poison_target):
                return Decision(
                    action=AgentAction.SKIP,
                    reasoning=thought,
                    suspicion=suspicion if suspicion else None,
                    skip_reason="requested_action_mismatch",
                    parse_failed=parse_failed,
                )
            if use_poison or poison_target is not None:
                target_id = self._resolve_target(poison_target, obs)
                if target_id:
                    return Decision(
                        action=AgentAction.POISON,
                        target_id=target_id,
                        reasoning=thought,
                        suspicion=suspicion if suspicion else None,
                        parse_failed=parse_failed,
                    )
            return Decision(
                action=AgentAction.SKIP,
                reasoning=thought,
                suspicion=suspicion if suspicion else None,
                skip_reason="witch_poison_skipped",
                parse_failed=parse_failed,
            )

        # 优先毒(若指示毒且有目标)
        if use_poison and poison_seat:
            target_id = self._resolve_target(poison_seat, obs)
            if target_id:
                return Decision(
                    action=AgentAction.POISON,
                    target_id=target_id,
                    reasoning=thought,
                    suspicion=suspicion if suspicion else None,
                    parse_failed=parse_failed,
                )
        if use_save and save_seat is not None:
            target_id = self._resolve_target(save_seat, obs, allow_self=True)
            if target_id:
                return Decision(
                    action=AgentAction.SAVE,
                    target_id=target_id,
                    reasoning=thought,
                    suspicion=suspicion if suspicion else None,
                    parse_failed=parse_failed,
                )
        return Decision(
            action=AgentAction.SKIP,
            reasoning=thought,
            suspicion=suspicion if suspicion else None,
            skip_reason="witch_no_action",
            parse_failed=parse_failed,
        )

    @staticmethod
    def _night_action_for_request(requested_action: str | None) -> AgentAction | None:
        return {
            "night_kill": AgentAction.NIGHT_KILL,
            "kill": AgentAction.NIGHT_KILL,
            "hunter_shot": AgentAction.NIGHT_KILL,
            "see": AgentAction.SEE,
            "save": AgentAction.SAVE,
            "poison": AgentAction.POISON,
            "guard": AgentAction.GUARD,
        }.get(requested_action or "")

    def _sanitize_speak(self, raw: dict, obs) -> Decision:
        thought = self._extract_str(raw, "thought") or ""
        parse_failed = self._parse_lossy(raw)
        bid = self._extract_int(raw, "bid")
        if bid is None:
            bid = 1
        speech = self._extract_str(raw, "speech") or ""
        if not speech.strip():
            speech = "(沉默)"
        suspicion = parse_suspicion(raw.get("suspicion"), obs.alive_seats, self.seat)
        if suspicion:
            self.apply_suspicion(suspicion)
        claim = self._sanitize_claim(raw.get("claim"), obs)
        # 结构化对话关系(方向A):reply_to 回应谁,accuses 指控谁。过滤非法座位。
        reply_to = self._extract_int(raw, "reply_to")
        if reply_to is not None and (
            reply_to == obs.my_seat or not any(s["seat"] == reply_to for s in obs.seats)
        ):
            reply_to = None
        accuses = self._extract_int_list(raw, "accuses", obs)
        # 二阶 ToM 态度网络(方向B):解析显式 attitudes,归一化到 support/oppose/neutral。
        # validator 已在 schemas 层做 mode=before 清洗,这里只负责提取与过滤非法座位。
        attitudes = self._extract_attitudes(raw.get("attitudes"), obs)
        # 欺骗策略结构化(DR/PS 提升):狼人显式声明本回合欺骗手段,归一化到 4 分类/none。
        deception = self._extract_deception(raw.get("deception"))
        return Decision(
            action=AgentAction.SPEAK,
            speech=speech,
            bid=bid,
            reasoning=thought,
            suspicion=suspicion if suspicion else None,
            claim=claim,
            reply_to=reply_to,
            accuses=accuses or None,
            attitudes=attitudes or None,
            deception=deception,
            parse_failed=parse_failed,
        )

    @staticmethod
    def _sanitize_claim(raw: Any, obs) -> dict[str, Any] | None:
        """校验 claim schema。仅接受规范的预言家跳身份声明,剔除乱格式自证。

        合法 claim: {"role":"seer","checked_seat":<活人int>,"result":"wolf"|"village"}
        非预言家的"我是村民"等不需要 claim(在 speech 里说即可),避免污染 claims 记录与前端。
        """
        if not isinstance(raw, dict):
            return None
        role = str(raw.get("role", "")).strip().lower()
        if role != "seer":
            # 只接受预言家跳身份 claim;其他(村民/狼人自证)一律剔除
            return None
        checked_seat = raw.get("checked_seat")
        try:
            checked_seat = int(checked_seat)
        except (ValueError, TypeError):
            return None
        result = str(raw.get("result", "")).strip().lower()
        if result not in ("wolf", "village"):
            return None
        # checked_seat 必须是存在的座位(非自己);允许已死者(预言家可能验过夜里死者)
        if checked_seat == obs.my_seat:
            return None
        if not any(s["seat"] == checked_seat for s in obs.seats):
            return None
        return {"role": "seer", "checked_seat": checked_seat, "result": result}

    def _sanitize_vote(self, raw: dict, obs) -> Decision:
        thought = self._extract_str(raw, "thought") or ""
        parse_failed = self._parse_lossy(raw)
        objective_summary = self._extract_str(raw, "objective_summary") or ""
        target_seat = self._extract_int(raw, "target_seat")
        suspicion = parse_suspicion(raw.get("suspicion"), obs.alive_seats, self.seat)
        if suspicion:
            self.apply_suspicion(suspicion)
        # PK 限制:投票目标必须在 PK 候选名单内
        pk_seats = set(obs.vote_targets) if obs.vote_targets else set()
        target_id = self._resolve_target(target_seat, obs)
        if pk_seats:
            # 若 LLM 投的不在 PK 候选内,就近修正到候选中怀疑度最高的(真实意图合理落地)
            tgt_seat = next((s["seat"] for s in obs.seats if s["id"] == target_id), None)
            if tgt_seat is None or tgt_seat not in pk_seats:
                target_id = None
                if suspicion:
                    # 只有 LLM 给出显式怀疑度时,才按真实意图在 PK 候选中修正。
                    cand = [s for s in pk_seats if s != self.seat]
                    cand.sort(key=lambda s: suspicion.get(s, 0.0), reverse=True)
                    if cand:
                        target_id = self._resolve_target(cand[0], obs)
        else:
            if target_id is None:
                # 投票必须有目标。就近选怀疑度最高的活人(真实意图的合理落地)
                if suspicion:
                    top = max(suspicion.items(), key=lambda kv: kv[1])
                    target_id = self._resolve_target(top[0], obs)
        if target_id is None:
            return Decision(
                action=AgentAction.SKIP,
                reasoning=thought,
                suspicion=suspicion if suspicion else None,
                skip_reason="vote_target_unresolved",
                objective_summary=objective_summary or None,
                parse_failed=parse_failed,
            )
        return Decision(
            action=AgentAction.VOTE,
            target_id=target_id,
            reasoning=thought,
            suspicion=suspicion if suspicion else None,
            objective_summary=objective_summary or None,
            parse_failed=parse_failed,
        )

    def _sanitize_last_words(self, raw: dict) -> Decision:
        thought = self._extract_str(raw, "thought") or ""
        speech = self._extract_str(raw, "speech") or "(无遗言)"
        return Decision(
            action=AgentAction.LAST_WORDS,
            speech=speech,
            reasoning=thought,
            parse_failed=self._parse_lossy(raw),
        )

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_lossy(raw: dict[str, Any]) -> bool:
        """LLM JSON 是否经过有损恢复。用于 Decision.parse_failed 透明审计。"""
        return bool(raw.get("_parse_lossy"))

    def _role_extras(self) -> dict[str, str]:
        """从记忆提取角色专属状态渲染到 prompt。"""
        extras: dict[str, str] = {}
        if self.role == Role.SEER:
            results = [m for m in self.memory.observations if m.kind == "seer_result"]
            if results:
                # 结构化渲染查验清单,让预言家清楚知道自己验过谁
                lines = []
                for m in results:
                    # metadata 含 target_seat/team(由 _push_night_results_to_memory 的 ev.payload 注入)
                    meta = m.metadata or {}
                    seat = meta.get("target_seat") or meta.get("seat")
                    team = meta.get("team")
                    tag = "狼人" if team == "werewolves" or team == "wolf" else ("好人" if team else "?")
                    lines.append(f"第{m.day}夜 验 {seat}号 → {tag}")
                extras["seer_results"] = "; ".join(lines)
            else:
                extras["seer_results"] = "尚无查验"
        elif self.role == Role.WITCH:
            saved = any(m.kind == "witch_save_used" for m in self.memory.observations)
            poisoned = any(m.kind == "witch_poison_used" for m in self.memory.observations)
            extras["witch_state"] = f"解药{'已用' if saved else '未用'},毒药{'已用' if poisoned else '未用'}"
        elif self.role == Role.GUARD:
            last_guard = [
                m for m in self.memory.observations if m.kind == "guard_target"
            ]
            extras["guard_state"] = (
                f"上一夜守了{last_guard[-1].text}" if last_guard else "上一夜未守护"
            )
        return extras

    def _resolve_target(self, seat: int | None, obs, *, allow_self: bool = False) -> str | None:
        """把座位号解析为 player_id。非法时返回 None(就近修正在上层)。"""
        if seat is None or seat == 0:
            return None
        for s in obs.seats:
            if s["seat"] == seat and s["alive"]:
                if not allow_self and seat == obs.my_seat:
                    return None
                return s["id"]
        return None

    @staticmethod
    def _extract_str(raw: dict, key: str) -> str | None:
        val = raw.get(key)
        if val is None:
            return None
        s = str(val).strip()
        return s or None

    @staticmethod
    def _extract_int(raw: dict, key: str) -> int | None:
        val = raw.get(key)
        if val is None or val == "":
            return None
        try:
            # 容忍 "3号" / 3.0 / "3"
            s = str(val).replace("号", "").strip()
            return int(float(s))
        except (ValueError, TypeError):
            return None

    def _extract_int_list(self, raw: dict, key: str, obs) -> list[int] | None:
        """解析座位号列表(如 accuses),过滤自己/不存在的座位。返回去重后的列表或 None。"""
        val = raw.get(key)
        if val is None:
            return None
        if not isinstance(val, (list, tuple)):
            # 容忍单个 int/"3" 形式
            single = self._extract_int(raw, key)
            return [single] if single is not None else None
        valid_seats = {s["seat"] for s in obs.seats}
        seen: set[int] = set()
        result: list[int] = []
        for item in val:
            try:
                seat = int(float(str(item).replace("号", "").strip()))
            except (ValueError, TypeError):
                continue
            if seat > 0 and seat != obs.my_seat and seat in valid_seats and seat not in seen:
                seen.add(seat)
                result.append(seat)
        return result or None

    def _extract_attitudes(self, raw: Any, obs) -> dict[int, str] | None:
        """解析二阶 ToM 显式态度(方向B),归一化到 support/oppose/neutral。
        容忍 "3号"/3.0 键 + 中文立场词,过滤自己/不存在的座位。"""
        if not isinstance(raw, dict):
            return None
        normalize = {
            "support": "support", "支持": "support", "帮腔": "support", "信任": "support", "agree": "support",
            "oppose": "oppose", "反对": "oppose", "指控": "oppose", "怀疑": "oppose", "disagree": "oppose",
            "neutral": "neutral", "中立": "neutral", "无": "neutral",
        }
        valid_seats = {s["seat"] for s in obs.seats}
        result: dict[int, str] = {}
        for k, val in raw.items():
            try:
                seat = int(float(str(k).replace("号", "").strip()))
            except (ValueError, TypeError):
                continue
            if seat <= 0 or seat == obs.my_seat or seat not in valid_seats:
                continue
            stance = normalize.get(str(val).strip().lower()) or normalize.get(str(val).strip())
            if stance is None:
                rv = str(val).strip()
                if any(ch in rv for ch in "反控疑敌"):
                    stance = "oppose"
                elif any(ch in rv for ch in "支帮信友同"):
                    stance = "support"
                else:
                    stance = "neutral"
            result[seat] = stance
        return result or None

    def _extract_deception(self, raw: Any) -> str | None:
        """解析欺骗策略(DR/PS 提升),归一化到 omission/distortion/fabrication/misdirection/none。
        容忍中文/大小写。狼人填策略,好人填 none。"""
        if raw is None:
            return None
        s = str(raw).strip().lower()
        norm = {
            "omission": "omission", "遗漏": "omission", "省略": "omission",
            "distortion": "distortion", "扭曲": "distortion", "曲解": "distortion",
            "fabrication": "fabrication", "捏造": "fabrication", "编造": "fabrication",
            "misdirection": "misdirection", "误导": "misdirection", "转移": "misdirection",
            "none": "none", "无": "none", "诚实": "none", "真话": "none",
        }
        return norm.get(s)

    def thinking_summary(self, decision: Decision, *, verbose: bool = False) -> AgentThinking:
        """从 Decision 生成思考摘要(推给前端,经整理)。

        verbose=False(默认,保公平):摘要取前 120 字,不暴露完整隐藏推理。
        verbose=True(上帝/复盘/研究):summary 与 reasoning 均填完整 thought,
        暴露 agent 的分析、欺骗算计、手段——用于多 agent 对抗过程可观察。
        """
        reasoning = (decision.reasoning or "").strip()
        if verbose:
            summary = reasoning
        else:
            summary = reasoning[:120] + ("..." if len(reasoning) > 120 else "")
        suspicion_top: list[dict[str, Any]] = []
        if decision.suspicion:
            items = sorted(decision.suspicion.items(), key=lambda kv: kv[1], reverse=True)[:3]
            suspicion_top = [{"seat": int(k), "suspicion": round(v, 2)} for k, v in items]
        return AgentThinking(
            seat=self.seat,
            action=decision.action.value,
            summary=summary,
            suspicion_top=suspicion_top,
            bid=decision.bid,
            reasoning=reasoning if verbose else None,
        )
