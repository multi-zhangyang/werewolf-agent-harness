"""新版编排器 v2 —— 目标:顶级多agent狼人杀对局体验。

核心改进(相对 v1):
1. 狼队真正私聊合谋:多狼并发内部讨论,达成一致 kill 目标。
2. 女巫分阶段救/毒:先告知狼刀目标决定救不救,再问毒不毒。
3. 守卫连守限制 + 同守同救正确结算。
4. 猎人开枪使用专用决策(不用 vote)。
5. PK 流程:平票后仅涉事者额外发言,再投一次。
6. 遗言与猎人开枪顺序正确(先遗言,再开枪,再结算胜负)。
7. 所有死亡事件正确入记忆(含死亡原因)并广播。
8. 信息隔离:夜间私有事件(查验/狼队友)只入对应记忆。
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections import Counter
from typing import Any, Awaitable, Callable

from ..config import (
    AGENT_DECISION_TIMEOUT,
    AGENT_DECISION_TIMEOUT_BY_PHASE,
    AGENT_PHASE_DEADLINE,
    AGENT_PHASE_DEADLINE_BY_PHASE,
)
from ..agent.actor import AgentActor, AgentDecisionError
from ..agent.information import attach_today_speeches, build_observation
from ..agent.schemas import AgentAction, AgentThinking, Decision
from ..game.models import (
    DeathReason,
    Event,
    EventVisibility,
    GameState,
    NightAction,
    NightActionType,
    Phase,
    Vote,
)
from ..game.roles import Role, Team, default_role_deck
from ..game.rules import RulesEngine
from ..llm.models import ModelConfig
from ..llm.router import LLMRouter

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]
ThinkingCallback = Callable[[dict[str, Any]], Awaitable[None]]

TURN_POLICIES = (
    "fixed_round_robin",
    "bid_only",
    "bid_reply",
    "bid_reply_caucus",
)
DEFAULT_TURN_POLICY = "bid_reply_caucus"

_CT_MARKERS = (
    "如果", "若", "假如", "则", "否则", "相比", "权衡", "风险", "收益",
    "代价", "反之", "一旦", "选择", "if", "then", "otherwise", "risk", "benefit",
)


def _bigrams(text: str) -> set[str]:
    """中文/通用 bigram 集合(字符级,容忍标点)。"""
    cleaned = "".join(c for c in text if c.isalnum())
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _is_duplicate_speech(
    text: str, today_speeches: list[dict[str, Any]], my_seat: int, *, relax_self: bool = False
) -> tuple[bool, str]:
    """检测发言是否重复。返回 (是否重复, 原因)。

    - 与自己今日已有发言 Jaccard≥0.55 → 重复(自说自话)
    - 与他人今日发言 Jaccard≥0.80 → 重复(逐字抄)
    - relax_self=True(反驳场景):放宽自重复阈值到 0.80,因为反驳本就会复述对方论点,
      避免误杀被点名者的辩护。
    """
    bg = _bigrams(text)
    if not bg:
        return False, ""
    self_threshold = 0.80 if relax_self else 0.55
    for s in today_speeches:
        if not s.get("text"):
            continue
        other_bg = _bigrams(s["text"])
        sim = _jaccard(bg, other_bg)
        if s.get("seat") == my_seat and sim >= self_threshold:
            return True, f"与你本日已发言高度重复(sim={sim:.2f})"
        if s.get("seat") != my_seat and sim >= 0.80:
            return True, f"与{s.get('seat')}号发言近乎逐字重复(sim={sim:.2f})"
    return False, ""


def _public_speech_memory_text(speech: dict[str, Any]) -> str:
    """Render a public speech observation for agent memory without hidden fields.

    Keep this strictly public: no role truth, no private reasoning, no wolf caucus.
    """
    parts: list[str] = []
    reply_to = speech.get("reply_to")
    if reply_to:
        parts.append(f"回应{reply_to}号")
    accuses = speech.get("accuses") or []
    if accuses:
        parts.append("指控" + ",".join(f"{a}号" for a in accuses))
    attitudes = speech.get("attitudes")
    if isinstance(attitudes, dict) and attitudes:
        label = {"support": "支持", "oppose": "反对", "neutral": "中立"}
        att = ",".join(f"{k}号{label.get(str(v), str(v))}" for k, v in attitudes.items())
        parts.append(f"态度:{att}")
    meta = f"({'/'.join(parts)})" if parts else ""
    return f"{speech.get('seat')}号{meta}说:{speech.get('text', '')}"


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _rounded_mean(values: list[float]) -> float | None:
    return round(_mean(values), 3) if values else None


def _normalized_entropy(values: list[Any]) -> float | None:
    if not values:
        return None
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    total = len(values)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return round(entropy / math.log(len(counts)), 3)


def _posterior_values(snapshot: dict[str, Any], key: str = "posterior") -> dict[str, float]:
    posterior = snapshot.get(key) or {}
    result: dict[str, float] = {}
    if not isinstance(posterior, dict):
        return result
    for seat, value in posterior.items():
        try:
            result[str(seat)] = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    return result


def _top_suspect_seat(snapshot: dict[str, Any]) -> int | None:
    top = snapshot.get("top_suspects") or []
    if not isinstance(top, list) or not top:
        return None
    first = top[0]
    if not isinstance(first, dict):
        return None
    return _as_int(first.get("seat"))


def _brier(records: list[tuple[float, int]]) -> float | None:
    if not records:
        return None
    return round(sum((p - y) ** 2 for p, y in records) / len(records), 4)


def _log_loss(records: list[tuple[float, int]], *, eps: float = 1e-6) -> float | None:
    if not records:
        return None
    total = 0.0
    for p, y in records:
        p = min(1.0 - eps, max(eps, p))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return round(total / len(records), 4)


def _calibration_bins(
    records: list[tuple[float, int]],
    *,
    bin_count: int = 5,
) -> tuple[list[dict[str, Any]], float | None]:
    if not records:
        return ([], None)
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bin_count)]
    for p, y in records:
        idx = min(bin_count - 1, max(0, int(p * bin_count)))
        buckets[idx].append((p, y))

    bins: list[dict[str, Any]] = []
    ece = 0.0
    total = len(records)
    for idx, bucket in enumerate(buckets):
        lo = idx / bin_count
        hi = (idx + 1) / bin_count
        if bucket:
            avg_pred = sum(p for p, _y in bucket) / len(bucket)
            wolf_rate = sum(y for _p, y in bucket) / len(bucket)
            ece += (len(bucket) / total) * abs(avg_pred - wolf_rate)
            bins.append({
                "range": [round(lo, 2), round(hi, 2)],
                "count": len(bucket),
                "avg_prediction": round(avg_pred, 4),
                "wolf_rate": round(wolf_rate, 4),
            })
        else:
            bins.append({
                "range": [round(lo, 2), round(hi, 2)],
                "count": 0,
                "avg_prediction": None,
                "wolf_rate": None,
            })
    return (bins, round(ece, 4))


class GameOrchestratorV2:
    """新版游戏编排器。"""

    def __init__(
        self,
        *,
        state: GameState,
        actors: dict[str, AgentActor],
        deck: list[Role] | None = None,
        rng: random.Random | None = None,
        on_event: EventCallback | None = None,
        on_thinking: ThinkingCallback | None = None,
        max_speak_rounds: int = 6,
        verbose_thinking: bool = False,
        turn_policy: str = DEFAULT_TURN_POLICY,
        decision_timeout: float | None = None,
        decision_timeouts: dict[str, float] | None = None,
        phase_deadline: float | None = None,
        phase_deadlines: dict[str, float] | None = None,
    ) -> None:
        if turn_policy not in TURN_POLICIES:
            raise ValueError(f"unknown turn_policy={turn_policy!r}; expected one of {TURN_POLICIES}")
        self.state = state
        self.actors = actors
        self.deck = deck or default_role_deck(len(state.players))
        self.rng = rng or random.Random()
        self.on_event = on_event
        self.on_thinking = on_thinking
        self.max_speak_rounds = max_speak_rounds
        self.verbose_thinking = verbose_thinking
        self.turn_policy = turn_policy
        self.decision_timeout = AGENT_DECISION_TIMEOUT if decision_timeout is None else decision_timeout
        default_decision_timeouts = (
            AGENT_DECISION_TIMEOUT_BY_PHASE
            if decision_timeout is None
            else {phase: self.decision_timeout for phase in AGENT_DECISION_TIMEOUT_BY_PHASE}
        )
        self.decision_timeouts = {
            **default_decision_timeouts,
            **(decision_timeouts or {}),
        }
        self.phase_deadline = AGENT_PHASE_DEADLINE if phase_deadline is None else phase_deadline
        default_phase_deadlines = (
            AGENT_PHASE_DEADLINE_BY_PHASE
            if phase_deadline is None
            else {phase: self.phase_deadline for phase in AGENT_PHASE_DEADLINE_BY_PHASE}
        )
        self.phase_deadlines = {
            **default_phase_deadlines,
            **(phase_deadlines or {}),
        }
        self.aborted = False
        self._failed_events: list[dict[str, Any]] = []
        self._last_round_silent = 0
        # 对局质量评分用的事件收集(轻量,只存评分需要的子集)
        self._speech_log: list[dict[str, Any]] = []
        self._vote_log: list[dict[str, Any]] = []
        self._thinking_log: list[dict[str, Any]] = []
        self._parse_decisions: list[dict[str, Any]] = []
        self._decision_failures: list[dict[str, Any]] = []
        # Per-agent visible RolePosterior snapshots for post-game trajectory metrics.
        # Analysis-only: never fed back into live prompts or decisions.
        self._posterior_log: list[dict[str, Any]] = []
        self._game_ended_emitted = False
        # 为人类玩家注册请求回调
        for actor in self.actors.values():
            actor.on_human_request = self._on_human_request

    async def _on_human_request(self, payload: dict[str, Any]) -> None:
        """把人类操作请求广播给前端(仅该 seat 的 play 模式可见)。"""
        await self._emit({
            **payload,
            "day": self.state.day,
            "phase": self.state.phase.value,
        })

    def _normalize_target(self, target_id: str | None) -> str | None:
        """把 target_id 归一化为 player_id(支持 seat 字符串/数字)。"""
        if target_id is None:
            return None
        try:
            self.state.get_player(target_id)
            return target_id
        except KeyError:
            pass
        try:
            seat = int(target_id)
            for player in self.state.players:
                if player.seat == seat and player.alive:
                    return player.id
        except (ValueError, TypeError):
            pass
        return None

    def _seat_to_pid(self, seat: int) -> str | None:
        """座位号 → player_id(无论死活)。供被提及者观察记录用。"""
        for player in self.state.players:
            if player.seat == seat:
                return player.id
        return None

    def _record_public_speech_memory(self, speech: dict[str, Any]) -> None:
        """Write one public speech into every living agent's memory.

        This fixes the evidence-chain gap where only the speaker remembered their
        own non-claim speech. It does not expose hidden role truth or private
        reasoning; it only mirrors the public table talk already observable.
        """
        text = _public_speech_memory_text(speech)
        for pid, actor in self.actors.items():
            if not self.state.get_player(pid).alive:
                continue
            actor.observe_event(
                int(speech.get("day") or self.state.day),
                "day",
                "speech",
                text,
                speaker_seat=speech.get("seat"),
                reply_to=speech.get("reply_to"),
                accuses=speech.get("accuses"),
                attitudes=speech.get("attitudes"),
            )

    def _record_posterior_snapshot(
        self,
        *,
        trigger: str,
        today_speeches: list[dict[str, Any]] | None = None,
        source_seat: int | None = None,
    ) -> None:
        """Record each living agent's visible evidence posterior for analysis.

        This is a post-game measurement hook. It rebuilds the same sanitized
        AgentObservation each live agent would receive, extracts compact
        posterior values, and stores them for trajectory metrics. It does not
        mutate memory, state, prompts, or decisions.
        """
        speeches = today_speeches or []
        phase = str(self.state.phase)
        for pid, actor in self.actors.items():
            if not self.state.get_player(pid).alive:
                continue
            obs = build_observation(
                self.state,
                pid,
                rng=random.Random(self.state.day * 1009 + actor.seat),
            )
            attach_today_speeches(obs, speeches)
            graph = obs.evidence_graph or {}
            posterior = graph.get("role_posterior") or {}
            compact = {
                str(seat): float((data or {}).get("werewolf_suspicion", 0.5))
                for seat, data in posterior.items()
            }
            compact_constrained = {
                str(seat): float((data or {}).get("constrained_werewolf_suspicion"))
                for seat, data in posterior.items()
                if (data or {}).get("constrained_werewolf_suspicion") is not None
            }
            deltas = graph.get("posterior_deltas") or []
            compact_deltas = [
                {
                    "target_seat": _as_int(delta.get("target_seat")),
                    "delta": float(delta.get("delta", 0.0)),
                    "after": float(delta.get("after", 0.5)),
                    "evidence_id": str(delta.get("evidence_id", "")),
                    "source_type": str(delta.get("source_type", "")),
                    "reason": str(delta.get("reason", ""))[:160],
                }
                for delta in deltas[:40]
                if isinstance(delta, dict)
            ]
            evidence_items = graph.get("evidence_items") or []
            compact_evidence = [
                {
                    "evidence_id": str(item.get("evidence_id", "")),
                    "type": str(item.get("type", "")),
                    "visibility": str(item.get("visibility", "")),
                    "provenance": str(item.get("provenance", "")),
                    "day": _as_int(item.get("day")),
                    "phase": str(item.get("phase", "")),
                    "source_seat": _as_int(item.get("source_seat")),
                    "target_seat": _as_int(item.get("target_seat")),
                    "confidence": float(item.get("confidence", 0.0)),
                }
                for item in evidence_items[:80]
                if isinstance(item, dict)
            ]
            legal_worlds = graph.get("legal_worlds") or {}
            self._posterior_log.append({
                "day": self.state.day,
                "phase": phase,
                "trigger": trigger,
                "source_seat": source_seat,
                "viewer_seat": actor.seat,
                "posterior": compact,
                "constrained_posterior": compact_constrained,
                "legal_worlds": legal_worlds,
                "evidence_items": compact_evidence,
                "posterior_deltas": compact_deltas,
                "top_suspects": graph.get("top_suspects") or [],
            })

    @staticmethod
    def _decision_action_value(action: Any) -> str:
        return str(getattr(action, "value", action))

    def _record_consumed_decision(self, actor: Any, decision: Any, *, phase: str) -> None:
        if not isinstance(decision, Decision):
            return
        self._parse_decisions.append({
            "day": self.state.day,
            "phase": phase,
            "seat": getattr(actor, "seat", None),
            "action": self._decision_action_value(decision.action),
            "parse_failed": bool(getattr(decision, "parse_failed", False)),
            "skip_reason": getattr(decision, "skip_reason", None),
        })

    def _decision_timeout_for(self, phase: str) -> float:
        return float(self.decision_timeouts.get(phase, self.decision_timeout))

    def _phase_deadline_for(self, phase: str) -> float:
        return float(self.phase_deadlines.get(phase, self.phase_deadline))

    def _start_phase_deadline(self, phase: str) -> float | None:
        seconds = self._phase_deadline_for(phase)
        if seconds <= 0:
            return None
        return time.monotonic() + seconds

    def _phase_deadline_error(
        self,
        actor: Any,
        *,
        phase: str,
        action: str,
        when: str,
    ) -> AgentDecisionError:
        seconds = self._phase_deadline_for(phase)
        seat = getattr(actor, "seat", None)
        err = AgentDecisionError(
            f"{phase}/{action} phase deadline exhausted {when} after {seconds:.1f}s"
            + (f" (seat={seat})" if seat is not None else "")
        )
        setattr(err, "timeout", True)
        setattr(err, "timeout_seconds", seconds)
        setattr(err, "phase_deadline_exhausted", True)
        setattr(err, "error_type", "PhaseDeadlineExceeded")
        return err

    @staticmethod
    def _close_unstarted_awaitable(awaitable: Awaitable[Any]) -> None:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()

    async def _with_decision_timeout(
        self,
        actor: Any,
        phase: str,
        action: str,
        awaitable: Awaitable[Any],
        *,
        phase_deadline: float | None = None,
    ) -> Any:
        """Apply a harness-level wall-clock limit to one real agent decision.

        Timeout is not a fallback decision. It becomes AgentDecisionError so the
        existing transparent failure/skip paths can handle it without inventing
        speech, votes, or targets.
        """
        timeout = self._decision_timeout_for(phase)
        deadline_limited = False
        if phase_deadline is not None:
            remaining = phase_deadline - time.monotonic()
            if remaining <= 0:
                self._close_unstarted_awaitable(awaitable)
                raise self._phase_deadline_error(
                    actor, phase=phase, action=action, when="before decision start"
                )
            if timeout <= 0:
                timeout = remaining
                deadline_limited = True
            elif remaining < timeout:
                timeout = remaining
                deadline_limited = True
        if timeout <= 0:
            return await awaitable
        try:
            return await asyncio.wait_for(awaitable, timeout=timeout)
        except asyncio.TimeoutError as err:
            seat = getattr(actor, "seat", None)
            if deadline_limited:
                phase_err = self._phase_deadline_error(
                    actor, phase=phase, action=action, when="during decision"
                )
                raise phase_err from err
            timeout_err = AgentDecisionError(
                f"{phase}/{action} decision timeout after {timeout:.1f}s"
                + (f" (seat={seat})" if seat is not None else "")
            )
            setattr(timeout_err, "timeout", True)
            setattr(timeout_err, "timeout_seconds", timeout)
            raise timeout_err from err

    def _agent_decision_failure_event(
        self,
        actor: Any | None,
        *,
        phase: str,
        action: str | None,
        err: BaseException,
        seat: int | None = None,
        prefix: str | None = None,
    ) -> dict[str, Any]:
        """Build a transparent no-fallback failure event for one actor call.

        Runtime errors and provider/actor failure messages may contain raw
        model output or request details. Public events expose structure only;
        internally generated timeout/deadline errors keep their timing text.
        """
        error_type = str(getattr(err, "error_type", type(err).__name__))
        where = f"{phase}/{action}" if action else phase
        is_public_timing_error = bool(
            getattr(err, "timeout", False)
            or getattr(err, "phase_deadline_exhausted", False)
        )
        if isinstance(err, AgentDecisionError) and is_public_timing_error:
            reason = str(err) or error_type
        else:
            reason = f"{error_type} during {where}"
        if prefix:
            reason = f"{prefix}:{reason}"
        payload: dict[str, Any] = {
            "type": "agent_decision_failed",
            "seat": seat if seat is not None else getattr(actor, "seat", None),
            "phase": phase,
            "reason": reason,
            "error_type": error_type,
        }
        if action:
            payload["action"] = action
        if bool(getattr(err, "timeout", False)) or "timeout" in reason.lower():
            payload["timeout"] = True
            timeout_seconds = getattr(err, "timeout_seconds", None)
            if timeout_seconds is not None:
                payload["timeout_seconds"] = timeout_seconds
        if not isinstance(err, AgentDecisionError):
            logger.error(
                "agent decision call failed(seat=%s phase=%s action=%s type=%s)",
                payload.get("seat"), phase, action, error_type,
                exc_info=(type(err), err, err.__traceback__),
            )
        return payload

    async def run(self) -> GameState:
        await self._emit({"type": "phase_started", "phase": "setup", "day": 0, "message": "角色分配完成"})
        await self._notify_role_assigned()
        await self._broadcast_phase(Phase.NIGHT)

        while self.state.phase != Phase.ENDED and not self.aborted:
            if self.state.phase == Phase.NIGHT:
                await self._run_night()
            elif self.state.phase == Phase.DAY:
                await self._run_day()
            elif self.state.phase == Phase.VOTING:
                await self._run_voting()

        await self._run_analysis()
        return self.state

    # ------------------------------------------------------------------
    # 夜晚
    # ------------------------------------------------------------------
    async def _run_night(self) -> None:
        day = self.state.day
        await self._emit({"type": "phase_started", "phase": "night", "day": day,
                          "message": f"第{day}天夜晚降临,请闭眼。"})
        night_deadline = self._start_phase_deadline("night")

        living_pids = {p.id for p in self.state.living_players()}

        # 1) 预言家查验
        await self._night_role_actions(Role.SEER, [NightActionType.SEE], phase_deadline=night_deadline)
        # 2) 守卫守护
        await self._night_role_actions(Role.GUARD, [NightActionType.GUARD], phase_deadline=night_deadline)
        # 3) 狼队合谋击杀
        await self._werewolf_deliberation(phase_deadline=night_deadline)
        # 4) 女巫:先救
        await self._witch_save_phase(phase_deadline=night_deadline)
        # 5) 女巫:再毒
        await self._witch_poison_phase(phase_deadline=night_deadline)

        # 结算
        self.state = RulesEngine.resolve_night(self.state)
        await self._push_night_results_to_memory()
        for ev in self._failed_events:
            await self._emit(ev)
        self._failed_events.clear()

        deaths = self.state.night_deaths
        await self._emit({
            "type": "night_resolved",
            "day": day,
            "deaths": deaths,
            "message": self._last_event_message("night_deaths"),
        })

        await self._reflect_all()

        if self.state.phase == Phase.ENDED:
            return
        await self._broadcast_phase(Phase.DAY)

    async def _night_role_actions(
        self,
        role: Role,
        allowed: list[NightActionType],
        *,
        phase_deadline: float | None = None,
    ) -> None:
        pids = [pid for pid, a in self.actors.items()
                if self.state.get_player(pid).alive and a.role == role]
        requested_action = "night_kill" if allowed and allowed[0] == NightActionType.KILL else (allowed[0].value if allowed else None)
        tasks = [
            self._with_decision_timeout(
                self.actors[pid], "night", f"{role.value}_action",
                self.actors[pid].decide_night_action(self.state, pid, requested_action=requested_action),
                phase_deadline=phase_deadline,
            )
            for pid in pids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for pid, res in zip(pids, results):
            actor = self.actors[pid]
            if isinstance(res, Exception):
                self._failed_events.append(self._agent_decision_failure_event(
                    actor,
                    phase="night",
                    action=f"{role.value}_action",
                    err=res,
                ))
                continue
            self._record_consumed_decision(actor, res, phase="night")
            await self._emit_thinking(actor, res)
            self._submit_safe(pid, res, allowed)

    async def _werewolf_deliberation(self, *, phase_deadline: float | None = None) -> None:
        """狼队私聊合谋:每个狼人提出想杀目标,多数决;平票随机。"""
        wolf_entries = [(pid, self.actors[pid]) for pid, a in self.actors.items()
                        if self.state.get_player(pid).alive and a.role == Role.WEREWOLF]
        if not wolf_entries:
            return

        tasks = [
            self._with_decision_timeout(
                actor, "night", "werewolf_deliberation",
                actor.decide_night_action(self.state, pid, requested_action="night_kill"),
                phase_deadline=phase_deadline,
            )
            for pid, actor in wolf_entries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        proposals: list[tuple[str, Decision, str]] = []
        for (pid, actor), res in zip(wolf_entries, results):
            if isinstance(res, Exception):
                self._failed_events.append(self._agent_decision_failure_event(
                    actor,
                    phase="night",
                    action="werewolf_deliberation",
                    err=res,
                ))
                continue
            self._record_consumed_decision(actor, res, phase="night")
            await self._emit_thinking(actor, res)
            if isinstance(res, Decision) and res.is_skip:
                continue
            if isinstance(res, Decision) and res.action == AgentAction.NIGHT_KILL and res.target_id:
                target_id = self._normalize_target(res.target_id)
                if not target_id:
                    continue
                proposals.append((pid, res, target_id))
                # 狼队友之间共享提案(私聊)
                for pid2, actor2 in wolf_entries:
                    actor2.observe_event(self.state.day, "night", "wolf_discuss",
                                         f"队友{actor.seat}号提议击杀{self.state.get_player(target_id).seat}号")

        if not proposals:
            return
        targets = [target_id for _, _d, target_id in proposals]
        tally = Counter(targets)
        top_count = tally.most_common(1)[0][1]
        tied = [t for t, c in tally.items() if c == top_count]
        chosen = self.rng.choice(tied)
        first_pid = proposals[0][0]
        self._submit_explicit(first_pid, NightActionType.KILL, chosen)

        # 狼人记忆:统一击杀目标
        for pid, actor in wolf_entries:
            actor.observe_event(self.state.day, "night", "wolf_kill_chosen",
                                f"狼队决定击杀{self.state.get_player(chosen).seat}号")

    async def _werewolf_day_caucus(self, *, phase_deadline: float | None = None) -> None:
        """狼队白天党团会议(方向C):白天发言前狼队私聊商定推人目标+口径。

        复用夜间 _werewolf_deliberation 的私聊拓扑(仅狼人可见的信息隔离通道)。
        弱协同:仅白天开始时1次私聊,不强制后续发言(harness 不写发言,守 no-fallback)。
        流程:每个狼人 LLM 提议 target_seat+strategy → 多数决聚共识 → 注入每个狼人
        私有记忆(wolf_caucus 事件,好人看不到)→ 狼人白天发言时自主决定是否照做。
        好人侧用方向B态度网络识别狼人抱团平衡。
        """
        wolf_entries = [(pid, self.actors[pid]) for pid, a in self.actors.items()
                        if self.state.get_player(pid).alive and a.role == Role.WEREWOLF]
        if len(wolf_entries) < 2:
            # 单狼或无狼:无需党团会议(单狼照自己判断即可)
            return

        tasks = [
            self._with_decision_timeout(
                actor, "day", "wolf_caucus",
                actor.decide_wolf_caucus(self.state, pid),
                phase_deadline=phase_deadline,
            )
            for pid, actor in wolf_entries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        proposals: list[tuple[int, str]] = []  # (target_seat, strategy)
        strategy_votes: list[str] = []
        for (pid, actor), res in zip(wolf_entries, results):
            if isinstance(res, Exception):
                await self._emit(self._agent_decision_failure_event(
                    actor,
                    phase="day",
                    action="wolf_caucus",
                    err=res,
                    prefix="党团会议",
                ))
                continue
            if not isinstance(res, dict):
                continue
            tgt = res.get("target_seat")
            strat = res.get("strategy") or ""
            reasoning = str(res.get("reasoning") or "").strip()
            if reasoning:
                await self._emit_thinking_payload(AgentThinking(
                    seat=actor.seat,
                    action="wolf_caucus",
                    summary=reasoning if self.verbose_thinking else reasoning[:120] + ("..." if len(reasoning) > 120 else ""),
                    reasoning=reasoning if self.verbose_thinking else None,
                ))
            # god 模式可见狼队私聊提案(信息隔离:_should_receive 仅 god/replay 收)
            await self._emit({
                "type": "wolf_caucus", "day": self.state.day, "seat": actor.seat,
                "target_seat": tgt, "strategy": strat,
                "text": f"提议今天推{tgt}号,口径:{strat}" if tgt else f"提议:{strat or '(未定)'}",
            })
            # 队友之间共享提案(私聊,仅狼人可见)
            for pid2, actor2 in wolf_entries:
                tgt_txt = f"{tgt}号" if tgt else "(未定)"
                actor2.observe_event(self.state.day, "day", "wolf_caucus",
                                     f"队友{actor.seat}号提议今天推{tgt_txt},口径:{strat}")
            if tgt:
                proposals.append((tgt, strat))
            if strat:
                strategy_votes.append(strat)

        if not proposals:
            return
        # 多数决目标
        from collections import Counter as _Counter
        tally = _Counter(t for t, _ in proposals)
        top_count = tally.most_common(1)[0][1]
        tied = [t for t, c in tally.items() if c == top_count]
        chosen_target = self.rng.choice(tied)
        # 策略取被提名最多的(简单:取第一个提议的策略,因口径需一致)
        chosen_strategy = next((s for t, s in proposals if t == chosen_target), "")
        consensus_text = (f"狼队白天共识:统一推{chosen_target}号出局;口径:"
                          f"{chosen_strategy or '(各自发挥,别跟太紧避免暴露抱团)'}")
        # god 模式可见共识
        await self._emit({
            "type": "wolf_caucus_consensus", "day": self.state.day,
            "target_seat": chosen_target, "strategy": chosen_strategy,
            "text": consensus_text,
        })
        # 共识注入每个狼人私有记忆
        for pid, actor in wolf_entries:
            actor.observe_event(
                self.state.day, "day", "wolf_caucus_consensus", consensus_text
            )

    async def _witch_save_phase(self, *, phase_deadline: float | None = None) -> None:
        witch_entries = [(pid, a) for pid, a in self.actors.items()
                         if self.state.get_player(pid).alive and a.role == Role.WITCH and self.state.witch_antidote]
        if not witch_entries:
            return
        # 告知女巫今夜被杀目标(女巫特有信息)
        kill_action = next((a for a in self.state.night_actions if a.action == NightActionType.KILL), None)
        kill_seat = self.state.get_player(kill_action.target_id).seat if kill_action else None
        for pid, actor in witch_entries:
            actor.observe_event(self.state.day, "night", "witch_kill_preview",
                                f"今夜{'无人死亡' if kill_seat is None else f'{kill_seat}号被杀'}")

        tasks = [
            self._with_decision_timeout(
                actor, "night", "witch_save",
                actor.decide_night_action(
                    self.state,
                    pid,
                    requested_action="save",
                    human_context={"killed_seat": kill_seat},
                ),
                phase_deadline=phase_deadline,
            )
            for pid, actor in witch_entries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for pid, res in zip([pid for pid, _ in witch_entries], results):
            actor = self.actors[pid]
            if isinstance(res, Exception):
                self._failed_events.append(self._agent_decision_failure_event(
                    actor,
                    phase="night",
                    action="witch_save",
                    err=res,
                ))
                continue
            self._record_consumed_decision(actor, res, phase="night")
            await self._emit_thinking(actor, res)
            submitted = self._submit_safe(pid, res, [NightActionType.SAVE])
            target_id = self._normalize_target(res.target_id) if isinstance(res, Decision) else None
            if submitted and isinstance(res, Decision) and res.action == AgentAction.SAVE and target_id:
                RulesEngine.apply_witch_save(self.state, used=True)
                actor.observe_event(self.state.day, "night", "witch_save_used",
                                    f"你救活了{self.state.get_player(target_id).seat}号")

    async def _witch_poison_phase(self, *, phase_deadline: float | None = None) -> None:
        witch_entries = [(pid, a) for pid, a in self.actors.items()
                         if self.state.get_player(pid).alive and a.role == Role.WITCH and self.state.witch_poison]
        if not witch_entries:
            return
        tasks = [
            self._with_decision_timeout(
                actor, "night", "witch_poison",
                actor.decide_night_action(self.state, pid, requested_action="poison"),
                phase_deadline=phase_deadline,
            )
            for pid, actor in witch_entries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for pid, res in zip([pid for pid, _ in witch_entries], results):
            actor = self.actors[pid]
            if isinstance(res, Exception):
                self._failed_events.append(self._agent_decision_failure_event(
                    actor,
                    phase="night",
                    action="witch_poison",
                    err=res,
                ))
                continue
            self._record_consumed_decision(actor, res, phase="night")
            await self._emit_thinking(actor, res)
            submitted = self._submit_safe(pid, res, [NightActionType.POISON])
            target_id = self._normalize_target(res.target_id) if isinstance(res, Decision) else None
            if submitted and isinstance(res, Decision) and res.action == AgentAction.POISON and target_id:
                RulesEngine.apply_witch_poison(self.state, used=True)
                actor.observe_event(self.state.day, "night", "witch_poison_used",
                                    f"你对{self.state.get_player(target_id).seat}号使用了毒药")

    def _submit_safe(self, pid: str, decision: Any, allowed: list[NightActionType]) -> bool:
        if not isinstance(decision, Decision):
            return False
        actor = self.actors[pid]
        if decision.is_skip:
            return False
        target_id = self._normalize_target(decision.target_id)
        if not target_id:
            return False
        action_map = {
            AgentAction.SEE: NightActionType.SEE,
            AgentAction.SAVE: NightActionType.SAVE,
            AgentAction.POISON: NightActionType.POISON,
            AgentAction.GUARD: NightActionType.GUARD,
            AgentAction.NIGHT_KILL: NightActionType.KILL,
        }
        na_type = action_map.get(decision.action)
        if na_type not in allowed:
            return False
        try:
            RulesEngine.submit_night_action(
                self.state, NightAction(actor_id=pid, action=na_type, target_id=target_id)
            )
            if na_type == NightActionType.SEE:
                actor.observe_event(self.state.day, "night", "seer_action", f"你查验了{self.state.get_player(target_id).seat}号")
            elif na_type == NightActionType.GUARD:
                actor.observe_event(self.state.day, "night", "guard_target",
                                    f"{self.state.get_player(target_id).seat}号")
            return True
        except Exception as err:  # noqa: BLE001
            logger.info("夜间行动被引擎拒绝(pid=%s): %s", pid, err)
            self._failed_events.append({
                "type": "agent_decision_failed",
                "seat": actor.seat,
                "phase": "night",
                "reason": str(err),
            })
            return False

    def _submit_explicit(self, pid: str, na_type: NightActionType, target_id: str) -> None:
        try:
            RulesEngine.submit_night_action(
                self.state, NightAction(actor_id=pid, action=na_type, target_id=target_id)
            )
        except Exception as err:  # noqa: BLE001
            logger.info("夜间行动被引擎拒绝(pid=%s %s): %s", pid, na_type, err)

    async def _push_night_results_to_memory(self) -> None:
        for pid, actor in self.actors.items():
            player = self.state.get_player(pid)
            if not player.alive:
                continue
            for d in self.state.night_deaths:
                actor.observe_event(self.state.day, "night", "death",
                                    f"{d['seat']}号{d['name']} 死亡")
            for ev in self.state.events:
                if ev.visibility == EventVisibility.PRIVATE and pid in ev.recipients and ev.day == self.state.day:
                    actor.observe_event(ev.day, "night", ev.type, ev.message, **(ev.payload or {}))
                    if ev.type == "seer_result":
                        team = ev.payload.get("team")
                        target_seat = ev.payload.get("target_seat")
                        if team == Team.WEREWOLVES:
                            actor.set_trust(target_seat, 1.0)
                        elif team == Team.VILLAGE:
                            actor.set_trust(target_seat, 0.0)

    # ------------------------------------------------------------------
    # 白天
    # ------------------------------------------------------------------
    def _fixed_speak_order(self, pids: list[str]) -> list[str]:
        return sorted(pids, key=lambda pid: self.actors[pid].seat)

    async def _run_day(self) -> None:
        day = self.state.day
        await self._emit({"type": "phase_started", "phase": "day", "day": day,
                          "message": f"第{day}天白天,请发言。"})
        day_deadline = self._start_phase_deadline("day")

        # 方向C:白天发言前狼队党团会议(复用夜间私聊拓扑)。
        # 平衡(20局统计暴露狼人70%偏强):党团会议仅 day1 举办——首日推人目标+统一口径
        # 一次性注入狼人记忆。后续天狼人独立发言,更易暴露抱团(好人态度网络可识别)。
        # 狼人发言仍自主——harness 不写发言,守 no-fallback + agent 自决。
        if day <= 1 and self.turn_policy == "bid_reply_caucus":
            await self._werewolf_day_caucus(phase_deadline=day_deadline)

        today_speeches: list[dict[str, Any]] = []
        # 被提及/被指控的座位集合(方向A:被点名者优先调度,但不强制回应)。
        # 跨轮累积——一旦被指控,后续轮次该 seat 在 bid≥4 时优先被叫起回应。
        mentioned_seats: set[int] = set()
        living_pids = [pid for pid, a in self.actors.items() if self.state.get_player(pid).alive]

        first_order = self._fixed_speak_order(living_pids)
        if self.turn_policy != "fixed_round_robin":
            self.rng.shuffle(first_order)

        for round_idx in range(self.max_speak_rounds):
            # 每轮的发言席位记录(每轮可重新发言)
            spoke_this_round: set[str] = set()
            if round_idx == 0:
                order = first_order
            elif self.turn_policy == "fixed_round_robin":
                order = self._fixed_speak_order(living_pids)
            else:
                order = await self._bid_order_by_llm(
                    living_pids,
                    today_speeches,
                    spoke_this_round,
                    mentioned_seats,
                    use_reply_priority=self.turn_policy in {"bid_reply", "bid_reply_caucus"},
                    phase_deadline=day_deadline,
                )
                # 收敛检测:本轮无人想发言(bid 全 0),讨论自然结束
                if not order:
                    break
                # 被点名但本轮未被叫起(bid<4 或已被排除)的 seat:记录"被点名未回应"观察。
                # 沉默也是信号,供后续判断。不强制叫起(铁律:harness 不替 agent 决策是否回应)。
                called_seats = {self.actors[pid].seat for pid in order}
                for seat in mentioned_seats:
                    if seat not in called_seats:
                        pid = self._seat_to_pid(seat)
                        if pid and pid in self.actors:
                            self.actors[pid].observe_event(
                                day, "day", "mentioned_silent",
                                f"你被点名/指控但本轮未发声(你的 bid 未达 4)。沉默可能被当心虚。"
                            )

            anyone_spoke = False
            for pid in order:
                if pid in spoke_this_round:
                    continue
                actor = self.actors[pid]
                if not self.state.get_player(pid).alive:
                    continue
                # 第 0 轮顺序发言;后续轮次已在 _bid_order_by_llm 中收集决策
                if round_idx == 0 or self.turn_policy == "fixed_round_robin":
                    try:
                        decision = await self._with_decision_timeout(
                            actor,
                            "day",
                            "speak",
                            actor.decide_speak(self.state, pid, today_speeches=today_speeches),
                            phase_deadline=day_deadline,
                        )
                    except AgentDecisionError as err:
                        await self._emit(self._agent_decision_failure_event(
                            actor,
                            phase="day",
                            action="speak",
                            err=err,
                        ))
                        continue
                    except Exception as err:  # noqa: BLE001
                        await self._emit(self._agent_decision_failure_event(
                            actor,
                            phase="day",
                            action="speak",
                            err=err,
                        ))
                        continue
                    self._record_consumed_decision(actor, decision, phase="day")
                else:
                    decision = getattr(actor, "_pending_speak_decision", None)
                    if decision is None:
                        continue
                    delattr(actor, "_pending_speak_decision")

                await self._emit_thinking(actor, decision)
                if decision.suspicion:
                    actor.apply_suspicion(decision.suspicion)
                await self._emit({
                    "type": "trust_update",
                    "seat": actor.seat,
                    "trust": {str(k): round(v, 3) for k, v in actor.memory.trust.items()},
                    "phase": "day",
                })
                speech = decision.speech or "(沉默)"
                # 去重硬约束:与今日已有发言高度重复则不广播,转为倾听(强制收敛,不伪造内容)。
                # 反驳豁免(方向A):reply_to 非空的发言是在回应/反驳,本就会复述对方论点,
                # 放宽自重复阈值避免误杀辩护。
                is_reply = decision.reply_to is not None
                is_dup, dup_reason = _is_duplicate_speech(
                    speech, today_speeches, actor.seat, relax_self=is_reply
                )
                if is_dup and round_idx > 0:
                    # 重复发言视为无新内容,本轮该 seat 倾听,不计入发言
                    spoke_this_round.add(pid)
                    actor.observe_event(day, "day", "speech_skipped_dup",
                                        f"你想说:{speech}(系统判定{dup_reason},转为倾听)")
                    continue
                speech_entry = {
                    "seat": actor.seat, "name": actor.name, "text": speech,
                    "bid": decision.bid, "reply_to": decision.reply_to, "accuses": decision.accuses,
                    "attitudes": decision.attitudes, "claim": decision.claim, "day": day,
                }
                today_speeches.append(speech_entry)
                # 结构化指控入 mentioned_seats:被指控者后续轮次 bid≥4 时优先被叫起回应
                if decision.accuses:
                    mentioned_seats.update(int(s) for s in decision.accuses)
                self._record_public_speech_memory(speech_entry)
                if decision.claim:
                    actor.record_claim(actor.seat, day, decision.claim)
                    # 所有人都听到该公开 claim,记录到自己视角的 claims(矛盾检测用)
                    for other_pid, other_actor in self.actors.items():
                        if other_pid != pid and self.state.get_player(other_pid).alive:
                            other_actor.record_claim(actor.seat, day, decision.claim)
                await self._emit({
                    "type": "speech",
                    "day": day, "seat": actor.seat, "name": actor.name,
                    "text": speech, "bid": decision.bid, "claim": decision.claim,
                    "reply_to": decision.reply_to, "accuses": decision.accuses,
                    "attitudes": decision.attitudes,
                    # Internal-only: keep wolf deception intent for post-game
                    # research metrics, but never broadcast it as a live speech
                    # field. Otherwise play/spectate clients can see wolves'
                    # declared strategy in real time.
                    "_analysis_deception": decision.deception,
                })
                self._record_posterior_snapshot(
                    trigger="speech",
                    today_speeches=today_speeches,
                    source_seat=actor.seat,
                )
                spoke_this_round.add(pid)
                anyone_spoke = True
                if round_idx == 0 and len(spoke_this_round) >= len(living_pids):
                    break
            if not anyone_spoke and round_idx > 0:
                break

        self.state = RulesEngine.start_vote(self.state)
        await self._emit({"type": "phase_started", "phase": "voting", "day": day,
                          "message": "讨论结束,开始投票。"})
        await self._run_voting(today_speeches=today_speeches)

    async def _bid_order_by_llm(
        self,
        living_pids: list[str],
        today_speeches: list[dict],
        spoke_this_round: set[str],
        mentioned_seats: set[int] | None = None,
        use_reply_priority: bool = True,
        phase_deadline: float | None = None,
    ) -> list[str]:
        """并发收集所有 eligible 玩家的发言+竞价,按优先级决定发言顺序。

        承 ARCHITECTURE.md §3.6/§5:"竞价发言循环:bid→speak",并补上"被提及者优先":
          1. 被提及/被指控(mentioned_seats)且 bid≥4 → 最优先(被点名必须回应,但仍由 agent 自评 bid)
          2. bid 降序
          3. 平局时被提及者优先
        bid=0 视为倾听,过滤不发言。不强制叫起 bid<4 的被提及者(铁律:harness 不替 agent 决策)。
        """
        mentioned_seats = mentioned_seats or set()
        eligible = [pid for pid in living_pids if pid not in spoke_this_round]
        tasks = [
            self._with_decision_timeout(
                self.actors[pid],
                "day",
                "bid_speak",
                self.actors[pid].decide_speak(self.state, pid, today_speeches=today_speeches),
                phase_deadline=phase_deadline,
            )
            for pid in eligible
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        decisions: list[tuple[str, Any]] = []
        for pid, res in zip(eligible, results):
            actor = self.actors[pid]
            if isinstance(res, Exception):
                await self._emit(self._agent_decision_failure_event(
                    actor,
                    phase="day",
                    action="bid_speak",
                    err=res,
                ))
                continue
            self._record_consumed_decision(actor, res, phase="day")
            actor._pending_speak_decision = res
            decisions.append((pid, res))

        rng = random.Random(self.state.day * 31 + len(today_speeches))
        # 排序键(升序,越小越先发言):
        #   (0) 被提及且 bid≥4 → 0(最优先),否则 1
        #   (1) -bid(bid 降序)
        #   (2) 被提及者优先(0 < 1)
        #   (3) 随机抖动
        def sort_key(item: tuple[str, Any]) -> tuple:
            pid, d = item
            seat = self.actors[pid].seat
            bid = d.bid or 0
            is_mentioned = use_reply_priority and seat in mentioned_seats
            must_reply = 0 if (is_mentioned and bid >= 4) else 1
            mentioned_rank = 0 if is_mentioned else 1
            return (must_reply, -bid, mentioned_rank, rng.random())

        decisions.sort(key=sort_key)
        # 记录本轮 bid=0 的数量,供收敛检测
        self._last_round_silent = sum(1 for _, d in decisions if (d.bid or 0) == 0)
        return [pid for pid, d in decisions if (d.bid or 0) > 0]

    # ------------------------------------------------------------------
    # 投票
    # ------------------------------------------------------------------
    async def _run_voting(
        self,
        *,
        pk_candidates: list[str] | None = None,
        today_speeches: list[dict[str, Any]] | None = None,
        pk_round: int = 0,
    ) -> None:
        """收票 + 结算。pk_candidates 非 None 时为 PK 重投,投票目标会被限制在候选内。"""
        day = self.state.day
        today_speeches = today_speeches or []
        voting_deadline = self._start_phase_deadline("voting")
        living_pids = [pid for pid, a in self.actors.items() if self.state.get_player(pid).alive]

        vote_tasks = [
            self._with_decision_timeout(
                self.actors[pid],
                "voting",
                "vote",
                self.actors[pid].decide_vote(
                    self.state, pid, today_speeches=today_speeches, pk_candidates=pk_candidates
                ),
                phase_deadline=voting_deadline,
            )
            for pid in living_pids
        ]
        results = await asyncio.gather(*vote_tasks, return_exceptions=True)

        for pid, res in zip(living_pids, results):
            actor = self.actors[pid]
            if isinstance(res, Exception):
                await self._emit(self._agent_decision_failure_event(
                    actor,
                    phase="voting",
                    action="vote",
                    err=res,
                ))
                continue
            if not isinstance(res, Decision):
                continue
            self._record_consumed_decision(actor, res, phase="voting")
            await self._emit_thinking(actor, res)
            target_id = self._normalize_target(res.target_id)
            if res.action == AgentAction.VOTE and target_id:
                try:
                    RulesEngine.submit_vote(self.state, Vote(voter_id=pid, target_id=target_id))
                    actor.observe_event(day, "voting", "vote",
                                        f"你投了{self.state.get_player(target_id).seat}号")
                    self._vote_log.append({
                        "voter_seat": actor.seat,
                        "target_seat": self.state.get_player(target_id).seat,
                        "day": day,
                        "objective_summary": getattr(res, "objective_summary", None),
                        # Private reasoning stays internal for analysis only; never broadcast.
                        "reasoning": getattr(res, "reasoning", None),
                    })
                    await self._emit({
                        "type": "vote_cast", "day": day, "seat": actor.seat, "name": actor.name,
                        "target_seat": self.state.get_player(target_id).seat,
                        # OSR 客观摘要(Beyond Survival):投票前的两段式第1段,可审计+前端展示
                        "objective_summary": getattr(res, "objective_summary", None),
                    })
                    self._record_posterior_snapshot(
                        trigger="vote",
                        today_speeches=today_speeches,
                        source_seat=actor.seat,
                    )
                except Exception as err:  # noqa: BLE001
                    logger.info("投票被拒绝(pid=%s): %s", pid, err)

        if len(self.state.votes) < len(living_pids):
            logger.warning("投票不完整(%d/%d),跳过结算", len(self.state.votes), len(living_pids))
            await self._emit({"type": "vote_incomplete", "day": day,
                              "cast": len(self.state.votes), "needed": len(living_pids)})
            self.state.phase = Phase.NIGHT
            self.state.day += 1
            self.state.votes.clear()
            return

        before_events = len(self.state.events)
        self.state = RulesEngine.resolve_vote(self.state)
        new_events = self.state.events[before_events:]
        message = next((ev.message for ev in reversed(new_events)
                        if ev.type in ("player_exiled", "vote_tied", "vote_tied_pk")), None)
        await self._emit({
            "type": "vote_resolved",
            "day": day,
            "message": message,
            "votes": dict(self.state.votes) if self.state.phase != Phase.NIGHT else {},
        })

        # PK 处理
        if self.state.phase == Phase.VOTING:
            await self._run_pk(today_speeches=today_speeches, pk_round=pk_round)
            return

        # 遗言 + 猎人开枪
        await self._process_deaths_and_hunter()

        await self._check_winner_and_advance()

    async def _run_pk(
        self,
        *,
        today_speeches: list[dict[str, Any]] | None = None,
        pk_round: int = 0,
        max_pk_rounds: int = 2,
    ) -> None:
        """平票 PK:仅候选者额外发言,然后重新投票(投票目标限制在 PK 候选内)。

        承载 max_pk_rounds 上限:超过后仍平票则当日无人放逐,进入夜晚。
        避免恶意/巧合导致的 PK 死循环。
        """
        day = self.state.day
        day_speeches = today_speeches or []
        pk_ids = list(self.state.pk_candidates)
        pk_speeches: list[dict[str, Any]] = []
        await self._emit({"type": "phase_started", "phase": "pk", "day": day,
                          "message": f"平票,进入 PK:{[self.state.get_player(p).seat for p in pk_ids]}"})
        pk_deadline = self._start_phase_deadline("pk")
        for pid in pk_ids:
            actor = self.actors.get(pid)
            if not actor or not self.state.get_player(pid).alive:
                continue
            try:
                decision = await self._with_decision_timeout(
                    actor,
                    "pk",
                    "speak",
                    actor.decide_speak(
                        self.state, pid, today_speeches=day_speeches + pk_speeches
                    ),
                    phase_deadline=pk_deadline,
                )
            except AgentDecisionError as err:
                await self._emit(self._agent_decision_failure_event(
                    actor,
                    phase="pk",
                    action="speak",
                    err=err,
                ))
                continue
            except Exception as err:  # noqa: BLE001
                await self._emit(self._agent_decision_failure_event(
                    actor,
                    phase="pk",
                    action="speak",
                    err=err,
                ))
                continue
            speech = decision.speech or "(沉默)"
            self._record_consumed_decision(actor, decision, phase="pk")
            await self._emit_thinking(actor, decision)
            speech_entry = {
                "seat": actor.seat, "name": actor.name, "text": speech, "bid": decision.bid,
                "reply_to": decision.reply_to, "accuses": decision.accuses,
                "attitudes": decision.attitudes, "claim": decision.claim, "day": day,
            }
            pk_speeches.append(speech_entry)
            self._record_public_speech_memory(speech_entry)
            if decision.claim:
                actor.record_claim(actor.seat, day, decision.claim)
                for other_pid, other_actor in self.actors.items():
                    if other_pid != pid and self.state.get_player(other_pid).alive:
                        other_actor.record_claim(actor.seat, day, decision.claim)
            await self._emit({"type": "speech", "day": day, "seat": actor.seat, "name": actor.name,
                              "text": speech, "bid": decision.bid, "claim": decision.claim, "pk": True,
                              "reply_to": decision.reply_to, "accuses": decision.accuses,
                              "attitudes": decision.attitudes,
                              "_analysis_deception": decision.deception})
            self._record_posterior_snapshot(
                trigger="pk_speech",
                today_speeches=day_speeches + pk_speeches,
                source_seat=actor.seat,
            )
        if pk_round + 1 >= max_pk_rounds:
            # 达到 PK 上限:清空 PK 候选,强制结算为"无人放逐"进入夜晚
            logger.info("PK 达到上限(%d 轮),当日无人放逐,进入夜晚", max_pk_rounds)
            await self._emit({"type": "vote_resolved", "day": day,
                              "message": f"PK {max_pk_rounds} 轮仍平票,无人放逐。", "votes": {}})
            self.state.pk_candidates = []
            self.state.phase = Phase.NIGHT
            self.state.day += 1
            self.state.votes.clear()
            return
        await self._run_voting(
            pk_candidates=pk_ids,
            today_speeches=day_speeches + pk_speeches,
            pk_round=pk_round + 1,
        )

    async def _process_deaths_and_hunter(self) -> None:
        # 遗言
        last_words_deadline = self._start_phase_deadline("last_words")
        while self.state.last_words_queue:
            q = self.state.last_words_queue[0]
            pid = q["id"]
            actor = self.actors.get(pid)
            if actor:
                try:
                    decision = await self._with_decision_timeout(
                        actor,
                        "last_words",
                        "last_words",
                        actor.decide_last_words(self.state, pid, q["reason"]),
                        phase_deadline=last_words_deadline,
                    )
                except Exception as err:  # noqa: BLE001
                    await self._emit(self._agent_decision_failure_event(
                        actor,
                        phase="last_words",
                        action="last_words",
                        err=err,
                        seat=q["seat"],
                    ))
                    self.state.last_words_queue.pop(0)
                    continue
                text = decision.speech or "(无遗言)"
                self.state = RulesEngine.record_last_words(self.state, pid, text)
                self._record_consumed_decision(actor, decision, phase="last_words")
                await self._emit_thinking(actor, decision)
                await self._emit({"type": "last_words", "day": self.state.day,
                                  "seat": q["seat"], "name": q["name"], "text": text})
            else:
                self.state.last_words_queue.pop(0)

        # 猎人开枪(含被投票放逐/夜间死亡但非毒杀)
        hunter_deadline = self._start_phase_deadline("hunter")
        while self.state.pending_hunter:
            hunter_id = self.state.pending_hunter[0]
            actor = self.actors.get(hunter_id)
            if not actor:
                self.state.pending_hunter.pop(0)
                continue
            try:
                # 用夜间决策复用:target_seat 即可
                decision = await self._with_decision_timeout(
                    actor,
                    "hunter",
                    "hunter_shot",
                    actor.decide_night_action(self.state, hunter_id, requested_action="hunter_shot"),
                    phase_deadline=hunter_deadline,
                )
            except Exception as err:  # noqa: BLE001
                await self._emit(self._agent_decision_failure_event(
                    actor,
                    phase="hunter",
                    action="hunter_shot",
                    err=err,
                ))
                self.state = RulesEngine.hunter_shoot(self.state, hunter_id, None)
                await self._emit({
                    "type": "hunter_shot", "day": self.state.day,
                    "seat": actor.seat, "name": actor.name,
                    "target_seat": None,
                    "skip_reason": "hunter_decision_failed",
                })
                continue
            self._record_consumed_decision(actor, decision, phase="hunter")
            await self._emit_thinking(actor, decision)
            target_id = self._normalize_target(decision.target_id) if isinstance(decision, Decision) else None
            try:
                self.state = RulesEngine.hunter_shoot(self.state, hunter_id, target_id)
            except Exception as err:  # noqa: BLE001
                logger.exception("猎人开枪路径异常(%s): %s", hunter_id, err)
                raise
            await self._emit({
                "type": "hunter_shot", "day": self.state.day,
                "seat": actor.seat, "name": actor.name,
                "target_seat": self.state.get_player(target_id).seat if target_id else None,
            })
            if target_id:
                RulesEngine.queue_last_words(self.state, target_id, reason="hunter_shot")

        # 被猎人带走者遗言
        last_words_deadline = self._start_phase_deadline("last_words")
        while self.state.last_words_queue:
            q = self.state.last_words_queue[0]
            pid = q["id"]
            actor = self.actors.get(pid)
            if actor:
                try:
                    decision = await self._with_decision_timeout(
                        actor,
                        "last_words",
                        "last_words",
                        actor.decide_last_words(self.state, pid, q["reason"]),
                        phase_deadline=last_words_deadline,
                    )
                except Exception as err:  # noqa: BLE001
                    await self._emit(self._agent_decision_failure_event(
                        actor,
                        phase="last_words",
                        action="last_words",
                        err=err,
                        seat=q["seat"],
                    ))
                    self.state.last_words_queue.pop(0)
                    continue
                text = decision.speech or "(无遗言)"
                self.state = RulesEngine.record_last_words(self.state, pid, text)
                self._record_consumed_decision(actor, decision, phase="last_words")
                await self._emit_thinking(actor, decision)
                await self._emit({"type": "last_words", "day": self.state.day,
                                  "seat": q["seat"], "name": q["name"], "text": text})
            else:
                self.state.last_words_queue.pop(0)

    async def _check_winner_and_advance(self) -> None:
        if self.state.phase == Phase.ENDED:
            await self._emit_game_ended()
            return
        # 否则 resolve_vote 已把 phase 设为 NIGHT 且 day+1

    # ------------------------------------------------------------------
    # 反思 + 复盘
    # ------------------------------------------------------------------
    async def _reflect_all(self) -> None:
        living = [pid for pid, a in self.actors.items() if self.state.get_player(pid).alive]
        reflection_deadline = self._start_phase_deadline("reflection")
        tasks = [
            self._with_decision_timeout(
                self.actors[pid],
                "reflection",
                "reflect",
                self.actors[pid].reflect(self.state, pid),
                phase_deadline=reflection_deadline,
            )
            for pid in living
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for pid, res in zip(living, results):
            if isinstance(res, Exception):
                await self._emit(self._agent_decision_failure_event(
                    self.actors[pid],
                    phase="reflection",
                    action="reflect",
                    err=res,
                ))
        seat_map = {self.state.get_player(pid).seat: pid for pid in living}
        payload = {}
        for seat, pid in seat_map.items():
            actor = self.actors[pid]
            recent = [r.text for r in actor.memory.reflections[-1:]]
            if recent:
                payload[str(seat)] = recent[-1]
        if payload:
            await self._emit({"type": "reflections_update", "reflections": payload})

    def _dialogue_metrics(self) -> dict[str, Any]:
        """方向A/B/C 对话对抗量化指标(客观统计,独立于 LLM 5维评审)。

        用于跨对局对比 A/B/C 落地前后的对抗质量提升:
        - reply_rate:含 reply_to 的发言占比(方向A 对话交锋度)
        - accuse_rate:含 accuses 的发言占比(方向A 指控结构化度)
        - attitude_rate:含 attitudes 的发言占比(方向B 信念建模度)
        - attitude_edges:support/oppose 边总数(态度网络密度)
        - wolf_coordination:狼人指控目标的一致度(方向C 协同度)——多狼指控同一好人的比例
        """
        speeches = self._speech_log
        n = len(speeches) or 1
        reply_n = sum(1 for s in speeches if s.get("reply_to"))
        accuse_n = sum(1 for s in speeches if s.get("accuses"))
        att_n = sum(1 for s in speeches if s.get("attitudes"))
        sup_edges = opp_edges = 0
        for s in speeches:
            atts = s.get("attitudes")
            if isinstance(atts, dict):
                for v in atts.values():
                    if v == "support":
                        sup_edges += 1
                    elif v == "oppose":
                        opp_edges += 1
        # 狼人协同度:按天统计"多少不同狼座位共同指控同一好人",并 clamp 到 [0,1]。
        # 旧算法按重复指控次数除以狼人数,单只狼多次重复点同一目标会把指标推到 >1。
        wolf_seats = {p.seat for p in self.state.players if p.role == Role.WEREWOLF}
        good_seats = {p.seat for p in self.state.players if p.role != Role.WEREWOLF}
        by_day_target: dict[int, dict[int, set[int]]] = {}
        for s in speeches:
            speaker = _as_int(s.get("seat"))
            if speaker not in wolf_seats or not s.get("accuses"):
                continue
            day = _as_int(s.get("day")) or 0
            for raw_target in s["accuses"]:
                target = _as_int(raw_target)
                if target in good_seats:
                    by_day_target.setdefault(day, {}).setdefault(target, set()).add(speaker)
        coord = 0.0
        if by_day_target and wolf_seats:
            day_scores = [
                max((len(wolf_cover) / len(wolf_seats)) for wolf_cover in target_map.values())
                for target_map in by_day_target.values()
                if target_map
            ]
            coord = round(max(0.0, min(1.0, max(day_scores, default=0.0))), 3)
        # 欺骗策略分布(方向DR):狼人声明用的欺骗手段——结构化后可量化狼人欺骗多样性
        from collections import Counter as _Counter2
        wolf_deceptions = [s.get("deception") for s in speeches
                           if s.get("seat") in wolf_seats and s.get("deception")
                           and s.get("deception") != "none"]
        deception_dist = dict(_Counter2(wolf_deceptions))
        return {
            "speech_count": len(speeches),
            "reply_rate": round(reply_n / n, 3),
            "accuse_rate": round(accuse_n / n, 3),
            "attitude_rate": round(att_n / n, 3),
            "support_edges": sup_edges,
            "oppose_edges": opp_edges,
            "wolf_coordination": coord,
            "wolf_seats": sorted(wolf_seats),
            "wolf_deception_count": len(wolf_deceptions),
            "wolf_deception_dist": deception_dist,
        }

    def _debate_process_metrics(self) -> dict[str, Any]:
        """赛后辩论过程指标,用于 turn_policy ablation。

        只使用公开 speech 结构化字段和调度配置;不读取 hidden role reasoning,
        不反向影响 live 发言顺序。
        """
        speeches = list(self._speech_log)
        speech_count = len(speeches)
        by_seat = Counter(_as_int(s.get("seat")) for s in speeches if _as_int(s.get("seat")) is not None)
        bids = [
            int(bid)
            for bid in (_as_int(s.get("bid")) for s in speeches)
            if bid is not None
        ]

        reply_latencies: list[float] = []
        for idx, speech in enumerate(speeches):
            reply_to = _as_int(speech.get("reply_to"))
            day = _as_int(speech.get("day"))
            if reply_to is None or day is None:
                continue
            for prev_idx in range(idx - 1, -1, -1):
                prev = speeches[prev_idx]
                if _as_int(prev.get("day")) == day and _as_int(prev.get("seat")) == reply_to:
                    reply_latencies.append(float(idx - prev_idx))
                    break

        claims = [
            (idx, _as_int(s.get("day")), _as_int(s.get("seat")))
            for idx, s in enumerate(speeches)
            if isinstance(s.get("claim"), dict)
        ]
        challenged = 0
        for idx, day, claimer in claims:
            if day is None or claimer is None:
                continue
            for later in speeches[idx + 1:]:
                if _as_int(later.get("day")) != day:
                    continue
                accuses = {_as_int(target) for target in (later.get("accuses") or [])}
                attitudes = later.get("attitudes") if isinstance(later.get("attitudes"), dict) else {}
                opposed = {
                    _as_int(target)
                    for target, stance in attitudes.items()
                    if str(stance) == "oppose"
                }
                if claimer in accuses or claimer in opposed:
                    challenged += 1
                    break

        accuse_targets: Counter[int] = Counter()
        support_edges: set[tuple[int, int]] = set()
        oppose_edges: set[tuple[int, int]] = set()
        for speech in speeches:
            source = _as_int(speech.get("seat"))
            if source is None:
                continue
            for target in speech.get("accuses") or []:
                target_seat = _as_int(target)
                if target_seat is not None and target_seat != source:
                    accuse_targets[target_seat] += 1
                    oppose_edges.add((source, target_seat))
            attitudes = speech.get("attitudes") if isinstance(speech.get("attitudes"), dict) else {}
            for raw_target, raw_stance in attitudes.items():
                target = _as_int(raw_target)
                if target is None or target == source:
                    continue
                stance = str(raw_stance)
                if stance == "support":
                    support_edges.add((source, target))
                elif stance == "oppose":
                    oppose_edges.add((source, target))

        def reciprocal_count(edges: set[tuple[int, int]]) -> int:
            return sum(
                1
                for a, b in edges
                if a < b and (b, a) in edges
            )

        accuse_edge_count = sum(accuse_targets.values())
        return {
            "turn_policy": self.turn_policy,
            "caucus_enabled": 1 if self.turn_policy == "bid_reply_caucus" else 0,
            "uses_bid_order": 0 if self.turn_policy == "fixed_round_robin" else 1,
            "uses_reply_priority": 1 if self.turn_policy in {"bid_reply", "bid_reply_caucus"} else 0,
            "speech_count": speech_count,
            "speaker_count": len(by_seat),
            "speaker_concentration": (
                round(max(by_seat.values()) / speech_count, 3) if speech_count and by_seat else None
            ),
            "bid_entropy": _normalized_entropy(bids),
            "avg_bid": _rounded_mean([float(bid) for bid in bids]),
            "reply_count": len(reply_latencies),
            "avg_reply_latency": _rounded_mean(reply_latencies),
            "claim_count": len(claims),
            "claim_challenged_count": challenged,
            "claim_challenged_rate": _rate(challenged, len(claims)),
            "accuse_target_count": len(accuse_targets),
            "top_accuse_target_share": (
                round(max(accuse_targets.values()) / accuse_edge_count, 3)
                if accuse_edge_count else None
            ),
            "support_loop_count": reciprocal_count(support_edges),
            "opposition_loop_count": reciprocal_count(oppose_edges),
        }

    def _objective_metrics(self) -> dict[str, Any]:
        """赛后确定性轨迹指标。

        这些指标使用完整真值,因此只能在 game analysis 中输出,不能进入 live prompt。
        它们用于补足单次 LLM judge 的不稳定性:先看可复算轨迹,再用五维评审解释原因。
        """
        role_by_seat = {
            p.seat: Role(p.role)
            for p in self.state.players
            if p.role is not None
        }
        wolf_seats = {seat for seat, role in role_by_seat.items() if role == Role.WEREWOLF}
        good_seats = set(role_by_seat) - wolf_seats

        votes = [
            {
                **v,
                "voter_seat": _as_int(v.get("voter_seat")),
                "target_seat": _as_int(v.get("target_seat")),
                "day": _as_int(v.get("day")) or 0,
            }
            for v in self._vote_log
        ]
        valid_votes = [v for v in votes if v["voter_seat"] in role_by_seat and v["target_seat"] in role_by_seat]
        good_votes = [v for v in valid_votes if v["voter_seat"] in good_seats]
        wolf_votes = [v for v in valid_votes if v["voter_seat"] in wolf_seats]
        good_vote_hits = sum(1 for v in good_votes if v["target_seat"] in wolf_seats)
        wolf_vote_hits = sum(1 for v in wolf_votes if v["target_seat"] in good_seats)

        accuse_records: list[tuple[int, int, int]] = []
        accused_same_day: set[tuple[int, int, int]] = set()
        attitude_events: list[tuple[int, int, int, str, int]] = []
        seer_wolf_claims: list[tuple[int, int, int]] = []
        for idx, s in enumerate(self._speech_log):
            speaker = _as_int(s.get("seat"))
            day = _as_int(s.get("day")) or 0
            if speaker not in role_by_seat:
                continue
            for raw_target in s.get("accuses") or []:
                target = _as_int(raw_target)
                if target in role_by_seat and target != speaker:
                    accuse_records.append((speaker, target, day))
                    accused_same_day.add((day, speaker, target))
            attitudes = s.get("attitudes")
            if isinstance(attitudes, dict):
                for raw_target, raw_stance in attitudes.items():
                    target = _as_int(raw_target)
                    stance = str(raw_stance)
                    if target in role_by_seat and stance in {"support", "oppose", "neutral"}:
                        attitude_events.append((day, speaker, target, stance, idx))
            claim = s.get("claim")
            if isinstance(claim, dict) and str(claim.get("role", "")).lower() == "seer":
                target = _as_int(claim.get("checked_seat"))
                result = str(claim.get("result", "")).lower()
                if target in role_by_seat and result in {"wolf", "werewolf", "werewolves", "狼人"}:
                    seer_wolf_claims.append((day, speaker, target))

        good_accuses = [r for r in accuse_records if r[0] in good_seats]
        wolf_accuses = [r for r in accuse_records if r[0] in wolf_seats]
        good_accuse_hits = sum(1 for _, target, _ in good_accuses if target in wolf_seats)
        wolf_accuse_hits = sum(1 for _, target, _ in wolf_accuses if target in good_seats)

        stance_votes = 0
        stance_consistent = 0
        for v in valid_votes:
            voter = v["voter_seat"]
            target = v["target_seat"]
            day = v["day"]
            prior = [
                ev for ev in attitude_events
                if ev[1] == voter and ev[2] == target and ev[0] <= day and ev[3] in {"support", "oppose"}
            ]
            if not prior:
                continue
            _d, _speaker, _target, stance, _idx = max(prior, key=lambda ev: (ev[0], ev[4]))
            stance_votes += 1
            if stance == "oppose":
                stance_consistent += 1

        accuse_vote_conversions = sum(
            1
            for v in valid_votes
            if (v["day"], v["voter_seat"], v["target_seat"]) in accused_same_day
        )

        osr_summary_votes = sum(1 for v in valid_votes if str(v.get("objective_summary") or "").strip())
        ct_marker_votes = 0
        for v in valid_votes:
            text = f"{v.get('objective_summary') or ''}\n{v.get('reasoning') or ''}".lower()
            if any(marker in text for marker in _CT_MARKERS):
                ct_marker_votes += 1

        seer_follow_den = 0
        seer_follow_num = 0
        for v in valid_votes:
            claimed_targets = {
                target
                for claim_day, _claimer, target in seer_wolf_claims
                if claim_day <= v["day"]
            }
            if not claimed_targets:
                continue
            seer_follow_den += 1
            if v["target_seat"] in claimed_targets:
                seer_follow_num += 1

        return {
            "vote_count": len(valid_votes),
            "good_vote_count": len(good_votes),
            "wolf_vote_count": len(wolf_votes),
            "vote_accuracy_good": _rate(good_vote_hits, len(good_votes)),
            "vote_accuracy_wolf": _rate(wolf_vote_hits, len(wolf_votes)),
            "accuse_count": len(accuse_records),
            "good_accuse_count": len(good_accuses),
            "wolf_accuse_count": len(wolf_accuses),
            "accuse_precision_good": _rate(good_accuse_hits, len(good_accuses)),
            "accuse_precision_wolf": _rate(wolf_accuse_hits, len(wolf_accuses)),
            "attitude_vote_consistency": _rate(stance_consistent, stance_votes),
            "attitude_vote_count": stance_votes,
            "accuse_to_vote_conversion": _rate(accuse_vote_conversions, len(valid_votes)),
            "osr_summary_rate": _rate(osr_summary_votes, len(valid_votes)),
            "ct_marker_rate": _rate(ct_marker_votes, len(valid_votes)),
            "seer_claim_follow_rate": _rate(seer_follow_num, seer_follow_den),
            "seer_claim_follow_vote_count": seer_follow_den,
        }

    def _posterior_speech_shift_groups(self) -> dict[tuple[int, int], list[list[dict[str, Any]]]]:
        """Group posterior before/after pairs by public speech source.

        The returned queue is keyed by (day, source_seat), matching _speech_log
        order. Each queue item contains per-viewer before/after posterior pairs
        for one speech. Analysis-only; no live state is mutated.
        """
        groups: list[dict[str, Any]] = []
        previous_by_viewer_day: dict[tuple[int, int], tuple[dict[str, float], str]] = {}
        for snap in self._posterior_log:
            day = _as_int(snap.get("day"))
            viewer = _as_int(snap.get("viewer_seat"))
            if day is None or viewer is None:
                continue
            trigger = str(snap.get("trigger") or "")
            current = _posterior_values(snap, "constrained_posterior") or _posterior_values(snap)
            source = _as_int(snap.get("source_seat"))
            if trigger not in {"speech", "pk_speech"} or source is None:
                previous_by_viewer_day[(viewer, day)] = (current, trigger)
                continue
            key = (day, str(snap.get("trigger")), source)
            # One _record_posterior_snapshot call emits at most one snapshot per
            # viewer. If the same source speaks twice in a row, the key is the
            # same; a repeated viewer marks the next speech boundary.
            last_viewers = groups[-1]["viewers"] if groups else set()
            if not groups or groups[-1]["key"] != key or viewer in last_viewers:
                groups.append({"key": key, "shifts": [], "viewers": set()})
            previous = previous_by_viewer_day.get((viewer, day))
            if previous is not None:
                before, previous_trigger = previous
                groups[-1]["shifts"].append({
                    "viewer_seat": viewer,
                    "before": before,
                    "after": current,
                    "previous_trigger": previous_trigger,
                    "snapshot": snap,
                })
            groups[-1]["viewers"].add(viewer)
            previous_by_viewer_day[(viewer, day)] = (current, trigger)

        result: dict[tuple[int, int], list[list[dict[str, Any]]]] = {}
        for group in groups:
            day, _trigger, source = group["key"]
            result.setdefault((day, source), []).append(group["shifts"])
        return result

    def _speech_evidence_refs(
        self,
        shifts: list[dict[str, Any]],
        *,
        day: int,
        speaker: int,
        target_seats: set[int],
    ) -> dict[str, Any]:
        """Link one speech audit record to public evidence ids seen by listeners."""
        evidence_types_by_id: dict[str, str] = {}
        delta_ids: set[str] = set()

        for shift in shifts:
            snap = shift.get("snapshot") if isinstance(shift, dict) else None
            if not isinstance(snap, dict):
                continue

            for item in snap.get("evidence_items") or []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("visibility") or "") == "private":
                    continue
                source = _as_int(item.get("source_seat"))
                if source != speaker:
                    continue
                item_day = _as_int(item.get("day"))
                if item_day is not None and item_day != day:
                    continue
                target = _as_int(item.get("target_seat"))
                if target_seats and target is not None and target not in target_seats:
                    continue
                evidence_id = str(item.get("evidence_id") or "")
                if not evidence_id:
                    continue
                evidence_types_by_id[evidence_id] = str(item.get("type") or "evidence")

            for delta in snap.get("posterior_deltas") or []:
                if not isinstance(delta, dict):
                    continue
                evidence_id = str(delta.get("evidence_id") or "")
                if evidence_id and evidence_id in evidence_types_by_id:
                    delta_ids.add(evidence_id)

        evidence_types = Counter(evidence_types_by_id.values())
        return {
            "evidence_ids": sorted(evidence_types_by_id)[:16],
            "posterior_delta_ids": sorted(delta_ids)[:16],
            "evidence_source_types": dict(sorted(evidence_types.items())),
        }

    def _deception_audit(self) -> dict[str, Any]:
        """Independent post-game deception audit.

        Speaker-declared `deception` is treated as intent, not truth. This
        deterministic pass uses role truth only after the game to audit coarse
        deception signals and listener posterior shifts. It never feeds results
        back into prompts or memories.
        """
        role_by_seat = {
            p.seat: Role(p.role)
            for p in self.state.players
            if p.role is not None
        }
        return self._deception_audit_from_roles(role_by_seat)

    def _collusion_audit(self) -> dict[str, Any]:
        """Post-game audit for public werewolf collusion signals.

        This is analysis-only. It uses role truth only after the game and only
        emits structured counts/ids; raw speech text, reasoning, and wolf caucus
        content stay out of the audit records.
        """
        role_by_seat = {
            p.seat: Role(p.role)
            for p in self.state.players
            if p.role is not None
        }
        wolf_seats = {seat for seat, role in role_by_seat.items() if role == Role.WEREWOLF}
        good_seats = set(role_by_seat) - wolf_seats
        wolf_pair_count = len(wolf_seats) * (len(wolf_seats) - 1) // 2
        if not wolf_seats:
            return {
                "wolf_speech_count": 0,
                "wolf_pair_count": 0,
                "active_wolf_pair_count": 0,
                "wolf_to_wolf_support_count": 0,
                "mutual_support_pair_count": 0,
                "shared_good_target_count": 0,
                "shared_good_target_speaker_coverage": None,
                "narrative_overlap_pair_count": 0,
                "avg_narrative_overlap": None,
                "coordinated_pressure_count": 0,
                "avg_shared_target_suspicion_gain": None,
                "avg_colluder_suspicion_gain": None,
                "evidence_linked_count": 0,
                "pair_listener_shift_sample_count": 0,
                "avg_pair_target_suspicion_gain": None,
                "pair_target_misdirected_rate": None,
                "windowed_relay_count": 0,
                "avg_windowed_relay_latency": None,
                "avg_relay_target_suspicion_gain": None,
                "relay_target_misdirected_rate": None,
                "deception_linked_pair_count": 0,
                "pair_listener_susceptibility_by_pair": {},
                "records": [],
            }

        shift_groups = self._posterior_speech_shift_groups()
        wolf_speeches: list[dict[str, Any]] = []
        support_edges: set[tuple[int, int]] = set()
        active_pairs: set[tuple[int, int]] = set()
        shared_targets: dict[tuple[int, int], dict[str, Any]] = {}
        speech_refs: dict[int, dict[str, Any]] = {}
        speech_shifts: dict[int, list[dict[str, Any]]] = {}
        target_gains_by_group: dict[tuple[int, int], list[float]] = {}
        colluder_suspicion_gains: list[float] = []
        pair_stats: dict[str, dict[str, Any]] = {}

        def pair_key(a: int, b: int) -> str:
            left, right = sorted((int(a), int(b)))
            return f"{left}-{right}"

        def pair_stat(a: int, b: int) -> dict[str, Any]:
            key = pair_key(a, b)
            if key not in pair_stats:
                left, right = (int(part) for part in key.split("-", 1))
                pair_stats[key] = {
                    "wolf_seats": [left, right],
                    "active_days": set(),
                    "shared_good_target_count": 0,
                    "wolf_to_wolf_support_count": 0,
                    "mutual_support_pair_count": 0,
                    "narrative_overlap_pair_count": 0,
                    "coordinated_pressure_count": 0,
                    "target_gains": [],
                    "target_misdirected": [],
                    "colluder_gains": [],
                    "overlap_values": [],
                    "windowed_relay_count": 0,
                    "relay_latencies": [],
                    "relay_target_gains": [],
                    "relay_target_misdirected": [],
                    "evidence_ids": set(),
                    "posterior_delta_ids": set(),
                    "evidence_linked_count": 0,
                    "target_good_seats": set(),
                    "deception_record_count": 0,
                    "successful_deception_record_count": 0,
                    "peer_detected_deception_record_count": 0,
                    "audited_deception_types": Counter(),
                }
            return pair_stats[key]

        def add_refs_to_pair(stat: dict[str, Any], refs: dict[str, Any]) -> None:
            evidence_ids = {str(eid) for eid in refs.get("evidence_ids") or [] if eid}
            delta_ids = {str(eid) for eid in refs.get("posterior_delta_ids") or [] if eid}
            if evidence_ids:
                stat["evidence_linked_count"] += 1
                stat["evidence_ids"].update(evidence_ids)
            stat["posterior_delta_ids"].update(delta_ids)

        def add_pair_shift_samples(
            stat: dict[str, Any],
            *,
            shifts: list[dict[str, Any]],
            target: int | None,
            colluder: int | None,
        ) -> None:
            for shift in shifts:
                viewer = _as_int(shift.get("viewer_seat"))
                if viewer not in good_seats:
                    continue
                before = shift.get("before") or {}
                after = shift.get("after") or {}
                if target is not None:
                    target_before = before.get(str(target))
                    target_after = after.get(str(target))
                    if target_before is not None and target_after is not None:
                        gain = float(target_after) - float(target_before)
                        stat["target_gains"].append(gain)
                        stat["target_misdirected"].append(1 if gain > 0.02 else 0)
                if colluder is not None:
                    colluder_before = before.get(str(colluder))
                    colluder_after = after.get(str(colluder))
                    if colluder_before is not None and colluder_after is not None:
                        stat["colluder_gains"].append(float(colluder_after) - float(colluder_before))

        for idx, speech in enumerate(self._speech_log):
            speaker = _as_int(speech.get("seat"))
            if speaker not in wolf_seats:
                continue
            day = _as_int(speech.get("day")) or 0
            accuses = {
                target
                for target in (_as_int(raw) for raw in (speech.get("accuses") or []))
                if target in role_by_seat and target != speaker
            }
            attitudes = speech.get("attitudes") if isinstance(speech.get("attitudes"), dict) else {}
            supports = {
                target
                for target, stance in ((_as_int(raw), str(value)) for raw, value in attitudes.items())
                if target in role_by_seat and target != speaker and stance == "support"
            }
            opposed = {
                target
                for target, stance in ((_as_int(raw), str(value)) for raw, value in attitudes.items())
                if target in role_by_seat and target != speaker and stance == "oppose"
            }
            good_targets = (accuses | opposed) & good_seats
            wolf_supports = supports & wolf_seats
            for target in wolf_supports:
                support_edges.add((speaker, target))
            for target in good_targets:
                group = shared_targets.setdefault(
                    (day, target),
                    {"speakers": set(), "speech_indices": []},
                )
                group["speakers"].add(speaker)
                group["speech_indices"].append(idx)

            shifts_queue = shift_groups.get((day, speaker)) or []
            shifts = shifts_queue.pop(0) if shifts_queue else []
            speech_shifts[idx] = shifts
            for shift in shifts:
                viewer = _as_int(shift.get("viewer_seat"))
                if viewer not in good_seats:
                    continue
                before = shift.get("before") or {}
                after = shift.get("after") or {}
                speaker_before = before.get(str(speaker))
                speaker_after = after.get(str(speaker))
                if speaker_before is not None and speaker_after is not None:
                    colluder_suspicion_gains.append(float(speaker_after) - float(speaker_before))
                for target in good_targets:
                    b = before.get(str(target))
                    a = after.get(str(target))
                    if b is not None and a is not None:
                        target_gains_by_group.setdefault((day, target), []).append(float(a) - float(b))

            evidence_refs = self._speech_evidence_refs(
                shifts,
                day=day,
                speaker=speaker,
                target_seats=good_targets | wolf_supports,
            )
            speech_refs[idx] = evidence_refs
            wolf_speeches.append({
                "idx": idx,
                "day": day,
                "seat": speaker,
                "text": str(speech.get("text") or ""),
                "good_targets": good_targets,
                "wolf_supports": wolf_supports,
            })

        by_day: dict[int, list[dict[str, Any]]] = {}
        for speech in wolf_speeches:
            by_day.setdefault(int(speech["day"]), []).append(speech)

        overlap_values: list[float] = []
        overlap_records: list[dict[str, Any]] = []
        for day, speeches in by_day.items():
            seats = sorted({int(s["seat"]) for s in speeches})
            for i, a in enumerate(seats):
                for b in seats[i + 1:]:
                    active_pairs.add((a, b))
                    pair_stat(a, b)["active_days"].add(day)
            for i, first in enumerate(speeches):
                for second in speeches[i + 1:]:
                    if first["seat"] == second["seat"]:
                        continue
                    pair = pair_stat(int(first["seat"]), int(second["seat"]))
                    overlap = _jaccard(_bigrams(str(first["text"])), _bigrams(str(second["text"])))
                    overlap_values.append(overlap)
                    if overlap >= 0.25:
                        pair["narrative_overlap_pair_count"] += 1
                        pair["coordinated_pressure_count"] += 1
                        pair["overlap_values"].append(overlap)
                        overlap_records.append({
                            "type": "narrative_overlap",
                            "day": day,
                            "wolf_seats": sorted([int(first["seat"]), int(second["seat"])]),
                            "narrative_overlap": round(overlap, 3),
                            "shared_good_targets": sorted(set(first["good_targets"]) & set(second["good_targets"])),
                        })

        def merge_refs(indices: list[int]) -> dict[str, Any]:
            evidence_ids: set[str] = set()
            delta_ids: set[str] = set()
            source_types: Counter[str] = Counter()
            for speech_idx in indices:
                refs = speech_refs.get(speech_idx) or {}
                evidence_ids.update(str(eid) for eid in refs.get("evidence_ids") or [])
                delta_ids.update(str(eid) for eid in refs.get("posterior_delta_ids") or [])
                for source_type, count in (refs.get("evidence_source_types") or {}).items():
                    source_types[str(source_type)] += int(count)
            return {
                "evidence_ids": sorted(evidence_ids)[:16],
                "posterior_delta_ids": sorted(delta_ids)[:16],
                "evidence_source_types": dict(sorted(source_types.items())),
            }

        def relay_target_gains_for(speech_idx: int, targets: set[int]) -> list[float]:
            values: list[float] = []
            if not targets:
                return values
            for shift in speech_shifts.get(speech_idx, []):
                viewer = _as_int(shift.get("viewer_seat"))
                if viewer not in good_seats:
                    continue
                before = shift.get("before") or {}
                after = shift.get("after") or {}
                viewer_gains: list[float] = []
                for target in targets:
                    b = before.get(str(target))
                    a = after.get(str(target))
                    if b is None or a is None:
                        continue
                    viewer_gains.append(float(a) - float(b))
                if viewer_gains:
                    values.append(_mean(viewer_gains))
            return values

        relay_records: list[dict[str, Any]] = []
        all_relay_latencies: list[float] = []
        all_relay_target_gains: list[float] = []
        all_relay_target_misdirected: list[int] = []
        evidence_linked_count = 0
        relay_window_speeches = 4
        for day, speeches in by_day.items():
            ordered = sorted(speeches, key=lambda s: int(s["idx"]))
            for i, first in enumerate(ordered):
                for second in ordered[i + 1:]:
                    first_seat = int(first["seat"])
                    second_seat = int(second["seat"])
                    if first_seat == second_seat:
                        continue
                    latency = int(second["idx"]) - int(first["idx"])
                    if latency < 1 or latency > relay_window_speeches:
                        continue
                    shared_good_targets = set(first["good_targets"]) & set(second["good_targets"])
                    follower_supports_lead = first_seat in set(second["wolf_supports"])
                    if not shared_good_targets and not follower_supports_lead:
                        continue

                    stat = pair_stat(first_seat, second_seat)
                    stat["active_days"].add(day)
                    stat["windowed_relay_count"] += 1
                    stat["relay_latencies"].append(float(latency))
                    all_relay_latencies.append(float(latency))

                    target_gains = relay_target_gains_for(int(second["idx"]), shared_good_targets)
                    if target_gains:
                        stat["relay_target_gains"].extend(target_gains)
                        all_relay_target_gains.extend(target_gains)
                        misdirected = [1 if gain > 0.02 else 0 for gain in target_gains]
                        stat["relay_target_misdirected"].extend(misdirected)
                        all_relay_target_misdirected.extend(misdirected)

                    refs = merge_refs([int(first["idx"]), int(second["idx"])])
                    if refs["evidence_ids"]:
                        evidence_linked_count += 1
                    add_refs_to_pair(stat, refs)
                    relay_records.append({
                        "type": "windowed_relay",
                        "day": day,
                        "wolf_seats": sorted([first_seat, second_seat]),
                        "lead_wolf_seat": first_seat,
                        "follow_wolf_seat": second_seat,
                        "relay_latency": latency,
                        "shared_good_targets": sorted(shared_good_targets),
                        "follower_supports_lead": bool(follower_supports_lead),
                        "avg_target_suspicion_gain": _rounded_mean(target_gains),
                        **refs,
                    })

        shared_records: list[dict[str, Any]] = []
        shared_coverages: list[float] = []
        shared_target_gains: list[float] = []
        for (day, target), group in sorted(shared_targets.items()):
            speakers = sorted(int(seat) for seat in group["speakers"])
            if len(speakers) < 2:
                continue
            coverage = len(speakers) / len(wolf_seats) if wolf_seats else 0.0
            shared_coverages.append(coverage)
            gains = target_gains_by_group.get((day, target), [])
            if gains:
                shared_target_gains.extend(gains)
            refs = merge_refs(list(group["speech_indices"]))
            if refs["evidence_ids"]:
                evidence_linked_count += 1
            group_speeches = [
                speech for speech in wolf_speeches
                if speech["idx"] in set(group["speech_indices"])
            ]
            max_overlap = 0.0
            for i, first in enumerate(group_speeches):
                for second in group_speeches[i + 1:]:
                    if first["seat"] == second["seat"]:
                        continue
                    max_overlap = max(
                        max_overlap,
                        _jaccard(_bigrams(str(first["text"])), _bigrams(str(second["text"]))),
                    )
            shared_records.append({
                "type": "shared_good_target",
                "day": day,
                "target_good_seat": target,
                "wolf_seats": speakers,
                "speaker_coverage": round(coverage, 3),
                "avg_target_suspicion_gain": _rounded_mean(gains),
                "max_narrative_overlap": round(max_overlap, 3),
                **refs,
            })
            for i, first in enumerate(speakers):
                for second in speakers[i + 1:]:
                    stat = pair_stat(first, second)
                    stat["active_days"].add(day)
                    stat["shared_good_target_count"] += 1
                    stat["coordinated_pressure_count"] += 1
                    stat["target_good_seats"].add(target)
                    add_refs_to_pair(stat, refs)
                    for speech_idx in group["speech_indices"]:
                        speech = next((item for item in wolf_speeches if item["idx"] == speech_idx), None)
                        if not speech or int(speech["seat"]) not in {first, second}:
                            continue
                        add_pair_shift_samples(
                            stat,
                            shifts=speech_shifts.get(int(speech_idx), []),
                            target=target,
                            colluder=int(speech["seat"]),
                        )

        mutual_support_pairs = {
            tuple(sorted((source, target)))
            for source, target in support_edges
            if (target, source) in support_edges
        }
        for source, target in support_edges:
            stat = pair_stat(source, target)
            stat["wolf_to_wolf_support_count"] += 1
            if tuple(sorted((source, target))) in mutual_support_pairs:
                stat["mutual_support_pair_count"] = 1
        support_records: list[dict[str, Any]] = []
        for source, target in sorted(support_edges):
            idxs = [
                int(speech["idx"])
                for speech in wolf_speeches
                if speech["seat"] == source and target in speech["wolf_supports"]
            ]
            refs = merge_refs(idxs)
            if refs["evidence_ids"]:
                evidence_linked_count += 1
            stat = pair_stat(source, target)
            add_refs_to_pair(stat, refs)
            support_records.append({
                "type": "wolf_support",
                "day": next((int(s["day"]) for s in wolf_speeches if s["idx"] in idxs), None),
                "source_wolf_seat": source,
                "target_wolf_seat": target,
                "mutual": tuple(sorted((source, target))) in mutual_support_pairs,
                **refs,
            })

        records = (shared_records + support_records + overlap_records + relay_records)[-40:]
        shared_good_target_count = len(shared_records)
        narrative_overlap_pair_count = len(overlap_records)
        coordinated_pressure_count = (
            shared_good_target_count
            + len(mutual_support_pairs)
            + narrative_overlap_pair_count
        )
        for pair in mutual_support_pairs:
            pair_stat(pair[0], pair[1])["coordinated_pressure_count"] += 1

        deception_records = self._deception_audit_from_roles(role_by_seat).get("records") or []
        for record in deception_records:
            if not isinstance(record, dict):
                continue
            seat = _as_int(record.get("seat"))
            day = _as_int(record.get("day"))
            if seat not in wolf_seats:
                continue
            target_goods = {
                target
                for target in (_as_int(raw) for raw in (record.get("target_good_seats") or []))
                if target in good_seats
            }
            for key, stat in pair_stats.items():
                seats = set(stat["wolf_seats"])
                if seat not in seats:
                    continue
                days = stat["active_days"]
                targets = stat["target_good_seats"]
                if day not in days and not (target_goods & targets):
                    continue
                stat["deception_record_count"] += 1
                if record.get("successful_misdirection"):
                    stat["successful_deception_record_count"] += 1
                peer_detection = record.get("peer_detection") if isinstance(record.get("peer_detection"), dict) else {}
                if peer_detection.get("detected"):
                    stat["peer_detected_deception_record_count"] += 1
                for audit_type in record.get("audited_types") or []:
                    stat["audited_deception_types"][str(audit_type)] += 1

        pair_listener_susceptibility_by_pair: dict[str, dict[str, Any]] = {}
        all_pair_target_gains: list[float] = []
        all_pair_misdirected: list[int] = []
        deception_linked_pair_count = 0
        for key, stat in sorted(pair_stats.items()):
            target_gains = [float(value) for value in stat["target_gains"]]
            target_misdirected = [int(value) for value in stat["target_misdirected"]]
            colluder_gains = [float(value) for value in stat["colluder_gains"]]
            relay_target_gains = [float(value) for value in stat["relay_target_gains"]]
            relay_target_misdirected = [int(value) for value in stat["relay_target_misdirected"]]
            all_pair_target_gains.extend(target_gains)
            all_pair_misdirected.extend(target_misdirected)
            if stat["deception_record_count"]:
                deception_linked_pair_count += 1
            pair_listener_susceptibility_by_pair[key] = {
                "wolf_seats": stat["wolf_seats"],
                "active_days": sorted(stat["active_days"]),
                "shared_good_target_count": stat["shared_good_target_count"],
                "wolf_to_wolf_support_count": stat["wolf_to_wolf_support_count"],
                "mutual_support_pair_count": stat["mutual_support_pair_count"],
                "narrative_overlap_pair_count": stat["narrative_overlap_pair_count"],
                "coordinated_pressure_count": stat["coordinated_pressure_count"],
                "target_shift_sample_count": len(target_gains),
                "avg_target_suspicion_gain": _rounded_mean(target_gains),
                "target_misdirected_rate": _rate(sum(target_misdirected), len(target_misdirected)),
                "colluder_shift_sample_count": len(colluder_gains),
                "avg_colluder_suspicion_gain": _rounded_mean(colluder_gains),
                "avg_narrative_overlap": _rounded_mean(stat["overlap_values"]),
                "windowed_relay_count": stat["windowed_relay_count"],
                "avg_windowed_relay_latency": _rounded_mean(stat["relay_latencies"]),
                "avg_relay_target_suspicion_gain": _rounded_mean(relay_target_gains),
                "relay_target_misdirected_rate": _rate(sum(relay_target_misdirected), len(relay_target_misdirected)),
                "evidence_linked_count": stat["evidence_linked_count"],
                "deception_record_count": stat["deception_record_count"],
                "successful_deception_record_count": stat["successful_deception_record_count"],
                "peer_detected_deception_record_count": stat["peer_detected_deception_record_count"],
                "audited_deception_types": dict(sorted(stat["audited_deception_types"].items())),
                "evidence_ids": sorted(stat["evidence_ids"])[:16],
                "posterior_delta_ids": sorted(stat["posterior_delta_ids"])[:16],
            }
        return {
            "wolf_speech_count": len(wolf_speeches),
            "wolf_pair_count": wolf_pair_count,
            "active_wolf_pair_count": len(active_pairs),
            "wolf_to_wolf_support_count": len(support_edges),
            "mutual_support_pair_count": len(mutual_support_pairs),
            "shared_good_target_count": shared_good_target_count,
            "shared_good_target_speaker_coverage": _rounded_mean(shared_coverages),
            "narrative_overlap_pair_count": narrative_overlap_pair_count,
            "avg_narrative_overlap": _rounded_mean(overlap_values),
            "coordinated_pressure_count": coordinated_pressure_count,
            "avg_shared_target_suspicion_gain": _rounded_mean(shared_target_gains),
            "avg_colluder_suspicion_gain": _rounded_mean(colluder_suspicion_gains),
            "evidence_linked_count": evidence_linked_count,
            "pair_listener_shift_sample_count": len(all_pair_target_gains),
            "avg_pair_target_suspicion_gain": _rounded_mean(all_pair_target_gains),
            "pair_target_misdirected_rate": _rate(sum(all_pair_misdirected), len(all_pair_misdirected)),
            "windowed_relay_count": len(relay_records),
            "avg_windowed_relay_latency": _rounded_mean(all_relay_latencies),
            "avg_relay_target_suspicion_gain": _rounded_mean(all_relay_target_gains),
            "relay_target_misdirected_rate": _rate(sum(all_relay_target_misdirected), len(all_relay_target_misdirected)),
            "deception_linked_pair_count": deception_linked_pair_count,
            "pair_listener_susceptibility_by_pair": pair_listener_susceptibility_by_pair,
            "records": records,
        }

    def _deception_audit_from_roles(self, role_by_seat: dict[int, Role]) -> dict[str, Any]:
        wolf_seats = {seat for seat, role in role_by_seat.items() if role == Role.WEREWOLF}
        good_seats = set(role_by_seat) - wolf_seats
        true_seers = {seat for seat, role in role_by_seat.items() if role == Role.SEER}

        shift_groups = self._posterior_speech_shift_groups()

        wolf_speech_count = 0
        declared_deception_count = 0
        audited_deception_count = 0
        declared_matches = 0
        declared_or_audited = 0
        successful_misdirection_count = 0
        target_good_audit_count = 0
        all_good_target_gains: list[float] = []
        all_speaker_suspicion_gains: list[float] = []
        listener_shift_sample_count = 0
        detected_deception_count = 0
        peer_detection_opportunity_count = 0
        evidence_linked_count = 0
        listener_stats: dict[int, dict[str, list[float] | list[int]]] = {}
        declared_by_type: Counter[str] = Counter()
        audited_by_type: Counter[str] = Counter()
        records: list[dict[str, Any]] = []

        good_accuse_count = 0
        good_false_positive_count = 0

        for speech in self._speech_log:
            speaker = _as_int(speech.get("seat"))
            if speaker not in role_by_seat:
                continue
            day = _as_int(speech.get("day")) or 0
            accuses = {_as_int(target) for target in (speech.get("accuses") or [])}
            accuses = {target for target in accuses if target is not None and target in role_by_seat}
            attitudes = speech.get("attitudes") if isinstance(speech.get("attitudes"), dict) else {}
            opposed = {
                _as_int(target)
                for target, stance in (attitudes or {}).items()
                if str(stance) == "oppose"
            }
            opposed = {target for target in opposed if target is not None and target in role_by_seat}

            if speaker in good_seats:
                for target in accuses:
                    if target == speaker:
                        continue
                    good_accuse_count += 1
                    if target in good_seats:
                        good_false_positive_count += 1
                continue

            if speaker not in wolf_seats:
                continue

            wolf_speech_count += 1
            declared = str(speech.get("deception") or "none").strip().lower()
            declared = declared if declared and declared != "none" else None
            if declared:
                declared_deception_count += 1
                declared_by_type[declared] += 1

            audited_types: set[str] = set()
            good_targets = (accuses | opposed) & good_seats
            wolf_targets = (accuses | opposed) & wolf_seats
            wolf_targets.discard(speaker)
            claim_checked: int | None = None

            claim = speech.get("claim")
            if isinstance(claim, dict):
                role_claim = str(claim.get("role") or "").lower()
                checked = _as_int(claim.get("checked_seat"))
                claim_checked = checked if checked in role_by_seat else None
                result = str(claim.get("result") or "").lower()
                if role_claim == "seer" and role_by_seat.get(speaker) != Role.SEER:
                    audited_types.add("fabrication")
                if checked in role_by_seat and result in {"wolf", "werewolf", "werewolves", "狼人", "village", "villager", "好人"}:
                    truth_is_wolf = checked in wolf_seats
                    claimed_wolf = result in {"wolf", "werewolf", "werewolves", "狼人"}
                    if claimed_wolf != truth_is_wolf:
                        audited_types.add("fabrication")

            text = str(speech.get("text") or "").lower()
            if any(token in text for token in ("我不是狼", "我是好人", "我是村民", "好人牌", "平民")):
                audited_types.add("fabrication")

            if good_targets:
                audited_types.add("misdirection")

            seer_targets = ((accuses | opposed) & true_seers)
            if seer_targets and any(token in text for token in ("预言家", "查", "验", "跳", "悍跳", "假", "急", "细节")):
                audited_types.add("distortion")

            if declared or audited_types:
                declared_or_audited += 1
                if declared and declared in audited_types:
                    declared_matches += 1

            if audited_types:
                audited_deception_count += 1
                for audit_type in audited_types:
                    audited_by_type[audit_type] += 1

            shifts_queue = shift_groups.get((day, speaker)) or []
            shifts = shifts_queue.pop(0) if shifts_queue else []
            gains: list[float] = []
            speaker_gains: list[float] = []
            detected_viewers: set[int] = set()
            listener_shift_records: list[dict[str, Any]] = []
            if good_targets:
                for shift in shifts:
                    viewer = _as_int(shift.get("viewer_seat"))
                    if viewer not in good_seats:
                        continue
                    before = shift.get("before") or {}
                    after = shift.get("after") or {}
                    target_gains: list[float] = []
                    for target in good_targets:
                        b = before.get(str(target))
                        a = after.get(str(target))
                        if b is None or a is None:
                            continue
                        gain = float(a) - float(b)
                        target_gains.append(gain)
                        gains.append(gain)
                    speaker_before = before.get(str(speaker))
                    speaker_after = after.get(str(speaker))
                    speaker_gain = (
                        float(speaker_after) - float(speaker_before)
                        if speaker_before is not None and speaker_after is not None
                        else None
                    )
                    if speaker_gain is not None:
                        speaker_gains.append(speaker_gain)
                        all_speaker_suspicion_gains.append(speaker_gain)
                        if speaker_gain > 0.02:
                            detected_viewers.add(viewer)
                    avg_target_gain = _mean(target_gains) if target_gains else None
                    if avg_target_gain is not None or speaker_gain is not None:
                        listener_shift_sample_count += 1
                        stats = listener_stats.setdefault(
                            viewer,
                            {
                                "target_gains": [],
                                "misdirected": [],
                                "speaker_gains": [],
                                "detected": [],
                            },
                        )
                        if avg_target_gain is not None:
                            stats["target_gains"].append(avg_target_gain)
                            stats["misdirected"].append(1 if avg_target_gain > 0.02 else 0)
                        if speaker_gain is not None:
                            stats["speaker_gains"].append(speaker_gain)
                            stats["detected"].append(1 if speaker_gain > 0.02 else 0)
                        listener_shift_records.append({
                            "viewer_seat": viewer,
                            "target_good_suspicion_gain": (
                                round(avg_target_gain, 3) if avg_target_gain is not None else None
                            ),
                            "speaker_suspicion_gain": (
                                round(speaker_gain, 3) if speaker_gain is not None else None
                            ),
                            "misdirected": bool(avg_target_gain is not None and avg_target_gain > 0.02),
                            "detected_speaker": bool(speaker_gain is not None and speaker_gain > 0.02),
                        })
                if gains:
                    all_good_target_gains.extend(gains)
                if "misdirection" in audited_types and gains:
                    target_good_audit_count += 1
                    avg_gain = _mean(gains) if gains else 0.0
                    if avg_gain > 0.02:
                        successful_misdirection_count += 1
            else:
                for shift in shifts:
                    viewer = _as_int(shift.get("viewer_seat"))
                    if viewer not in good_seats:
                        continue
                    before = shift.get("before") or {}
                    after = shift.get("after") or {}
                    speaker_before = before.get(str(speaker))
                    speaker_after = after.get(str(speaker))
                    if speaker_before is None or speaker_after is None:
                        continue
                    speaker_gain = float(speaker_after) - float(speaker_before)
                    speaker_gains.append(speaker_gain)
                    all_speaker_suspicion_gains.append(speaker_gain)
                    if speaker_gain > 0.02:
                        detected_viewers.add(viewer)
                    listener_shift_sample_count += 1
                    stats = listener_stats.setdefault(
                        viewer,
                        {
                            "target_gains": [],
                            "misdirected": [],
                            "speaker_gains": [],
                            "detected": [],
                        },
                    )
                    stats["speaker_gains"].append(speaker_gain)
                    stats["detected"].append(1 if speaker_gain > 0.02 else 0)
                    listener_shift_records.append({
                        "viewer_seat": viewer,
                        "target_good_suspicion_gain": None,
                        "speaker_suspicion_gain": round(speaker_gain, 3),
                        "misdirected": False,
                        "detected_speaker": bool(speaker_gain > 0.02),
                    })

            if audited_types and speaker_gains:
                peer_detection_opportunity_count += 1
                if detected_viewers:
                    detected_deception_count += 1

            target_seats = set(good_targets) | set(wolf_targets) | set(seer_targets)
            if claim_checked is not None:
                target_seats.add(claim_checked)
            evidence_refs = self._speech_evidence_refs(
                shifts,
                day=day,
                speaker=speaker,
                target_seats=target_seats,
            )
            if evidence_refs["evidence_ids"]:
                evidence_linked_count += 1

            if audited_types or declared:
                records.append({
                    "day": day,
                    "seat": speaker,
                    "declared": declared,
                    "audited_types": sorted(audited_types),
                    "target_good_seats": sorted(good_targets),
                    "target_wolf_seats": sorted(wolf_targets),
                    "avg_good_target_suspicion_gain": _rounded_mean(gains),
                    "successful_misdirection": bool(gains and _mean(gains) > 0.02),
                    "evidence_ids": evidence_refs["evidence_ids"],
                    "posterior_delta_ids": evidence_refs["posterior_delta_ids"],
                    "evidence_source_types": evidence_refs["evidence_source_types"],
                    "listener_shifts": listener_shift_records[:12],
                    "peer_detection": {
                        "detected": bool(detected_viewers),
                        "detector_seats": sorted(detected_viewers),
                        "avg_speaker_suspicion_gain": _rounded_mean(speaker_gains),
                    },
                })

        listener_susceptibility_by_seat: dict[str, dict[str, Any]] = {}
        for seat, stats in sorted(listener_stats.items()):
            target_gains = [float(v) for v in stats.get("target_gains", [])]
            misdirected = [int(v) for v in stats.get("misdirected", [])]
            speaker_gains = [float(v) for v in stats.get("speaker_gains", [])]
            detected = [int(v) for v in stats.get("detected", [])]
            listener_susceptibility_by_seat[str(seat)] = {
                "misdirection_samples": len(target_gains),
                "avg_good_target_suspicion_gain": _rounded_mean(target_gains),
                "misdirected_rate": _rate(sum(misdirected), len(misdirected)),
                "detection_samples": len(speaker_gains),
                "avg_speaker_suspicion_gain": _rounded_mean(speaker_gains),
                "peer_detection_rate": _rate(sum(detected), len(detected)),
            }

        return {
            "wolf_speech_count": wolf_speech_count,
            "declared_deception_count": declared_deception_count,
            "audited_deception_count": audited_deception_count,
            "declared_vs_audited_agreement": _rate(declared_matches, declared_or_audited),
            "deception_success_rate": _rate(successful_misdirection_count, target_good_audit_count),
            "successful_misdirection_count": successful_misdirection_count,
            "target_good_audit_count": target_good_audit_count,
            "misdirection_shift_coverage": _rate(target_good_audit_count, audited_by_type.get("misdirection", 0)),
            "unauditable_misdirection_count": max(
                0,
                int(audited_by_type.get("misdirection", 0)) - target_good_audit_count,
            ),
            "avg_good_target_suspicion_gain": _rounded_mean(all_good_target_gains),
            "detected_deception_count": detected_deception_count,
            "peer_detection_opportunity_count": peer_detection_opportunity_count,
            "peer_detection_rate": _rate(detected_deception_count, peer_detection_opportunity_count),
            "avg_speaker_suspicion_gain": _rounded_mean(all_speaker_suspicion_gains),
            "listener_shift_sample_count": listener_shift_sample_count,
            "evidence_linked_count": evidence_linked_count,
            "listener_susceptibility_by_seat": listener_susceptibility_by_seat,
            "villager_false_positive_rate": _rate(good_false_positive_count, good_accuse_count),
            "villager_false_positive_count": good_false_positive_count,
            "good_accuse_count": good_accuse_count,
            "declared_by_type": dict(sorted(declared_by_type.items())),
            "audited_by_type": dict(sorted(audited_by_type.items())),
            "records": records[-40:],
        }

    def _posterior_metrics(self) -> dict[str, Any]:
        """赛后信念轨迹指标,基于 EvidenceGraph posterior snapshots。

        使用真实阵营只做赛后校准评测,绝不进入 live prompt。核心目标是把
        persuasion/deception 从"judge 觉得有说服力"推进到"发言后他人信念
        是否发生了可解释变化"。
        """
        role_by_seat = {
            p.seat: Role(p.role)
            for p in self.state.players
            if p.role is not None
        }
        wolf_seats = {seat for seat, role in role_by_seat.items() if role == Role.WEREWOLF}
        good_seats = set(role_by_seat) - wolf_seats

        snapshots = list(self._posterior_log)
        speech_snapshots = [
            s for s in snapshots if s.get("trigger") in {"speech", "pk_speech"}
        ]
        if not snapshots:
            return {
                "snapshot_count": 0,
                "speech_snapshot_count": 0,
                "avg_speech_posterior_shift": None,
                "good_final_wolf_suspicion_gap": None,
                "good_final_top_suspect_accuracy": None,
                "herding_index": None,
                "herding_event_count": 0,
                "correct_herding_rate": None,
                "wrong_herding_rate": None,
                "final_brier_score": None,
                "final_log_loss": None,
                "good_final_brier_score": None,
                "good_final_log_loss": None,
                "constrained_final_brier_score": None,
                "constrained_final_log_loss": None,
                "constrained_good_final_brier_score": None,
                "constrained_good_final_log_loss": None,
                "constrained_calibration_ece": None,
                "constrained_calibration_bins": [],
                "calibration_ece": None,
                "calibration_bins": [],
            }

        shifts: list[float] = []
        previous_by_viewer_day: dict[tuple[int, int], dict[str, float]] = {}
        for snap in snapshots:
            viewer = _as_int(snap.get("viewer_seat"))
            day = _as_int(snap.get("day"))
            if viewer is None or day is None:
                continue
            current = _posterior_values(snap)
            if snap.get("trigger") in {"speech", "pk_speech"}:
                prev = previous_by_viewer_day.get((viewer, day))
                if prev:
                    common = set(prev) & set(current)
                    if common:
                        shifts.append(sum(abs(current[s] - prev[s]) for s in common) / len(common))
            previous_by_viewer_day[(viewer, day)] = current

        latest_by_viewer: dict[int, dict[str, Any]] = {}
        for snap in snapshots:
            viewer = _as_int(snap.get("viewer_seat"))
            if viewer is not None:
                latest_by_viewer[viewer] = snap

        good_gaps: list[float] = []
        top_hits = 0
        top_total = 0
        final_records: list[tuple[float, int]] = []
        good_final_records: list[tuple[float, int]] = []
        constrained_final_records: list[tuple[float, int]] = []
        constrained_good_final_records: list[tuple[float, int]] = []
        for viewer, snap in latest_by_viewer.items():
            posterior = _posterior_values(snap)
            constrained_posterior = _posterior_values(snap, "constrained_posterior")
            for seat, role in role_by_seat.items():
                if seat == viewer:
                    continue
                value = posterior.get(str(seat))
                target = 1 if role == Role.WEREWOLF else 0
                if value is not None:
                    record = (value, target)
                    final_records.append(record)
                    if viewer in good_seats:
                        good_final_records.append(record)
                constrained_value = constrained_posterior.get(str(seat))
                if constrained_value is not None:
                    constrained_record = (constrained_value, target)
                    constrained_final_records.append(constrained_record)
                    if viewer in good_seats:
                        constrained_good_final_records.append(constrained_record)

        for viewer in sorted(good_seats):
            snap = latest_by_viewer.get(viewer)
            if not snap:
                continue
            posterior = _posterior_values(snap)
            wolf_vals = [posterior[str(seat)] for seat in wolf_seats if str(seat) in posterior]
            good_vals = [
                posterior[str(seat)]
                for seat in good_seats
                if seat != viewer and str(seat) in posterior
            ]
            if wolf_vals and good_vals:
                good_gaps.append(_mean(wolf_vals) - _mean(good_vals))
            top = _top_suspect_seat(snap)
            if top is not None:
                top_total += 1
                if top in wolf_seats:
                    top_hits += 1

        day_viewer_latest: dict[tuple[int, int], dict[str, Any]] = {}
        for snap in speech_snapshots:
            day = _as_int(snap.get("day"))
            viewer = _as_int(snap.get("viewer_seat"))
            if day is None or viewer not in good_seats:
                continue
            day_viewer_latest[(day, viewer)] = snap
        tops_by_day: dict[int, list[int]] = {}
        for (day, _viewer), snap in day_viewer_latest.items():
            top = _top_suspect_seat(snap)
            if top is not None:
                tops_by_day.setdefault(day, []).append(top)
        herding_scores: list[float] = []
        herding_event_count = 0
        correct_herding_count = 0
        wrong_herding_count = 0
        for tops in tops_by_day.values():
            if len(tops) < 2:
                continue
            counts = Counter(tops)
            consensus_target, consensus_count = counts.most_common(1)[0]
            score = consensus_count / len(tops)
            herding_scores.append(score)
            if score > 0.5:
                herding_event_count += 1
                if consensus_target in wolf_seats:
                    correct_herding_count += 1
                elif consensus_target in good_seats:
                    wrong_herding_count += 1

        bins, ece = _calibration_bins(good_final_records)
        constrained_bins, constrained_ece = _calibration_bins(constrained_good_final_records)

        return {
            "snapshot_count": len(snapshots),
            "speech_snapshot_count": len(speech_snapshots),
            "avg_speech_posterior_shift": _rounded_mean(shifts),
            "good_final_wolf_suspicion_gap": _rounded_mean(good_gaps),
            "good_final_top_suspect_accuracy": _rate(top_hits, top_total),
            "herding_index": _rounded_mean(herding_scores),
            "herding_event_count": herding_event_count,
            "correct_herding_rate": _rate(correct_herding_count, herding_event_count),
            "wrong_herding_rate": _rate(wrong_herding_count, herding_event_count),
            "final_brier_score": _brier(final_records),
            "final_log_loss": _log_loss(final_records),
            "good_final_brier_score": _brier(good_final_records),
            "good_final_log_loss": _log_loss(good_final_records),
            "constrained_final_brier_score": _brier(constrained_final_records),
            "constrained_final_log_loss": _log_loss(constrained_final_records),
            "constrained_good_final_brier_score": _brier(constrained_good_final_records),
            "constrained_good_final_log_loss": _log_loss(constrained_good_final_records),
            "constrained_calibration_ece": constrained_ece,
            "constrained_calibration_bins": constrained_bins,
            "calibration_ece": ece,
            "calibration_bins": bins,
        }

    def _parse_metrics(self) -> dict[str, Any]:
        failed = [d for d in self._parse_decisions if d.get("parse_failed")]
        failed_by_action = Counter(str(d.get("action") or "unknown") for d in failed)
        failed_by_phase = Counter(str(d.get("phase") or "unknown") for d in failed)
        decision_count = len(self._parse_decisions)
        parse_failed_count = len(failed)
        return {
            "decision_count": decision_count,
            "parse_failed_count": parse_failed_count,
            "parse_failed_rate": (
                parse_failed_count / decision_count if decision_count else None
            ),
            "parse_failed_by_action": dict(sorted(failed_by_action.items())),
            "parse_failed_by_phase": dict(sorted(failed_by_phase.items())),
        }

    def _decision_failure_metrics(self) -> dict[str, Any]:
        failures = list(self._decision_failures)
        by_phase = Counter(str(f.get("phase") or "unknown") for f in failures)
        by_action = Counter(str(f.get("action") or "unknown") for f in failures)
        by_seat = Counter(str(f.get("seat") or "unknown") for f in failures)
        by_error_type = Counter(str(f.get("error_type") or "unknown") for f in failures)
        timeout_count = sum(1 for f in failures if f.get("timeout"))
        return {
            "failure_count": len(failures),
            "timeout_count": timeout_count,
            "by_phase": dict(sorted(by_phase.items())),
            "by_action": dict(sorted(by_action.items())),
            "by_seat": dict(sorted(by_seat.items())),
            "by_error_type": dict(sorted(by_error_type.items())),
            "records": failures[-80:],
        }

    async def _run_analysis(self) -> None:
        analysis = {
            "winner": self.state.winner,
            "days": self.state.day,
            "seats": [
                {"seat": p.seat, "name": p.name, "role": p.role, "team": p.team,
                 "alive": p.alive, "death_reason": p.death_reason, "death_day": p.death_day}
                for p in self.state.players
            ],
            "agent_summaries": [
                {"seat": a.seat, "role": a.role.value, "persona": a.persona_name,
                 "trust_final": a.memory.snapshot().get("trust", {}),
                 "claims": a.memory.snapshot().get("claims", {})}
                for a in self.actors.values()
            ],
        }
        # 方向A/B/C 对话对抗量化指标(客观,独立于 LLM 评审):
        # 回应率/指控率/态度网络密度/狼人协同度——用于跨对局对比 A/B/C 落地前后提升。
        analysis["turn_policy"] = self.turn_policy
        analysis["dialogue_metrics"] = self._dialogue_metrics()
        analysis["debate_process_metrics"] = self._debate_process_metrics()
        analysis["objective_metrics"] = self._objective_metrics()
        analysis["posterior_metrics"] = self._posterior_metrics()
        analysis["posterior_trace"] = list(self._posterior_log)
        analysis["posterior_trace_total_count"] = len(self._posterior_log)
        analysis["posterior_trace_truncated"] = False
        analysis["posterior_trace_dropped_count"] = 0
        analysis["parse_metrics"] = self._parse_metrics()
        analysis["decision_failure_metrics"] = self._decision_failure_metrics()
        analysis["deception_audit"] = self._deception_audit()
        analysis["collusion_audit"] = self._collusion_audit()

        # 五维对局质量评分(Beyond Survival WereAlign)。真实 LLM 调用,失败不致命。
        quality: dict[str, Any] | None = None
        try:
            any_actor = next(iter(self.actors.values()))
            from ..agent.quality import score_game_quality
            judge_config = any_actor.model_config.model_copy(update={"temperature": 0.0})
            quality = await score_game_quality(
                router=any_actor.router,
                config=judge_config,
                winner=self.state.winner.value if self.state.winner else None,
                days=self.state.day,
                seats=[{"seat": p.seat, "name": p.name, "role": p.role,
                        "team": p.team, "alive": p.alive} for p in self.state.players],
                speeches=self._speech_log,
                votes=self._vote_log,
                thinking_digest=self._thinking_log,
            )
        except Exception as err:  # noqa: BLE001
            logger.warning("对局质量评分异常(跳过,不致命): %s", err)
        if quality is not None:
            analysis["quality"] = quality

        await self._emit_game_ended()
        await self._emit({"type": "analysis", "analysis": analysis})

    async def _notify_role_assigned(self) -> None:
        for pid, actor in self.actors.items():
            player = self.state.get_player(pid)
            teammates = []
            if actor.role == Role.WEREWOLF:
                teammates = [
                    {"seat": p.seat, "name": p.name}
                    for p in self.state.players
                    if p.role == Role.WEREWOLF and p.id != pid
                ]
            actor.observe_event(0, "setup", "role_assigned", f"你的身份是{actor.role.value}",
                                role=actor.role.value, teammates=teammates)
            if actor.role == Role.WEREWOLF:
                for t in teammates:
                    actor.observe_event(0, "setup", "teammate", f"你的狼队友是{t['seat']}号{t['name']}")

    async def _broadcast_phase(self, phase: Phase) -> None:
        pass

    async def _emit_game_ended(self) -> None:
        """Broadcast game_ended exactly once; analysis remains the final replay event."""
        if self._game_ended_emitted:
            return
        self._game_ended_emitted = True
        await self._emit({"type": "game_ended",
                          "winner": self.state.winner.value if self.state.winner else None})

    async def _emit(self, payload: dict[str, Any]) -> None:
        # 收集评分用子集
        etype = payload.get("type")
        if etype == "agent_decision_failed":
            reason = str(payload.get("reason") or payload.get("error") or "")
            self._decision_failures.append({
                "day": self.state.day,
                "phase": payload.get("phase"),
                "seat": payload.get("seat"),
                "action": payload.get("action"),
                "error_type": payload.get("error_type"),
                "reason": reason[:240],
                "timeout": bool(payload.get("timeout") or "timeout" in reason.lower()),
                "timeout_seconds": payload.get("timeout_seconds"),
            })
        if etype == "speech":
            self._speech_log.append({"seat": payload.get("seat"), "text": payload.get("text", ""),
                                     "claim": payload.get("claim"), "day": payload.get("day"),
                                     "bid": payload.get("bid"),
                                     # 方向A/B 结构化对话关系:供5维评分识别对话交锋/态度网络
                                     "reply_to": payload.get("reply_to"),
                                     "accuses": payload.get("accuses"),
                                     "attitudes": payload.get("attitudes"),
                                     "deception": payload.get("_analysis_deception",
                                                              payload.get("deception"))})
        if self.on_event:
            try:
                public_payload = {
                    key: value
                    for key, value in payload.items()
                    if not str(key).startswith("_analysis_")
                }
                await self.on_event(public_payload)
            except Exception as err:  # noqa: BLE001
                logger.debug("on_event 回调失败: %s", err)

    async def _emit_thinking(self, actor: AgentActor, decision: Decision) -> None:
        if not isinstance(decision, Decision):
            return
        if self.on_thinking:
            try:
                thinking = actor.thinking_summary(decision, verbose=self.verbose_thinking)
                await self._emit_thinking_payload(thinking)
            except Exception as err:  # noqa: BLE001
                logger.debug("on_thinking 回调失败: %s", err)

    async def _emit_thinking_payload(self, thinking: AgentThinking) -> None:
        if not self.on_thinking:
            return
        self._thinking_log.append({
            "seat": thinking.seat,
            "action": thinking.action,
            "summary": thinking.summary,
            "reasoning": thinking.reasoning,
        })
        await self.on_thinking(thinking.model_dump())

    def _last_event_message(self, etype: str) -> str | None:
        for ev in reversed(self.state.events):
            if ev.type == etype:
                return ev.message
        return None


def build_actors(
    state: GameState,
    *,
    model_config: ModelConfig,
    router: LLMRouter,
    seat_configs: dict[int, ModelConfig] | None = None,
    human_seats: set[int] | None = None,
    rng: random.Random | None = None,
) -> dict[str, AgentActor]:
    rng = rng or random.Random()
    human_seats = human_seats or set()
    actors: dict[str, AgentActor] = {}
    for player in state.players:
        cfg = model_config.merge((seat_configs or {}).get(player.seat))
        actor = AgentActor(
            seat=player.seat,
            name=player.name,
            role=Role(player.role),
            model_config=cfg,
            router=router,
            rng=random.Random(player.seat * 104729 + rng.randint(0, 9999)),
            is_human=player.seat in human_seats,
        )
        actors[player.id] = actor
    return actors
