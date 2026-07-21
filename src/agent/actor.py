"""LLM/human agent adapter for the harness decision protocol.

Agent decision boundary:
- 每个 AI 决策必须来自真实 LLM 调用,绝不伪造。
- Router 重试瞬时网络故障；Actor 只重试不完整/不合格的模型响应。
- 彻底失败抛 AgentDecisionError。
- JSON/response failures produce no envelope; invalid target intent remains in
  the envelope so protocol validation can reject it without inventing SKIP.
- reasoning 私有保存(上帝/复盘可见,不广播)。
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from ..game.roles import Role
from ..llm.models import ModelConfig
from ..llm.router import LLMError, LLMResponseError, LLMRouter
from ..harness.agent_protocol import ActionRequest, DecisionEnvelope
from ..harness.errors import AgentDecisionError
from .cognition import PrivateAgentState
from .memory import AgentMemory
from .prompts import (
    assign_persona,
    build_messages,
    build_tool_loop_turn,
    last_words_instruction,
    night_action_instruction,
    render_observation,
    role_prompt,
    speak_instruction,
    vote_instruction,
    wolf_council_instruction,
)
from .schemas import AgentAction, AgentObservation, Decision
from .session import AgentSession, AgentSessionError, AgentSessionLimits
from .werewolf_tools import build_werewolf_tool_registry

logger = logging.getLogger(__name__)

DECISION_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 0.05
RETRY_MAX_DELAY_SECONDS = 0.8
# Real models often spend several turns reading private/public state and
# repairing a rejected cognition update before emitting the terminal action.
# Keep the loop bounded, but do not make the actor stricter than the generic
# AgentSession default; the outer decision deadline and tool-call cap remain
# independent hard limits.
TOOL_LOOP_MAX_STEPS = 12
# Response-shape retries are provider generations too, even though they do not
# advance the logical tool-loop step.  Bound them independently per request.
TOOL_LOOP_MAX_GENERATIONS = 18
TOOL_LOOP_MAX_TOOL_CALLS = 24
TOOL_LOOP_MAX_NO_PROGRESS = 5
# This is cumulative provider-reported usage for one ActionRequest.  It does
# not set or forward any provider output-token parameter.
TOOL_LOOP_MAX_TOTAL_TOKENS = 64_000
_MAX_VALIDATION_FEEDBACK_ISSUES = 8
_VALIDATION_FEEDBACK_FIELDS = frozenset({
    "action",
    "target_seat",
    "save_target",
    "poison_target",
    "use_save",
    "use_poison",
    "speech",
    "team_message",
    "bid",
    "thought",
    "claim",
    "reply_to",
    "accuses",
    "private_state",
    "beliefs",
    "seat",
    "wolf_probability",
    "likely_role",
    "confidence",
    "evidence",
    "candidate_plans",
    "selected_plan",
    "public_cover_role",
    "perceived_image",
    "deception_plan",
    "team_plan",
})


class AgentActor:
    """单个 agent 的执行器。

    持有 memory + persona + model_config。每次 decide() 产生一个 Decision。
    Production decisions have one entry point: ``decide(ActionRequest)``.
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
        budget_scope: str | None = None,
    ) -> None:
        self.seat = seat
        self.name = name
        self.role = role
        self.model_config = model_config
        self.router = router
        self.is_human = is_human
        self.on_human_request = on_human_request
        self.budget_scope = budget_scope
        self.memory = AgentMemory(seat=seat, role=role.value)
        self.private_state = PrivateAgentState(owner_seat=seat, owner_role=role.value)
        self.rng = rng or random.Random(seat * 104729)
        persona_name, persona_desc = assign_persona(seat, self.rng)
        self.persona_name = persona_name
        self.persona_desc = persona_desc
        # A session is created per ActionRequest; memory/private_state remain
        # seat-owned across turns, while tool history and terminal state never
        # leak into the next environment request.
        self.agent_session: AgentSession | None = None
        self.on_agent_trace: Callable[[dict[str, Any]], Any] | None = None
        # A seat owns one mutable memory/private-state/RNG tuple.  Serialize
        # the complete decision lifecycle so overlapping protocol requests
        # cannot interleave model turns or replace ``agent_session`` midway.
        self._decide_lock = asyncio.Lock()
        # 人类玩家操作队列(人机混合模式)
        self.human_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.current_human_request: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # 记忆维护(编排器在状态推进时调用)
    # ------------------------------------------------------------------
    def observe_event(self, day: int, phase: str, kind: str, text: str, **meta: Any) -> None:
        self.memory.observe(day, phase, kind, text, **meta)

    def record_claim(self, seat: int, day: int, claim: dict[str, Any]) -> None:
        self.memory.record_claim(seat, day, claim)

    def record_public_commitment(
        self,
        *,
        day: int,
        phase: str,
        kind: str,
        text: str,
        claim: dict[str, Any] | None = None,
    ) -> None:
        """Persist exact public output after the environment accepted it."""
        self.private_state.record_public_commitment(
            day=day,
            phase=phase,
            kind=kind,
            text=text,
            claim=claim,
        )

    def context_provenance(self) -> dict[str, Any]:
        """Return non-secret digests for the exact seat-owned prompt context."""
        return {
            "context_version": "werewolf.agent-context.v1",
            "memory_digest": self.memory.digest(),
            "memory_observation_count": self.memory.snapshot()["observation_count"],
            "private_state_digest": self.private_state.digest(),
            "private_state_revision": self.private_state.snapshot()["revision"],
        }

    # ------------------------------------------------------------------
    # 人类玩家操作(人机混合)
    # ------------------------------------------------------------------
    def enqueue_human_action(self, data: dict[str, Any]) -> tuple[bool, str]:
        """Validate and enqueue a frontend human action for the current request only."""
        if not isinstance(data, dict):
            return False, "invalid_payload"
        current = self.current_human_request
        if not current:
            return False, "no_pending_request"
        ok, reason = self._validate_human_action(data, current)
        if not ok:
            return False, reason
        self.human_queue.put_nowait(data)
        return True, "queued"

    def _clear_human_queue(self) -> None:
        while True:
            try:
                self.human_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _matches_current_human_request(self, data: dict[str, Any], request_id: str) -> bool:
        current = self.current_human_request
        if not isinstance(data, dict) or not current:
            return False
        ok, _reason = self._validate_human_action(data, current)
        return ok and str(data.get("request_id") or "") == request_id

    def _validate_human_action(self, data: dict[str, Any], current: dict[str, Any]) -> tuple[bool, str]:
        if str(data.get("request_id") or "") != str(current.get("request_id") or ""):
            return False, "request_id_mismatch"
        if "phase" not in data:
            return False, "phase_missing"
        if "day" not in data:
            return False, "day_missing"
        action_phase = data.get("phase")
        if str(action_phase) != str(current.get("phase")):
            return False, "phase_mismatch"
        action_day = data.get("day")
        try:
            if int(action_day) != int(current.get("day")):
                return False, "day_mismatch"
        except (TypeError, ValueError):
            return False, "day_invalid"
        user_action = str(data.get("action") or "").strip().lower()
        if not user_action:
            return False, "action_missing"
        if user_action == "skip":
            if not bool(current.get("can_skip")):
                return False, "skip_not_allowed"
            if any(
                data.get(field) is not None and data.get(field) != ""
                for field in ("target_seat", "speech", "bid")
            ):
                return False, "skip_payload_not_empty"
            return True, "queued"
        if user_action and user_action != "skip":
            accepted_actions = set(str(item) for item in current.get("accepted_actions") or [])
            if accepted_actions and user_action not in accepted_actions:
                return False, "action_type_mismatch"
        if user_action != "skip" and current.get("requires_target") and data.get("target_seat") is None:
            return False, "target_required"
        allowed = current.get("allowed_target_seats")
        if data.get("target_seat") is not None:
            target_seat = _parse_single_seat(data.get("target_seat"))
            if target_seat is None:
                return False, "target_invalid"
            if allowed is not None and target_seat not in set(int(item) for item in allowed):
                return False, "target_not_allowed"
        if user_action in {"speak", "last_words"} and not str(data.get("speech") or "").strip():
            return False, "speech_required"
        if user_action == "speak" and data.get("bid") is None:
            return False, "bid_required"
        if data.get("bid") is not None:
            try:
                bid = int(str(data.get("bid")).strip())
            except (TypeError, ValueError):
                return False, "bid_invalid"
            if bid < 0 or bid > 4:
                return False, "bid_out_of_range"
            if user_action == "speak" and bid == 0:
                return False, "bid_zero_requires_skip"
        return True, "queued"

    # ------------------------------------------------------------------
    # 决策入口
    # ------------------------------------------------------------------
    async def decide(self, request: ActionRequest) -> DecisionEnvelope:
        """Serialize decisions for this seat's mutable private state."""
        async with self._decide_lock:
            return await self._decide_serial(request)

    async def _decide_serial(self, request: ActionRequest) -> DecisionEnvelope:
        """Handle one harness ``ActionRequest`` with one model decision.

        The environment has already projected the full game state into the
        request observation and advertised the legal action space.  The agent
        never receives ``GameState`` through this boundary.
        """
        if request.seat != self.seat:
            err = AgentDecisionError(
                f"request seat {request.seat} does not match agent seat {self.seat}"
            )
            setattr(err, "error_type", "AgentRequestSeatMismatch")
            setattr(err, "request_id", request.request_id)
            raise err
        supported_actions = {
            "speak",
            "vote",
            "last_words",
            "wolf_council",
            "night_kill",
            "kill",
            "see",
            "save",
            "poison",
            "guard",
            "hunter_shot",
        }
        if request.action_kind not in supported_actions:
            err = AgentDecisionError(
                f"unsupported ActionRequest action_kind: {request.action_kind!r}"
            )
            setattr(err, "error_type", "UnsupportedActionRequest")
            setattr(err, "request_id", request.request_id)
            raise err
        obs = AgentObservation(**request.observation)
        if obs.my_seat != self.seat:
            err = AgentDecisionError(
                f"observation seat {obs.my_seat} does not match agent seat {self.seat}"
            )
            setattr(err, "error_type", "AgentObservationSeatMismatch")
            setattr(err, "request_id", request.request_id)
            raise err
        if obs.my_role != self.role.value:
            err = AgentDecisionError(
                f"observation role {obs.my_role!r} does not match agent role {self.role.value!r}"
            )
            setattr(err, "error_type", "AgentObservationRoleMismatch")
            setattr(err, "request_id", request.request_id)
            raise err
        started = time.monotonic()
        if self.is_human:
            decision = await self._decide_human_request(request, obs)
            return DecisionEnvelope(
                request_id=request.request_id,
                seat=self.seat,
                decision=decision,
                latency_seconds=round(time.monotonic() - started, 6),
                parse_status="not_applicable",
                metadata={"agent_kind": "human"},
            )

        # Production routers expose a provider-neutral tool turn.  Narrow
        # complete_json-only test doubles keep the legacy adapter below so old
        # protocol tests can isolate response parsing without pretending to be
        # a tool-capable Agent harness.
        if callable(getattr(self.router, "complete_tools", None)):
            return await self._decide_with_tool_loop(
                request,
                obs,
                started=started,
            )

        role_text = role_prompt(
            self.role.value,
            teammates=obs.my_teammates if self.role == Role.WEREWOLF else None,
            extras=self._role_extras(),
        )
        action = request.action_kind
        if action == "wolf_council":
            instruction = wolf_council_instruction(obs)
            required: list[str | tuple[str, ...]] = ["team_message", "target_seat"]
        elif action == "speak":
            instruction = speak_instruction(obs)
            required = ["speech?", "bid"]
        elif action == "vote":
            instruction = vote_instruction(obs)
            required = ["target_seat"]
        elif action == "last_words":
            instruction = last_words_instruction(str(request.private_context.get("reason") or ""))
            required = ["speech?"]
        else:
            instruction = night_action_instruction(obs, self.role.value, requested_action=action)
            required = self._required_night_fields(action)
        required.append("private_state")

        system, messages, _ = build_messages(
            persona_name=self.persona_name,
            persona_desc=self.persona_desc,
            role_text=role_text,
            observation_text=render_observation(
                obs,
                self.memory.render_for_prompt(),
                self.private_state.render_for_prompt(),
            ),
            action_instruction=instruction,
        )
        raw = await self._call_with_retry(
            messages,
            system,
            max_attempts=DECISION_MAX_ATTEMPTS,
            required_fields=required,
            trace_context={
                "request_id": request.request_id,
                "run_id": request.run_id,
                "budget_scope": self.budget_scope,
                "seat": self.seat,
                "role": self.role.value,
                "day": request.day,
                "phase": request.phase,
                "action": action,
            },
        )
        self.private_state.apply_model_update(
            raw["private_state"],
            visible_seats={int(item["seat"]) for item in obs.seats},
            day=request.day,
            phase=request.phase,
            known_wolf_seats={
                int(item["seat"])
                for item in obs.my_teammates
                if item.get("seat") is not None
            } | _private_checked_seats(obs, wolf=True),
            known_village_seats=_private_checked_seats(obs, wolf=False),
            total_wolves=int(obs.role_counts.get(Role.WEREWOLF.value, 0)),
        )
        if action == "wolf_council":
            decision = self._sanitize_wolf_council(raw)
        elif action == "speak":
            decision = self._sanitize_speak(raw, obs)
        elif action == "vote":
            decision = self._sanitize_vote(raw)
        elif action == "last_words":
            decision = self._sanitize_last_words(raw)
        else:
            decision = self._sanitize_night(raw, requested_action=action)
        self._attach_llm_trace(decision, raw)
        trace = raw.get("_llm_call_trace") if isinstance(raw.get("_llm_call_trace"), dict) else {}
        return DecisionEnvelope(
            request_id=request.request_id,
            seat=self.seat,
            decision=decision,
            latency_seconds=round(time.monotonic() - started, 6),
            model_call_id=trace.get("call_id"),
            prompt_hash=trace.get("request_hash"),
            response_hash=trace.get("response_hash"),
            parse_status="recovered" if raw.get("_parse_recovered") else "ok",
            metadata={
                "agent_kind": "llm",
                "provider": self.model_config.provider,
                "model": self.model_config.model,
            },
        )

    async def _decide_human_request(
        self,
        request: ActionRequest,
        obs: AgentObservation,
    ) -> Decision:
        """Wait for a human response using the same request/envelope boundary."""
        from ..config import HUMAN_TIMEOUT

        remaining = request.seconds_remaining()
        timeout = float(HUMAN_TIMEOUT if remaining is None else remaining)
        self._clear_human_queue()
        legal = request.legal_actions[0] if request.legal_actions else None
        allowed_targets = list(legal.target_seats) if legal else []
        requires_target = bool(legal.requires_target) if legal else False
        context = {
            **request.private_context,
            "day": request.day,
            "phase": request.phase,
            "requested_action": request.action_kind,
            "allowed_target_seats": allowed_targets,
            "requires_target": requires_target,
            "can_skip": bool(legal.can_skip) if legal else False,
            "timeout": timeout,
            "timeout_ms": int(timeout * 1000),
        }
        payload = {
            "type": "human_action_request",
            "request_id": request.request_id,
            "seat": self.seat,
            "action_type": request.action_kind,
            "context": context,
            "timeout": timeout,
            "day": request.day,
            "phase": request.phase,
        }
        self.current_human_request = {
            "request_id": request.request_id,
            "action_type": request.action_kind,
            "accepted_actions": [request.action_kind],
            "requires_target": requires_target,
            "can_skip": bool(legal.can_skip) if legal else False,
            "day": request.day,
            "phase": request.phase,
            "allowed_target_seats": allowed_targets,
        }
        if self.on_human_request:
            await self.on_human_request(payload)
        try:
            if request.metadata.get("deadline_owner") == "decision_runtime":
                data = await self.human_queue.get()
            else:
                data = await asyncio.wait_for(self.human_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            await self.on_decision_timeout(request)
            raise _human_timeout_error(request, timeout)
        finally:
            self.current_human_request = None

        action = str(data.get("action") or request.action_kind)
        if action == "skip":
            return Decision(action=AgentAction.SKIP, skip_reason="human_skip")
        target_seat = _parse_single_seat(data.get("target_seat"))
        mapped = {
            "night_kill": AgentAction.NIGHT_KILL,
            "kill": AgentAction.NIGHT_KILL,
            "see": AgentAction.SEE,
            "save": AgentAction.SAVE,
            "poison": AgentAction.POISON,
            "guard": AgentAction.GUARD,
            "hunter_shot": AgentAction.NIGHT_KILL,
            "speak": AgentAction.SPEAK,
            "vote": AgentAction.VOTE,
            "last_words": AgentAction.LAST_WORDS,
        }.get(action)
        if mapped is None:
            err = AgentDecisionError(f"human response action is unknown: {action!r}")
            setattr(err, "error_type", "HumanResponseInvalid")
            raise err
        raw_speech = str(data.get("speech") or "")
        speech = raw_speech if raw_speech.strip() else None
        if action in {"speak", "last_words"} and speech is None:
            err = AgentDecisionError("human text action requires non-empty speech")
            setattr(err, "error_type", "HumanResponseInvalid")
            raise err
        return Decision(
            action=mapped,
            target_seat=target_seat,
            speech=speech,
            bid=int(data["bid"]) if action == "speak" else None,
        )

    async def _decide_with_tool_loop(
        self,
        request: ActionRequest,
        obs: AgentObservation,
        *,
        started: float,
    ) -> DecisionEnvelope:
        """Run one bounded, seat-private model/tool loop.

        A terminal tool only creates a ``Decision``.  The shared
        ``DecisionRuntime`` and Werewolf rules remain the sole consumers of
        that decision, so a model cannot mutate the environment from a tool
        handler or silently turn a failed turn into SKIP.
        """
        action = str(request.action_kind)
        role_text = role_prompt(
            self.role.value,
            teammates=obs.my_teammates if self.role == Role.WEREWOLF else None,
            extras=self._role_extras(),
        )
        visible_seats = sorted({
            parsed
            for item in obs.seats
            if isinstance(item, dict)
            if (parsed := _parse_single_seat(item.get("seat"))) is not None
        })
        visible_seat_set = set(visible_seats)
        alive_seats = sorted({
            parsed
            for item in obs.alive_seats
            if (parsed := _parse_single_seat(item)) in visible_seat_set
        })
        system, initial_messages = build_tool_loop_turn(
            seat=self.seat,
            visible_seats=visible_seats,
            alive_seats=alive_seats,
            persona_name=self.persona_name,
            persona_desc=self.persona_desc,
            role_text=role_text,
            phase=request.phase,
            day=request.day,
            action=action,
            request_id=request.request_id,
        )
        registry = build_werewolf_tool_registry(self, request, obs)
        session = AgentSession(
            seat=self.seat,
            role=self.role.value,
            session_id=f"{request.request_id}:agent-session",
            registry=registry,
            limits=AgentSessionLimits(
                max_steps=TOOL_LOOP_MAX_STEPS,
                max_model_generations=TOOL_LOOP_MAX_GENERATIONS,
                max_tool_calls=TOOL_LOOP_MAX_TOOL_CALLS,
                max_no_progress_steps=TOOL_LOOP_MAX_NO_PROGRESS,
                max_total_tokens=TOOL_LOOP_MAX_TOTAL_TOKENS,
                max_wall_time_seconds=None,
            ),
            private_state=self.private_state,
            memory=self.memory,
            trace_sink=self.on_agent_trace,
        )
        self.agent_session = session
        trace_context = {
            "request_id": request.request_id,
            "run_id": request.run_id,
            "actor_id": f"seat:{self.seat}",
            "seat": self.seat,
            "role": self.role.value,
            "day": request.day,
            "phase": request.phase,
            "action": action,
            "stage": "agent_tool_loop",
            "budget_scope": self.budget_scope,
        }
        try:
            result = await session.run(
                request,
                router=self.router,
                config=self.model_config,
                system=system,
                initial_messages=initial_messages,
                trace_context=trace_context,
                budget_scope=self.budget_scope,
            )
        except (asyncio.CancelledError, Exception):
            # Reclaim any external handler task before the per-seat lock is
            # released.  A task that ignores cancellation remains a bounded,
            # attributable failure rather than continuing with seat state.
            try:
                await session.aclose()
            except Exception:
                logger.exception("agent tool cleanup failed (seat=%s request=%s)", self.seat, request.request_id)
            raise
        try:
            await session.aclose()
        except AgentSessionError as cleanup_error:
            failure = AgentDecisionError(
                f"agent {self.seat}({self.role.value}) tool cleanup failed"
            )
            setattr(failure, "error_type", "AgentToolCleanupFailure")
            setattr(failure, "agent_session_error", cleanup_error.code)
            setattr(failure, "request_id", request.request_id)
            raise failure from cleanup_error
        attempts = _tool_loop_response_attempts(result, request)
        session_summary = result.public_summary()
        if result.error is not None or result.decision is None:
            failure = AgentDecisionError(
                f"agent {self.seat}({self.role.value}) tool loop failed: "
                f"{result.error.code if result.error else 'terminal_decision_missing'}"
            )
            setattr(failure, "error_type", "AgentToolLoopFailure")
            setattr(failure, "agent_session_error", result.error.code if result.error else "terminal_decision_missing")
            setattr(failure, "agent_session_telemetry", session_summary)
            setattr(failure, "llm_call_attempts", attempts)
            setattr(failure, "request_id", request.request_id)
            raise failure from (result.error.cause if result.error is not None else None)

        decision = result.require_decision()
        final_trace = attempts[-1].get("llm_call") if attempts else None
        if not isinstance(final_trace, dict):
            final_trace = {
                "call_id": None,
                "context": {"request_id": request.request_id},
                "usage": {},
                "latency": 0.0,
                "finish_reason": None,
            }
        else:
            final_trace = dict(final_trace)
        final_trace["actor_response_attempt_count"] = len(attempts)
        final_trace["actor_response_attempts"] = [dict(row) for row in attempts]
        final_trace["agent_session"] = session_summary
        private_reasoning = "\n".join(
            str(row.get("reasoning") or "").strip()
            for row in session.private_trace
            if row.get("type") == "model_generation" and str(row.get("reasoning") or "").strip()
        ).strip()
        if private_reasoning:
            decision = decision.model_copy(update={"reasoning": private_reasoning[:4000]})
        setattr(decision, "llm_call_trace", final_trace)
        final_call_id = final_trace.get("call_id")
        return DecisionEnvelope(
            request_id=request.request_id,
            seat=self.seat,
            decision=decision,
            latency_seconds=round(time.monotonic() - started, 6),
            model_call_id=final_call_id,
            prompt_hash=final_trace.get("request_hash"),
            response_hash=final_trace.get("response_hash"),
            parse_status="not_applicable",
            metadata={
                "agent_kind": "llm",
                "runtime": "tool_loop",
                "provider": self.model_config.provider,
                "model": self.model_config.model,
                "agent_session": session_summary,
            },
        )

    async def on_decision_timeout(self, request: ActionRequest) -> None:
        """Emit human-input expiry when DecisionRuntime owns the deadline."""
        if not self.is_human or not self.on_human_request:
            return
        await self.on_human_request({
            "type": "human_action_expired",
            "request_id": request.request_id,
            "seat": self.seat,
            "action_type": request.action_kind,
            "reason": "human_timeout",
            "day": request.day,
            "phase": request.phase,
        })

    # ------------------------------------------------------------------
    # LLM 调用 + 重试
    # ------------------------------------------------------------------
    async def _call_with_retry(
        self,
        messages: list[dict],
        system: str,
        *,
        max_attempts: int,
        required_fields: list[str | tuple[str, ...]] | None = None,
        trace_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """真实 LLM 调用；只为响应级错误重新请求模型。

        - 网络/网关故障由 Router 在一次调用内重试，Actor 不叠加请求。
        - JSON 不完整、字段缺失或 schema 不合格时最多重新请求 max_attempts 次。
        - 彻底失败抛 AgentDecisionError,绝不返回伪造 dict。
        """
        last_err: Exception | None = None
        attempts = max(1, max_attempts)
        response_attempts: list[dict[str, Any]] = []
        retry_feedback: str | None = None
        for attempt in range(attempts):
            raw: dict[str, Any] | None = None
            attempt_messages = [dict(message) for message in messages]
            if retry_feedback:
                attempt_messages.append({"role": "user", "content": retry_feedback})
            try:
                try:
                    raw = await self.router.complete_json(
                        attempt_messages,
                        self.model_config,
                        system=system,
                        allow_lossy=False,
                        include_parse_metadata=True,
                        trace_context=trace_context,
                    )
                except TypeError as err:
                    if trace_context is None or "trace_context" not in str(err):
                        raise
                    raw = await self.router.complete_json(
                        attempt_messages,
                        self.model_config,
                        system=system,
                        allow_lossy=False,
                        include_parse_metadata=True,
                    )
                if not isinstance(raw, dict):
                    raise ValueError(f"LLM 返回非对象: {type(raw)}")
                self._ensure_required_fields(raw, required_fields)
                if raw.get("_parse_lossy"):
                    raise ValueError(f"LLM JSON 有损恢复结果不可落地: {raw.get('_parse_method')}")
                call_trace = raw.get("_llm_call_trace")
                response_attempts.append(_actor_response_attempt(
                    attempt=attempt + 1,
                    status="accepted",
                    call_trace=call_trace if isinstance(call_trace, dict) else None,
                ))
                if isinstance(call_trace, dict):
                    augmented_trace = dict(call_trace)
                    augmented_trace["actor_response_attempt_count"] = len(response_attempts)
                    augmented_trace["actor_response_attempts"] = [
                        dict(row) for row in response_attempts
                    ]
                    raw["_llm_call_trace"] = augmented_trace
                return raw
            except LLMResponseError as err:
                last_err = err
                retry_feedback = None
                response_attempts.append(_actor_response_attempt(
                    attempt=attempt + 1,
                    status="response_rejected",
                    error=err,
                    call_trace=_error_or_raw_call_trace(err, raw),
                ))
                logger.warning(
                    "agent %s(%s) 模型响应不合格 attempt=%d/%d error_type=%s",
                    self.seat, self.role.value, attempt + 1, attempts, type(err).__name__,
                )
                if attempt < attempts - 1:
                    await self._sleep_before_retry(attempt)
                    continue
                break
            except LLMError as err:
                # Router has already exhausted its transport/provider retry
                # budget. Reissuing the whole actor request here would
                # multiply real API traffic without improving provenance.
                last_err = err
                response_attempts.append(_actor_response_attempt(
                    attempt=attempt + 1,
                    status="provider_failed",
                    error=err,
                    call_trace=_error_or_raw_call_trace(err, raw),
                ))
                logger.warning(
                    "agent %s(%s) LLM调用失败(Router已重试) error_type=%s",
                    self.seat, self.role.value, type(err).__name__,
                )
                break
            except ValidationError as err:
                retry_feedback = _validation_retry_feedback(err)
                last_err = ValueError(_validation_failure_summary(err))
                response_attempts.append(_actor_response_attempt(
                    attempt=attempt + 1,
                    status="response_rejected",
                    error=err,
                    call_trace=_error_or_raw_call_trace(err, raw),
                ))
                logger.warning(
                    "agent %s(%s) schema校验失败 attempt=%d/%d issue_count=%d",
                    self.seat,
                    self.role.value,
                    attempt + 1,
                    attempts,
                    min(
                        len(err.errors(
                            include_url=False,
                            include_context=False,
                            include_input=False,
                        )),
                        _MAX_VALIDATION_FEEDBACK_ISSUES,
                    ),
                )
                if attempt < attempts - 1:
                    await self._sleep_before_retry(attempt)
                    continue
                break
            except (ValueError, KeyError, TypeError) as err:
                last_err = err
                retry_feedback = None
                response_attempts.append(_actor_response_attempt(
                    attempt=attempt + 1,
                    status="response_rejected",
                    error=err,
                    call_trace=_error_or_raw_call_trace(err, raw),
                ))
                logger.warning(
                    "agent %s(%s) 解析失败 attempt=%d/%d error_type=%s",
                    self.seat, self.role.value, attempt + 1, attempts, type(err).__name__,
                )
                if attempt < attempts - 1:
                    await self._sleep_before_retry(attempt)
                    continue
                break
        failure = AgentDecisionError(
            f"agent {self.seat}({self.role.value}) 决策失败(最多{max_attempts}次响应尝试): {last_err}"
        )
        setattr(failure, "llm_call_attempts", [dict(row) for row in response_attempts])
        raise failure

    @staticmethod
    def _ensure_required_fields(raw: dict[str, Any], required_fields: list[str | tuple[str, ...]] | None) -> None:
        """Validate action-critical JSON fields before a Decision can be consumed."""
        for field in required_fields or []:
            if isinstance(field, tuple):
                present = [candidate for candidate in field if AgentActor._has_required_field(raw, candidate)]
                if not present:
                    joined = "|".join(field)
                    raise ValueError(f"LLM JSON 缺少必需字段组: {joined}")
                for candidate in present:
                    AgentActor._validate_required_value(raw, candidate)
                continue
            if not AgentActor._has_required_field(raw, field):
                raise ValueError(f"LLM JSON 缺少必需字段: {field}")
            AgentActor._validate_required_value(raw, field)

    @staticmethod
    def _validate_required_value(raw: dict[str, Any], field: str) -> None:
        field_name = field[:-1] if field.endswith("?") else field
        value = raw.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            return
        if field_name == "bid":
            bid = AgentActor._extract_int(raw, "bid")
            if bid is None or bid < 0 or bid > 4:
                raise ValueError(f"LLM JSON bid 超出合法范围 0..4: {value!r}")
        if field_name in {"target_seat", "save_target", "poison_target"}:
            if AgentActor._extract_int(raw, field_name) is None:
                raise ValueError(f"LLM JSON {field_name} 不是整数座位: {value!r}")
        if field_name in {"speech", "team_message"} and not str(value).strip():
            raise ValueError(f"LLM JSON {field_name} 必须是非空文本")
        if field_name == "private_state":
            PrivateAgentState.validate_model_update(value)

    @staticmethod
    def _has_required_field(raw: dict[str, Any], field: str) -> bool:
        allow_none = field.endswith("?")
        field_name = field[:-1] if allow_none else field
        if field_name not in raw:
            return False
        value = raw.get(field_name)
        if value is None:
            return allow_none
        if isinstance(value, str) and not value.strip():
            return allow_none
        return True

    @staticmethod
    def _attach_llm_trace(decision: Decision, raw: dict[str, Any]) -> Decision:
        trace = raw.get("_llm_call_trace")
        if isinstance(trace, dict):
            setattr(decision, "llm_call_trace", trace)
        return decision

    async def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(RETRY_MAX_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
        jitter = self.rng.uniform(0.0, delay * 0.25)
        await asyncio.sleep(delay + jitter)

    # ------------------------------------------------------------------
    # Response normalization. Optional explicit non-action becomes SKIP;
    # malformed or illegal intent is preserved for protocol rejection. The
    # environment never invents a replacement target or action.
    # ------------------------------------------------------------------
    def _sanitize_night(self, raw: dict, *, requested_action: str | None = None) -> Decision:
        thought = self._extract_str(raw, "thought") or ""
        target_seat = self._extract_int(raw, "target_seat")
        action = self._night_action_for_request(requested_action)
        if action is None:
            err = AgentDecisionError(
                f"unsupported ActionRequest action_kind: {requested_action!r}"
            )
            setattr(err, "error_type", "UnsupportedActionRequest")
            raise err

        # 女巫 save/poison 响应有兼容两种目标字段的结构化适配。
        if self.role == Role.WITCH and requested_action in {"save", "poison"}:
            return self._sanitize_witch(raw, thought, requested_action=requested_action)
        if target_seat is None and requested_action in {"save", "hunter_shot"}:
            return Decision(
                action=AgentAction.SKIP,
                reasoning=thought,
                skip_reason=f"{requested_action}_declined",
            )
        return Decision(
            action=action,
            target_seat=target_seat,
            reasoning=thought,
        )

    def _sanitize_wolf_council(self, raw: dict[str, Any]) -> Decision:
        """Preserve one wolf's exact team-private message and tentative target."""
        return Decision(
            action=AgentAction.WOLF_COUNCIL,
            target_seat=self._extract_int(raw, "target_seat"),
            team_message=self._extract_exact_text(raw, "team_message"),
            reasoning=self._extract_str(raw, "thought") or "",
        )

    def _sanitize_witch(
        self,
        raw: dict,
        thought: str,
        *,
        requested_action: str | None = None,
    ) -> Decision:
        """女巫:可能救人(save)+ 毒人(poison),但引擎逐个处理。优先毒人(更主动)。

        编排器会分两次询问女巫(先救后毒),这里根据 raw 决定本步动作。
        实际编排:女巫夜间被问两次——一次决定救,一次决定毒。此处用 use_save/use_poison。
        """
        # 默认本步为 save(编排器先问救)
        use_save = bool(raw.get("use_save", False))
        use_poison = bool(raw.get("use_poison", False))
        # 引擎通过 available_actions 告知当前问的是哪步;这里简化:若 use_poison 且有 poison_target 返回毒
        poison_seat = self._extract_int(raw, "poison_target")
        save_seat = self._extract_int(raw, "save_target")
        if save_seat is None:
            save_seat = self._extract_int(raw, "target_seat")

        if requested_action == "save":
            if use_poison and poison_seat and not (use_save or save_seat):
                return Decision(
                    action=AgentAction.POISON,
                    target_seat=poison_seat,
                    reasoning=thought,
                )
            if use_save or save_seat is not None:
                return Decision(
                    action=AgentAction.SAVE,
                    target_seat=save_seat,
                    reasoning=thought,
                )
            return Decision(
                action=AgentAction.SKIP,
                reasoning=thought,
                skip_reason="witch_save_skipped",
            )

        if requested_action == "poison":
            poison_target = poison_seat
            if poison_target is None:
                poison_target = self._extract_int(raw, "target_seat")
            if use_save and save_seat is not None and not (use_poison or poison_target):
                return Decision(
                    action=AgentAction.SAVE,
                    target_seat=save_seat,
                    reasoning=thought,
                )
            if use_poison or poison_target is not None:
                return Decision(
                    action=AgentAction.POISON,
                    target_seat=poison_target,
                    reasoning=thought,
                )
            return Decision(
                action=AgentAction.SKIP,
                reasoning=thought,
                skip_reason="witch_poison_skipped",
            )

        err = AgentDecisionError(
            f"witch adapter cannot handle ActionRequest action_kind: {requested_action!r}"
        )
        setattr(err, "error_type", "UnsupportedActionRequest")
        raise err

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

    def _required_night_fields(self, requested_action: str | None) -> list[str | tuple[str, ...]]:
        action = (requested_action or "").strip().lower()
        if action == "save":
            if self.role != Role.WITCH:
                return ["target_seat?"]
            return [("target_seat?", "save_target?")]
        if action == "poison":
            return [("target_seat?", "poison_target?")]
        if action == "hunter_shot":
            return ["target_seat?"]
        if action in {"night_kill", "kill", "see", "guard"}:
            return ["target_seat"]
        return ["target_seat?"]

    def _sanitize_speak(self, raw: dict, obs) -> Decision:
        thought = self._extract_str(raw, "thought") or ""
        bid = self._extract_int(raw, "bid")
        if bid is None:
            bid = 0
        speech = self._extract_exact_text(raw, "speech") or ""
        claim = self._sanitize_claim(raw.get("claim"), obs)
        # Public relationship metadata from this same Decision; filter invalid seats.
        reply_to = self._extract_int(raw, "reply_to")
        if reply_to is not None and (
            reply_to == obs.my_seat or not any(s["seat"] == reply_to for s in obs.seats)
        ):
            reply_to = None
        accuses = self._extract_int_list(raw, "accuses", obs)
        if bid <= 0:
            return Decision(
                action=AgentAction.SKIP,
                reasoning=thought,
                speech=speech or None,
                claim=claim,
                reply_to=reply_to,
                accuses=accuses or None,
                skip_reason="speech_declined",
            )
        return Decision(
            action=AgentAction.SPEAK,
            speech=speech or None,
            bid=bid,
            reasoning=thought,
            claim=claim,
            reply_to=reply_to,
            accuses=accuses or None,
        )

    @staticmethod
    def _sanitize_claim(raw: Any, obs) -> dict[str, Any] | None:
        """Validate a public role claim without comparing it to hidden truth.

        Any configured role may be claimed. A seer claim may additionally carry
        one checked seat/result pair. This deliberately permits bluffing.
        """
        if not isinstance(raw, dict):
            return None
        role = str(raw.get("role", "")).strip().lower()
        if role not in {item.value for item in Role}:
            return None
        if role != Role.SEER.value:
            return {"role": role}
        checked_seat = AgentActor._extract_int(raw, "checked_seat")
        result = str(raw.get("result", "")).strip().lower()
        if checked_seat is None and not result:
            return {"role": Role.SEER.value}
        if checked_seat is None:
            return None
        if result not in ("wolf", "village"):
            return None
        # checked_seat 必须是存在的座位(非自己);允许已死者(预言家可能验过夜里死者)
        if checked_seat == obs.my_seat:
            return None
        if not any(s["seat"] == checked_seat for s in obs.seats):
            return None
        return {"role": "seer", "checked_seat": checked_seat, "result": result}

    def _sanitize_vote(self, raw: dict) -> Decision:
        thought = self._extract_str(raw, "thought") or ""
        target_seat = self._extract_int(raw, "target_seat")
        # Preserve the exact seat intent. Protocol validation, not the Agent
        # adapter, decides whether it belongs to the advertised target set.
        return Decision(
            action=AgentAction.VOTE,
            target_seat=target_seat,
            reasoning=thought,
        )

    def _sanitize_last_words(self, raw: dict) -> Decision:
        thought = self._extract_str(raw, "thought") or ""
        speech = self._extract_exact_text(raw, "speech") or ""
        if not speech.strip():
            return Decision(
                action=AgentAction.SKIP,
                reasoning=thought,
                skip_reason="last_words_declined",
            )
        return Decision(
            action=AgentAction.LAST_WORDS,
            speech=speech,
            reasoning=thought,
        )

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
        elif self.role == Role.DOCTOR:
            protected = [
                m for m in self.memory.observations if m.kind == "doctor_protect_target"
            ]
            extras["doctor_state"] = (
                protected[-1].text if protected else "尚无保护记录"
            )
        return extras

    @staticmethod
    def _extract_str(raw: dict, key: str) -> str | None:
        val = raw.get(key)
        if val is None:
            return None
        s = str(val).strip()
        return s or None

    @staticmethod
    def _extract_exact_text(raw: dict, key: str) -> str | None:
        """Return nonblank model text exactly as authored."""
        val = raw.get(key)
        if val is None:
            return None
        text = str(val)
        return text if text.strip() else None

    @staticmethod
    def _extract_int(raw: dict, key: str) -> int | None:
        val = raw.get(key)
        if val is None or val == "":
            return None
        try:
            # 容忍 "3号" / 3.0 / "3"，但不把 2.9 截断成 2。
            s = str(val).replace("号", "").strip()
            number = float(s)
            if not math.isfinite(number) or not number.is_integer():
                return None
            return int(number)
        except (ValueError, TypeError):
            return None

    def _extract_int_list(self, raw: dict, key: str, obs) -> list[int] | None:
        """解析当前指控座位，过滤自己、死亡及不存在的座位。"""
        val = raw.get(key)
        if val is None:
            return None
        visible_seats = {
            int(item["seat"])
            for item in obs.seats
            if isinstance(item, dict) and item.get("seat") is not None
        }
        valid_seats = visible_seats & {int(seat) for seat in obs.alive_seats}
        if not isinstance(val, (list, tuple)):
            # 容忍单个 int/"3" 形式
            single = self._extract_int(raw, key)
            if single is None or single == obs.my_seat or single not in valid_seats:
                return None
            return [single]
        seen: set[int] = set()
        result: list[int] = []
        for item in val:
            try:
                number = float(str(item).replace("号", "").strip())
                if not math.isfinite(number) or not number.is_integer():
                    continue
                seat = int(number)
            except (ValueError, TypeError):
                continue
            if seat > 0 and seat != obs.my_seat and seat in valid_seats and seat not in seen:
                seen.add(seat)
                result.append(seat)
        return result or None


def _error_or_raw_call_trace(
    err: BaseException,
    raw: dict[str, Any] | None,
) -> dict[str, Any] | None:
    trace = getattr(err, "llm_call_trace", None)
    if not isinstance(trace, dict) and isinstance(raw, dict):
        trace = raw.get("_llm_call_trace")
    return dict(trace) if isinstance(trace, dict) else None


def _validation_retry_feedback(err: ValidationError) -> str:
    """Build bounded field-only feedback without rejected values or context."""
    issues = _safe_validation_issues(err)
    rendered = "\n".join(f"- {path}: {code}" for path, code in issues)
    return (
        "上一响应未通过 JSON schema 校验。请重新返回完整 JSON 对象，只修正以下字段级错误；"
        "不要复述此前的字段值、私有上下文或任何凭据。\n"
        f"{rendered}"
    )[:1200]


def _validation_failure_summary(err: ValidationError) -> str:
    issues = _safe_validation_issues(err)
    rendered = ", ".join(f"{path}:{code}" for path, code in issues)
    return f"LLM JSON schema validation failed ({rendered})"


def _safe_validation_issues(err: ValidationError) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    rows = err.errors(include_url=False, include_context=False, include_input=False)
    for row in rows[:_MAX_VALIDATION_FEEDBACK_ISSUES]:
        location = row.get("loc")
        parts: list[str] = []
        if isinstance(location, (list, tuple)):
            for part in location:
                if isinstance(part, int):
                    parts.append(str(part) if 0 <= part <= 999 else "<index>")
                elif isinstance(part, str) and part in _VALIDATION_FEEDBACK_FIELDS:
                    parts.append(part)
                else:
                    parts.append("<field>")
        if err.title == "PrivateStateUpdate":
            parts.insert(0, "private_state")
        path = ".".join(parts) or "<root>"
        raw_code = str(row.get("type") or "invalid")
        code = (
            raw_code
            if len(raw_code) <= 64
            and raw_code.replace("_", "").isascii()
            and raw_code.replace("_", "").isalnum()
            else "invalid"
        )
        issues.append((path, code))
    return issues or [("<root>", "invalid")]


def _actor_response_attempt(
    *,
    attempt: int,
    status: str,
    call_trace: dict[str, Any] | None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "attempt": attempt,
        "status": status,
        "error_type": type(error).__name__ if error is not None else None,
        "llm_call": dict(call_trace) if isinstance(call_trace, dict) else None,
    }
    if isinstance(error, ValidationError):
        row["validation_issues"] = [
            {"path": path, "code": code}
            for path, code in _safe_validation_issues(error)
        ]
    return row


def _tool_loop_response_attempts(result: Any, request: ActionRequest) -> list[dict[str, Any]]:
    """Convert model generations in one Agent loop to existing call provenance."""
    generations = [
        row
        for row in result.private_trace()
        if isinstance(row, dict)
        and row.get("type") in {"model_generation", "model_generation_failed"}
    ]
    attempts: list[dict[str, Any]] = []
    for index, row in enumerate(generations, start=1):
        if row.get("type") == "model_generation_failed":
            trace = row.get("router_trace")
            if not isinstance(trace, dict):
                trace = {
                    "call_id": row.get("call_id"),
                    "context": {"request_id": request.request_id},
                    "request_hash": row.get("request_hash"),
                    "response_hash": None,
                    "usage": {},
                    "latency": 0.0,
                    "finish_reason": None,
                    "transport_attempt_count": 1,
                    "transport_attempts": [],
                    "parse": None,
                }
            attempts.append({
                "attempt": index,
                "status": "response_rejected",
                "error_type": row.get("error_type") or "LLMResponseError",
                "llm_call": dict(trace),
            })
            continue
        trace = row.get("router_trace")
        if not isinstance(trace, dict):
            trace = {
                "call_id": row.get("call_id"),
                "context": {"request_id": request.request_id},
                "request_hash": row.get("request_hash"),
                "response_hash": row.get("response_hash"),
                "usage": dict(row.get("usage") or {}),
                "latency": float(row.get("latency") or 0.0),
                "finish_reason": None,
                "transport_attempt_count": 1,
                "transport_attempts": [],
                "parse": None,
            }
        is_last = index == len(generations)
        if result.completed and is_last:
            status = "accepted"
        elif result.failed and is_last:
            status = "session_failed"
        else:
            status = "tool_continued"
        attempts.append({
            "attempt": index,
            "status": status,
            "error_type": result.error.code if result.error is not None and is_last else None,
            "llm_call": dict(trace),
        })

    cause = result.error.cause if result.error is not None else None
    failure_trace = getattr(cause, "llm_call_trace", None)
    if isinstance(failure_trace, dict):
        failure_call_id = failure_trace.get("call_id")
        if failure_call_id and all(
            row.get("llm_call", {}).get("call_id") != failure_call_id
            for row in attempts
            if isinstance(row.get("llm_call"), dict)
        ):
            attempts.append({
                "attempt": len(attempts) + 1,
                "status": "provider_failed",
                "error_type": type(cause).__name__,
                "llm_call": dict(failure_trace),
            })
    return attempts


def _human_timeout_error(request: ActionRequest, timeout: float) -> AgentDecisionError:
    err = AgentDecisionError(
        f"human decision timeout for request {request.request_id} after {timeout:.3f}s"
    )
    setattr(err, "timeout", True)
    setattr(err, "timeout_seconds", timeout)
    setattr(err, "error_type", "HumanDecisionTimeout")
    setattr(err, "request_id", request.request_id)
    return err


def _parse_single_seat(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        seat = value
    else:
        text = str(value).strip()
        if text.endswith("号"):
            text = text[:-1].strip()
        if not text.isdigit():
            return None
        try:
            seat = int(text)
        except (TypeError, ValueError):
            return None
    return seat if seat > 0 else None


def _private_checked_seats(obs: AgentObservation, *, wolf: bool) -> set[int]:
    """Extract only seer results actually delivered to this seat."""
    expected = {"werewolves", "werewolf", "wolf"} if wolf else {"village", "villager", "good"}
    seats: set[int] = set()
    for event in obs.private_events:
        if event.get("type") != "seer_result":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        team = str(payload.get("team") or "").strip().lower()
        seat = _parse_single_seat(payload.get("target_seat"))
        if seat is not None and team in expected:
            seats.add(seat)
    return seats
