"""Werewolf environment orchestrator for the Agent Harness.

It constructs seat-scoped requests, enforces deadlines, validates envelopes,
submits accepted actions to the rules engine, and records factual traces.
Werewolf night kills are independent per-wolf proposals resolved by plurality
with a seeded tie-break; this is not presented as a multi-round conversation.
"""
from __future__ import annotations

import asyncio
import hashlib
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
from ..agent.actor import AgentActor
from ..agent.information import attach_today_speeches, build_observation
from ..agent.schemas import AgentAction, Decision
from ..harness.agent_protocol import ActionRequest, DecisionEnvelope, LegalAction
from ..harness.decision_runtime import DecisionRuntime
from ..harness.errors import AgentDecisionError
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
from ..game.rules import RulesEngine, RulesError
from ..llm.models import ModelConfig
from ..llm.router import LLMRouter

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]
TraceCallback = Callable[[dict[str, Any]], None]

BELIEF_TRACE_SCHEMA_VERSION = "werewolf.agent-belief-trace.v1"

TURN_POLICIES = (
    "fixed_round_robin",
    "bid_reply",
)
DEFAULT_TURN_POLICY = "fixed_round_robin"
# These guards are deliberately conservative enough not to change ordinary
# games, while bounding a provider outage before the outer 900s run deadline.
DEFAULT_MAX_CONSECUTIVE_DECISION_FAILURES = 3
DEFAULT_MAX_CONSECUTIVE_NO_PROGRESS_ROUNDS = 3
DEFAULT_MAX_GAME_ROUNDS = 20
AGENT_SESSION_BUDGET_CODES = frozenset({
    "agent_budget_exhausted",
    "max_model_generations",
    "max_steps",
    "max_tool_calls",
    "max_total_tokens",
    "no_progress",
    "token_usage_unavailable",
    "wall_time_exceeded",
})
RULE_EVENT_TYPES_TO_EMIT = {"role_assigned", "night_action_submitted", "seer_result"}
LIVE_PUBLIC_EVENT_TYPES = {
    "phase_started",
    "night_resolved",
    "speech",
    "vote_cast",
    "vote_resolved",
    "vote_incomplete",
    "vote_rejected",
    "last_words",
    "last_words_skipped",
    "hunter_shot",
    "game_ended",
    "room_status",
    "game_error",
    "agent_decision_failed",
    "decision_envelope_rejected",
    "decision_validation_failed",
}
RESTRICTED_LIVE_VISIBILITIES = {"god", "admin", "team"}
FORBIDDEN_LIVE_PUBLIC_KEYS = {"role", "team", "teammates", "private_context", "reasoning", "thought"}

def _trace_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _public_speech_memory_text(speech: dict[str, Any]) -> str:
    """Render a public speech observation for agent memory without hidden fields.

    Keep this strictly public: no role truth, private reasoning, or team-only context.
    """
    parts: list[str] = []
    reply_to = speech.get("reply_to")
    if reply_to:
        parts.append(f"回应{reply_to}号")
    accuses = speech.get("accuses") or []
    if accuses:
        parts.append("指控" + ",".join(f"{a}号" for a in accuses))
    meta = f"({'/'.join(parts)})" if parts else ""
    return f"{speech.get('seat')}号{meta}说:{speech.get('text', '')}"


def _public_vote_memory_text(*, voter_seat: int, target_seat: int) -> str:
    """Render an accepted public vote for each seat's memory.

    The vote is an environment fact, so the text is deliberately identical
    for the voter and observers.  Whether the vote was a PK vote is carried in
    bounded metadata rather than folded into prose that a model might have to
    parse.
    """
    return f"{int(voter_seat)}号投了{int(target_seat)}号"


def _public_last_words_memory_text(*, speaker_seat: int, text: str) -> str:
    """Render one accepted public last-words observation without inference."""
    return f"{int(speaker_seat)}号遗言:{text}"


def _as_int(value: Any) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _sanitize_public_claim(claim: Any) -> dict[str, Any] | None:
    """Validate public claim shape without correcting strategic lies."""
    if not isinstance(claim, dict):
        return None
    role = str(claim.get("role") or "").strip().lower()
    if role not in {item.value for item in Role}:
        return None
    if role != Role.SEER.value:
        return {"role": role}
    checked_seat = _as_int(claim.get("checked_seat"))
    result = str(claim.get("result") or "").strip().lower()
    if checked_seat is None and not result:
        return {"role": Role.SEER.value}
    if checked_seat is None or checked_seat <= 0 or result not in {"wolf", "village"}:
        return None
    return {"role": Role.SEER.value, "checked_seat": checked_seat, "result": result}


def _binding_role_value(value: Any, *, field: str) -> str:
    try:
        return Role(value).value
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"actor binding {field} must be a supported role, got {value!r}"
        ) from err


def _validate_one_actor_binding(player: Any, actor: Any) -> None:
    player_id = str(player.id)
    if type(getattr(actor, "seat", None)) is not int or actor.seat != player.seat:
        raise ValueError(
            f"actor binding mismatch for player {player_id}: "
            f"actor.seat={getattr(actor, 'seat', None)!r}, player.seat={player.seat!r}"
        )
    if getattr(actor, "name", None) != player.name:
        raise ValueError(
            f"actor binding mismatch for player {player_id}: "
            f"actor.name={getattr(actor, 'name', None)!r}, player.name={player.name!r}"
        )
    player_role = _binding_role_value(player.role, field="player.role")
    actor_role = _binding_role_value(getattr(actor, "role", None), field="actor.role")
    if actor_role != player_role:
        raise ValueError(
            f"actor binding mismatch for player {player_id}: "
            f"actor.role={actor_role!r}, player.role={player_role!r}"
        )

    memory = getattr(actor, "memory", None)
    if memory is None:
        raise ValueError(f"actor binding for player {player_id} requires a memory object")
    if type(getattr(memory, "seat", None)) is not int or memory.seat != player.seat:
        raise ValueError(
            f"actor binding mismatch for player {player_id}: "
            f"memory.seat={getattr(memory, 'seat', None)!r}, player.seat={player.seat!r}"
        )
    memory_role = _binding_role_value(getattr(memory, "role", None), field="memory.role")
    if memory_role != player_role:
        raise ValueError(
            f"actor binding mismatch for player {player_id}: "
            f"memory.role={memory_role!r}, player.role={player_role!r}"
        )


def _validate_actor_bindings(state: GameState, actors: dict[str, Any]) -> None:
    player_ids = [str(player.id) for player in state.players]
    if len(player_ids) != len(set(player_ids)):
        raise ValueError("state player IDs must be unique before binding actors")
    player_seats = [player.seat for player in state.players]
    if len(player_seats) != len(set(player_seats)):
        raise ValueError("state player seats must be unique before binding actors")

    expected_ids = set(player_ids)
    actual_ids = set(actors)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(str(actor_id) for actor_id in actual_ids - expected_ids)
        raise ValueError(
            "actor keys must exactly cover state players "
            f"(missing={missing}, extra={extra})"
        )

    actor_owners: dict[int, str] = {}
    for player_id in player_ids:
        actor = actors[player_id]
        owner = actor_owners.setdefault(id(actor), player_id)
        if owner != player_id:
            raise ValueError(
                "actor object must be unique per player; "
                f"players {owner!r} and {player_id!r} share one actor"
            )

    memory_owners: dict[int, str] = {}
    for player_id in player_ids:
        memory = getattr(actors[player_id], "memory", None)
        if memory is None:
            raise ValueError(f"actor binding for player {player_id} requires a memory object")
        owner = memory_owners.setdefault(id(memory), player_id)
        if owner != player_id:
            raise ValueError(
                "memory object must be unique per player; "
                f"players {owner!r} and {player_id!r} share one memory"
            )

    players_by_id = {str(player.id): player for player in state.players}
    for player_id in player_ids:
        _validate_one_actor_binding(players_by_id[player_id], actors[player_id])


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
        on_trace: TraceCallback | None = None,
        internal_events: bool = False,
        max_speak_rounds: int = 6,
        turn_policy: str = DEFAULT_TURN_POLICY,
        decision_timeout: float | None = None,
        decision_timeouts: dict[str, float] | None = None,
        phase_deadline: float | None = None,
        phase_deadlines: dict[str, float] | None = None,
        decision_runtime: DecisionRuntime | None = None,
        max_consecutive_decision_failures: int = DEFAULT_MAX_CONSECUTIVE_DECISION_FAILURES,
        max_consecutive_no_progress_rounds: int = DEFAULT_MAX_CONSECUTIVE_NO_PROGRESS_ROUNDS,
        max_game_rounds: int = DEFAULT_MAX_GAME_ROUNDS,
    ) -> None:
        if turn_policy not in TURN_POLICIES:
            raise ValueError(f"unknown turn_policy={turn_policy!r}; expected one of {TURN_POLICIES}")
        if type(max_consecutive_decision_failures) is not int or max_consecutive_decision_failures <= 0:
            raise ValueError("max_consecutive_decision_failures must be a positive integer")
        if type(max_consecutive_no_progress_rounds) is not int or max_consecutive_no_progress_rounds <= 0:
            raise ValueError("max_consecutive_no_progress_rounds must be a positive integer")
        if type(max_game_rounds) is not int or max_game_rounds <= 0:
            raise ValueError("max_game_rounds must be a positive integer")
        _validate_actor_bindings(state, actors)
        self.state = state
        self.actors = actors
        self.deck = deck or default_role_deck(len(state.players))
        self.rng = rng or random.Random()
        self.on_event = on_event
        self.on_trace = on_trace
        self.internal_events = internal_events
        self.max_speak_rounds = max_speak_rounds
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
        self.max_consecutive_decision_failures = max_consecutive_decision_failures
        self.max_consecutive_no_progress_rounds = max_consecutive_no_progress_rounds
        self.max_game_rounds = max_game_rounds
        self.aborted = False
        self.termination_status = "running"
        self.termination_reason: str | None = None
        self.termination_details: dict[str, Any] = {}
        self._failed_events: list[dict[str, Any]] = []
        self._consumed_decisions: list[dict[str, Any]] = []
        self._decision_failures: list[dict[str, Any]] = []
        self._decision_trace: list[dict[str, Any]] = []
        if decision_runtime is None:
            self._decision_runtime = DecisionRuntime(on_trace=self._append_trace)
        else:
            self._decision_runtime = decision_runtime
            self._decision_runtime.add_trace_listener(self._store_runtime_trace)
        self._emitted_rule_event_ids: set[str] = set()
        self._game_ended_emitted = False
        self._request_seq = 0
        # Requests may execute concurrently. Store terminal outcomes by the
        # deterministic request sequence and consume only the contiguous
        # prefix, so completion timing cannot change the failure guard.
        self._request_terminal_outcomes: dict[int, bool] = {}
        self._request_outcome_cursor = 1
        self._consecutive_decision_failures = 0
        self._max_observed_decision_failure_streak = 0
        self._termination_pending: dict[str, Any] | None = None
        self._round_start_living_ids = {
            player.id for player in self.state.living_players()
        }
        self._round_had_valid_vote = False
        self._consecutive_no_progress_rounds = 0
        self._progress_round_history: list[dict[str, Any]] = []
        # 为人类玩家注册请求回调
        for actor in self.actors.values():
            actor.on_human_request = self._on_human_request
            # Tool-loop rows are admin-only decision trace entries.  They do
            # not pass through the public event projection or another seat's
            # observation.
            if hasattr(actor, "on_agent_trace"):
                actor.on_agent_trace = self._append_trace

    async def _on_human_request(self, payload: dict[str, Any]) -> None:
        """Broadcast a human request lifecycle event only to that play seat."""
        recipient = next((p.id for p in self.state.players if p.seat == payload.get("seat")), None)
        recipients = [recipient] if recipient else [str(payload.get("seat"))]
        await self._emit({
            **payload,
            "day": self.state.day,
            "phase": self.state.phase.value,
            "visibility": "private",
            "recipients": recipients,
        })

    async def _emit_new_rule_events(self, event_types: set[str] | None = None) -> None:
        allowed_types = event_types or RULE_EVENT_TYPES_TO_EMIT
        for event in self.state.events:
            if event.type not in allowed_types:
                continue
            if str(event.id) in self._emitted_rule_event_ids:
                continue
            self._emitted_rule_event_ids.add(str(event.id))
            await self._emit(_rule_event_payload(event, self.state))

    def _seat_to_pid(self, seat: int) -> str | None:
        """座位号 → player_id(无论死活)。供被提及者观察记录用。"""
        for player in self.state.players:
            if player.seat == seat:
                return player.id
        return None

    async def _request_agent_decision(
        self,
        actor: AgentActor,
        player_id: str,
        *,
        action_kind: str,
        phase: str,
        today_speeches: list[dict[str, Any]] | None = None,
        pk_candidates: list[str] | None = None,
        private_context: dict[str, Any] | None = None,
        phase_deadline: float | None = None,
    ) -> DecisionEnvelope:
        """Send one versioned request through the harness agent boundary."""
        player = self.state.get_player(player_id)
        if self.actors.get(player_id) is not actor:
            raise ValueError(
                f"actor argument is not the actor bound to player {player_id!r}"
            )
        _validate_one_actor_binding(player, actor)
        legal_target_seats = self._legal_target_seats(
            actor,
            action_kind,
            pk_candidates=pk_candidates,
            private_context=private_context,
        )
        obs = build_observation(
            self.state,
            player_id,
            rng=getattr(actor, "rng", self.rng),
            available_actions=[action_kind],
            candidate_targets=legal_target_seats,
            vote_targets=legal_target_seats if action_kind == "vote" else None,
            in_pk=bool(pk_candidates),
        )
        # ``build_observation`` shuffles a detached copy with this Actor's RNG.
        # Reuse that exact order everywhere the request advertises legality.
        target_seats = list(obs.candidate_targets)
        if action_kind == "vote":
            obs.vote_targets = list(target_seats)
        player_role = _binding_role_value(player.role, field="player.role")
        if obs.my_seat != player.seat or obs.my_role != player_role:
            raise ValueError(
                f"observation identity mismatch for player {player_id!r}: "
                f"observation seat/role=({obs.my_seat!r}, {obs.my_role!r}), "
                f"player seat/role=({player.seat!r}, {player_role!r})"
            )
        if today_speeches:
            attach_today_speeches(obs, today_speeches)
        expected_action = {
            "kill": "night_kill",
            "hunter_shot": "night_kill",
        }.get(action_kind, action_kind)
        can_skip = action_kind in {"save", "poison", "hunter_shot", "speak", "last_words"}
        deadline_started = time.monotonic()
        request_deadline = phase_deadline
        deadline_source = "phase" if phase_deadline is not None else None
        decision_timeout = self._decision_timeout_for(phase)
        if decision_timeout > 0:
            decision_deadline = deadline_started + decision_timeout
            if request_deadline is None or decision_deadline < request_deadline:
                request_deadline = decision_deadline
                deadline_source = "decision"
        effective_timeout = (
            round(max(0.0, request_deadline - deadline_started), 6)
            if request_deadline is not None
            else None
        )
        context_provenance: dict[str, Any] = {}
        context_reader = getattr(actor, "context_provenance", None)
        if callable(context_reader):
            context_provenance = dict(context_reader())
        visible_event_ids = [
            str(event.get("id"))
            for event in [*obs.public_events, *obs.private_events]
            if event.get("id")
        ]
        team_event_ids = [
            str(event.get("id"))
            for event in obs.private_events
            if event.get("id") and event.get("type") == "wolf_council_message"
        ]
        request_id = self._next_request_id()
        request_sequence = self._request_seq
        request = ActionRequest(
            request_id=request_id,
            run_id=self.state.id,
            seat=player.seat,
            phase=phase,
            day=self.state.day,
            action_kind=action_kind,
            observation=obs.model_dump(),
            legal_actions=[LegalAction(
                action=expected_action,
                target_seats=target_seats,
                target_required=action_kind in {
                    "night_kill", "kill", "wolf_council", "see", "save", "poison",
                    "guard", "hunter_shot", "vote",
                },
                can_skip=can_skip,
            )],
            # Advertise the exact deadline enforced by DecisionRuntime.
            deadline_monotonic=request_deadline,
            private_context=private_context or {},
            metadata={
                "deadline_source": deadline_source,
                "effective_timeout_seconds": effective_timeout,
                "agent_context": context_provenance,
                "visible_event_ids": visible_event_ids,
                "team_event_ids": team_event_ids,
            },
        )
        try:
            envelope = await self._decision_runtime.execute(actor, request)
            if not isinstance(envelope.decision, Decision):
                raise AgentDecisionError("Werewolf environment requires a Decision response")
        except Exception:
            self._record_request_terminal_outcome(request_sequence, succeeded=False)
            raise
        self._record_request_terminal_outcome(request_sequence, succeeded=True)
        return envelope

    def _next_request_id(self) -> str:
        """Return a deterministic, run-scoped protocol correlation ID."""
        self._request_seq += 1
        return f"{self.state.id}:request:{self._request_seq:06d}"

    def _record_request_terminal_outcome(self, sequence: int, *, succeeded: bool) -> None:
        """Advance the deterministic request-failure guard after one terminal call."""
        if sequence in self._request_terminal_outcomes:
            return
        self._request_terminal_outcomes[sequence] = bool(succeeded)
        while self._request_outcome_cursor in self._request_terminal_outcomes:
            ok = self._request_terminal_outcomes.pop(self._request_outcome_cursor)
            self._request_outcome_cursor += 1
            if ok:
                self._consecutive_decision_failures = 0
                continue
            self._consecutive_decision_failures += 1
            self._max_observed_decision_failure_streak = max(
                self._max_observed_decision_failure_streak,
                self._consecutive_decision_failures,
            )
            if (
                self._termination_pending is None
                and self._consecutive_decision_failures
                >= self.max_consecutive_decision_failures
            ):
                self._termination_pending = {
                    "reason": "consecutive_decision_failures",
                    "threshold": self.max_consecutive_decision_failures,
                    "observed": self._consecutive_decision_failures,
                    "last_request_sequence": self._request_outcome_cursor - 1,
                }

    def _termination_snapshot(self) -> dict[str, Any]:
        """Return bounded, credential-free termination evidence for artifacts."""
        return {
            "status": self.termination_status,
            "reason": self.termination_reason,
            "details": dict(self.termination_details),
            "decision_failure_streak": self._consecutive_decision_failures,
            "max_observed_decision_failure_streak": self._max_observed_decision_failure_streak,
            "max_consecutive_decision_failures": self.max_consecutive_decision_failures,
            "consecutive_no_progress_rounds": self._consecutive_no_progress_rounds,
            "max_consecutive_no_progress_rounds": self.max_consecutive_no_progress_rounds,
            "max_game_rounds": self.max_game_rounds,
            "progress_rounds": list(self._progress_round_history[-20:]),
        }

    async def _terminate_if_pending(self) -> bool:
        """Commit a pending incomplete terminal state, if a guard tripped.

        All requests that caused the guard are already awaited by their caller;
        this method only flushes public failure events and never invents an
        action. A rules winner always takes precedence over a guard that became
        visible during optional death-resolution requests.
        """
        pending = self._termination_pending
        if pending is None:
            return False
        winner = RulesEngine.check_winner(self.state)
        if winner:
            self.state.phase = Phase.ENDED
            self.state.winner = winner
            self.termination_status = "completed"
            self.termination_reason = "winner_declared"
            self.termination_details = {"winner": winner.value}
            RulesEngine._emit_win_event(self.state, winner)
            await self._emit_game_ended()
            self._termination_pending = None
            return True

        for event in self._failed_events:
            await self._emit(event)
        self._failed_events.clear()
        self.state.phase = Phase.ENDED
        self.state.winner = None
        self.termination_status = "incomplete"
        self.termination_reason = str(pending.get("reason") or "incomplete")
        self.termination_details = {
            **pending,
            "request_count": self._request_seq,
            "round_count": len(self._progress_round_history),
        }
        await self._emit_game_ended()
        self._termination_pending = None
        return True

    async def _complete_progress_round(self) -> bool:
        """Close one night/day/vote cycle and enforce the no-progress limit."""
        living_ids = {player.id for player in self.state.living_players()}
        record = {
            "day": max(0, self.state.day - 1),
            "had_death": living_ids != self._round_start_living_ids,
            "had_valid_vote": self._round_had_valid_vote,
        }
        record["progress"] = bool(record["had_death"] or record["had_valid_vote"])
        if record["progress"]:
            self._consecutive_no_progress_rounds = 0
        else:
            self._consecutive_no_progress_rounds += 1
        record["consecutive_no_progress_rounds"] = self._consecutive_no_progress_rounds
        self._progress_round_history.append(record)

        if (
            not record["progress"]
            and self._consecutive_no_progress_rounds
            >= self.max_consecutive_no_progress_rounds
            and self._termination_pending is None
        ):
            self._termination_pending = {
                "reason": "consecutive_no_progress_rounds",
                "threshold": self.max_consecutive_no_progress_rounds,
                "observed": self._consecutive_no_progress_rounds,
                "last_completed_day": record["day"],
            }
        if (
            len(self._progress_round_history) >= self.max_game_rounds
            and self._termination_pending is None
        ):
            self._termination_pending = {
                "reason": "max_game_rounds",
                "threshold": self.max_game_rounds,
                "observed": len(self._progress_round_history),
                "last_completed_day": record["day"],
            }
        if await self._terminate_if_pending():
            return True

        self._round_start_living_ids = living_ids
        self._round_had_valid_vote = False
        return False

    def _legal_target_seats(
        self,
        actor: AgentActor,
        action_kind: str,
        *,
        pk_candidates: list[str] | None,
        private_context: dict[str, Any] | None,
    ) -> list[int]:
        living = self.state.living_players()
        if action_kind in {"speak", "last_words"}:
            return []
        if action_kind in {"night_kill", "kill", "wolf_council"}:
            return [p.seat for p in living if p.role != Role.WEREWOLF]
        if action_kind == "vote":
            allowed_ids = set(pk_candidates or [])
            return [
                p.seat for p in living
                if p.seat != actor.seat and (not allowed_ids or p.id in allowed_ids)
            ]
        if action_kind == "see":
            return [p.seat for p in living if p.seat != actor.seat]
        if action_kind == "guard":
            return [p.seat for p in living if p.seat != self.state.last_guarded_seat]
        if action_kind == "save" and actor.role == Role.WITCH:
            killed = (private_context or {}).get("killed_seat")
            return [int(killed)] if killed is not None else []
        if action_kind in {"poison", "hunter_shot"}:
            return [p.seat for p in living if p.seat != actor.seat]
        if action_kind == "save":
            return [p.seat for p in living]
        return []

    def _player_id_to_seat(self, player_id: Any) -> int | None:
        if not player_id:
            return None
        try:
            return self.state.get_player(str(player_id)).seat
        except KeyError:
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
            )

    def _record_public_vote_memory(
        self,
        *,
        day: int,
        voter_seat: int,
        target_seat: int,
        pk: bool,
    ) -> None:
        """Mirror one rules-accepted vote into every living seat's memory.

        This is only a projection of an already accepted public action.  It
        does not infer an intent, update beliefs, or expose any private state.
        Failed/illegal votes never reach this method.
        """
        text = _public_vote_memory_text(
            voter_seat=int(voter_seat),
            target_seat=int(target_seat),
        )
        for pid, observer in self.actors.items():
            if not self.state.get_player(pid).alive:
                continue
            observer.observe_event(
                int(day),
                "voting",
                "vote",
                text,
                voter_seat=int(voter_seat),
                target_seat=int(target_seat),
                pk=bool(pk),
            )

    def _record_public_last_words_memory(
        self,
        *,
        day: int,
        speaker_seat: int,
        text: str,
    ) -> None:
        """Persist one rules-accepted last statement for every living seat.

        This mirrors only exact public text after ``RulesEngine`` accepts it.
        It does not parse the statement, infer a claim, or expose the dead
        speaker's role/private reasoning.
        """
        memory_text = _public_last_words_memory_text(
            speaker_seat=int(speaker_seat),
            text=str(text),
        )
        for pid, observer in self.actors.items():
            if not self.state.get_player(pid).alive:
                continue
            observer.observe_event(
                int(day),
                "last_words",
                "last_words",
                memory_text,
                speaker_seat=int(speaker_seat),
            )

    @staticmethod
    def _record_actor_public_commitment(
        actor: Any,
        *,
        day: int,
        phase: str,
        kind: str,
        text: str,
        claim: dict[str, Any] | None,
    ) -> None:
        """Commit accepted output when the Actor supports private cognition."""
        recorder = getattr(actor, "record_public_commitment", None)
        if callable(recorder):
            recorder(
                day=day,
                phase=phase,
                kind=kind,
                text=text,
                claim=claim,
            )

    @staticmethod
    def _decision_action_value(action: Any) -> str:
        return str(getattr(action, "value", action))

    def _record_consumed_decision(
        self,
        actor: Any,
        envelope: Any,
        *,
        phase: str,
    ) -> None:
        if not isinstance(envelope, DecisionEnvelope):
            return
        decision = envelope.decision
        llm_trace = getattr(decision, "llm_call_trace", None)
        raw_model_call_id = envelope.model_call_id
        if not (isinstance(raw_model_call_id, str) and raw_model_call_id.strip()):
            raw_model_call_id = (
                llm_trace.get("call_id") if isinstance(llm_trace, dict) else None
            )
        model_call_id = (
            raw_model_call_id.strip()
            if isinstance(raw_model_call_id, str) and raw_model_call_id.strip()
            else None
        )
        parse = llm_trace.get("parse") if isinstance(llm_trace, dict) else None
        if isinstance(parse, dict):
            if bool(parse.get("lossy")):
                parse_status = "lossy"
            elif bool(parse.get("recovered")):
                parse_status = "recovered"
            else:
                parse_status = "ok"
            parse_method = str(parse.get("method") or "unknown")
        else:
            envelope_parse_status = str(envelope.parse_status or "").strip().lower()
            parse_status_declared = "parse_status" in envelope.model_fields_set
            if bool(getattr(actor, "is_human", False)):
                parse_status = "not_applicable"
            elif (
                envelope_parse_status in {"ok", "recovered", "not_applicable"}
                and (parse_status_declared or model_call_id is not None)
            ):
                parse_status = envelope_parse_status
            else:
                parse_status = "unavailable"
            parse_method = (
                "unknown" if parse_status in {"ok", "recovered"} else None
            )
        self._consumed_decisions.append({
            "request_id": envelope.request_id,
            "model_call_id": model_call_id,
            "day": self.state.day,
            "phase": phase,
            "seat": getattr(actor, "seat", None),
            "action": self._decision_action_value(decision.action),
            "parse_status": parse_status,
            "parse_method": parse_method,
            "skip_reason": getattr(decision, "skip_reason", None),
        })
        self._record_decision_trace(actor, envelope, phase=phase)

    def _record_decision_trace(
        self,
        actor: Any,
        envelope: DecisionEnvelope,
        *,
        phase: str,
    ) -> None:
        """Record an admin-only decision audit item without raw prompt/key leakage."""
        decision = envelope.decision
        llm_trace = getattr(decision, "llm_call_trace", None)
        if llm_trace is not None and not isinstance(llm_trace, dict):
            llm_trace = None
        item = {
            "type": "decision_consumed",
            "request_id": envelope.request_id,
            # Keep the protocol's canonical provenance name on the consumed
            # row as well as the historical ``call_id`` audit alias.  The
            # smoke gate joins this row to the accepted nested model attempt
            # through ``model_call_id``.
            "model_call_id": envelope.model_call_id or (
                llm_trace.get("call_id") if llm_trace else None
            ),
            "call_id": envelope.model_call_id or (llm_trace.get("call_id") if llm_trace else None),
            "day": self.state.day,
            "phase": phase,
            "seat": getattr(actor, "seat", None),
            "role": getattr(getattr(actor, "role", None), "value", getattr(actor, "role", None)),
            "action": self._decision_action_value(decision.action),
            "target_seat": self._decision_target_seat(decision),
            "decision": self._decision_trace_view(decision),
            "llm_call": llm_trace,
        }
        belief_state = self._belief_state_trace_view(actor)
        if belief_state is not None:
            item["belief_state_after"] = belief_state
        self._append_trace(item)

    @staticmethod
    def _belief_state_trace_view(actor: Any) -> dict[str, Any] | None:
        """Return a bounded admin-only belief checkpoint for offline analysis.

        The checkpoint deliberately excludes plans, free-form evidence, role
        truth, and public commitments. It is emitted only through the decision
        trace, which is never delivered as a game event or Agent observation.
        """
        private_state = getattr(actor, "private_state", None)
        snapshot_reader = getattr(private_state, "snapshot", None)
        if not callable(snapshot_reader):
            return None
        try:
            snapshot = snapshot_reader()
        except Exception:  # noqa: BLE001 - trace instrumentation cannot break play
            return None
        if not isinstance(snapshot, dict):
            return None
        raw_beliefs = snapshot.get("beliefs")
        if not isinstance(raw_beliefs, dict):
            return None

        beliefs: dict[str, dict[str, Any]] = {}
        for raw_seat, raw_belief in raw_beliefs.items():
            seat = _as_int(raw_seat)
            if seat is None or seat < 1 or not isinstance(raw_belief, dict):
                continue
            try:
                probability = float(raw_belief.get("wolf_probability"))
                confidence = float(raw_belief.get("confidence"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                continue
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                continue
            likely_role = raw_belief.get("likely_role")
            updated_day = _as_int(raw_belief.get("updated_day"))
            updated_phase = raw_belief.get("updated_phase")
            beliefs[str(seat)] = {
                "wolf_probability": probability,
                "likely_role": (
                    str(likely_role) if isinstance(likely_role, str) and likely_role else None
                ),
                "confidence": confidence,
                "updated_day": updated_day if updated_day is not None and updated_day >= 0 else 0,
                "updated_phase": (
                    str(updated_phase) if isinstance(updated_phase, str) else ""
                ),
            }
        revision = _as_int(snapshot.get("revision"))
        owner_seat = _as_int(snapshot.get("owner_seat"))
        return {
            "schema_version": BELIEF_TRACE_SCHEMA_VERSION,
            "owner_seat": owner_seat or getattr(actor, "seat", None),
            "revision": revision if revision is not None and revision >= 0 else 0,
            "beliefs": dict(sorted(beliefs.items(), key=lambda item: int(item[0]))),
        }

    def _record_rules_trace(
        self,
        actor: Any,
        envelope: DecisionEnvelope,
        *,
        phase: str,
        status: str,
        rule_action: str | None = None,
        target_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        decision = envelope.decision
        llm_trace = getattr(decision, "llm_call_trace", None)
        target_seat = self._player_id_to_seat(target_id) if target_id else self._decision_target_seat(decision)
        self._append_trace({
            "type": "rules_result",
            "request_id": envelope.request_id,
            "call_id": envelope.model_call_id or (
                llm_trace.get("call_id") if isinstance(llm_trace, dict) else None
            ),
            "day": self.state.day,
            "phase": phase,
            "seat": getattr(actor, "seat", None),
            "action": self._decision_action_value(decision.action),
            "target_seat": target_seat,
            "rules": {
                "status": status,
                "action": rule_action,
                "reason": reason,
            },
        })

    def _append_trace(self, item: dict[str, Any]) -> None:
        stored = dict(item)
        self._decision_trace.append(stored)
        if self.on_trace:
            try:
                self.on_trace(stored)
            except Exception as err:  # noqa: BLE001
                logger.debug("on_trace 回调失败 error_type=%s", type(err).__name__)

    def _store_runtime_trace(self, item: dict[str, Any]) -> None:
        """Keep shared-runtime rows in environment metrics without re-emitting."""
        self._decision_trace.append(dict(item))

    def _decision_trace_view(self, decision: Decision) -> dict[str, Any]:
        speech = getattr(decision, "speech", None) or ""
        return {
            "action": self._decision_action_value(decision.action),
            "target_seat": self._decision_target_seat(decision),
            "speech_hash": _trace_hash(speech) if speech else None,
            "speech_len": len(speech),
            "bid": getattr(decision, "bid", None),
            "claim": getattr(decision, "claim", None),
            "reply_to": getattr(decision, "reply_to", None),
            "accuses": getattr(decision, "accuses", None),
            "skip_reason": getattr(decision, "skip_reason", None),
            "reasoning_hash": _trace_hash(getattr(decision, "reasoning", None) or ""),
        }

    def _decision_timeout_for(self, phase: str) -> float:
        return float(self.decision_timeouts.get(phase, self.decision_timeout))

    def _phase_deadline_for(self, phase: str) -> float:
        return float(self.phase_deadlines.get(phase, self.phase_deadline))

    def _start_phase_deadline(self, phase: str) -> float | None:
        seconds = self._phase_deadline_for(phase)
        if seconds <= 0:
            return None
        return time.monotonic() + seconds

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
        envelope_rejected = error_type == "DecisionEnvelopeRejected"
        validator_failed = error_type == "DecisionValidatorError"
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
            "type": (
                "decision_validation_failed"
                if validator_failed
                else (
                    "decision_envelope_rejected"
                    if envelope_rejected
                    else "agent_decision_failed"
                )
            ),
            "seat": seat if seat is not None else getattr(actor, "seat", None),
            "phase": phase,
            "reason": reason,
            "error_type": error_type,
            "agent_kind": "human" if bool(getattr(actor, "is_human", False)) else "llm",
        }
        if action:
            payload["action"] = action
        request_id = getattr(err, "request_id", None)
        if request_id:
            payload["request_id"] = str(request_id)
        if bool(getattr(err, "timeout", False)) or "timeout" in reason.lower():
            payload["timeout"] = True
            timeout_seconds = getattr(err, "timeout_seconds", None)
            if timeout_seconds is not None:
                payload["timeout_seconds"] = timeout_seconds
        if not isinstance(err, AgentDecisionError):
            logger.error(
                "agent decision call failed(seat=%s phase=%s action=%s type=%s)",
                payload.get("seat"), phase, action, error_type,
            )
        return payload

    @staticmethod
    def _action_rejected_event(
        actor: Any,
        *,
        player_id: str,
        phase: str,
        action: str,
        request_id: str,
        reason_code: str,
    ) -> dict[str, Any]:
        """Describe a RulesEngine rejection without relabeling it as Agent failure."""
        return {
            "type": "action_rejected",
            "request_id": request_id,
            "seat": getattr(actor, "seat", None),
            "phase": phase,
            "action": action,
            "reason_code": reason_code,
            "reason": "RulesEngine rejected the validated action.",
            "visibility": "private",
            "recipients": [player_id],
        }

    def _record_private_action_rejection(
        self,
        actor: Any,
        *,
        player_id: str,
        phase: str,
        action: str,
        request_id: str,
        reason_code: str,
    ) -> dict[str, Any]:
        """Persist bounded, seat-private feedback for an uncommitted action.

        ``_failed_events`` is a delivery queue, not game state.  Keeping a
        rejection only there meant the next ``ActionRequest`` was rebuilt from
        ``state.events`` without any explanation of why the prior intent did
        not take effect.  The domain event below is the authoritative outcome:
        it contains no raw RulesError text, target identity, or replacement
        action, and explicitly marks the attempted action as uncommitted.
        Seat-owned belief/plan edits are cognition, not game transitions, so
        they remain available for the Agent to revise after receiving this
        feedback rather than being silently rolled back by the environment.

        The returned payload retains the existing live-event shape and is still
        queued by callers at their original phase boundary.  This separation
        keeps public/replay timing stable while making the feedback available
        to the affected seat's subsequent observation and private snapshot.
        """
        # These values originate at the environment boundary, but keep the
        # persisted event bounded in case a future caller passes external data.
        bounded_request_id = str(request_id)[:256]
        bounded_phase = str(phase)[:48]
        bounded_action = str(action)[:64]
        bounded_reason_code = str(reason_code)[:96]
        event = Event(
            phase=self.state.phase,
            day=int(self.state.day),
            type="action_rejected",
            message="The validated action was rejected and did not change game state.",
            visibility=EventVisibility.PRIVATE,
            recipients=[str(player_id)],
            payload={
                "request_id": bounded_request_id,
                "request_phase": bounded_phase,
                "action": bounded_action,
                "reason_code": bounded_reason_code,
                "committed": False,
            },
        )
        # RulesEngine owns deterministic event IDs for the mutable domain
        # state; use its append helper rather than an ad-hoc list append.
        RulesEngine._append_event(self.state, event)

        live = self._action_rejected_event(
            actor,
            player_id=str(player_id),
            phase=bounded_phase,
            action=bounded_action,
            request_id=bounded_request_id,
            reason_code=bounded_reason_code,
        )
        live["day"] = int(self.state.day)
        live["committed"] = False
        return live

    async def _emit_vote_rejected(
        self,
        actor: Any,
        *,
        day: int,
        target_id: str | None,
        reason_code: str,
        reason: str,
        allowed_seats: list[int] | None = None,
        request_id: str | None = None,
    ) -> None:
        target_seat = self._player_id_to_seat(target_id)
        payload: dict[str, Any] = {
            "type": "vote_rejected",
            "day": day,
            "seat": getattr(actor, "seat", None),
            "name": getattr(actor, "name", None),
            "target_seat": target_seat,
            "reason_code": reason_code,
            "reason": reason,
        }
        if request_id:
            payload["request_id"] = request_id
        if allowed_seats is not None:
            payload["allowed_seats"] = allowed_seats
        await self._emit(payload)
        try:
            actor.observe_event(
                day,
                "voting",
                "vote_rejected",
                f"你的投票无效:{reason}",
            )
        except Exception:  # noqa: BLE001
            pass

    async def run(self) -> GameState:
        await self._emit({"type": "phase_started", "phase": "setup", "day": 0, "message": "角色分配完成"})
        await self._notify_role_assigned()
        await self._emit_new_rule_events({"role_assigned"})

        while self.state.phase != Phase.ENDED and not self.aborted:
            if self.state.phase == Phase.NIGHT:
                await self._run_night()
            elif self.state.phase == Phase.DAY:
                await self._run_day()
            elif self.state.phase == Phase.VOTING:
                await self._run_voting()
            else:
                raise RuntimeError(f"orchestrator cannot execute phase {self.state.phase!s}")

        if self.aborted and self.state.phase != Phase.ENDED:
            self._termination_pending = {
                "reason": "aborted",
                "threshold": None,
                "observed": None,
            }
            await self._terminate_if_pending()

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
        if await self._terminate_if_pending():
            return
        # 2) 守卫守护
        await self._night_role_actions(Role.GUARD, [NightActionType.GUARD], phase_deadline=night_deadline)
        if await self._terminate_if_pending():
            return
        # 3) 医生独立选择本夜保护目标；医生看不到狼刀预览。
        await self._night_role_actions(Role.DOCTOR, [NightActionType.SAVE], phase_deadline=night_deadline)
        if await self._terminate_if_pending():
            return
        # 4) 狼人先独立发团队消息，再各自读取消息并提交最终击杀票。
        await self._collect_werewolf_kill_proposals(phase_deadline=night_deadline)
        if await self._terminate_if_pending():
            return
        # 5) 女巫:先救
        await self._witch_save_phase(phase_deadline=night_deadline)
        if await self._terminate_if_pending():
            return
        # 6) 女巫:再毒
        await self._witch_poison_phase(phase_deadline=night_deadline)
        if await self._terminate_if_pending():
            return

        # 结算
        self.state = RulesEngine.resolve_night(self.state)
        await self._push_night_results_to_memory()
        await self._emit_new_rule_events({"seer_result"})
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

        self._queue_last_words_for_night_deaths()
        await self._process_deaths_and_hunter()
        if await self._terminate_if_pending():
            return
        if await self._finish_death_resolution(next_phase=Phase.DAY, increment_day=False):
            return

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
            self._request_agent_decision(
                self.actors[pid],
                pid,
                action_kind=str(requested_action),
                phase="night",
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
            await self._submit_safe(pid, res, allowed)

    async def _collect_werewolf_kill_proposals(self, *, phase_deadline: float | None = None) -> None:
        """Run a two-stage council, then resolve independent final wolf votes."""
        wolf_entries = sorted(
            [
                (pid, actor)
                for pid, actor in self.actors.items()
                if self.state.get_player(pid).alive and actor.role == Role.WEREWOLF
            ],
            key=lambda item: self.state.get_player(item[0]).seat,
        )
        if not wolf_entries:
            return

        council_entries = [
            (pid, actor)
            for pid, actor in wolf_entries
            if not bool(getattr(actor, "is_human", False))
        ]
        council_tasks = [
            self._request_agent_decision(
                actor,
                pid,
                action_kind="wolf_council",
                phase="wolf_council",
                phase_deadline=phase_deadline,
            )
            for pid, actor in council_entries
        ]
        council_results = await asyncio.gather(*council_tasks, return_exceptions=True)
        for (pid, actor), result in zip(council_entries, council_results):
            if isinstance(result, Exception):
                self._failed_events.append(self._agent_decision_failure_event(
                    actor,
                    phase="wolf_council",
                    action="wolf_council",
                    err=result,
                ))
                continue
            self._record_consumed_decision(actor, result, phase="wolf_council")
            decision = result.decision
            target_id = self._decision_target_id(decision)
            if (
                decision.action != AgentAction.WOLF_COUNCIL
                or target_id is None
                or not str(decision.team_message or "").strip()
            ):
                continue
            try:
                RulesEngine.record_wolf_council_message(
                    self.state,
                    actor_id=pid,
                    target_id=target_id,
                    message=decision.team_message,
                )
            except RulesError as err:
                self._record_rules_trace(
                    actor,
                    result,
                    phase="wolf_council",
                    status="rejected",
                    rule_action="wolf_council",
                    target_id=target_id,
                    reason=type(err).__name__,
                )
                self._failed_events.append(self._record_private_action_rejection(
                    actor,
                    player_id=pid,
                    phase="wolf_council",
                    action="wolf_council",
                    request_id=result.request_id,
                    reason_code="rules_rejected",
                ))
                continue
            self._record_rules_trace(
                actor,
                result,
                phase="wolf_council",
                status="accepted",
                rule_action="wolf_council",
                target_id=target_id,
            )
        await self._emit_new_rule_events({"wolf_council_message"})
        # All council requests above have reached a terminal trace row. Do not
        # issue a second-stage vote after the deterministic failure guard trips.
        if self._termination_pending is not None:
            return

        final_vote_tasks = [
            self._request_agent_decision(
                actor,
                pid,
                action_kind="night_kill",
                phase="wolf_final_vote",
                phase_deadline=phase_deadline,
            )
            for pid, actor in wolf_entries
        ]
        final_vote_results = await asyncio.gather(
            *final_vote_tasks,
            return_exceptions=True,
        )

        proposals: list[tuple[str, DecisionEnvelope, str]] = []
        for (pid, actor), res in zip(wolf_entries, final_vote_results):
            if isinstance(res, Exception):
                self._failed_events.append(self._agent_decision_failure_event(
                    actor,
                    phase="wolf_final_vote",
                    action="werewolf_final_kill_vote",
                    err=res,
                ))
                continue
            self._record_consumed_decision(actor, res, phase="wolf_final_vote")
            decision = res.decision
            if decision.is_skip:
                continue
            if decision.action == AgentAction.NIGHT_KILL and decision.target_seat is not None:
                target_id = self._decision_target_id(decision)
                if not target_id:
                    continue
                proposals.append((pid, res, target_id))

        if not proposals:
            return
        targets = [target_id for _, _d, target_id in proposals]
        tally = Counter(targets)
        top_count = tally.most_common(1)[0][1]
        tied = sorted(
            (target_id for target_id, count in tally.items() if count == top_count),
            key=lambda target_id: self.state.get_player(target_id).seat,
        )
        chosen = self.rng.choice(tied)
        for proposal_pid, proposal_envelope, proposal_target in proposals:
            proposal_actor = self.actors[proposal_pid]
            selected = proposal_target == chosen
            self._record_rules_trace(
                proposal_actor,
                proposal_envelope,
                phase="wolf_final_vote",
                status="accepted" if selected else "not_selected",
                rule_action="kill_vote",
                target_id=proposal_target,
                reason=(
                    "selected_by_plurality"
                    if selected
                    else "not_selected_by_plurality"
                ),
            )
        first_pid, first_envelope, _ = next(
            (proposal for proposal in proposals if proposal[2] == chosen),
            proposals[0],
        )
        await self._submit_explicit(
            first_pid,
            NightActionType.KILL,
            chosen,
            envelope=first_envelope,
        )

        # Final result remains private to each living wolf.
        for pid, actor in wolf_entries:
            actor.observe_event(self.state.day, "night", "wolf_kill_chosen",
                                f"狼队决定击杀{self.state.get_player(chosen).seat}号")

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
            self._request_agent_decision(
                actor,
                pid,
                action_kind="save",
                phase="night",
                private_context={"killed_seat": kill_seat},
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
            decision = res.decision
            target_id = self._decision_target_id(decision)
            if (
                decision.action == AgentAction.SAVE
                and target_id
                and (kill_action is None or target_id != kill_action.target_id)
            ):
                self._record_rules_trace(
                    actor,
                    res,
                    phase="night",
                    status="rejected",
                    rule_action="save",
                    target_id=target_id,
                    reason="witch_save_target_mismatch",
                )
                self._failed_events.append(self._record_private_action_rejection(
                    actor,
                    player_id=pid,
                    phase="night",
                    action="save",
                    request_id=res.request_id,
                    reason_code="witch_save_target_mismatch",
                ))
                continue
            submitted = await self._submit_safe(pid, res, [NightActionType.SAVE])
            if submitted and decision.action == AgentAction.SAVE and target_id:
                RulesEngine.apply_witch_save(self.state, used=True)
                actor.observe_event(self.state.day, "night", "witch_save_used",
                                    f"你救活了{self.state.get_player(target_id).seat}号")

    async def _witch_poison_phase(self, *, phase_deadline: float | None = None) -> None:
        witch_entries = [(pid, a) for pid, a in self.actors.items()
                         if self.state.get_player(pid).alive and a.role == Role.WITCH and self.state.witch_poison]
        saved_witches = {
            action.actor_id
            for action in self.state.night_actions
            if action.action == NightActionType.SAVE
        }
        witch_entries = [(pid, actor) for pid, actor in witch_entries if pid not in saved_witches]
        if not witch_entries:
            return
        tasks = [
            self._request_agent_decision(
                actor,
                pid,
                action_kind="poison",
                phase="night",
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
            submitted = await self._submit_safe(pid, res, [NightActionType.POISON])
            decision = res.decision
            target_id = self._decision_target_id(decision)
            if submitted and decision.action == AgentAction.POISON and target_id:
                RulesEngine.apply_witch_poison(self.state, used=True)
                actor.observe_event(self.state.day, "night", "witch_poison_used",
                                    f"你对{self.state.get_player(target_id).seat}号使用了毒药")

    async def _submit_safe(
        self,
        pid: str,
        envelope: Any,
        allowed: list[NightActionType],
    ) -> bool:
        if not isinstance(envelope, DecisionEnvelope):
            return False
        decision = envelope.decision
        actor = self.actors[pid]
        requested_rule_action = allowed[0].value if len(allowed) == 1 else "night_action"
        if decision.is_skip:
            self._record_rules_trace(
                actor,
                envelope,
                phase="night",
                status="skipped",
                rule_action=requested_rule_action,
                reason=getattr(decision, "skip_reason", None) or "agent_skip",
            )
            return False
        target_id = self._decision_target_id(decision)
        if not target_id:
            self._record_rules_trace(
                actor,
                envelope,
                phase="night",
                status="rejected",
                rule_action=requested_rule_action,
                reason="target_unresolved",
            )
            self._failed_events.append(self._record_private_action_rejection(
                actor,
                player_id=pid,
                phase="night",
                action="night_action",
                request_id=envelope.request_id,
                reason_code="target_unresolved",
            ))
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
            self._record_rules_trace(
                actor,
                envelope,
                phase="night",
                status="rejected",
                rule_action=str(na_type.value if na_type else decision.action.value),
                target_id=target_id,
                reason="action_not_allowed",
            )
            self._failed_events.append(self._record_private_action_rejection(
                actor,
                player_id=pid,
                phase="night",
                action=str(na_type.value if na_type else decision.action.value),
                request_id=envelope.request_id,
                reason_code="action_not_allowed",
            ))
            return False
        try:
            RulesEngine.submit_night_action(
                self.state, NightAction(actor_id=pid, action=na_type, target_id=target_id)
            )
            await self._emit_new_rule_events({"night_action_submitted"})
            self._record_rules_trace(
                actor,
                envelope,
                phase="night",
                status="accepted",
                rule_action=na_type.value,
                target_id=target_id,
            )
            if na_type == NightActionType.SEE:
                actor.observe_event(self.state.day, "night", "seer_action", f"你查验了{self.state.get_player(target_id).seat}号")
            elif na_type == NightActionType.GUARD:
                actor.observe_event(self.state.day, "night", "guard_target",
                                    f"{self.state.get_player(target_id).seat}号")
            elif na_type == NightActionType.SAVE and actor.role == Role.DOCTOR:
                target_seat = self.state.get_player(target_id).seat
                actor.observe_event(
                    self.state.day,
                    "night",
                    "doctor_protect_target",
                    f"第{self.state.day}夜你选择保护{target_seat}号",
                    target_seat=target_seat,
                )
            return True
        except RulesError as err:
            logger.info("夜间行动被引擎拒绝(pid=%s): %s", pid, err)
            self._record_rules_trace(
                actor,
                envelope,
                phase="night",
                status="rejected",
                rule_action=na_type.value,
                target_id=target_id,
                reason=str(err),
            )
            self._failed_events.append(self._record_private_action_rejection(
                actor,
                player_id=pid,
                phase="night",
                action=na_type.value,
                request_id=envelope.request_id,
                reason_code="rules_rejected",
            ))
            return False

    async def _submit_explicit(
        self,
        pid: str,
        na_type: NightActionType,
        target_id: str,
        *,
        envelope: DecisionEnvelope | None = None,
    ) -> None:
        try:
            RulesEngine.submit_night_action(
                self.state, NightAction(actor_id=pid, action=na_type, target_id=target_id)
            )
            await self._emit_new_rule_events({"night_action_submitted"})
            if envelope is not None:
                self._record_rules_trace(
                    self.actors[pid],
                    envelope,
                    phase="night",
                    status="accepted",
                    rule_action=na_type.value,
                    target_id=target_id,
                )
        except RulesError as err:
            logger.info("夜间行动被引擎拒绝(pid=%s %s): %s", pid, na_type, err)
            if envelope is not None:
                self._record_rules_trace(
                    self.actors[pid],
                    envelope,
                    phase="night",
                    status="rejected",
                    rule_action=na_type.value,
                    target_id=target_id,
                    reason=str(err),
                )
                self._failed_events.append(self._record_private_action_rejection(
                    self.actors[pid],
                    player_id=pid,
                    phase="night",
                    action=na_type.value,
                    request_id=envelope.request_id,
                    reason_code="rules_rejected",
                ))

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

        today_speeches: list[dict[str, Any]] = []
        # 公开 Decision 中被提及/指控的座位；只影响调度优先级，不强制回应。
        mentioned_seats: set[int] = set()
        living_pids = [pid for pid, a in self.actors.items() if self.state.get_player(pid).alive]

        first_order = self._fixed_speak_order(living_pids)
        if self.turn_policy != "fixed_round_robin":
            self.rng.shuffle(first_order)

        for round_idx in range(self.max_speak_rounds):
            # 每轮的发言席位记录(每轮可重新发言)
            spoke_this_round: set[str] = set()
            scheduled_decisions: dict[str, DecisionEnvelope] = {}
            if round_idx == 0:
                order = first_order
            elif self.turn_policy == "fixed_round_robin":
                order = self._fixed_speak_order(living_pids)
            else:
                scheduled = await self._collect_scheduled_speech_decisions(
                    living_pids,
                    today_speeches,
                    spoke_this_round,
                    mentioned_seats,
                    use_reply_priority=self.turn_policy == "bid_reply",
                    phase_deadline=day_deadline,
                )
                if await self._terminate_if_pending():
                    return
                order = [pid for pid, _decision in scheduled]
                scheduled_decisions = dict(scheduled)
                # 收敛检测:本轮无人想发言(bid 全 0),讨论自然结束
                if not order:
                    break
                # 只记录调度事实，不替 Agent 推断心理状态。
                called_seats = {self.actors[pid].seat for pid in order}
                for seat in mentioned_seats:
                    if seat not in called_seats:
                        pid = self._seat_to_pid(seat)
                        if pid and pid in self.actors:
                            self.actors[pid].observe_event(
                                day, "day", "mentioned_silent",
                                "你在本轮被公开点名，但没有产生被调度的公开发言；"
                                "你的 bid 未达到调度阈值。"
                            )

            anyone_spoke = False
            for pid in order:
                if pid in spoke_this_round:
                    continue
                actor = self.actors[pid]
                if not self.state.get_player(pid).alive:
                    continue
                # 第 0 轮/固定策略现场请求；bid 策略复用本轮已收集的同一 Decision。
                if round_idx == 0 or self.turn_policy == "fixed_round_robin":
                    try:
                        envelope = await self._request_agent_decision(
                            actor,
                            pid,
                            action_kind="speak",
                            phase="day",
                            today_speeches=today_speeches,
                            phase_deadline=day_deadline,
                        )
                    except AgentDecisionError as err:
                        await self._emit(self._agent_decision_failure_event(
                            actor,
                            phase="day",
                            action="speak",
                            err=err,
                        ))
                        if await self._terminate_if_pending():
                            return
                        continue
                    except Exception as err:  # noqa: BLE001
                        await self._emit(self._agent_decision_failure_event(
                            actor,
                            phase="day",
                            action="speak",
                            err=err,
                        ))
                        if await self._terminate_if_pending():
                            return
                        continue
                    self._record_consumed_decision(actor, envelope, phase="day")
                else:
                    envelope = scheduled_decisions.get(pid)
                    if envelope is None:
                        continue
                decision = envelope.decision

                if decision.is_skip or not (decision.speech or "").strip():
                    self._record_rules_trace(
                        actor,
                        envelope,
                        phase="day",
                        status="skipped",
                        rule_action="speech",
                        reason=decision.skip_reason or "empty_speech",
                    )
                    spoke_this_round.add(pid)
                    continue
                draft_speech = decision.speech
                # The public text is exactly the speech selected by this agent's
                # single structured decision call.  The harness must not replace
                # it with a second model generation or censor legal bluffing.
                speech = draft_speech
                public_claim = _sanitize_public_claim(decision.claim)
                speech_entry = {
                    "seat": actor.seat, "name": actor.name, "text": speech,
                    "bid": decision.bid, "reply_to": decision.reply_to, "accuses": decision.accuses,
                    "claim": public_claim, "day": day,
                }
                self._record_rules_trace(
                    actor,
                    envelope,
                    phase="day",
                    status="accepted",
                    rule_action="speech",
                )
                today_speeches.append(speech_entry)
                # 结构化指控入 mentioned_seats:被指控者后续轮次 bid≥4 时优先被叫起回应
                if decision.accuses:
                    mentioned_seats.update(int(s) for s in decision.accuses)
                self._record_public_speech_memory(speech_entry)
                self._record_actor_public_commitment(
                    actor,
                    day=day,
                    phase="day",
                    kind="speech",
                    text=speech,
                    claim=public_claim,
                )
                if public_claim:
                    actor.record_claim(actor.seat, day, public_claim)
                    # 所有人都听到该公开 claim,记录到自己视角的 claims(矛盾检测用)
                    for other_pid, other_actor in self.actors.items():
                        if other_pid != pid and self.state.get_player(other_pid).alive:
                            other_actor.record_claim(actor.seat, day, public_claim)
                await self._emit({
                    "type": "speech",
                    "day": day, "seat": actor.seat, "name": actor.name,
                    "text": speech, "bid": decision.bid, "claim": public_claim,
                    "reply_to": decision.reply_to, "accuses": decision.accuses,
                })
                spoke_this_round.add(pid)
                anyone_spoke = True
                if round_idx == 0 and len(spoke_this_round) >= len(living_pids):
                    break
            if not anyone_spoke:
                break

        if await self._terminate_if_pending():
            return
        self.state = RulesEngine.start_vote(self.state)
        await self._emit({"type": "phase_started", "phase": "voting", "day": day,
                          "message": "讨论结束,开始投票。"})
        await self._run_voting(today_speeches=today_speeches)

    async def _collect_scheduled_speech_decisions(
        self,
        living_pids: list[str],
        today_speeches: list[dict],
        spoke_this_round: set[str],
        mentioned_seats: set[int] | None = None,
        use_reply_priority: bool = True,
        phase_deadline: float | None = None,
    ) -> list[tuple[str, DecisionEnvelope]]:
        """Collect one Decision per eligible Agent and return scheduled decisions.

        The scheduler owns the returned decisions. They are never stored as
        ad-hoc attributes on Agent objects.

          1. 被提及/被指控(mentioned_seats)且 bid≥4 → 最优先
          2. bid 降序
          3. 平局时被提及者优先
        bid=0/SKIP 不产生公开发言。不强制叫起被提及者。
        """
        mentioned_seats = mentioned_seats or set()
        eligible = [pid for pid in living_pids if pid not in spoke_this_round]
        tasks = [
            self._request_agent_decision(
                self.actors[pid],
                pid,
                action_kind="speak",
                phase="day",
                today_speeches=today_speeches,
                phase_deadline=phase_deadline,
            )
            for pid in eligible
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        decisions: list[tuple[str, DecisionEnvelope]] = []
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
            decision = res.decision
            if decision.is_skip or (decision.bid or 0) <= 0 or not (decision.speech or "").strip():
                self._record_rules_trace(
                    actor,
                    res,
                    phase="day",
                    status="skipped",
                    rule_action="speech_schedule",
                    reason=decision.skip_reason or "bid_zero",
                )
                continue
            decisions.append((pid, res))

        rng = random.Random(self.state.day * 31 + len(today_speeches))
        # 排序键(升序,越小越先发言):
        #   (0) 被提及且 bid≥4 → 0(最优先),否则 1
        #   (1) -bid(bid 降序)
        #   (2) 被提及者优先(0 < 1)
        #   (3) 随机抖动
        def sort_key(item: tuple[str, DecisionEnvelope]) -> tuple:
            pid, envelope = item
            decision = envelope.decision
            seat = self.actors[pid].seat
            bid = decision.bid or 0
            is_mentioned = use_reply_priority and seat in mentioned_seats
            must_reply = 0 if (is_mentioned and bid >= 4) else 1
            mentioned_rank = 0 if is_mentioned else 1
            return (must_reply, -bid, mentioned_rank, rng.random())

        decisions.sort(key=sort_key)
        return [
            (pid, envelope)
            for pid, envelope in decisions
            if (envelope.decision.bid or 0) > 0
        ]

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
            self._request_agent_decision(
                self.actors[pid],
                pid,
                action_kind="vote",
                phase="voting",
                today_speeches=today_speeches,
                pk_candidates=pk_candidates,
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
            if not isinstance(res, DecisionEnvelope):
                continue
            self._record_consumed_decision(actor, res, phase="voting")
            decision = res.decision
            target_id = self._decision_target_id(decision)
            if pk_candidates and target_id and target_id not in set(pk_candidates):
                self._record_rules_trace(
                    actor,
                    res,
                    phase="voting",
                    status="rejected",
                    rule_action="vote",
                    target_id=target_id,
                    reason="pk_target_not_allowed",
                )
                await self._emit_vote_rejected(
                    actor,
                    day=day,
                    target_id=target_id,
                    reason_code="pk_target_not_allowed",
                    reason="PK 阶段只能投平票候选人。",
                    allowed_seats=[
                        self.state.get_player(pid).seat
                        for pid in pk_candidates
                        if pid in {p.id for p in self.state.players}
                    ],
                    request_id=res.request_id,
                )
                continue
            if decision.action == AgentAction.VOTE and target_id:
                try:
                    RulesEngine.submit_vote(self.state, Vote(voter_id=pid, target_id=target_id))
                    self._record_rules_trace(
                        actor,
                        res,
                        phase="voting",
                        status="accepted",
                        rule_action="vote",
                        target_id=target_id,
                    )
                    self._record_public_vote_memory(
                        day=day,
                        voter_seat=actor.seat,
                        target_seat=self.state.get_player(target_id).seat,
                        pk=pk_candidates is not None,
                    )
                    await self._emit({
                        "type": "vote_cast", "day": day, "seat": actor.seat, "name": actor.name,
                        "target_seat": self.state.get_player(target_id).seat,
                    })
                except RulesError as err:
                    logger.info("投票被拒绝(pid=%s): %s", pid, err)
                    self._record_rules_trace(
                        actor,
                        res,
                        phase="voting",
                        status="rejected",
                        rule_action="vote",
                        target_id=target_id,
                        reason=str(err),
                    )
                    await self._emit_vote_rejected(
                        actor,
                        day=day,
                        target_id=target_id,
                        reason_code="rules_rejected",
                        reason="投票目标不合法,本票无效。",
                        request_id=res.request_id,
                    )
            else:
                reject_reason = decision.skip_reason or "vote_target_unresolved"
                status = "skipped" if decision.is_skip else "rejected"
                self._record_rules_trace(
                    actor,
                    res,
                    phase="voting",
                    status=status,
                    rule_action="vote",
                    reason=reject_reason,
                )
                await self._emit_vote_rejected(
                    actor,
                    day=day,
                    target_id=target_id,
                    reason_code="vote_skipped" if decision.is_skip else "target_unresolved",
                    reason="选择弃票。" if decision.is_skip else "未能解析有效投票目标,本票无效。",
                    request_id=res.request_id,
                )

        if len(self.state.votes) < len(living_pids):
            logger.warning("投票不完整(%d/%d),未投票视为无有效票", len(self.state.votes), len(living_pids))
            await self._emit({"type": "vote_incomplete", "day": day,
                              "cast": len(self.state.votes), "needed": len(living_pids)})

        resolved_votes = dict(self.state.votes)
        self._round_had_valid_vote = self._round_had_valid_vote or bool(resolved_votes)
        if await self._terminate_if_pending():
            return
        before_events = len(self.state.events)
        self.state = RulesEngine.resolve_vote(
            self.state,
            require_all=len(self.state.votes) >= len(living_pids),
        )
        new_events = self.state.events[before_events:]
        resolution_event = next(
            (ev for ev in reversed(new_events) if ev.type in ("player_exiled", "vote_tied", "vote_tied_pk")),
            None,
        )
        message = resolution_event.message if resolution_event else None
        resolution_payload = resolution_event.payload if resolution_event else {}
        await self._emit({
            "type": "vote_resolved",
            "day": day,
            "message": message,
            "votes": resolved_votes,
            "exiled_seat": self._player_id_to_seat(resolution_payload.get("exiled_player_id")),
            "tied_seats": [
                self._player_id_to_seat(pid)
                for pid in resolution_payload.get("tied_player_ids", [])
                if self._player_id_to_seat(pid) is not None
            ],
            "no_exile": resolution_event.type == "vote_tied" if resolution_event else False,
        })

        # PK 处理
        if self.state.pk_candidates:
            await self._run_pk(today_speeches=today_speeches, pk_round=pk_round)
            return

        # 遗言 + 猎人开枪
        await self._process_deaths_and_hunter()
        if await self._terminate_if_pending():
            return

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
        if pk_round >= max_pk_rounds:
            # 达到 PK 上限:不再追加一轮没有投票结算的 PK 发言。
            logger.info("PK 达到上限(%d 轮),当日无人放逐,进入夜晚", max_pk_rounds)
            await self._emit({"type": "vote_resolved", "day": day,
                              "message": f"PK {max_pk_rounds} 轮仍平票,无人放逐。",
                              "votes": {}, "tied_seats": [self.state.get_player(pid).seat for pid in pk_ids],
                              "no_exile": True})
            self.state.pk_candidates = []
            await self._check_winner_and_advance()
            return
        await self._emit({"type": "phase_started", "phase": "pk", "day": day,
                          "message": f"平票,进入 PK:{[self.state.get_player(p).seat for p in pk_ids]}"})
        pk_deadline = self._start_phase_deadline("pk")
        for pid in pk_ids:
            actor = self.actors.get(pid)
            if not actor or not self.state.get_player(pid).alive:
                continue
            try:
                envelope = await self._request_agent_decision(
                    actor,
                    pid,
                    action_kind="speak",
                    phase="pk",
                    today_speeches=day_speeches + pk_speeches,
                    phase_deadline=pk_deadline,
                )
            except AgentDecisionError as err:
                await self._emit(self._agent_decision_failure_event(
                    actor,
                    phase="pk",
                    action="speak",
                    err=err,
                ))
                if await self._terminate_if_pending():
                    return
                continue
            except Exception as err:  # noqa: BLE001
                await self._emit(self._agent_decision_failure_event(
                    actor,
                    phase="pk",
                    action="speak",
                    err=err,
                ))
                if await self._terminate_if_pending():
                    return
                continue
            self._record_consumed_decision(actor, envelope, phase="pk")
            decision = envelope.decision
            if decision.is_skip or not (decision.speech or "").strip():
                self._record_rules_trace(
                    actor,
                    envelope,
                    phase="pk",
                    status="skipped",
                    rule_action="speech",
                    reason=decision.skip_reason or "empty_speech",
                )
                continue
            speech = decision.speech
            public_claim = _sanitize_public_claim(decision.claim)
            speech_entry = {
                "seat": actor.seat, "name": actor.name, "text": speech, "bid": decision.bid,
                "reply_to": decision.reply_to, "accuses": decision.accuses,
                "claim": public_claim, "day": day,
            }
            self._record_rules_trace(
                actor,
                envelope,
                phase="pk",
                status="accepted",
                rule_action="speech",
            )
            pk_speeches.append(speech_entry)
            self._record_public_speech_memory(speech_entry)
            self._record_actor_public_commitment(
                actor,
                day=day,
                phase="pk",
                kind="speech",
                text=speech,
                claim=public_claim,
            )
            if public_claim:
                actor.record_claim(actor.seat, day, public_claim)
                for other_pid, other_actor in self.actors.items():
                    if other_pid != pid and self.state.get_player(other_pid).alive:
                        other_actor.record_claim(actor.seat, day, public_claim)
            await self._emit({"type": "speech", "day": day, "seat": actor.seat, "name": actor.name,
                              "text": speech, "bid": decision.bid, "claim": public_claim, "pk": True,
                              "reply_to": decision.reply_to, "accuses": decision.accuses})
        await self._run_voting(
            pk_candidates=pk_ids,
            today_speeches=day_speeches + pk_speeches,
            pk_round=pk_round + 1,
        )

    async def _process_deaths_and_hunter(self) -> None:
        await self._process_last_words_queue()
        if self.state.phase == Phase.ENDED:
            return

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
                envelope = await self._request_agent_decision(
                    actor,
                    hunter_id,
                    action_kind="hunter_shot",
                    phase="hunter",
                    phase_deadline=hunter_deadline,
                )
            except Exception as err:  # noqa: BLE001
                await self._emit(self._agent_decision_failure_event(
                    actor,
                    phase="hunter",
                    action="hunter_shot",
                    err=err,
                ))
                if await self._terminate_if_pending():
                    return
                self.state = RulesEngine.hunter_shoot(self.state, hunter_id, None)
                await self._emit({
                    "type": "hunter_shot", "day": self.state.day,
                    "seat": actor.seat, "name": actor.name,
                    "target_seat": None,
                    "request_id": getattr(err, "request_id", None),
                    "resolution_reason": "decision_failed",
                })
                continue
            self._record_consumed_decision(actor, envelope, phase="hunter")
            decision = envelope.decision
            target_id = self._decision_target_id(decision)
            try:
                self.state = RulesEngine.hunter_shoot(self.state, hunter_id, target_id)
            except RulesError as err:
                self._record_rules_trace(
                    actor,
                    envelope,
                    phase="hunter",
                    status="rejected",
                    rule_action="hunter_shot",
                    target_id=target_id,
                    reason="hunter_target_invalid",
                )
                await self._emit(self._record_private_action_rejection(
                    actor,
                    player_id=hunter_id,
                    phase="hunter",
                    action="hunter_shot",
                    request_id=envelope.request_id,
                    reason_code="hunter_target_invalid",
                ))
                self.state = RulesEngine.hunter_shoot(self.state, hunter_id, None)
                await self._emit({
                    "type": "hunter_shot", "day": self.state.day,
                    "seat": actor.seat, "name": actor.name,
                    "target_seat": None,
                    "request_id": envelope.request_id,
                    "resolution_reason": "rules_rejected",
                })
                continue
            resolution_status = "skipped" if decision.is_skip else "accepted"
            self._record_rules_trace(
                actor,
                envelope,
                phase="hunter",
                status=resolution_status,
                rule_action="hunter_shot",
                target_id=target_id,
                reason=decision.skip_reason if decision.is_skip else None,
            )
            shot_event: dict[str, Any] = {
                "type": "hunter_shot", "day": self.state.day,
                "seat": actor.seat, "name": actor.name,
                "request_id": envelope.request_id,
                "target_seat": self.state.get_player(target_id).seat if target_id else None,
            }
            if decision.is_skip:
                shot_event["skip_reason"] = decision.skip_reason
            await self._emit(shot_event)
            actor.observe_event(
                self.state.day,
                "hunter",
                "hunter_shot",
                (
                    f"你选择不开枪。"
                    if decision.is_skip
                    else (
                        f"你开枪带走{self.state.get_player(target_id).seat}号。"
                        if target_id
                        else "你没有产生有效开枪目标。"
                    )
                ),
                target_seat=(self.state.get_player(target_id).seat if target_id else None),
                resolution="skipped" if decision.is_skip else "accepted",
            )
            if target_id:
                RulesEngine.queue_last_words(self.state, target_id, reason="hunter_shot")

        # Hunter victims use the exact same last-words protocol path.
        await self._process_last_words_queue()

    async def _process_last_words_queue(self) -> None:
        """Consume every queued last-words opportunity through AgentProtocol."""
        last_words_deadline = self._start_phase_deadline("last_words")
        while self.state.last_words_queue:
            q = self.state.last_words_queue[0]
            pid = q["id"]
            actor = self.actors.get(pid)
            if actor:
                try:
                    envelope = await self._request_agent_decision(
                        actor,
                        pid,
                        action_kind="last_words",
                        phase="last_words",
                        private_context={"reason": q["reason"]},
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
                    if await self._terminate_if_pending():
                        return
                    continue
                self._record_consumed_decision(actor, envelope, phase="last_words")
                decision = envelope.decision
                if decision.is_skip or not (decision.speech or "").strip():
                    self.state.last_words_queue.pop(0)
                    self._record_rules_trace(
                        actor,
                        envelope,
                        phase="last_words",
                        status="skipped",
                        rule_action="last_words",
                        reason=decision.skip_reason or "empty_speech",
                    )
                    await self._emit({
                        "type": "last_words_skipped",
                        "day": self.state.day,
                        "seat": q["seat"],
                        "name": q["name"],
                        "skip_reason": decision.skip_reason or "empty_speech",
                    })
                    continue
                text = decision.speech
                self.state = RulesEngine.record_last_words(self.state, pid, text)
                self._record_public_last_words_memory(
                    day=self.state.day,
                    speaker_seat=q["seat"],
                    text=text,
                )
                self._record_actor_public_commitment(
                    actor,
                    day=self.state.day,
                    phase="last_words",
                    kind="last_words",
                    text=text,
                    claim=None,
                )
                self._record_rules_trace(
                    actor,
                    envelope,
                    phase="last_words",
                    status="accepted",
                    rule_action="last_words",
                )
                await self._emit({"type": "last_words", "day": self.state.day,
                                  "seat": q["seat"], "name": q["name"], "text": text,
                                  })
            else:
                self.state.last_words_queue.pop(0)

    def _queue_last_words_for_night_deaths(self) -> None:
        queued = {item["id"] for item in self.state.last_words_queue}
        for death in self.state.night_deaths:
            pid = death.get("id")
            if isinstance(pid, str) and pid not in queued:
                RulesEngine.queue_last_words(self.state, pid, reason=death.get("reason") or "night_death")
                queued.add(pid)

    async def _check_winner_and_advance(self) -> None:
        if await self._finish_death_resolution(next_phase=Phase.NIGHT, increment_day=True):
            return
        await self._complete_progress_round()

    async def _finish_death_resolution(self, *, next_phase: Phase, increment_day: bool) -> bool:
        winner = RulesEngine.check_winner(self.state)
        if winner:
            self.state.phase = Phase.ENDED
            self.state.winner = winner
            self.termination_status = "completed"
            self.termination_reason = "winner_declared"
            self.termination_details = {"winner": winner.value}
            RulesEngine._emit_win_event(self.state, winner)
            await self._emit_game_ended()
            return True
        self.state.phase = next_phase
        if increment_day:
            self.state.day += 1
        self.state.votes.clear()
        return False

    def _parse_metrics(self) -> dict[str, Any]:
        parsed = [
            d for d in self._consumed_decisions
            if d.get("parse_status") in {"ok", "recovered", "lossy"}
        ]
        recovered = [d for d in parsed if d.get("parse_status") == "recovered"]
        recovered_by_action = Counter(str(d.get("action") or "unknown") for d in recovered)
        recovered_by_phase = Counter(str(d.get("phase") or "unknown") for d in recovered)
        method_counts = Counter(str(d.get("parse_method") or "unknown") for d in parsed)
        parsed_count = len(parsed)
        recovered_count = len(recovered)
        return {
            "decision_count": len(self._consumed_decisions),
            "parsed_model_decision_count": parsed_count,
            "clean_parse_count": sum(1 for d in parsed if d.get("parse_status") == "ok"),
            "parse_recovered_count": recovered_count,
            "parse_recovered_rate": (
                recovered_count / parsed_count if parsed_count else None
            ),
            "parse_recovered_by_action": dict(sorted(recovered_by_action.items())),
            "parse_recovered_by_phase": dict(sorted(recovered_by_phase.items())),
            "parse_method_counts": dict(sorted(method_counts.items())),
            "lossy_consumed_count": sum(1 for d in parsed if d.get("parse_status") == "lossy"),
            "missing_provenance_count": sum(
                1 for d in self._consumed_decisions if d.get("parse_status") == "unavailable"
            ),
            "not_applicable_count": sum(
                1 for d in self._consumed_decisions if d.get("parse_status") == "not_applicable"
            ),
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

    def _decision_trace_metrics(self) -> dict[str, Any]:
        """Count protocol, tool-loop, and rules rows without exposing payloads."""
        request_ids = [
            str(item.get("request", {}).get("request_id"))
            for item in self._decision_trace
            if item.get("kind") == "agent_request"
            and item.get("request", {}).get("request_id")
        ]
        terminal_ids = [
            str(item.get("request_id"))
            for item in self._decision_trace
            if item.get("kind") in {
                "agent_response", "agent_response_failed", "agent_response_cancelled",
                "agent_response_validation_failed",
            }
            and item.get("request_id")
        ]
        request_id_counts = Counter(request_ids)
        terminal_id_counts = Counter(terminal_ids)
        generation_by_request: Counter[str] = Counter()
        tool_call_by_request: Counter[str] = Counter()
        tool_failure_by_code: Counter[str] = Counter()
        tool_failure_by_tool: Counter[str] = Counter()
        requests_with_tool_failures: set[str] = set()
        model_generation_failure_count = 0
        tool_call_count = 0
        tool_result_count = 0
        tool_success_count = 0
        tool_failure_count = 0
        terminal_tool_result_count = 0
        terminal_tool_failure_count = 0
        history_compaction_count = 0
        history_compaction_requests: set[str] = set()
        max_compacted_tool_groups = 0
        max_history_chars_before_compaction = 0
        max_model_history_chars_after_compaction = 0
        history_compaction_limit_unsatisfied_count = 0
        max_unsatisfied_model_history_chars = 0
        for item in self._decision_trace:
            event_type = str(item.get("type") or "")
            request_id = str(item.get("request_id") or "")
            if event_type == "model_generation":
                generation_by_request[request_id] += 1
            elif event_type == "model_generation_failed":
                model_generation_failure_count += 1
            elif event_type == "tool_call_requested":
                tool_call_count += 1
                tool_call_by_request[request_id] += 1
            elif event_type == "tool_result":
                tool_result_count += 1
                if bool(item.get("ok")):
                    tool_success_count += 1
                else:
                    tool_failure_count += 1
                    requests_with_tool_failures.add(request_id)
                    error = item.get("error")
                    if isinstance(error, dict):
                        code = str(error.get("code") or "unknown")
                    else:
                        code = "unknown"
                    tool_failure_by_code[code] += 1
                    tool_failure_by_tool[str(item.get("tool") or "unknown")] += 1
                if bool(item.get("terminal")):
                    terminal_tool_result_count += 1
                    if not bool(item.get("ok")):
                        terminal_tool_failure_count += 1
            elif event_type == "agent_history_compacted":
                compacted_groups = _as_int(item.get("compacted_tool_groups"))
                original_chars = _as_int(item.get("original_chars"))
                model_chars = _as_int(item.get("model_chars"))
                if compacted_groups is not None and compacted_groups >= 0:
                    if compacted_groups > 0:
                        history_compaction_count += 1
                        history_compaction_requests.add(request_id)
                        max_compacted_tool_groups = max(
                            max_compacted_tool_groups,
                            compacted_groups,
                        )
                        if original_chars is not None and original_chars >= 0:
                            max_history_chars_before_compaction = max(
                                max_history_chars_before_compaction,
                                original_chars,
                            )
                        if model_chars is not None and model_chars >= 0:
                            max_model_history_chars_after_compaction = max(
                                max_model_history_chars_after_compaction,
                                model_chars,
                            )
                if item.get("limit_satisfied") is False:
                    history_compaction_limit_unsatisfied_count += 1
                    if model_chars is not None and model_chars >= 0:
                        max_unsatisfied_model_history_chars = max(
                            max_unsatisfied_model_history_chars,
                            model_chars,
                        )
        response_count = sum(
            1 for item in self._decision_trace if item.get("kind") == "agent_response"
        )
        response_failure_count = sum(
            1 for item in self._decision_trace if item.get("kind") == "agent_response_failed"
        )
        response_cancelled_count = sum(
            1 for item in self._decision_trace if item.get("kind") == "agent_response_cancelled"
        )
        response_validation_failure_count = sum(
            1
            for item in self._decision_trace
            if item.get("kind") == "agent_response_validation_failed"
        )
        metrics = {
            "trace_row_count": len(self._decision_trace),
            "request_count": len(request_ids),
            "response_count": response_count,
            "response_failure_count": response_failure_count,
            "response_cancelled_count": response_cancelled_count,
            "response_validation_failure_count": response_validation_failure_count,
            "terminal_response_count": (
                response_count
                + response_failure_count
                + response_cancelled_count
                + response_validation_failure_count
            ),
            "unpaired_request_count": sum(
                count
                for request_id, count in request_id_counts.items()
                if terminal_id_counts.get(request_id, 0) == 0
            ),
            "duplicate_terminal_count": sum(
                max(0, count - 1) for count in terminal_id_counts.values()
            ),
            "orphan_terminal_count": sum(
                count
                for request_id, count in terminal_id_counts.items()
                if request_id not in request_id_counts
            ),
            "consumed_decision_count": sum(
                1 for item in self._decision_trace if item.get("type") == "decision_consumed"
            ),
            "rules_resolution_count": sum(
                1 for item in self._decision_trace if item.get("type") == "rules_result"
            ),
            "model_generation_count": sum(generation_by_request.values()),
            "model_generation_failure_count": model_generation_failure_count,
            "tool_call_count": tool_call_count,
            "tool_result_count": tool_result_count,
            "tool_success_count": tool_success_count,
            "tool_failure_count": tool_failure_count,
            "tool_failure_by_code": dict(sorted(tool_failure_by_code.items())),
            "tool_failure_by_tool": dict(sorted(tool_failure_by_tool.items())),
            "terminal_tool_result_count": terminal_tool_result_count,
            "terminal_tool_failure_count": terminal_tool_failure_count,
            "requests_with_tool_failures": len(requests_with_tool_failures),
            "max_model_generations_per_request": max(generation_by_request.values(), default=0),
            "max_tool_calls_per_request": max(tool_call_by_request.values(), default=0),
            "history_compaction_count": history_compaction_count,
            "requests_with_history_compaction": len(history_compaction_requests),
            "max_compacted_tool_groups": max_compacted_tool_groups,
            "max_history_chars_before_compaction": max_history_chars_before_compaction,
            "max_model_history_chars_after_compaction": max_model_history_chars_after_compaction,
            "history_compaction_limit_unsatisfied_count": history_compaction_limit_unsatisfied_count,
            "max_unsatisfied_model_history_chars": max_unsatisfied_model_history_chars,
        }
        metrics.update(self._agent_turn_finished_metrics())
        return metrics

    def _agent_turn_finished_metrics(self) -> dict[str, Any]:
        """Roll up one auditable, payload-free telemetry row per request.

        ``agent_turn_finished`` is emitted by the seat-owned session at the end
        of its lifecycle.  Only a uniquely paired row whose request/seat
        identity agrees with the originating ``agent_request`` contributes to
        cost totals.  Duplicate and orphan rows remain visible as integrity
        facts but cannot inflate those totals.
        """

        def nonnegative_int(value: Any) -> int | None:
            parsed = _as_int(value)
            return parsed if parsed is not None and parsed >= 0 else None

        def nonnegative_float(value: Any) -> float | None:
            if isinstance(value, bool) or value is None or value == "":
                return None
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            return parsed if math.isfinite(parsed) and parsed >= 0.0 else None

        request_rows: dict[str, list[dict[str, Any]]] = {}
        request_seats: dict[str, int | None] = {}
        for item in self._decision_trace:
            if item.get("kind") != "agent_request":
                continue
            request = item.get("request")
            if not isinstance(request, dict):
                continue
            raw_request_id = request.get("request_id")
            if not isinstance(raw_request_id, str) or not raw_request_id.strip():
                continue
            request_id = raw_request_id.strip()
            request_rows.setdefault(request_id, []).append(item)
            seat = nonnegative_int(request.get("seat"))
            request_seats.setdefault(request_id, seat if seat and seat > 0 else None)

        finished_rows = [
            item
            for item in self._decision_trace
            if item.get("type") == "agent_turn_finished"
        ]
        finished_by_request: dict[str, list[dict[str, Any]]] = {}
        missing_id_finished_count = 0
        for item in finished_rows:
            raw_request_id = item.get("request_id")
            if not isinstance(raw_request_id, str) or not raw_request_id.strip():
                missing_id_finished_count += 1
                continue
            finished_by_request.setdefault(raw_request_id.strip(), []).append(item)

        duplicate_finished_count = sum(
            max(0, len(rows) - 1) for rows in finished_by_request.values()
        )
        orphan_finished_count = missing_id_finished_count + sum(
            len(rows)
            for request_id, rows in finished_by_request.items()
            if request_id not in request_rows
        )
        paired_finished_ids = set(request_rows).intersection(finished_by_request)
        ambiguous_request_ids = {
            request_id
            for request_id in paired_finished_ids
            if len(request_rows[request_id]) != 1
            or len(finished_by_request[request_id]) != 1
        }

        seat_rows: dict[int, dict[str, Any]] = {}

        def seat_row(seat: int) -> dict[str, Any]:
            return seat_rows.setdefault(seat, {
                "seat": seat,
                "request_count": 0,
                "finished_request_count": 0,
                "telemetry_request_count": 0,
                "duplicate_finished_count": 0,
                "invalid_telemetry_count": 0,
                "generation_attempts": 0,
                "model_generations": 0,
                "generation_failures": 0,
                "response_retries": 0,
                "tool_calls": 0,
                "tool_successes": 0,
                "tool_failures": 0,
                "model_latency_seconds": 0.0,
                "tool_latency_seconds": 0.0,
                "elapsed_seconds": 0.0,
                "total_tokens": 0,
                "token_usage_complete_count": 0,
                "token_usage_incomplete_count": 0,
                "budget_failure_count": 0,
                "budget_failure_by_code": Counter(),
                "max_generation_attempts_per_request": 0,
                "max_model_generations_per_request": 0,
                "max_response_retries_per_request": 0,
                "max_tool_calls_per_request": 0,
                "max_total_tokens_per_request": 0,
                "max_model_latency_seconds_per_request": 0.0,
                "max_tool_latency_seconds_per_request": 0.0,
                "max_elapsed_seconds_per_request": 0.0,
            })

        for request_id, rows in request_rows.items():
            if len(rows) != 1:
                continue
            seat = request_seats.get(request_id)
            if seat is None:
                continue
            current = seat_row(seat)
            current["request_count"] += 1
            finished_for_request = finished_by_request.get(request_id, [])
            if finished_for_request:
                current["finished_request_count"] += 1
                current["duplicate_finished_count"] += max(
                    0,
                    len(finished_for_request) - 1,
                )

        totals = {
            "generation_attempts": 0,
            "model_generations": 0,
            "generation_failures": 0,
            "response_retries": 0,
            "tool_calls": 0,
            "tool_successes": 0,
            "tool_failures": 0,
            "model_latency_seconds": 0.0,
            "tool_latency_seconds": 0.0,
            "elapsed_seconds": 0.0,
            "total_tokens": 0,
        }
        maxima = {
            "generation_attempts": 0,
            "model_generations": 0,
            "response_retries": 0,
            "tool_calls": 0,
            "total_tokens": 0,
            "model_latency_seconds": 0.0,
            "tool_latency_seconds": 0.0,
            "elapsed_seconds": 0.0,
        }
        telemetry_request_count = 0
        invalid_telemetry_count = 0
        identity_mismatch_count = 0
        token_usage_complete_count = 0
        token_usage_incomplete_count = 0
        budget_failures: Counter[str] = Counter()

        integer_fields = (
            "generation_attempts",
            "model_generations",
            "generation_failures",
            "response_retries",
            "tool_calls",
            "tool_successes",
            "tool_failures",
            "total_tokens",
        )
        duration_fields = (
            "model_latency_seconds",
            "tool_latency_seconds",
            "elapsed_seconds",
        )

        def normalized_telemetry(
            telemetry: dict[str, Any],
        ) -> tuple[dict[str, int | float], bool, str | None] | None:
            parsed: dict[str, int | float] = {}
            for field in integer_fields:
                value = nonnegative_int(telemetry.get(field))
                if value is None:
                    return None
                parsed[field] = value
            for field in duration_fields:
                value = nonnegative_float(telemetry.get(field))
                if value is None:
                    return None
                parsed[field] = value

            token_complete = telemetry.get("token_usage_complete")
            if not isinstance(token_complete, bool):
                return None
            raw_budget_code = telemetry.get("budget_exhausted")
            if raw_budget_code is None:
                budget_code = None
            elif (
                isinstance(raw_budget_code, str)
                and raw_budget_code.strip() in AGENT_SESSION_BUDGET_CODES
            ):
                budget_code = raw_budget_code.strip()
            else:
                return None
            if (
                parsed["model_generations"] + parsed["generation_failures"]
                != parsed["generation_attempts"]
                or parsed["response_retries"] > parsed["generation_failures"]
                or parsed["tool_successes"] + parsed["tool_failures"]
                != parsed["tool_calls"]
            ):
                return None
            return parsed, token_complete, budget_code

        for request_id in sorted(paired_finished_ids - ambiguous_request_ids):
            item = finished_by_request[request_id][0]
            telemetry = item.get("telemetry")
            expected_seat = request_seats.get(request_id)
            if not isinstance(telemetry, dict) or expected_seat is None:
                invalid_telemetry_count += 1
                if expected_seat is not None:
                    seat_row(expected_seat)["invalid_telemetry_count"] += 1
                continue

            telemetry_request_id = telemetry.get("request_id")
            row_seat = nonnegative_int(item.get("seat"))
            telemetry_seat = nonnegative_int(telemetry.get("seat"))
            if (
                not isinstance(telemetry_request_id, str)
                or telemetry_request_id.strip() != request_id
                or row_seat != expected_seat
                or telemetry_seat != expected_seat
            ):
                invalid_telemetry_count += 1
                identity_mismatch_count += 1
                seat_row(expected_seat)["invalid_telemetry_count"] += 1
                continue

            normalized = normalized_telemetry(telemetry)
            if normalized is None:
                invalid_telemetry_count += 1
                seat_row(expected_seat)["invalid_telemetry_count"] += 1
                continue
            parsed, token_complete, budget_code = normalized
            telemetry_request_count += 1
            current = seat_row(expected_seat)
            current["telemetry_request_count"] += 1
            for field in integer_fields:
                amount = int(parsed[field])
                totals[field] += amount
                current[field] += amount
            for field in duration_fields:
                amount = float(parsed[field])
                totals[field] += amount
                current[field] += amount
            for field in maxima:
                maxima[field] = max(maxima[field], parsed[field])
            current["max_generation_attempts_per_request"] = max(
                current["max_generation_attempts_per_request"],
                parsed["generation_attempts"],
            )
            current["max_model_generations_per_request"] = max(
                current["max_model_generations_per_request"],
                parsed["model_generations"],
            )
            current["max_response_retries_per_request"] = max(
                current["max_response_retries_per_request"],
                parsed["response_retries"],
            )
            current["max_tool_calls_per_request"] = max(
                current["max_tool_calls_per_request"],
                parsed["tool_calls"],
            )
            current["max_total_tokens_per_request"] = max(
                current["max_total_tokens_per_request"],
                parsed["total_tokens"],
            )
            current["max_model_latency_seconds_per_request"] = max(
                current["max_model_latency_seconds_per_request"],
                parsed["model_latency_seconds"],
            )
            current["max_tool_latency_seconds_per_request"] = max(
                current["max_tool_latency_seconds_per_request"],
                parsed["tool_latency_seconds"],
            )
            current["max_elapsed_seconds_per_request"] = max(
                current["max_elapsed_seconds_per_request"],
                parsed["elapsed_seconds"],
            )
            if token_complete:
                token_usage_complete_count += 1
                current["token_usage_complete_count"] += 1
            else:
                token_usage_incomplete_count += 1
                current["token_usage_incomplete_count"] += 1
            if budget_code is not None:
                budget_failures[budget_code] += 1
                current["budget_failure_count"] += 1
                current["budget_failure_by_code"][budget_code] += 1

        normalized_seat_rows: list[dict[str, Any]] = []
        for seat in sorted(seat_rows):
            row = seat_rows[seat]
            count = int(row["telemetry_request_count"])
            row["missing_finished_count"] = max(
                0,
                int(row["request_count"]) - int(row["finished_request_count"]),
            )
            for field in duration_fields:
                row[field] = round(float(row[field]), 6)
            for field in (
                "max_model_latency_seconds_per_request",
                "max_tool_latency_seconds_per_request",
                "max_elapsed_seconds_per_request",
            ):
                row[field] = round(float(row[field]), 6)
            for field in (
                "generation_attempts",
                "model_generations",
                "response_retries",
                "tool_calls",
                "tool_failures",
                "model_latency_seconds",
                "tool_latency_seconds",
                "elapsed_seconds",
                "total_tokens",
            ):
                row[f"{field}_per_request"] = (
                    round(float(row[field]) / count, 6) if count else None
                )
            row["budget_failure_by_code"] = dict(
                sorted(row["budget_failure_by_code"].items())
            )
            normalized_seat_rows.append(row)

        def extrema(field: str) -> dict[str, Any] | None:
            evidence = [
                (int(row["seat"]), row.get(field))
                for row in normalized_seat_rows
                if isinstance(row.get(field), (int, float))
                and not isinstance(row.get(field), bool)
            ]
            if not evidence:
                return None
            values = [float(value) for _seat, value in evidence]
            minimum = min(values)
            maximum = max(values)
            integral = all(float(value).is_integer() for value in values)

            def normalized(value: float) -> int | float:
                return int(value) if integral else round(value, 6)

            return {
                "minimum": normalized(minimum),
                "maximum": normalized(maximum),
                "spread": normalized(maximum - minimum),
                "max_to_min_ratio": (
                    round(maximum / minimum, 6) if minimum > 0 else None
                ),
                "minimum_seats": [
                    seat for seat, value in evidence if float(value) == minimum
                ],
                "maximum_seats": [
                    seat for seat, value in evidence if float(value) == maximum
                ],
            }

        fairness_fields = (
            "request_count",
            "finished_request_count",
            "telemetry_request_count",
            "generation_attempts_per_request",
            "model_generations_per_request",
            "response_retries_per_request",
            "tool_calls_per_request",
            "tool_failures_per_request",
            "model_latency_seconds_per_request",
            "tool_latency_seconds_per_request",
            "elapsed_seconds_per_request",
            "total_tokens_per_request",
            "max_generation_attempts_per_request",
            "max_model_generations_per_request",
            "max_tool_calls_per_request",
            "max_total_tokens_per_request",
        )
        fairness_facts = {
            field: fact
            for field in fairness_fields
            if (fact := extrema(field)) is not None
        }

        return {
            "agent_turn_finished_count": len(finished_rows),
            "unique_agent_turn_finished_count": len(finished_by_request),
            "duplicate_agent_turn_finished_count": duplicate_finished_count,
            "orphan_agent_turn_finished_count": orphan_finished_count,
            "requests_with_agent_turn_finished": len(paired_finished_ids),
            "requests_without_agent_turn_finished": sum(
                1 for request_id in request_rows if request_id not in finished_by_request
            ),
            "ambiguous_agent_turn_finished_request_count": len(ambiguous_request_ids),
            "agent_turn_telemetry_request_count": telemetry_request_count,
            "invalid_agent_turn_telemetry_count": invalid_telemetry_count,
            "agent_turn_telemetry_identity_mismatch_count": identity_mismatch_count,
            "agent_turn_generation_attempts": totals["generation_attempts"],
            "agent_turn_model_generations": totals["model_generations"],
            "agent_turn_generation_failures": totals["generation_failures"],
            "agent_turn_response_retries": totals["response_retries"],
            "agent_turn_tool_calls": totals["tool_calls"],
            "agent_turn_tool_successes": totals["tool_successes"],
            "agent_turn_tool_failures": totals["tool_failures"],
            "agent_turn_model_latency_seconds": round(
                float(totals["model_latency_seconds"]),
                6,
            ),
            "agent_turn_tool_latency_seconds": round(
                float(totals["tool_latency_seconds"]),
                6,
            ),
            "agent_turn_elapsed_seconds": round(
                float(totals["elapsed_seconds"]),
                6,
            ),
            "agent_turn_total_tokens": totals["total_tokens"],
            "agent_turn_token_usage_complete_count": token_usage_complete_count,
            "agent_turn_token_usage_incomplete_count": token_usage_incomplete_count,
            "agent_turn_token_usage_unavailable_count": (
                invalid_telemetry_count
                + len(ambiguous_request_ids)
                + sum(
                    1
                    for request_id in request_rows
                    if request_id not in finished_by_request
                )
            ),
            "agent_turn_token_usage_complete": (
                token_usage_incomplete_count == 0
                and invalid_telemetry_count == 0
                and not ambiguous_request_ids
                and telemetry_request_count == len(request_rows)
                if request_rows
                else None
            ),
            "agent_turn_budget_failure_count": sum(budget_failures.values()),
            "agent_turn_budget_failure_by_code": dict(sorted(budget_failures.items())),
            "max_agent_turn_generation_attempts_per_request": maxima["generation_attempts"],
            "max_agent_turn_model_generations_per_request": maxima["model_generations"],
            "max_agent_turn_response_retries_per_request": maxima["response_retries"],
            "max_agent_turn_tool_calls_per_request": maxima["tool_calls"],
            "max_agent_turn_total_tokens_per_request": maxima["total_tokens"],
            "max_agent_turn_model_latency_seconds_per_request": round(
                float(maxima["model_latency_seconds"]),
                6,
            ),
            "max_agent_turn_tool_latency_seconds_per_request": round(
                float(maxima["tool_latency_seconds"]),
                6,
            ),
            "max_agent_turn_elapsed_seconds_per_request": round(
                float(maxima["elapsed_seconds"]),
                6,
            ),
            "agent_turn_by_seat": normalized_seat_rows,
            "agent_turn_seat_fairness_facts": fairness_facts,
        }

    async def _run_analysis(self) -> None:
        """Emit a factual run summary derived from accepted state and traces.

        Research heuristics and model-graded quality scores do not belong in
        the production runtime.  Offline evaluators may consume the transcript
        separately, but their outputs must not be presented as run truth.
        """
        analysis = {
            "winner": self.state.winner,
            "days": self.state.day,
            "turn_policy": self.turn_policy,
            "seats": [
                {"seat": p.seat, "name": p.name, "role": p.role, "team": p.team,
                 "alive": p.alive, "death_reason": p.death_reason, "death_day": p.death_day}
                for p in self.state.players
            ],
            # A decision is one validated response consumed by the environment.
            # Request/response/rules trace rows have separate factual counters.
            "decision_count": len(self._consumed_decisions),
            "decision_trace_metrics": self._decision_trace_metrics(),
            "parse_metrics": self._parse_metrics(),
            "decision_failure_metrics": self._decision_failure_metrics(),
            "agent_strategy_metrics": self._agent_strategy_metrics(),
        }
        if self.termination_status != "completed":
            analysis["termination"] = self._termination_snapshot()

        await self._emit_game_ended()
        await self._emit({"type": "analysis", "analysis": analysis})

    def _agent_strategy_metrics(self) -> dict[str, Any]:
        """Compute factual metrics from private state, accepted claims and traces."""
        players_by_seat = {player.seat: player for player in self.state.players}
        seat_rows: list[dict[str, Any]] = []
        all_brier_terms: list[float] = []
        claim_count = 0
        false_role_claim_count = 0
        false_seer_result_count = 0
        seer_result_contradiction_count = 0

        for player in sorted(self.state.players, key=lambda item: item.seat):
            actor = self.actors.get(player.id)
            private_state = getattr(actor, "private_state", None)
            snapshot_reader = getattr(private_state, "snapshot", None)
            if not callable(snapshot_reader):
                continue
            snapshot = snapshot_reader()
            seat_brier_terms: list[float] = []
            for seat_text, belief in (snapshot.get("beliefs") or {}).items():
                try:
                    target = players_by_seat[int(seat_text)]
                    probability = float(belief["wolf_probability"])
                except (KeyError, TypeError, ValueError):
                    continue
                truth = 1.0 if target.role == Role.WEREWOLF else 0.0
                seat_brier_terms.append((probability - truth) ** 2)
            all_brier_terms.extend(seat_brier_terms)

            role_claims: set[str] = set()
            seer_results: dict[int, set[str]] = {}
            seat_claim_count = 0
            seat_false_role_count = 0
            seat_false_result_count = 0
            for commitment in snapshot.get("commitments") or []:
                claim = commitment.get("claim")
                if not isinstance(claim, dict):
                    continue
                claimed_role = str(claim.get("role") or "").strip().lower()
                if not claimed_role:
                    continue
                seat_claim_count += 1
                role_claims.add(claimed_role)
                if claimed_role != str(player.role):
                    seat_false_role_count += 1
                checked_seat = _as_int(claim.get("checked_seat"))
                result = str(claim.get("result") or "").strip().lower()
                if claimed_role == Role.SEER.value and checked_seat is not None and result:
                    seer_results.setdefault(checked_seat, set()).add(result)
                    target = players_by_seat.get(checked_seat)
                    expected = (
                        "wolf"
                        if target is not None and target.role == Role.WEREWOLF
                        else "village"
                    )
                    if result != expected:
                        seat_false_result_count += 1
            seat_contradictions = sum(
                max(0, len(results) - 1) for results in seer_results.values()
            )
            claim_count += seat_claim_count
            false_role_claim_count += seat_false_role_count
            false_seer_result_count += seat_false_result_count
            seer_result_contradiction_count += seat_contradictions
            seat_rows.append({
                "seat": player.seat,
                "private_state_revision": snapshot.get("revision", 0),
                "belief_count": len(snapshot.get("beliefs") or {}),
                "belief_brier_sum": (
                    round(math.fsum(seat_brier_terms), 12)
                    if seat_brier_terms
                    else 0.0
                ),
                "belief_brier": (
                    round(sum(seat_brier_terms) / len(seat_brier_terms), 6)
                    if seat_brier_terms
                    else None
                ),
                "public_commitment_count": len(snapshot.get("commitments") or []),
                "structured_claim_count": seat_claim_count,
                "false_role_claim_count": seat_false_role_count,
                "false_seer_result_count": seat_false_result_count,
                "role_claim_switch_count": max(0, len(role_claims) - 1),
                "seer_result_contradiction_count": seat_contradictions,
            })

        final_vote_rows = [
            row
            for row in self._decision_trace
            if row.get("type") == "rules_result"
            and (row.get("rules") or {}).get("action") == "kill_vote"
        ]
        final_targets = [
            int(row["target_seat"])
            for row in final_vote_rows
            if isinstance(row.get("target_seat"), int)
        ]
        return {
            "schema_version": "werewolf.agent-strategy-metrics.v1",
            "private_state_seat_count": len(seat_rows),
            "belief_observation_count": len(all_brier_terms),
            "belief_brier_sum": round(math.fsum(all_brier_terms), 12),
            "belief_brier": (
                round(sum(all_brier_terms) / len(all_brier_terms), 6)
                if all_brier_terms
                else None
            ),
            "structured_claim_count": claim_count,
            "false_role_claim_count": false_role_claim_count,
            "false_seer_result_count": false_seer_result_count,
            "seer_result_contradiction_count": seer_result_contradiction_count,
            "wolf_council_message_count": sum(
                event.type == "wolf_council_message" for event in self.state.events
            ),
            "wolf_final_vote_count": len(final_targets),
            "wolf_final_vote_target_count": len(set(final_targets)),
            "wolf_final_vote_agreement": (
                len(set(final_targets)) == 1 if final_targets else None
            ),
            "seats": seat_rows,
        }

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

    async def _emit_game_ended(self) -> None:
        """Broadcast game_ended exactly once; analysis remains the final replay event."""
        if self._game_ended_emitted:
            return
        self._game_ended_emitted = True
        payload: dict[str, Any] = {
            "type": "game_ended",
            "winner": self.state.winner.value if self.state.winner else None,
        }
        if self.termination_status != "completed":
            payload.update({
                "status": self.termination_status,
                "reason": self.termination_reason,
                "details": dict(self.termination_details),
            })
        await self._emit(payload)

    async def _emit(self, payload: dict[str, Any]) -> None:
        etype = payload.get("type")
        if etype in {
            "agent_decision_failed", "decision_envelope_rejected", "decision_validation_failed",
        }:
            reason = str(payload.get("reason") or payload.get("error") or "")
            self._decision_failures.append({
                "request_id": payload.get("request_id"),
                "day": self.state.day,
                "phase": payload.get("phase"),
                "seat": payload.get("seat"),
                "action": payload.get("action"),
                "error_type": payload.get("error_type"),
                "terminal_kind": (
                    "validation_failure"
                    if etype == "decision_validation_failed"
                    else (
                        "envelope_rejected"
                        if etype == "decision_envelope_rejected"
                        else "no_envelope"
                    )
                ),
                "reason": reason[:240],
                "timeout": bool(payload.get("timeout") or "timeout" in reason.lower()),
                "timeout_seconds": payload.get("timeout_seconds"),
            })
        if self.on_event:
            try:
                if self.internal_events:
                    await self.on_event(dict(payload))
                else:
                    public_payload = _project_live_public_event(payload)
                    if public_payload is not None:
                        await self.on_event(public_payload)
            except Exception as err:  # noqa: BLE001
                logger.debug("on_event 回调失败 error_type=%s", type(err).__name__)

    def _decision_target_seat(self, decision: Decision) -> int | None:
        return decision.target_seat

    def _decision_target_id(self, decision: Decision) -> str | None:
        seat = decision.target_seat
        if seat is None:
            return None
        player = next((item for item in self.state.players if item.seat == seat), None)
        return player.id if player is not None else None

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
    budget_scope: str | None = None,
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
            budget_scope=budget_scope,
        )
        actors[player.id] = actor
    return actors


def _rule_event_payload(event: Event, state: GameState) -> dict[str, Any]:
    payload = dict(event.payload or {})
    out: dict[str, Any] = {
        "type": event.type,
        "phase": str(event.phase),
        "day": event.day,
        "message": event.message,
        "visibility": str(event.visibility),
        "recipients": list(event.recipients),
        **payload,
    }
    if event.recipients:
        try:
            actor = state.get_player(str(event.recipients[0]))
            out.setdefault("seat", actor.seat)
        except KeyError:
            pass
    target_id = payload.get("target_id")
    if target_id:
        try:
            target = state.get_player(str(target_id))
            out["target_seat"] = target.seat
            out["target_name"] = target.name
        except KeyError:
            pass
    return out


def _project_live_public_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Project internal orchestrator events to the default public callback.

    RoomManager opts into `internal_events=True` and performs per-client
    projection itself. Direct orchestrator consumers get a fail-closed public
    stream by default.
    """
    event_type = str(payload.get("type") or "")
    visibility = str(payload.get("visibility") or "")
    recipients = payload.get("recipients") or []
    if visibility == "private" or recipients or visibility in RESTRICTED_LIVE_VISIBILITIES:
        return None
    if event_type not in LIVE_PUBLIC_EVENT_TYPES:
        return None
    visible = {
        key: value
        for key, value in payload.items()
        if not str(key).startswith("_")
    }
    visible.pop("visibility", None)
    visible.pop("recipients", None)
    for key in FORBIDDEN_LIVE_PUBLIC_KEYS:
        visible.pop(key, None)
    if "claim" in visible:
        claim = _sanitize_public_claim(visible.get("claim"))
        if claim:
            visible["claim"] = claim
        else:
            visible.pop("claim", None)
    if event_type == "night_resolved":
        visible["deaths"] = [
            {"seat": item.get("seat"), "name": item.get("name")}
            for item in visible.get("deaths", [])
            if isinstance(item, dict)
        ]
    return visible
