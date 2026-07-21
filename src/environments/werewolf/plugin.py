"""Generic-harness plugin backed by the existing Werewolf domain engine."""
from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...agent.actor import AgentActor
from ...game.models import GameState, Phase
from ...game.orchestrator import GameOrchestratorV2
from ...game.roles import (
    CLASSIC_RULESET_ID,
    Role,
    validate_role_deck,
    validate_ruleset_id,
)
from ...game.rules import RulesEngine
from ...game.state import new_game
from ...harness.agent_protocol import DecisionEnvelope
from ...harness.agents import validate_decision_against_legal_actions
from ...harness.core_spec import ActorSpec
from ...harness.environment import (
    DecisionContract,
    EnvironmentDescriptor,
    EnvironmentOutcome,
    EnvironmentRunContext,
    EnvironmentSession,
)
from ...harness.spec import ModelConfigManifest
from ...llm.models import ModelConfig


class WerewolfEnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    player_names: list[str] = Field(min_length=6, max_length=12)
    role_deck: list[str]
    ruleset_id: str = CLASSIC_RULESET_ID
    turn_policy: Literal["fixed_round_robin", "bid_reply"] = "fixed_round_robin"
    max_speak_rounds: int = Field(default=3, ge=1, le=20)
    decision_timeout_seconds: float = Field(gt=0)
    decision_timeouts: dict[str, float] = Field(default_factory=dict)
    phase_deadline_seconds: float = Field(default=0, ge=0)
    phase_deadlines: dict[str, float] = Field(default_factory=dict)
    max_consecutive_decision_failures: int = Field(default=3, ge=1, le=1000)
    max_consecutive_no_progress_rounds: int = Field(default=3, ge=1, le=1000)
    max_game_rounds: int = Field(default=20, ge=1, le=1000)

    @field_validator("ruleset_id")
    @classmethod
    def _supported_ruleset(cls, value: str) -> str:
        return validate_ruleset_id(value)

    @field_validator("decision_timeouts", "phase_deadlines")
    @classmethod
    def _valid_deadline_overrides(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for raw_phase, raw_seconds in value.items():
            phase = str(raw_phase).strip()
            seconds = float(raw_seconds)
            if not phase:
                raise ValueError("deadline override phase must not be empty")
            if seconds < 0 or not math.isfinite(seconds):
                raise ValueError("deadline override must be a finite non-negative number")
            normalized[phase] = seconds
        return normalized

    @model_validator(mode="after")
    def _valid_deck(self) -> "WerewolfEnvironmentConfig":
        validate_role_deck(
            self.role_deck,
            player_count=len(self.player_names),
            ruleset_id=self.ruleset_id,
        )
        return self


class WerewolfEnvironmentPlugin:
    descriptor = EnvironmentDescriptor(
        id="werewolf.classic",
        version="1",
        required_seeds=("role", "actor", "orchestrator"),
        capabilities=(
            "multi_agent",
            "hidden_information",
            "simultaneous_actions",
            "adversarial_teams",
        ),
    )
    decision_contract = DecisionContract(
        envelope_type=DecisionEnvelope,
        validate_envelope=validate_decision_against_legal_actions,
    )

    def __init__(
        self,
        *,
        on_state_ready: Callable[[GameState], None] | None = None,
        room_state: GameState | None = None,
        on_session_ready: Callable[["_WerewolfSession"], None] | None = None,
    ) -> None:
        # The legacy wrapper may need the dealt state to construct its real
        # AgentActor objects. It can prepare its resolver here, but the plugin
        # still obtains every execution object through context.resolve_agent.
        self.on_state_ready = on_state_ready
        # Interactive rooms stage and durably publish one already-dealt state.
        # A dedicated plugin instance may consume that exact state once so the
        # Core lifecycle does not create a second game or a second actor graph.
        self.room_state = room_state
        self.on_session_ready = on_session_ready
        self._room_state_consumed = False

    def resolve_config(
        self,
        raw_config: Mapping[str, Any],
        _seeds: Mapping[str, int],
    ) -> BaseModel:
        return WerewolfEnvironmentConfig(**dict(raw_config))

    async def create_session(self, context: EnvironmentRunContext) -> EnvironmentSession:
        config = WerewolfEnvironmentConfig.model_validate(context.config)
        deck = [Role(value) for value in config.role_deck]
        state = self._resolve_state(context=context, config=config, deck=deck)
        self._validate_actor_spec(state, context.actor_spec)
        if self.on_state_ready is not None:
            self.on_state_ready(state)
        actors: dict[str, AgentActor] = {}
        private_resources: dict[int, tuple[str, str]] = {}
        for player in state.players:
            actor_id = f"seat:{player.seat}"
            actor = context.resolve_agent(actor_id)
            self._validate_actor_provenance(
                actor_id=actor_id,
                actor=actor,
                actor_spec=context.actor_spec,
            )
            self._validate_actor_identity(actor_id=actor_id, actor=actor, player=player)
            self._validate_private_resource_ownership(
                actor_id=actor_id,
                actor=actor,
                seen=private_resources,
            )
            actors[player.id] = actor
        analysis: dict[str, Any] = {}

        async def on_event(payload: dict[str, Any]) -> None:
            if payload.get("type") == "analysis" and isinstance(payload.get("analysis"), dict):
                analysis.update(payload["analysis"])
            await context.emit_event(payload)

        orchestrator = GameOrchestratorV2(
            state=state,
            actors=actors,
            deck=deck,
            rng=random.Random(context.seeds["orchestrator"]),
            on_event=on_event,
            on_trace=context.emit_trace,
            internal_events=True,
            max_speak_rounds=config.max_speak_rounds,
            turn_policy=config.turn_policy,
            decision_timeout=config.decision_timeout_seconds,
            decision_timeouts=config.decision_timeouts,
            phase_deadline=config.phase_deadline_seconds,
            phase_deadlines=config.phase_deadlines,
            decision_runtime=context.decision_runtime,
            max_consecutive_decision_failures=config.max_consecutive_decision_failures,
            max_consecutive_no_progress_rounds=config.max_consecutive_no_progress_rounds,
            max_game_rounds=config.max_game_rounds,
        )
        session = _WerewolfSession(state=state, orchestrator=orchestrator, analysis=analysis)
        if self.on_session_ready is not None:
            self.on_session_ready(session)
        return session

    def _resolve_state(
        self,
        *,
        context: EnvironmentRunContext,
        config: WerewolfEnvironmentConfig,
        deck: list[Role],
    ) -> GameState:
        if self.room_state is None:
            state = new_game(config.player_names, game_id=context.run_id)
            for player in state.players:
                player.id = f"{context.run_id}-seat-{player.seat}"
            RulesEngine.deal_roles(
                state,
                deck=deck,
                seed=context.seeds["role"],
                ruleset_id=config.ruleset_id,
            )
            return state

        if self._room_state_consumed:
            raise RuntimeError("room-owned Werewolf state may only create one session")
        state = self.room_state
        self._validate_room_state(
            state=state,
            run_id=context.run_id,
            config=config,
            deck=deck,
            role_seed=context.seeds["role"],
        )
        self._room_state_consumed = True
        return state

    @staticmethod
    def _validate_room_state(
        *,
        state: GameState,
        run_id: str,
        config: WerewolfEnvironmentConfig,
        deck: list[Role],
        role_seed: int,
    ) -> None:
        if state.id != run_id:
            raise ValueError("room-owned Werewolf state id does not match run id")
        seats = [player.seat for player in state.players]
        if seats != list(range(1, len(config.player_names) + 1)):
            raise ValueError("room-owned Werewolf state seats do not match config")
        if [player.name for player in state.players] != list(config.player_names):
            raise ValueError("room-owned Werewolf player names do not match config")
        player_ids = [player.id for player in state.players]
        if any(not str(player_id).strip() for player_id in player_ids) or len(
            set(player_ids)
        ) != len(player_ids):
            raise ValueError("room-owned Werewolf player identities are invalid")
        if state.phase != Phase.NIGHT or state.day != 1 or state.winner is not None:
            raise ValueError("room-owned Werewolf state is not a fresh dealt game")
        if any(player.role is None or not player.alive for player in state.players):
            raise ValueError("room-owned Werewolf state has incomplete role bindings")
        actual_deck = Counter(Role(player.role) for player in state.players)
        if actual_deck != Counter(deck):
            raise ValueError("room-owned Werewolf role deck does not match config")
        expected_state = new_game(config.player_names, game_id=run_id)
        RulesEngine.deal_roles(
            expected_state,
            deck=list(deck),
            seed=role_seed,
            ruleset_id=config.ruleset_id,
        )
        if [Role(player.role) for player in state.players] != [
            Role(player.role) for player in expected_state.players
        ]:
            raise ValueError("room-owned Werewolf role assignment does not match seed")

    @staticmethod
    def _validate_actor_identity(
        *,
        actor_id: str,
        actor: Any,
        player: Any,
    ) -> None:
        try:
            actor_role = Role(getattr(actor, "role"))
            player_role = Role(player.role)
        except (TypeError, ValueError) as err:
            raise ValueError(f"resolved actor role is invalid for {actor_id}") from err
        if (
            getattr(actor, "seat", None) != player.seat
            or getattr(actor, "name", None) != player.name
            or actor_role != player_role
        ):
            raise ValueError(f"resolved actor identity does not match {actor_id}")

    @staticmethod
    def _validate_private_resource_ownership(
        *,
        actor_id: str,
        actor: Any,
        seen: dict[int, tuple[str, str]],
    ) -> None:
        # Generic harness integrations may provide another AgentProtocol
        # implementation. Object uniqueness is still enforced by AgentRegistry;
        # these concrete mutable resources are the stronger AgentActor-specific
        # ownership contract used by the interactive runtime.
        if not isinstance(actor, AgentActor):
            return
        for field in ("memory", "private_state", "rng", "human_queue", "_decide_lock"):
            resource = getattr(actor, field, None)
            if resource is None:
                raise ValueError(f"resolved actor has no {field} resource for {actor_id}")
            resource_key = id(resource)
            previous = seen.get(resource_key)
            if previous is not None and previous[0] != actor_id:
                raise ValueError(
                    "resolved actors share private execution state "
                    f"({previous[0]} and {actor_id}: {field}/{previous[1]})"
                )
            seen[resource_key] = (actor_id, field)

    @staticmethod
    def _validate_actor_spec(state: GameState, actor_spec: ActorSpec) -> None:
        expected = {f"seat:{player.seat}" for player in state.players}
        declared = set(actor_spec.model_overrides) | set(actor_spec.human_actor_ids)
        unknown = sorted(declared - expected)
        if unknown:
            raise ValueError(
                "Werewolf ActorSpec contains actors outside the player seats: "
                + ",".join(unknown)
            )
        missing = sorted(
            actor_id
            for actor_id in expected
            if actor_id not in actor_spec.human_actor_ids
            and actor_id not in actor_spec.model_overrides
            and actor_spec.default_model is None
        )
        if missing:
            raise ValueError(
                "Werewolf ActorSpec has no execution binding for: "
                + ",".join(missing)
            )

    @staticmethod
    def _validate_actor_provenance(
        *,
        actor_id: str,
        actor: Any,
        actor_spec: ActorSpec,
    ) -> None:
        expected_human = actor_id in actor_spec.human_actor_ids
        actual_human = getattr(actor, "is_human", False) is True
        if actual_human != expected_human:
            raise ValueError(
                f"resolved actor kind does not match ActorSpec for {actor_id}"
            )
        if expected_human:
            return

        raw_manifest = actor_spec.model_overrides.get(
            actor_id,
            actor_spec.default_model,
        )
        if raw_manifest is None:
            raise ValueError(f"ActorSpec has no model binding for {actor_id}")
        try:
            expected_manifest = ModelConfigManifest.model_validate(raw_manifest)
        except ValueError as err:
            raise ValueError(
                f"ActorSpec model binding is invalid for {actor_id}"
            ) from err
        actual_config = getattr(actor, "model_config", None)
        if not isinstance(actual_config, ModelConfig):
            raise ValueError(
                f"resolved model actor has no attestable ModelConfig for {actor_id}"
            )
        actual_manifest = ModelConfigManifest.from_config(actual_config)
        if (
            actual_manifest.model_dump(mode="json")
            != expected_manifest.model_dump(mode="json")
        ):
            raise ValueError(
                f"resolved model actor does not match ActorSpec for {actor_id}"
            )


class _WerewolfSession:
    def __init__(
        self,
        *,
        state: GameState,
        orchestrator: GameOrchestratorV2,
        analysis: dict[str, Any],
    ) -> None:
        self.state = state
        self.orchestrator = orchestrator
        self.analysis = analysis

    async def run(self) -> EnvironmentOutcome:
        await self.orchestrator.run()
        winner = self.state.winner.value if self.state.winner else None
        status = self.orchestrator.termination_status
        if status == "completed":
            if self.state.phase != Phase.ENDED or winner is None:
                raise RuntimeError(
                    "Werewolf orchestrator reported completed without an ended state and winner"
                )
        elif status == "incomplete":
            if (
                self.state.phase != Phase.ENDED
                or winner is not None
                or not (self.orchestrator.termination_reason or "").strip()
            ):
                raise RuntimeError(
                    "Werewolf orchestrator reported an invalid incomplete terminal state"
                )
        else:
            raise RuntimeError(
                f"Werewolf orchestrator returned non-terminal status {status!r}"
            )
        outcome = {
            "winner": winner,
            "days": self.state.day,
        }
        if status == "incomplete":
            outcome["termination_reason"] = self.orchestrator.termination_reason
            outcome["termination_details"] = dict(self.orchestrator.termination_details)
        return EnvironmentOutcome(
            terminal=True,
            status="incomplete" if status == "incomplete" else "completed",
            termination_reason=(
                self.orchestrator.termination_reason if status == "incomplete" else None
            ),
            outcome=outcome,
            metrics={"analysis": dict(self.analysis)},
        )

    async def aclose(self) -> None:
        return None
