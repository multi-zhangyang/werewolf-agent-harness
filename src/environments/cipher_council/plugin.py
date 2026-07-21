"""A production hidden-information environment built only on the Core protocol.

Cipher Council is intentionally distinct from Werewolf.  A hidden Cipher
minority tries to cause secret missions to fail while the public Council tries
to identify them.  The public phases make bluffing, coalition-building, and
strategic disclosure useful; only each agent's own private identity is placed
in its observation.

All terminal behavior is environment-owned and explicit:

* a missing or skipped speech creates no speech;
* a missing or skipped Cipher strategy message creates no substitute message;
* a missing or skipped nomination fails that proposal attempt;
* a missing or skipped vote is an absence, never a fabricated rejection;
* a missing secret mission commitment voids the mission and ends the run as
  incomplete rather than inventing a support/sabotage decision.

No Werewolf module is imported here.  This is deliberately a second production
consumer of the generic Core ``ActionRequest``/``DecisionEnvelope`` contract.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...harness.core_protocol import (
    ActionChoice,
    ActionOption,
    ActionRequest,
    DecisionEnvelope,
    SkipChoice,
    SkipPolicy,
    validate_decision_envelope,
)
from ...harness.environment import (
    DecisionContract,
    EnvironmentDescriptor,
    EnvironmentOutcome,
    EnvironmentRunContext,
    EnvironmentSession,
)
from ...harness.errors import AgentDecisionError


_PUBLIC_HISTORY_ENTRY_LIMIT = 600
_CIPHER_COUNCIL_HISTORY_ENTRY_LIMIT = 120


class CipherCouncilConfig(BaseModel):
    """Validated rules for one deterministic Cipher Council run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    player_names: list[str] = Field(min_length=5, max_length=10)
    cipher_count: int = Field(default=2, ge=1)
    mission_sizes: list[int] = Field(
        default_factory=lambda: [2, 3, 2],
        min_length=1,
        max_length=5,
    )
    victory_target: int = Field(default=2, ge=1)
    max_proposals_per_mission: int = Field(default=2, ge=1, le=10)
    public_history_limit: int = Field(default=24, ge=4, le=100)

    @field_validator("player_names")
    @classmethod
    def _normalized_unique_names(cls, value: list[str]) -> list[str]:
        names = [str(item).strip() for item in value]
        if any(not name for name in names):
            raise ValueError("player names must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("player names must be unique")
        if any(len(name) > 120 for name in names):
            raise ValueError("player names must be at most 120 characters")
        return names

    @model_validator(mode="after")
    def _valid_factions_and_missions(self) -> "CipherCouncilConfig":
        player_count = len(self.player_names)
        if self.cipher_count >= player_count:
            raise ValueError("cipher_count must leave at least one Council player")
        if any(size < 2 or size > player_count for size in self.mission_sizes):
            raise ValueError("each mission size must be between 2 and player count")
        if self.victory_target > len(self.mission_sizes):
            raise ValueError("victory_target cannot exceed the configured mission count")
        return self


class CipherCouncilEnvironmentPlugin:
    """Exact v1 plugin for a multi-agent hidden-faction council."""

    descriptor = EnvironmentDescriptor(
        id="council.cipher",
        version="1",
        required_seeds=("roles", "order"),
        capabilities=(
            "multi_agent",
            "hidden_information",
            "simultaneous_actions",
            "adversarial_teams",
            "strategic_deception",
            "public_deliberation",
        ),
    )
    decision_contract = DecisionContract(
        envelope_type=DecisionEnvelope,
        validate_envelope=validate_decision_envelope,
    )
    enable_cipher_council = False

    def resolve_config(
        self,
        raw_config: Mapping[str, Any],
        _seeds: Mapping[str, int],
    ) -> BaseModel:
        return CipherCouncilConfig.model_validate(dict(raw_config))

    async def create_session(self, context: EnvironmentRunContext) -> EnvironmentSession:
        config = CipherCouncilConfig.model_validate(context.config)
        session = CipherCouncilSession(
            context=context,
            config=config,
            enable_cipher_council=self.enable_cipher_council,
        )
        await session.initialize()
        return session


class CipherCouncilV2EnvironmentPlugin(CipherCouncilEnvironmentPlugin):
    """Exact v2 plugin with simultaneous, faction-private strategy council."""

    descriptor = EnvironmentDescriptor(
        id="council.cipher",
        version="2",
        required_seeds=("roles", "order"),
        capabilities=(
            "multi_agent",
            "hidden_information",
            "simultaneous_actions",
            "adversarial_teams",
            "strategic_deception",
            "public_deliberation",
            "private_faction_coordination",
        ),
    )
    enable_cipher_council = True


class CipherCouncilSession:
    """One isolated game state; every action is delegated to its actor."""

    def __init__(
        self,
        *,
        context: EnvironmentRunContext,
        config: CipherCouncilConfig,
        enable_cipher_council: bool = False,
    ) -> None:
        self.context = context
        self.config = config
        self.enable_cipher_council = bool(enable_cipher_council)
        self.actor_ids = tuple(
            f"council:{seat}"
            for seat in range(1, len(config.player_names) + 1)
        )
        self._actors: dict[str, Any] = {}
        self._roles: dict[str, Literal["cipher", "council"]] = {}
        self._proposer_order: tuple[str, ...] = ()
        self._public_history: list[dict[str, Any]] = []
        self._cipher_council_history: list[dict[str, Any]] = []
        self._decision_count = 0
        self._decision_failure_count = 0
        self._explicit_skip_count = 0
        self._proposal_count = 0
        self._mission_count = 0
        self._cipher_council_round_count = 0
        self._cipher_council_request_count = 0
        self._cipher_council_message_count = 0
        self._cipher_council_absent_count = 0
        self._initialized = False
        self._closed = False

    async def initialize(self) -> None:
        if self._initialized:
            raise RuntimeError("Cipher Council session may only be initialized once")
        self._validate_actor_spec()
        self._roles = self._deal_roles()
        self._proposer_order = self._deal_proposer_order()
        for actor_id in self.actor_ids:
            actor = self.context.resolve_agent(actor_id)
            self._validate_actor(actor_id, actor)
            set_trace_sink = getattr(actor, "set_trace_sink", None)
            if callable(set_trace_sink):
                set_trace_sink(self.context.emit_trace)
            self._actors[actor_id] = actor

        for actor_id in self.actor_ids:
            await self.context.emit_event({
                "type": "council_role_assigned",
                "visibility": "private",
                "recipients": [actor_id],
                "actor_id": actor_id,
                "role": self._roles[actor_id],
                "teammates": self._cipher_teammates(actor_id),
            })
        await self._emit_public(
            "council_started",
            actor_ids=list(self.actor_ids),
            player_names=list(self.config.player_names),
            mission_count=len(self.config.mission_sizes),
            victory_target=self.config.victory_target,
        )
        self._initialized = True

    async def run(self) -> EnvironmentOutcome:
        if not self._initialized:
            raise RuntimeError("Cipher Council session was not initialized")
        if self._closed:
            raise RuntimeError("Cipher Council session is closed")

        council_successes = 0
        cipher_failures = 0
        proposer_index = 0
        for mission, mission_size in enumerate(self.config.mission_sizes, start=1):
            proposal_accepted = False
            for proposal_attempt in range(1, self.config.max_proposals_per_mission + 1):
                proposer = self._proposer_order[proposer_index]
                proposer_index = (proposer_index + 1) % len(self._proposer_order)
                self._proposal_count += 1
                await self._emit_public(
                    "council_round_started",
                    mission=mission,
                    proposal_attempt=proposal_attempt,
                    proposer=proposer,
                    mission_size=mission_size,
                )
                await self._run_cipher_council(
                    mission=mission,
                    proposal_attempt=proposal_attempt,
                    proposer=proposer,
                    mission_size=mission_size,
                )
                await self._run_deliberation(
                    mission=mission,
                    proposal_attempt=proposal_attempt,
                    proposer=proposer,
                    mission_size=mission_size,
                )
                team = await self._request_nomination(
                    mission=mission,
                    proposal_attempt=proposal_attempt,
                    proposer=proposer,
                    mission_size=mission_size,
                )
                if team is None:
                    await self._emit_public(
                        "council_proposal_unavailable",
                        mission=mission,
                        proposal_attempt=proposal_attempt,
                        proposer=proposer,
                        reason="no_nomination",
                    )
                    continue
                await self._emit_public(
                    "council_proposal_submitted",
                    mission=mission,
                    proposal_attempt=proposal_attempt,
                    proposer=proposer,
                    members=team,
                )
                votes = await self._request_votes(
                    mission=mission,
                    proposal_attempt=proposal_attempt,
                    proposer=proposer,
                    team=team,
                )
                approvals = sum(1 for approve in votes.values() if approve)
                rejections = sum(1 for approve in votes.values() if not approve)
                absent = len(self.actor_ids) - len(votes)
                approved = approvals > len(self.actor_ids) / 2
                await self._emit_public(
                    "council_proposal_resolved",
                    mission=mission,
                    proposal_attempt=proposal_attempt,
                    proposer=proposer,
                    members=team,
                    approved=approved,
                    approvals=approvals,
                    rejections=rejections,
                    absent=absent,
                )
                if not approved:
                    continue

                proposal_accepted = True
                await self._emit_public(
                    "council_mission_started",
                    mission=mission,
                    members=team,
                )
                commitments = await self._request_secret_commitments(
                    mission=mission,
                    team=team,
                    proposer=proposer,
                )
                if commitments is None:
                    await self._emit_public(
                        "council_mission_void",
                        mission=mission,
                        members=team,
                        reason="mission_commitment_missing",
                    )
                    return await self._incomplete_outcome(
                        "mission_commitment_missing",
                        council_successes=council_successes,
                        cipher_failures=cipher_failures,
                    )

                sabotage_count = sum(
                    commitment == "sabotage" for commitment in commitments.values()
                )
                self._mission_count += 1
                mission_result = "failed" if sabotage_count else "succeeded"
                await self._emit_public(
                    "council_mission_resolved",
                    mission=mission,
                    members=team,
                    result=mission_result,
                )
                if sabotage_count:
                    cipher_failures += 1
                else:
                    council_successes += 1
                break

            if not proposal_accepted:
                await self._emit_public(
                    "council_proposal_limit_exhausted",
                    mission=mission,
                    max_proposals=self.config.max_proposals_per_mission,
                )
                return await self._incomplete_outcome(
                    "proposal_limit_exhausted",
                    council_successes=council_successes,
                    cipher_failures=cipher_failures,
                )
            if council_successes >= self.config.victory_target:
                return await self._completed_outcome(
                    "council",
                    council_successes=council_successes,
                    cipher_failures=cipher_failures,
                )
            if cipher_failures >= self.config.victory_target:
                return await self._completed_outcome(
                    "cipher",
                    council_successes=council_successes,
                    cipher_failures=cipher_failures,
                )

        # Config validation permits a target equal to mission count, so this
        # branch is theoretically reachable only with an unusual tied rule
        # configuration. It remains explicit rather than inferring a winner.
        return await self._incomplete_outcome(
            "victory_target_unreached",
            council_successes=council_successes,
            cipher_failures=cipher_failures,
        )

    async def aclose(self) -> None:
        self._closed = True

    def _validate_actor_spec(self) -> None:
        declared = set(self.context.actor_spec.model_overrides) | set(
            self.context.actor_spec.human_actor_ids
        )
        unknown = sorted(declared - set(self.actor_ids))
        if unknown:
            raise ValueError(
                "Cipher Council ActorSpec contains actors outside the council: "
                + ",".join(unknown)
            )

    @staticmethod
    def _validate_actor(actor_id: str, actor: Any) -> None:
        if getattr(actor, "actor_id", None) != actor_id:
            raise ValueError(f"resolved actor identity does not match {actor_id}")
        if not callable(getattr(actor, "decide", None)):
            raise ValueError(f"resolved actor does not implement decide for {actor_id}")

    def _deal_roles(self) -> dict[str, Literal["cipher", "council"]]:
        ordered = list(self.actor_ids)
        self.context.rng("roles").shuffle(ordered)
        cipher_ids = set(ordered[: self.config.cipher_count])
        return {
            actor_id: "cipher" if actor_id in cipher_ids else "council"
            for actor_id in self.actor_ids
        }

    def _deal_proposer_order(self) -> tuple[str, ...]:
        ordered = list(self.actor_ids)
        self.context.rng("order").shuffle(ordered)
        return tuple(ordered)

    def _cipher_teammates(self, actor_id: str) -> list[str]:
        if self._roles[actor_id] != "cipher":
            return []
        return [
            candidate
            for candidate in self.actor_ids
            if candidate != actor_id and self._roles[candidate] == "cipher"
        ]

    def _cipher_actor_ids(self) -> tuple[str, ...]:
        return tuple(
            actor_id
            for actor_id in self.actor_ids
            if self._roles[actor_id] == "cipher"
        )

    async def _run_cipher_council(
        self,
        *,
        mission: int,
        proposal_attempt: int,
        proposer: str,
        mission_size: int,
    ) -> None:
        """Collect one simultaneous strategy message from each Cipher.

        The current round's messages are not appended until every Cipher
        decision resolves. Each model therefore sees only prior council rounds
        while composing its own message, and no process acts as a shared team
        brain. A skip or failure is an absent message, never a fabricated
        strategy, and remains visible only in the admin decision trace.
        """
        if not self.enable_cipher_council:
            return
        cipher_ids = self._cipher_actor_ids()
        if len(cipher_ids) < 2:
            return
        self._cipher_council_round_count += 1
        self._cipher_council_request_count += len(cipher_ids)
        tasks = [
            asyncio.create_task(self._decide(
                actor_id=actor_id,
                stage="cipher_council",
                mission=mission,
                proposal_attempt=proposal_attempt,
                proposer=proposer,
                mission_size=mission_size,
                legal_actions=[_cipher_council_message_option()],
                skip_policy=SkipPolicy(allowed=True, reason_required=False),
            ))
            for actor_id in cipher_ids
        ]
        choices = await asyncio.gather(*tasks)
        for actor_id, choice in zip(cipher_ids, choices, strict=True):
            if choice is None:
                self._cipher_council_absent_count += 1
                continue
            message = str(choice.arguments["message"])
            entry = {
                "mission": mission,
                "proposal_attempt": proposal_attempt,
                "actor_id": actor_id,
                "message": message,
            }
            self._cipher_council_history.append(entry)
            if len(self._cipher_council_history) > _CIPHER_COUNCIL_HISTORY_ENTRY_LIMIT:
                del self._cipher_council_history[:-_CIPHER_COUNCIL_HISTORY_ENTRY_LIMIT]
            self._cipher_council_message_count += 1
            await self.context.emit_event({
                "type": "council_cipher_message",
                "visibility": "private",
                "recipients": list(cipher_ids),
                **deepcopy(entry),
            })

    async def _run_deliberation(
        self,
        *,
        mission: int,
        proposal_attempt: int,
        proposer: str,
        mission_size: int,
    ) -> None:
        for actor_id in self.actor_ids:
            choice = await self._decide(
                actor_id=actor_id,
                stage="deliberation",
                mission=mission,
                proposal_attempt=proposal_attempt,
                proposer=proposer,
                mission_size=mission_size,
                legal_actions=[_speech_option()],
                skip_policy=SkipPolicy(allowed=True, reason_required=False),
            )
            # A skipped/missing speech is intentionally not represented as a
            # synthetic public statement.
            if choice is None:
                continue
            await self._emit_public(
                "council_speech",
                mission=mission,
                proposal_attempt=proposal_attempt,
                actor_id=actor_id,
                text=str(choice.arguments["message"]),
            )

    async def _request_nomination(
        self,
        *,
        mission: int,
        proposal_attempt: int,
        proposer: str,
        mission_size: int,
    ) -> list[str] | None:
        choice = await self._decide(
            actor_id=proposer,
            stage="nomination",
            mission=mission,
            proposal_attempt=proposal_attempt,
            proposer=proposer,
            mission_size=mission_size,
            legal_actions=[_nomination_option(self.actor_ids, mission_size)],
            # A proposer has a well-defined legal team set and must submit a
            # proposal. Provider failure or an invalid/attempted skip remains
            # a missing nomination and fails this proposal attempt; it is not
            # silently converted into a default team.
            skip_policy=SkipPolicy(allowed=False),
        )
        if choice is None:
            return None
        return [str(member) for member in choice.arguments["members"]]

    async def _request_votes(
        self,
        *,
        mission: int,
        proposal_attempt: int,
        proposer: str,
        team: list[str],
    ) -> dict[str, bool]:
        votes: dict[str, bool] = {}
        for actor_id in self.actor_ids:
            choice = await self._decide(
                actor_id=actor_id,
                stage="vote",
                mission=mission,
                proposal_attempt=proposal_attempt,
                proposer=proposer,
                mission_size=len(team),
                team=team,
                legal_actions=[_vote_option()],
                skip_policy=SkipPolicy(allowed=True, reason_required=False),
            )
            # An absent vote is deliberately excluded from ``votes``. It is
            # neither an affirmative nor a fabricated rejection.
            if choice is None:
                continue
            approve = bool(choice.arguments["approve"])
            votes[actor_id] = approve
            await self._emit_public(
                "council_vote_cast",
                mission=mission,
                proposal_attempt=proposal_attempt,
                actor_id=actor_id,
                approve=approve,
            )
        return votes

    async def _request_secret_commitments(
        self,
        *,
        mission: int,
        team: list[str],
        proposer: str,
    ) -> dict[str, str] | None:
        tasks = [
            asyncio.create_task(self._decide(
                actor_id=actor_id,
                stage="mission_commitment",
                mission=mission,
                proposal_attempt=0,
                proposer=proposer,
                mission_size=len(team),
                team=team,
                legal_actions=[_commitment_option(self._roles[actor_id])],
                skip_policy=SkipPolicy(allowed=False),
            ))
            for actor_id in team
        ]
        choices = await asyncio.gather(*tasks)
        if any(choice is None for choice in choices):
            return None
        commitments: dict[str, str] = {}
        for actor_id, choice in zip(team, choices, strict=True):
            if choice is None:  # Narrowed above; retain a defensive guard.
                return None
            commitment = str(choice.arguments["commitment"])
            commitments[actor_id] = commitment
            await self.context.emit_event({
                "type": "council_mission_commitment",
                "visibility": "private",
                "recipients": [actor_id],
                "mission": mission,
                "actor_id": actor_id,
                "commitment": commitment,
            })
        return commitments

    async def _decide(
        self,
        *,
        actor_id: str,
        stage: str,
        mission: int,
        proposal_attempt: int,
        proposer: str,
        mission_size: int,
        legal_actions: list[ActionOption],
        skip_policy: SkipPolicy,
        team: list[str] | None = None,
    ) -> ActionChoice | None:
        request = ActionRequest(
            request_id=(
                f"{self.context.run_id}:m{mission}:p{proposal_attempt}:"
                f"{stage}:{actor_id}"
            ),
            run_id=self.context.run_id,
            actor_id=actor_id,
            observation=self._observation(
                actor_id=actor_id,
                stage=stage,
                mission=mission,
                proposal_attempt=proposal_attempt,
                proposer=proposer,
                mission_size=mission_size,
                team=team,
            ),
            legal_actions=legal_actions,
            skip_policy=skip_policy,
            labels={
                "environment": "council.cipher",
                "stage": stage,
                "mission": mission,
                "proposal_attempt": proposal_attempt,
            },
            metadata={
                "decision_stage": stage,
                "public_action": stage in {"deliberation", "nomination", "vote"},
            },
        )
        try:
            envelope = await self.context.decision_runtime.execute(
                self._actors[actor_id],
                request,
            )
        except AgentDecisionError:
            self._decision_failure_count += 1
            # Secret commitment failures are evidenced in the admin-only
            # DecisionRuntime trace and collapse publicly into the later void
            # result. Other missing public actions may be shown as absent.
            if stage not in {"mission_commitment", "cipher_council"}:
                await self._emit_public(
                    "council_action_unavailable",
                    mission=mission,
                    proposal_attempt=proposal_attempt,
                    stage=stage,
                    actor_id=actor_id,
                )
            return None
        self._decision_count += 1
        self._record_consumed_decision(request, envelope, stage=stage)
        if isinstance(envelope.choice, SkipChoice):
            self._explicit_skip_count += 1
            return None
        if not isinstance(envelope.choice, ActionChoice):
            raise RuntimeError("Core decision did not contain an action choice")
        return envelope.choice

    def _observation(
        self,
        *,
        actor_id: str,
        stage: str,
        mission: int,
        proposal_attempt: int,
        proposer: str,
        mission_size: int,
        team: list[str] | None,
    ) -> dict[str, Any]:
        identity: dict[str, Any] = {
            "actor_id": actor_id,
            "faction": self._roles[actor_id],
            "objective": (
                "Cause enough missions to fail without exposing the Cipher faction. "
                "Public claims may be strategically deceptive."
                if self._roles[actor_id] == "cipher"
                else "Complete enough missions and use public evidence to identify Cipher agents."
            ),
        }
        if self._roles[actor_id] == "cipher":
            identity["cipher_teammates"] = self._cipher_teammates(actor_id)
            if self.enable_cipher_council:
                identity["cipher_council_messages"] = deepcopy(
                    self._cipher_council_history[-_CIPHER_COUNCIL_HISTORY_ENTRY_LIMIT :]
                )
        public_state: dict[str, Any] = {
            "mission": mission,
            "proposal_attempt": proposal_attempt,
            "proposer": proposer,
            "mission_size": mission_size,
            "stage": stage,
            "actor_ids": list(self.actor_ids),
        }
        if team is not None:
            public_state["proposed_team"] = list(team)
        stage_guidance = {
            "cipher_council": (
                "You are a Cipher. Send one faction-private strategy message or use an "
                "advertised explicit skip. Messages from this same council round are "
                "delivered only after every Cipher response resolves."
            ),
            "deliberation": "You may speak publicly or use an advertised explicit skip.",
            "nomination": "You are the current proposer. Submit exactly one legal mission team.",
            "vote": "Cast a public approval/rejection vote, or abstain only if skip is advertised.",
            "mission_commitment": "Submit your secret legal mission commitment now.",
        }
        return {
            "environment": "Cipher Council",
            "private_identity": identity,
            "public_state": public_state,
            "public_history": deepcopy(
                self._public_history[-self.config.public_history_limit :]
            ),
            "action_visibility": (
                "team_private"
                if stage == "cipher_council"
                else "secret"
                if stage == "mission_commitment"
                else "public"
            ),
            "stage_guidance": stage_guidance.get(stage, "Use one advertised legal action."),
        }

    def _record_consumed_decision(
        self,
        request: ActionRequest,
        envelope: DecisionEnvelope,
        *,
        stage: str,
    ) -> None:
        """Record the environment's acceptance of one validated Core choice.

        ``DecisionRuntime`` owns request/terminal pairing. The environment owns
        the separate fact that it consumed a valid choice under its own rules.
        These rows are admin-only decision evidence, never public game events.
        """
        choice = envelope.choice
        if isinstance(choice, ActionChoice):
            action = choice.action
            decision: dict[str, Any] = {
                "kind": "action",
                "action": action,
                "arguments": deepcopy(choice.arguments),
            }
            status = "accepted"
        else:
            action = "skip"
            decision = {"kind": "skip", "reason": choice.reason}
            status = "skipped"
        metadata = dict(envelope.metadata)
        raw_llm_call = metadata.get("llm_call")
        llm_call = (
            deepcopy(raw_llm_call)
            if isinstance(raw_llm_call, Mapping)
            else None
        )
        self.context.emit_trace({
            "type": "decision_consumed",
            "visibility": "admin",
            "audience": "admin",
            "request_id": request.request_id,
            "actor_id": request.actor_id,
            "stage": stage,
            "action": action,
            "decision": decision,
            "model_call_id": envelope.model_call_id,
            "call_id": envelope.model_call_id,
            "llm_call": llm_call,
        })
        self.context.emit_trace({
            "type": "rules_result",
            "visibility": "admin",
            "audience": "admin",
            "request_id": request.request_id,
            "actor_id": request.actor_id,
            "stage": stage,
            "rules": {"status": status, "action": action},
        })

    async def _emit_public(self, event_type: str, **fields: Any) -> None:
        payload = {"type": event_type, "visibility": "public", **fields}
        await self.context.emit_event(payload)
        history_entry = {"type": event_type, **deepcopy(fields)}
        self._public_history.append(history_entry)
        if len(self._public_history) > _PUBLIC_HISTORY_ENTRY_LIMIT:
            del self._public_history[:-_PUBLIC_HISTORY_ENTRY_LIMIT]

    async def _completed_outcome(
        self,
        winner: Literal["cipher", "council"],
        *,
        council_successes: int,
        cipher_failures: int,
    ) -> EnvironmentOutcome:
        await self._emit_public(
            "council_game_ended",
            winner=winner,
            reason="victory_target_reached",
            council_successes=council_successes,
            cipher_failures=cipher_failures,
        )
        return EnvironmentOutcome(
            terminal=True,
            outcome={
                "winner": winner,
                "council_successes": council_successes,
                "cipher_failures": cipher_failures,
                "missions_resolved": self._mission_count,
            },
            metrics=self._metrics(),
        )

    async def _incomplete_outcome(
        self,
        reason: str,
        *,
        council_successes: int,
        cipher_failures: int,
    ) -> EnvironmentOutcome:
        await self._emit_public(
            "council_game_incomplete",
            reason=reason,
            council_successes=council_successes,
            cipher_failures=cipher_failures,
        )
        return EnvironmentOutcome(
            terminal=True,
            status="incomplete",
            termination_reason=reason,
            outcome={
                "winner": None,
                "council_successes": council_successes,
                "cipher_failures": cipher_failures,
                "missions_resolved": self._mission_count,
            },
            metrics=self._metrics(),
        )

    def _metrics(self) -> dict[str, int]:
        metrics = {
            "player_count": len(self.actor_ids),
            "decision_count": self._decision_count,
            "decision_failure_count": self._decision_failure_count,
            "explicit_skip_count": self._explicit_skip_count,
            "proposal_count": self._proposal_count,
            "missions_resolved": self._mission_count,
        }
        if self.enable_cipher_council:
            cipher_faction_size = len(self._cipher_actor_ids())
            if (
                self._cipher_council_message_count
                + self._cipher_council_absent_count
                != self._cipher_council_request_count
            ):
                raise RuntimeError("Cipher council request accounting is inconsistent")
            if (
                self._cipher_council_request_count
                != self._cipher_council_round_count * cipher_faction_size
            ):
                raise RuntimeError("Cipher council round accounting is inconsistent")
            metrics.update({
                "cipher_council_faction_size": cipher_faction_size,
                "cipher_council_round_count": self._cipher_council_round_count,
                "cipher_council_request_count": self._cipher_council_request_count,
                "cipher_council_message_count": self._cipher_council_message_count,
                "cipher_council_absent_count": self._cipher_council_absent_count,
            })
        return metrics


def _speech_option() -> ActionOption:
    return ActionOption(
        name="speak",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "minLength": 1, "maxLength": 600},
            },
            "required": ["message"],
            "additionalProperties": False,
        },
        metadata={"visibility": "public", "stage": "deliberation"},
    )


def _cipher_council_message_option() -> ActionOption:
    return ActionOption(
        name="send_cipher_strategy_message",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "minLength": 1, "maxLength": 1000},
            },
            "required": ["message"],
            "additionalProperties": False,
        },
        metadata={"visibility": "private", "stage": "cipher_council"},
    )


def _nomination_option(actor_ids: tuple[str, ...], mission_size: int) -> ActionOption:
    return ActionOption(
        name="nominate_mission_team",
        input_schema={
            "type": "object",
            "properties": {
                "members": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(actor_ids)},
                    "minItems": mission_size,
                    "maxItems": mission_size,
                    "uniqueItems": True,
                },
            },
            "required": ["members"],
            "additionalProperties": False,
        },
        metadata={"visibility": "public", "stage": "nomination"},
    )


def _vote_option() -> ActionOption:
    return ActionOption(
        name="cast_public_vote",
        input_schema={
            "type": "object",
            "properties": {"approve": {"type": "boolean"}},
            "required": ["approve"],
            "additionalProperties": False,
        },
        metadata={"visibility": "public", "stage": "vote"},
    )


def _commitment_option(role: Literal["cipher", "council"]) -> ActionOption:
    commitments = ["support", "sabotage"] if role == "cipher" else ["support"]
    return ActionOption(
        name="commit_secret_mission_action",
        input_schema={
            "type": "object",
            "properties": {
                "commitment": {"type": "string", "enum": commitments},
            },
            "required": ["commitment"],
            "additionalProperties": False,
        },
        metadata={"visibility": "private", "stage": "mission_commitment"},
    )
