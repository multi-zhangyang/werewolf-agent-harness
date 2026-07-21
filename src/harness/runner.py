"""Production entry point for one real Werewolf harness run."""
from __future__ import annotations

import asyncio
import random
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..config import (
    AGENT_DECISION_TIMEOUT,
    AGENT_PHASE_DEADLINE,
    LLM_CONCURRENCY,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
)
from ..environments.werewolf import WerewolfEnvironmentPlugin
from ..game.orchestrator import build_actors
from ..game.roles import (
    default_role_deck,
    validate_role_deck,
    validate_ruleset_id,
)
from ..llm.models import ModelConfig
from ..llm.router import LLMRouter, STANDARD_PROTOCOLS
from .core_runner import run_environment_run
from .registry import EnvironmentRegistry
from .schedule import resolved_role_layout_metadata, validate_persona_provenance
from .spec import ModelConfigManifest, RunSpec
from .spec_loader import legacy_werewolf_run_to_core
from .statistics import router_stats_delta


class HarnessRunResult(BaseModel):
    """Factual outcome, provenance, cost, and immutable transcript for one run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str
    termination_reason: str | None = None
    winner: str | None = None
    days: int = 0
    elapsed_seconds: float = 0.0
    error_type: str | None = None
    error: str | None = None
    run_spec_hash: str
    role_seed: int
    actor_seed: int
    orchestrator_seed: int
    run_spec: dict[str, Any] = Field(default_factory=dict)
    event_count: int = 0
    decision_trace_count: int = 0
    transcript_digest: str
    transcript: dict[str, Any] = Field(default_factory=dict)
    analysis: dict[str, Any] | None = None
    harness_metrics: dict[str, Any] = Field(default_factory=dict)
    router_stats: dict[str, Any] = Field(default_factory=dict)
    router_stats_delta: dict[str, Any] = Field(default_factory=dict)


async def run_werewolf_run(
    run_spec: RunSpec,
    *,
    model_config: ModelConfig,
    seat_model_configs: dict[int, ModelConfig | dict[str, Any]] | None = None,
    router: LLMRouter | None = None,
    close_router: bool | None = None,
) -> HarnessRunResult:
    """Execute one real LLM-only run from a fully recorded ``RunSpec``.

    There is intentionally no scripted/replay factory branch. Human seats are
    interactive and belong to the room runtime, not this unattended runner.
    """
    _require_reproducible_seeds(run_spec)
    if run_spec.human_seats:
        raise ValueError("offline run_werewolf_run does not support interactive human seats")

    resolved_seat_configs = _normalize_seat_model_configs(seat_model_configs)
    resolved_run_spec = resolve_run_spec(
        run_spec,
        model_config=model_config,
        seat_model_configs=resolved_seat_configs,
    )
    _require_complete_model_configs(
        player_count=len(resolved_run_spec.player_names),
        model_config=model_config,
        seat_model_configs=resolved_seat_configs,
    )
    core_spec = legacy_werewolf_run_to_core(resolved_run_spec)
    owned_router = router is None
    active_router = router or LLMRouter(
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
        concurrency=LLM_CONCURRENCY,
    )
    stats_before = active_router.stats.snapshot()
    runtime_agents: dict[str, Any] = {}
    actor_builder = _actor_builder_for_run(resolved_run_spec)

    def prepare_runtime_agents(state: Any) -> None:
        if runtime_agents:
            raise RuntimeError("Werewolf runtime agents were prepared more than once")
        built = actor_builder(
            state,
            model_config=model_config,
            router=active_router,
            seat_configs=resolved_seat_configs,
            human_seats=set(),
            rng=random.Random(int(resolved_run_spec.actor_seed)),
        )
        expected_player_ids = {str(player.id) for player in state.players}
        if set(built) != expected_player_ids:
            raise ValueError("actor builder must exactly cover the dealt Werewolf state")
        for player in state.players:
            runtime_agents[f"seat:{player.seat}"] = built[str(player.id)]

    def resolve_runtime_agent(actor_id: str) -> Any:
        try:
            return runtime_agents[actor_id]
        except KeyError as err:
            raise LookupError(f"no prepared Werewolf actor for {actor_id}") from err

    plugin = WerewolfEnvironmentPlugin(on_state_ready=prepare_runtime_agents)
    registry = EnvironmentRegistry()
    registry.register(plugin)
    try:
        core_result = await run_environment_run(
            core_spec,
            registry=registry,
            resolve_agent=resolve_runtime_agent,
        )
    finally:
        stats_after = active_router.stats.snapshot()
        should_close = owned_router if close_router is None else close_router
        if should_close:
            await active_router.aclose()
    analysis_value = core_result.metrics.get("analysis")
    analysis = dict(analysis_value) if isinstance(analysis_value, dict) else None
    entries = core_result.transcript.get("entries") or []
    return HarnessRunResult(
        run_id=resolved_run_spec.run_id,
        status=core_result.status,
        termination_reason=core_result.termination_reason,
        winner=core_result.outcome.get("winner"),
        days=int(core_result.outcome.get("days") or 0),
        elapsed_seconds=core_result.elapsed_seconds,
        error_type=core_result.error_type,
        error=core_result.error,
        run_spec_hash=resolved_run_spec.spec_hash,
        role_seed=int(resolved_run_spec.role_seed),
        actor_seed=int(resolved_run_spec.actor_seed),
        orchestrator_seed=int(resolved_run_spec.orchestrator_seed),
        run_spec=resolved_run_spec.model_dump(),
        event_count=sum(1 for entry in entries if entry.get("kind") == "event"),
        decision_trace_count=sum(1 for entry in entries if entry.get("kind") == "decision"),
        transcript_digest=core_result.transcript_digest,
        transcript=core_result.transcript,
        analysis=analysis,
        harness_metrics=core_result.harness_metrics,
        router_stats=stats_after,
        router_stats_delta=router_stats_delta(stats_before, stats_after),
    )


def run_werewolf_run_sync(run_spec: RunSpec, **kwargs: Any) -> HarnessRunResult:
    return asyncio.run(run_werewolf_run(run_spec, **kwargs))


def _require_reproducible_seeds(run_spec: RunSpec) -> None:
    missing = [
        name for name in ("role_seed", "actor_seed", "orchestrator_seed")
        if getattr(run_spec, name) is None
    ]
    if missing:
        raise ValueError("harness run requires explicit reproducibility seeds: " + ",".join(missing))


def _normalize_seat_model_configs(
    seat_model_configs: dict[int, ModelConfig | dict[str, Any]] | None,
) -> dict[int, ModelConfig]:
    return {
        int(seat): config if isinstance(config, ModelConfig) else ModelConfig(**config)
        for seat, config in (seat_model_configs or {}).items()
    }


def _require_complete_model_configs(
    *,
    player_count: int,
    model_config: ModelConfig,
    seat_model_configs: dict[int, ModelConfig],
) -> None:
    errors: list[str] = []
    for seat in range(1, player_count + 1):
        config = model_config.merge(seat_model_configs.get(seat))
        missing: list[str] = []
        if config.provider not in STANDARD_PROTOCOLS:
            missing.append("provider")
        if not (config.model or "").strip():
            missing.append("model")
        if not (config.api_key or "").strip():
            missing.append("api_key")
        if missing:
            errors.append(f"seat {seat}: {','.join(missing)}")
    if errors:
        raise ValueError("incomplete real model configuration: " + "; ".join(errors))


def resolve_run_spec(
    run_spec: RunSpec,
    *,
    model_config: ModelConfig,
    seat_model_configs: dict[int, ModelConfig | dict[str, Any]] | None = None,
) -> RunSpec:
    """Resolve every execution-relevant default before hashing or resuming."""
    _require_reproducible_seeds(run_spec)
    validate_persona_provenance(
        run_spec.metadata,
        player_names=run_spec.player_names,
    )
    normalized = _normalize_seat_model_configs(seat_model_configs)
    valid_seats = set(range(1, len(run_spec.player_names) + 1))
    unknown_seats = sorted(set(normalized) - valid_seats)
    if unknown_seats:
        raise ValueError(f"seat model config outside player range: {unknown_seats}")
    ruleset_id = validate_ruleset_id(run_spec.ruleset_id)
    role_deck = validate_role_deck(
        run_spec.role_deck or default_role_deck(len(run_spec.player_names)),
        player_count=len(run_spec.player_names),
        ruleset_id=ruleset_id,
    )
    resolved = _run_spec_with_model_manifests(
        run_spec,
        model_config=model_config,
        seat_model_configs=normalized,
    )
    resolved_role_deck = [role.value for role in role_deck]
    resolved_metadata = dict(resolved.metadata)
    resolved_metadata.update(resolved_role_layout_metadata(
        resolved_metadata,
        ruleset_id=ruleset_id,
        role_deck=resolved_role_deck,
        role_seed=int(resolved.role_seed),
    ))
    return resolved.model_copy(update={
        "ruleset_id": ruleset_id,
        "role_deck": resolved_role_deck,
        "metadata": resolved_metadata,
        "decision_timeout_seconds": (
            resolved.decision_timeout_seconds
            if resolved.decision_timeout_seconds is not None
            else AGENT_DECISION_TIMEOUT
        ),
        "phase_deadline_seconds": (
            resolved.phase_deadline_seconds
            if resolved.phase_deadline_seconds is not None
            else AGENT_PHASE_DEADLINE
        ),
    })


def _run_spec_with_model_manifests(
    run_spec: RunSpec,
    *,
    model_config: ModelConfig,
    seat_model_configs: dict[int, ModelConfig],
) -> RunSpec:
    seat_manifests = dict(run_spec.seat_models)
    for seat, override in seat_model_configs.items():
        seat_manifests[seat] = ModelConfigManifest.from_config(model_config.merge(override))
    return run_spec.model_copy(update={
        "default_model": ModelConfigManifest.from_config(model_config),
        "seat_models": seat_manifests,
    })


def _actor_builder_for_run(run_spec: RunSpec) -> Any:
    """Bind an explicit persona plan without changing the one-Actor-per-seat path."""

    assignments = validate_persona_provenance(
        run_spec.metadata,
        player_names=run_spec.player_names,
    )
    if not assignments:
        return build_actors
    assignment_by_seat = {int(item["seat"]): item for item in assignments}

    def build_persona_actors(state: Any, **kwargs: Any) -> dict[str, Any]:
        actors = build_actors(state, **kwargs)
        if len(actors) != len(state.players) or len({id(actor) for actor in actors.values()}) != len(actors):
            raise ValueError("persona controls require one unique AgentActor per player")
        for player in state.players:
            assignment = assignment_by_seat.get(int(player.seat))
            actor = actors.get(str(player.id))
            if assignment is None or actor is None:
                raise ValueError("persona controls must cover every player Actor")
            if actor.seat != player.seat or actor.name != player.name:
                raise ValueError("persona controls cannot change Actor seat/name ownership")
            actor.persona_name = str(assignment["name"])
            actor.persona_desc = str(assignment["description"])
        return actors

    return build_persona_actors
