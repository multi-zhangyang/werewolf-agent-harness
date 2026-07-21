"""Production entry point for generic Core environments backed by real models.

The legacy Werewolf runner owns a concrete state adapter and its historical
``RunSpec``. Generic environments instead receive one independent
``CoreToolActor`` per actor ID on demand through the environment's resolver.
This module keeps that normal OpenAI/Responses/Anthropic router path available
without adding endpoint, vendor, or model-name branches.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..llm.models import ModelConfig
from ..llm.router import LLMRouter, STANDARD_PROTOCOLS
from .core_llm_actor import CoreToolActor
from .core_protocol import DecisionEnvelope as CoreDecisionEnvelope
from .core_runner import EnvironmentRunResult, run_environment_run
from .core_spec import CoreRunSpec
from .model_manifest import ModelConfigManifest
from .registry import EnvironmentRegistry
from .statistics import router_stats_delta


async def run_core_llm_environment(
    run_spec: CoreRunSpec,
    *,
    registry: EnvironmentRegistry,
    model_config: ModelConfig,
    actor_model_configs: Mapping[str, ModelConfig | Mapping[str, Any]] | None = None,
    router: LLMRouter | None = None,
    close_router: bool | None = None,
) -> EnvironmentRunResult:
    """Run a generic environment with one normal real-model Actor per ID.

    Credentials stay in the supplied in-memory ``ModelConfig`` objects. They
    do not enter ``CoreRunSpec`` or its artifacts. Every actor receives an
    independent adapter, lock, config copy and trace sink through the regular
    Core lifecycle; the Router is shared only as stateless transport/cache
    infrastructure.
    """
    if not isinstance(run_spec, CoreRunSpec):
        run_spec = CoreRunSpec.model_validate(run_spec)
    if not isinstance(registry, EnvironmentRegistry):
        raise TypeError("registry must be an EnvironmentRegistry")
    if not isinstance(model_config, ModelConfig):
        raise TypeError("model_config must be a ModelConfig")

    _require_core_tool_contract(run_spec, registry)
    overrides = _normalize_overrides(actor_model_configs)
    if run_spec.actors.human_actor_ids:
        raise ValueError(
            "run_core_llm_environment cannot construct declared human actors: "
            + ",".join(run_spec.actors.human_actor_ids)
        )
    owned_router = router is None
    active_router = router or LLMRouter()
    _require_router_shape(active_router)
    stats_before = active_router.stats.snapshot()
    actors: dict[str, CoreToolActor] = {}

    def resolve_agent(actor_id: str) -> CoreToolActor:
        normalized_actor_id = str(actor_id).strip()
        if not normalized_actor_id:
            raise ValueError("environment requested an empty actor_id")
        existing = actors.get(normalized_actor_id)
        if existing is not None:
            return existing
        resolved_config = model_config.merge(overrides.get(normalized_actor_id))
        _require_complete_model_config(resolved_config, actor_id=normalized_actor_id)
        _require_actor_provenance(
            run_spec,
            actor_id=normalized_actor_id,
            config=resolved_config,
        )
        actor = CoreToolActor(
            actor_id=normalized_actor_id,
            model_config=resolved_config,
            router=active_router,
            budget_scope=run_spec.run_id,
        )
        actors[normalized_actor_id] = actor
        return actor

    try:
        result = await run_environment_run(
            run_spec,
            registry=registry,
            resolve_agent=resolve_agent,
        )
    finally:
        stats_after = active_router.stats.snapshot()
        should_close = owned_router if close_router is None else bool(close_router)
        if should_close:
            await active_router.aclose()

    delta = router_stats_delta(stats_before, stats_after)
    metrics = dict(result.metrics)
    metrics["router_stats_delta"] = delta
    metrics["model_calls"] = int(delta.get("calls") or 0)
    harness_metrics = dict(result.harness_metrics)
    harness_metrics["router_stats_delta"] = delta
    harness_metrics["model_actor_count"] = len(actors)
    return result.model_copy(update={
        "metrics": metrics,
        "harness_metrics": harness_metrics,
    })


def _normalize_overrides(
    value: Mapping[str, ModelConfig | Mapping[str, Any]] | None,
) -> dict[str, ModelConfig]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("actor_model_configs must be a mapping")
    normalized: dict[str, ModelConfig] = {}
    for raw_actor_id, raw_config in value.items():
        actor_id = str(raw_actor_id).strip()
        if not actor_id:
            raise ValueError("actor_model_configs contains an empty actor_id")
        if actor_id in normalized:
            raise ValueError(f"duplicate actor_model_configs actor_id: {actor_id}")
        if isinstance(raw_config, ModelConfig):
            normalized[actor_id] = raw_config.model_copy(deep=True)
        elif isinstance(raw_config, Mapping):
            normalized[actor_id] = ModelConfig.model_validate(dict(raw_config))
        else:
            raise TypeError(f"model override for {actor_id!r} must be a ModelConfig or mapping")
    return normalized


def _require_core_tool_contract(
    run_spec: CoreRunSpec,
    registry: EnvironmentRegistry,
) -> None:
    """Reject an environment whose contract cannot consume ``CoreToolActor``.

    A registered environment can deliberately use another decision protocol
    (the legacy Werewolf adapter does). ``run_core_llm_environment`` creates
    only Core-tool Actors, so accepting such a plugin would defer a guaranteed
    type mismatch until after setup and obscure the actual boundary. The
    legacy adapter remains responsible for its own actor protocol.
    """
    plugin = registry.get(run_spec.environment.id, run_spec.environment.version)
    contract = getattr(plugin, "decision_contract", None)
    envelope_type = getattr(contract, "envelope_type", None)
    if envelope_type is not CoreDecisionEnvelope:
        raise ValueError(
            "environment is not compatible with the Core tool decision protocol: "
            f"{run_spec.environment.id}@{run_spec.environment.version}"
        )


def _require_router_shape(router: Any) -> None:
    if not callable(getattr(router, "complete_tools", None)):
        raise TypeError("router must provide complete_tools")
    stats = getattr(router, "stats", None)
    if not callable(getattr(stats, "snapshot", None)):
        raise TypeError("router.stats must provide snapshot")
    if not callable(getattr(router, "aclose", None)):
        raise TypeError("router must provide aclose")


def _require_complete_model_config(config: ModelConfig, *, actor_id: str) -> None:
    missing: list[str] = []
    if config.provider not in STANDARD_PROTOCOLS:
        missing.append("provider")
    if not config.model.strip():
        missing.append("model")
    if not config.api_key.strip():
        missing.append("api_key")
    if missing:
        raise ValueError(
            f"incomplete real model configuration for {actor_id}: " + ",".join(missing)
        )


def _require_actor_provenance(
    run_spec: CoreRunSpec,
    *,
    actor_id: str,
    config: ModelConfig,
) -> None:
    """Require the actual model call to match the spec's safe binding."""
    if actor_id in run_spec.actors.human_actor_ids:
        raise ValueError(
            f"CoreRunSpec declares {actor_id} as human; generic model runner cannot bind it"
        )
    raw_manifest = run_spec.actors.model_overrides.get(
        actor_id,
        run_spec.actors.default_model,
    )
    if raw_manifest is None:
        raise ValueError(f"CoreRunSpec ActorSpec has no model binding for {actor_id}")
    try:
        expected_manifest = ModelConfigManifest.model_validate(raw_manifest)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"CoreRunSpec ActorSpec model binding is invalid for {actor_id}"
        ) from err
    actual_manifest = ModelConfigManifest.from_config(config)
    if (
        actual_manifest.model_dump(mode="json")
        != expected_manifest.model_dump(mode="json")
    ):
        raise ValueError(
            f"resolved model actor does not match CoreRunSpec ActorSpec for {actor_id}"
        )
