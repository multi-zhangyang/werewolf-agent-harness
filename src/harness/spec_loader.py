"""Exact-version loading and migration into the environment-neutral RunSpec."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .core_spec import (
    CORE_RUN_SPEC_VERSION,
    ActorSpec,
    CoreRunSpec,
    EnvironmentRef,
    ExecutionSpec,
)
from .spec import SPEC_SCHEMA_VERSION, RunSpec


class RunSpecLoadError(ValueError):
    """Raised when a RunSpec version is missing, unknown, or cannot migrate."""


def load_core_run_spec(value: CoreRunSpec | RunSpec | Mapping[str, Any]) -> CoreRunSpec:
    """Load an exact Core v1 spec or explicitly migrate a Werewolf v3 spec."""
    if isinstance(value, CoreRunSpec):
        return CoreRunSpec.model_validate(value.model_dump())
    if isinstance(value, RunSpec):
        return legacy_werewolf_run_to_core(value)
    if not isinstance(value, Mapping):
        raise TypeError("RunSpec input must be a mapping, CoreRunSpec, or RunSpec")

    raw = dict(value)
    schema_version = raw.get("schema_version")
    try:
        if schema_version == CORE_RUN_SPEC_VERSION:
            return CoreRunSpec.model_validate(raw)
        if schema_version == SPEC_SCHEMA_VERSION:
            return legacy_werewolf_run_to_core(RunSpec.model_validate(raw))
    except ValueError as err:
        raise RunSpecLoadError(
            f"invalid {schema_version or 'unversioned'} RunSpec"
        ) from err
    raise RunSpecLoadError(f"unsupported RunSpec schema_version: {schema_version!r}")


def legacy_werewolf_run_to_core(run_spec: RunSpec) -> CoreRunSpec:
    """Translate one v3 Werewolf spec without claiming its hash is a Core hash."""
    legacy = RunSpec.model_validate(run_spec.model_dump())
    if legacy.environment_id != "werewolf.classic" or legacy.environment_version != "1":
        raise RunSpecLoadError(
            "Werewolf v3 migration supports only exact environment werewolf.classic@1"
        )

    seeds = {
        namespace: int(seed)
        for namespace, seed in (
            ("role", legacy.role_seed),
            ("actor", legacy.actor_seed),
            ("orchestrator", legacy.orchestrator_seed),
        )
        if seed is not None
    }
    default_model = (
        legacy.default_model.model_dump()
        if legacy.default_model is not None
        else None
    )
    model_overrides = {
        _seat_actor_id(seat): manifest.model_dump()
        for seat, manifest in sorted(legacy.seat_models.items())
    }
    actors = ActorSpec(
        default_model=default_model,
        model_overrides=model_overrides,
        human_actor_ids=[_seat_actor_id(seat) for seat in legacy.human_seats],
    )
    return CoreRunSpec(
        run_id=legacy.run_id,
        environment=EnvironmentRef(
            id=legacy.environment_id,
            version=legacy.environment_version,
        ),
        environment_config={
            "ruleset_id": legacy.ruleset_id,
            "player_names": list(legacy.player_names),
            "role_deck": list(legacy.role_deck),
            "turn_policy": legacy.turn_policy,
            "max_speak_rounds": legacy.max_speak_rounds,
            "decision_timeout_seconds": legacy.decision_timeout_seconds,
            "phase_deadline_seconds": legacy.phase_deadline_seconds,
            "max_consecutive_decision_failures": legacy.max_consecutive_decision_failures,
            "max_consecutive_no_progress_rounds": legacy.max_consecutive_no_progress_rounds,
            "max_game_rounds": legacy.max_game_rounds,
        },
        seeds=seeds,
        execution=ExecutionSpec(
            run_timeout_seconds=legacy.run_timeout_seconds,
            decision_timeout_seconds=legacy.decision_timeout_seconds,
        ),
        actors=actors,
        metadata={
            **legacy.metadata,
            "source_schema_version": legacy.schema_version,
            "legacy_spec_hash": legacy.spec_hash,
        },
    )


def _seat_actor_id(seat: int) -> str:
    return f"seat:{int(seat)}"
