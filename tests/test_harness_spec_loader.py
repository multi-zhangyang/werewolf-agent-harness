"""Exact schema dispatch and Werewolf v3 to Core v1 migration tests."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.harness.core_spec import ActorSpec, CoreRunSpec
from src.harness.spec import ModelConfigManifest, RunSpec
from src.harness.spec_loader import (
    RunSpecLoadError,
    legacy_werewolf_run_to_core,
    load_core_run_spec,
)


def _legacy_spec(**updates) -> RunSpec:
    spec = RunSpec(
        run_id="legacy-migration",
        player_names=["A", "B", "C", "D", "E", "F"],
        role_deck=[
            "werewolf",
            "werewolf",
            "seer",
            "villager",
            "villager",
            "villager",
        ],
        role_seed=11,
        actor_seed=22,
        orchestrator_seed=33,
        max_speak_rounds=2,
        run_timeout_seconds=120,
        decision_timeout_seconds=30,
        phase_deadline_seconds=60,
        default_model=ModelConfigManifest(
            provider="openai",
            model="model-default",
            api_base="https://gateway.example.invalid/v1",
            configured=True,
        ),
        seat_models={
            2: ModelConfigManifest(
                provider="openai_responses",
                model="model-seat-2",
                api_base="https://gateway.example.invalid/v1",
                configured=True,
            ),
        },
        metadata={"experiment_id": "migration-test"},
    )
    return spec.model_copy(update=updates) if updates else spec


def test_legacy_migration_preserves_execution_and_records_distinct_hash_provenance():
    legacy = _legacy_spec()

    core = legacy_werewolf_run_to_core(legacy)

    assert core.environment.model_dump() == {"id": "werewolf.classic", "version": "1"}
    assert core.environment_config == {
        "ruleset_id": "classic.v1",
        "player_names": ["A", "B", "C", "D", "E", "F"],
        "role_deck": [
            "werewolf",
            "werewolf",
            "seer",
            "villager",
            "villager",
            "villager",
        ],
        "turn_policy": "fixed_round_robin",
        "max_speak_rounds": 2,
        "decision_timeout_seconds": 30.0,
        "phase_deadline_seconds": 60.0,
        "max_consecutive_decision_failures": 3,
        "max_consecutive_no_progress_rounds": 3,
        "max_game_rounds": 20,
    }
    assert core.seeds == {"role": 11, "actor": 22, "orchestrator": 33}
    assert core.execution.run_timeout_seconds == 120
    assert core.actors.default_model["model"] == "model-default"
    assert core.actors.model_overrides["seat:2"]["model"] == "model-seat-2"
    assert core.metadata["source_schema_version"] == "werewolf.harness.spec.v3"
    assert core.metadata["legacy_spec_hash"] == legacy.spec_hash
    assert core.spec_hash != legacy.spec_hash
    serialized = json.dumps(core.model_dump())
    assert "api_key" not in serialized


def test_loader_dispatches_only_exact_legacy_and_core_schema_versions():
    legacy = _legacy_spec()
    migrated = load_core_run_spec(legacy.model_dump())
    loaded_core = load_core_run_spec(migrated.model_dump())

    assert migrated == loaded_core
    assert load_core_run_spec(legacy) == migrated
    assert load_core_run_spec(migrated) == migrated

    for invalid in (
        {**legacy.model_dump(), "schema_version": "werewolf.harness.spec.v999"},
        {key: value for key, value in legacy.model_dump().items() if key != "schema_version"},
    ):
        with pytest.raises(RunSpecLoadError, match="unsupported RunSpec schema_version"):
            load_core_run_spec(invalid)


def test_run_spec_models_reject_wrong_schema_version_directly():
    legacy = _legacy_spec().model_dump()
    legacy["schema_version"] = "werewolf.harness.spec.v2"
    with pytest.raises(ValidationError, match="schema_version"):
        RunSpec.model_validate(legacy)

    core = legacy_werewolf_run_to_core(_legacy_spec()).model_dump()
    core["schema_version"] = "agent-harness.run-spec.v2"
    with pytest.raises(ValidationError, match="schema_version"):
        CoreRunSpec.model_validate(core)


def test_migration_maps_human_actor_ids_and_rejects_credential_bearing_metadata():
    human = _legacy_spec(seat_models={}, human_seats=[1, 3])
    migrated = legacy_werewolf_run_to_core(human)
    assert migrated.actors.human_actor_ids == ["seat:1", "seat:3"]

    leaked = _legacy_spec(metadata={"note": "Bearer must-not-enter-core-spec"})
    with pytest.raises(ValidationError, match="credentials are forbidden"):
        legacy_werewolf_run_to_core(leaked)


def test_actor_spec_rejects_human_model_overlap_and_credential_values():
    with pytest.raises(ValidationError, match="cannot have model overrides"):
        ActorSpec(
            human_actor_ids=["seat:1"],
            model_overrides={"seat:1": {"provider": "openai", "model": "m"}},
        )

    with pytest.raises(ValidationError, match="credentials are forbidden"):
        CoreRunSpec(
            run_id="actor-secret",
            environment={"id": "counter", "version": "1"},
            actors={
                "default_model": {
                    "provider": "openai",
                    "model": "m",
                    "authorization": "Bearer must-not-enter-core-spec",
                },
            },
        )
