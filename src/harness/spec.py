"""Experiment and run specifications for the multi-agent harness."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..game.roles import CLASSIC_RULESET_ID, validate_role_deck, validate_ruleset_id
from .core_spec import ArtifactIntegrity
from .model_manifest import ModelConfigManifest, _safe_api_base

SPEC_SCHEMA_VERSION = "werewolf.harness.spec.v3"
MANIFEST_SCHEMA_VERSION = "agent-harness.manifest.v2"
LEGACY_TRANSCRIPT_INTEGRITY_VERSION = "agent-harness.transcript-integrity.v1"
TurnPolicy = Literal["fixed_round_robin", "bid_reply"]


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class RunSpec(BaseModel):
    """One concrete Werewolf harness run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SPEC_SCHEMA_VERSION] = SPEC_SCHEMA_VERSION
    run_id: str
    environment_id: str = "werewolf.classic"
    environment_version: str = "1"
    ruleset_id: str = CLASSIC_RULESET_ID
    player_names: list[str] = Field(min_length=6, max_length=12)
    role_deck: list[str] = Field(default_factory=list)
    turn_policy: TurnPolicy = "fixed_round_robin"
    role_seed: int | None = None
    actor_seed: int | None = None
    orchestrator_seed: int | None = None
    max_speak_rounds: int = Field(default=3, ge=1, le=20)
    run_timeout_seconds: float | None = Field(default=900.0, gt=0)
    decision_timeout_seconds: float | None = Field(default=None, gt=0)
    phase_deadline_seconds: float | None = Field(default=None, ge=0)
    max_consecutive_decision_failures: int = Field(default=3, ge=1, le=1000)
    max_consecutive_no_progress_rounds: int = Field(default=3, ge=1, le=1000)
    max_game_rounds: int = Field(default=20, ge=1, le=1000)
    human_seats: list[int] = Field(default_factory=list)
    seat_models: dict[int, ModelConfigManifest] = Field(default_factory=dict)
    default_model: ModelConfigManifest | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def _safe_run_id(cls, value: str) -> str:
        return _safe_path_component(value, field_name="run_id")

    @field_validator("human_seats")
    @classmethod
    def _normalize_human_seats(cls, seats: list[int]) -> list[int]:
        return sorted({int(seat) for seat in seats})

    @field_validator("ruleset_id")
    @classmethod
    def _supported_ruleset(cls, value: str) -> str:
        return validate_ruleset_id(value)

    @model_validator(mode="after")
    def _validate_environment_shape(self) -> "RunSpec":
        valid_seats = set(range(1, len(self.player_names) + 1))
        invalid_humans = [seat for seat in self.human_seats if seat not in valid_seats]
        if invalid_humans:
            raise ValueError(f"human_seats outside player range: {invalid_humans}")
        if self.role_deck:
            validate_role_deck(
                self.role_deck,
                player_count=len(self.player_names),
                ruleset_id=self.ruleset_id,
            )
        return self

    @property
    def spec_hash(self) -> str:
        return _canonical_hash(self.model_dump())


class ExperimentSpec(BaseModel):
    """Versioned experiment plan that expands into concrete run specs.

    ``replicates`` is the number of runs scheduled for each turn policy, not
    the total number of runs in the experiment.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[SPEC_SCHEMA_VERSION] = SPEC_SCHEMA_VERSION
    experiment_id: str
    ruleset_id: str = CLASSIC_RULESET_ID
    player_names: list[str] = Field(min_length=6, max_length=12)
    role_deck: list[str] = Field(default_factory=list)
    turn_policies: list[TurnPolicy] = Field(default_factory=lambda: ["fixed_round_robin"])
    replicates: int = Field(
        default=1,
        ge=1,
        description="Number of scheduled runs per turn policy.",
    )
    base_seed: int
    policy_order: Literal["sequential", "abba"] = "sequential"
    human_seats: list[int] = Field(default_factory=list)
    max_speak_rounds: int = Field(default=3, ge=1, le=20)
    run_timeout_seconds: float | None = Field(default=900.0, gt=0)
    decision_timeout_seconds: float | None = Field(default=None, gt=0)
    phase_deadline_seconds: float | None = Field(default=None, ge=0)
    max_consecutive_decision_failures: int = Field(default=3, ge=1, le=1000)
    max_consecutive_no_progress_rounds: int = Field(default=3, ge=1, le=1000)
    max_game_rounds: int = Field(default=20, ge=1, le=1000)
    default_model: ModelConfigManifest | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id")
    @classmethod
    def _safe_experiment_id(cls, value: str) -> str:
        return _safe_path_component(value, field_name="experiment_id")

    @field_validator("turn_policies")
    @classmethod
    def _nonempty_unique_policies(cls, policies: list[TurnPolicy]) -> list[TurnPolicy]:
        cleaned = [str(policy).strip() for policy in policies if str(policy).strip()]
        if not cleaned:
            raise ValueError("turn_policies must not be empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("turn_policies must be unique")
        return cleaned

    @field_validator("human_seats")
    @classmethod
    def _normalize_human_seats(cls, seats: list[int]) -> list[int]:
        return sorted({int(seat) for seat in seats})

    @field_validator("ruleset_id")
    @classmethod
    def _supported_ruleset(cls, value: str) -> str:
        return validate_ruleset_id(value)

    @model_validator(mode="after")
    def _validate_experiment_shape(self) -> "ExperimentSpec":
        valid_seats = set(range(1, len(self.player_names) + 1))
        invalid_humans = [seat for seat in self.human_seats if seat not in valid_seats]
        if invalid_humans:
            raise ValueError(f"human_seats outside player range: {invalid_humans}")
        if self.role_deck:
            validate_role_deck(
                self.role_deck,
                player_count=len(self.player_names),
                ruleset_id=self.ruleset_id,
            )
        return self

    @property
    def spec_hash(self) -> str:
        return _canonical_hash(self.model_dump())

    def expand_runs(self) -> list[RunSpec]:
        """Build deterministic run specs from the canonical policy schedule."""
        # Keep scheduling semantics in one module. The local import avoids
        # pulling the game orchestrator into protocol-only imports of spec.py.
        from .schedule import (
            apply_seat_permutation,
            build_policy_schedule,
            experiment_metadata,
            persona_assignment_metadata,
        )

        seat_permutation = str(self.metadata.get("seat_permutation_mode") or "fixed")
        role_layout_mode = str(self.metadata.get("role_layout_mode") or "legacy")
        persona_mode = str(self.metadata.get("persona_mode") or "legacy")
        schedule = build_policy_schedule(
            self.replicates,
            list(self.turn_policies),
            policy_order=self.policy_order,
            seed=self.base_seed,
            experiment_id=self.experiment_id,
            seat_count=len(self.player_names),
            seat_permutation=seat_permutation,
            role_layout_mode=role_layout_mode,
            role_layout_seed=self.metadata.get("role_layout_seed"),
            role_layout_count=self.metadata.get("role_layout_count"),
            persona_mode=persona_mode,
            persona_seed=self.metadata.get("persona_seed"),
            persona_profiles=self.metadata.get("persona_profile_ids"),
        )
        experiment_spec_hash = self.spec_hash
        runs: list[RunSpec] = []
        for row in schedule:
            run_player_names = apply_seat_permutation(self.player_names, row)
            schedule_metadata = experiment_metadata(row, player_names=run_player_names)
            schedule_metadata.update(persona_assignment_metadata(
                row,
                source_player_names=self.player_names,
                player_names=run_player_names,
            ))
            if seat_permutation != "fixed" or persona_mode != "legacy":
                schedule_metadata["source_player_names"] = list(self.player_names)
            runs.append(RunSpec(
                run_id=str(row["game_id"]),
                ruleset_id=self.ruleset_id,
                player_names=run_player_names,
                role_deck=list(self.role_deck),
                turn_policy=str(row["turn_policy"]),
                role_seed=int(row["role_seed"]),
                actor_seed=int(row["actor_seed"]),
                orchestrator_seed=int(row["orchestrator_seed"]),
                human_seats=list(self.human_seats),
                max_speak_rounds=self.max_speak_rounds,
                run_timeout_seconds=self.run_timeout_seconds,
                decision_timeout_seconds=self.decision_timeout_seconds,
                phase_deadline_seconds=self.phase_deadline_seconds,
                max_consecutive_decision_failures=self.max_consecutive_decision_failures,
                max_consecutive_no_progress_rounds=self.max_consecutive_no_progress_rounds,
                max_game_rounds=self.max_game_rounds,
                default_model=self.default_model,
                metadata={
                    **self.metadata,
                    **schedule_metadata,
                    "experiment_spec_hash": experiment_spec_hash,
                },
            ))
        return runs


class RunManifest(BaseModel):
    """Safe manifest attached to one live room or offline run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[MANIFEST_SCHEMA_VERSION] = MANIFEST_SCHEMA_VERSION
    run: RunSpec
    transcript_schema_version: str = "werewolf.harness.transcript.v1"
    # Optional so pre-extension v2 manifests remain readable. New writers set
    # this marker together with metadata/counts, enabling offline digest
    # reconstruction without the original live result object.
    transcript_integrity_version: Literal[LEGACY_TRANSCRIPT_INTEGRITY_VERSION] | None = None
    transcript_metadata: dict[str, Any] = Field(default_factory=dict)
    transcript_counts_by_kind: dict[str, int] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    artifact_integrity: dict[str, ArtifactIntegrity] = Field(default_factory=dict)
    transcript_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def manifest_hash(self) -> str:
        return _canonical_hash(self.model_dump())


def _safe_path_component(value: str, *, field_name: str) -> str:
    component = str(value).strip()
    if not component or component in {".", ".."}:
        raise ValueError(f"{field_name} must not be empty or dot-only")
    if len(component) > 200:
        raise ValueError(f"{field_name} must be at most 200 characters")
    if any(character in component for character in ("/", "\\", "\x00")):
        raise ValueError(f"{field_name} must be a single safe path component")
    return component
