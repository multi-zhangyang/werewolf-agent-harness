"""Canonical environment-neutral run specification."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CORE_RUN_SPEC_VERSION = "agent-harness.run-spec.v1"
CORE_RUN_MANIFEST_VERSION = "agent-harness.core-manifest.v1"
_CREDENTIAL_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "access_token",
    "auth_token",
    "admin_token",
    "bearer_token",
    "client_secret",
    "credential",
    "credentials",
    "private_key",
    "refresh_token",
    "seat_token",
    "seat_tokens",
    "secret",
    "token",
}
_SECRET_VALUE_RE = re.compile(r"(?i)(?:\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]{8,})")


class EnvironmentRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)

    @field_validator("id", "version")
    @classmethod
    def _strip_identity(cls, value: str) -> str:
        identity = str(value).strip()
        if not identity:
            raise ValueError("environment id/version must not be empty")
        return identity


class ExecutionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_timeout_seconds: float | None = Field(default=900.0, gt=0)
    decision_timeout_seconds: float | None = Field(default=None, gt=0)
    # A deadline bounds useful execution. These two limits separately bound
    # cooperative task cancellation and environment-owned resource cleanup.
    # A coroutine that still ignores cancellation after the grace period is
    # reported as a fatal cleanup failure; an in-process runner cannot kill it.
    cancellation_grace_seconds: float = Field(default=1.0, ge=0, le=60)
    cleanup_timeout_seconds: float = Field(default=5.0, gt=0, le=300)


class ArtifactIntegrity(BaseModel):
    """Content identity for one file committed by an artifact manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class ActorSpec(BaseModel):
    """Credential-free model and human bindings keyed by environment actor ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_model: dict[str, Any] | None = None
    model_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    human_actor_ids: list[str] = Field(default_factory=list)

    @field_validator("model_overrides", mode="before")
    @classmethod
    def _normalize_model_override_ids(
        cls,
        value: Any,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(value, dict):
            raise ValueError("model_overrides must be an object")
        normalized: dict[str, dict[str, Any]] = {}
        for raw_actor_id, manifest in value.items():
            actor_id = _actor_id(raw_actor_id)
            if not isinstance(manifest, dict):
                raise ValueError(f"model override for {actor_id!r} must be an object")
            if actor_id in normalized:
                raise ValueError(f"duplicate model override actor_id: {actor_id}")
            normalized[actor_id] = dict(manifest)
        return normalized

    @field_validator("human_actor_ids", mode="before")
    @classmethod
    def _normalize_human_actor_ids(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("human_actor_ids must be a list")
        normalized = [_actor_id(actor_id) for actor_id in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("human_actor_ids must be unique")
        return sorted(normalized)

    @model_validator(mode="after")
    def _human_and_model_bindings_do_not_overlap(self) -> "ActorSpec":
        overlap = sorted(set(self.human_actor_ids) & set(self.model_overrides))
        if overlap:
            raise ValueError("human actors cannot have model overrides: " + ",".join(overlap))
        return self


class CoreRunSpec(BaseModel):
    """Generic spec; environment-specific fields live in environment_config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[CORE_RUN_SPEC_VERSION] = CORE_RUN_SPEC_VERSION
    run_id: str = Field(min_length=1)
    environment: EnvironmentRef
    environment_config: dict[str, Any] = Field(default_factory=dict)
    seeds: dict[str, int] = Field(default_factory=dict)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    actors: ActorSpec = Field(default_factory=ActorSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def _strip_run_id(cls, value: str) -> str:
        run_id = str(value).strip()
        if not run_id:
            raise ValueError("run_id must not be empty")
        if run_id in {".", ".."} or len(run_id) > 200 or any(
            character in run_id for character in ("/", "\\", "\x00")
        ):
            raise ValueError("run_id must be a single safe path component")
        return run_id

    @field_validator("seeds", mode="before")
    @classmethod
    def _strict_namespaced_seeds(cls, value: Any) -> dict[str, int]:
        if not isinstance(value, dict):
            raise ValueError("seeds must be an object")
        normalized: dict[str, int] = {}
        for raw_name, seed in value.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("seed namespace must not be empty")
            if type(seed) is not int:
                raise ValueError(f"seed {name!r} must be an integer")
            normalized[name] = seed
        return normalized

    @model_validator(mode="after")
    def _credentials_do_not_enter_spec(self) -> "CoreRunSpec":
        leaked = _credential_paths({
            "environment_config": self.environment_config,
            "actors": self.actors.model_dump(),
            "metadata": self.metadata,
        })
        if leaked:
            raise ValueError("credentials are forbidden in CoreRunSpec: " + ",".join(leaked))
        return self

    @property
    def spec_hash(self) -> str:
        return _canonical_hash(self.model_dump())


class CoreRunManifest(BaseModel):
    """Environment-neutral commit marker for one generic harness run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[CORE_RUN_MANIFEST_VERSION] = CORE_RUN_MANIFEST_VERSION
    run: CoreRunSpec
    run_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_schema_version: str = Field(min_length=1)
    transcript_schema_version: str = Field(min_length=1)
    transcript_metadata: dict[str, Any] = Field(default_factory=dict)
    transcript_counts_by_kind: dict[str, int] = Field(default_factory=dict)
    artifact_paths: dict[str, str]
    artifact_integrity: dict[str, ArtifactIntegrity]
    transcript_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _run_hash_matches_embedded_spec(self) -> "CoreRunManifest":
        if self.run_spec_hash != self.run.spec_hash:
            raise ValueError("run_spec_hash does not match embedded CoreRunSpec")
        return self

    @property
    def manifest_hash(self) -> str:
        return _canonical_hash(self.model_dump())


def _credential_paths(value: Any, path: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key)
            current = f"{path}.{key}" if path else key
            normalized = key.lower().replace("-", "_")
            if normalized in _CREDENTIAL_KEYS:
                paths.append(current)
            paths.extend(_credential_paths(item, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_credential_paths(item, f"{path}[{index}]"))
    elif isinstance(value, str) and _looks_like_credential(value):
        paths.append(path or "<root>")
    return paths


def _actor_id(value: Any) -> str:
    actor_id = str(value).strip()
    if not actor_id or len(actor_id) > 200 or "\x00" in actor_id:
        raise ValueError("actor_id must be non-empty and at most 200 characters")
    return actor_id


def _looks_like_credential(value: str) -> bool:
    if _SECRET_VALUE_RE.search(value):
        return True
    if "://" not in value:
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if parsed.username is not None or parsed.password is not None:
        return True
    return any(
        key.lower().replace("-", "_") in _CREDENTIAL_KEYS
        for key, _item in parse_qsl(parsed.query, keep_blank_values=True)
    )


def _canonical_hash(value: Any) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()
