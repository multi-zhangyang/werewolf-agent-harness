"""Run one trusted generic Core environment with real tool-using Actors.

The legacy ``src.harness.cli`` command owns the historical Werewolf batch
adapter. This command consumes one exact ``CoreRunSpec`` instead: its plugin
is explicitly named, its model binding is credential-free in the spec, and
runtime credentials remain in the normal ``WEREWOLF_LLM_*`` configuration.
It prints only a compact safe report rather than a transcript or model output.
"""
from __future__ import annotations

import argparse
import asyncio
from importlib import import_module
import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..config import DEFAULT_MODEL_CONFIG
from ..llm.models import ModelConfig
from ..llm.router import STANDARD_PROTOCOLS
from .artifacts import write_run_artifacts
from .core_llm_runner import run_core_llm_environment
from .core_spec import CORE_RUN_SPEC_VERSION, CoreRunSpec
from .environment import EnvironmentPlugin
from .registry import EnvironmentRegistry, EnvironmentRegistryError
from .smoke import verify_real_model_smoke_artifacts


CORE_CLI_REPORT_VERSION = "agent-harness.core-cli-report.v1"


class CoreCliError(ValueError):
    """Raised for invalid local Core CLI input before an environment runs."""


class CoreCliReport(BaseModel):
    """Compact, credential-free outcome of one generic CLI invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = CORE_CLI_REPORT_VERSION
    run_id: str = Field(min_length=1)
    environment_id: str = Field(min_length=1)
    environment_version: str = Field(min_length=1)
    status: str = Field(min_length=1)
    termination_reason: str | None = None
    artifact_dir: str = Field(min_length=1)
    transcript_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_calls: int = Field(ge=0)
    smoke_verified: bool = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.harness.core_cli",
        description=(
            "Run one exact CoreRunSpec through a trusted environment plugin "
            "with real tool-using model Actors."
        ),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path to one credential-free agent-harness.run-spec.v1 JSON document.",
    )
    parser.add_argument(
        "--plugin",
        required=True,
        help=(
            "Trusted local Python reference module:attribute for the exact "
            "environment plugin class or instance."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts"),
        help="Directory in which the three-file committed run artifact is written.",
    )
    parser.add_argument(
        "--verify-smoke",
        action="store_true",
        help="Require the credential-free real-model smoke verifier to pass after writing.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    _parse_plugin_reference(args.plugin)
    return args


async def run_core_cli(
    args: argparse.Namespace,
    *,
    model_config: ModelConfig | None = None,
    router: Any | None = None,
) -> CoreCliReport:
    """Execute parsed Core CLI arguments without printing provider content.

    ``model_config`` and ``router`` are injectable for integration tests. The
    normal command path builds the former from the process-scoped safe config
    and lets ``run_core_llm_environment`` own its Router lifecycle.
    """
    spec = _load_core_spec(args.spec)
    registry = EnvironmentRegistry()
    plugin = _load_trusted_plugin(args.plugin)
    try:
        descriptor = registry.register(plugin)
    except EnvironmentRegistryError as err:
        raise CoreCliError("trusted plugin is not a valid environment plugin") from err
    if (
        descriptor.id != spec.environment.id
        or descriptor.version != spec.environment.version
    ):
        raise CoreCliError(
            "trusted plugin does not match the spec environment: "
            f"expected={spec.environment.id}@{spec.environment.version} "
            f"actual={descriptor.id}@{descriptor.version}"
        )

    active_model_config = model_config or ModelConfig(**DEFAULT_MODEL_CONFIG)
    _require_real_model_config(active_model_config)
    try:
        result = await run_core_llm_environment(
            spec,
            registry=registry,
            model_config=active_model_config,
            router=router,
        )
    except ValueError as err:
        raise CoreCliError(str(err)) from err
    paths = write_run_artifacts(result, spec, args.artifact_root)

    model_calls = _nonnegative_model_calls(result.metrics.get("model_calls"))
    smoke_verified = False
    if args.verify_smoke:
        smoke = verify_real_model_smoke_artifacts(paths["run_dir"])
        model_calls = smoke.model_calls
        smoke_verified = True

    return CoreCliReport(
        run_id=result.run_id,
        environment_id=result.environment_id,
        environment_version=result.environment_version,
        status=result.status,
        termination_reason=result.termination_reason,
        artifact_dir=paths["run_dir"],
        transcript_digest=result.transcript_digest,
        model_calls=model_calls,
        smoke_verified=smoke_verified,
    )


def _load_core_spec(path: Path) -> CoreRunSpec:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as err:
        raise CoreCliError(f"could not read CoreRunSpec: {path}") from err
    if not isinstance(raw, dict):
        raise CoreCliError("CoreRunSpec JSON must be an object")
    if raw.get("schema_version") != CORE_RUN_SPEC_VERSION:
        raise CoreCliError(
            "Core CLI requires exact "
            f"{CORE_RUN_SPEC_VERSION}; use the legacy harness CLI for legacy specs"
        )
    try:
        return CoreRunSpec.model_validate(raw)
    except ValueError as err:
        raise CoreCliError("CoreRunSpec is invalid") from err


def _load_trusted_plugin(reference: str) -> EnvironmentPlugin:
    module_name, attribute = _parse_plugin_reference(reference)
    try:
        module = import_module(module_name)
        candidate = getattr(module, attribute)
    except (ImportError, AttributeError) as err:
        raise CoreCliError(f"could not load trusted plugin: {reference}") from err
    plugin = candidate() if isinstance(candidate, type) else candidate
    return plugin  # The registry validates the structural plugin contract.


def _parse_plugin_reference(reference: str) -> tuple[str, str]:
    raw = str(reference).strip()
    module_name, separator, attribute = raw.partition(":")
    module_parts = module_name.split(".") if module_name else []
    if (
        separator != ":"
        or not module_parts
        or any(not part.isidentifier() for part in module_parts)
        or not attribute.isidentifier()
        or ":" in attribute
    ):
        raise CoreCliError(
            "--plugin must be a trusted module:attribute reference"
        )
    return module_name, attribute


def _require_real_model_config(config: ModelConfig) -> None:
    missing = [
        field
        for field, value in (
            ("WEREWOLF_LLM_PROVIDER", config.provider in STANDARD_PROTOCOLS),
            ("WEREWOLF_LLM_MODEL", config.model.strip()),
            ("WEREWOLF_LLM_API_KEY", config.api_key.strip()),
        )
        if not value
    ]
    if missing:
        raise CoreCliError("missing real model configuration: " + ", ".join(missing))


def _nonnegative_model_calls(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CoreCliError("generic run result has an invalid model_calls metric")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = asyncio.run(run_core_cli(parse_args(argv)))
    except CoreCliError as err:
        raise SystemExit(str(err)) from err
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    return 0 if report.status == "completed" else 1


if __name__ == "__main__":  # pragma: no cover - module entry point.
    raise SystemExit(main())
