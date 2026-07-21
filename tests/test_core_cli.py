"""Generic Core CLI tests without a provider or a persistent process."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

import src.harness.core_cli as core_cli
from src.harness.artifacts import verify_run_artifacts
from src.harness.core_protocol import DecisionEnvelope, validate_decision_envelope
from src.harness.core_runner import EnvironmentRunResult
from src.harness.core_spec import ActorSpec, CoreRunSpec, EnvironmentRef
from src.harness.environment import DecisionContract, EnvironmentDescriptor
from src.harness.model_manifest import ModelConfigManifest
from src.harness.transcript import Transcript
from src.llm.models import ModelConfig


class _CliPlugin:
    descriptor = EnvironmentDescriptor(
        id="test.core-cli",
        version="1",
        capabilities=("multi_agent",),
    )
    decision_contract = DecisionContract(
        envelope_type=DecisionEnvelope,
        validate_envelope=validate_decision_envelope,
    )

    def resolve_config(self, raw_config, _seeds) -> BaseModel:
        return BaseModel.model_validate(raw_config)

    async def create_session(self, _context):  # pragma: no cover - runner is injected.
        raise AssertionError("CLI test must use the injected runner")


def _model_config() -> ModelConfig:
    return ModelConfig(
        provider="openai",
        model="core-cli-test-model",
        api_base="https://example.invalid/v1",
        api_key="test-key-not-written-to-artifacts",
        max_tokens=0,
    )


def _spec() -> CoreRunSpec:
    config = _model_config()
    return CoreRunSpec(
        run_id="core-cli-run",
        environment=EnvironmentRef(id="test.core-cli", version="1"),
        environment_config={},
        actors=ActorSpec(
            default_model=ModelConfigManifest.from_config(config).model_dump(
                mode="json"
            ),
        ),
        metadata={"suite": "core-cli"},
    )


def _write_spec(tmp_path: Path, spec: CoreRunSpec) -> Path:
    path = tmp_path / "core-run.json"
    path.write_text(
        json.dumps(spec.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _result(spec: CoreRunSpec) -> EnvironmentRunResult:
    transcript = Transcript(
        run_id=spec.run_id,
        metadata={"run_spec_hash": spec.spec_hash},
    )
    transcript.append("harness", {
        "type": "run_started",
        "environment_id": spec.environment.id,
        "environment_version": spec.environment.version,
    })
    exported = transcript.export()
    return EnvironmentRunResult(
        run_id=spec.run_id,
        status="completed",
        environment_id=spec.environment.id,
        environment_version=spec.environment.version,
        run_spec_hash=spec.spec_hash,
        elapsed_seconds=0.01,
        outcome={"winner": "alpha"},
        metrics={"model_calls": 3},
        transcript_digest=exported["stable_digest"],
        transcript=exported,
    )


def test_core_cli_rejects_an_untrusted_plugin_reference_shape() -> None:
    with pytest.raises(core_cli.CoreCliError, match="module:attribute"):
        core_cli.parse_args(["--spec", "run.json", "--plugin", "not-a-reference"])


def test_core_cli_requires_an_exact_core_spec(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text('{"schema_version":"werewolf.harness.spec.v3"}', encoding="utf-8")

    with pytest.raises(core_cli.CoreCliError, match="requires exact"):
        core_cli._load_core_spec(path)


@pytest.mark.asyncio
async def test_core_cli_runs_one_exact_spec_writes_an_artifact_and_verifies_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    spec_path = _write_spec(tmp_path, spec)
    args = core_cli.parse_args([
        "--spec",
        str(spec_path),
        "--plugin",
        "trusted.module:Plugin",
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--verify-smoke",
    ])
    seen = {}

    async def fake_runner(run_spec, *, registry, model_config, router):
        seen["run_spec"] = run_spec
        seen["model_config"] = model_config
        assert registry.get("test.core-cli", "1").descriptor == _CliPlugin.descriptor
        assert router is None
        return _result(run_spec)

    monkeypatch.setattr(core_cli, "_load_trusted_plugin", lambda _reference: _CliPlugin())
    monkeypatch.setattr(core_cli, "run_core_llm_environment", fake_runner)
    monkeypatch.setattr(
        core_cli,
        "verify_real_model_smoke_artifacts",
        lambda _run_dir: SimpleNamespace(model_calls=3),
    )

    report = await core_cli.run_core_cli(args, model_config=_model_config())

    assert seen["run_spec"] == spec
    assert seen["model_config"].model == "core-cli-test-model"
    assert report.status == "completed"
    assert report.model_calls == 3
    assert report.smoke_verified is True
    assert verify_run_artifacts(report.artifact_dir).run == spec


@pytest.mark.asyncio
async def test_core_cli_rejects_a_plugin_that_does_not_match_the_spec_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = _write_spec(tmp_path, _spec())
    args = core_cli.parse_args([
        "--spec",
        str(spec_path),
        "--plugin",
        "trusted.module:Plugin",
    ])

    class WrongPlugin(_CliPlugin):
        descriptor = EnvironmentDescriptor(id="test.other", version="1")

    async def must_not_run(*_args, **_kwargs):  # pragma: no cover - assertion path.
        raise AssertionError("mismatched plugin must be rejected before execution")

    monkeypatch.setattr(core_cli, "_load_trusted_plugin", lambda _reference: WrongPlugin())
    monkeypatch.setattr(core_cli, "run_core_llm_environment", must_not_run)

    with pytest.raises(core_cli.CoreCliError, match="does not match the spec environment"):
        await core_cli.run_core_cli(args, model_config=_model_config())


@pytest.mark.asyncio
async def test_core_cli_rejects_missing_runtime_credentials_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = _write_spec(tmp_path, _spec())
    args = core_cli.parse_args([
        "--spec",
        str(spec_path),
        "--plugin",
        "trusted.module:Plugin",
    ])

    async def must_not_run(*_args, **_kwargs):  # pragma: no cover - assertion path.
        raise AssertionError("missing credentials must be rejected before execution")

    monkeypatch.setattr(core_cli, "_load_trusted_plugin", lambda _reference: _CliPlugin())
    monkeypatch.setattr(core_cli, "run_core_llm_environment", must_not_run)

    with pytest.raises(core_cli.CoreCliError, match="WEREWOLF_LLM_MODEL"):
        await core_cli.run_core_cli(
            args,
            model_config=ModelConfig(provider="openai", model="", api_key=""),
        )


@pytest.mark.asyncio
async def test_core_cli_rejects_a_non_core_decision_contract_without_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = _write_spec(tmp_path, _spec())
    args = core_cli.parse_args([
        "--spec",
        str(spec_path),
        "--plugin",
        "trusted.module:Plugin",
    ])

    class LegacyContractPlugin(_CliPlugin):
        decision_contract = DecisionContract(
            envelope_type=object,
            validate_envelope=lambda _envelope, _request: None,
        )

    monkeypatch.setattr(
        core_cli,
        "_load_trusted_plugin",
        lambda _reference: LegacyContractPlugin(),
    )

    with pytest.raises(
        core_cli.CoreCliError,
        match="not compatible with the Core tool decision protocol",
    ):
        await core_cli.run_core_cli(args, model_config=_model_config())


def test_core_cli_main_prints_only_the_safe_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = core_cli.CoreCliReport(
        run_id="core-cli-run",
        environment_id="test.core-cli",
        environment_version="1",
        status="completed",
        artifact_dir="/tmp/core-cli-run",
        transcript_digest="a" * 64,
        model_calls=3,
        smoke_verified=True,
    )

    async def fake_run(_args):
        return report

    monkeypatch.setattr(core_cli, "run_core_cli", fake_run)
    assert core_cli.main([
        "--spec",
        "run.json",
        "--plugin",
        "trusted.module:Plugin",
    ]) == 0

    output = capsys.readouterr().out
    assert json.loads(output) == report.model_dump(mode="json")
    assert "api_key" not in output
    assert '"transcript":' not in output
