"""Sequential batch runner for real harness experiments."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..config import LLM_CONCURRENCY, LLM_MAX_RETRIES, LLM_TIMEOUT
from ..llm.models import ModelConfig
from ..llm.router import LLMRouter
from .artifacts import load_verified_run_summary, write_run_artifacts
from .results import RunSummaryRow, run_summary_from_result
from .runner import HarnessRunResult, resolve_run_spec, run_werewolf_run
from .schedule import apply_seat_mapping_permutation
from .spec import ExperimentSpec, RunSpec
from .summary import ExperimentSummary, summarize_runs


class HarnessBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    scheduled_runs: int
    completed_runs: int
    failed_runs: int
    results: list[HarnessRunResult] = Field(default_factory=list)
    rows: list[RunSummaryRow] = Field(default_factory=list)
    summary: ExperimentSummary
    artifact_paths: dict[str, dict[str, str]] = Field(default_factory=dict)
    resumed_run_ids: list[str] = Field(default_factory=list)


async def run_experiment_spec(
    spec: ExperimentSpec,
    *,
    model_config: ModelConfig,
    seat_model_configs: dict[int, ModelConfig | dict[str, Any]] | None = None,
    router: LLMRouter | None = None,
    artifact_root: str | Path | None = None,
    summary_jsonl: str | Path | None = None,
    resume_jsonl: bool = False,
) -> HarnessBatchResult:
    """Run every scheduled spec with real model-backed agents."""
    scheduled_runs = [
        (
            run_spec,
            apply_seat_mapping_permutation(seat_model_configs, run_spec.metadata),
        )
        for run_spec in spec.expand_runs()
    ]
    resolved_runs = [
        (
            resolve_run_spec(
                run_spec,
                model_config=model_config,
                seat_model_configs=permuted_configs,
            ),
            permuted_configs,
        )
        for run_spec, permuted_configs in scheduled_runs
    ]
    run_specs = [run_spec for run_spec, _configs in resolved_runs]
    resumed = (
        _load_resume_rows(
            summary_jsonl,
            run_specs,
            artifact_root=artifact_root,
        )
        if resume_jsonl
        else {}
    )
    rows: list[RunSummaryRow] = list(resumed.values())
    results: list[HarnessRunResult] = []
    artifact_paths: dict[str, dict[str, str]] = {}
    owned_router = router is None
    active_router = router or LLMRouter(
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
        concurrency=LLM_CONCURRENCY,
    )
    jsonl_path = Path(summary_jsonl) if summary_jsonl else None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume_jsonl:
            jsonl_path.write_text("", encoding="utf-8")

    try:
        for run_spec, run_seat_configs in resolved_runs:
            if run_spec.run_id in resumed:
                continue
            result = await run_werewolf_run(
                run_spec,
                model_config=model_config,
                seat_model_configs=run_seat_configs,
                router=active_router,
                close_router=False,
            )
            if result.run_spec_hash != run_spec.spec_hash:
                raise RuntimeError(
                    f"runner returned a mismatched run spec hash for {run_spec.run_id}"
                )
            row = run_summary_from_result(result)
            results.append(result)
            rows.append(row)
            if artifact_root is not None:
                artifact_paths[result.run_id] = write_run_artifacts(result, run_spec, artifact_root)
            if jsonl_path is not None:
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    # Optional evidence stays absent rather than serializing null;
                    # legacy v3 rows remain readable alongside new v4 rows.
                    handle.write(
                        json.dumps(row.model_dump(exclude_none=True), ensure_ascii=False, default=str)
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
    finally:
        if owned_router:
            await active_router.aclose()

    rows.sort(key=lambda row: row.run_id)
    summary = summarize_runs(rows)
    return HarnessBatchResult(
        experiment_id=spec.experiment_id,
        scheduled_runs=len(run_specs),
        completed_runs=summary.completed_runs,
        failed_runs=summary.failed_runs,
        results=results,
        rows=rows,
        summary=summary,
        artifact_paths=artifact_paths,
        resumed_run_ids=sorted(resumed),
    )


def run_experiment_spec_sync(spec: ExperimentSpec, **kwargs: Any) -> HarnessBatchResult:
    return asyncio.run(run_experiment_spec(spec, **kwargs))


def _load_resume_rows(
    path: str | Path | None,
    run_specs: list[RunSpec],
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, RunSummaryRow]:
    if path is None or not Path(path).exists():
        return {}
    scheduled = {run_spec.run_id: run_spec for run_spec in run_specs}
    rows: dict[str, RunSummaryRow] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                row = RunSummaryRow(**data)
            except (json.JSONDecodeError, ValueError, TypeError) as err:
                raise ValueError(
                    f"invalid resume row at {Path(path)}:{line_number}"
                ) from err
            expected = scheduled.get(row.run_id)
            if expected is None:
                continue
            if row.run_spec_hash != expected.spec_hash:
                raise ValueError(
                    f"resume run_spec_hash mismatch for {row.run_id}: "
                    f"expected {expected.spec_hash}, got {row.run_spec_hash}"
                )
            previous = rows.get(row.run_id)
            if (
                previous is not None
                and previous.model_dump(mode="json") != row.model_dump(mode="json")
            ):
                raise ValueError(f"conflicting duplicate resume rows for {row.run_id}")
            rows[row.run_id] = row
    if artifact_root is not None:
        root = Path(artifact_root)
        for run_id, cached in list(rows.items()):
            run_dir = root / run_id
            if not run_dir.exists():
                continue
            try:
                verified = load_verified_run_summary(run_dir)
            except (OSError, ValueError, TypeError) as err:
                raise ValueError(
                    f"invalid resume artifact for {run_id}: {run_dir}"
                ) from err
            if verified.model_dump(mode="json") != cached.model_dump(mode="json"):
                raise ValueError(
                    f"resume summary row does not match verified artifact for {run_id}"
                )
            rows[run_id] = verified
    return rows
