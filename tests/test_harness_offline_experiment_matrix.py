"""Offline matrix proof for scheduling, artifacts, and summary recomputation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.agent.actor import AgentActor
from src.agent.schemas import AgentAction, Decision
from src.harness.agent_protocol import ActionRequest, DecisionEnvelope
from src.harness.artifacts import load_verified_run_summary, verify_run_artifacts
from src.harness.batch import run_experiment_spec
from src.harness.results import RunSummaryRow
from src.harness.spec import ExperimentSpec
from src.harness.summary import summarize_runs
from src.llm.models import ModelConfig


class _Stats:
    @staticmethod
    def snapshot() -> dict[str, float]:
        return {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_latency": 0.0,
            "avg_latency": 0.0,
        }


class _Router:
    stats = _Stats()

    async def aclose(self) -> None:
        return None


async def _offline_decide(self: AgentActor, request: ActionRequest) -> DecisionEnvelope:
    legal = request.legal_actions[0]
    target = legal.target_seats[0] if legal.target_seats else None
    action = request.action_kind
    if action == "speak":
        decision = Decision(
            action=AgentAction.SPEAK,
            speech=f"offline exact speech from seat {self.seat}",
            bid=1,
            reasoning=f"offline private reasoning seat {self.seat}",
        )
    elif action == "vote":
        decision = Decision(
            action=AgentAction.VOTE,
            target_seat=target,
            reasoning="offline deterministic vote",
        )
    elif action == "last_words":
        decision = Decision(
            action=AgentAction.LAST_WORDS,
            speech=f"offline last words from seat {self.seat}",
        )
    elif action == "wolf_council":
        decision = Decision(
            action=AgentAction.WOLF_COUNCIL,
            target_seat=target,
            team_message=f"offline council from seat {self.seat}",
            reasoning="offline private council",
        )
    elif action in {"save", "poison", "hunter_shot"}:
        decision = Decision(action=AgentAction.SKIP, skip_reason="offline_explicit_skip")
    else:
        mapped = {
            "night_kill": AgentAction.NIGHT_KILL,
            "see": AgentAction.SEE,
            "guard": AgentAction.GUARD,
        }[action]
        decision = Decision(
            action=mapped,
            target_seat=target,
            reasoning="offline deterministic night action",
        )
    return DecisionEnvelope(
        request_id=request.request_id,
        seat=self.seat,
        decision=decision,
        model_call_id=f"offline:{request.request_id}",
        prompt_hash="a" * 64,
        response_hash="b" * 64,
        metadata={"agent_kind": "offline-test"},
    )


@pytest.mark.asyncio
async def test_offline_multiseed_cross_play_artifacts_recompute(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(AgentActor, "decide", _offline_decide)
    spec = ExperimentSpec(
        experiment_id="offline-matrix",
        player_names=["A", "B", "C", "D", "E", "F"],
        role_deck=["werewolf", "werewolf", "seer", "villager", "villager", "villager"],
        turn_policies=["fixed_round_robin", "bid_reply"],
        replicates=2,
        base_seed=200,
        policy_order="abba",
        max_speak_rounds=1,
        run_timeout_seconds=20,
        metadata={
            "seat_permutation_mode": "cyclic",
            "role_layout_mode": "fixed",
            "role_layout_seed": 777,
            "persona_mode": "randomized",
            "persona_seed": 888,
        },
    )
    default = ModelConfig(
        provider="openai",
        model="offline-default",
        api_base="https://example.invalid/v1",
        api_key="offline-only-key",
    )
    source_models = {
        1: ModelConfig(provider="openai", model="offline-a", api_key="offline-a-key"),
        2: ModelConfig(provider="openai", model="offline-b", api_key="offline-b-key"),
    }
    artifact_root = tmp_path / "artifacts"
    summary_path = tmp_path / "summary.jsonl"

    batch = await run_experiment_spec(
        spec,
        model_config=default,
        seat_model_configs=source_models,
        router=_Router(),  # type: ignore[arg-type]
        artifact_root=artifact_root,
        summary_jsonl=summary_path,
    )

    assert batch.scheduled_runs == 4
    assert batch.completed_runs == 4
    assert batch.failed_runs == 0
    assert [result.role_seed for result in batch.results] == [777, 777, 777, 777]
    assert [result.actor_seed for result in batch.results] == [100201, 100201, 100202, 100202]
    assert [row.metadata["seat_rotation"] for row in batch.rows] == [0, 0, 1, 1]
    assert (
        batch.rows[0].metadata["persona_assignment_id"]
        == batch.rows[1].metadata["persona_assignment_id"]
    )
    assert (
        batch.rows[2].metadata["persona_assignment_id"]
        == batch.rows[3].metadata["persona_assignment_id"]
    )
    assert batch.rows[0].metadata["persona_assignment_id"] != batch.rows[2].metadata["persona_assignment_id"]

    first_models = batch.results[0].run_spec["seat_models"]
    rotated_models = batch.results[2].run_spec["seat_models"]
    assert first_models[1]["model"] == "offline-a"
    assert first_models[2]["model"] == "offline-b"
    assert rotated_models[1]["model"] == "offline-b"
    assert rotated_models[6]["model"] == "offline-a"

    for paths in batch.artifact_paths.values():
        verify_run_artifacts(paths["run_dir"])

    reloaded = [
        RunSummaryRow.model_validate(json.loads(line))
        for line in summary_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recomputed = summarize_runs(reloaded)
    assert recomputed.evaluation_evidence_run_count == 0
    assert recomputed.cache_only_run_count == 4
    assert recomputed.strategy_evaluation is None
    assert recomputed.operational_evaluation is None
    assert recomputed.deception_evaluation is None
    assert recomputed.comparative_evaluation is None

    verified_rows = [
        load_verified_run_summary(paths["run_dir"])
        for paths in batch.artifact_paths.values()
    ]
    verified_summary = summarize_runs(verified_rows)
    assert verified_summary == batch.summary
    assert verified_summary.evaluation_evidence_run_count == 4
    assert verified_summary.strategy_evaluation is not None
    assert verified_summary.operational_evaluation is not None
    assert verified_summary.deception_evaluation is not None
    assert verified_summary.deception_evaluation.causal is False
    assert verified_summary.operational_evaluation.overall.private_information_leak_count == 0
    assert verified_summary.runs_by_persona_mode == {"randomized": 4}
    assert len(verified_summary.runs_by_role_layout) == 1
    assert next(iter(verified_summary.runs_by_role_layout.values())) == 4

    serialized = json.dumps(
        {
            "rows": [row.model_dump() for row in reloaded],
            "summary": recomputed.model_dump(),
        },
        ensure_ascii=False,
    )
    for secret in ("offline-only-key", "offline-a-key", "offline-b-key"):
        assert secret not in serialized
