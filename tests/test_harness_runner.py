"""Production harness runner tests with a test-local AgentProtocol double."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.agent.memory import AgentMemory
from src.agent.schemas import AgentAction, Decision
from src.game.roles import CLASSIC_RULESET_ID, Role, default_role_deck
from src.harness.agent_protocol import ActionRequest, DecisionEnvelope
from src.harness.artifacts import verify_run_artifacts, write_run_artifacts
from src.harness.runner import resolve_run_spec, run_werewolf_run
from src.harness.spec import ExperimentSpec, RunSpec
from src.llm.models import ModelConfig


@dataclass
class _Stats:
    def snapshot(self) -> dict[str, Any]:
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


class _ProtocolActor:
    """Deterministic test double; never imported by production code."""

    def __init__(self, player, state, model_config: ModelConfig) -> None:
        self.seat = player.seat
        self.name = player.name
        self.role = player.role
        self.state = state
        self.model_config = model_config
        self.is_human = False
        self.memory = AgentMemory(seat=player.seat, role=player.role.value)
        self.persona_name = "test"
        self.on_human_request = None

    async def decide(self, request: ActionRequest) -> DecisionEnvelope:
        legal = request.legal_actions[0]
        action = request.action_kind
        target_seat: int | None = None
        if action == "vote":
            candidates = [p for p in self.state.living_players() if p.seat in legal.target_seats]
            preferred = (
                [p for p in candidates if p.role != Role.WEREWOLF]
                if self.role == Role.WEREWOLF
                else [p for p in candidates if p.role == Role.WEREWOLF]
            )
            selected = (preferred or candidates)[0]
            target_seat = selected.seat
            decision = Decision(action=AgentAction.VOTE, target_seat=selected.seat, reasoning="test")
        elif action == "speak":
            decision = Decision(
                action=AgentAction.SPEAK,
                speech=f"exact output from seat {self.seat}",
                bid=2,
                reasoning="private test reasoning",
            )
        elif action == "last_words":
            decision = Decision(action=AgentAction.LAST_WORDS, speech="test last words")
        elif action == "wolf_council":
            decision = Decision(
                action=AgentAction.WOLF_COUNCIL,
                target_seat=legal.target_seats[0],
                team_message=f"test council from seat {self.seat}",
                reasoning="private test council reasoning",
            )
        elif action in {"save", "poison", "hunter_shot"}:
            decision = Decision(action=AgentAction.SKIP, skip_reason="test_skip")
        else:
            target_seat = legal.target_seats[0]
            mapped = {
                "night_kill": AgentAction.NIGHT_KILL,
                "see": AgentAction.SEE,
                "guard": AgentAction.GUARD,
            }[action]
            decision = Decision(action=mapped, target_seat=target_seat, reasoning="test")
        return DecisionEnvelope(
            request_id=request.request_id,
            seat=self.seat,
            decision=decision,
            model_call_id=f"test-call-{request.request_id}",
            prompt_hash="a" * 64,
            response_hash="b" * 64,
            metadata={"agent_kind": "test"},
        )

    def observe_event(self, day: int, phase: str, kind: str, text: str, **meta: Any) -> None:
        self.memory.observe(day, phase, kind, text, **meta)

    def record_claim(self, seat: int, day: int, claim: dict[str, Any]) -> None:
        self.memory.record_claim(seat, day, claim)


class _FailingProtocolActor(_ProtocolActor):
    async def decide(self, request: ActionRequest) -> DecisionEnvelope:
        raise RuntimeError("private provider outage detail")


def _spec(run_id: str = "runner-test") -> RunSpec:
    return RunSpec(
        run_id=run_id,
        player_names=["A", "B", "C", "D", "E", "F"],
        role_deck=["werewolf", "werewolf", "seer", "villager", "villager", "villager"],
        role_seed=11,
        actor_seed=22,
        orchestrator_seed=33,
        max_speak_rounds=1,
        run_timeout_seconds=20,
    )


@pytest.mark.asyncio
async def test_runner_uses_real_protocol_boundary_and_emits_factual_result(monkeypatch):
    def build_test_actors(state, **kwargs):
        model_config = kwargs["model_config"]
        seat_configs = kwargs.get("seat_configs") or {}
        return {
            player.id: _ProtocolActor(
                player,
                state,
                model_config.merge(seat_configs.get(player.seat)),
            )
            for player in state.players
        }

    monkeypatch.setattr("src.harness.runner.build_actors", build_test_actors)
    result = await run_werewolf_run(
        _spec(),
        model_config=ModelConfig(
            provider="openai",
            model="test-model",
            api_base="https://example.invalid/v1",
            api_key="test-only-key",
        ),
        router=_Router(),  # type: ignore[arg-type]
        close_router=False,
    )

    assert result.status == "completed"
    assert result.run_spec["ruleset_id"] == CLASSIC_RULESET_ID
    assert result.winner in {"village", "werewolves"}
    assert result.analysis is not None
    assert result.harness_metrics["resolved_actor_count"] == 6
    assert result.harness_metrics["resolved_actor_ids"] == [
        f"seat:{seat}" for seat in range(1, 7)
    ]
    assert set(result.analysis) == {
        "winner",
        "days",
        "turn_policy",
        "seats",
        "decision_count",
        "decision_trace_metrics",
        "parse_metrics",
        "decision_failure_metrics",
        "agent_strategy_metrics",
    }
    assert not hasattr(result, "social_spec")
    assert not hasattr(result, "interaction_graph")
    assert not hasattr(result, "replay_capability")
    decision_rows = [
        entry["payload"] for entry in result.transcript["entries"]
        if entry["kind"] == "decision"
    ]
    assert any(row.get("kind") == "agent_request" for row in decision_rows)
    assert any(row.get("kind") == "agent_response" for row in decision_rows)
    binding_rows = [
        entry["payload"]
        for entry in result.transcript["entries"]
        if entry["kind"] == "harness"
        and entry["payload"].get("type") == "agent_bindings_finalized"
    ]
    assert binding_rows == [{
        "type": "agent_bindings_finalized",
        "actor_count": 6,
        "actor_ids": [f"seat:{seat}" for seat in range(1, 7)],
    }]
    assert (
        result.transcript["metadata"]["caller_metadata"]["legacy_spec_hash"]
        == result.run_spec_hash
    )
    assert result.transcript["metadata"]["run_spec_hash"] != result.run_spec_hash
    request_ids = {
        row["request"]["request_id"]
        for row in decision_rows
        if row.get("kind") == "agent_request"
    }
    terminal_ids = [
        row["request_id"]
        for row in decision_rows
        if row.get("kind") in {"agent_response", "agent_response_failed"}
    ]
    assert len(terminal_ids) == len(request_ids)
    assert set(terminal_ids) == request_ids
    assert all(
        row.get("request_id") in request_ids
        for row in decision_rows
        if row.get("type") in {"decision_consumed", "rules_result"}
    )
    speech_events = [
        entry["payload"] for entry in result.transcript["entries"]
        if entry["kind"] == "event" and entry["payload"].get("type") == "speech"
    ]
    assert speech_events
    assert all(event["text"].startswith("exact output from seat") for event in speech_events)
    serialized = str(result.model_dump())
    assert "test-only-key" not in serialized


@pytest.mark.asyncio
async def test_runner_and_artifact_report_incomplete_failure_guard(monkeypatch, tmp_path):
    def build_failing_actors(state, **kwargs):
        model_config = kwargs["model_config"]
        seat_configs = kwargs.get("seat_configs") or {}
        return {
            player.id: _FailingProtocolActor(
                player,
                state,
                model_config.merge(seat_configs.get(player.seat)),
            )
            for player in state.players
        }

    monkeypatch.setattr("src.harness.runner.build_actors", build_failing_actors)
    config = ModelConfig(
        provider="openai",
        model="test-model",
        api_base="https://example.invalid/v1",
        api_key="test-only-key",
    )
    unresolved = _spec("failure-guard-artifact").model_copy(update={
        "max_consecutive_decision_failures": 2,
        "max_consecutive_no_progress_rounds": 99,
        "max_game_rounds": 99,
    })
    spec = resolve_run_spec(unresolved, model_config=config)

    result = await run_werewolf_run(
        spec,
        model_config=config,
        router=_Router(),  # type: ignore[arg-type]
        close_router=False,
    )

    assert result.status == "incomplete"
    assert result.termination_reason == "consecutive_decision_failures"
    assert result.winner is None
    assert result.error_type is None
    assert result.error is None
    assert result.analysis is not None
    assert result.analysis["termination"]["reason"] == "consecutive_decision_failures"

    decision_rows = [
        entry["payload"] for entry in result.transcript["entries"]
        if entry["kind"] == "decision"
    ]
    request_ids = {
        row["request"]["request_id"]
        for row in decision_rows
        if row.get("kind") == "agent_request"
    }
    terminal_ids = [
        row["request_id"]
        for row in decision_rows
        if row.get("kind") in {
            "agent_response",
            "agent_response_failed",
            "agent_response_cancelled",
            "agent_response_validation_failed",
        }
    ]
    assert len(terminal_ids) == len(request_ids)
    assert set(terminal_ids) == request_ids
    harness_rows = [
        entry["payload"] for entry in result.transcript["entries"]
        if entry["kind"] == "harness"
    ]
    assert any(row.get("type") == "run_incomplete" for row in harness_rows)
    assert not any(row.get("type") == "run_failed" for row in harness_rows)

    paths = write_run_artifacts(result, spec, tmp_path)
    verify_run_artifacts(paths["run_dir"])
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["status"] == "incomplete"
    assert summary["termination_reason"] == "consecutive_decision_failures"
    assert summary["winner"] is None
    assert "private provider outage detail" not in json.dumps(
        result.model_dump(), ensure_ascii=False, default=str
    )


@pytest.mark.asyncio
async def test_same_local_seeds_produce_stable_request_ids_and_transcript(monkeypatch):
    def build_test_actors(state, **kwargs):
        model_config = kwargs["model_config"]
        seat_configs = kwargs.get("seat_configs") or {}
        return {
            player.id: _ProtocolActor(
                player,
                state,
                model_config.merge(seat_configs.get(player.seat)),
            )
            for player in state.players
        }

    monkeypatch.setattr("src.harness.runner.build_actors", build_test_actors)
    config = ModelConfig(
        provider="openai",
        model="test-model",
        api_base="https://example.invalid/v1",
        api_key="test-only-key",
    )

    first = await run_werewolf_run(
        _spec("stable-local-run"),
        model_config=config,
        router=_Router(),  # type: ignore[arg-type]
        close_router=False,
    )
    second = await run_werewolf_run(
        _spec("stable-local-run"),
        model_config=config,
        router=_Router(),  # type: ignore[arg-type]
        close_router=False,
    )

    def request_ids(result) -> list[str]:
        return [
            entry["payload"]["request"]["request_id"]
            for entry in result.transcript["entries"]
            if entry["kind"] == "decision"
            and entry["payload"].get("kind") == "agent_request"
        ]

    assert request_ids(first) == request_ids(second)
    assert request_ids(first)[0] == "stable-local-run:request:000001"
    assert first.transcript_digest == second.transcript_digest, _first_stable_difference(
        first.transcript,
        second.transcript,
    )


def _first_stable_difference(first: Any, second: Any) -> str:
    timing_keys = {"_ts", "deadline_monotonic", "latency_seconds", "elapsed_seconds"}

    def normalized(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: normalized(item)
                for key, item in value.items()
                if key not in timing_keys and key not in {"ts_monotonic", "payload_hash"}
            }
        if isinstance(value, list):
            return [normalized(item) for item in value]
        return value

    def visit(left: Any, right: Any, path: str = "$ first") -> str | None:
        if type(left) is not type(right):
            return f"{path}: type {type(left).__name__} != {type(right).__name__}"
        if isinstance(left, dict):
            if left.keys() != right.keys():
                return f"{path}: keys {sorted(left)} != {sorted(right)}"
            for key in left:
                difference = visit(left[key], right[key], f"{path}.{key}")
                if difference is not None:
                    return difference
            return None
        if isinstance(left, list):
            if len(left) != len(right):
                return f"{path}: length {len(left)} != {len(right)}"
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
                difference = visit(left_item, right_item, f"{path}[{index}]")
                if difference is not None:
                    return difference
            return None
        if left != right:
            return f"{path}: {left!r} != {right!r}"
        return None

    return visit(normalized(first), normalized(second)) or "no normalized field difference"


@pytest.mark.asyncio
async def test_runner_rejects_missing_reproducibility_seed_before_execution():
    spec = _spec().model_copy(update={"actor_seed": None})
    with pytest.raises(ValueError, match="actor_seed"):
        await run_werewolf_run(
            spec,
            model_config=ModelConfig(provider="openai", model="m", api_key="k"),
            router=_Router(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_runner_has_no_scripted_actor_factory_parameter():
    with pytest.raises(TypeError):
        await run_werewolf_run(  # type: ignore[call-arg]
            _spec(),
            model_config=ModelConfig(provider="openai", model="m", api_key="k"),
            router=_Router(),  # type: ignore[arg-type]
            actor_factory=lambda *_args: {},
        )


@pytest.mark.asyncio
async def test_runner_applies_explicit_personas_and_records_exact_role_layout(monkeypatch):
    created: dict[int, _ProtocolActor] = {}

    def build_test_actors(state, **kwargs):
        model_config = kwargs["model_config"]
        seat_configs = kwargs.get("seat_configs") or {}
        actors = {
            player.id: _ProtocolActor(
                player,
                state,
                model_config.merge(seat_configs.get(player.seat)),
            )
            for player in state.players
        }
        created.update({actor.seat: actor for actor in actors.values()})
        return actors

    monkeypatch.setattr("src.harness.runner.build_actors", build_test_actors)
    run_spec = ExperimentSpec(
        experiment_id="explicit-controls",
        player_names=["A", "B", "C", "D", "E", "F"],
        role_deck=["werewolf", "werewolf", "seer", "villager", "villager", "villager"],
        turn_policies=["fixed_round_robin"],
        replicates=1,
        base_seed=70,
        max_speak_rounds=1,
        metadata={
            "role_layout_mode": "fixed",
            "role_layout_seed": 800,
            "persona_mode": "fixed",
            "persona_seed": 900,
        },
    ).expand_runs()[0]
    config = ModelConfig(
        provider="openai",
        model="test-model",
        api_base="https://example.invalid/v1",
        api_key="test-only-key",
    )

    result = await run_werewolf_run(
        run_spec,
        model_config=config,
        router=_Router(),  # type: ignore[arg-type]
        close_router=False,
    )

    assignments = {
        int(item["seat"]): item
        for item in result.run_spec["metadata"]["persona_assignments"]
    }
    assert len(created) == len(assignments) == 6
    assert {
        seat: (actor.persona_name, actor.persona_desc)
        for seat, actor in created.items()
    } == {
        seat: (item["name"], item["description"])
        for seat, item in assignments.items()
    }
    expected_layout = {
        int(item["seat"]): item["role"]
        for item in result.run_spec["metadata"]["role_layout"]
    }
    actual_layout = {
        int(item["seat"]): item["role"]
        for item in result.analysis["seats"]
    }
    assert actual_layout == expected_layout
    assert result.run_spec["metadata"]["role_layout_id"].startswith("role-layout-")


def test_resolve_run_spec_records_default_ruleset_and_role_deck():
    unresolved = _spec("resolved-provenance").model_copy(update={"role_deck": []})
    resolved = resolve_run_spec(
        unresolved,
        model_config=ModelConfig(provider="openai", model="m", api_key="k"),
    )

    assert resolved.ruleset_id == CLASSIC_RULESET_ID
    assert resolved.role_deck == [role.value for role in default_role_deck(6)]
    assert resolved.model_dump()["ruleset_id"] == "classic.v1"


def test_run_spec_rejects_unknown_ruleset_and_invalid_custom_deck():
    values = _spec("invalid-ruleset").model_dump()
    values["ruleset_id"] = "classic.v2"
    with pytest.raises(ValidationError, match="unsupported Werewolf ruleset"):
        RunSpec(**values)

    values = _spec("invalid-deck").model_dump()
    values["role_deck"] = ["villager"] * 6
    with pytest.raises(ValidationError, match="at least one werewolf"):
        RunSpec(**values)


def test_resolve_run_spec_revalidates_model_copy_updates():
    bypassed = _spec("bypassed-validation").model_copy(update={"ruleset_id": "classic.v2"})

    with pytest.raises(ValueError, match="unsupported Werewolf ruleset"):
        resolve_run_spec(
            bypassed,
            model_config=ModelConfig(provider="openai", model="m", api_key="k"),
        )


def test_experiment_expansion_preserves_ruleset_provenance():
    runs = ExperimentSpec(
        experiment_id="ruleset-provenance",
        player_names=["A", "B", "C", "D", "E", "F"],
        role_deck=[],
        replicates=1,
        base_seed=7,
    ).expand_runs()

    assert len(runs) == 1
    assert runs[0].ruleset_id == CLASSIC_RULESET_ID
    assert runs[0].model_dump()["ruleset_id"] == "classic.v1"
