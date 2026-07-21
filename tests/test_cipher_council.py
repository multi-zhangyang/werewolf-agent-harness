"""Cipher Council production-plugin and Core-boundary tests."""
from __future__ import annotations

import asyncio
import ast
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

import src.environments.cipher_council.evidence as cipher_council_evidence
from src.environments.cipher_council import (
    CipherCouncilArtifactEvidenceError,
    CipherCouncilConfig,
    CipherCouncilEnvironmentPlugin,
    CipherCouncilV2EnvironmentPlugin,
    verify_cipher_council_v2_artifacts,
)
from src.harness.artifacts import (
    ArtifactIntegrityError,
    load_verified_artifact_snapshot,
    verify_run_artifacts,
    write_run_artifacts,
)
from src.harness.core_protocol import ActionChoice, ActionRequest, DecisionEnvelope, SkipChoice
from src.harness.core_runner import run_environment_run
from src.harness.core_spec import CoreRunSpec, EnvironmentRef, ExecutionSpec
from src.harness.registry import EnvironmentRegistry
from src.harness.visibility import audit_transcript_visibility, project_transcript_rows


class CipherCouncilBarrier:
    """Fails a sequential implementation of the v2 faction council quickly."""

    def __init__(self, expected_participants: int) -> None:
        self.expected_participants = expected_participants
        self.participants: set[str] = set()
        self._released = asyncio.Event()

    async def wait(self, actor_id: str) -> None:
        self.participants.add(actor_id)
        if len(self.participants) == self.expected_participants:
            self._released.set()
        await asyncio.wait_for(self._released.wait(), timeout=0.2)


class ScriptedCouncilAgent:
    """Independent test Actor that chooses only from its own Core request."""

    def __init__(
        self,
        actor_id: str,
        *,
        skip_vote: bool = False,
        skip_nomination: bool = False,
        skip_commitment: bool = False,
        skip_cipher_council: bool = False,
        sabotage_when_cipher: bool = False,
        cipher_council_barrier: CipherCouncilBarrier | None = None,
    ) -> None:
        self.actor_id = actor_id
        self.skip_vote = skip_vote
        self.skip_nomination = skip_nomination
        self.skip_commitment = skip_commitment
        self.skip_cipher_council = skip_cipher_council
        self.sabotage_when_cipher = sabotage_when_cipher
        self.cipher_council_barrier = cipher_council_barrier
        self.requests: list[ActionRequest] = []

    async def decide(self, request: ActionRequest) -> DecisionEnvelope:
        self.requests.append(request.model_copy(deep=True))
        stage = str(request.labels["stage"])
        actor_ids = list(request.observation["public_state"]["actor_ids"])
        if stage == "cipher_council":
            if self.cipher_council_barrier is not None:
                await self.cipher_council_barrier.wait(self.actor_id)
            choice = (
                SkipChoice(reason="withhold faction strategy")
                if self.skip_cipher_council
                else ActionChoice(
                    action="send_cipher_strategy_message",
                    arguments={"message": f"private Cipher strategy from {self.actor_id}"},
                )
            )
        elif stage == "deliberation":
            choice = ActionChoice(
                action="speak",
                arguments={"message": f"{self.actor_id} makes a public claim."},
            )
        elif stage == "nomination":
            if self.skip_nomination:
                choice = SkipChoice(reason="decline nomination")
            else:
                mission_size = int(request.observation["public_state"]["mission_size"])
                choice = ActionChoice(
                    action="nominate_mission_team",
                    arguments={"members": actor_ids[:mission_size]},
                )
        elif stage == "vote":
            choice = (
                SkipChoice(reason="abstain")
                if self.skip_vote
                else ActionChoice(action="cast_public_vote", arguments={"approve": True})
            )
        elif stage == "mission_commitment":
            if self.skip_commitment:
                choice = SkipChoice(reason="withhold commitment")
            else:
                faction = request.observation["private_identity"]["faction"]
                commitment = (
                    "sabotage"
                    if faction == "cipher" and self.sabotage_when_cipher
                    else "support"
                )
                choice = ActionChoice(
                    action="commit_secret_mission_action",
                    arguments={"commitment": commitment},
                )
        else:  # pragma: no cover - detects an unexpected environment stage.
            raise AssertionError(f"unexpected stage: {stage}")
        return DecisionEnvelope(
            request_id=request.request_id,
            actor_id=self.actor_id,
            choice=choice,
            private_reasoning=f"private reasoning for {self.actor_id}/{stage}",
            parse_status="not_applicable",
        )


def _spec(
    *,
    run_id: str = "cipher-council-run",
    version: str = "1",
    mission_sizes: list[int] | None = None,
    victory_target: int = 1,
    max_proposals: int = 1,
) -> CoreRunSpec:
    return CoreRunSpec(
        run_id=run_id,
        environment=EnvironmentRef(id="council.cipher", version=version),
        environment_config={
            "player_names": ["A", "B", "C", "D", "E"],
            "cipher_count": 2,
            "mission_sizes": mission_sizes or [2],
            "victory_target": victory_target,
            "max_proposals_per_mission": max_proposals,
            "public_history_limit": 12,
        },
        seeds={"roles": 101, "order": 202},
        execution=ExecutionSpec(decision_timeout_seconds=2),
        metadata={"suite": "cipher-council-test"},
    )


def _agents(**overrides: dict[str, Any]) -> dict[str, ScriptedCouncilAgent]:
    return {
        f"council:{seat}": ScriptedCouncilAgent(
            f"council:{seat}",
            **overrides.get(f"council:{seat}", {}),
        )
        for seat in range(1, 6)
    }


async def _run(
    spec: CoreRunSpec,
    agents: dict[str, ScriptedCouncilAgent],
    *,
    plugin: CipherCouncilEnvironmentPlugin | CipherCouncilV2EnvironmentPlugin | None = None,
):
    registry = EnvironmentRegistry()
    registry.register(plugin or CipherCouncilEnvironmentPlugin())
    return await run_environment_run(
        spec,
        registry=registry,
        resolve_agent=agents.__getitem__,
    )


def _payloads(result: Any, event_type: str) -> list[dict[str, Any]]:
    return [
        row["payload"]
        for row in result.transcript["entries"]
        if row["kind"] == "event" and row["payload"].get("type") == event_type
    ]


def _verify_v2_evidence_rows(
    result: Any,
    spec: CoreRunSpec,
    rows: list[dict[str, Any]],
    *,
    summary: dict[str, Any] | None = None,
) -> None:
    cipher_council_evidence._verify_rows(
        run_id=result.run_id,
        transcript_digest=result.transcript_digest,
        environment_config=spec.environment_config,
        summary=(
            {"metrics": deepcopy(result.metrics)}
            if summary is None
            else summary
        ),
        rows=rows,
    )


@pytest.mark.asyncio
async def test_cipher_council_runs_with_independent_actors_and_safe_projections():
    spec = _spec()
    agents = _agents()
    result = await _run(spec, agents)

    assert result.status == "completed"
    assert result.outcome["winner"] == "council"
    assert result.metrics["player_count"] == 5
    assert result.metrics["missions_resolved"] == 1
    assert "cipher_council_round_count" not in result.metrics
    assert "cipher_council_request_count" not in result.metrics
    assert "cipher_council_message_count" not in result.metrics
    assert not _payloads(result, "council_cipher_message")
    assert len({id(actor) for actor in agents.values()}) == 5
    assert any(
        row["payload"].get("type") == "agent_bindings_finalized"
        and row["payload"].get("actor_count") == 5
        for row in result.transcript["entries"]
        if row["kind"] == "harness"
    )

    private_assignments = _payloads(result, "council_role_assigned")
    assert len(private_assignments) == 5
    roles = {row["actor_id"]: row["role"] for row in private_assignments}
    assert list(roles.values()).count("cipher") == 2
    for assignment in private_assignments:
        assert assignment["visibility"] == "private"
        assert assignment["recipients"] == [assignment["actor_id"]]

    for actor_id, actor in agents.items():
        assert actor.requests
        for request in actor.requests:
            identity = request.observation["private_identity"]
            assert request.actor_id == actor_id
            assert identity["actor_id"] == actor_id
            assert identity["faction"] == roles[actor_id]
            if roles[actor_id] == "cipher":
                assert set(identity["cipher_teammates"]) == {
                    other_id
                    for other_id, role in roles.items()
                    if role == "cipher" and other_id != actor_id
                }
            else:
                assert "cipher_teammates" not in identity

    rows = result.transcript["entries"]
    public = project_transcript_rows(rows, audience="public")
    player_one = project_transcript_rows(
        rows,
        audience="player",
        player_id="council:1",
    )
    god = project_transcript_rows(rows, audience="god")
    admin = project_transcript_rows(rows, audience="admin")
    public_text = json.dumps(public, ensure_ascii=False)
    player_text = json.dumps(player_one, ensure_ascii=False)
    god_text = json.dumps(god, ensure_ascii=False)
    admin_text = json.dumps(admin, ensure_ascii=False)

    assert "council_role_assigned" not in public_text
    assert "private reasoning for" not in public_text
    assert "private reasoning for" not in player_text
    assert "private reasoning for" not in god_text
    assert "private reasoning for" in admin_text
    assert "council_role_assigned" in player_text
    assert "council_role_assigned" in god_text
    assert not [issue for issue in audit_transcript_visibility(rows) if issue.severity == "error"]


@pytest.mark.asyncio
async def test_cipher_council_v2_runs_simultaneous_private_faction_council() -> None:
    spec = _spec(run_id="cipher-council-v2", version="2")
    agents = _agents()
    barrier = CipherCouncilBarrier(expected_participants=2)
    for actor in agents.values():
        actor.cipher_council_barrier = barrier

    result = await _run(
        spec,
        agents,
        plugin=CipherCouncilV2EnvironmentPlugin(),
    )

    assert result.status == "completed"
    assignments = _payloads(result, "council_role_assigned")
    roles = {row["actor_id"]: row["role"] for row in assignments}
    cipher_ids = tuple(
        actor_id for actor_id in agents if roles[actor_id] == "cipher"
    )
    council_ids = tuple(
        actor_id for actor_id in agents if roles[actor_id] == "council"
    )
    assert len(cipher_ids) == 2
    assert barrier.participants == set(cipher_ids)

    message_events = _payloads(result, "council_cipher_message")
    assert len(message_events) == 2
    assert {row["actor_id"] for row in message_events} == set(cipher_ids)
    assert all(row["recipients"] == list(cipher_ids) for row in message_events)
    assert all(row["message"] == f"private Cipher strategy from {row['actor_id']}" for row in message_events)

    for actor_id in cipher_ids:
        actor = agents[actor_id]
        council_request = next(
            request
            for request in actor.requests
            if request.labels["stage"] == "cipher_council"
        )
        # Every current-round request was built before any current-round
        # strategy was committed, so no Cipher can read an early teammate reply.
        assert council_request.observation["private_identity"]["cipher_council_messages"] == []
        deliberation_request = next(
            request
            for request in actor.requests
            if request.labels["stage"] == "deliberation"
        )
        visible_messages = deliberation_request.observation["private_identity"][
            "cipher_council_messages"
        ]
        assert {row["actor_id"] for row in visible_messages} == set(cipher_ids)
        assert len(visible_messages) == 2

    for actor_id in council_ids:
        deliberation_request = next(
            request
            for request in agents[actor_id].requests
            if request.labels["stage"] == "deliberation"
        )
        assert "cipher_council_messages" not in deliberation_request.observation[
            "private_identity"
        ]

    rows = result.transcript["entries"]
    public = project_transcript_rows(rows, audience="public")
    cipher_player = project_transcript_rows(
        rows,
        audience="player",
        player_id=cipher_ids[0],
    )
    council_player = project_transcript_rows(
        rows,
        audience="player",
        player_id=council_ids[0],
    )
    god = project_transcript_rows(rows, audience="god")
    public_text = json.dumps(public, ensure_ascii=False)
    cipher_text = json.dumps(cipher_player, ensure_ascii=False)
    council_text = json.dumps(council_player, ensure_ascii=False)
    god_text = json.dumps(god, ensure_ascii=False)

    assert "council_cipher_message" not in public_text
    assert "private Cipher strategy" not in public_text
    assert "council_cipher_message" in cipher_text
    assert "private Cipher strategy" in cipher_text
    assert "council_cipher_message" not in council_text
    assert "private Cipher strategy" not in council_text
    assert "council_cipher_message" in god_text
    assert result.metrics["cipher_council_faction_size"] == 2
    assert result.metrics["cipher_council_round_count"] == 1
    assert result.metrics["cipher_council_request_count"] == 2
    assert result.metrics["cipher_council_message_count"] == 2
    assert result.metrics["cipher_council_absent_count"] == 0
    assert not [issue for issue in audit_transcript_visibility(rows) if issue.severity == "error"]


@pytest.mark.asyncio
async def test_cipher_council_v2_missing_faction_messages_remain_absent_and_private(
    tmp_path: Path,
) -> None:
    spec = _spec(run_id="cipher-council-v2-missing-message", version="2")
    agents = _agents(**{
        f"council:{seat}": {"skip_cipher_council": True}
        for seat in range(1, 6)
    })

    result = await _run(
        spec,
        agents,
        plugin=CipherCouncilV2EnvironmentPlugin(),
    )

    assert result.status == "completed"
    assert not _payloads(result, "council_cipher_message")
    assert result.metrics["cipher_council_faction_size"] == 2
    assert result.metrics["cipher_council_round_count"] == 1
    assert result.metrics["cipher_council_request_count"] == 2
    assert result.metrics["cipher_council_message_count"] == 0
    assert result.metrics["cipher_council_absent_count"] == 2
    assert result.metrics["explicit_skip_count"] == 2
    assert not [
        row
        for row in _payloads(result, "council_action_unavailable")
        if row["stage"] == "cipher_council"
    ]
    public = project_transcript_rows(result.transcript["entries"], audience="public")
    assert "cipher_council" not in json.dumps(public, ensure_ascii=False)
    paths = write_run_artifacts(result, spec, tmp_path)
    evidence = verify_cipher_council_v2_artifacts(paths["run_dir"])
    assert evidence.cipher_council_message_count == 0
    assert evidence.cipher_council_absent_count == 2


@pytest.mark.asyncio
async def test_cipher_council_v2_artifact_evidence_recomputes_private_coordination(
    tmp_path: Path,
) -> None:
    spec = _spec(run_id="cipher-council-v2-artifact", version="2")
    result = await _run(
        spec,
        _agents(),
        plugin=CipherCouncilV2EnvironmentPlugin(),
    )

    paths = write_run_artifacts(result, spec, tmp_path)
    evidence = verify_cipher_council_v2_artifacts(paths["run_dir"])

    assert evidence.run_id == spec.run_id
    assert evidence.transcript_digest == result.transcript_digest
    assert evidence.cipher_faction_size == 2
    assert evidence.cipher_council_round_count == 1
    assert evidence.cipher_council_request_count == 2
    assert evidence.cipher_council_message_count == 2
    assert evidence.cipher_council_absent_count == 0
    assert evidence.message_delivery_barrier_verified is True
    assert evidence.observation_isolation_verified is True

    # The wrapper first checks hashes, so a raw artifact mutation cannot be
    # accepted as an environment-evidence report.
    transcript_path = Path(paths["transcript_jsonl"])
    transcript_path.write_text(transcript_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="integrity mismatch"):
        verify_cipher_council_v2_artifacts(paths["run_dir"])


@pytest.mark.asyncio
async def test_cipher_council_v2_artifact_evidence_rejects_semantic_violations() -> None:
    spec = _spec(run_id="cipher-council-v2-semantic-evidence", version="2")
    result = await _run(
        spec,
        _agents(),
        plugin=CipherCouncilV2EnvironmentPlugin(),
    )
    baseline_rows = result.transcript["entries"]

    wrong_recipients = deepcopy(baseline_rows)
    message = next(
        row["payload"]
        for row in wrong_recipients
        if row["kind"] == "event"
        and row["payload"].get("type") == "council_cipher_message"
    )
    message["recipients"] = [message["actor_id"]]
    with pytest.raises(
        CipherCouncilArtifactEvidenceError,
        match="invalid private recipient route",
    ):
        _verify_v2_evidence_rows(result, spec, wrong_recipients)

    before_barrier = deepcopy(baseline_rows)
    message_row = next(
        row
        for row in before_barrier
        if row["kind"] == "event"
        and row["payload"].get("type") == "council_cipher_message"
    )
    message_row["seq"] = 1
    with pytest.raises(
        CipherCouncilArtifactEvidenceError,
        match="before the round barrier",
    ):
        _verify_v2_evidence_rows(result, spec, before_barrier)

    assignments = {
        row["payload"]["actor_id"]: row["payload"]["role"]
        for row in baseline_rows
        if row["kind"] == "event"
        and row["payload"].get("type") == "council_role_assigned"
    }
    council_id = next(
        actor_id for actor_id, role in assignments.items() if role == "council"
    )
    council_observation_leak = deepcopy(baseline_rows)
    council_request = next(
        row["payload"]["request"]
        for row in council_observation_leak
        if row["kind"] == "decision"
        and row["payload"].get("kind") == "agent_request"
        and row["payload"]["request"]["actor_id"] == council_id
    )
    council_request["observation"]["private_identity"][
        "cipher_council_messages"
    ] = []
    with pytest.raises(
        CipherCouncilArtifactEvidenceError,
        match="Council Actor observation",
    ):
        _verify_v2_evidence_rows(result, spec, council_observation_leak)

    current_round_leak = deepcopy(baseline_rows)
    cipher_request = next(
        row["payload"]["request"]
        for row in current_round_leak
        if row["kind"] == "decision"
        and row["payload"].get("kind") == "agent_request"
        and row["payload"]["request"]["labels"].get("stage") == "cipher_council"
    )
    labels = cipher_request["labels"]
    cipher_request["observation"]["private_identity"][
        "cipher_council_messages"
    ].append({
        "mission": labels["mission"],
        "proposal_attempt": labels["proposal_attempt"],
        "actor_id": cipher_request["actor_id"],
        "message": "forged current-round strategy",
    })
    with pytest.raises(
        CipherCouncilArtifactEvidenceError,
        match="current-round message",
    ):
        _verify_v2_evidence_rows(result, spec, current_round_leak)

    bad_tool_route = deepcopy(baseline_rows)
    strategy_request = next(
        row["payload"]["request"]
        for row in bad_tool_route
        if row["kind"] == "decision"
        and row["payload"].get("kind") == "agent_request"
        and row["payload"]["request"]["labels"].get("stage") == "cipher_council"
    )
    strategy_request["legal_actions"][0]["metadata"]["visibility"] = "public"
    with pytest.raises(
        CipherCouncilArtifactEvidenceError,
        match="message tool private",
    ):
        _verify_v2_evidence_rows(result, spec, bad_tool_route)

    inconsistent_metrics = {"metrics": deepcopy(result.metrics)}
    inconsistent_metrics["metrics"]["cipher_council_message_count"] += 1
    with pytest.raises(
        CipherCouncilArtifactEvidenceError,
        match="does not match transcript evidence",
    ):
        _verify_v2_evidence_rows(
            result,
            spec,
            deepcopy(baseline_rows),
            summary=inconsistent_metrics,
        )


@pytest.mark.asyncio
async def test_cipher_council_v2_evidence_uses_the_verified_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(run_id="cipher-council-v2-evidence-snapshot", version="2")
    result = await _run(
        spec,
        _agents(),
        plugin=CipherCouncilV2EnvironmentPlugin(),
    )
    paths = write_run_artifacts(result, spec, tmp_path)
    snapshot = load_verified_artifact_snapshot(paths["run_dir"])

    def replace_after_snapshot(_run_dir: str | Path):
        Path(paths["transcript_jsonl"]).write_text("not verified content\n", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(
        cipher_council_evidence,
        "load_verified_artifact_snapshot",
        replace_after_snapshot,
    )
    evidence = verify_cipher_council_v2_artifacts(paths["run_dir"])
    assert evidence.transcript_digest == result.transcript_digest


@pytest.mark.asyncio
async def test_cipher_council_missing_vote_is_absent_not_a_fabricated_rejection():
    spec = _spec()
    agents = _agents(**{"council:5": {"skip_vote": True}})
    result = await _run(spec, agents)

    assert result.status == "completed"
    resolution = _payloads(result, "council_proposal_resolved")[0]
    assert resolution["approved"] is True
    assert resolution["approvals"] == 4
    assert resolution["rejections"] == 0
    assert resolution["absent"] == 1
    assert "council:5" not in {
        row["actor_id"] for row in _payloads(result, "council_vote_cast")
    }
    assert result.metrics["explicit_skip_count"] == 1


@pytest.mark.asyncio
async def test_cipher_council_missing_nomination_fails_the_proposal_without_inventing_one():
    spec = _spec(max_proposals=1)
    agents = _agents(**{
        f"council:{seat}": {"skip_nomination": True}
        for seat in range(1, 6)
    })
    result = await _run(spec, agents)

    assert result.status == "incomplete"
    assert result.termination_reason == "proposal_limit_exhausted"
    assert not _payloads(result, "council_proposal_submitted")
    assert _payloads(result, "council_proposal_unavailable")
    assert _payloads(result, "council_proposal_limit_exhausted")
    assert _payloads(result, "council_game_incomplete")


@pytest.mark.asyncio
async def test_cipher_council_missing_secret_commitment_voids_the_mission():
    spec = _spec()
    agents = _agents(**{"council:1": {"skip_commitment": True}})
    result = await _run(spec, agents)

    assert result.status == "incomplete"
    assert result.termination_reason == "mission_commitment_missing"
    assert not _payloads(result, "council_mission_commitment")
    assert not _payloads(result, "council_mission_resolved")
    assert _payloads(result, "council_mission_void")
    assert not [
        row
        for row in _payloads(result, "council_action_unavailable")
        if row["stage"] == "mission_commitment"
    ]
    failure_rows = [
        row["payload"]
        for row in result.transcript["entries"]
        if row["kind"] == "decision"
        and row["payload"].get("kind") == "agent_response"
        and row["payload"].get("request_id", "").endswith(":mission_commitment:council:1")
    ]
    assert failure_rows
    assert failure_rows[0]["validation"]["valid"] is False


@pytest.mark.asyncio
async def test_cipher_council_seeded_runs_and_generic_artifacts_are_reproducible(tmp_path: Path):
    spec = _spec(run_id="cipher-council-artifact")
    first = await _run(spec, _agents())
    second = await _run(spec, _agents())

    assert first.status == second.status == "completed"
    assert first.transcript_digest == second.transcript_digest

    paths = write_run_artifacts(first, spec, tmp_path)
    manifest = verify_run_artifacts(paths["run_dir"])
    assert manifest.run.environment.id == "council.cipher"
    assert manifest.run.environment.version == "1"
    assert manifest.transcript_digest == first.transcript_digest


def test_cipher_council_configuration_and_import_boundary_are_strict():
    assert CipherCouncilEnvironmentPlugin.descriptor.version == "1"
    assert CipherCouncilV2EnvironmentPlugin.descriptor.version == "2"
    assert "private_faction_coordination" in CipherCouncilV2EnvironmentPlugin.descriptor.capabilities
    with pytest.raises(ValueError, match="leave at least one Council"):
        CipherCouncilConfig(
            player_names=["A", "B", "C", "D", "E"],
            cipher_count=5,
        )
    with pytest.raises(ValueError, match="victory_target"):
        CipherCouncilConfig(
            player_names=["A", "B", "C", "D", "E"],
            mission_sizes=[2],
            victory_target=2,
        )

    source = Path("src/environments/cipher_council/plugin.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]
    assert not any("werewolf" in module.lower() for module in imports)
