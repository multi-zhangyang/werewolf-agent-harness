"""Agent memory and decision-schema boundary tests."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agent.memory import AgentMemory
from src.agent.schemas import AgentAction, Decision


def test_memory_returns_recent_observations_in_actual_chronological_order():
    memory = AgentMemory(seat=1, role="villager")
    for index in range(6):
        memory.observe(1, "day", "speech", f"event-{index}")

    recent = memory.recent_observations(limit=3)

    assert [item.text for item in recent] == ["event-3", "event-4", "event-5"]


def test_memory_records_public_claims_without_inventing_truth_or_contradictions():
    memory = AgentMemory(seat=1, role="villager")
    memory.record_claim(2, 1, {"role": "seer", "checked_seat": 5, "result": "wolf"})
    memory.record_claim(3, 1, {"role": "seer", "checked_seat": 5, "result": "village"})

    rendered = memory.render_for_prompt()

    assert "仅记录，不代表真实" in rendered
    assert "2号" in rendered and "3号" in rendered
    assert "必有狼人" not in rendered
    assert "硬信号" not in rendered


def test_memory_snapshot_contains_counts_and_claims_only():
    memory = AgentMemory(seat=4, role="werewolf")
    memory.observe(2, "night", "private_result", "private observation")
    memory.record_claim(5, 2, {"role": "seer"})

    snapshot = memory.snapshot()

    assert snapshot["seat"] == 4
    assert snapshot["role"] == "werewolf"
    assert snapshot["observation_count"] == 1
    assert snapshot["retained_observation_count"] == 1
    assert snapshot["observation_summary"]["archived_count"] == 0
    assert len(snapshot["observation_summary"]["archived_digest"]) == 64
    assert snapshot["claim_count"] == 1
    assert snapshot["retained_claim_count"] == 1
    assert snapshot["claims"] == {"5": [{"role": "seer", "day": 2}]}
    assert snapshot["claim_summary"]["5"] == {
        "total_count": 1,
        "retained_count": 1,
        "omitted_count": 0,
        "first_day": 2,
        "last_day": 2,
        "transition_count": 0,
        "role_counts": {"seer": 1},
        "seer_result_counts": {},
    }
    assert "trust" not in snapshot
    assert "reflection_count" not in snapshot


def test_memory_deep_copies_nested_observation_and_claim_inputs():
    memory = AgentMemory(seat=1, role="villager")
    observation_payload = {"targets": [2], "detail": {"source": ["private"]}}
    claim = {
        "role": "seer",
        "evidence": {"checked": [3], "labels": {"result": "wolf"}},
    }

    memory.observe(1, "day", "speech", "observed", payload=observation_payload)
    memory.record_claim(2, 1, claim)
    observation_payload["targets"].append(6)
    observation_payload["detail"]["source"].append("mutated")
    claim["evidence"]["checked"].append(5)
    claim["evidence"]["labels"]["result"] = "village"

    stored = memory.observations[0].metadata["payload"]
    assert stored == {"targets": [2], "detail": {"source": ["private"]}}
    assert memory.claims[2][0]["evidence"] == {
        "checked": [3],
        "labels": {"result": "wolf"},
    }


def test_memory_public_reads_and_snapshot_are_detached():
    memory = AgentMemory(seat=1, role="villager")
    memory.observe(1, "day", "speech", "observed", payload={"targets": [2]})
    memory.record_claim(2, 1, {"role": "seer", "evidence": {"checked": [3]}})

    observations = memory.observations
    recent = memory.recent_observations()
    claims = memory.claims
    snapshot = memory.snapshot()
    observations[0].metadata["payload"]["targets"].append(9)
    recent[0].metadata["payload"]["targets"].append(8)
    claims[2][0]["evidence"]["checked"].append(7)
    snapshot["claims"]["2"][0]["evidence"]["checked"].append(6)

    assert memory.observations[0].metadata["payload"]["targets"] == [2]
    assert memory.recent_observations()[0].metadata["payload"]["targets"] == [2]
    assert memory.claims[2][0]["evidence"]["checked"] == [3]
    assert memory.snapshot()["claims"]["2"][0]["evidence"]["checked"] == [3]


def test_memories_do_not_share_nested_observation_or_claim_state():
    first = AgentMemory(seat=1, role="villager")
    second = AgentMemory(seat=2, role="villager")
    shared_targets = [3]
    shared_claim = {"role": "seer", "evidence": {"checked": [4]}}
    for memory in (first, second):
        memory.observe(1, "day", "speech", "shared", accuses=shared_targets)
        memory.record_claim(3, 1, shared_claim)

    shared_targets.append(5)
    shared_claim["evidence"]["checked"].append(6)
    first_observations = first.observations
    first_claims = first.claims
    first_observations[0].metadata["accuses"].append(7)
    first_claims[3][0]["evidence"]["checked"].append(8)

    assert first.observations[0].metadata["accuses"] == [3]
    assert second.observations[0].metadata["accuses"] == [3]
    assert first.claims[3][0]["evidence"]["checked"] == [4]
    assert second.claims[3][0]["evidence"]["checked"] == [4]


def test_long_memory_is_bounded_and_keeps_durable_facts_and_claim_anchors():
    memory = AgentMemory(
        seat=1,
        role="seer",
        max_observations=4,
        max_claims_per_seat=3,
    )
    memory.observe(1, "night", "seer_result", "2号是狼人", target_seat=2, team="wolf")
    for index in range(9):
        memory.observe(index + 1, "day", "speech", f"routine-{index}")

    first_claim = {"role": "seer", "checked_seat": 2, "result": "wolf"}
    claims = [
        first_claim,
        first_claim,
        {"role": "villager"},
        {"role": "seer", "checked_seat": 3, "result": "village"},
        {"role": "werewolf"},
    ]
    for day, claim in enumerate(claims, start=1):
        memory.record_claim(2, day, claim)

    snapshot = memory.snapshot()
    retained = memory.observations
    retained_claims = memory.claims[2]
    rendered = memory.render_for_prompt(obs_limit=2)

    assert snapshot["observation_count"] == 10
    assert snapshot["retained_observation_count"] == 4
    assert snapshot["observation_summary"]["archived_count"] == 6
    assert len(retained) == 4
    assert any(item.kind == "seer_result" and item.metadata["target_seat"] == 2 for item in retained)
    assert [item.text for item in retained if item.kind == "speech"] == [
        "routine-6",
        "routine-7",
        "routine-8",
    ]
    assert snapshot["claim_count"] == 5
    assert snapshot["retained_claim_count"] == 3
    assert snapshot["claim_summary"]["2"]["omitted_count"] == 2
    assert snapshot["claim_summary"]["2"]["transition_count"] == 3
    assert retained_claims[0] == {**first_claim, "day": 1}
    assert retained_claims[-1] == {"role": "werewolf", "day": 5}
    assert "程序统计，不是模型总结" in rendered
    assert "可见事件累计=10" in rendered
    assert "公开结构化声明累计=5" in rendered
    assert '"role": "seer"' in rendered
    assert '"role": "werewolf"' in rendered


def test_compacted_history_digest_still_commits_to_evicted_content():
    first = AgentMemory(seat=1, role="villager", max_observations=2)
    second = AgentMemory(seat=1, role="villager", max_observations=2)
    first.observe(1, "day", "speech", "evicted-a")
    second.observe(1, "day", "speech", "evicted-b")
    for memory in (first, second):
        memory.observe(2, "day", "speech", "same-retained-1")
        memory.observe(3, "day", "speech", "same-retained-2")

    assert [item.text for item in first.observations] == [
        item.text for item in second.observations
    ]
    assert first.digest() != second.digest()


def test_public_vote_ledger_survives_observation_eviction_and_archives_boundedly():
    memory = AgentMemory(
        seat=1,
        role="villager",
        max_observations=2,
        max_public_votes=2,
    )
    memory.observe(
        1,
        "voting",
        "vote",
        "1号投了3号",
        voter_seat=1,
        target_seat=3,
        pk=False,
    )
    for day in range(2, 8):
        memory.observe(day, "day", "speech", f"routine-{day}")

    # The ordinary observation window may evict the vote, but the independent
    # public ledger still has it until its own cap is reached.
    assert not any(item.kind == "vote" for item in memory.observations)
    assert memory.read_public_votes() == [{
        "day": 1,
        "phase": "voting",
        "voter_seat": 1,
        "target_seat": 3,
        "pk": False,
    }]

    memory.observe(8, "voting", "vote", "2号投了4号", voter_seat=2, target_seat=4, pk=False)
    memory.observe(9, "voting", "vote", "1号投了5号", voter_seat=1, target_seat=5, pk=True)
    memory.observe(10, "voting", "vote", "3号投了1号", voter_seat=3, target_seat=1, pk=False)

    snapshot = memory.snapshot()
    assert snapshot["public_vote_count"] == 4
    assert snapshot["retained_public_vote_count"] == 2
    assert snapshot["public_vote_ledger"] == [
        {
            "day": 9,
            "phase": "voting",
            "voter_seat": 1,
            "target_seat": 5,
            "pk": True,
        },
        {
            "day": 10,
            "phase": "voting",
            "voter_seat": 3,
            "target_seat": 1,
            "pk": False,
        },
    ]
    assert snapshot["public_vote_summary"]["archived_count"] == 2
    assert len(snapshot["public_vote_summary"]["archived_digest"]) == 64

    # A malformed vote observation is still an observation, but cannot become
    # a structured accepted-vote record.
    memory.observe(11, "voting", "vote", "unstructured")
    memory.observe(12, "voting", "vote", "bad pk", voter_seat=4, target_seat=5, pk="false")
    memory.observe(13, "voting", "vote", "string seat", voter_seat="4", target_seat=5, pk=False)
    memory.observe(14, "voting", "vote", "float seat", voter_seat=4.0, target_seat=5, pk=False)
    assert memory.snapshot()["public_vote_count"] == 4


def test_public_vote_ledger_reads_and_snapshots_are_detached():
    memory = AgentMemory(seat=1, role="villager", max_public_votes=3)
    for day, voter, target, pk in (
        (1, 1, 3, False),
        (1, 2, 3, True),
        (2, 2, 4, False),
        (2, 3, 3, False),
    ):
        memory.observe(
            day,
            "voting",
            "vote",
            f"{voter}号投了{target}号",
            voter_seat=voter,
            target_seat=target,
            pk=pk,
        )

    before_digest = memory.digest()
    ledger = memory.public_vote_ledger
    snapshot = memory.snapshot()
    filtered = memory.read_public_votes(target_seat=3, pk=False)
    ledger[0]["target_seat"] = 999
    snapshot["public_vote_ledger"][0]["target_seat"] = 998
    filtered[0]["target_seat"] = 997

    assert memory.public_vote_ledger[0]["target_seat"] == 3
    assert memory.snapshot()["public_vote_ledger"][0]["target_seat"] == 3
    assert memory.digest() == before_digest
    assert memory.read_public_votes(target_seat=3, pk=False) == [{
        "day": 2,
        "phase": "voting",
        "voter_seat": 3,
        "target_seat": 3,
        "pk": False,
    }]


def test_public_vote_archive_digest_distinguishes_evicted_vote_facts():
    first = AgentMemory(seat=1, role="villager", max_public_votes=1)
    second = AgentMemory(seat=1, role="villager", max_public_votes=1)
    for memory, first_target in ((first, 2), (second, 3)):
        memory.observe(
            1,
            "voting",
            "vote",
            "1号首票",
            voter_seat=1,
            target_seat=first_target,
            pk=False,
        )
        memory.observe(
            2,
            "voting",
            "vote",
            "2号末票",
            voter_seat=2,
            target_seat=4,
            pk=True,
        )

    assert first.public_vote_ledger == second.public_vote_ledger
    assert (
        first.snapshot()["public_vote_summary"]["archived_digest"]
        != second.snapshot()["public_vote_summary"]["archived_digest"]
    )


def test_removed_self_reported_audit_fields_are_rejected():
    for field, value in (
        ("attitudes", {3: "oppose"}),
        ("deception", "fabrication"),
        ("suspicion", {3: 0.9}),
        ("objective_summary", "model-authored summary"),
        ("parse_failed", True),
    ):
        with pytest.raises(ValidationError):
            Decision(action=AgentAction.SPEAK, **{field: value})


def test_accuses_validator_filters_invalid_values():
    decision = Decision(action=AgentAction.SPEAK, accuses=["3号", 5, 5, "0", "abc", 3.0, 2.9])
    assert decision.accuses == [3, 5]


def test_reply_to_validator_tolerates_seat_text_and_rejects_non_positive():
    assert Decision(action=AgentAction.SPEAK, reply_to="3号").reply_to == 3
    assert Decision(action=AgentAction.SPEAK, reply_to=0).reply_to is None
    assert Decision(action=AgentAction.SPEAK, reply_to=2.9).reply_to is None
