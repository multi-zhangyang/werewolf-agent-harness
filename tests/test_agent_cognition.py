"""Seat-owned private cognition and commitment isolation tests."""
from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.agent.cognition import PrivateAgentState


def _update(*, seat: int = 2, wolf_probability: float = 0.2) -> dict:
    return {
        "beliefs": [{
            "seat": seat,
            "wolf_probability": wolf_probability,
            "likely_role": "villager",
            "confidence": 0.7,
            "evidence": ["D1: 2号的公开投票与发言一致"],
        }],
        "candidate_plans": ["公开支持2号", "公开攻击2号以隐藏真实判断"],
        "selected_plan": "公开攻击2号以隐藏真实判断",
        "public_cover_role": "seer",
        "perceived_image": "其他人可能把我当作激进的预言家",
        "deception_plan": "把2号描述成狼，但不要把主观判断写成私有事实",
        "team_plan": None,
    }


def _belief_patch(
    seat: int,
    *,
    wolf_probability: float = 0.2,
    confidence: float = 0.7,
    evidence: str = "public evidence",
) -> dict:
    return {
        "seat": seat,
        "wolf_probability": wolf_probability,
        "likely_role": "villager",
        "confidence": confidence,
        "evidence": [evidence],
    }


def test_private_belief_and_public_claim_can_intentionally_diverge() -> None:
    state = PrivateAgentState(owner_seat=1, owner_role="werewolf")
    state.apply_model_update(
        _update(wolf_probability=0.05),
        visible_seats={1, 2, 3, 4, 5, 6},
        day=1,
        phase="day",
    )
    state.record_public_commitment(
        day=1,
        phase="day",
        kind="speech",
        text="我是预言家，昨晚查验2号是狼人。",
        claim={"role": "seer", "checked_seat": 2, "result": "wolf"},
    )

    snapshot = state.snapshot()

    assert snapshot["beliefs"]["2"]["wolf_probability"] == 0.05
    assert snapshot["commitments"][0]["claim"]["result"] == "wolf"
    assert snapshot["deception_plan"]


def test_public_commitment_keeps_exact_accepted_text() -> None:
    state = PrivateAgentState(owner_seat=1, owner_role="werewolf")
    statement = "  我是好人，先观察。\n"

    state.record_public_commitment(
        day=1,
        phase="day",
        kind="speech",
        text=statement,
    )

    assert state.snapshot()["commitments"][0]["text"] == statement


def test_two_agents_never_share_private_state_or_nested_values() -> None:
    first = PrivateAgentState(owner_seat=1, owner_role="werewolf")
    second = PrivateAgentState(owner_seat=2, owner_role="werewolf")
    raw = _update(seat=3, wolf_probability=0.8)

    first.apply_model_update(raw, visible_seats={1, 2, 3}, day=1, phase="day")
    first.record_public_commitment(
        day=1,
        phase="day",
        kind="speech",
        text="3号很可疑",
        claim={"role": "villager"},
    )

    assert first is not second
    assert first.snapshot()["beliefs"]
    assert second.snapshot()["beliefs"] == {}
    assert second.snapshot()["commitments"] == []


def test_private_state_deep_copies_ingress_and_egress() -> None:
    state = PrivateAgentState(owner_seat=1, owner_role="villager")
    raw = _update()
    original = deepcopy(raw)
    state.apply_model_update(raw, visible_seats={1, 2, 3}, day=1, phase="day")
    claim = {"role": "seer", "checked_seat": 2, "result": "wolf"}
    state.record_public_commitment(
        day=1,
        phase="day",
        kind="speech",
        text="claim",
        claim=claim,
    )

    raw["beliefs"][0]["evidence"].append("mutated ingress")
    claim["result"] = "village"
    detached = state.snapshot()
    detached["beliefs"]["2"]["evidence"].append("mutated egress")
    detached["commitments"][0]["claim"]["result"] = "village"

    fresh = state.snapshot()
    assert fresh["beliefs"]["2"]["evidence"] == original["beliefs"][0]["evidence"]
    assert fresh["commitments"][0]["claim"]["result"] == "wolf"


def test_batch_belief_update_commits_all_seats_once() -> None:
    state = PrivateAgentState(owner_seat=1, owner_role="villager")
    patches = [
        _belief_patch(2, wolf_probability=0.8, evidence="vote mismatch"),
        _belief_patch(3, wolf_probability=0.1, evidence="consistent claim"),
    ]

    updated = state.update_beliefs(
        patches,
        visible_seats={1, 2, 3},
        day=2,
        phase="voting",
    )
    patches[0]["evidence"].append("mutated after commit")

    snapshot = state.snapshot()
    assert updated == (2, 3)
    assert snapshot["revision"] == 1
    assert snapshot["beliefs"]["2"]["evidence"] == ["vote mismatch"]
    assert snapshot["beliefs"]["3"]["evidence"] == ["consistent claim"]
    assert snapshot["beliefs"]["2"]["updated_day"] == 2
    assert snapshot["beliefs"]["3"]["updated_phase"] == "voting"


def test_batch_belief_update_remains_owned_by_one_agent() -> None:
    first = PrivateAgentState(owner_seat=1, owner_role="villager")
    second = PrivateAgentState(owner_seat=2, owner_role="villager")

    first.update_beliefs(
        [_belief_patch(3, wolf_probability=0.75)],
        visible_seats={1, 2, 3},
        day=1,
        phase="day",
    )

    assert first.snapshot()["revision"] == 1
    assert first.snapshot()["beliefs"]["3"]["wolf_probability"] == 0.75
    assert second.snapshot()["revision"] == 0
    assert second.snapshot()["beliefs"] == {}


@pytest.mark.parametrize(
    ("patches", "fact_kwargs"),
    [
        (
            [
                _belief_patch(3),
                _belief_patch(4, confidence=2.0),
            ],
            {},
        ),
        (
            [
                _belief_patch(3),
                _belief_patch(3, wolf_probability=0.9),
            ],
            {},
        ),
        (
            [
                _belief_patch(3),
                _belief_patch(9),
            ],
            {},
        ),
        (
            [_belief_patch(3)],
            {"known_wolf_seats": {2}, "known_village_seats": {2}},
        ),
        (
            [_belief_patch(2, wolf_probability=0.2)],
            {"known_wolf_seats": {2}},
        ),
        (
            [_belief_patch(2, wolf_probability=0.8)],
            {"known_village_seats": {2}},
        ),
    ],
    ids=(
        "invalid-later-patch",
        "duplicate-seat",
        "hidden-seat",
        "conflicting-facts",
        "known-wolf-contradiction",
        "known-village-contradiction",
    ),
)
def test_batch_belief_update_rolls_back_every_invalid_transaction(
    patches: list[dict],
    fact_kwargs: dict,
) -> None:
    state = PrivateAgentState(owner_seat=1, owner_role="villager")
    state.update_belief(
        _belief_patch(2, wolf_probability=0.6, evidence="initial"),
        visible_seats={1, 2, 3, 4},
        day=1,
        phase="day",
    )
    before = state.snapshot()
    before_digest = state.digest()

    with pytest.raises(ValueError):
        state.update_beliefs(
            patches,
            visible_seats={1, 2, 3, 4},
            day=2,
            phase="voting",
            **fact_kwargs,
        )

    assert state.snapshot() == before
    assert state.digest() == before_digest


def test_private_state_ignores_self_and_unobserved_seats() -> None:
    state = PrivateAgentState(owner_seat=1, owner_role="villager")
    raw = _update(seat=1)
    raw["beliefs"].append({
        "seat": 9,
        "wolf_probability": 0.9,
        "likely_role": "werewolf",
        "confidence": 0.8,
        "evidence": ["not visible"],
    })

    state.apply_model_update(raw, visible_seats={1, 2, 3}, day=1, phase="day")

    beliefs = state.snapshot()["beliefs"]
    assert set(beliefs) == {"2", "3"}
    assert "1" not in beliefs and "9" not in beliefs


def test_private_state_requires_distinct_strategy_candidates() -> None:
    raw = _update()
    raw["candidate_plans"] = ["same plan", " same   plan "]

    with pytest.raises(ValidationError):
        PrivateAgentState.validate_model_update(raw)


def test_set_plan_late_validation_is_atomic() -> None:
    state = PrivateAgentState(owner_seat=1, owner_role="villager")
    state.set_plan(
        selected_plan="initial",
        candidate_plans=["observe", "accuse"],
        perceived_image="quiet village",
        deception_plan="hold back",
    )
    before = state.snapshot()
    before_digest = state.digest()

    with pytest.raises(ValueError, match="perceived_image"):
        state.set_plan(
            selected_plan="replacement",
            candidate_plans=["new a", "new b"],
            perceived_image="   ",
            deception_plan="changed",
        )

    assert state.snapshot() == before
    assert state.digest() == before_digest


def test_prompt_render_labels_beliefs_as_subjective_and_commitments_as_public() -> None:
    state = PrivateAgentState(owner_seat=1, owner_role="werewolf")
    state.apply_model_update(_update(), visible_seats={1, 2}, day=1, phase="day")
    state.record_public_commitment(
        day=1,
        phase="day",
        kind="speech",
        text="我是预言家",
        claim={"role": "seer"},
    )

    rendered = state.render_for_prompt()

    assert "私有主观状态" in rendered
    assert "不是环境真值" in rendered
    assert "此前已经公开说过" in rendered
    assert "我是预言家" in rendered
    assert len(state.digest()) == 64


def test_role_count_and_private_checks_constrain_subjective_marginals() -> None:
    state = PrivateAgentState(owner_seat=1, owner_role="villager")
    raw = _update(seat=2, wolf_probability=0.2)
    raw["beliefs"].extend([
        {
            "seat": 3,
            "wolf_probability": 0.9,
            "likely_role": "werewolf",
            "confidence": 0.8,
            "evidence": ["public behavior"],
        },
        {
            "seat": 4,
            "wolf_probability": 0.8,
            "likely_role": "werewolf",
            "confidence": 0.7,
            "evidence": ["public behavior"],
        },
    ])

    state.apply_model_update(
        raw,
        visible_seats={1, 2, 3, 4, 5, 6},
        known_wolf_seats={2},
        known_village_seats={3},
        total_wolves=2,
        day=2,
        phase="day",
    )

    beliefs = state.snapshot()["beliefs"]
    assert beliefs["2"]["wolf_probability"] == 1.0
    assert beliefs["3"]["wolf_probability"] == 0.0
    assert sum(item["wolf_probability"] for item in beliefs.values()) == pytest.approx(2.0)
    assert all(0.0 <= item["wolf_probability"] <= 1.0 for item in beliefs.values())
