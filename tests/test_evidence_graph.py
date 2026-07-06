"""Visible EvidenceGraph / RolePosterior tests.

These tests lock the privacy boundary: the graph may structure visible public
evidence and the viewer's own private seer result, but it must not consume
hidden role truth, wolf caucus, deception strategy, or private reasoning.
"""
from __future__ import annotations

import json

import sys

sys.path.insert(0, ".")

from src.agent.evidence import build_evidence_graph
from src.agent.information import attach_today_speeches, build_observation
from src.agent.prompts import render_observation
from src.agent.schemas import AgentObservation
from src.game.models import Event, EventVisibility, Phase
from src.game.roles import Role
from src.game.state import new_game


def _contains_subset(items: list[dict], expected: dict) -> bool:
    return any(all(item.get(k) == v for k, v in expected.items()) for item in items)


def _obs(**overrides) -> AgentObservation:
    base = {
        "my_seat": 1,
        "my_role": "villager",
        "my_team": "village",
        "my_teammates": [],
        "seats": [
            {"seat": i, "id": f"p{i}", "name": f"P{i}", "alive": True}
            for i in range(1, 7)
        ],
        "alive_seats": [1, 2, 3, 4, 5, 6],
        "phase": "day",
        "day": 1,
        "public_events": [],
        "private_events": [],
        "today_speeches": [],
        "available_actions": [],
        "candidate_targets": [2, 3, 4, 5, 6],
        "vote_targets": [],
        "in_pk": False,
    }
    base.update(overrides)
    obs = AgentObservation(**base)
    obs.evidence_graph = build_evidence_graph(obs)
    return obs


def test_claims_and_conflicts_build_role_posterior_without_truth():
    obs = _obs(today_speeches=[
        {
            "seat": 2,
            "name": "P2",
            "text": "我跳预言家,验5号是狼。",
            "claim": {"role": "seer", "checked_seat": 5, "result": "wolf"},
            "day": 1,
        },
        {
            "seat": 3,
            "name": "P3",
            "text": "我才是预言家,5号是好人。",
            "claim": {"role": "seer", "checked_seat": 5, "result": "village"},
            "day": 1,
        },
    ])

    graph = obs.evidence_graph
    assert len(graph["claims"]) == 2
    assert {c["type"] for c in graph["claim_conflicts"]} >= {
        "seer_counterclaim",
        "seer_result_conflict",
    }
    assert graph["role_posterior"]["2"]["seer_claim"] is True
    assert graph["role_posterior"]["3"]["seer_claim"] is True
    assert all(c.get("evidence_id", "").startswith("claim:") for c in graph["claims"])
    assert all(c.get("evidence_id", "").startswith("claim_conflict:") for c in graph["claim_conflicts"])
    # Conflict raises suspicion for both claimers, but does not know who is true.
    assert graph["role_posterior"]["2"]["werewolf_suspicion"] > 0.5
    assert graph["role_posterior"]["3"]["werewolf_suspicion"] > 0.5
    assert graph["role_posterior"]["5"]["posterior_deltas"][0]["evidence_id"].startswith("claim:")
    assert all("target_seat" in d and "before" in d and "after" in d for d in graph["posterior_deltas"])
    item_ids = {item["evidence_id"] for item in graph["evidence_items"]}
    assert item_ids >= {claim["evidence_id"] for claim in graph["claims"]}
    assert item_ids >= {conflict["evidence_id"] for conflict in graph["claim_conflicts"]}
    assert {delta["evidence_id"] for delta in graph["posterior_deltas"]} <= item_ids
    assert all(
        {"evidence_id", "type", "visibility", "provenance", "confidence"} <= set(item)
        for item in graph["evidence_items"]
    )


def test_attitude_and_vote_edges_are_visible_evidence():
    obs = _obs(
        today_speeches=[
            {
                "seat": 4,
                "name": "P4",
                "text": "2号在带节奏,3号这轮说得像好人。",
                "accuses": [2],
                "attitudes": {2: "oppose", 3: "support"},
                "day": 1,
            }
        ],
        public_events=[
            {
                "type": "vote_cast",
                "day": 1,
                "message": "P5 voted for P2.",
                "payload": {"voter_seat": 5, "target_seat": 2},
            }
        ],
    )

    graph = obs.evidence_graph
    assert _contains_subset(
        graph["attitude_edges"],
        {"source": 4, "target": 2, "stance": "oppose", "source_type": "accuse", "day": 1},
    )
    assert _contains_subset(
        graph["attitude_edges"],
        {"source": 4, "target": 3, "stance": "support", "source_type": "attitude", "day": 1},
    )
    assert _contains_subset(
        graph["vote_edges"],
        {"source": 5, "target": 2, "stance": "oppose", "source_type": "vote", "day": 1},
    )
    assert graph["attitude_edges"][0]["evidence_id"]
    assert graph["vote_edges"][0]["evidence_id"].startswith("vote:")
    assert _contains_subset(
        graph["evidence_items"],
        {
            "evidence_id": graph["attitude_edges"][0]["evidence_id"],
            "visibility": "public",
            "provenance": "today_speech",
        },
    )
    assert _contains_subset(
        graph["evidence_items"],
        {
            "evidence_id": graph["vote_edges"][0]["evidence_id"],
            "visibility": "public",
            "provenance": "public_event",
        },
    )
    assert graph["role_posterior"]["2"]["werewolf_suspicion"] > 0.5
    assert graph["role_posterior"]["3"]["werewolf_suspicion"] < 0.5


def test_private_seer_result_only_reaches_that_viewer():
    state = new_game(["P1", "P2", "P3", "P4", "P5", "P6"])
    roles = [Role.SEER, Role.WEREWOLF, Role.WEREWOLF, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER]
    for player, role in zip(state.players, roles, strict=True):
        player.role = role
    state.phase = Phase.DAY
    state.day = 1
    seer = state.players[0]
    wolf = state.players[1]
    villager = state.players[3]
    state.events.append(
        Event(
            phase=Phase.NIGHT,
            day=1,
            type="seer_result",
            message=f"你查验了 {wolf.name}({wolf.seat}号),结果:狼人",
            visibility=EventVisibility.PRIVATE,
            recipients=[seer.id],
            payload={"target_id": wolf.id, "target_seat": wolf.seat, "team": "werewolves"},
        )
    )

    seer_obs = build_observation(state, seer.id)
    villager_obs = build_observation(state, villager.id)

    assert _contains_subset(
        seer_obs.evidence_graph["private_results"],
        {"target": 2, "result": "wolf", "day": 1, "source": "private_seer_result"},
    )
    assert seer_obs.evidence_graph["private_results"][0]["evidence_id"].startswith("private_seer_result:")
    assert _contains_subset(
        seer_obs.evidence_graph["evidence_items"],
        {
            "evidence_id": seer_obs.evidence_graph["private_results"][0]["evidence_id"],
            "visibility": "private",
            "provenance": "private_event",
            "target_seat": 2,
        },
    )
    assert seer_obs.evidence_graph["role_posterior"]["2"]["werewolf_suspicion"] > 0.9
    assert villager_obs.evidence_graph["private_results"] == []
    assert villager_obs.evidence_graph["role_posterior"]["2"]["werewolf_suspicion"] == 0.5
    assert all(item["visibility"] == "public" for item in villager_obs.evidence_graph["evidence_items"])


def test_public_seer_result_is_not_treated_as_private_hard_evidence():
    state = new_game(["P1", "P2", "P3", "P4", "P5", "P6"])
    roles = [Role.VILLAGER, Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.VILLAGER, Role.VILLAGER]
    for player, role in zip(state.players, roles, strict=True):
        player.role = role
    state.phase = Phase.DAY
    state.day = 1
    state.events.append(
        Event(
            phase=Phase.NIGHT,
            day=1,
            type="seer_result",
            message="错误标记为公开的查验结果不应成为私有硬信息",
            visibility=EventVisibility.PUBLIC,
            payload={"target_seat": 2, "team": "werewolves"},
        )
    )

    obs = build_observation(state, state.players[0].id)

    assert obs.evidence_graph["private_results"] == []
    assert all(item["visibility"] == "public" for item in obs.evidence_graph["evidence_items"])
    assert obs.evidence_graph["role_posterior"]["2"]["werewolf_suspicion"] == 0.5


def test_legal_worlds_use_public_deck_and_viewer_facts_not_truth():
    state = new_game(["P1", "P2", "P3", "P4", "P5", "P6"])
    roles = [Role.VILLAGER, Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.VILLAGER, Role.VILLAGER]
    for player, role in zip(state.players, roles, strict=True):
        player.role = role
    state.phase = Phase.DAY
    state.day = 1

    obs = build_observation(state, state.players[0].id)
    graph = obs.evidence_graph

    assert graph["role_counts"] == {"seer": 1, "villager": 3, "werewolf": 2}
    assert graph["legal_worlds"]["wolf_count"] == 2
    assert graph["legal_worlds"]["known_wolves"] == []
    assert graph["legal_worlds"]["known_villagers"] == [1]
    assert graph["legal_worlds"]["world_count"] == 10
    assert graph["role_posterior"]["1"]["constrained_werewolf_suspicion"] == 0.0
    assert graph["role_posterior"]["2"]["constrained_werewolf_suspicion"] == 0.4
    assert graph["role_posterior"]["3"]["constrained_werewolf_suspicion"] == 0.4


def test_legal_worlds_apply_private_seer_result_only_for_viewer():
    state = new_game(["P1", "P2", "P3", "P4", "P5", "P6"])
    roles = [Role.SEER, Role.WEREWOLF, Role.WEREWOLF, Role.VILLAGER, Role.VILLAGER, Role.VILLAGER]
    for player, role in zip(state.players, roles, strict=True):
        player.role = role
    state.phase = Phase.DAY
    state.day = 1
    seer = state.players[0]
    wolf = state.players[1]
    villager = state.players[3]
    state.events.append(
        Event(
            phase=Phase.NIGHT,
            day=1,
            type="seer_result",
            message=f"你查验了 {wolf.name}({wolf.seat}号),结果:狼人",
            visibility=EventVisibility.PRIVATE,
            recipients=[seer.id],
            payload={"target_id": wolf.id, "target_seat": wolf.seat, "team": "werewolves"},
        )
    )

    seer_obs = build_observation(state, seer.id)
    villager_obs = build_observation(state, villager.id)

    assert seer_obs.evidence_graph["legal_worlds"]["known_wolves"] == [2]
    assert seer_obs.evidence_graph["legal_worlds"]["known_villagers"] == [1]
    assert seer_obs.evidence_graph["legal_worlds"]["world_count"] == 4
    assert seer_obs.evidence_graph["role_posterior"]["2"]["constrained_werewolf_suspicion"] == 1.0
    assert villager_obs.evidence_graph["legal_worlds"]["known_wolves"] == []
    assert villager_obs.evidence_graph["legal_worlds"]["known_villagers"] == [4]
    assert villager_obs.evidence_graph["legal_worlds"]["world_count"] == 10


def test_legal_worlds_for_werewolf_include_known_teammate_without_other_truth():
    state = new_game(["P1", "P2", "P3", "P4", "P5", "P6"])
    roles = [Role.VILLAGER, Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.VILLAGER, Role.VILLAGER]
    for player, role in zip(state.players, roles, strict=True):
        player.role = role
    state.phase = Phase.DAY
    state.day = 1

    obs = build_observation(state, state.players[1].id)
    graph = obs.evidence_graph

    assert graph["legal_worlds"]["known_wolves"] == [2, 3]
    assert graph["legal_worlds"]["world_count"] == 1
    assert graph["role_posterior"]["2"]["constrained_werewolf_suspicion"] == 1.0
    assert graph["role_posterior"]["3"]["constrained_werewolf_suspicion"] == 1.0
    assert graph["role_posterior"]["1"]["constrained_werewolf_suspicion"] == 0.0


def test_non_wolf_observation_ignores_teammates_field_defensively():
    obs = _obs(my_role="villager", my_team="village", my_teammates=[{"seat": 2, "name": "P2"}])
    graph = obs.evidence_graph

    assert graph["legal_worlds"]["known_wolves"] == []
    assert graph["legal_worlds"]["known_villagers"] == [1]
    assert graph["role_posterior"]["2"]["constrained_werewolf_suspicion"] == 0.4


def test_impossible_legal_worlds_do_not_emit_fake_zero_constrained_posterior():
    obs = _obs(
        my_role="seer",
        my_team="village",
        role_counts={"werewolf": 1, "seer": 1, "villager": 4},
        private_events=[
            {
                "type": "seer_result",
                "visibility": "private",
                "phase": "night",
                "day": 1,
                "payload": {"target_seat": 2, "team": "werewolves"},
            },
            {
                "type": "seer_result",
                "visibility": "private",
                "phase": "night",
                "day": 2,
                "payload": {"target_seat": 3, "team": "werewolves"},
            },
        ],
    )
    graph = obs.evidence_graph

    assert graph["legal_worlds"]["is_contradictory"] is True
    assert graph["legal_worlds"]["world_count"] == 0
    assert all("constrained_werewolf_suspicion" not in data for data in graph["role_posterior"].values())


def test_role_posterior_does_not_initialize_from_true_roles():
    state = new_game(["P1", "P2", "P3", "P4", "P5", "P6"])
    roles = [Role.VILLAGER, Role.WEREWOLF, Role.WEREWOLF, Role.SEER, Role.VILLAGER, Role.VILLAGER]
    for player, role in zip(state.players, roles, strict=True):
        player.role = role
    state.phase = Phase.DAY
    state.day = 1

    obs = build_observation(state, state.players[0].id)
    posterior = obs.evidence_graph["role_posterior"]

    assert posterior["2"]["werewolf_suspicion"] == 0.5
    assert posterior["3"]["werewolf_suspicion"] == 0.5
    assert posterior["4"]["werewolf_suspicion"] == 0.5


def test_attach_today_speeches_sanitizes_hidden_fields_and_rebuilds_graph():
    obs = _obs()
    attach_today_speeches(obs, [
        {
            "seat": 2,
            "name": "P2",
            "text": "我跳预言家,验5号是狼。",
            "claim": {
                "role": "seer",
                "checked_seat": 5,
                "result": "wolf",
                "reasoning": "hidden-claim-reasoning",
            },
            "accuses": [5],
            "attitudes": {5: "oppose"},
            "deception": "fabrication",
            "reasoning": "hidden-wolf-reasoning",
            "wolf_caucus": "hidden-caucus",
            "role": "hidden-role-werewolf",
            "team": "hidden-team",
            "day": 1,
        }
    ])

    speech = obs.today_speeches[0]
    assert "deception" not in speech
    assert "reasoning" not in speech
    assert "wolf_caucus" not in speech
    assert "role" not in speech
    assert "team" not in speech
    assert speech["claim"] == {"role": "seer", "checked_seat": 5, "result": "wolf"}
    assert obs.evidence_graph["claims"][0]["claimer"] == 2

    dumped = json.dumps(obs.evidence_graph, ensure_ascii=False)
    assert "fabrication" not in dumped
    assert "hidden-wolf-reasoning" not in dumped
    assert "hidden-caucus" not in dumped
    assert "hidden-role-werewolf" not in dumped


def test_render_observation_includes_evidence_block_without_hidden_fields():
    obs = _obs()
    attach_today_speeches(obs, [
        {
            "seat": 2,
            "name": "P2",
            "text": "我跳预言家,验5号是狼。",
            "claim": {"role": "seer", "checked_seat": 5, "result": "wolf"},
            "accuses": [5],
            "deception": "fabrication",
            "reasoning": "hidden-wolf-reasoning",
            "wolf_caucus": "hidden-caucus",
            "day": 1,
        }
    ])

    rendered = render_observation(obs, "(尚无记忆)")

    assert "【公开证据图 / 角色后验(非真值)】" in rendered
    assert "2号声称seer,查5号=狼人" in rendered
    assert "fabrication" not in rendered
    assert "hidden-wolf-reasoning" not in rendered
    assert "hidden-caucus" not in rendered
