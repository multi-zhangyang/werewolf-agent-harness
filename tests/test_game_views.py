from __future__ import annotations

import json

from src.agent.information import build_observation
from src.game.models import Event, EventVisibility, Phase
from src.game.roles import Role
from src.game.state import new_game


def test_public_view_hides_role_resource_state() -> None:
    state = new_game(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    state.witch_antidote = False
    state.witch_poison = False
    state.last_guarded_seat = 4
    state.pending_hunter = [state.players[0].id]

    view = state.public_view()

    assert "witch_antidote" not in view
    assert "witch_poison" not in view
    assert "last_guarded_seat" not in view
    assert "pending_hunter" not in view


def test_private_view_exposes_only_own_role_resource_state() -> None:
    state = new_game(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    roles = [
        Role.WITCH,
        Role.GUARD,
        Role.HUNTER,
        Role.VILLAGER,
        Role.VILLAGER,
        Role.VILLAGER,
        Role.SEER,
        Role.WEREWOLF,
        Role.WEREWOLF,
    ]
    for player, role in zip(state.players, roles, strict=True):
        player.role = role
    state.witch_antidote = False
    state.witch_poison = True
    state.last_guarded_seat = 5
    state.pending_hunter = [state.players[2].id]

    witch_view = state.private_view_for(state.players[0].id)
    guard_view = state.private_view_for(state.players[1].id)
    hunter_view = state.private_view_for(state.players[2].id)
    villager_view = state.private_view_for(state.players[3].id)

    assert witch_view["role_state"] == {
        "witch_antidote": False,
        "witch_poison": True,
    }
    assert guard_view["role_state"] == {"last_guarded_seat": 5}
    assert hunter_view["role_state"] == {"pending_hunter": True}
    assert "role_state" not in villager_view

    for view in (witch_view, guard_view, hunter_view, villager_view):
        assert "witch_antidote" not in view
        assert "witch_poison" not in view
        assert "last_guarded_seat" not in view
        assert "pending_hunter" not in view


def test_game_views_and_agent_observation_strip_nested_model_reasoning() -> None:
    state = new_game(["A", "B", "C", "D", "E", "F"])
    viewer = state.players[0]
    marker = "model-private-reasoning-sentinel"
    state.events.extend([
        Event(
            phase=Phase.DAY,
            day=1,
            type="speech",
            message="public statement",
            payload={
                "reasoning": marker,
                "nested": {
                    "thought": marker,
                    "items": [{"private_reasoning": marker, "visible": "kept"}],
                },
            },
        ),
        Event(
            phase=Phase.NIGHT,
            day=1,
            type="seer_result",
            message="private result",
            visibility=EventVisibility.PRIVATE,
            recipients=[viewer.id],
            payload={
                "target_seat": 2,
                "nested": {
                    "reasoning": marker,
                    "thought": marker,
                    "private_reasoning": marker,
                    "visible": "kept",
                },
            },
        ),
    ])

    projected = (
        state.public_view(),
        state.private_view_for(viewer.id),
        build_observation(state, viewer.id).model_dump(),
    )
    for value in projected:
        serialized = json.dumps(value, ensure_ascii=False, default=str)
        assert marker not in serialized
        for field in ("reasoning", "thought", "private_reasoning"):
            assert f'"{field}"' not in serialized

    # Projection is read-only: trusted server state still has its original
    # malformed payload, while no player-facing view can observe it.
    assert state.events[0].payload["reasoning"] == marker
    assert state.events[1].payload["nested"]["private_reasoning"] == marker
