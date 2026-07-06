"""Convenience state constructors and view helpers."""
from __future__ import annotations

from typing import Any

from .models import GameState
from .rules import RulesEngine


def new_game(player_names: list[str], *, game_id: str | None = None) -> GameState:
    """Create an undealt game in setup phase."""

    return RulesEngine.create_game(player_names, game_id=game_id)


def public_state(state: GameState) -> dict[str, Any]:
    """Return a public-safe state view."""

    return state.public_view()


def private_state(state: GameState, player_id: str) -> dict[str, Any]:
    """Return a player-specific state view with hidden data only for that player."""

    return state.private_view_for(player_id)
