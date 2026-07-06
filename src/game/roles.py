"""Role and team definitions for the Werewolf game domain."""
from __future__ import annotations

from enum import StrEnum


class Team(StrEnum):
    """Win-condition team."""

    VILLAGE = "village"
    WEREWOLVES = "werewolves"


class Role(StrEnum):
    """Supported roles."""

    VILLAGER = "villager"
    WEREWOLF = "werewolf"
    SEER = "seer"
    DOCTOR = "doctor"
    WITCH = "witch"
    GUARD = "guard"
    HUNTER = "hunter"


ROLE_TEAM: dict[Role, Team] = {
    Role.VILLAGER: Team.VILLAGE,
    Role.SEER: Team.VILLAGE,
    Role.DOCTOR: Team.VILLAGE,
    Role.WITCH: Team.VILLAGE,
    Role.GUARD: Team.VILLAGE,
    Role.HUNTER: Team.VILLAGE,
    Role.WEREWOLF: Team.WEREWOLVES,
}


def role_team(role: Role) -> Team:
    """Return the team that a role belongs to."""

    return ROLE_TEAM[role]


def default_role_deck(player_count: int, *, include_hunter: bool | None = None) -> list[Role]:
    """Build a balanced classic deck for 6-12 players.

    Village power roles scale with player count to keep wolves ~1/3 of seats:
    - Seer always present (village's information backbone).
    - Witch (save+poison) from 7 players.
    - Guard from 9 players.
    - Hunter from 8 players.
    Wolves: 2 (≤8), 3 (≤11), 4 (12).
    """

    if player_count < 6 or player_count > 12:
        raise ValueError("classic supports 6-12 players")

    wolf_count = 2 if player_count <= 8 else 3 if player_count <= 11 else 4
    deck: list[Role] = [Role.WEREWOLF] * wolf_count + [Role.SEER]
    if player_count >= 7:
        deck.append(Role.WITCH)
    if player_count >= 9:
        deck.append(Role.GUARD)
    use_hunter = player_count >= 8 if include_hunter is None else include_hunter
    if use_hunter:
        deck.append(Role.HUNTER)

    villager_count = player_count - len(deck)
    if villager_count < 0:
        raise ValueError("role deck has more special roles than players")
    deck.extend([Role.VILLAGER] * villager_count)
    return deck
