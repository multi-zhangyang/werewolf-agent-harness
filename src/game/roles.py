"""Role and team definitions for the Werewolf game domain."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
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


CLASSIC_RULESET_ID = "classic.v1"
SUPPORTED_RULESET_IDS = frozenset({CLASSIC_RULESET_ID})

# This explicit roster keeps adding an enum value from silently advertising an
# ability that the classic orchestrator and rules engine do not implement.
CLASSIC_V1_IMPLEMENTED_ROLES = frozenset({
    Role.VILLAGER,
    Role.WEREWOLF,
    Role.SEER,
    Role.DOCTOR,
    Role.WITCH,
    Role.GUARD,
    Role.HUNTER,
})

# Classic v1 uses single-card power roles; Witch and Guard also have per-game
# singleton state. Wolves and ordinary villagers are the repeatable cards.
CLASSIC_V1_SINGLETON_ROLES = frozenset({
    Role.SEER,
    Role.DOCTOR,
    Role.WITCH,
    Role.GUARD,
    Role.HUNTER,
})


def role_team(role: Role) -> Team:
    """Return the team that a role belongs to."""

    return ROLE_TEAM[role]


def validate_ruleset_id(ruleset_id: str) -> str:
    """Return the canonical supported ruleset ID or fail closed."""

    normalized = str(ruleset_id).strip()
    if normalized not in SUPPORTED_RULESET_IDS:
        raise ValueError(f"unsupported Werewolf ruleset: {normalized!r}")
    return normalized


def validate_role_deck(
    deck: Iterable[Role | str],
    *,
    player_count: int,
    ruleset_id: str = CLASSIC_RULESET_ID,
) -> list[Role]:
    """Validate and normalize a playable deck for one implemented ruleset."""

    ruleset_id = validate_ruleset_id(ruleset_id)
    if not 6 <= player_count <= 12:
        raise ValueError(f"{ruleset_id} supports 6-12 players")

    raw_deck = list(deck)
    if len(raw_deck) != player_count:
        raise ValueError("role deck size must match player count")

    roles: list[Role] = []
    for value in raw_deck:
        try:
            roles.append(value if isinstance(value, Role) else Role(value))
        except (TypeError, ValueError) as err:
            raise ValueError(f"role deck contains an unknown role: {value!r}") from err

    unsupported = sorted(
        (role.value for role in set(roles) - CLASSIC_V1_IMPLEMENTED_ROLES),
    )
    if unsupported:
        raise ValueError(
            f"{ruleset_id} has no implemented capability for roles: {','.join(unsupported)}"
        )

    counts = Counter(roles)
    if counts[Role.WEREWOLF] < 1:
        raise ValueError("role deck must contain at least one werewolf")
    if counts[Role.WEREWOLF] == len(roles):
        raise ValueError("role deck must contain at least one non-werewolf")

    duplicates = sorted(
        role.value
        for role in CLASSIC_V1_SINGLETON_ROLES
        if counts[role] > 1
    )
    if duplicates:
        raise ValueError(
            f"{ruleset_id} allows at most one of each power role: {','.join(duplicates)}"
        )
    return roles


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
    return validate_role_deck(deck, player_count=player_count)
