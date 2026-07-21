"""Reproducible scheduling and experimental-control provenance."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from ..agent.prompts import PERSONAS
from ..game.orchestrator import TURN_POLICIES


ROLE_LAYOUT_MODES = ("legacy", "fixed", "counterbalanced")
PERSONA_MODES = ("legacy", "fixed", "randomized", "counterbalanced")
PERSONA_CATALOG_VERSION = "werewolf.persona-catalog.v1"
ROLE_LAYOUT_SCHEMA_VERSION = "werewolf.role-layout.v1"

_PERSONA_PROFILE_IDS = (
    "observe_wait",
    "direct_confrontation",
    "patient_disguise",
    "alliance_builder",
    "anti_pattern",
    "consistency_auditor",
)
if len(_PERSONA_PROFILE_IDS) != len(PERSONAS):
    raise RuntimeError("harness persona catalog does not match the runtime catalog")
_PERSONA_CATALOG = {
    profile_id: {
        "profile_id": profile_id,
        "name": str(profile[0]),
        "description": str(profile[1]),
    }
    for profile_id, profile in zip(_PERSONA_PROFILE_IDS, PERSONAS, strict=True)
}


def persona_catalog() -> list[dict[str, str]]:
    """Return the exact, prompt-bearing persona catalog used by the harness."""

    return [dict(_PERSONA_CATALOG[profile_id]) for profile_id in _PERSONA_PROFILE_IDS]


def persona_profile_ids() -> tuple[str, ...]:
    return _PERSONA_PROFILE_IDS


def default_experiment_id(policies: list[str], *, policy_order: str) -> str:
    return f"{policy_order}:" + ",".join(policies)


def build_policy_schedule(
    n_games: int,
    policies: list[str],
    *,
    policy_order: str,
    seed: int | None,
    experiment_id: str,
    seat_count: int | None = None,
    seat_permutation: str = "fixed",
    role_layout_mode: str = "legacy",
    role_layout_seed: int | None = None,
    role_layout_count: int | None = None,
    persona_mode: str = "legacy",
    persona_seed: int | None = None,
    persona_profiles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build a reproducible turn-policy schedule.

    `n_games` means games per policy. In ABBA mode it must be even and produces
    paired cases: A/B on the first seed, then B/A on the second seed.
    """
    if n_games < 0:
        raise ValueError("n_games must be non-negative")
    if not policies:
        raise ValueError("at least one turn policy is required")
    invalid = [policy for policy in policies if policy not in TURN_POLICIES]
    if invalid:
        raise ValueError(f"unknown turn policies: {invalid}")
    if policy_order not in {"sequential", "abba"}:
        raise ValueError("policy_order must be 'sequential' or 'abba'")
    if policy_order == "abba" and len(policies) != 2:
        raise ValueError("abba policy_order requires exactly two turn policies")
    if policy_order == "abba" and n_games % 2 != 0:
        raise ValueError("abba policy_order requires an even number of runs per policy")
    if len(policies) > 1 and seed is None:
        raise ValueError("multi-policy experiments require --seed/--experiment-seed for reproducible paired cases")
    if seat_permutation not in {"fixed", "cyclic"}:
        raise ValueError("seat_permutation must be 'fixed' or 'cyclic'")
    if seat_permutation == "cyclic" and (seat_count is None or seat_count < 1):
        raise ValueError("cyclic seat_permutation requires a positive seat_count")
    if role_layout_mode not in ROLE_LAYOUT_MODES:
        raise ValueError(f"role_layout_mode must be one of {ROLE_LAYOUT_MODES}")
    if persona_mode not in PERSONA_MODES:
        raise ValueError(f"persona_mode must be one of {PERSONA_MODES}")
    if role_layout_seed is not None and type(role_layout_seed) is not int:
        raise ValueError("role_layout_seed must be an integer")
    if persona_seed is not None and type(persona_seed) is not int:
        raise ValueError("persona_seed must be an integer")
    if role_layout_count is not None and (
        type(role_layout_count) is not int or role_layout_count < 1
    ):
        raise ValueError("role_layout_count must be a positive integer")
    if role_layout_mode == "legacy" and (
        role_layout_seed is not None or role_layout_count is not None
    ):
        raise ValueError(
            "role_layout_seed/count require fixed or counterbalanced role_layout_mode"
        )
    if role_layout_mode == "fixed" and role_layout_count not in {None, 1}:
        raise ValueError("fixed role_layout_mode supports exactly one layout")
    if persona_mode == "legacy" and (
        persona_seed is not None or persona_profiles is not None
    ):
        raise ValueError(
            "persona_seed/profiles require fixed, randomized, or counterbalanced persona_mode"
        )

    normalized_profiles = _normalize_persona_profiles(persona_profiles)
    if persona_mode != "legacy" and seat_count is None:
        raise ValueError("explicit persona_mode requires a positive seat_count")
    role_seed_base = _control_seed(
        role_layout_seed,
        base_seed=seed,
        offset=1,
        label="role layout",
        enabled=role_layout_mode != "legacy",
    )
    persona_seed_base = _control_seed(
        persona_seed,
        base_seed=seed,
        offset=300_001,
        label="persona",
        enabled=persona_mode != "legacy",
    )
    seat_cycle = int(seat_count) if seat_permutation == "cyclic" else 1
    persona_cycle = (
        len(normalized_profiles) if persona_mode == "counterbalanced" else 1
    )
    # These controls must be crossed, not advanced in lockstep. Using an LCM
    # would sample only a diagonal subset when both cycles share factors and
    # would therefore keep identity/seat and persona effects confounded.
    control_cycle = seat_cycle * persona_cycle
    if (
        persona_mode == "counterbalanced"
        and n_games > 0
        and n_games % control_cycle
    ):
        raise ValueError(
            "counterbalanced personas require runs per policy to be a multiple "
            f"of the complete seat-by-persona control cycle ({control_cycle})"
        )
    resolved_role_layout_count: int | None = None
    if role_layout_mode == "fixed":
        resolved_role_layout_count = 1
    elif role_layout_mode == "counterbalanced":
        if n_games == 0:
            resolved_role_layout_count = role_layout_count or 1
        else:
            if n_games % control_cycle:
                raise ValueError(
                    "counterbalanced role layouts require runs per policy to be a "
                    f"multiple of the complete control cycle ({control_cycle})"
                )
            available_blocks = n_games // control_cycle
            resolved_role_layout_count = role_layout_count or available_blocks
            if available_blocks % resolved_role_layout_count:
                raise ValueError(
                    "runs per policy must contain an equal number of complete cycles "
                    "for every requested role layout"
                )

    schedule: list[dict[str, Any]] = []
    per_policy_counts = {policy: 0 for policy in policies}

    def append(
        policy: str,
        *,
        case_idx: int,
        counterbalance_order: str | None = None,
        abba_block_idx: int | None = None,
        abba_position: int | None = None,
    ) -> None:
        per_policy_counts[policy] += 1
        policy_game_idx = per_policy_counts[policy]
        case_seed = seed + case_idx if seed is not None else None
        role_layout_index: int | None = None
        if role_layout_mode == "legacy":
            resolved_role_seed = case_seed
        elif role_layout_mode == "fixed":
            resolved_role_seed = role_seed_base
            role_layout_index = 1
        else:
            assert resolved_role_layout_count is not None
            role_layout_index = (
                ((case_idx - 1) // control_cycle) % resolved_role_layout_count
            ) + 1
            assert role_seed_base is not None
            resolved_role_seed = role_seed_base + role_layout_index - 1
        actor_seed = case_seed + 100_000 if case_seed is not None else None
        orchestrator_seed = case_seed + 200_000 if case_seed is not None else None
        global_idx = len(schedule) + 1
        policy_index = policies.index(policy)
        schedule.append({
            "global_game_idx": global_idx,
            "experiment_id": experiment_id,
            "policy_order": policy_order,
            "policy_set": list(policies),
            "policy_alias": chr(ord("A") + policy_index),
            "policy_index": policy_index,
            "policy_count": len(policies),
            "runs_per_policy": n_games,
            "policy_game_idx": policy_game_idx,
            "base_seed": seed,
            "case_seed": case_seed,
            "role_seed": resolved_role_seed,
            "actor_seed": actor_seed,
            "orchestrator_seed": orchestrator_seed,
            "pair_id": f"pair-{case_idx:04d}" if len(policies) > 1 else None,
            "counterbalance_order": counterbalance_order,
            "abba_block_idx": abba_block_idx,
            "abba_position": abba_position,
            "game_id": f"{experiment_id}-g{global_idx:04d}",
            "turn_policy": policy,
        })
        if role_layout_mode != "legacy":
            assert resolved_role_seed is not None
            assert resolved_role_layout_count is not None
            schedule[-1].update({
                "role_layout_mode": role_layout_mode,
                "role_layout_seed_base": role_seed_base,
                "role_layout_index": role_layout_index,
                "role_layout_count": resolved_role_layout_count,
                "role_layout_control_cycle": control_cycle,
                "role_layout_block_id": (
                    f"role-layout-{int(role_layout_index or 1):04d}"
                ),
            })
        if persona_mode != "legacy":
            assert seat_count is not None
            assert persona_seed_base is not None
            schedule[-1].update(_persona_schedule_fields(
                case_idx=case_idx,
                seat_count=seat_count,
                mode=persona_mode,
                seed_base=persona_seed_base,
                profile_ids=normalized_profiles,
                counterbalance_position=(
                    (case_idx - 1) // seat_cycle
                    if persona_mode == "counterbalanced"
                    else None
                ),
            ))
        if seat_permutation == "cyclic":
            assert seat_count is not None
            rotation = (case_idx - 1) % seat_count
            schedule[-1].update({
                "seat_permutation_mode": "cyclic",
                "seat_rotation": rotation,
                # New seat i receives the player originally in this 1-based seat.
                "seat_permutation": [
                    ((index + rotation) % seat_count) + 1
                    for index in range(seat_count)
                ],
                "permutation_id": f"seat-rotation-{rotation:02d}",
            })

    if policy_order == "sequential" or len(policies) == 1:
        for policy in policies:
            for case_idx in range(1, n_games + 1):
                append(policy, case_idx=case_idx, counterbalance_order="batch")
    else:
        a, b = policies
        for block_idx in range(n_games // 2):
            first_case = block_idx * 2 + 1
            second_case = first_case + 1
            append(a, case_idx=first_case, counterbalance_order="AB", abba_block_idx=block_idx + 1, abba_position=1)
            append(b, case_idx=first_case, counterbalance_order="AB", abba_block_idx=block_idx + 1, abba_position=2)
            append(b, case_idx=second_case, counterbalance_order="BA", abba_block_idx=block_idx + 1, abba_position=3)
            append(a, case_idx=second_case, counterbalance_order="BA", abba_block_idx=block_idx + 1, abba_position=4)
    for row in schedule:
        row["scheduled_total"] = len(schedule)
    return schedule


def experiment_metadata(meta: dict[str, Any], *, player_names: list[str]) -> dict[str, Any]:
    if not meta:
        return {}
    output = {
        "protocol_version": "turn_policy_ablation.v1",
        "experiment_id": meta.get("experiment_id"),
        "policy_order": meta.get("policy_order"),
        "policy_set": meta.get("policy_set"),
        "policy_alias": meta.get("policy_alias"),
        "policy_index": meta.get("policy_index"),
        "policy_count": meta.get("policy_count"),
        "runs_per_policy": meta.get("runs_per_policy"),
        "schedule_index": meta.get("global_game_idx"),
        "scheduled_total": meta.get("scheduled_total"),
        "global_game_idx": meta.get("global_game_idx"),
        "policy_game_idx": meta.get("policy_game_idx"),
        "game_idx_global": meta.get("global_game_idx"),
        "game_idx_within_policy": meta.get("policy_game_idx"),
        "replicate_idx": meta.get("policy_game_idx"),
        "pair_id": meta.get("pair_id"),
        "counterbalance_order": meta.get("counterbalance_order"),
        "abba_block_idx": meta.get("abba_block_idx"),
        "abba_position": meta.get("abba_position"),
        "base_seed": meta.get("base_seed"),
        "case_seed": meta.get("case_seed"),
        "role_seed": meta.get("role_seed"),
        "actor_seed": meta.get("actor_seed"),
        "orchestrator_seed": meta.get("orchestrator_seed"),
        "game_id": meta.get("game_id"),
        "player_names": list(player_names),
        "turn_policy": meta.get("turn_policy"),
    }
    for key in (
        "seat_permutation_mode",
        "seat_rotation",
        "seat_permutation",
        "permutation_id",
        "role_layout_mode",
        "role_layout_seed_base",
        "role_layout_index",
        "role_layout_count",
        "role_layout_control_cycle",
        "role_layout_block_id",
        "persona_mode",
        "persona_seed_base",
        "persona_case_seed",
        "persona_profile_ids",
        "persona_cycle_length",
        "persona_counterbalance_position",
        "source_persona_assignment_id",
        "source_persona_profile_ids",
        "persona_catalog_version",
    ):
        if key in meta:
            output[key] = meta[key]
    return output


def persona_assignment_metadata(
    meta: dict[str, Any],
    *,
    source_player_names: list[str],
    player_names: list[str],
) -> dict[str, Any]:
    """Resolve source-player persona profiles onto physical seats."""

    mode = str(meta.get("persona_mode") or "legacy")
    if mode == "legacy":
        return {}
    source_profiles = meta.get("source_persona_profile_ids")
    if not isinstance(source_profiles, list) or len(source_profiles) != len(source_player_names):
        raise ValueError("persona source assignment must cover every source player")
    permutation = _normalized_permutation(meta, len(source_player_names))
    if len(player_names) != len(source_player_names):
        raise ValueError("persona assignment player-name lengths must match")
    assignments: list[dict[str, Any]] = []
    for physical_seat, source_seat in enumerate(permutation, start=1):
        profile_id = str(source_profiles[source_seat - 1])
        profile = _PERSONA_CATALOG.get(profile_id)
        if profile is None:
            raise ValueError(f"unknown persona profile: {profile_id!r}")
        assignments.append({
            "seat": physical_seat,
            "player_name": str(player_names[physical_seat - 1]),
            "source_seat": source_seat,
            "source_player_name": str(source_player_names[source_seat - 1]),
            **profile,
        })
    assignment_id = _short_hash({
        "catalog_version": PERSONA_CATALOG_VERSION,
        "mode": mode,
        "assignments": assignments,
    }, prefix="persona")
    return {
        "persona_assignment_id": assignment_id,
        "persona_assignments": assignments,
        "persona_catalog_version": PERSONA_CATALOG_VERSION,
    }


def validate_persona_provenance(
    metadata: dict[str, Any],
    *,
    player_names: list[str],
) -> list[dict[str, Any]]:
    """Fail closed on persona metadata before it can alter actor prompts."""

    mode = str(metadata.get("persona_mode") or "legacy")
    assignments = metadata.get("persona_assignments")
    if mode == "legacy":
        if assignments is not None and assignments != [] and assignments != ():
            raise ValueError("legacy persona mode cannot contain persona assignments")
        return []
    if mode not in PERSONA_MODES[1:]:
        raise ValueError(f"unsupported persona_mode: {mode!r}")
    if metadata.get("persona_catalog_version") != PERSONA_CATALOG_VERSION:
        raise ValueError("persona catalog version does not match the runtime catalog")
    source_player_names = metadata.get("source_player_names")
    if (
        not isinstance(source_player_names, list)
        or len(source_player_names) != len(player_names)
        or any(not isinstance(name, str) or not name for name in source_player_names)
    ):
        raise ValueError("persona provenance requires every source player name")
    expected_source_profiles = _expected_source_persona_profiles(
        metadata,
        seat_count=len(player_names),
    )
    source_profiles = metadata.get("source_persona_profile_ids")
    if source_profiles != expected_source_profiles:
        raise ValueError("source persona assignment does not match its deterministic controls")
    expected_source_assignment_id = _short_hash({
        "catalog_version": PERSONA_CATALOG_VERSION,
        "mode": mode,
        "source_profile_ids": expected_source_profiles,
    }, prefix="source-persona")
    if metadata.get("source_persona_assignment_id") != expected_source_assignment_id:
        raise ValueError("source_persona_assignment_id does not match deterministic controls")
    if not isinstance(assignments, list) or len(assignments) != len(player_names):
        raise ValueError("persona assignments must cover every physical seat")
    normalized: list[dict[str, Any]] = []
    source_seats: set[int] = set()
    for expected_seat, raw in enumerate(assignments, start=1):
        if not isinstance(raw, dict):
            raise ValueError("each persona assignment must be an object")
        seat = raw.get("seat")
        source_seat = raw.get("source_seat")
        if type(seat) is not int or seat != expected_seat:
            raise ValueError("persona assignments must be ordered by physical seat")
        if type(source_seat) is not int or source_seat not in range(1, len(player_names) + 1):
            raise ValueError("persona source seat is outside the player range")
        if source_seat in source_seats:
            raise ValueError("persona source seats must form a permutation")
        source_seats.add(source_seat)
        profile_id = str(raw.get("profile_id") or "")
        profile = _PERSONA_CATALOG.get(profile_id)
        if profile is None:
            raise ValueError(f"unknown persona profile: {profile_id!r}")
        if raw.get("name") != profile["name"] or raw.get("description") != profile["description"]:
            raise ValueError("persona prompt text does not match the versioned catalog")
        if raw.get("player_name") != player_names[expected_seat - 1]:
            raise ValueError("persona assignment player name does not match its seat")
        if raw.get("source_player_name") != source_player_names[source_seat - 1]:
            raise ValueError("persona assignment source player does not match its source seat")
        if profile_id != expected_source_profiles[source_seat - 1]:
            raise ValueError("persona assignment profile does not match its source player")
        normalized.append(dict(raw))
    expected_id = _short_hash({
        "catalog_version": PERSONA_CATALOG_VERSION,
        "mode": mode,
        "assignments": normalized,
    }, prefix="persona")
    if metadata.get("persona_assignment_id") != expected_id:
        raise ValueError("persona_assignment_id does not match persona assignments")
    return normalized


def _expected_source_persona_profiles(
    metadata: dict[str, Any],
    *,
    seat_count: int,
) -> list[str]:
    mode = str(metadata.get("persona_mode") or "")
    raw_profiles = metadata.get("persona_profile_ids")
    if not isinstance(raw_profiles, list):
        raise ValueError("persona_profile_ids must be a list")
    profile_ids = _normalize_persona_profiles(raw_profiles)
    seed_base = metadata.get("persona_seed_base")
    case_seed = metadata.get("persona_case_seed")
    if type(seed_base) is not int or type(case_seed) is not int:
        raise ValueError("persona provenance requires integer base and case seeds")
    expected_cycle_length = len(profile_ids) if mode == "counterbalanced" else 1
    if metadata.get("persona_cycle_length") != expected_cycle_length:
        raise ValueError("persona_cycle_length does not match persona mode")

    base_seed = metadata.get("base_seed")
    paired_case_seed = metadata.get("case_seed")
    if type(base_seed) is not int or type(paired_case_seed) is not int:
        raise ValueError("persona provenance requires integer experiment case seeds")
    case_index = paired_case_seed - base_seed
    if case_index < 1:
        raise ValueError("persona provenance case index is invalid")

    if mode == "randomized":
        if case_seed != seed_base + case_index - 1:
            raise ValueError("randomized persona case seed does not match experiment case")
        if metadata.get("persona_counterbalance_position") is not None:
            raise ValueError("randomized persona provenance cannot claim a counterbalance position")
        expected = [profile_ids[index % len(profile_ids)] for index in range(seat_count)]
        random.Random(case_seed).shuffle(expected)
        return expected

    if case_seed != seed_base:
        raise ValueError("fixed/counterbalanced persona case seed must equal its base seed")
    if mode == "fixed":
        expected_position = 1
        position = 0
    elif mode == "counterbalanced":
        seat_cycle = (
            seat_count
            if metadata.get("seat_permutation_mode") == "cyclic"
            else 1
        )
        position = (case_index - 1) // seat_cycle
        expected_position = (position % len(profile_ids)) + 1
    else:
        raise ValueError(f"unsupported persona_mode: {mode!r}")
    if metadata.get("persona_counterbalance_position") != expected_position:
        raise ValueError("persona counterbalance position does not match experiment case")
    offset = random.Random(seed_base).randrange(len(profile_ids))
    return [
        profile_ids[(source_seat - 1 + offset + position) % len(profile_ids)]
        for source_seat in range(1, seat_count + 1)
    ]


def resolved_role_layout_metadata(
    metadata: dict[str, Any],
    *,
    ruleset_id: str,
    role_deck: list[str],
    role_seed: int,
) -> dict[str, Any]:
    """Record the exact deterministic seat-role layout for explicit controls."""

    mode = str(metadata.get("role_layout_mode") or "legacy")
    if mode == "legacy":
        return {}
    if mode not in ROLE_LAYOUT_MODES[1:]:
        raise ValueError(f"unsupported role_layout_mode: {mode!r}")
    recorded_base = metadata.get("role_layout_seed_base")
    index = metadata.get("role_layout_index")
    count = metadata.get("role_layout_count")
    cycle = metadata.get("role_layout_control_cycle")
    if any(type(value) is not int for value in (recorded_base, index, count, cycle)):
        raise ValueError("role-layout provenance requires integer seed/index/count/cycle")
    if index < 1 or count < 1 or index > count or cycle < 1:
        raise ValueError("role-layout provenance indexes are invalid")
    expected_seed = recorded_base if mode == "fixed" else recorded_base + index - 1
    if role_seed != expected_seed:
        raise ValueError("role_seed does not match explicit role-layout provenance")
    if mode == "fixed" and (index != 1 or count != 1):
        raise ValueError("fixed role-layout provenance must use one layout")
    shuffled = list(role_deck)
    random.Random(role_seed).shuffle(shuffled)
    layout = [
        {"seat": seat, "role": role}
        for seat, role in enumerate(shuffled, start=1)
    ]
    layout_id = _short_hash({
        "schema_version": ROLE_LAYOUT_SCHEMA_VERSION,
        "ruleset_id": ruleset_id,
        "layout": layout,
    }, prefix="role-layout")
    return {
        "role_layout_schema_version": ROLE_LAYOUT_SCHEMA_VERSION,
        "role_layout_id": layout_id,
        "role_layout": layout,
    }


def apply_seat_permutation(
    player_names: list[str],
    meta: dict[str, Any],
) -> list[str]:
    """Apply the schedule's recorded permutation to player names.

    The default/fixed schedule deliberately returns a copy in the original
    order, preserving legacy run specs and hashes. A cyclic permutation is
    explicit in metadata and therefore part of each concrete RunSpec.
    """

    permutation = meta.get("seat_permutation")
    if not isinstance(permutation, list):
        return list(player_names)
    if len(permutation) != len(player_names):
        raise ValueError("seat_permutation length must match player_names")
    try:
        normalized = [int(index) for index in permutation]
    except (TypeError, ValueError) as err:
        raise ValueError("seat_permutation must contain integer seat indexes") from err
    expected = list(range(1, len(player_names) + 1))
    if sorted(normalized) != expected:
        raise ValueError("seat_permutation must be a permutation of 1..seat_count")
    return [player_names[index - 1] for index in normalized]


def apply_seat_mapping_permutation(
    values: dict[int, Any] | None,
    meta: dict[str, Any],
) -> dict[int, Any]:
    """Move source-seat keyed values with the scheduled player permutation."""

    source = {int(seat): value for seat, value in (values or {}).items()}
    permutation = meta.get("seat_permutation")
    if not isinstance(permutation, list):
        return source
    try:
        normalized = [int(index) for index in permutation]
    except (TypeError, ValueError) as err:
        raise ValueError("seat_permutation must contain integer seat indexes") from err
    expected = list(range(1, len(normalized) + 1))
    if sorted(normalized) != expected:
        raise ValueError("seat_permutation must be a permutation of 1..seat_count")
    return {
        new_seat: source[source_seat]
        for new_seat, source_seat in enumerate(normalized, start=1)
        if source_seat in source
    }


def _normalize_persona_profiles(values: list[str] | None) -> list[str]:
    if values is None:
        return list(_PERSONA_PROFILE_IDS)
    if not isinstance(values, list) or not values:
        raise ValueError("persona_profiles must be a non-empty list")
    normalized = [str(value).strip() for value in values]
    if any(not value for value in normalized):
        raise ValueError("persona profile IDs must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("persona_profiles must be unique")
    unknown = [value for value in normalized if value not in _PERSONA_CATALOG]
    if unknown:
        raise ValueError(
            f"unknown persona profiles: {unknown}; allowed={list(_PERSONA_PROFILE_IDS)}"
        )
    return normalized


def _control_seed(
    explicit: int | None,
    *,
    base_seed: int | None,
    offset: int,
    label: str,
    enabled: bool,
) -> int | None:
    if not enabled:
        return None
    if explicit is not None:
        return explicit
    if base_seed is None:
        raise ValueError(f"explicit {label} control requires its own seed or a base seed")
    return base_seed + offset


def _persona_schedule_fields(
    *,
    case_idx: int,
    seat_count: int,
    mode: str,
    seed_base: int,
    profile_ids: list[str],
    counterbalance_position: int | None,
) -> dict[str, Any]:
    cycle_length = len(profile_ids) if mode == "counterbalanced" else 1
    if mode == "randomized":
        case_seed = seed_base + case_idx - 1
        source_profiles = [
            profile_ids[index % len(profile_ids)]
            for index in range(seat_count)
        ]
        random.Random(case_seed).shuffle(source_profiles)
        counterbalance_position = None
    else:
        case_seed = seed_base
        offset = random.Random(seed_base).randrange(len(profile_ids))
        position = (
            int(counterbalance_position or 0)
            if mode == "counterbalanced"
            else 0
        )
        source_profiles = [
            profile_ids[(source_seat - 1 + offset + position) % len(profile_ids)]
            for source_seat in range(1, seat_count + 1)
        ]
        counterbalance_position = (position % len(profile_ids)) + 1
    source_assignment_id = _short_hash({
        "catalog_version": PERSONA_CATALOG_VERSION,
        "mode": mode,
        "source_profile_ids": source_profiles,
    }, prefix="source-persona")
    return {
        "persona_mode": mode,
        "persona_seed_base": seed_base,
        "persona_case_seed": case_seed,
        "persona_profile_ids": list(profile_ids),
        "persona_cycle_length": cycle_length,
        "persona_counterbalance_position": counterbalance_position,
        "source_persona_profile_ids": source_profiles,
        "source_persona_assignment_id": source_assignment_id,
        "persona_catalog_version": PERSONA_CATALOG_VERSION,
    }


def _normalized_permutation(meta: dict[str, Any], seat_count: int) -> list[int]:
    raw = meta.get("seat_permutation")
    if raw is None:
        return list(range(1, seat_count + 1))
    if not isinstance(raw, list) or len(raw) != seat_count:
        raise ValueError("seat_permutation length must match player count")
    if any(type(value) is not int for value in raw):
        raise ValueError("seat_permutation must contain integer seat indexes")
    normalized = list(raw)
    if sorted(normalized) != list(range(1, seat_count + 1)):
        raise ValueError("seat_permutation must be a permutation of 1..seat_count")
    return normalized


def _short_hash(value: Any, *, prefix: str) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(encoded).hexdigest()[:20]}"
