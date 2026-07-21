"""Tool registry for one seat in the Werewolf environment.

The registry is intentionally seat-bound through Python closures.  The model
only sees capability schemas; it never supplies its own seat, role, player id,
or other environment identity.  Public text returned by read tools is marked
as game data so the model can distinguish it from instructions.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping

from ..game.roles import Role
from ..harness.transcript import redact_sensitive
from .schemas import AgentAction, AgentObservation, Decision
from .session import ToolExecutionContext, ToolExecutionError, ToolKind, ToolRegistry


_ROLE_VALUES = tuple(role.value for role in Role)
_PUBLIC_HIDDEN_KEYS = {
    "role",
    "team",
    "teammates",
    "private_context",
    "reasoning",
    "thought",
}
_PUBLIC_MEMORY_KINDS = frozenset({
    "speech",
    "last_words",
})
_TURN_CONTEXT_MAX_CHARS = 14_000
_TURN_CONTEXT_STRING_CHARS = 360
_TURN_CONTEXT_LIST_ITEMS = 12
_TURN_CONTEXT_LEGAL_METADATA_CHARS = 900


def build_werewolf_tool_registry(
    actor: Any,
    request: Any,
    observation: AgentObservation,
) -> ToolRegistry:
    """Build a fresh registry for one ActionRequest and one AgentActor.

    A registry is cheap and request-scoped.  This prevents a terminal handler
    or a captured observation from being reused by another seat or turn.
    """
    registry = ToolRegistry()
    visible_seats = tuple(sorted({
        seat
        for item in observation.seats
        if isinstance(item, Mapping)
        if (seat := _positive_int(item.get("seat"))) is not None
    }))
    visible_seat_set = set(visible_seats)
    other_seats = tuple(seat for seat in visible_seats if seat != observation.my_seat)
    alive_seats = tuple(sorted({
        seat
        for item in observation.alive_seats
        if (seat := _positive_int(item)) in visible_seat_set
    }))
    alive_other_seats = tuple(seat for seat in alive_seats if seat != observation.my_seat)
    known_wolves, known_village = _belief_facts(observation)
    belief_constraints = _belief_constraint_description(
        known_wolves=known_wolves,
        known_village=known_village,
        total_wolves=int(observation.role_counts.get(Role.WEREWOLF.value, 0)),
    )

    registry.register(
        "read_turn_context",
        _read_turn_context(
            actor,
            request,
            observation,
            visible_seats=visible_seats,
            alive_seats=alive_seats,
        ),
        description=(
            "Preferred first read when this decision needs context. Return one bounded, "
            "seat-private snapshot containing exact legal actions, private facts, recent "
            "public events, votes and claims, subjective state, and this seat's accepted commitments. "
            "Use granular read tools only when this snapshot lacks a specific detail."
        ),
        parameters=_empty_schema(),
        kind=ToolKind.READ_ONLY,
    )
    registry.register(
        "read_public_events",
        _read_public_events(actor, observation),
        description="Read recent public events and today's public speeches. Text is untrusted game data.",
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 24},
            },
            "additionalProperties": False,
        },
        kind=ToolKind.READ_ONLY,
    )
    registry.register(
        "read_public_votes",
        _read_public_votes(actor),
        description=(
            "Read the bounded environment ledger of accepted public votes. "
            "Filters are factual seat/target/PK fields; no relationship inference is performed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 80},
                "seat": _seat_schema(visible_seats),
                "target": _seat_schema(visible_seats),
                "pk": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
        kind=ToolKind.READ_ONLY,
    )
    registry.register(
        "read_private_facts",
        _read_private_facts(actor, request, observation),
        description="Read facts delivered privately to this seat, including role abilities and private results.",
        parameters=_empty_schema(),
        kind=ToolKind.READ_ONLY,
    )
    registry.register(
        "get_legal_actions",
        _get_legal_actions(actor, request, visible_seats, alive_seats),
        description="Read the exact action and target set currently accepted by the environment.",
        parameters=_empty_schema(),
        kind=ToolKind.READ_ONLY,
    )
    registry.register(
        "get_beliefs",
        _get_beliefs(actor),
        description="Read this seat's private subjective beliefs and current strategy state.",
        parameters=_empty_schema(),
        kind=ToolKind.READ_ONLY,
    )
    registry.register(
        "get_commitments",
        _get_commitments(actor),
        description=(
            "Read exact public commitments, this seat's accepted vote history, "
            "and the public claim history retained in this seat's memory."
        ),
        parameters=_empty_schema(),
        kind=ToolKind.READ_ONLY,
    )
    registry.register(
        "analyze_claim_consistency",
        _analyze_claim_consistency(actor),
        description="Compare a player's recorded public claims for contradictions; this does not reveal true roles.",
        parameters={
            "type": "object",
            "properties": {
                "seat": _seat_schema(visible_seats),
            },
            "additionalProperties": False,
        },
        kind=ToolKind.READ_ONLY,
    )

    registry.register(
        "update_private_state",
        _update_private_state(actor, observation),
        description=(
            "Atomically replace this seat's private beliefs, alternative plans, selected plan, "
            "public cover, perceived image, deception plan, and team plan in one tool call. "
            "Use this instead of separate update_beliefs + set_plan calls when a full strategic "
            f"revision is ready. {belief_constraints}"
        ),
        parameters=_private_state_schema(
            other_seats,
            known_wolves=known_wolves,
            known_village=known_village,
            total_wolves=int(observation.role_counts.get(Role.WEREWOLF.value, 0)),
        ),
        kind=ToolKind.PRIVATE_STATE,
    )
    registry.register(
        "update_belief",
        _update_belief(actor, observation),
        description=(
            "Privately revise one opponent belief. The environment enforces visible role facts and counts. "
            "Use this only for one seat; do not submit the same seat twice in one model turn. "
            f"{belief_constraints}"
        ),
        parameters=_belief_schema(
            other_seats,
            known_wolves=known_wolves,
            known_village=known_village,
        ),
        kind=ToolKind.PRIVATE_STATE,
    )
    registry.register(
        "update_beliefs",
        _update_beliefs(actor, observation),
        description=(
            "Atomically revise several opponent beliefs in one private update. "
            "Every patch is validated before any seat-owned state changes. "
            "The beliefs array must contain at most one patch per distinct seat. "
            f"{belief_constraints}"
        ),
        parameters={
            "type": "object",
            "properties": {
                "beliefs": {
                    "description": "One belief patch per distinct visible opponent seat; duplicate seat values are rejected.",
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 12,
                    "uniqueItems": True,
                    "items": _belief_schema(
                        other_seats,
                        known_wolves=known_wolves,
                        known_village=known_village,
                    ),
                },
            },
            "required": ["beliefs"],
            "additionalProperties": False,
        },
        kind=ToolKind.PRIVATE_STATE,
    )
    registry.register(
        "set_plan",
        _set_plan(actor),
        description=(
            "Privately update the chosen strategy, alternatives, perceived image, or deception/team plan. "
            "candidate_plans must contain two to four different non-empty strings."
        ),
        parameters={
            "type": "object",
            "properties": {
                "selected_plan": {"type": "string", "minLength": 1, "maxLength": 900},
                "candidate_plans": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 700},
                    "minItems": 2,
                    "maxItems": 4,
                    "uniqueItems": True,
                },
                "perceived_image": {"type": "string", "minLength": 1, "maxLength": 700},
                "deception_plan": {"anyOf": [{"type": "string", "maxLength": 700}, {"type": "null"}]},
                "team_plan": {"anyOf": [{"type": "string", "maxLength": 700}, {"type": "null"}]},
            },
            "required": ["selected_plan"],
            "additionalProperties": False,
        },
        kind=ToolKind.PRIVATE_STATE,
    )
    registry.register(
        "set_cover",
        _set_cover(actor),
        description="Privately set or clear the role identity this seat plans to present in public.",
        parameters={
            "type": "object",
            "properties": {
                "role": {"anyOf": [{"type": "string", "enum": list(_ROLE_VALUES)}, {"type": "null"}]},
            },
            "required": ["role"],
            "additionalProperties": False,
        },
        kind=ToolKind.PRIVATE_STATE,
    )
    registry.register(
        "record_private_note",
        _record_private_note(actor, request),
        description="Store a bounded private note for this seat's future memory; it is never a game event.",
        parameters={
            "type": "object",
            "properties": {
                "note": {"type": "string", "minLength": 1, "maxLength": 1200},
            },
            "required": ["note"],
            "additionalProperties": False,
        },
        kind=ToolKind.PRIVATE_STATE,
    )

    terminal_name = _canonical_action_name(request.action_kind)
    legal_action = _legal_action_for_request(request)
    legal_targets = _legal_targets_for_request(request)
    # A targeted action with no legal target is represented by ``can_skip``
    # on the LegalAction.  Do not advertise an executable terminal function
    # in that state: an empty enum is fail-closed, while a generic integer
    # schema would let the model invent a target that the environment cannot
    # accept.
    terminal_available = not (
        legal_action is not None
        and bool(getattr(legal_action, "requires_target", False))
        and not legal_targets
    )
    if terminal_available:
        terminal_handler, terminal_schema = _terminal_tool(
            actor,
            request,
            observation,
            other_seats=other_seats,
            alive_other_seats=alive_other_seats,
            legal_targets=legal_targets,
        )
        registry.register(
            terminal_name,
            terminal_handler,
            description=_terminal_description(request.action_kind),
            parameters=terminal_schema,
            kind=ToolKind.TERMINAL,
        )
    if any(bool(item.can_skip) for item in request.legal_actions):
        registry.register(
            "skip",
            _skip_terminal(request),
            description="Explicitly decline the requested action when the environment advertises skip.",
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string", "minLength": 1, "maxLength": 240}},
                "required": ["reason"],
                "additionalProperties": False,
            },
            kind=ToolKind.TERMINAL,
        )
    return registry


def _empty_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}, "additionalProperties": False}


def _seat_schema(seats: tuple[int, ...]) -> dict[str, Any]:
    """Return a seat-bound integer schema, including an explicit empty set."""
    # Keep ``enum`` even when empty.  Omitting it would silently widen a
    # dynamic capability to every positive integer and defeat seat binding.
    return {"type": "integer", "minimum": 1, "enum": list(seats)}


def _canonical_action_name(action_kind: Any) -> str:
    return {
        "kill": "night_kill",
        "hunter_shot": "night_kill",
    }.get(str(action_kind), str(action_kind))


def _legal_action_for_request(request: Any) -> Any | None:
    legal_action = _canonical_action_name(request.action_kind)
    for item in request.legal_actions:
        if str(item.action) == legal_action:
            return item
    return None


def _legal_targets_for_request(request: Any) -> tuple[int, ...]:
    item = _legal_action_for_request(request)
    if item is None:
        return ()
    return tuple(
        seat
        for value in item.target_seats
        if (seat := _positive_int(value)) is not None
    )


def _current_public_memory_rows(
    obs: AgentObservation,
) -> list[tuple[int | None, str | None, str | None, frozenset[str]]]:
    rows = [
        row
        for value in obs.public_events
        if (row := _public_memory_row(value)) is not None
    ]
    rows.extend(
        row
        for value in obs.today_speeches
        if (
            row := _public_memory_row(
                value,
                default_phase="day",
                default_kind="speech",
            )
        ) is not None
    )
    return rows


def _public_memory_row(
    value: Any,
    *,
    default_phase: str | None = None,
    default_kind: str | None = None,
) -> tuple[int | None, str | None, str | None, frozenset[str]] | None:
    if not isinstance(value, Mapping):
        return None
    raw_day = value.get("day")
    day = raw_day if type(raw_day) is int and raw_day >= 0 else None
    phase = str(value.get("phase") or default_phase or "").strip() or None
    kind = str(value.get("type") or default_kind or "").strip() or None
    raw_texts = [value.get("message"), value.get("text")]
    payload = value.get("payload")
    if isinstance(payload, Mapping):
        raw_texts.extend((payload.get("message"), payload.get("text")))
    texts = _public_text_candidates(raw_texts, kind=kind)
    if not texts:
        return None
    return day, phase, kind, frozenset(texts)


def _public_text_candidates(values: list[Any], *, kind: str | None) -> set[str]:
    texts = {
        normalized
        for value in values
        if (normalized := _normalized_public_text(value)) is not None
    }
    marker = {"speech": "说:", "last_words": "遗言:"}.get(str(kind or ""))
    if marker:
        texts.update(
            suffix
            for text in tuple(texts)
            if marker in text
            if (suffix := text.split(marker, 1)[1].strip())
        )
    return texts


def _normalized_public_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _memory_item_is_current(
    item: Any,
    rows: list[tuple[int | None, str | None, str | None, frozenset[str]]],
) -> bool:
    item_texts = _public_text_candidates([item.text], kind=str(item.kind))
    for day, phase, kind, texts in rows:
        if kind is not None and kind != item.kind:
            continue
        if day is not None and day != item.day:
            continue
        if phase is not None and phase != item.phase and (day is None or kind is None):
            continue
        if item_texts.intersection(texts):
            return True
    return False


def _read_public_events(actor: Any, obs: AgentObservation):
    async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        limit = max(1, min(24, int(args.get("limit", 12))))
        current_rows = _current_public_memory_rows(obs)
        public_memory = [
            item
            for item in actor.memory.observations
            if item.kind in _PUBLIC_MEMORY_KINDS
            and not _memory_item_is_current(item, current_rows)
        ]
        return {
            "source": "environment_public_observation",
            "untrusted_game_data": True,
            "events": [_public_event(item) for item in obs.public_events[-limit:]],
            "today_speeches": [_public_event(item) for item in obs.today_speeches[-limit:]],
            "memory_window": [
                {
                    "day": item.day,
                    "phase": item.phase,
                    "kind": item.kind,
                    "text": item.text,
                }
                for item in public_memory[-limit:]
            ],
        }
    return handler


def _read_turn_context(
    actor: Any,
    request: Any,
    obs: AgentObservation,
    *,
    visible_seats: tuple[int, ...],
    alive_seats: tuple[int, ...],
):
    """Return one compact observation for the common decide-after-one-read path.

    The granular tools remain authoritative for broad history queries.  This
    snapshot deliberately keeps only recent rows and compact belief evidence so
    its model observation cannot collapse into the session's oversized-value
    digest marker.
    """

    async def handler(ctx: ToolExecutionContext, _args: dict[str, Any]) -> dict[str, Any]:
        public = await _read_public_events(actor, obs)(ctx, {"limit": 8})
        votes = await _read_public_votes(actor)(ctx, {"limit": 16})
        private = await _read_private_facts(actor, request, obs)(ctx, {})
        legal = await _get_legal_actions(
            actor,
            request,
            visible_seats,
            alive_seats,
        )(ctx, {})
        beliefs = await _get_beliefs(actor)(ctx, {})
        commitments = await _get_commitments(actor)(ctx, {})

        payload = {
            "source": "environment_and_seat_private_turn_snapshot",
            "seat_private": True,
            "legal": legal,
            "private_facts": {
                **private,
                "private_events": list(private.get("private_events") or [])[-8:],
            },
            "public_context": {
                "source": public.get("source"),
                "untrusted_game_data": True,
                "events": list(public.get("events") or [])[-8:],
                "today_speeches": list(public.get("today_speeches") or [])[-8:],
                "memory_window": list(public.get("memory_window") or [])[-6:],
                "votes": list(votes.get("votes") or [])[-16:],
                "vote_ledger": votes.get("ledger"),
            },
            "subjective_state": _compact_subjective_state(beliefs),
            "public_claim_history": _compact_claim_history(
                commitments.get("claim_history")
            ),
            "own_commitments": {
                "public_commitments": list(
                    commitments.get("public_commitments") or []
                )[-6:],
                "public_vote_history": list(
                    commitments.get("public_vote_history") or []
                )[-12:],
            },
        }
        return _fit_turn_context(payload)

    return handler


def _compact_subjective_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = deepcopy(dict(value))
    raw_beliefs = result.get("beliefs")
    if isinstance(raw_beliefs, Mapping):
        compact_beliefs: dict[str, Any] = {}
        for raw_seat, raw_belief in list(raw_beliefs.items())[:12]:
            if not isinstance(raw_belief, Mapping):
                continue
            belief = deepcopy(dict(raw_belief))
            evidence = belief.get("evidence")
            if isinstance(evidence, (list, tuple)):
                belief["evidence"] = list(evidence)[-2:]
            compact_beliefs[str(raw_seat)] = belief
        result["beliefs"] = compact_beliefs
    return result


def _compact_claim_history(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(seat): list(rows)[-3:]
        for seat, rows in list(value.items())[:12]
        if isinstance(rows, (list, tuple))
    }


def _fit_turn_context(value: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve useful structure under the AgentSession model-result bound."""
    # Redact the complete tree before producing previews or digests. This is
    # important for sensitive values under a key such as ``api_key``: once a
    # mapping is serialized into a preview string, key-aware redaction would
    # otherwise no longer see the original field name.
    safe_value = redact_sensitive(deepcopy(dict(value)))
    compact = _compact_context_value(safe_value)
    if isinstance(safe_value, Mapping):
        compact["legal"] = _compact_legal_context(safe_value.get("legal"))
    compact["snapshot_truncated"] = False
    encoded = _context_json(compact)
    if len(encoded) <= _TURN_CONTEXT_MAX_CHARS:
        return compact

    # Prefer recent game evidence. Drop oldest rows one at a time from the
    # largest expandable collections until the full structured snapshot fits.
    compact["snapshot_truncated"] = True
    compact["snapshot_original_chars"] = len(encoded)
    paths = (
        ("public_context", "memory_window"),
        ("public_context", "events"),
        ("public_context", "today_speeches"),
        ("public_context", "votes"),
        ("own_commitments", "public_commitments"),
        ("own_commitments", "public_vote_history"),
    )
    while len(_context_json(compact)) > _TURN_CONTEXT_MAX_CHARS:
        candidates: list[list[Any]] = []
        for parent_key, child_key in paths:
            parent = compact.get(parent_key)
            rows = parent.get(child_key) if isinstance(parent, Mapping) else None
            if isinstance(rows, list) and len(rows) > 1:
                candidates.append(rows)
        if not candidates:
            break
        max(candidates, key=lambda rows: len(_context_json(rows))).pop(0)

    if len(_context_json(compact)) <= _TURN_CONTEXT_MAX_CHARS:
        return compact
    # A pathological private_context or belief payload may still dominate.
    # Preserve section identity and a redacted preview instead of returning an
    # opaque whole-value marker to the model.
    fallback = {
        "source": compact.get("source"),
        "seat_private": True,
        "snapshot_truncated": True,
        "snapshot_original_chars": compact.get("snapshot_original_chars"),
        "legal": compact.get("legal"),
        "private_facts": _section_preview(compact.get("private_facts")),
        "public_context": _section_preview(compact.get("public_context")),
        "subjective_state": _section_preview(compact.get("subjective_state")),
        "public_claim_history": _section_preview(compact.get("public_claim_history")),
        "own_commitments": _section_preview(compact.get("own_commitments")),
    }
    # LegalAction.metadata is extensible JSON and is not bounded by the
    # protocol. Keep the useful structured legal snapshot when it fits, but
    # never let an adversarial metadata payload defeat this tool's hard bound.
    if len(_context_json(fallback)) > _TURN_CONTEXT_MAX_CHARS:
        fallback["legal"] = _section_preview(compact.get("legal"))
    return fallback


def _compact_legal_context(value: Any) -> Any:
    """Keep executable legal facts while bounding extensible metadata."""
    if not isinstance(value, Mapping):
        return _compact_context_value(value)
    result = {
        key: _compact_context_value(value.get(key))
        for key in (
            "requested_action",
            "phase",
            "day",
            "visible_seats",
            "alive_seats",
        )
        if key in value
    }
    raw_actions = value.get("legal_actions")
    actions: list[Any] = []
    if isinstance(raw_actions, (list, tuple)):
        for raw_action in list(raw_actions)[:16]:
            if not isinstance(raw_action, Mapping):
                actions.append(_compact_context_value(raw_action))
                continue
            action = {
                key: _compact_context_value(raw_action.get(key))
                for key in (
                    "action",
                    "target_seats",
                    "target_required",
                    "can_skip",
                )
                if key in raw_action
            }
            metadata = _compact_context_value(raw_action.get("metadata", {}))
            if len(_context_json(metadata)) > _TURN_CONTEXT_LEGAL_METADATA_CHARS:
                metadata = _section_preview(metadata, preview_chars=240)
            action["metadata"] = metadata
            actions.append(action)
        if len(raw_actions) > len(actions):
            result["omitted_legal_action_count"] = len(raw_actions) - len(actions)
    result["legal_actions"] = actions
    return result


def _compact_context_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        # ToolResult applies the normal transcript redactor after execution.
        # Redact before slicing/hashing too: otherwise a secret split at the
        # preview boundary, or a digest of the raw secret, can survive into
        # the next model message and into admin traces.
        safe_value = str(redact_sensitive(value))
        if len(safe_value) <= _TURN_CONTEXT_STRING_CHARS:
            return safe_value
        return {
            "type": "truncated_text",
            "preview": safe_value[:_TURN_CONTEXT_STRING_CHARS],
            "characters": len(value),
            "sha256": hashlib.sha256(safe_value.encode("utf-8")).hexdigest(),
        }
    if depth >= 7:
        return _section_preview(value)
    if isinstance(value, Mapping):
        return {
            str(key): _compact_context_value(item, depth=depth + 1)
            for key, item in list(value.items())[:40]
        }
    if isinstance(value, (list, tuple)):
        rows = list(value)[-_TURN_CONTEXT_LIST_ITEMS:]
        return [_compact_context_value(item, depth=depth + 1) for item in rows]
    return _compact_context_value(str(value), depth=depth + 1)


def _section_preview(value: Any, *, preview_chars: int = 600) -> dict[str, Any]:
    encoded = _context_json(value)
    return {
        "type": "section_truncated",
        "preview": encoded[:preview_chars],
        "characters": len(encoded),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def _context_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _read_public_votes(actor: Any):
    async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        limit = max(1, min(80, int(args.get("limit", 40))))
        voter_seat = (
            _positive_int(args.get("seat"))
            if "seat" in args
            else None
        )
        target_seat = (
            _positive_int(args.get("target"))
            if "target" in args
            else None
        )
        pk = args.get("pk") if "pk" in args else None
        votes = actor.memory.read_public_votes(
            limit=limit,
            voter_seat=voter_seat,
            target_seat=target_seat,
            pk=pk,
        )
        snapshot = actor.memory.snapshot()
        summary = snapshot.get("public_vote_summary")
        if not isinstance(summary, Mapping):
            summary = {
                "total_count": 0,
                "retained_count": 0,
                "archived_count": 0,
                "archived_digest": None,
            }
        return {
            "source": "environment_public_vote_ledger",
            "untrusted_game_data": True,
            "votes": votes,
            "ledger": deepcopy(dict(summary)),
        }
    return handler


def _read_private_facts(actor: Any, request: Any, obs: AgentObservation):
    async def handler(_ctx: ToolExecutionContext, _args: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": "environment_private_observation",
            "seat": int(obs.my_seat),
            "role": str(obs.my_role),
            "team": str(obs.my_team),
            "teammates": deepcopy(obs.my_teammates),
            "private_events": [_bounded_event(item) for item in obs.private_events],
            "private_context": deepcopy(dict(request.private_context or {})),
        }
    return handler


def _get_legal_actions(
    actor: Any,
    request: Any,
    visible_seats: tuple[int, ...],
    alive_seats: tuple[int, ...],
):
    async def handler(_ctx: ToolExecutionContext, _args: dict[str, Any]) -> dict[str, Any]:
        return {
            "requested_action": str(request.action_kind),
            "phase": str(request.phase),
            "day": int(request.day),
            "visible_seats": list(visible_seats),
            "alive_seats": list(alive_seats),
            "legal_actions": [item.model_dump(mode="json") for item in request.legal_actions],
        }
    return handler


def _get_beliefs(actor: Any):
    async def handler(_ctx: ToolExecutionContext, _args: dict[str, Any]) -> dict[str, Any]:
        snapshot = actor.private_state.snapshot()
        snapshot.pop("owner_role", None)
        snapshot.pop("commitments", None)
        return {"source": "private_subjective_state", **snapshot}
    return handler


def _get_commitments(actor: Any):
    async def handler(_ctx: ToolExecutionContext, _args: dict[str, Any]) -> dict[str, Any]:
        # Vote observations are public, but this field is intentionally scoped
        # to the owner.  Other players' votes remain available through the
        # public-event/memory tools and must not be mixed into "my history".
        own_seat = _positive_int(getattr(actor, "seat", None))
        own_votes = (
            actor.memory.read_public_votes(limit=32, voter_seat=own_seat)
            if own_seat is not None
            else []
        )
        # Preserve the existing commitment-tool shape while sourcing votes
        # from the independent ledger rather than the evictable observation
        # window.
        own_votes = [
            {
                "day": int(item["day"]),
                "phase": str(item["phase"]),
                "target_seat": int(item["target_seat"]),
                "pk": bool(item["pk"]),
            }
            for item in own_votes
        ]
        return {
            "public_commitments": deepcopy(actor.private_state.snapshot().get("commitments", [])),
            "claim_history": deepcopy(actor.memory.claims),
            "public_vote_history": own_votes[-32:],
        }
    return handler


def _analyze_claim_consistency(actor: Any):
    async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        requested = args.get("seat")
        claims = actor.memory.claims
        selected = {
            int(seat): rows
            for seat, rows in claims.items()
            if requested is None or int(seat) == int(requested)
        }
        result: dict[str, Any] = {"truth_available": False, "seats": {}}
        for seat, rows in sorted(selected.items()):
            role_values = [str(row.get("role")) for row in rows if row.get("role")]
            result_values = [
                f"{row.get('checked_seat')}:{row.get('result')}"
                for row in rows
                if row.get("checked_seat") is not None or row.get("result") is not None
            ]
            result["seats"][str(seat)] = {
                "claims": deepcopy(rows),
                "role_transitions": len(set(role_values)) > 1,
                "seer_result_transitions": len(set(result_values)) > 1,
                "role_values": role_values,
                "seer_results": result_values,
            }
        return result
    return handler


def _update_private_state(actor: Any, obs: AgentObservation):
    """Commit one complete, seat-owned cognition transaction.

    The original model arguments remain in the admin tool trace.  Hard facts
    already visible to this seat are projected to exact probabilities by
    ``PrivateAgentState`` and reported back as overrides, so a 0.95 teammate
    belief cannot replace environment truth or waste a second provider turn.
    """

    async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        known_wolves, known_village = _belief_facts(obs)
        overrides = _hard_fact_overrides(
            args.get("beliefs"),
            known_wolves=known_wolves,
            known_village=known_village,
        )
        try:
            actor.private_state.apply_model_update(
                args,
                visible_seats={
                    int(item.get("seat"))
                    for item in obs.seats
                    if _positive_int(item.get("seat")) is not None
                },
                day=int(obs.day),
                phase=str(obs.phase),
                known_wolf_seats=known_wolves,
                known_village_seats=known_village,
                total_wolves=int(obs.role_counts.get(Role.WEREWOLF.value, 0)),
            )
        except ValueError as exc:
            raise _invalid_cognition_update(exc) from exc
        snapshot = actor.private_state.snapshot()
        supplied_seats = sorted({
            int(item["seat"])
            for item in args.get("beliefs", [])
            if isinstance(item, Mapping) and _positive_int(item.get("seat")) is not None
        })
        return {
            "updated": True,
            "revision": snapshot["revision"],
            "model_supplied_seats": supplied_seats,
            "belief_seats": sorted(int(seat) for seat in snapshot.get("beliefs", {})),
            "hard_fact_overrides": overrides,
        }

    return handler


def _update_belief(actor: Any, obs: AgentObservation):
    async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        known_wolves, known_village = _belief_facts(obs)
        belief_args = dict(args)
        belief_args.setdefault("likely_role", None)
        try:
            actor.private_state.update_belief(
                belief_args,
                visible_seats={int(item.get("seat")) for item in obs.seats if _positive_int(item.get("seat")) is not None},
                day=int(obs.day),
                phase=str(obs.phase),
                known_wolf_seats=known_wolves,
                known_village_seats=known_village,
                total_wolves=int(obs.role_counts.get(Role.WEREWOLF.value, 0)),
            )
        except ValueError as exc:
            raise _invalid_cognition_update(exc) from exc
        return {"updated": int(args["seat"]), "revision": actor.private_state.snapshot()["revision"]}
    return handler


def _update_beliefs(actor: Any, obs: AgentObservation):
    async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        known_wolves, known_village = _belief_facts(obs)
        try:
            updated = actor.private_state.update_beliefs(
                args["beliefs"],
                visible_seats={
                    int(item.get("seat"))
                    for item in obs.seats
                    if _positive_int(item.get("seat")) is not None
                },
                day=int(obs.day),
                phase=str(obs.phase),
                known_wolf_seats=known_wolves,
                known_village_seats=known_village,
                total_wolves=int(obs.role_counts.get(Role.WEREWOLF.value, 0)),
            )
        except ValueError as exc:
            raise _invalid_cognition_update(exc) from exc
        return {
            "updated": list(updated),
            "revision": actor.private_state.snapshot()["revision"],
        }
    return handler


def _belief_facts(obs: AgentObservation) -> tuple[set[int], set[int]]:
    known_wolves = {
        int(item.get("seat"))
        for item in obs.my_teammates
        if isinstance(item, Mapping) and _positive_int(item.get("seat")) is not None
    }
    known_wolves.update(_checked_seats(obs, expected={"wolf", "werewolf", "werewolves"}))
    known_village = _checked_seats(obs, expected={"village", "villager", "good"})
    return known_wolves, known_village


def _invalid_cognition_update(error: ValueError | str) -> ToolExecutionError:
    """Return a bounded, actionable constraint without echoing model text."""
    text = str(error).strip().lower()
    if "at most one patch per seat" in text:
        constraint = "duplicate_seat_patch"
    elif "visible to this agent" in text:
        constraint = "seat_not_visible"
    elif "known wolf" in text:
        constraint = "known_wolf_conflict"
    elif "known village" in text:
        constraint = "known_village_conflict"
    elif "probabilit" in text or "confidence" in text:
        constraint = "probability_bounds"
    elif "evidence" in text:
        constraint = "evidence_required"
    elif "belief updates" in text:
        constraint = "belief_array_shape"
    else:
        constraint = "private_state_constraint"
    return ToolExecutionError(
        "invalid_cognition_update",
        f"belief update rejected ({constraint}); correct the arguments and retry",
        details={"constraint": constraint},
    )


def _belief_schema(
    valid_seats: tuple[int, ...],
    *,
    known_wolves: set[int] | None = None,
    known_village: set[int] | None = None,
) -> dict[str, Any]:
    constraints = _belief_constraint_description(
        known_wolves=known_wolves or set(),
        known_village=known_village or set(),
        total_wolves=None,
    )
    return {
        "type": "object",
        "description": constraints,
        "properties": {
            "seat": _seat_schema(valid_seats),
            "wolf_probability": {"type": "number", "minimum": 0, "maximum": 1},
            "likely_role": {
                "anyOf": [
                    {
                        "type": "string",
                        "enum": [
                            *_ROLE_VALUES,
                            "unknown",
                            "uncertain",
                            "none",
                            "null",
                        ],
                    },
                    {"type": "null"},
                ],
                "description": "Use null (or the compatibility strings unknown/uncertain/none/null) when role is not known.",
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {
                "description": "One or more concise evidence references; do not repeat the same seat in a batch.",
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 6,
            },
        },
        "required": ["seat", "wolf_probability", "confidence", "evidence"],
        "additionalProperties": False,
    }


def _private_state_schema(
    valid_seats: tuple[int, ...],
    *,
    known_wolves: set[int],
    known_village: set[int],
    total_wolves: int,
) -> dict[str, Any]:
    return {
        "type": "object",
        "description": (
            "One atomic private cognition revision. "
            + _belief_constraint_description(
                known_wolves=known_wolves,
                known_village=known_village,
                total_wolves=total_wolves,
            )
        ),
        "properties": {
            "beliefs": {
                "type": "array",
                "minItems": 0,
                "maxItems": 12,
                "uniqueItems": True,
                "items": _belief_schema(
                    valid_seats,
                    known_wolves=known_wolves,
                    known_village=known_village,
                ),
            },
            "candidate_plans": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 700},
                "minItems": 2,
                "maxItems": 4,
                "uniqueItems": True,
            },
            "selected_plan": {"type": "string", "minLength": 1, "maxLength": 900},
            "public_cover_role": {
                "anyOf": [
                    {"type": "string", "enum": [*_ROLE_VALUES, "unclaimed", "none", "null"]},
                    {"type": "null"},
                ]
            },
            "perceived_image": {"type": "string", "minLength": 1, "maxLength": 700},
            "deception_plan": {
                "anyOf": [
                    {"type": "string", "maxLength": 700},
                    {"type": "null"},
                ]
            },
            "team_plan": {
                "anyOf": [
                    {"type": "string", "maxLength": 700},
                    {"type": "null"},
                ]
            },
        },
        "required": [
            "beliefs",
            "candidate_plans",
            "selected_plan",
            "public_cover_role",
            "perceived_image",
            "deception_plan",
            "team_plan",
        ],
        "additionalProperties": False,
    }


def _belief_constraint_description(
    *,
    known_wolves: set[int],
    known_village: set[int],
    total_wolves: int | None,
) -> str:
    clauses = [
        "Environment-visible hard facts override subjective estimates.",
    ]
    if known_wolves:
        clauses.append(
            "Known wolf seats "
            + ",".join(str(seat) for seat in sorted(known_wolves))
            + " require wolf_probability=1.0 and likely_role werewolf or null."
        )
    if known_village:
        clauses.append(
            "Known village seats "
            + ",".join(str(seat) for seat in sorted(known_village))
            + " require wolf_probability=0.0 and likely_role must not be werewolf."
        )
    if total_wolves is not None and total_wolves > 0:
        clauses.append(
            f"The configured game contains {int(total_wolves)} wolves in total; "
            "unknown-seat weights are projected onto the remaining probability mass."
        )
    return " ".join(clauses)


def _hard_fact_overrides(
    raw_beliefs: Any,
    *,
    known_wolves: set[int],
    known_village: set[int],
) -> list[dict[str, Any]]:
    overrides: list[dict[str, Any]] = []
    if not isinstance(raw_beliefs, (list, tuple)):
        return overrides
    for item in raw_beliefs:
        if not isinstance(item, Mapping):
            continue
        seat = _positive_int(item.get("seat"))
        probability = item.get("wolf_probability")
        likely_role = str(item.get("likely_role") or "").strip().lower()
        if seat in known_wolves and (
            probability != 1.0 or likely_role not in {"", "none", "null", Role.WEREWOLF.value}
        ):
            overrides.append({"seat": seat, "constraint": "known_wolf", "wolf_probability": 1.0})
        elif seat in known_village and (
            probability != 0.0 or likely_role == Role.WEREWOLF.value
        ):
            overrides.append({"seat": seat, "constraint": "known_village", "wolf_probability": 0.0})
    return overrides


def _set_plan(actor: Any):
    async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "selected_plan": args["selected_plan"],
        }
        for key in ("candidate_plans", "perceived_image", "deception_plan", "team_plan"):
            if key in args:
                kwargs[key] = args[key]
        try:
            actor.private_state.set_plan(**kwargs)
        except ValueError as exc:
            raise _invalid_cognition_update(exc) from exc
        return {"updated": True, "revision": actor.private_state.snapshot()["revision"]}
    return handler


def _set_cover(actor: Any):
    async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        actor.private_state.set_public_cover(args.get("role"))
        return {
            "public_cover_role": actor.private_state.snapshot().get("public_cover_role"),
            "revision": actor.private_state.snapshot()["revision"],
        }
    return handler


def _record_private_note(actor: Any, request: Any):
    async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> dict[str, Any]:
        actor.memory.observe(
            int(request.day),
            str(request.phase),
            "agent_private_note",
            str(args["note"])[:1200],
            owner_seat=int(actor.seat),
        )
        return {"recorded": True, "memory_digest": actor.memory.digest()}
    return handler


def _terminal_tool(
    actor: Any,
    request: Any,
    obs: AgentObservation,
    *,
    other_seats: tuple[int, ...],
    alive_other_seats: tuple[int, ...],
    legal_targets: tuple[int, ...],
):
    action_kind = str(request.action_kind)
    if action_kind in {"speak"}:
        schema = {
            "type": "object",
            "properties": {
                "speech": {"type": "string", "minLength": 1, "maxLength": 4000},
                "bid": {"type": "integer", "minimum": 1, "maximum": 4},
                "claim": _claim_schema(other_seats),
                "reply_to": {"anyOf": [_seat_schema(other_seats), {"type": "null"}]},
                "accuses": {"type": "array", "items": _seat_schema(alive_other_seats), "maxItems": 16},
            },
            "required": ["speech", "bid"],
            "additionalProperties": False,
        }

        async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> Decision:
            claim = actor._sanitize_claim(args.get("claim"), obs)
            reply_to = _valid_other_seat(args.get("reply_to"), obs)
            accuses = _valid_alive_other_seats(args.get("accuses"), obs)
            return Decision(
                action=AgentAction.SPEAK,
                speech=_exact_nonblank_text(args["speech"], field="speech"),
                bid=int(args["bid"]),
                claim=claim,
                reply_to=reply_to,
                accuses=accuses or None,
            )
        return handler, schema

    if action_kind == "vote":
        schema = _target_schema(legal_targets, required=True)

        async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> Decision:
            return Decision(action=AgentAction.VOTE, target_seat=int(args["target_seat"]))
        return handler, schema

    if action_kind == "wolf_council":
        schema = {
            "type": "object",
            "properties": {
                "target_seat": _seat_schema(legal_targets),
                "team_message": {"type": "string", "minLength": 1, "maxLength": 2000},
            },
            "required": ["target_seat", "team_message"],
            "additionalProperties": False,
        }

        async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> Decision:
            return Decision(
                action=AgentAction.WOLF_COUNCIL,
                target_seat=int(args["target_seat"]),
                team_message=_exact_nonblank_text(
                    args["team_message"],
                    field="team_message",
                ),
            )
        return handler, schema

    if action_kind == "last_words":
        schema = {
            "type": "object",
            "properties": {"speech": {"type": "string", "minLength": 1, "maxLength": 4000}},
            "required": ["speech"],
            "additionalProperties": False,
        }

        async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> Decision:
            return Decision(
                action=AgentAction.LAST_WORDS,
                speech=_exact_nonblank_text(args["speech"], field="speech"),
            )
        return handler, schema

    target_schema = _target_schema(legal_targets, required=True)

    async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> Decision:
        action = {
            "night_kill": AgentAction.NIGHT_KILL,
            "kill": AgentAction.NIGHT_KILL,
            "hunter_shot": AgentAction.NIGHT_KILL,
            "see": AgentAction.SEE,
            "save": AgentAction.SAVE,
            "poison": AgentAction.POISON,
            "guard": AgentAction.GUARD,
        }.get(action_kind)
        if action is None:
            raise ValueError(f"unsupported terminal action: {action_kind}")
        return Decision(action=action, target_seat=int(args["target_seat"]))
    return handler, target_schema


def _skip_terminal(request: Any):
    async def handler(_ctx: ToolExecutionContext, args: dict[str, Any]) -> Decision:
        return Decision(action=AgentAction.SKIP, skip_reason=str(args["reason"]).strip())
    return handler


def _terminal_description(action_kind: str) -> str:
    return (
        f"Submit the exact terminal {action_kind} action for this request. "
        "The environment validates the target and consumes it once."
    )


def _target_schema(targets: tuple[int, ...], *, required: bool) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"target_seat": _seat_schema(targets)},
        "required": ["target_seat"] if required else [],
        "additionalProperties": False,
    }


def _claim_schema(valid_seats: tuple[int, ...]) -> dict[str, Any]:
    return {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": list(_ROLE_VALUES)},
                    "checked_seat": _seat_schema(valid_seats),
                    "result": {"type": "string", "enum": ["wolf", "village"]},
                },
                "required": ["role"],
                "additionalProperties": False,
            },
            {"type": "null"},
        ]
    }


def _public_event(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"text": str(value), "untrusted_game_data": True}
    result = _sanitize_public_value(value)
    result["untrusted_game_data"] = True
    return result


def _sanitize_public_value(value: Any, *, claim: bool = False) -> Any:
    """Recursively remove private fields while preserving public role claims."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.startswith("_"):
                continue
            if key in _PUBLIC_HIDDEN_KEYS and not (claim and key == "role"):
                continue
            result[key] = _sanitize_public_value(item, claim=key == "claim")
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize_public_value(item, claim=claim) for item in value]
    return deepcopy(value)


def _bounded_event(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"text": str(value)}
    return {
        str(key): deepcopy(item)
        for key, item in value.items()
        if str(key) not in {"reasoning", "thought"} and not str(key).startswith("_")
    }


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _exact_nonblank_text(value: Any, *, field: str) -> str:
    """Validate text without rewriting the Agent's accepted utterance."""
    text = str(value)
    if not text.strip():
        raise ValueError(f"{field} must contain non-whitespace text")
    return text


def _checked_seats(obs: AgentObservation, *, expected: set[str]) -> set[int]:
    seats: set[int] = set()
    for event in obs.private_events:
        if event.get("type") != "seer_result":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        team = str(payload.get("team") or "").strip().lower()
        seat = _positive_int(payload.get("target_seat"))
        if seat is not None and team in expected:
            seats.add(seat)
    return seats


def _valid_other_seat(value: Any, obs: AgentObservation) -> int | None:
    seat = _positive_int(value)
    if seat is None or seat == obs.my_seat:
        return None
    return seat if any(int(item.get("seat")) == seat for item in obs.seats if _positive_int(item.get("seat")) is not None) else None


def _valid_alive_other_seats(value: Any, obs: AgentObservation) -> list[int]:
    """Keep current accusations scoped to visible, living opponents."""
    if not isinstance(value, (list, tuple)):
        return []
    visible = {
        seat
        for item in obs.seats
        if isinstance(item, Mapping)
        if (seat := _positive_int(item.get("seat"))) is not None
    }
    alive = {
        seat
        for value in obs.alive_seats
        if (seat := _positive_int(value)) is not None
    }
    allowed = (visible & alive) - {obs.my_seat}
    result: list[int] = []
    for item in value:
        seat = _positive_int(item)
        if seat in allowed and seat not in result:
            result.append(seat)
    return result
