"""Transcript visibility and audience leak checks."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..privacy import MODEL_PRIVATE_REASONING_KEYS, strip_model_private_reasoning


PRIVATE_EVENT_TYPES = {
    "role_assigned",
    "seer_result",
    "night_action_submitted",
    "action_rejected",
    "council_role_assigned",
    "council_mission_commitment",
    "council_cipher_message",
}
ADMIN_EVENT_TYPES = {"analysis"}
RESTRICTED_VISIBILITIES = {"god", "admin", "team"}
PUBLIC_EVENT_TYPES = {
    "phase_started",
    "night_resolved",
    "speech",
    "vote_cast",
    "vote_resolved",
    "vote_incomplete",
    "vote_rejected",
    "last_words",
    "last_words_skipped",
    "hunter_shot",
    "game_ended",
    "room_status",
    "game_error",
    "agent_decision_failed",
    "decision_envelope_rejected",
    "decision_validation_failed",
}
FORBIDDEN_PUBLIC_TOP_LEVEL_KEYS = {
    "role",
    "team",
    "teammates",
    "private_context",
    *MODEL_PRIVATE_REASONING_KEYS,
}
_PUBLIC_HIDDEN_FIELD_KEYS = frozenset({
    *FORBIDDEN_PUBLIC_TOP_LEVEL_KEYS,
    "private_state",
    "private_events",
    "private_memory",
    "private_facts",
    "private_plan",
    "team_plan",
    "deception_plan",
    "hidden_state",
    "secret",
    "secrets",
    "cipher_teammates",
    "cipher_council_messages",
    "wolf_council_messages",
    "wolf_council_message",
    "team_message",
})
_PUBLIC_HIDDEN_FIELD_PREFIXES = ("private_", "secret_", "hidden_")
_PUBLIC_HIDDEN_FIELD_SUFFIXES = ("_private", "_role", "_team")
PUBLIC_EVENT_TYPES_FOR_PROJECTION = PUBLIC_EVENT_TYPES


class VisibilityIssue(BaseModel):
    """One transcript visibility/audience problem."""

    model_config = ConfigDict(extra="forbid")

    row_index: int
    seq: int | None = None
    code: str
    severity: Literal["warning", "error"]
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


def project_transcript_rows(
    rows: list[dict[str, Any]],
    *,
    audience: Literal["public", "player", "god", "admin"],
    seat: int | None = None,
    player_id: str | None = None,
) -> list[dict[str, Any]]:
    """Project an omniscient transcript for a specific audience.

    `admin` is the full local research artifact. `god` keeps omniscient events
    but strips internal routing fields. `public` and `player` are safe client
    projections; player mode additionally receives rows explicitly addressed to
    that seat/player.
    """
    projected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        visible_payload = project_payload_for_audience(
            _payload(row),
            kind=str(row.get("kind") or ""),
            audience=audience,
            seat=seat,
            player_id=player_id,
        )
        if visible_payload is None:
            continue
        projected.append({
            key: value
            for key, value in row.items()
            if key != "payload"
        } | {"payload": visible_payload})
    return projected


def project_payload_for_audience(
    payload: Mapping[str, Any],
    *,
    kind: str = "event",
    audience: Literal["public", "player", "god", "admin"],
    seat: int | None = None,
    player_id: str | None = None,
) -> dict[str, Any] | None:
    """Return a safe audience-specific payload, or None if hidden."""
    # Decision rows are the sole admin-only reasoning channel. Even the admin
    # projection of an event row is sanitized so a malformed event cannot turn
    # into a second reasoning transport by accident.
    raw = (
        strip_model_private_reasoning(payload)
        if kind == "event"
        else dict(payload)
    )
    if audience == "admin":
        return raw
    if audience == "god":
        if kind == "decision":
            return None
        return _strip_transport_fields(_omniscient_payload(raw))
    if kind == "decision":
        return None
    if not _client_should_receive(raw, audience=audience, seat=seat, player_id=player_id):
        return None
    return _client_payload(raw, audience=audience)


def infer_visibility(row: Mapping[str, Any]) -> str:
    """Infer a coarse channel for a transcript row."""
    payload = _payload(row)
    kind = str(row.get("kind") or "")
    event_type = str(payload.get("type") or "")
    explicit = str(payload.get("visibility") or row.get("visibility") or "")
    recipients = payload.get("recipients")
    has_recipients = isinstance(recipients, list) and bool(recipients)
    if explicit == "private" or has_recipients:
        return "private"
    if explicit in RESTRICTED_VISIBILITIES:
        return explicit
    if kind == "decision":
        return "admin"
    if event_type in ADMIN_EVENT_TYPES:
        return "admin"
    if event_type in PRIVATE_EVENT_TYPES:
        return "private_expected"
    if explicit == "public" or event_type in PUBLIC_EVENT_TYPES:
        return "public"
    return "system"


def audit_transcript_visibility(rows: list[dict[str, Any]]) -> list[VisibilityIssue]:
    """Scan transcript rows for private/public channel mistakes."""
    issues: list[VisibilityIssue] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            issues.append(_issue(idx, None, "row_not_object", "error", "Transcript row is not an object."))
            continue
        payload = _payload(row)
        seq = _as_int(row.get("seq"))
        kind = str(row.get("kind") or "")
        event_type = str(payload.get("type") or "")
        explicit = str(payload.get("visibility") or row.get("visibility") or "")
        recipients = payload.get("recipients")
        has_recipients = isinstance(recipients, list) and bool(recipients)
        inferred = infer_visibility(row)

        if event_type in PRIVATE_EVENT_TYPES:
            if explicit != "private" and not has_recipients:
                issues.append(_issue(
                    idx,
                    seq,
                    "private_event_without_private_visibility",
                    "error",
                    "Private game event lacks private visibility or recipients.",
                    type=event_type,
                    inferred=inferred,
                ))
            if not has_recipients:
                issues.append(_issue(
                    idx,
                    seq,
                    "private_event_without_recipients",
                    "error",
                    "Private game event has no recipient list.",
                    type=event_type,
                ))
        if explicit == "private" and not has_recipients:
            issues.append(_issue(
                idx,
                seq,
                "private_visibility_without_recipients",
                "error",
                "Private visibility requires explicit recipients.",
                type=event_type,
            ))
        if event_type in ADMIN_EVENT_TYPES and explicit == "public":
            issues.append(_issue(
                idx,
                seq,
                "admin_event_public",
                "error",
                "Admin/god analysis event must not be public.",
                type=event_type,
            ))
        if explicit in RESTRICTED_VISIBILITIES and event_type in PUBLIC_EVENT_TYPES:
            issues.append(_issue(
                idx,
                seq,
                "restricted_visibility_public_event_type",
                "warning",
                "Restricted channel event uses a public event type; client projection must keep it hidden.",
                type=event_type,
                visibility=explicit,
            ))
        if inferred == "public":
            issues.extend(_public_payload_issues(idx, seq, payload))
    return issues


def _client_should_receive(
    payload: Mapping[str, Any],
    *,
    audience: Literal["public", "player"],
    seat: int | None,
    player_id: str | None,
) -> bool:
    event_type = str(payload.get("type") or "")
    recipients = payload.get("recipients") or []
    explicit = str(payload.get("visibility") or "")
    # A generic environment may define public event names outside the
    # Werewolf-specific allowlist. It must opt in explicitly, but a malformed
    # plugin can never override the hard private/admin classification of the
    # existing protocol event types.
    if event_type in ADMIN_EVENT_TYPES:
        return False
    # Known private protocol events still need to reach their explicitly
    # addressed player. A bare/malicious ``visibility: public`` label cannot
    # make one visible because it has no private recipient route.
    if event_type in PRIVATE_EVENT_TYPES and not (
        explicit == "private"
        or (isinstance(recipients, list) and bool(recipients))
    ):
        return False
    if explicit in RESTRICTED_VISIBILITIES:
        return False
    if explicit == "private" or recipients:
        if audience != "player":
            return False
        recipient_set = {str(item) for item in recipients} if isinstance(recipients, list) else set()
        return bool(
            (player_id and player_id in recipient_set)
            or (seat is not None and str(seat) in recipient_set)
            or (seat is not None and f"seat-{seat}" in recipient_set)
        )
    if event_type == "human_action_request":
        return audience == "player" and seat is not None and payload.get("seat") == seat
    if explicit == "public":
        return True
    return event_type in PUBLIC_EVENT_TYPES_FOR_PROJECTION


def _client_payload(payload: Mapping[str, Any], *, audience: Literal["public", "player"]) -> dict[str, Any]:
    private_like = _is_private_client_payload(payload)
    visible = _strip_transport_fields(payload)
    for key in list(visible):
        if str(key).startswith("_analysis_"):
            visible.pop(key, None)
    event_type = visible.get("type")
    if event_type == "night_resolved":
        visible["deaths"] = [
            {"seat": item.get("seat"), "name": item.get("name")}
            for item in visible.get("deaths", [])
            if isinstance(item, Mapping)
        ]
    elif event_type in {
        "agent_decision_failed",
        "decision_envelope_rejected",
        "decision_validation_failed",
    }:
        phase = str(visible.get("phase") or "")
        agent_kind = "human" if visible.get("agent_kind") == "human" else "llm"
        envelope_rejected = event_type == "decision_envelope_rejected"
        validator_failed = event_type == "decision_validation_failed"
        sanitized: dict[str, Any] = {
            "type": event_type,
            "phase": phase,
            "reason": (
                "Harness 校验器失败,本请求未被执行;该故障不归因于 Agent。"
                if validator_failed
                else (
                    "DecisionEnvelope 未通过协议校验,本请求未被执行。"
                    if envelope_rejected
                    else (
                        "真人决策超时或失败,本请求未产生 DecisionEnvelope。"
                        if agent_kind == "human"
                        else "AI 决策失败,本请求未产生 DecisionEnvelope。"
                    )
                )
            ),
        }
        if visible.get("request_id"):
            sanitized["request_id"] = str(visible.get("request_id"))
        if visible.get("error_type"):
            sanitized["error_type"] = str(visible.get("error_type"))
        sanitized["agent_kind"] = agent_kind
        if phase in {"day", "voting", "pk", "last_words", "hunter"} and visible.get("seat") is not None:
            sanitized["seat"] = visible.get("seat")
            if visible.get("action"):
                sanitized["action"] = str(visible.get("action"))
        if bool(visible.get("timeout")):
            sanitized["timeout"] = True
            if visible.get("timeout_seconds") is not None:
                sanitized["timeout_seconds"] = visible.get("timeout_seconds")
        return sanitized
    if not private_like:
        visible = _strip_public_hidden_fields(visible)
    return visible


def _is_private_client_payload(payload: Mapping[str, Any]) -> bool:
    recipients = payload.get("recipients")
    event_type = str(payload.get("type") or "")
    return (
        str(payload.get("visibility") or "") == "private"
        or (isinstance(recipients, list) and bool(recipients))
        or event_type == "human_action_request"
    )


def _omniscient_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(payload)


def _strip_public_hidden_fields(
    value: Any,
    *,
    claim_object: bool = False,
) -> Any:
    """Remove hidden fields recursively from a public/client event payload.

    Environment events are authoritative only after their visibility boundary
    is applied. This second, defensive pass prevents a malformed or hostile
    plugin payload from smuggling a private state tree through an otherwise
    public event. A structured public role claim remains an intentional
    exception: only the direct ``role`` field of a ``claim`` object is public.
    """
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = _normalized_public_field_name(key)
            if key.startswith("_"):
                continue
            if _is_hidden_public_field(normalized) and not (
                claim_object and normalized == "role"
            ):
                continue
            sanitized[key] = _strip_public_hidden_fields(
                item,
                claim_object=normalized == "claim",
            )
        return sanitized
    if isinstance(value, list):
        return [
            _strip_public_hidden_fields(item, claim_object=claim_object)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _strip_public_hidden_fields(item, claim_object=claim_object)
            for item in value
        ]
    return value


def _normalized_public_field_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _is_hidden_public_field(normalized: str) -> bool:
    return (
        normalized in _PUBLIC_HIDDEN_FIELD_KEYS
        or normalized.startswith(_PUBLIC_HIDDEN_FIELD_PREFIXES)
        or normalized.endswith(_PUBLIC_HIDDEN_FIELD_SUFFIXES)
    )


def _strip_transport_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    visible = dict(payload)
    visible.pop("visibility", None)
    visible.pop("recipients", None)
    return visible


def _public_payload_issues(row_index: int, seq: int | None, payload: Mapping[str, Any]) -> list[VisibilityIssue]:
    issues: list[VisibilityIssue] = []
    event_type = str(payload.get("type") or "")
    for key in sorted(FORBIDDEN_PUBLIC_TOP_LEVEL_KEYS):
        if key in payload:
            issues.append(_issue(
                row_index,
                seq,
                "public_hidden_top_level_field",
                "error",
                "Public event contains a hidden/private top-level field.",
                type=event_type,
                field=key,
            ))
    private_context_paths = _find_key_paths(payload, key_name="private_context")
    for path in private_context_paths:
        issues.append(_issue(
            row_index,
            seq,
            "public_private_context_field",
            "error",
            "Public event contains private_context.",
            type=event_type,
            path=path,
        ))
    for path, field in _find_public_hidden_paths(payload):
        normalized = _normalized_public_field_name(field)
        # Existing checks retain their stable, more specific codes.
        if normalized == "private_context":
            continue
        if path == field and normalized in FORBIDDEN_PUBLIC_TOP_LEVEL_KEYS:
            continue
        issues.append(_issue(
            row_index,
            seq,
            "public_hidden_nested_field",
            "error",
            "Public event contains a nested hidden/private field.",
            type=event_type,
            path=path,
            field=field,
        ))
    return issues


def _find_key_paths(value: Any, *, prefix: str | None = None, key_name: str | None = None, path: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            current = f"{path}.{key_text}" if path else key_text
            if (prefix and key_text.startswith(prefix)) or (key_name and key_text == key_name):
                paths.append(current)
            paths.extend(_find_key_paths(item, prefix=prefix, key_name=key_name, path=current))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            current = f"{path}[{idx}]" if path else f"[{idx}]"
            paths.extend(_find_key_paths(item, prefix=prefix, key_name=key_name, path=current))
    return paths


def _find_public_hidden_paths(
    value: Any,
    *,
    path: str = "",
    claim_object: bool = False,
) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            normalized = _normalized_public_field_name(key_text)
            current = f"{path}.{key_text}" if path else key_text
            hidden = _is_hidden_public_field(normalized) and not (
                claim_object and normalized == "role"
            )
            if hidden:
                paths.append((current, key_text))
                continue
            paths.extend(_find_public_hidden_paths(
                item,
                path=current,
                claim_object=normalized == "claim",
            ))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            current = f"{path}[{idx}]" if path else f"[{idx}]"
            paths.extend(_find_public_hidden_paths(
                item,
                path=current,
                claim_object=claim_object,
            ))
    return paths


def _payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _issue(
    row_index: int,
    seq: int | None,
    code: str,
    severity: Literal["warning", "error"],
    message: str,
    **evidence: Any,
) -> VisibilityIssue:
    return VisibilityIssue(
        row_index=row_index,
        seq=seq,
        code=code,
        severity=severity,
        message=message,
        evidence=evidence,
    )
