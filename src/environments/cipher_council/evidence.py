"""Offline, content-free evidence verification for Cipher Council v2 artifacts.

The generic artifact verifier establishes file integrity and transcript shape.
This module adds the environment-specific facts needed to substantiate v2's
private faction-coordination contract without replaying Agents or exposing
private strategy messages in its report.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...harness.artifacts import load_verified_artifact_snapshot
from ...harness.core_spec import CoreRunManifest


CIPHER_COUNCIL_V2_ARTIFACT_EVIDENCE_VERSION = (
    "council.cipher.v2-artifact-evidence.v1"
)
_CIPHER_COUNCIL_ID = "council.cipher"
_CIPHER_COUNCIL_V2 = "2"
_CIPHER_COUNCIL_ACTION = "send_cipher_strategy_message"
_TERMINAL_RESPONSE_KINDS = frozenset({
    "agent_response",
    "agent_response_failed",
    "agent_response_cancelled",
    "agent_response_validation_failed",
})

RoundKey = tuple[int, int]


class CipherCouncilArtifactEvidenceError(ValueError):
    """Raised when a verified artifact lacks v2 coordination evidence."""


class CipherCouncilV2ArtifactEvidence(BaseModel):
    """Safe recomputation report with no strategy text or model reasoning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "council.cipher.v2-artifact-evidence.v1"
    ] = CIPHER_COUNCIL_V2_ARTIFACT_EVIDENCE_VERSION
    run_id: str = Field(min_length=1)
    transcript_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    cipher_faction_size: int = Field(ge=1)
    cipher_council_round_count: int = Field(ge=0)
    cipher_council_request_count: int = Field(ge=0)
    cipher_council_message_count: int = Field(ge=0)
    cipher_council_absent_count: int = Field(ge=0)
    message_delivery_barrier_verified: Literal[True] = True
    observation_isolation_verified: Literal[True] = True


def verify_cipher_council_v2_artifacts(
    run_dir: str | Path,
) -> CipherCouncilV2ArtifactEvidence:
    """Verify v2 faction-coordination evidence from one committed artifact.

    File hashes, manifest/result identity and transcript structure are checked
    into one in-memory snapshot before this verifier recomputes coordination
    facts and compares them to the stored summary metrics. It never contacts a
    provider and its return value deliberately excludes private message
    content, tool arguments and reasoning.
    """
    snapshot = load_verified_artifact_snapshot(run_dir)
    manifest = snapshot.manifest
    if not isinstance(manifest, CoreRunManifest):
        raise CipherCouncilArtifactEvidenceError(
            "Cipher Council v2 evidence requires a Core artifact manifest"
        )
    if (
        manifest.run.environment.id != _CIPHER_COUNCIL_ID
        or manifest.run.environment.version != _CIPHER_COUNCIL_V2
    ):
        raise CipherCouncilArtifactEvidenceError(
            "artifact is not council.cipher@2"
        )

    return _verify_rows(
        run_id=manifest.run.run_id,
        transcript_digest=manifest.transcript_digest,
        environment_config=manifest.run.environment_config,
        summary=snapshot.summary,
        rows=snapshot.transcript_rows,
    )


def _verify_rows(
    *,
    run_id: str,
    transcript_digest: str,
    environment_config: Mapping[str, Any],
    summary: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> CipherCouncilV2ArtifactEvidence:
    expected_player_count = _require_nonnegative_int(
        environment_config.get("player_names_count"),
        label="environment_config.player_names_count",
        allow_missing=True,
    )
    players = environment_config.get("player_names")
    if isinstance(players, list):
        expected_player_count = len(players)
    if expected_player_count is None or expected_player_count < 1:
        raise CipherCouncilArtifactEvidenceError(
            "Cipher Council environment_config has no valid player_names"
        )
    expected_cipher_count = _require_nonnegative_int(
        environment_config.get("cipher_count"),
        label="environment_config.cipher_count",
    )
    if expected_cipher_count is None or expected_cipher_count < 1:
        raise CipherCouncilArtifactEvidenceError(
            "Cipher Council environment_config has no valid cipher_count"
        )

    roles: dict[str, str] = {}
    actor_order: list[str] = []
    public_rounds: dict[RoundKey, int] = {}
    cipher_requests: dict[str, dict[str, Any]] = {}
    cipher_requests_by_round: dict[RoundKey, dict[str, str]] = defaultdict(dict)
    terminal_sequences: dict[str, list[int]] = defaultdict(list)
    consumed_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    message_events: list[tuple[int, dict[str, Any]]] = []
    agent_observations: list[tuple[str, int, dict[str, Any]]] = []

    for row_index, row in enumerate(rows):
        payload = _mapping(row.get("payload"), label=f"transcript[{row_index}].payload")
        kind = str(row.get("kind") or "")
        seq = _require_sequence(row.get("seq"), row_index=row_index)
        event_type = str(payload.get("type") or "")

        if kind == "event" and event_type == "council_role_assigned":
            actor_id = _require_actor_id(payload.get("actor_id"), label="role assignment")
            role = str(payload.get("role") or "")
            if role not in {"cipher", "council"}:
                raise CipherCouncilArtifactEvidenceError(
                    "role assignment has an unsupported faction"
                )
            if actor_id in roles:
                raise CipherCouncilArtifactEvidenceError(
                    "Cipher Council role assignment is duplicated"
                )
            if payload.get("visibility") != "private" or payload.get("recipients") != [actor_id]:
                raise CipherCouncilArtifactEvidenceError(
                    "Cipher Council role assignment has an invalid private route"
                )
            roles[actor_id] = role
            actor_order.append(actor_id)
            continue

        if kind == "event" and event_type == "council_round_started":
            if payload.get("visibility") != "public":
                raise CipherCouncilArtifactEvidenceError(
                    "council_round_started must be explicitly public"
                )
            round_key = _round_key(payload, label="council_round_started")
            if round_key in public_rounds:
                raise CipherCouncilArtifactEvidenceError(
                    "council_round_started is duplicated for one round"
                )
            public_rounds[round_key] = seq
            continue

        if kind == "event" and event_type == "council_cipher_message":
            message_events.append((seq, payload))
            continue

        if kind != "decision":
            continue

        trace_kind = str(payload.get("kind") or "")
        if trace_kind == "agent_request":
            request = _mapping(payload.get("request"), label="agent_request.request")
            actor_id = _require_actor_id(request.get("actor_id"), label="agent_request")
            observation = _mapping(
                request.get("observation"),
                label="agent_request.observation",
            )
            identity = _mapping(
                observation.get("private_identity"),
                label="agent_request.private_identity",
            )
            agent_observations.append((actor_id, seq, identity))
            labels = _mapping(request.get("labels"), label="agent_request.labels")
            if labels.get("stage") != "cipher_council":
                continue
            request_id = _require_request_id(request.get("request_id"), label="cipher council request")
            if request_id in cipher_requests:
                raise CipherCouncilArtifactEvidenceError(
                    "Cipher council request_id is duplicated"
                )
            _verify_cipher_council_request_shape(request)
            round_key = _round_key(labels, label="cipher council request labels")
            cipher_requests[request_id] = {
                "actor_id": actor_id,
                "round": round_key,
                "seq": seq,
                "identity": identity,
            }
            if actor_id in cipher_requests_by_round[round_key]:
                raise CipherCouncilArtifactEvidenceError(
                    "one Cipher has multiple strategy requests in one round"
                )
            cipher_requests_by_round[round_key][actor_id] = request_id
            continue

        if trace_kind in _TERMINAL_RESPONSE_KINDS:
            request_id = payload.get("request_id")
            if isinstance(request_id, str) and request_id.strip():
                terminal_sequences[request_id.strip()].append(seq)
            continue

        if (
            payload.get("type") == "decision_consumed"
            and payload.get("stage") == "cipher_council"
        ):
            request_id = _require_request_id(
                payload.get("request_id"),
                label="cipher council consumed decision",
            )
            consumed_by_request[request_id].append(payload)

    cipher_ids = _verify_role_assignments(
        roles=roles,
        actor_order=actor_order,
        expected_player_count=expected_player_count,
        expected_cipher_count=expected_cipher_count,
    )
    _verify_observation_isolation(
        agent_observations=agent_observations,
        cipher_ids=set(cipher_ids),
        cipher_requests=cipher_requests,
        message_events=message_events,
    )

    eligible_rounds = set(public_rounds) if len(cipher_ids) >= 2 else set()
    _verify_scheduled_cipher_requests(
        cipher_ids=set(cipher_ids),
        eligible_rounds=eligible_rounds,
        requests_by_round=cipher_requests_by_round,
        terminal_sequences=terminal_sequences,
    )
    message_by_request = _verify_message_delivery(
        cipher_ids=cipher_ids,
        cipher_requests=cipher_requests,
        requests_by_round=cipher_requests_by_round,
        terminal_sequences=terminal_sequences,
        consumed_by_request=consumed_by_request,
        message_events=message_events,
    )
    _verify_consumed_strategy_actions(
        cipher_requests=cipher_requests,
        consumed_by_request=consumed_by_request,
        message_by_request=message_by_request,
    )

    request_count = len(cipher_requests)
    message_count = len(message_events)
    absent_count = request_count - message_count
    if absent_count < 0:
        raise CipherCouncilArtifactEvidenceError(
            "Cipher council message count exceeds scheduled requests"
        )
    expected_metrics = {
        "cipher_council_faction_size": len(cipher_ids),
        "cipher_council_round_count": len(eligible_rounds),
        "cipher_council_request_count": request_count,
        "cipher_council_message_count": message_count,
        "cipher_council_absent_count": absent_count,
    }
    metrics = _mapping(summary.get("metrics"), label="summary.metrics")
    for field, expected in expected_metrics.items():
        actual = _require_nonnegative_int(metrics.get(field), label=f"summary.metrics.{field}")
        if actual != expected:
            raise CipherCouncilArtifactEvidenceError(
                f"summary metric {field} does not match transcript evidence"
            )

    return CipherCouncilV2ArtifactEvidence(
        run_id=run_id,
        transcript_digest=transcript_digest,
        cipher_faction_size=len(cipher_ids),
        cipher_council_round_count=len(eligible_rounds),
        cipher_council_request_count=request_count,
        cipher_council_message_count=message_count,
        cipher_council_absent_count=absent_count,
    )


def _verify_role_assignments(
    *,
    roles: Mapping[str, str],
    actor_order: list[str],
    expected_player_count: int,
    expected_cipher_count: int,
) -> tuple[str, ...]:
    if len(roles) != expected_player_count:
        raise CipherCouncilArtifactEvidenceError(
            "role assignment count does not match the configured player count"
        )
    cipher_ids = tuple(actor_id for actor_id in actor_order if roles[actor_id] == "cipher")
    if len(cipher_ids) != expected_cipher_count:
        raise CipherCouncilArtifactEvidenceError(
            "Cipher faction size does not match the configured cipher_count"
        )
    if not cipher_ids:
        raise CipherCouncilArtifactEvidenceError("Cipher Council run has no Cipher faction")
    return cipher_ids


def _verify_observation_isolation(
    *,
    agent_observations: list[tuple[str, int, dict[str, Any]]],
    cipher_ids: set[str],
    cipher_requests: Mapping[str, Mapping[str, Any]],
    message_events: list[tuple[int, dict[str, Any]]],
) -> None:
    strategy_request_rounds = {
        id(request["identity"]): request["round"]
        for request in cipher_requests.values()
    }
    delivered_history = [
        (
            seq,
            _cipher_message_history_entry(
                payload,
                label="Cipher strategy message event",
            ),
        )
        for seq, payload in message_events
    ]
    for actor_id, request_seq, identity in agent_observations:
        has_messages = "cipher_council_messages" in identity
        if actor_id not in cipher_ids:
            if has_messages:
                raise CipherCouncilArtifactEvidenceError(
                    "a Council Actor observation contains Cipher council messages"
                )
            continue
        history = identity.get("cipher_council_messages")
        if not isinstance(history, list):
            raise CipherCouncilArtifactEvidenceError(
                "a Cipher observation lacks the bounded council-message history"
            )
        observed_history = [
            _cipher_message_history_entry(
                entry,
                label="Cipher observation council-message history entry",
                require_exact_keys=True,
            )
            for entry in history
        ]
        current_round = strategy_request_rounds.get(id(identity))
        if current_round is not None and any(
            _round_key(entry, label="cipher council message history entry")
            == current_round
            for entry in observed_history
        ):
            raise CipherCouncilArtifactEvidenceError(
                "a Cipher strategy request observed a current-round message"
            )
        available_history = [
            entry
            for event_seq, entry in delivered_history
            if event_seq < request_seq
        ]
        if observed_history and observed_history != available_history[-len(observed_history):]:
            raise CipherCouncilArtifactEvidenceError(
                "a Cipher observation contains messages not delivered before its request"
            )


def _verify_scheduled_cipher_requests(
    *,
    cipher_ids: set[str],
    eligible_rounds: set[RoundKey],
    requests_by_round: Mapping[RoundKey, Mapping[str, str]],
    terminal_sequences: Mapping[str, list[int]],
) -> None:
    if set(requests_by_round) != eligible_rounds:
        raise CipherCouncilArtifactEvidenceError(
            "Cipher strategy requests do not match the eligible public rounds"
        )
    for round_key in sorted(eligible_rounds):
        request_ids = requests_by_round[round_key]
        if set(request_ids) != cipher_ids:
            raise CipherCouncilArtifactEvidenceError(
                "each eligible Cipher council round must request every Cipher Actor"
            )
        for request_id in request_ids.values():
            if len(terminal_sequences.get(request_id, [])) != 1:
                raise CipherCouncilArtifactEvidenceError(
                    "each Cipher strategy request must have one response terminal"
                )


def _verify_message_delivery(
    *,
    cipher_ids: tuple[str, ...],
    cipher_requests: Mapping[str, Mapping[str, Any]],
    requests_by_round: Mapping[RoundKey, Mapping[str, str]],
    terminal_sequences: Mapping[str, list[int]],
    consumed_by_request: Mapping[str, list[dict[str, Any]]],
    message_events: list[tuple[int, dict[str, Any]]],
) -> set[str]:
    delivered: set[str] = set()
    seen_round_actor: set[tuple[RoundKey, str]] = set()
    for seq, event in message_events:
        if event.get("visibility") != "private" or event.get("recipients") != list(cipher_ids):
            raise CipherCouncilArtifactEvidenceError(
                "Cipher strategy message has an invalid private recipient route"
            )
        history_entry = _cipher_message_history_entry(
            event,
            label="Cipher strategy message",
        )
        actor_id = history_entry["actor_id"]
        round_key = _round_key(history_entry, label="Cipher strategy message")
        request_id = requests_by_round.get(round_key, {}).get(actor_id)
        if request_id is None or request_id not in cipher_requests:
            raise CipherCouncilArtifactEvidenceError(
                "Cipher strategy message has no matching scheduled request"
            )
        marker = (round_key, actor_id)
        if marker in seen_round_actor:
            raise CipherCouncilArtifactEvidenceError(
                "Cipher strategy message is duplicated for one Actor/round"
            )
        seen_round_actor.add(marker)
        round_request_ids = requests_by_round[round_key].values()
        terminal_sequences_for_round = [
            terminal_sequences[request_id][0]
            for request_id in round_request_ids
        ]
        if seq <= max(terminal_sequences_for_round):
            raise CipherCouncilArtifactEvidenceError(
                "Cipher strategy messages were emitted before the round barrier"
            )
        consumed = consumed_by_request.get(request_id, [])
        if len(consumed) != 1 or consumed[0].get("action") != _CIPHER_COUNCIL_ACTION:
            raise CipherCouncilArtifactEvidenceError(
                "Cipher strategy message is not backed by one consumed strategy action"
            )
        if consumed[0].get("actor_id") != actor_id:
            raise CipherCouncilArtifactEvidenceError(
                "Cipher strategy message actor does not match its consumed action"
            )
        delivered.add(request_id)
    return delivered


def _verify_consumed_strategy_actions(
    *,
    cipher_requests: Mapping[str, Mapping[str, Any]],
    consumed_by_request: Mapping[str, list[dict[str, Any]]],
    message_by_request: set[str],
) -> None:
    for request_id, request in cipher_requests.items():
        consumed = consumed_by_request.get(request_id, [])
        if len(consumed) > 1:
            raise CipherCouncilArtifactEvidenceError(
                "Cipher strategy request has multiple consumed decisions"
            )
        if not consumed:
            if request_id in message_by_request:
                raise CipherCouncilArtifactEvidenceError(
                    "Cipher strategy message lacks a consumed decision"
                )
            continue
        action = consumed[0].get("action")
        if action == _CIPHER_COUNCIL_ACTION:
            if request_id not in message_by_request:
                raise CipherCouncilArtifactEvidenceError(
                    "consumed Cipher strategy action has no private message event"
                )
        elif action == "skip":
            if request_id in message_by_request:
                raise CipherCouncilArtifactEvidenceError(
                    "skipped Cipher strategy request emitted a private message"
                )
        else:
            raise CipherCouncilArtifactEvidenceError(
                "Cipher strategy request consumed an unexpected action"
            )
        if consumed[0].get("actor_id") != request["actor_id"]:
            raise CipherCouncilArtifactEvidenceError(
                "consumed Cipher strategy actor does not match its request"
            )


def _verify_cipher_council_request_shape(request: Mapping[str, Any]) -> None:
    actions = request.get("legal_actions")
    if not isinstance(actions, list) or len(actions) != 1:
        raise CipherCouncilArtifactEvidenceError(
            "Cipher strategy request must advertise exactly one action option"
        )
    option = _mapping(actions[0], label="Cipher strategy action option")
    if option.get("name") != _CIPHER_COUNCIL_ACTION:
        raise CipherCouncilArtifactEvidenceError(
            "Cipher strategy request advertises the wrong action option"
        )
    schema = _mapping(
        option.get("input_schema"),
        label="Cipher strategy action input_schema",
    )
    properties = _mapping(
        schema.get("properties"),
        label="Cipher strategy action input_schema.properties",
    )
    message_schema = _mapping(
        properties.get("message"),
        label="Cipher strategy message input schema",
    )
    if (
        schema.get("type") != "object"
        or schema.get("required") != ["message"]
        or schema.get("additionalProperties") is not False
        or set(properties) != {"message"}
        or message_schema.get("type") != "string"
        or message_schema.get("minLength") != 1
        or message_schema.get("maxLength") != 1000
    ):
        raise CipherCouncilArtifactEvidenceError(
            "Cipher strategy request has an invalid private message schema"
        )
    metadata = _mapping(option.get("metadata"), label="Cipher strategy action metadata")
    if metadata.get("visibility") != "private" or metadata.get("stage") != "cipher_council":
        raise CipherCouncilArtifactEvidenceError(
            "Cipher strategy request does not mark its message tool private"
        )
    skip_policy = _mapping(request.get("skip_policy"), label="Cipher strategy skip policy")
    if skip_policy.get("allowed") is not True:
        raise CipherCouncilArtifactEvidenceError(
            "Cipher strategy request must allow an explicit absent message"
        )


def _cipher_message_history_entry(
    value: Any,
    *,
    label: str,
    require_exact_keys: bool = False,
) -> dict[str, Any]:
    entry = _mapping(value, label=label)
    if require_exact_keys and set(entry) != {
        "mission",
        "proposal_attempt",
        "actor_id",
        "message",
    }:
        raise CipherCouncilArtifactEvidenceError(
            f"{label} has unexpected private fields"
        )
    mission, proposal_attempt = _round_key(entry, label=label)
    actor_id = _require_actor_id(entry.get("actor_id"), label=label)
    message = entry.get("message")
    if not isinstance(message, str) or not message or len(message) > 1000:
        raise CipherCouncilArtifactEvidenceError(
            f"{label} has an invalid strategy message"
        )
    return {
        "mission": mission,
        "proposal_attempt": proposal_attempt,
        "actor_id": actor_id,
        "message": message,
    }


def _round_key(value: Mapping[str, Any], *, label: str) -> RoundKey:
    mission = _require_nonnegative_int(value.get("mission"), label=f"{label}.mission")
    attempt = _require_nonnegative_int(
        value.get("proposal_attempt"),
        label=f"{label}.proposal_attempt",
    )
    if mission is None or mission < 1 or attempt is None or attempt < 1:
        raise CipherCouncilArtifactEvidenceError(f"{label} has an invalid round identity")
    return mission, attempt


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CipherCouncilArtifactEvidenceError(f"{label} must be an object")
    return dict(value)


def _require_actor_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CipherCouncilArtifactEvidenceError(f"{label} has no actor_id")
    return value.strip()


def _require_request_id(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CipherCouncilArtifactEvidenceError(f"{label} has no request_id")
    return value.strip()


def _require_nonnegative_int(
    value: Any,
    *,
    label: str,
    allow_missing: bool = False,
) -> int | None:
    if value is None and allow_missing:
        return None
    if type(value) is not int or value < 0:
        raise CipherCouncilArtifactEvidenceError(f"{label} must be a non-negative integer")
    return value


def _require_sequence(value: Any, *, row_index: int) -> int:
    if type(value) is not int or value < 1:
        raise CipherCouncilArtifactEvidenceError(
            f"transcript[{row_index}] has an invalid sequence"
        )
    return value
