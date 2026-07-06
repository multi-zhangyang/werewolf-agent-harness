"""Visible evidence graph and soft role posterior for agents.

This module is deterministic harness-side structuring, not a decision maker.
It only reorganizes information already visible in ``AgentObservation``:
public claims, public attitudes/votes/deaths, and the viewer's own private
seer results when present. It must never infer from hidden true roles.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math
from typing import Any


def build_evidence_graph(obs: Any) -> dict[str, Any]:
    """Build a compact evidence graph from one player's visible observation.

    The output is intentionally soft: ``role_posterior`` is a heuristic
    suspicion summary with cited evidence, not role truth. Private evidence is
    included only when it is already present in ``obs.private_events`` for this
    viewer.
    """
    seats = _seat_numbers(obs)
    alive = {int(s) for s in getattr(obs, "alive_seats", []) if _to_int(s) is not None}
    id_to_seat = _id_to_seat(obs)
    phase = _phase_value(getattr(obs, "phase", None))

    posterior: dict[int, dict[str, Any]] = {
        seat: {
            "werewolf_suspicion": 0.5,
            "seer_claim": False,
            "alive": seat in alive,
            "evidence": [],
            "posterior_deltas": [],
        }
        for seat in seats
    }
    if getattr(obs, "my_seat", None) in posterior:
        mine = posterior[int(obs.my_seat)]
        mine["werewolf_suspicion"] = 0.0
        mine["evidence"].append("你知道自己的身份,该后验不用于判断自己")

    claims = _collect_claims(getattr(obs, "today_speeches", []) or [], phase)
    attitude_edges = _collect_attitude_edges(getattr(obs, "today_speeches", []) or [], phase)
    vote_edges = _collect_vote_edges(getattr(obs, "public_events", []) or [])
    death_events = _collect_death_events(getattr(obs, "public_events", []) or [], id_to_seat)
    private_results = _collect_private_seer_results(getattr(obs, "private_events", []) or [])
    claim_conflicts = _claim_conflicts(claims)

    _score_claims(claims, posterior)
    _score_claim_conflicts(claim_conflicts, posterior)
    _score_attitudes(attitude_edges, posterior)
    _score_votes(vote_edges, posterior)
    _score_private_results(private_results, posterior)

    posterior_deltas = [
        delta
        for seat, data in sorted(posterior.items())
        for delta in data.get("posterior_deltas", [])
    ]
    role_counts = _role_counts(obs, seats)
    legal_worlds = _legal_team_worlds(obs, seats, role_counts, private_results)
    constrained_marginals = _constrained_wolf_marginals(
        legal_worlds.get("_all_worlds", []),
        seats,
        {
            seat: _clamp01(float(data.get("werewolf_suspicion", 0.5)))
            for seat, data in posterior.items()
        },
    )
    for seat, value in constrained_marginals.items():
        if seat in posterior:
            posterior[seat]["constrained_werewolf_suspicion"] = value
    evidence_items = _build_evidence_items(
        phase=phase,
        claims=claims,
        claim_conflicts=claim_conflicts,
        attitude_edges=attitude_edges,
        vote_edges=vote_edges,
        death_events=death_events,
        private_results=private_results,
        posterior_deltas=posterior_deltas,
    )
    role_posterior = {
        str(seat): {
            **data,
            "werewolf_suspicion": _clamp01(float(data["werewolf_suspicion"])),
            "evidence": data["evidence"][:6],
            "posterior_deltas": data["posterior_deltas"][:12],
        }
        for seat, data in sorted(posterior.items())
    }
    top_suspects = sorted(
        (
            {"seat": int(seat), "werewolf_suspicion": data["werewolf_suspicion"]}
            for seat, data in role_posterior.items()
            if int(seat) != getattr(obs, "my_seat", None) and data.get("alive", False)
        ),
        key=lambda item: item["werewolf_suspicion"],
        reverse=True,
    )[:4]

    return {
        "claims": claims,
        "claim_conflicts": claim_conflicts,
        "attitude_edges": attitude_edges,
        "vote_edges": vote_edges,
        "death_events": death_events,
        "private_results": private_results,
        "role_counts": role_counts,
        "legal_worlds": _public_legal_worlds(legal_worlds),
        "evidence_items": evidence_items,
        "posterior_deltas": posterior_deltas,
        "role_posterior": role_posterior,
        "top_suspects": top_suspects,
    }


def _seat_numbers(obs: Any) -> list[int]:
    values: set[int] = set()
    for seat_info in getattr(obs, "seats", []) or []:
        seat = _to_int((seat_info or {}).get("seat"))
        if seat:
            values.add(seat)
    for seat in getattr(obs, "alive_seats", []) or []:
        normalized = _to_int(seat)
        if normalized:
            values.add(normalized)
    return sorted(values)


def _id_to_seat(obs: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for seat_info in getattr(obs, "seats", []) or []:
        pid = (seat_info or {}).get("id")
        seat = _to_int((seat_info or {}).get("seat"))
        if pid and seat:
            result[str(pid)] = seat
    return result


def _collect_claims(speeches: list[dict[str, Any]], phase: str | None) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for speech in speeches:
        claim = speech.get("claim")
        if not isinstance(claim, dict):
            continue
        claimer = _to_int(speech.get("seat"))
        if not claimer:
            continue
        normalized = {
            "claimer": claimer,
            "day": _to_int(speech.get("day")) or None,
            "role": str(claim.get("role", "")).strip().lower() or None,
            "checked_seat": _to_int(claim.get("checked_seat")),
            "result": _normalize_result(claim.get("result")),
            "phase": phase,
        }
        normalized["evidence_id"] = _evidence_id(
            "claim",
            normalized.get("day"),
            normalized.get("claimer"),
            normalized.get("role"),
            normalized.get("checked_seat"),
            normalized.get("result"),
        )
        claims.append({k: v for k, v in normalized.items() if v is not None})
    return claims


def _collect_attitude_edges(speeches: list[dict[str, Any]], phase: str | None) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str, str, int | None]] = set()
    for speech in speeches:
        source = _to_int(speech.get("seat"))
        if not source:
            continue
        day = _to_int(speech.get("day"))
        for target in speech.get("accuses") or []:
            target_seat = _to_int(target)
            if target_seat and target_seat != source:
                _append_edge(edges, seen, source, target_seat, "oppose", "accuse", day, phase)
        attitudes = speech.get("attitudes")
        if isinstance(attitudes, dict):
            for raw_target, raw_stance in attitudes.items():
                target = _to_int(raw_target)
                stance = _normalize_stance(raw_stance)
                if target and target != source:
                    _append_edge(edges, seen, source, target, stance, "attitude", day, phase)
    return edges


def _append_edge(
    edges: list[dict[str, Any]],
    seen: set[tuple[int, int, str, str, int | None]],
    source: int,
    target: int,
    stance: str,
    source_type: str,
    day: int | None,
    phase: str | None,
) -> None:
    key = (source, target, stance, source_type, day)
    if key in seen:
        return
    seen.add(key)
    edge = {"source": source, "target": target, "stance": stance, "source_type": source_type}
    if day is not None:
        edge["day"] = day
    if phase:
        edge["phase"] = phase
    edge["evidence_id"] = _evidence_id(source_type, day, source, target, stance)
    edges.append(edge)


def _collect_vote_edges(public_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for event in public_events:
        if event.get("type") != "vote_cast":
            continue
        payload = event.get("payload") or {}
        voter = _to_int(payload.get("voter_seat") or payload.get("voter"))
        target = _to_int(payload.get("target_seat") or payload.get("target"))
        if not voter or not target or voter == target:
            continue
        edges.append({
            "source": voter,
            "target": target,
            "stance": "oppose",
            "source_type": "vote",
            "day": event.get("day"),
            "phase": _phase_value(event.get("phase")),
            "evidence_id": _evidence_id("vote", event.get("day"), voter, target),
        })
    return edges


def _collect_death_events(
    public_events: list[dict[str, Any]], id_to_seat: dict[str, int]
) -> list[dict[str, Any]]:
    deaths: list[dict[str, Any]] = []
    for event in public_events:
        etype = event.get("type")
        payload = event.get("payload") or {}
        seats: list[int] = []
        if etype == "night_deaths":
            for pid in payload.get("dead_player_ids") or []:
                seat = id_to_seat.get(str(pid))
                if seat:
                    seats.append(seat)
        elif etype == "player_exiled":
            seat = id_to_seat.get(str(payload.get("exiled_player_id")))
            if seat:
                seats.append(seat)
        elif etype == "hunter_shot":
            seat = id_to_seat.get(str(payload.get("target_id")))
            if seat:
                seats.append(seat)
        if seats:
            unique_seats = sorted(set(seats))
            deaths.append({
                "type": etype,
                "day": event.get("day"),
                "phase": _phase_value(event.get("phase")),
                "seats": unique_seats,
                "evidence_id": _evidence_id("death", event.get("day"), etype, unique_seats),
            })
    return deaths


def _collect_private_seer_results(private_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for event in private_events:
        if event.get("type") != "seer_result":
            continue
        if _visibility_value(event.get("visibility")) != "private":
            continue
        payload = event.get("payload") or {}
        target = _to_int(payload.get("target_seat"))
        team = _normalize_result(payload.get("team"))
        if not target or team not in {"wolf", "village"}:
            continue
        results.append({
            "target": target,
            "result": team,
            "day": event.get("day"),
            "phase": _phase_value(event.get("phase")),
            "source": "private_seer_result",
            "evidence_id": _evidence_id("private_seer_result", event.get("day"), target, team),
        })
    return results


def _claim_conflicts(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    seer_claims = [c for c in claims if c.get("role") == "seer"]
    seer_claimers = sorted({int(c["claimer"]) for c in seer_claims if c.get("claimer")})
    if len(seer_claimers) >= 2:
        conflicts.append({
            "evidence_id": _evidence_id("claim_conflict", "seer_counterclaim", seer_claimers),
            "type": "seer_counterclaim",
            "claimers": seer_claimers,
            "description": "多人声称预言家,至多一个为真",
        })

    by_target: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    by_claimer_target: dict[tuple[int, int], set[str]] = defaultdict(set)
    for claim in seer_claims:
        claimer = _to_int(claim.get("claimer"))
        target = _to_int(claim.get("checked_seat"))
        result = claim.get("result")
        if not claimer or not target or result not in {"wolf", "village"}:
            continue
        by_target[target][str(result)].append(claimer)
        by_claimer_target[(claimer, target)].add(str(result))

    for target, result_map in sorted(by_target.items()):
        if len(result_map) >= 2:
            results = {k: sorted(set(v)) for k, v in result_map.items()}
            conflicts.append({
                "evidence_id": _evidence_id("claim_conflict", "seer_result_conflict", target, sorted(results)),
                "type": "seer_result_conflict",
                "target": target,
                "results": results,
                "description": f"对{target}号的查验声明互相冲突",
            })
    for (claimer, target), results in sorted(by_claimer_target.items()):
        if len(results) >= 2:
            conflicts.append({
                "evidence_id": _evidence_id("claim_conflict", "self_contradictory_claim", claimer, target, sorted(results)),
                "type": "self_contradictory_claim",
                "claimer": claimer,
                "target": target,
                "results": sorted(results),
                "description": f"{claimer}号对{target}号先后报出不同查验结果",
            })
    return conflicts


def _build_evidence_items(
    *,
    phase: str | None,
    claims: list[dict[str, Any]],
    claim_conflicts: list[dict[str, Any]],
    attitude_edges: list[dict[str, Any]],
    vote_edges: list[dict[str, Any]],
    death_events: list[dict[str, Any]],
    private_results: list[dict[str, Any]],
    posterior_deltas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    for claim in claims:
        _append_evidence_item(
            items,
            seen,
            evidence_id=claim.get("evidence_id"),
            kind="claim",
            visibility="public",
            provenance="today_speech",
            day=claim.get("day"),
            phase=claim.get("phase") or phase,
            source_seat=claim.get("claimer"),
            target_seat=claim.get("checked_seat"),
            confidence=0.65,
            payload={
                "role": claim.get("role"),
                "result": claim.get("result"),
            },
        )

    for conflict in claim_conflicts:
        _append_evidence_item(
            items,
            seen,
            evidence_id=conflict.get("evidence_id"),
            kind=conflict.get("type") or "claim_conflict",
            visibility="public",
            provenance="derived_claim_conflict",
            day=conflict.get("day"),
            phase=phase,
            source_seat=conflict.get("claimer"),
            target_seat=conflict.get("target"),
            confidence=0.8,
            payload={
                "claimers": conflict.get("claimers"),
                "results": conflict.get("results"),
                "description": conflict.get("description"),
            },
        )

    for edge in attitude_edges:
        _append_evidence_item(
            items,
            seen,
            evidence_id=edge.get("evidence_id"),
            kind=edge.get("source_type") or "attitude",
            visibility="public",
            provenance="today_speech",
            day=edge.get("day"),
            phase=edge.get("phase") or phase,
            source_seat=edge.get("source"),
            target_seat=edge.get("target"),
            confidence=0.55,
            payload={"stance": edge.get("stance")},
        )

    for edge in vote_edges:
        _append_evidence_item(
            items,
            seen,
            evidence_id=edge.get("evidence_id"),
            kind="vote",
            visibility="public",
            provenance="public_event",
            day=edge.get("day"),
            phase=edge.get("phase"),
            source_seat=edge.get("source"),
            target_seat=edge.get("target"),
            confidence=0.7,
            payload={"stance": edge.get("stance")},
        )

    for death in death_events:
        for seat in death.get("seats") or []:
            _append_evidence_item(
                items,
                seen,
                evidence_id=_evidence_id(death.get("evidence_id"), seat),
                kind=death.get("type") or "death",
                visibility="public",
                provenance="public_event",
                day=death.get("day"),
                phase=death.get("phase"),
                source_seat=None,
                target_seat=seat,
                confidence=0.75,
                payload={"death_event_id": death.get("evidence_id")},
            )

    for result in private_results:
        _append_evidence_item(
            items,
            seen,
            evidence_id=result.get("evidence_id"),
            kind="private_seer_result",
            visibility="private",
            provenance="private_event",
            day=result.get("day"),
            phase=result.get("phase"),
            source_seat=None,
            target_seat=result.get("target"),
            confidence=1.0,
            payload={"result": result.get("result")},
        )

    for delta in posterior_deltas:
        evidence_id = str(delta.get("evidence_id") or "")
        if not evidence_id or evidence_id in seen:
            continue
        _append_evidence_item(
            items,
            seen,
            evidence_id=evidence_id,
            kind=delta.get("source_type") or "derived",
            visibility="public",
            provenance="derived_posterior_delta",
            day=None,
            phase=phase,
            source_seat=None,
            target_seat=delta.get("target_seat"),
            confidence=0.5,
            payload={"reason": delta.get("reason")},
        )

    return items


def _append_evidence_item(
    items: list[dict[str, Any]],
    seen: set[str],
    *,
    evidence_id: Any,
    kind: Any,
    visibility: str,
    provenance: str,
    day: Any,
    phase: Any,
    source_seat: Any,
    target_seat: Any,
    confidence: float,
    payload: dict[str, Any] | None = None,
) -> None:
    eid = str(evidence_id or "")
    if not eid or eid in seen:
        return
    seen.add(eid)
    item = {
        "evidence_id": eid,
        "type": str(kind or "evidence"),
        "visibility": visibility,
        "provenance": provenance,
        "confidence": _clamp01(confidence),
    }
    normalized_day = _to_int(day)
    if normalized_day is not None:
        item["day"] = normalized_day
    normalized_phase = _phase_value(phase)
    if normalized_phase:
        item["phase"] = normalized_phase
    source = _to_int(source_seat)
    if source is not None:
        item["source_seat"] = source
    target = _to_int(target_seat)
    if target is not None:
        item["target_seat"] = target
    clean_payload = {
        str(k): v
        for k, v in (payload or {}).items()
        if v is not None and v != [] and v != {}
    }
    if clean_payload:
        item["payload"] = clean_payload
    items.append(item)


def _role_counts(obs: Any, seats: list[int]) -> dict[str, int]:
    counts = getattr(obs, "role_counts", None)
    if isinstance(counts, dict) and counts:
        normalized: dict[str, int] = {}
        for role, count in counts.items():
            try:
                n = int(count)
            except (TypeError, ValueError):
                continue
            if n > 0:
                normalized[str(role).strip().lower()] = n
        if normalized:
            return dict(sorted(normalized.items()))
    # Six-player classic fallback. Larger games should normally pass role_counts.
    wolf_count = 2 if len(seats) <= 8 else 3 if len(seats) <= 11 else 4
    villager_count = max(0, len(seats) - wolf_count - 1)
    return dict(sorted({"seer": 1, "villager": villager_count, "werewolf": wolf_count}.items()))


def _legal_team_worlds(
    obs: Any,
    seats: list[int],
    role_counts: dict[str, int],
    private_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Enumerate legal wolf-seat worlds from public deck counts and private facts.

    This is not a full role assignment enumerator. It only constrains the
    werewolf team seats, which is the probability space used by current
    posterior/calibration metrics.
    """
    wolf_count = int(role_counts.get("werewolf", 0) or 0)
    known_wolves: set[int] = set()
    known_villagers: set[int] = set()

    my_seat = _to_int(getattr(obs, "my_seat", None))
    my_role = str(getattr(obs, "my_role", "") or "").lower()
    if my_seat:
        if my_role == "werewolf":
            known_wolves.add(my_seat)
        else:
            known_villagers.add(my_seat)
    if my_role == "werewolf":
        for teammate in getattr(obs, "my_teammates", []) or []:
            seat = _to_int((teammate or {}).get("seat"))
            if seat:
                known_wolves.add(seat)

    for result in private_results:
        target = _to_int(result.get("target"))
        if not target:
            continue
        if result.get("result") == "wolf":
            known_wolves.add(target)
            known_villagers.discard(target)
        elif result.get("result") == "village":
            known_villagers.add(target)
            known_wolves.discard(target)

    contradiction = bool(known_wolves & known_villagers)
    unknown = [
        seat
        for seat in seats
        if seat not in known_wolves and seat not in known_villagers
    ]
    remaining_wolves = wolf_count - len(known_wolves)
    worlds: list[tuple[int, ...]] = []
    if not contradiction and 0 <= remaining_wolves <= len(unknown):
        for extra in combinations(unknown, remaining_wolves):
            worlds.append(tuple(sorted([*known_wolves, *extra])))

    return {
        "wolf_count": wolf_count,
        "known_wolves": sorted(known_wolves),
        "known_villagers": sorted(known_villagers),
        "world_count": len(worlds),
        "is_contradictory": contradiction or remaining_wolves < 0 or remaining_wolves > len(unknown),
        "is_truncated": len(worlds) > 60,
        "worlds": [{"wolf_seats": list(world)} for world in worlds[:60]],
        "_all_worlds": worlds,
    }


def _constrained_wolf_marginals(
    worlds: list[tuple[int, ...]],
    seats: list[int],
    heuristic: dict[int, float],
) -> dict[int, float]:
    if not worlds:
        return {}

    logs: list[float] = []
    world_sets = [set(world) for world in worlds]
    for wolf_set in world_sets:
        total = 0.0
        for seat in seats:
            p = min(0.99, max(0.01, float(heuristic.get(seat, 0.5))))
            total += math.log(p if seat in wolf_set else 1.0 - p)
        logs.append(total)

    max_log = max(logs)
    weights = [math.exp(value - max_log) for value in logs]
    denom = sum(weights)
    if denom <= 0:
        uniform = 1.0 / len(worlds)
        weights = [uniform for _ in worlds]
        denom = 1.0

    marginals: dict[int, float] = {}
    for seat in seats:
        value = sum(weight for weight, wolf_set in zip(weights, world_sets, strict=True) if seat in wolf_set) / denom
        marginals[seat] = _clamp01(value)
    return marginals


def _public_legal_worlds(worlds: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in worlds.items()
        if not key.startswith("_")
    }


def _score_claims(claims: list[dict[str, Any]], posterior: dict[int, dict[str, Any]]) -> None:
    for claim in claims:
        claimer = _to_int(claim.get("claimer"))
        target = _to_int(claim.get("checked_seat"))
        result = claim.get("result")
        if claimer in posterior and claim.get("role") == "seer":
            posterior[claimer]["seer_claim"] = True
            posterior[claimer]["evidence"].append("公开声称预言家")
        if target not in posterior or result not in {"wolf", "village"}:
            continue
        if result == "wolf":
            _add_score(
                posterior,
                target,
                0.25,
                f"{claimer}号预言家声明查杀{target}号",
                evidence_id=claim.get("evidence_id"),
                source_type="claim",
            )
        else:
            _add_score(
                posterior,
                target,
                -0.18,
                f"{claimer}号预言家声明{target}号为好人",
                evidence_id=claim.get("evidence_id"),
                source_type="claim",
            )


def _score_claim_conflicts(conflicts: list[dict[str, Any]], posterior: dict[int, dict[str, Any]]) -> None:
    for conflict in conflicts:
        claimers: set[int] = set()
        if "claimers" in conflict:
            claimers.update(_to_int(s) for s in conflict.get("claimers", []) if _to_int(s))
        if "results" in conflict and isinstance(conflict["results"], dict):
            for seats in conflict["results"].values():
                claimers.update(_to_int(s) for s in seats if _to_int(s))
        if "claimer" in conflict:
            claimer = _to_int(conflict.get("claimer"))
            if claimer:
                claimers.add(claimer)
        for claimer in claimers:
            _add_score(
                posterior,
                claimer,
                0.15,
                conflict.get("description") or "身份声明存在冲突",
                evidence_id=conflict.get("evidence_id"),
                source_type="claim_conflict",
            )


def _score_attitudes(edges: list[dict[str, Any]], posterior: dict[int, dict[str, Any]]) -> None:
    incoming_oppose: dict[int, int] = defaultdict(int)
    for edge in edges:
        target = _to_int(edge.get("target"))
        source = _to_int(edge.get("source"))
        if target not in posterior or not source:
            continue
        if edge.get("stance") == "support":
            _add_score(
                posterior,
                target,
                -0.04,
                f"{source}号公开支持{target}号",
                evidence_id=edge.get("evidence_id"),
                source_type=edge.get("source_type") or "attitude",
            )
        elif edge.get("stance") == "oppose":
            incoming_oppose[target] += 1
            label = "指控" if edge.get("source_type") == "accuse" else "反对"
            _add_score(
                posterior,
                target,
                0.06,
                f"{source}号公开{label}{target}号",
                evidence_id=edge.get("evidence_id"),
                source_type=edge.get("source_type") or "attitude",
            )
    for target, count in incoming_oppose.items():
        if count >= 2:
            _add_score(
                posterior,
                target,
                min(0.12, 0.03 * count),
                f"{target}号被{count}条反对/指控边集中指向",
                evidence_id=_evidence_id("attitude_cluster", target, count),
                source_type="attitude_cluster",
            )


def _score_votes(edges: list[dict[str, Any]], posterior: dict[int, dict[str, Any]]) -> None:
    voters_by_target: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        target = _to_int(edge.get("target"))
        source = _to_int(edge.get("source"))
        if target in posterior and source:
            voters_by_target[target].append(source)
            _add_score(
                posterior,
                target,
                0.08,
                f"{source}号投票给{target}号",
                evidence_id=edge.get("evidence_id"),
                source_type="vote",
            )
    for target, voters in voters_by_target.items():
        if len(voters) >= 2:
            _add_score(
                posterior,
                target,
                min(0.12, 0.03 * len(voters)),
                f"{target}号被多人投票:{','.join(f'{v}号' for v in voters)}",
                evidence_id=_evidence_id("vote_cluster", target, sorted(voters)),
                source_type="vote_cluster",
            )


def _score_private_results(results: list[dict[str, Any]], posterior: dict[int, dict[str, Any]]) -> None:
    for result in results:
        target = _to_int(result.get("target"))
        if target not in posterior:
            continue
        if result.get("result") == "wolf":
            _add_score(
                posterior,
                target,
                0.42,
                f"你的查验结果:{target}号是狼人",
                evidence_id=result.get("evidence_id"),
                source_type="private_seer_result",
            )
        elif result.get("result") == "village":
            _add_score(
                posterior,
                target,
                -0.35,
                f"你的查验结果:{target}号是好人",
                evidence_id=result.get("evidence_id"),
                source_type="private_seer_result",
            )


def _add_score(
    posterior: dict[int, dict[str, Any]],
    seat: int,
    delta: float,
    evidence: str,
    *,
    evidence_id: Any = None,
    source_type: str | None = None,
) -> None:
    if seat not in posterior:
        return
    before = float(posterior[seat]["werewolf_suspicion"])
    after = _clamp01(before + delta)
    posterior[seat]["werewolf_suspicion"] = after
    posterior[seat]["posterior_deltas"].append({
        "target_seat": seat,
        "delta": round(float(delta), 3),
        "before": _clamp01(before),
        "after": after,
        "reason": evidence,
        "evidence_id": str(evidence_id) if evidence_id else _evidence_id("derived", seat, evidence),
        "source_type": source_type or "derived",
    })
    if evidence and evidence not in posterior[seat]["evidence"]:
        posterior[seat]["evidence"].append(evidence)


def _normalize_stance(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in {"support", "支持", "帮腔", "信任", "agree"}:
        return "support"
    if raw in {"oppose", "反对", "指控", "怀疑", "disagree"}:
        return "oppose"
    return "neutral"


def _normalize_result(value: Any) -> str | None:
    raw = str(value).strip().lower()
    if raw in {"wolf", "werewolf", "werewolves", "狼人", "狼", "team.werewolves"}:
        return "wolf"
    if raw in {"village", "villager", "good", "human", "好人", "村民", "team.village"}:
        return "village"
    return None


def _phase_value(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    # StrEnum values arrive as "day"; plain Enum repr may arrive as "Phase.DAY".
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1]
    return raw.lower()


def _visibility_value(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1]
    return raw.lower()


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(float(str(value).replace("号", "").strip()))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _clamp01(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _evidence_id(prefix: str, *parts: Any) -> str:
    """Stable, human-readable id for visible evidence provenance."""
    tokens = [_safe_token(prefix)]
    for part in parts:
        if isinstance(part, (list, tuple, set)):
            token = ",".join(_safe_token(v) for v in sorted(part, key=lambda x: str(x)))
        else:
            token = _safe_token(part)
        tokens.append(token)
    return ":".join(tokens)


def _safe_token(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text:
        return "-"
    return (
        text.replace(" ", "_")
        .replace("\n", "_")
        .replace("\t", "_")
        .replace(":", "_")
        .replace("|", "_")
    )[:80]
