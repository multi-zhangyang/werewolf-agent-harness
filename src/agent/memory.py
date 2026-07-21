"""Per-agent memory of observations actually delivered by the environment.

Memory is not an evaluator and does not manufacture beliefs.  It stores a
chronological log of visible events plus public structured claims so later
requests can include the agent's own history.  Hidden role truth only appears
when the environment explicitly delivered a private event to that agent.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any


DEFAULT_MAX_STORED_OBSERVATIONS = 256
DEFAULT_MAX_STORED_CLAIMS_PER_SEAT = 24
# Public votes are kept in a separate bounded ledger because ordinary
# observation compaction must not erase the table's accepted vote sequence.
DEFAULT_MAX_STORED_PUBLIC_VOTES = 1024
DEFAULT_RENDERED_CLAIMS_PER_SEAT = 6
_MAX_TEXT_CHARS = 2400
_MAX_VALUE_TEXT_CHARS = 1200
_MAX_COLLECTION_ITEMS = 32
_MAX_STRUCTURE_DEPTH = 5
_MAX_SUMMARY_KINDS = 32

# These observations carry durable private capabilities or results. Under
# ordinary game limits they survive preferentially when routine table talk is
# compacted, but the overall store still has a hard cap.
_DURABLE_OBSERVATION_KINDS = frozenset({
    "role_assigned",
    "teammate",
    "seer_result",
    "witch_save_used",
    "witch_poison_used",
    "guard_target",
    "doctor_protect_target",
    "last_words",
})


@dataclass(frozen=True)
class MemoryItem:
    """One environment-delivered observation."""

    day: int
    phase: str
    kind: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return f"[D{self.day} {self.phase}]({self.kind}) {self.text}"


@dataclass(frozen=True)
class PublicVoteRecord:
    """One rules-accepted public vote in a seat-owned ledger."""

    day: int
    phase: str
    voter_seat: int
    target_seat: int
    pk: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "phase": self.phase,
            "voter_seat": self.voter_seat,
            "target_seat": self.target_seat,
            "pk": self.pk,
        }


class AgentMemory:
    """Chronological visible history owned by one agent adapter."""

    def __init__(
        self,
        seat: int,
        role: str,
        *,
        max_observations: int = DEFAULT_MAX_STORED_OBSERVATIONS,
        max_claims_per_seat: int = DEFAULT_MAX_STORED_CLAIMS_PER_SEAT,
        max_public_votes: int = DEFAULT_MAX_STORED_PUBLIC_VOTES,
    ) -> None:
        if type(max_observations) is not int or max_observations <= 0:
            raise ValueError("max_observations must be a positive integer")
        if type(max_claims_per_seat) is not int or max_claims_per_seat < 2:
            raise ValueError("max_claims_per_seat must be an integer of at least 2")
        if type(max_public_votes) is not int or max_public_votes <= 0:
            raise ValueError("max_public_votes must be a positive integer")
        self.seat = seat
        self.role = role
        self._max_observations = max_observations
        self._max_claims_per_seat = max_claims_per_seat
        self._max_public_votes = max_public_votes
        self._observations: list[MemoryItem] = []
        self._public_votes: list[PublicVoteRecord] = []
        self._public_vote_count = 0
        self._archived_public_vote_count = 0
        self._archived_public_vote_digest = hashlib.sha256(b"").hexdigest()
        self._claims: dict[int, list[dict[str, Any]]] = {}
        self._observation_count = 0
        self._archived_observation_count = 0
        self._archived_observation_min_day: int | None = None
        self._archived_observation_max_day: int | None = None
        self._archived_observation_kinds: Counter[str] = Counter()
        self._archived_observation_digest = hashlib.sha256(b"").hexdigest()
        self._claim_count = 0
        self._claim_stats: dict[int, dict[str, Any]] = {}
        self._archived_claim_digest = hashlib.sha256(b"").hexdigest()

    @property
    def observations(self) -> list[MemoryItem]:
        """Return a detached view of observations owned by this memory."""
        return deepcopy(self._observations)

    @property
    def claims(self) -> dict[int, list[dict[str, Any]]]:
        """Return a detached view of public claims owned by this memory."""
        return deepcopy(self._claims)

    @property
    def public_votes(self) -> list[dict[str, Any]]:
        """Return a detached view of the bounded public vote ledger."""
        return [record.as_dict() for record in self._public_votes]

    @property
    def public_vote_ledger(self) -> list[dict[str, Any]]:
        """Alias for callers that want to name the backing structure explicitly."""
        return self.public_votes

    def observe(self, day: int, phase: str, kind: str, text: str, **meta: Any) -> None:
        # ``importance`` belonged to the removed heuristic retrieval scorer.
        # Ignore it when reading an older caller instead of preserving a fake
        # score in current artifacts.
        meta.pop("importance", None)
        item = MemoryItem(
            int(day),
            str(phase)[:80],
            str(kind)[:80],
            str(text)[:_MAX_TEXT_CHARS],
            _bounded_structure(meta),
        )
        if item.kind == "vote":
            vote = _public_vote_from_memory_item(item)
            if vote is not None:
                self._record_public_vote(vote)
        self._observation_count += 1
        self._observations.append(item)
        while len(self._observations) > self._max_observations:
            eviction_index = next(
                (
                    index
                    for index, stored in enumerate(self._observations)
                    if stored.kind not in _DURABLE_OBSERVATION_KINDS
                ),
                0,
            )
            self._archive_observation(self._observations.pop(eviction_index))

    def read_public_votes(
        self,
        *,
        limit: int = 40,
        voter_seat: int | None = None,
        target_seat: int | None = None,
        pk: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Read recent structured public votes with factual filters only.

        The ledger contains observations already delivered by the environment;
        this method performs no inference or relation/stance calculation.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return []
        if voter_seat is not None:
            voter_seat = _positive_int(voter_seat)
            if voter_seat is None:
                return []
        if target_seat is not None:
            target_seat = _positive_int(target_seat)
            if target_seat is None:
                return []
        if pk is not None and not isinstance(pk, bool):
            return []
        selected = [
            record
            for record in self._public_votes
            if (voter_seat is None or record.voter_seat == voter_seat)
            and (target_seat is None or record.target_seat == target_seat)
            and (pk is None or record.pk is pk)
        ]
        return [record.as_dict() for record in selected[-limit:]]

    def record_claim(self, seat: int, day: int, claim: dict[str, Any]) -> None:
        """Record a bounded public claim as a claim, never as role truth.

        The first exact claim for a seat and its recent exact claims are kept.
        Repeated middle entries are evicted first; deterministic counters retain
        role/result frequencies and contradiction transitions without asking a
        model to summarize them.
        """
        if not isinstance(claim, dict):
            return
        claim_seat = int(seat)
        bounded_claim = _bounded_structure(claim)
        entry = {**bounded_claim, "day": int(day)}
        claim_fingerprint = _claim_fingerprint(bounded_claim)
        stats = self._claim_stats.setdefault(claim_seat, {
            "total_count": 0,
            "omitted_count": 0,
            "first_day": int(day),
            "last_day": int(day),
            "transition_count": 0,
            "last_fingerprint": None,
            "role_counts": Counter(),
            "seer_result_counts": Counter(),
        })
        previous_fingerprint = stats["last_fingerprint"]
        if previous_fingerprint is not None and previous_fingerprint != claim_fingerprint:
            stats["transition_count"] += 1
        stats["last_fingerprint"] = claim_fingerprint
        stats["total_count"] += 1
        stats["last_day"] = int(day)
        role = str(bounded_claim.get("role") or "unspecified")[:80]
        _increment_bounded_counter(stats["role_counts"], role)
        checked_seat = bounded_claim.get("checked_seat")
        result = bounded_claim.get("result")
        if checked_seat is not None or result is not None:
            result_key = f"{checked_seat}:{str(result)[:80]}"
            _increment_bounded_counter(stats["seer_result_counts"], result_key)

        self._claim_count += 1
        stored_claims = self._claims.setdefault(claim_seat, [])
        stored_claims.append(entry)
        while len(stored_claims) > self._max_claims_per_seat:
            fingerprints = Counter(
                _claim_fingerprint({key: value for key, value in item.items() if key != "day"})
                for item in stored_claims
            )
            # Preserve the first commitment and current commitment. Among the
            # middle entries, discard a duplicate before a unique contradiction.
            eviction_index = next(
                (
                    index
                    for index in range(1, len(stored_claims) - 1)
                    if fingerprints[
                        _claim_fingerprint({
                            key: value
                            for key, value in stored_claims[index].items()
                            if key != "day"
                        })
                    ] > 1
                ),
                1,
            )
            evicted = stored_claims.pop(eviction_index)
            stats["omitted_count"] += 1
            self._archived_claim_digest = _extend_digest(
                self._archived_claim_digest,
                {"seat": claim_seat, "claim": evicted},
            )

    def recent_observations(self, limit: int = 30) -> list[MemoryItem]:
        """Return the most recently observed events in chronological order."""
        if limit <= 0:
            return []
        return deepcopy(self._observations[-limit:])

    def render_for_prompt(self, *, obs_limit: int = 25) -> str:
        """Render a bounded window plus deterministic, non-model summaries."""
        parts: list[str] = []
        observations = self.recent_observations(obs_limit)
        if self._observation_count or self._claim_count:
            displayed_observation_count = len(observations)
            omitted_observation_count = self._observation_count - displayed_observation_count
            parts.extend([
                "【机械记忆计数（程序统计，不是模型总结）】",
                (
                    f"- 可见事件累计={self._observation_count}，"
                    f"当前保留={len(self._observations)}，"
                    f"本次展示={displayed_observation_count}，"
                    f"本次省略={omitted_observation_count}"
                ),
                (
                    f"- 公开结构化声明累计={self._claim_count}，"
                    f"当前保留={sum(len(items) for items in self._claims.values())}"
                ),
            ])
            if omitted_observation_count:
                omitted_kinds = Counter(self._archived_observation_kinds)
                retained_prefix_length = max(0, len(self._observations) - displayed_observation_count)
                for item in self._observations[:retained_prefix_length]:
                    omitted_kinds[item.kind] += 1
                parts.append(
                    "- 省略事件类型计数=" + _render_counter(omitted_kinds)
                )
        if observations:
            parts.append("【你此前实际看到的事件】")
            parts.extend(f"- {item.render()}" for item in observations)
        if self._claims:
            parts.append("\n【公开结构化声明（仅记录，不代表真实）】")
            for seat, claim_list in sorted(self._claims.items()):
                stats = self._claim_stats[seat]
                selected_claims = _select_claims_for_prompt(
                    claim_list,
                    limit=DEFAULT_RENDERED_CLAIMS_PER_SEAT,
                )
                parts.append(
                    f"- {seat}号机械计数：累计={stats['total_count']}，"
                    f"存储省略={stats['omitted_count']}，"
                    f"声明变化={stats['transition_count']}，"
                    f"身份次数={_render_counter(stats['role_counts'])}，"
                    f"查验次数={_render_counter(stats['seer_result_counts'])}"
                )
                if len(claim_list) > len(selected_claims):
                    parts.append(
                        f"  - 精确声明窗口省略={len(claim_list) - len(selected_claims)}"
                    )
                parts.extend(
                    "  - D{day} {claim}".format(
                        day=claim.get("day"),
                        claim=json.dumps(
                            {key: value for key, value in claim.items() if key != "day"},
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        ),
                    )
                    for claim in selected_claims
                )
        return "\n".join(parts) if parts else "(尚无记忆)"

    def snapshot(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "role": self.role,
            "observation_count": self._observation_count,
            "retained_observation_count": len(self._observations),
            "observation_summary": {
                "archived_count": self._archived_observation_count,
                "archived_day_min": self._archived_observation_min_day,
                "archived_day_max": self._archived_observation_max_day,
                "archived_kind_counts": dict(sorted(self._archived_observation_kinds.items())),
                "archived_digest": self._archived_observation_digest,
            },
            "claim_count": self._claim_count,
            "retained_claim_count": sum(len(items) for items in self._claims.values()),
            "claims": {
                str(seat): deepcopy(items)
                for seat, items in self._claims.items()
            },
            "claim_summary": {
                str(seat): _public_claim_stats(stats, retained=len(self._claims.get(seat, ())))
                for seat, stats in sorted(self._claim_stats.items())
            },
            "public_vote_count": self._public_vote_count,
            "retained_public_vote_count": len(self._public_votes),
            "public_vote_ledger": self.public_votes,
            "public_vote_summary": {
                "total_count": self._public_vote_count,
                "retained_count": len(self._public_votes),
                "archived_count": self._archived_public_vote_count,
                "archived_digest": self._archived_public_vote_digest,
            },
        }

    def digest(self) -> str:
        """Hash retained context plus digests/summaries of compacted history."""
        payload = {
            "seat": self.seat,
            "role": self.role,
            "observation_count": self._observation_count,
            "archived_observation_count": self._archived_observation_count,
            "archived_observation_digest": self._archived_observation_digest,
            "observations": [
                {
                    "day": item.day,
                    "phase": item.phase,
                    "kind": item.kind,
                    "text": item.text,
                    "metadata": item.metadata,
                }
                for item in self._observations
            ],
            "claim_count": self._claim_count,
            "archived_claim_digest": self._archived_claim_digest,
            "claims": self._claims,
            "claim_summary": {
                str(seat): _public_claim_stats(stats, retained=len(self._claims.get(seat, ())))
                for seat, stats in sorted(self._claim_stats.items())
            },
            "public_vote_count": self._public_vote_count,
            "archived_public_vote_count": self._archived_public_vote_count,
            "archived_public_vote_digest": self._archived_public_vote_digest,
            "public_vote_ledger": self.public_votes,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _archive_observation(self, item: MemoryItem) -> None:
        self._archived_observation_count += 1
        self._archived_observation_min_day = (
            item.day
            if self._archived_observation_min_day is None
            else min(self._archived_observation_min_day, item.day)
        )
        self._archived_observation_max_day = (
            item.day
            if self._archived_observation_max_day is None
            else max(self._archived_observation_max_day, item.day)
        )
        _increment_bounded_counter(self._archived_observation_kinds, item.kind)
        self._archived_observation_digest = _extend_digest(
            self._archived_observation_digest,
            {
                "day": item.day,
                "phase": item.phase,
                "kind": item.kind,
                "text": item.text,
                "metadata": item.metadata,
            },
        )

    def _record_public_vote(self, vote: PublicVoteRecord) -> None:
        self._public_vote_count += 1
        self._public_votes.append(vote)
        while len(self._public_votes) > self._max_public_votes:
            evicted = self._public_votes.pop(0)
            self._archived_public_vote_count += 1
            self._archived_public_vote_digest = _extend_digest(
                self._archived_public_vote_digest,
                evicted.as_dict(),
            )


def _bounded_structure(value: Any, *, depth: int = 0) -> Any:
    """Make a detached, size-bounded copy of environment-delivered metadata."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:_MAX_VALUE_TEXT_CHARS]
    if depth >= _MAX_STRUCTURE_DEPTH:
        return str(value)[:_MAX_VALUE_TEXT_CHARS]
    if isinstance(value, dict):
        return {
            deepcopy(key): _bounded_structure(item, depth=depth + 1)
            for key, item in list(value.items())[:_MAX_COLLECTION_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_structure(item, depth=depth + 1)
            for item in value[:_MAX_COLLECTION_ITEMS]
        ]
    if isinstance(value, set):
        return [
            _bounded_structure(item, depth=depth + 1)
            for item in sorted(value, key=repr)[:_MAX_COLLECTION_ITEMS]
        ]
    return str(value)[:_MAX_VALUE_TEXT_CHARS]


def _public_vote_from_memory_item(item: MemoryItem) -> PublicVoteRecord | None:
    """Extract only the structured fields of an accepted public vote."""
    voter_seat = _positive_int(item.metadata.get("voter_seat"))
    target_seat = _positive_int(item.metadata.get("target_seat"))
    pk = item.metadata.get("pk", False)
    if voter_seat is None or target_seat is None or not isinstance(pk, bool):
        return None
    return PublicVoteRecord(
        day=int(item.day),
        phase=str(item.phase),
        voter_seat=voter_seat,
        target_seat=target_seat,
        pk=pk,
    )


def _positive_int(value: Any) -> int | None:
    if type(value) is not int or value <= 0:
        return None
    return value


def _claim_fingerprint(claim: dict[str, Any]) -> str:
    encoded = json.dumps(
        claim,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _extend_digest(previous: str, value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(previous.encode("ascii") + b"\0" + encoded).hexdigest()


def _increment_bounded_counter(counter: Counter[str], key: str) -> None:
    normalized = str(key)[:80]
    if normalized in counter or len(counter) < _MAX_SUMMARY_KINDS:
        counter[normalized] += 1
    else:
        counter["(other)"] += 1


def _render_counter(counter: Counter[str] | dict[str, int]) -> str:
    if not counter:
        return "无"
    return ",".join(
        f"{key}:{count}" for key, count in sorted(counter.items())
    )


def _select_claims_for_prompt(
    claims: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if len(claims) <= limit:
        return claims
    return [claims[0], *claims[-(limit - 1):]]


def _public_claim_stats(stats: dict[str, Any], *, retained: int) -> dict[str, Any]:
    return {
        "total_count": int(stats["total_count"]),
        "retained_count": int(retained),
        "omitted_count": int(stats["omitted_count"]),
        "first_day": int(stats["first_day"]),
        "last_day": int(stats["last_day"]),
        "transition_count": int(stats["transition_count"]),
        "role_counts": dict(sorted(stats["role_counts"].items())),
        "seer_result_counts": dict(sorted(stats["seer_result_counts"].items())),
    }
