"""Private, seat-owned cognition for one Werewolf agent.

The objects in this module are not environment truth and are not evaluators.
They store what one agent currently believes, the strategy it selected, and
the public commitments that the environment actually accepted from that seat.
Nothing here is shared between seats.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..game.roles import Role


_ROLE_VALUES = {role.value for role in Role}
_MAX_COMMITMENTS = 40
_MAX_RENDERED_COMMITMENTS = 12
_MAX_BELIEF_UPDATES = 12
_UNSET = object()


class BeliefUpdate(BaseModel):
    """One model-authored update to this agent's opponent model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seat: int = Field(ge=1)
    wolf_probability: float = Field(ge=0.0, le=1.0)
    likely_role: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(min_length=1, max_length=6)

    @field_validator("wolf_probability", "confidence")
    @classmethod
    def _finite_probability(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("belief probabilities must be finite")
        return float(value)

    @field_validator("likely_role", mode="before")
    @classmethod
    def _known_role_or_unknown(cls, value: Any) -> str | None:
        if value is None:
            return None
        role = str(value).strip().lower()
        if role in {"", "unknown", "uncertain", "none", "null"}:
            return None
        if role not in _ROLE_VALUES:
            raise ValueError("likely_role must be a configured Werewolf role or null")
        return role

    @field_validator("evidence")
    @classmethod
    def _bounded_evidence(cls, value: list[str]) -> list[str]:
        evidence = [str(item).strip()[:280] for item in value if str(item).strip()]
        if not evidence:
            raise ValueError("belief update requires at least one evidence reference")
        return evidence


class PrivateStateUpdate(BaseModel):
    """Structured private state emitted with one model decision.

    Requiring distinct candidate plans makes the model consider alternatives
    before committing to the exact action carried by the same response. These
    fields remain private and never become game facts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    beliefs: list[BeliefUpdate] = Field(max_length=12)
    candidate_plans: list[str] = Field(min_length=2, max_length=4)
    selected_plan: str = Field(min_length=1, max_length=900)
    public_cover_role: str | None
    perceived_image: str = Field(min_length=1, max_length=700)
    deception_plan: str | None = Field(max_length=700)
    team_plan: str | None = Field(max_length=700)

    @field_validator("candidate_plans")
    @classmethod
    def _distinct_candidate_plans(cls, value: list[str]) -> list[str]:
        plans = [str(item).strip()[:700] for item in value if str(item).strip()]
        if len(plans) < 2:
            raise ValueError("at least two non-empty candidate plans are required")
        normalized = {" ".join(item.lower().split()) for item in plans}
        if len(normalized) != len(plans):
            raise ValueError("candidate plans must be distinct")
        return plans

    @field_validator("selected_plan", "perceived_image")
    @classmethod
    def _required_private_text(cls, value: str) -> str:
        return str(value).strip()

    @field_validator("deception_plan", "team_plan", mode="before")
    @classmethod
    def _optional_private_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("public_cover_role", mode="before")
    @classmethod
    def _valid_cover_role(cls, value: Any) -> str | None:
        if value is None:
            return None
        role = str(value).strip().lower()
        if role in {"", "unclaimed", "none", "null"}:
            return None
        if role not in _ROLE_VALUES:
            raise ValueError("public_cover_role must be a configured Werewolf role or null")
        return role


class SeatBelief(BaseModel):
    """Latest private belief held by one owner about one other seat."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seat: int
    wolf_probability: float
    likely_role: str | None
    confidence: float
    evidence: tuple[str, ...]
    updated_day: int
    updated_phase: str


class PublicCommitment(BaseModel):
    """Exact public output accepted from this seat by the environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    day: int
    phase: str
    kind: str
    text: str
    claim: dict[str, Any] | None = None


class PrivateAgentState:
    """Mutable runtime state owned by exactly one AgentActor instance."""

    def __init__(self, *, owner_seat: int, owner_role: str) -> None:
        self.owner_seat = int(owner_seat)
        self.owner_role = str(owner_role)
        self._beliefs: dict[int, SeatBelief] = {}
        self._candidate_plans: tuple[str, ...] = ()
        self._selected_plan = "尚未形成持续策略"
        self._public_cover_role: str | None = None
        self._perceived_image = "尚未判断其他玩家如何看待自己"
        self._deception_plan: str | None = None
        self._team_plan: str | None = None
        self._commitments: list[PublicCommitment] = []
        self._revision = 0

    @staticmethod
    def validate_model_update(raw: Any) -> PrivateStateUpdate:
        """Validate a model-authored update before it can affect private state."""
        return PrivateStateUpdate.model_validate(raw)

    def apply_model_update(
        self,
        raw: Any,
        *,
        visible_seats: set[int],
        day: int,
        phase: str,
        known_wolf_seats: set[int] | None = None,
        known_village_seats: set[int] | None = None,
        total_wolves: int | None = None,
    ) -> None:
        """Commit one update and optionally enforce visible role-count facts."""
        update = self.validate_model_update(deepcopy(raw))
        allowed = {int(seat) for seat in visible_seats if int(seat) != self.owner_seat}
        incoming = {item.seat: item for item in update.beliefs if item.seat in allowed}
        probabilities = {
            seat: (
                incoming[seat].wolf_probability
                if seat in incoming
                else self._beliefs.get(seat, _uninformed_belief(seat)).wolf_probability
            )
            for seat in allowed
        }
        known_wolves = set(known_wolf_seats or ()) & allowed
        known_village = set(known_village_seats or ()) & allowed
        if known_wolves & known_village:
            raise ValueError("one seat cannot be known as both wolf and village")
        if total_wolves is not None:
            owner_is_wolf = self.owner_role == Role.WEREWOLF.value
            unknown = sorted(allowed - known_wolves - known_village)
            remaining = int(total_wolves) - int(owner_is_wolf) - len(known_wolves)
            projected = _project_probability_mass(
                [probabilities[seat] for seat in unknown],
                target=float(max(0, min(len(unknown), remaining))),
            )
            probabilities.update(dict(zip(unknown, projected, strict=True)))
            probabilities.update({seat: 1.0 for seat in known_wolves})
            probabilities.update({seat: 0.0 for seat in known_village})

        for seat in sorted(allowed):
            item = incoming.get(seat)
            previous = self._beliefs.get(seat)
            probability = probabilities[seat]
            if seat in known_wolves:
                likely_role = Role.WEREWOLF.value
                confidence = 1.0
                evidence = ("由该座位可见的私有角色信息确认",)
            elif seat in known_village:
                likely_role = (
                    item.likely_role
                    if item is not None and item.likely_role != Role.WEREWOLF.value
                    else None
                )
                confidence = 1.0
                evidence = ("由该座位可见的私有查验信息确认阵营",)
            elif item is not None:
                likely_role = item.likely_role
                confidence = item.confidence
                evidence = tuple(item.evidence)
            elif previous is not None:
                likely_role = previous.likely_role
                confidence = previous.confidence
                evidence = previous.evidence
            else:
                likely_role = None
                confidence = 0.1
                evidence = ("仅依据公开角色数量形成的未信息化先验",)
            if probability == 0.0 and likely_role == Role.WEREWOLF.value:
                likely_role = None
            self._beliefs[seat] = SeatBelief(
                seat=seat,
                wolf_probability=probability,
                likely_role=likely_role,
                confidence=confidence,
                evidence=evidence,
                updated_day=int(day),
                updated_phase=str(phase),
            )
        self._candidate_plans = tuple(update.candidate_plans)
        self._selected_plan = update.selected_plan
        self._public_cover_role = update.public_cover_role
        self._perceived_image = update.perceived_image
        self._deception_plan = update.deception_plan
        self._team_plan = update.team_plan
        self._revision += 1

    def update_belief(
        self,
        raw: Any,
        *,
        visible_seats: set[int],
        day: int,
        phase: str,
        known_wolf_seats: set[int] | None = None,
        known_village_seats: set[int] | None = None,
        total_wolves: int | None = None,
    ) -> None:
        """Apply one tool-authored belief patch without rewriting private state.

        The environment still owns hard role facts and the configured wolf
        count.  A model can revise one subjective belief, but it cannot use this
        method to contradict a role fact visible to this seat.
        """
        self.update_beliefs(
            [deepcopy(raw)],
            visible_seats=visible_seats,
            day=day,
            phase=phase,
            known_wolf_seats=known_wolf_seats,
            known_village_seats=known_village_seats,
            total_wolves=total_wolves,
        )

    def update_beliefs(
        self,
        raw: Any,
        *,
        visible_seats: set[int],
        day: int,
        phase: str,
        known_wolf_seats: set[int] | None = None,
        known_village_seats: set[int] | None = None,
        total_wolves: int | None = None,
    ) -> tuple[int, ...]:
        """Atomically validate and commit several subjective belief patches."""
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("belief updates must be a non-empty list")
        if len(raw) > _MAX_BELIEF_UPDATES:
            raise ValueError(f"at most {_MAX_BELIEF_UPDATES} belief updates are allowed")
        normalized: list[Any] = []
        for item in raw:
            candidate = deepcopy(item)
            if isinstance(candidate, Mapping):
                candidate = dict(candidate)
                candidate.setdefault("likely_role", None)
            normalized.append(candidate)
        updates = tuple(BeliefUpdate.model_validate(item) for item in normalized)
        seats = tuple(item.seat for item in updates)
        if len(set(seats)) != len(seats):
            raise ValueError("belief updates must contain at most one patch per seat")

        allowed = {int(seat) for seat in visible_seats if int(seat) != self.owner_seat}
        hidden = sorted(set(seats) - allowed)
        if hidden:
            raise ValueError(
                "belief seat must be another seat visible to this agent: "
                + ",".join(str(seat) for seat in hidden)
            )
        incoming = {item.seat: item for item in updates}
        known_wolves = set(known_wolf_seats or ()) & allowed
        known_village = set(known_village_seats or ()) & allowed
        if known_wolves & known_village:
            raise ValueError("one seat cannot be known as both wolf and village")
        for seat, item in incoming.items():
            if seat in known_wolves and (
                item.wolf_probability != 1.0
                or item.likely_role not in {None, Role.WEREWOLF.value}
            ):
                raise ValueError("belief update contradicts a known wolf seat")
            if seat in known_village and (
                item.wolf_probability != 0.0
                or item.likely_role == Role.WEREWOLF.value
            ):
                raise ValueError("belief update contradicts a known village seat")

        probabilities = {
            seat: self._beliefs.get(seat, _uninformed_belief(seat)).wolf_probability
            for seat in allowed
        }
        probabilities.update({seat: item.wolf_probability for seat, item in incoming.items()})
        if total_wolves is not None:
            owner_is_wolf = self.owner_role == Role.WEREWOLF.value
            unknown = sorted(allowed - known_wolves - known_village)
            remaining = int(total_wolves) - int(owner_is_wolf) - len(known_wolves)
            projected = _project_probability_mass(
                [probabilities[seat] for seat in unknown],
                target=float(max(0, min(len(unknown), remaining))),
            )
            probabilities.update(dict(zip(unknown, projected, strict=True)))
        probabilities.update({seat: 1.0 for seat in known_wolves})
        probabilities.update({seat: 0.0 for seat in known_village})

        candidate_beliefs = dict(self._beliefs)
        for seat in sorted(allowed):
            item = incoming.get(seat)
            previous = self._beliefs.get(seat)
            if seat in known_wolves:
                likely_role = Role.WEREWOLF.value
                confidence = 1.0
                evidence = ("由该座位可见的私有角色信息确认",)
            elif seat in known_village:
                likely_role = (
                    item.likely_role
                    if item is not None and item.likely_role != Role.WEREWOLF.value
                    else (
                        previous.likely_role
                        if previous is not None and previous.likely_role != Role.WEREWOLF.value
                        else None
                    )
                )
                confidence = 1.0
                evidence = ("由该座位可见的私有查验信息确认阵营",)
            elif item is not None:
                likely_role = item.likely_role
                confidence = item.confidence
                evidence = tuple(item.evidence)
            elif previous is not None:
                likely_role = previous.likely_role
                confidence = previous.confidence
                evidence = previous.evidence
            else:
                likely_role = None
                confidence = 0.1
                evidence = ("仅依据公开角色数量形成的未信息化先验",)
            if probabilities[seat] == 0.0 and likely_role == Role.WEREWOLF.value:
                likely_role = None
            candidate_beliefs[seat] = SeatBelief(
                seat=seat,
                wolf_probability=probabilities[seat],
                likely_role=likely_role,
                confidence=confidence,
                evidence=evidence,
                updated_day=int(day),
                updated_phase=str(phase),
            )
        self._beliefs = candidate_beliefs
        self._revision += 1
        return tuple(sorted(incoming))

    def set_plan(
        self,
        *,
        selected_plan: str,
        candidate_plans: list[str] | None = None,
        perceived_image: str | None = None,
        deception_plan: str | None | object = _UNSET,
        team_plan: str | None | object = _UNSET,
    ) -> None:
        """Update macro/micro strategy fields from a private state tool call."""
        selected = str(selected_plan).strip()[:900]
        if not selected:
            raise ValueError("selected_plan must be non-empty")
        next_candidates = self._candidate_plans
        next_image = self._perceived_image
        next_deception = self._deception_plan
        next_team = self._team_plan
        if candidate_plans is not None:
            plans = [str(item).strip()[:700] for item in candidate_plans if str(item).strip()]
            normalized = {" ".join(item.lower().split()) for item in plans}
            if len(plans) < 2 or len(normalized) != len(plans) or len(plans) > 4:
                raise ValueError("candidate_plans must contain 2-4 distinct non-empty plans")
            next_candidates = tuple(plans)
        if perceived_image is not None:
            image = str(perceived_image).strip()[:700]
            if not image:
                raise ValueError("perceived_image must be non-empty when provided")
            next_image = image
        if deception_plan is not _UNSET:
            next_deception = _optional_bounded_text(deception_plan, max_length=700)
        if team_plan is not _UNSET:
            next_team = _optional_bounded_text(team_plan, max_length=700)
        self._candidate_plans = next_candidates
        self._perceived_image = next_image
        self._deception_plan = next_deception
        self._team_plan = next_team
        self._selected_plan = selected
        self._revision += 1

    def set_public_cover(self, role: str | None) -> None:
        """Set or clear the role identity this Agent intends to present publicly."""
        if role is None:
            normalized = None
        else:
            normalized = str(role).strip().lower()
            if normalized in {"", "unclaimed", "none", "null"}:
                normalized = None
            elif normalized not in _ROLE_VALUES:
                raise ValueError("public cover must be a configured Werewolf role or null")
        self._public_cover_role = normalized
        self._revision += 1

    def record_public_commitment(
        self,
        *,
        day: int,
        phase: str,
        kind: str,
        text: str,
        claim: dict[str, Any] | None = None,
    ) -> None:
        """Record public output only after the environment accepted it."""
        statement = str(text)
        if not statement.strip() and not claim:
            return
        self._commitments.append(PublicCommitment(
            day=int(day),
            phase=str(phase),
            kind=str(kind),
            text=statement,
            claim=deepcopy(claim) if isinstance(claim, dict) else None,
        ))
        if len(self._commitments) > _MAX_COMMITMENTS:
            del self._commitments[:-_MAX_COMMITMENTS]
        self._revision += 1

    def render_for_prompt(self) -> str:
        """Render bounded subjective state with explicit epistemic labels."""
        lines = [
            "【你的私有主观状态：不是环境真值，不得共享给其他座位】",
            f"当前选择的策略：{self._selected_plan}",
            f"你认为别人如何看你：{self._perceived_image}",
            f"当前公开伪装身份：{self._public_cover_role or '未声明'}",
            f"欺骗/信息隐藏计划：{self._deception_plan or '无'}",
            f"狼队协作计划：{self._team_plan or '无'}",
        ]
        if self._candidate_plans:
            lines.append("本轮考虑过的不同策略：")
            lines.extend(f"- {plan}" for plan in self._candidate_plans)
        if self._beliefs:
            lines.append("你对其他座位的主观判断：")
            for seat, belief in sorted(self._beliefs.items()):
                likely = belief.likely_role or "未知"
                evidence = "；".join(belief.evidence)
                lines.append(
                    f"- {seat}号：狼人概率={belief.wolf_probability:.2f}，"
                    f"可能身份={likely}，信心={belief.confidence:.2f}；依据={evidence}"
                )
        if self._commitments:
            lines.append("你此前已经公开说过的话（维护叙事一致性，可有策略地修正但不可遗忘）：")
            for item in self._commitments[-_MAX_RENDERED_COMMITMENTS:]:
                claim = f"；结构化声明={item.claim}" if item.claim else ""
                lines.append(f"- D{item.day} {item.kind}: {item.text}{claim}")
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        """Return a detached admin/test snapshot; callers cannot mutate state."""
        snapshot = {
            "owner_seat": self.owner_seat,
            "owner_role": self.owner_role,
            "revision": self._revision,
            "beliefs": {
                str(seat): belief.model_dump(mode="json")
                for seat, belief in sorted(self._beliefs.items())
            },
            "candidate_plans": list(self._candidate_plans),
            "selected_plan": self._selected_plan,
            "public_cover_role": self._public_cover_role,
            "perceived_image": self._perceived_image,
            "deception_plan": self._deception_plan,
            "team_plan": self._team_plan,
            "commitments": [item.model_dump(mode="json") for item in self._commitments],
        }
        return deepcopy(snapshot)

    def digest(self) -> str:
        """Stable digest for request provenance without exposing private text."""
        encoded = json.dumps(
            self.snapshot(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _uninformed_belief(seat: int) -> SeatBelief:
    return SeatBelief(
        seat=seat,
        wolf_probability=0.5,
        likely_role=None,
        confidence=0.0,
        evidence=("uninformed",),
        updated_day=0,
        updated_phase="setup",
    )


def _optional_bounded_text(value: Any, *, max_length: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:max_length] or None


def _project_probability_mass(values: list[float], *, target: float) -> list[float]:
    """Project non-negative weights onto a capped simplex deterministically."""
    if not values:
        return []
    target = max(0.0, min(float(len(values)), float(target)))
    if target == 0.0:
        return [0.0 for _ in values]
    if target == float(len(values)):
        return [1.0 for _ in values]
    weights = [max(0.0, min(1.0, float(value))) for value in values]
    result = [0.0 for _ in weights]
    active = set(range(len(weights)))
    remaining = target
    while active:
        total_weight = sum(weights[index] for index in active)
        if total_weight <= 0:
            share = remaining / len(active)
            for index in active:
                result[index] = share
            break
        saturated = {
            index
            for index in active
            if remaining * weights[index] / total_weight >= 1.0
        }
        if not saturated:
            for index in active:
                result[index] = remaining * weights[index] / total_weight
            break
        for index in saturated:
            result[index] = 1.0
            remaining -= 1.0
        active -= saturated
    return [max(0.0, min(1.0, value)) for value in result]
