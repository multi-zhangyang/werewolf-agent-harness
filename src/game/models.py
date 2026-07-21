"""Pydantic domain models for a Werewolf game."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from ..privacy import strip_model_private_reasoning
from .roles import Role, Team, role_team


class Phase(StrEnum):
    SETUP = "setup"
    NIGHT = "night"
    DAY = "day"
    VOTING = "voting"
    ENDED = "ended"


class EventVisibility(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"


class NightActionType(StrEnum):
    KILL = "kill"
    SEE = "see"
    SAVE = "save"
    POISON = "poison"
    GUARD = "guard"


class DeathReason(StrEnum):
    """死亡原因(影响猎人能否开枪:被毒不能开枪)。"""

    WOLF_KILL = "wolf_kill"
    EXILED = "exiled"
    POISONED = "poisoned"
    HUNTER_SHOT = "hunter_shot"
    WITCH_POISON = "witch_poison"


class Event(BaseModel):
    """A public announcement or private player-facing event."""

    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    phase: Phase
    day: int
    type: str
    message: str
    visibility: EventVisibility = EventVisibility.PUBLIC
    recipients: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("recipients")
    @classmethod
    def _private_events_need_recipients(cls, v: list[str], info: Any) -> list[str]:
        visibility = info.data.get("visibility")
        if visibility == EventVisibility.PRIVATE and not v:
            raise ValueError("private events require at least one recipient")
        return v

    def is_visible_to(self, player_id: str | None = None) -> bool:
        if self.visibility == EventVisibility.PUBLIC:
            return True
        return bool(player_id and player_id in self.recipients)


class PlayerState(BaseModel):
    """State for one seat."""

    model_config = ConfigDict(use_enum_values=True)

    id: str
    name: str
    seat: int
    role: Role | None = None
    alive: bool = True
    death_reason: DeathReason | None = None  # 死亡原因(被毒不能开枪)
    death_day: int | None = None

    @computed_field
    @property
    def team(self) -> Team | None:
        if self.role is None:
            return None
        return role_team(Role(self.role))

    def public_view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "seat": self.seat,
            "alive": self.alive,
        }

    def private_view(self) -> dict[str, Any]:
        view = self.public_view()
        view["role"] = self.role
        view["team"] = self.team
        return view


class NightAction(BaseModel):
    """A hidden night action submitted by a living role."""

    model_config = ConfigDict(use_enum_values=True)

    actor_id: str
    action: NightActionType
    target_id: str


class Vote(BaseModel):
    """A public daytime exile vote."""

    voter_id: str
    target_id: str


class GameState(BaseModel):
    """Complete mutable game state.

    Use ``public_view`` or ``private_view_for`` when sending state to players.
    Direct serialization contains hidden data and is intended only for trusted
    server-side storage.
    """

    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    phase: Phase = Phase.SETUP
    day: int = 0
    players: list[PlayerState] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    night_actions: list[NightAction] = Field(default_factory=list)
    votes: dict[str, str] = Field(default_factory=dict)
    winner: Team | None = None
    # —— 扩展状态:角色技能持久信息 ——
    witch_antidote: bool = True  # 女巫解药是否还在
    witch_poison: bool = True    # 女巫毒药是否还在
    last_guarded_seat: int | None = None  # 守卫上一夜守护的座位(连守限制)
    pending_hunter: list[str] = Field(default_factory=list)  # 待开枪的猎人 id(被毒不能开枪则空)
    last_words_queue: list[dict[str, Any]] = Field(default_factory=list)  # 待发表的遗言
    pk_candidates: list[str] = Field(default_factory=list)  # 平票 PK 候选 id
    night_kill_target: str | None = None  # 本夜狼人击杀目标(结算前暂存)
    night_deaths: list[dict[str, Any]] = Field(default_factory=list)  # 本夜死亡记录(供复盘/遗言)

    def get_player(self, player_id: str) -> PlayerState:
        for player in self.players:
            if player.id == player_id:
                return player
        raise KeyError(f"unknown player id: {player_id}")

    def living_players(self) -> list[PlayerState]:
        return [player for player in self.players if player.alive]

    def public_events(self) -> list[Event]:
        return [event for event in self.events if event.visibility == EventVisibility.PUBLIC]

    def events_for(self, player_id: str) -> list[Event]:
        return [event for event in self.events if event.is_visible_to(player_id)]

    def public_view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase,
            "day": self.day,
            "players": [player.public_view() for player in self.players],
            "events": [
                strip_model_private_reasoning(event.model_dump())
                for event in self.public_events()
            ],
            "votes": dict(self.votes) if self.phase in {Phase.VOTING, Phase.DAY} else {},
            "winner": self.winner,
        }

    def private_view_for(self, player_id: str) -> dict[str, Any]:
        viewer = self.get_player(player_id)
        view = {
            **self.public_view(),
            "self": viewer.private_view(),
            "events": [
                strip_model_private_reasoning(event.model_dump())
                for event in self.events_for(player_id)
            ],
        }
        role = Role(viewer.role) if viewer.role is not None else None
        if role == Role.WITCH:
            view["role_state"] = {
                "witch_antidote": self.witch_antidote,
                "witch_poison": self.witch_poison,
            }
        elif role == Role.GUARD:
            view["role_state"] = {
                "last_guarded_seat": self.last_guarded_seat,
            }
        elif role == Role.HUNTER:
            view["role_state"] = {
                "pending_hunter": viewer.id in self.pending_hunter,
            }
        return view


PublicView = dict[str, Any]
PrivateView = dict[str, Any]
RoleAssignmentMode = Literal["random", "fixed"]
