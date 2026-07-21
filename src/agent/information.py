"""信息隔离层 —— GameState → AgentObservation 投影。

承 ARCHITECTURE.md §3.2 / §6.2:agent 只能基于 AgentObservation 决策,看不到完整 GameState。
特权信息(查验结果/队友/夜间反馈)只注入对应角色。候选目标随机化对抗位置偏差。
"""
from __future__ import annotations

import random
from collections import Counter
from copy import deepcopy
from typing import Any

from ..game.models import EventVisibility, GameState, Phase
from ..game.roles import Role, default_role_deck
from ..privacy import strip_model_private_reasoning
from .schemas import AgentObservation

_PUBLIC_SPEECH_KEYS = {
    "seat", "name", "text", "bid", "reply_to", "accuses", "claim", "day", "pk"
}
_PUBLIC_CLAIM_KEYS = {"role", "checked_seat", "result"}
_PUBLIC_EVENT_KEYS = {"id", "phase", "day", "type", "message", "visibility"}
_PUBLIC_EVENT_PAYLOAD_KEYS = {
    "dead_player_ids",
    "exiled_player_id",
    "hunter_id",
    "missing_player_ids",
    "player_id",
    "target_id",
    "target_seat",
    "text",
    "tied_player_ids",
    "voter_id",
    "voter_seat",
    "votes",
    "winner",
    "claim",
}


def build_observation(
    state: GameState,
    player_id: str,
    *,
    rng: random.Random | None = None,
    available_actions: list[str] | None = None,
    candidate_targets: list[int] | None = None,
    vote_targets: list[int] | None = None,
    in_pk: bool = False,
) -> AgentObservation:
    """从完整状态投影出某个玩家的观察。

    这是信息隔离的出口。绝不泄露该玩家无权看到的信息。
    candidate_targets: 环境已计算的本请求精确合法目标；None 时使用
        通用“存活且非自己”集合。列表只在副本上随机化。
    vote_targets: PK 时投票只能投这些座位(候选);None/空表示不限。
    """
    rng = rng or random.Random()
    viewer = state.get_player(player_id)
    role = Role(viewer.role) if viewer.role else Role.VILLAGER

    # —— 公开事件(所有人可见) ——
    public_events = [
        _sanitize_public_event_for_agent(e.model_dump())
        for e in state.events
        if e.visibility == EventVisibility.PUBLIC
    ]

    # —— 该玩家的私有事件(不混入 public; public 已在 public_events 中) ——
    private_events = [
        strip_model_private_reasoning(e.model_dump())
        for e in state.events
        if e.visibility == EventVisibility.PRIVATE and e.is_visible_to(player_id)
    ]

    # —— 队友(仅狼人) ——
    teammates: list[dict[str, Any]] = []
    if role == Role.WEREWOLF:
        teammates = [
            {"seat": p.seat, "name": p.name, "id": p.id}
            for p in state.players
            if p.role == Role.WEREWOLF and p.id != player_id
        ]

    alive_seats = [p.seat for p in state.living_players()]
    seats = [
        {"seat": p.seat, "id": p.id, "name": p.name, "alive": p.alive} for p in state.players
    ]
    role_counts = _public_role_counts(state)

    # —— 候选目标:只随机化环境投递的精确合法集合 ——
    # Never sort after this point: one Actor's private seeded RNG owns the
    # presentation order for this request, while set membership stays fixed.
    randomized_targets = list(
        candidate_targets
        if candidate_targets is not None
        else (seat for seat in alive_seats if seat != viewer.seat)
    )
    rng.shuffle(randomized_targets)
    randomized_vote_targets: list[int] = []
    if vote_targets:
        vote_set = set(vote_targets)
        randomized_vote_targets = [
            seat for seat in randomized_targets if seat in vote_set
        ]
        # This fallback only matters to direct callers that supplied a vote
        # restriction different from candidate_targets. Preserve every legal
        # vote target without introducing a deterministic tail.
        missing_vote_targets = [
            seat for seat in vote_targets if seat not in randomized_vote_targets
        ]
        rng.shuffle(missing_vote_targets)
        randomized_vote_targets.extend(missing_vote_targets)

    obs = AgentObservation(
        my_seat=viewer.seat,
        my_role=role.value,
        my_team=viewer.team or "village",
        my_teammates=teammates,
        seats=seats,
        alive_seats=alive_seats,
        role_counts=role_counts,
        phase=state.phase,
        day=state.day,
        public_events=public_events,
        private_events=private_events,
        today_speeches=[],  # 由编排器填入当天已有发言
        available_actions=available_actions or [],
        candidate_targets=randomized_targets,
        vote_targets=randomized_vote_targets,
        in_pk=in_pk,
    )
    return obs


def attach_today_speeches(obs: AgentObservation, speeches: list[dict[str, Any]]) -> AgentObservation:
    """把当天已有的发言注入观察(竞价调度时,后发言者能看到前面发言)。"""
    obs.today_speeches = [_sanitize_public_speech_for_agent(s) for s in speeches]
    return obs


def _sanitize_public_speech_for_agent(speech: dict[str, Any]) -> dict[str, Any]:
    """Return only fields observable by ordinary agents from a public speech.

    Private reasoning and internal routing fields must not reach another
    agent's observation.
    """
    clean = {k: deepcopy(speech.get(k)) for k in _PUBLIC_SPEECH_KEYS if k in speech}
    claim = clean.get("claim")
    if isinstance(claim, dict):
        clean["claim"] = {k: claim.get(k) for k in _PUBLIC_CLAIM_KEYS if k in claim}
    else:
        clean.pop("claim", None)
    return clean


def _sanitize_public_event_for_agent(event: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on fields nested in a nominally public domain event."""
    clean = {
        key: deepcopy(event.get(key))
        for key in _PUBLIC_EVENT_KEYS
        if key in event
    }
    payload = event.get("payload")
    if isinstance(payload, dict):
        visible_payload = {
            key: deepcopy(payload.get(key))
            for key in _PUBLIC_EVENT_PAYLOAD_KEYS
            if key in payload
        }
        claim = visible_payload.get("claim")
        if isinstance(claim, dict):
            visible_payload["claim"] = {
                key: deepcopy(claim.get(key))
                for key in _PUBLIC_CLAIM_KEYS
                if key in claim
            }
        elif "claim" in visible_payload:
            visible_payload.pop("claim", None)
        clean["payload"] = visible_payload
    else:
        clean["payload"] = {}
    return clean


def _public_role_counts(state: GameState) -> dict[str, int]:
    roles = [Role(p.role).value for p in state.players if p.role is not None]
    if len(roles) != len(state.players):
        roles = [role.value for role in default_role_deck(len(state.players))]
    return dict(sorted(Counter(roles).items()))
