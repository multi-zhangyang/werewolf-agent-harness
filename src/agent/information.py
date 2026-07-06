"""信息隔离层 —— GameState → AgentObservation 投影。

承 ARCHITECTURE.md §3.2 / §6.2:agent 只能基于 AgentObservation 决策,看不到完整 GameState。
特权信息(查验结果/队友/夜间反馈)只注入对应角色。候选目标随机化对抗位置偏差。
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any

from ..game.models import EventVisibility, GameState, Phase
from ..game.roles import Role, default_role_deck
from .evidence import build_evidence_graph
from .schemas import AgentObservation

_PUBLIC_SPEECH_KEYS = {
    "seat", "name", "text", "bid", "reply_to", "accuses", "attitudes", "claim", "day", "pk"
}
_PUBLIC_CLAIM_KEYS = {"role", "checked_seat", "result"}


def build_observation(
    state: GameState,
    player_id: str,
    *,
    rng: random.Random | None = None,
    available_actions: list[str] | None = None,
    vote_targets: list[int] | None = None,
    in_pk: bool = False,
) -> AgentObservation:
    """从完整状态投影出某个玩家的观察。

    这是信息隔离的出口。绝不泄露该玩家无权看到的信息。
    vote_targets: PK 时投票只能投这些座位(候选);None/空表示不限。
    """
    rng = rng or random.Random()
    viewer = state.get_player(player_id)
    role = Role(viewer.role) if viewer.role else Role.VILLAGER

    # —— 公开事件(所有人可见) ——
    public_events = [
        e.model_dump() for e in state.events if e.visibility == EventVisibility.PUBLIC
    ]

    # —— 该玩家的私有事件(不混入 public; public 已在 public_events 中) ——
    private_events = [
        e.model_dump()
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

    # —— 候选目标:活人(去掉自己),随机化对抗位置偏差 ——
    candidate_targets = [s for s in alive_seats if s != viewer.seat]
    rng.shuffle(candidate_targets)

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
        candidate_targets=candidate_targets,
        vote_targets=list(vote_targets) if vote_targets else [],
        in_pk=in_pk,
    )
    obs.evidence_graph = build_evidence_graph(obs)
    return obs


def attach_today_speeches(obs: AgentObservation, speeches: list[dict[str, Any]]) -> AgentObservation:
    """把当天已有的发言注入观察(竞价调度时,后发言者能看到前面发言)。"""
    obs.today_speeches = [_sanitize_public_speech_for_agent(s) for s in speeches]
    obs.evidence_graph = build_evidence_graph(obs)
    return obs


def _sanitize_public_speech_for_agent(speech: dict[str, Any]) -> dict[str, Any]:
    """Return only fields observable by ordinary agents from a public speech.

    The orchestrator keeps extra metadata for god/replay/analysis, including
    deception strategy and private reasoning. Those fields must not reach the
    live agent observation or evidence graph.
    """
    clean = {k: speech.get(k) for k in _PUBLIC_SPEECH_KEYS if k in speech}
    claim = clean.get("claim")
    if isinstance(claim, dict):
        clean["claim"] = {k: claim.get(k) for k in _PUBLIC_CLAIM_KEYS if k in claim}
    else:
        clean.pop("claim", None)
    return clean


def _public_role_counts(state: GameState) -> dict[str, int]:
    roles = [Role(p.role).value for p in state.players if p.role is not None]
    if len(roles) != len(state.players):
        roles = [role.value for role in default_role_deck(len(state.players))]
    return dict(sorted(Counter(roles).items()))
