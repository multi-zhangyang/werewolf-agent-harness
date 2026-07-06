"""Agent 记忆 —— 双记忆流(观察+反思)+ 纵向信任网络。

设计来源(ARCHITECTURE.md §3.2-3.3):
- Werewolf Arena + Generative Agents:观察记忆(游戏事件+特权信息)+ 反思记忆(每轮总结)。
- WOLF:LLM 检测欺骗整体仅 ~52%,但多轮交互提升召回率且不累积误判诚实角色。
  → 信任度随轮次平滑更新(纵向性),非单轮重判。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryItem:
    """一条记忆。"""

    day: int
    phase: str  # night/day/voting/setup
    kind: str  # event/reflection/claim/suspicion
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 0.0  # 0-1,三因子检索用(Generative Agents 2304.03442)

    def render(self) -> str:
        tag = f"[D{self.day} {self.phase}]"
        return f"{tag}({self.kind}) {self.text}"


# importance 启发式:按事件类型赋基础分(免 LLM 调用,成本低)。
# claim/contradiction/death/seer_result 是硬信号(高);speech/vote 中;phase_started/setup 低。
_IMPORTANCE_BY_KIND: dict[str, float] = {
    "claim": 0.9,
    "seer_action": 0.9,        # 预言家查验结果(特权硬信号)
    "death": 0.85,             # 死亡公告
    "wolf_caucus_consensus": 0.85,  # 狼队共识(私有硬信号)
    "wolf_caucus": 0.7,
    "wolf_kill_chosen": 0.8,
    "vote": 0.7,
    "speech": 0.5,
    "mentioned_silent": 0.6,   # 被点名未回应(心虚信号)
    "speech_skipped_dup": 0.2,
    "reflection": 0.55,
    "phase_started": 0.1,
    "role_assigned": 0.4,
    "teammate": 0.5,
}


def _importance_for(kind: str, text: str) -> float:
    """启发式 importance 打分(Generative Agents 用 LLM 打 1-10;此处免调用按类型估)。"""
    base = _IMPORTANCE_BY_KIND.get(kind, 0.4)
    # 文本含矛盾/对跳/查验等关键词再加权(claim 矛盾是狼人硬破绽)
    if any(k in text for k in ("对跳", "矛盾", "冲突", "假查验", "自相矛盾")):
        base = min(1.0, base + 0.1)
    return base


class AgentMemory:
    """单个 agent 的记忆流。

    - observations: GM 推送的可观察事件 + 角色特权信息(按时间序)。
    - reflections: 每轮结束 agent 自己生成的总结(key insights)。
    - trust: 纵向信任网络 {seat: suspicion 0-1},跨轮平滑更新。
    - claims: 各 seat 公开声称的身份/信息(矛盾检测用)。
    """

    def __init__(self, seat: int, role: str) -> None:
        self.seat = seat
        self.role = role
        self.observations: list[MemoryItem] = []
        self.reflections: list[MemoryItem] = []
        self.trust: dict[int, float] = {}  # seat -> suspicion
        self.claims: dict[int, list[dict[str, Any]]] = {}  # seat -> 历次声明

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def observe(self, day: int, phase: str, kind: str, text: str, **meta: Any) -> None:
        imp = meta.pop("importance", None)
        if imp is None:
            imp = _importance_for(kind, text)
        self.observations.append(MemoryItem(day, phase, kind, text, meta, importance=float(imp)))

    def reflect(self, day: int, phase: str, text: str, **meta: Any) -> None:
        imp = meta.pop("importance", None)
        if imp is None:
            imp = _importance_for("reflection", text)
        self.reflections.append(MemoryItem(day, phase, "reflection", text, meta, importance=float(imp)))

    def record_claim(self, seat: int, day: int, claim: dict[str, Any]) -> None:
        self.claims.setdefault(seat, []).append({"day": day, **claim})

    def detect_claim_contradictions(self) -> list[str]:
        """检测公开 claim 的矛盾(GRATR 证据图思路:跨轮追踪 claim 找前后不一)。

        返回矛盾描述列表,供渲染给 agent + 注入怀疑度。狼人易前后矛盾,这是硬信号。
        检测:
        - 对跳预言家:多人 claim role=seer
        - 同人查验结果矛盾:同一 seat 对同一 checked_seat 报不同 result
        - 互斥查杀:A 说 X 是狼,B(预言家)说 X 是好人
        """
        contradictions: list[str] = []
        # 收集每个 seat 的预言家声明
        seer_claims: dict[int, list[dict[str, Any]]] = {}
        for seat, clist in self.claims.items():
            seers = [c for c in clist if str(c.get("role", "")).lower() == "seer"]
            if seers:
                seer_claims[seat] = seers
        # 对跳预言家
        if len(seer_claims) >= 2:
            seats = sorted(seer_claims.keys())
            contradictions.append(
                f"对跳预言家:{','.join(f'{s}号' for s in seats)} 都声称是预言家,至多一个为真,其余是狼悍跳"
            )
        # 同人查验结果矛盾 + 互斥查杀
        # 建立 (checked_seat -> {claimer_seat -> result}) 映射
        checked_map: dict[int, dict[int, str]] = {}
        for claimer_seat, seers in seer_claims.items():
            for c in seers:
                cs = c.get("checked_seat")
                res = str(c.get("result", "")).lower()
                if cs is not None and res:
                    try:
                        cs = int(cs)
                    except (ValueError, TypeError):
                        continue
                    checked_map.setdefault(cs, {})[claimer_seat] = res
        for target, claimers in checked_map.items():
            results = set(claimers.values())
            if len(results) > 1:
                # 有多个预言家对同一目标报不同结果
                parts = [f"{s}号说{r}" for s, r in sorted(claimers.items())]
                contradictions.append(
                    f"查验冲突:对{target}号,{', '.join(parts)} —— 必有人报假查验"
                )
        # 同一预言家对同一目标前后报不同(自相矛盾)
        for claimer_seat, seers in seer_claims.items():
            by_target: dict[int, set[str]] = {}
            for c in seers:
                cs = c.get("checked_seat")
                res = str(c.get("result", "")).lower()
                if cs is not None and res:
                    try:
                        cs = int(cs)
                    except (ValueError, TypeError):
                        continue
                    by_target.setdefault(cs, set()).add(res)
            for target, results in by_target.items():
                if len(results) > 1:
                    contradictions.append(
                        f"{claimer_seat}号自相矛盾:对{target}号先后报不同结果{results}"
                    )
        return contradictions

    def update_trust(self, suspicion: dict[int, float], *, smoothing: float = 0.35) -> None:
        """纵向平滑更新信任度。

        新值 = (1-smoothing)*旧值 + smoothing*本轮值。smoothing 较小避免单轮误判
        暴涨(WOLF 警告会误判诚实村民),但允许多轮累积证据逐步改变判断。
        """
        for seat, new_val in suspicion.items():
            try:
                seat = int(seat)
            except (ValueError, TypeError):
                continue
            if seat == self.seat:
                continue
            old = self.trust.get(seat, 0.5)
            self.trust[seat] = (1 - smoothing) * old + smoothing * max(0.0, min(1.0, float(new_val)))

    def set_trust(self, seat: int, value: float) -> None:
        """直接设置信任度(特权信息触发,如预言家查验结果)。"""
        try:
            seat = int(seat)
        except (ValueError, TypeError):
            return
        if seat != self.seat:
            self.trust[seat] = max(0.0, min(1.0, float(value)))

    # ------------------------------------------------------------------
    # 读取(渲染给 prompt)
    # ------------------------------------------------------------------
    def recent_observations(self, limit: int = 30) -> list[MemoryItem]:
        """三因子检索(Generative Agents 2304.03442):recency×0.1 + relevance×1.0 + importance×1.0。
        替换纯 recency 的 [-limit:]。relevance = 与当前高怀疑对象的相关度(提及怀疑 seat 加权),
        让跨天 claim 矛盾/查验结果等关键记忆不被近期噪声淹没。"""
        if not self.observations:
            return []
        # 当前高怀疑座位(top-3),作为 relevance 查询的代理
        suspect_seats = {s for s, v in sorted(self.trust.items(), key=lambda kv: kv[1], reverse=True)[:3]
                         if v >= 0.5}
        current_day = max((m.day for m in self.observations), default=1)
        scored: list[tuple[float, MemoryItem]] = []
        for m in self.observations:
            # recency:按天指数衰减(0.5/天),越近越大
            age = max(0, current_day - m.day)
            recency = 0.5 ** age
            # relevance:文本提及高怀疑座位 +0.5
            relevance = 0.5 if any(f"{s}号" in m.text or f"{s}" in m.text for s in suspect_seats) else 0.0
            score = recency * 0.1 + relevance * 1.0 + m.importance * 1.0
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def render_for_prompt(self, *, obs_limit: int = 25, refl_limit: int = 6) -> str:
        """渲染记忆为 prompt 文本块。"""
        parts: list[str] = []
        obs = self.recent_observations(obs_limit)
        if obs:
            parts.append("【观察记忆】")
            parts.extend(f"- {m.render()}" for m in obs)
        if self.reflections:
            parts.append("\n【反思记忆(历轮关键洞察)】")
            parts.extend(f"- {m.render()}" for m in self.reflections[-refl_limit:])
        if self.trust:
            parts.append("\n【当前怀疑度】")
            sorted_trust = sorted(self.trust.items(), key=lambda kv: kv[1], reverse=True)
            parts.extend(f"- {seat}号: {val:.2f}" for seat, val in sorted_trust)
        if self.claims:
            parts.append("\n【各玩家身份声明记录】")
            for seat, claim_list in self.claims.items():
                claims_text = "; ".join(
                    f"D{c.get('day')}{c.get('claim','')}" if "claim" in c else f"D{c.get('day')}{c}"
                    for c in claim_list
                )
                parts.append(f"- {seat}号 声称: {claims_text}")
        # claim 矛盾检测(GRATR 证据图):跨轮追踪 claim,检测对跳/结果不一致
        contradictions = self.detect_claim_contradictions()
        if contradictions:
            parts.append("\n【⚠️ 声称矛盾检测(硬信号)】")
            parts.extend(f"- {c}" for c in contradictions)
            parts.append("→ 出现对跳/查验冲突,必有狼人报假查验,重点关注谁的前后发言更不一致。")
        return "\n".join(parts) if parts else "(尚无记忆)"

    def snapshot(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "role": self.role,
            "trust": dict(self.trust),
            "observation_count": len(self.observations),
            "reflection_count": len(self.reflections),
            "claims": {str(k): v for k, v in self.claims.items()},
        }
