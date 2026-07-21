"""Agent 决策数据模型 —— 统一意图接口。

无论 LLM 还是 Human,所有行为都表达为 Decision,由 RulesEngine 校验推进。
解耦"决策来源"与"规则执行"。承 ARCHITECTURE.md §3.1。
"""
from __future__ import annotations

import json
import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentAction(StrEnum):
    """Agent 可表达的意图类型。"""

    NIGHT_KILL = "night_kill"      # 狼人夜间杀人(目标)
    WOLF_COUNCIL = "wolf_council"  # 狼人发送团队私有建议(不直接结算)
    SEE = "see"                    # 预言家查验(目标)
    SAVE = "save"                  # 女巫救人或医生保护(目标可为自己)
    POISON = "poison"              # 女巫毒人(目标)
    GUARD = "guard"                # 守卫守护(目标)
    SPEAK = "speak"                # 白天发言(speech+bid)
    VOTE = "vote"                  # 投票放逐(目标)
    LAST_WORDS = "last_words"      # 遗言(speech)
    SKIP = "skip"                  # 不行动(合法弃权)


class Decision(BaseModel):
    """一次 agent 决策(意图)。引擎校验合法性后执行。

    所有字段都是 agent 的"主张",未必合法——harness/引擎负责校验或拒绝，
    不会替 Agent 改成另一个策略。
    reasoning 是私有推理,上帝/复盘可见,不广播给对手(保公平)。
    """

    model_config = ConfigDict(extra="forbid")

    action: AgentAction
    # Seat-native intent exactly as selected by the Agent. Internal player IDs
    # belong to the environment and are resolved only after protocol validation.
    target_seat: int | None = Field(default=None, strict=True)
    speech: str | None = None
    # Exact team-private text. Only WOLF_COUNCIL may carry this field; it is
    # delivered to living werewolves and never projected as public speech.
    team_message: str | None = None
    bid: int | None = Field(default=None, strict=True, ge=0, le=4)  # 0-4 发言意愿(竞价调度)
    # 私有推理，仅在授权的 DecisionEnvelope trace 中可见。
    reasoning: str | None = None
    # 声明:本回合公开声称的身份/信息(如"我是预言家,查3号是狼")
    claim: dict[str, Any] | None = None
    # Optional public relationship metadata selected by the Agent.
    # reply_to:本次发言回应/反驳的发言者座位(被点名/指控时填);accuses:本次点名指控的座位列表
    # Environment filters nonexistent/self seats for both fields. reply_to may
    # reference a dead player's historical speech/last words; accuses is a
    # current pressure signal and is restricted to living visible seats.
    reply_to: int | None = None
    accuses: list[int] | None = None

    skip_reason: str | None = None
    # Provenance from the one model call that produced this decision.  It is
    # stored in the harness trace, never serialized as part of the game action.
    llm_call_trace: dict[str, Any] | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _payload_is_finite_json(self) -> "Decision":
        try:
            json.dumps(self.claim, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as err:
            raise ValueError("decision claim must contain only finite JSON values") from err
        return self

    @field_validator("reply_to", mode="before")
    @classmethod
    def _validate_reply_to(cls, v: Any) -> int | None:
        """mode=before:容忍 "3号"/3.0 等,清洗为 int 或 None。"""
        if v is None:
            return None
        try:
            number = float(str(v).replace("号", "").strip())
            if not math.isfinite(number) or not number.is_integer():
                return None
            seat = int(number)
        except (ValueError, TypeError):
            return None
        return seat if seat > 0 else None

    @field_validator("accuses", mode="before")
    @classmethod
    def _validate_accuses(cls, v: Any) -> list[int] | None:
        """mode=before:在 pydantic 强制 list[int] 之前先清洗,容忍 "9号"/3.0 等。
        过滤非法/重复/非正座位。返回清洗后的 int 列表或 None。"""
        if v is None:
            return None
        if not isinstance(v, (list, tuple)):
            # 容忍单个值
            v = [v]
        seen: set[int] = set()
        result: list[int] = []
        for item in v:
            try:
                number = float(str(item).replace("号", "").strip())
                if not math.isfinite(number) or not number.is_integer():
                    continue
                seat = int(number)
            except (ValueError, TypeError):
                continue
            if seat > 0 and seat not in seen:
                seen.add(seat)
                result.append(seat)
        return result or None

    @property
    def is_skip(self) -> bool:
        return self.action == AgentAction.SKIP


class AgentObservation(BaseModel):
    """喂给 agent 的观察(从该 agent 视角投影的游戏状态 + 记忆)。

    这是信息隔离的出口:agent 只能基于此决策,看不到完整 GameState。
    """

    model_config = ConfigDict(use_enum_values=True)

    # 我的身份
    my_seat: int
    my_role: str
    my_team: str
    my_teammates: list[dict[str, Any]] = Field(default_factory=list)  # 狼人队友(seat/name)
    # 全部座位(公开信息)
    seats: list[dict[str, Any]] = Field(default_factory=list)
    alive_seats: list[int] = Field(default_factory=list)
    # 公开牌组分布(只含每种角色数量,不含座位身份)
    role_counts: dict[str, int] = Field(default_factory=dict)
    # 当前阶段
    phase: str
    day: int
    # 我能看到的公开事件历史(发言/投票/死亡公告)
    public_events: list[dict[str, Any]] = Field(default_factory=list)
    # 我的私有事件(查验结果/夜间行动反馈)
    private_events: list[dict[str, Any]] = Field(default_factory=list)
    # 当天已有的发言(竞价调度用)
    today_speeches: list[dict[str, Any]] = Field(default_factory=list)
    # 本回合可执行的动作提示(引擎告知 agent 此刻能做什么)
    available_actions: list[str] = Field(default_factory=list)
    # 可选目标座位(已随机化,对抗位置偏差)
    candidate_targets: list[int] = Field(default_factory=list)
    # 投票限制目标(PK 时只允许投 PK 候选;非 PK 时为空表示可投任意活人)
    vote_targets: list[int] = Field(default_factory=list)
    # 当前是否处于 PK(平票加赛)
    in_pk: bool = False
