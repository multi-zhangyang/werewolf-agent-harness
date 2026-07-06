"""Agent 决策数据模型 —— 统一意图接口。

无论 LLM 还是 Human,所有行为都表达为 Decision,由 RulesEngine 校验推进。
解耦"决策来源"与"规则执行"。承 ARCHITECTURE.md §3.1。
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentAction(StrEnum):
    """Agent 可表达的意图类型。"""

    NIGHT_KILL = "night_kill"      # 狼人夜间杀人(目标)
    SEE = "see"                    # 预言家查验(目标)
    SAVE = "save"                  # 女巫救人(目标,可为自己)
    POISON = "poison"              # 女巫毒人(目标)
    GUARD = "guard"                # 守卫守护(目标)
    SPEAK = "speak"                # 白天发言(speech+bid)
    VOTE = "vote"                  # 投票放逐(目标)
    LAST_WORDS = "last_words"      # 遗言(speech)
    SKIP = "skip"                  # 不行动(合法弃权)


class Decision(BaseModel):
    """一次 agent 决策(意图)。引擎校验合法性后执行。

    所有字段都是 agent 的"主张",未必合法——引擎负责校验/修正/拒绝。
    reasoning 是私有推理,上帝/复盘可见,不广播给对手(保公平)。
    """

    model_config = ConfigDict(extra="allow")

    action: AgentAction
    target_id: str | None = None
    speech: str | None = None
    bid: int | None = None  # 0-4 发言意愿(竞价调度)
    # 信任模型更新:对每个 seat 的怀疑度 0-1(纵向累积)
    suspicion: dict[int, float] | None = None
    # 私有推理(AI思考流/复盘可见)
    reasoning: str | None = None
    # 声明:本回合公开声称的身份/信息(如"我是预言家,查3号是狼")
    claim: dict[str, Any] | None = None
    # 结构化对话关系(方向A:让 agent 之间真正对话,而非轮流独白)
    # reply_to:本次发言回应/反驳的发言者座位(被点名/指控时填);accuses:本次点名指控的座位列表
    # 这俩是 agent 的"主张",引擎不校验合法性(铁律:harness 不替 agent 决策),
    # 用于发言调度(被提及者优先)和后续二阶 ToM 态度网络。
    reply_to: int | None = None
    accuses: list[int] | None = None
    # 二阶 ToM 态度网络(方向B):agent 显式声明本回合对其他座位的立场
    # (support=支持/帮腔, oppose=反对/指控, neutral=中立)。
    # 学术依据:S2§3.2 explicit belief graph(Sclar/Kassner/Li 2023,prompt 工程显式信念状态增强多 agent 协作)
    # + S2§3.3 Suspicion-Agent 二阶 ToM(预测对手相信我会做什么)。
    # 同样是 agent 的"主张",引擎不校验,用于聚合成 attitude_edges 注入他人观察(信念图)。
    attitudes: dict[int, str] | None = None
    # OSR 客观发言重写(Beyond Survival):投票前显式两段式——先客观摘要他人发言,
    # 再基于摘要投票。结构化字段强制 LLM 真的执行摘要(可审计),而非只在 thought 里要求。
    # 防御"被说服成坏选择":LLM 易把有说服力的话当字面指令,客观摘要剥离情绪话术后投票。
    objective_summary: str | None = None
    # 欺骗策略结构化(Werewolf Arena/WOLF 4 分类):狼人显式声明本回合用哪种欺骗手段。
    # 取值:omission(遗漏)/distortion(扭曲)/fabrication(捏造)/misdirection(误导)/none(无)。
    # 仅狼人发言有意义;结构化后 5维评审可对 DR(欺骗推理)/PS(劝说)打分有据,
    # 前端可展示"狼人用了X策略"。引擎不校验,agent 主张。
    deception: str | None = None

    # —— 清洗标记(承 no-fallback-design,透明审计) ——
    parse_failed: bool = False
    skip_reason: str | None = None

    @field_validator("bid")
    @classmethod
    def _clamp_bid(cls, v: int | None) -> int | None:
        if v is None:
            return None
        return max(0, min(4, int(v)))

    @field_validator("reply_to", mode="before")
    @classmethod
    def _validate_reply_to(cls, v: Any) -> int | None:
        """mode=before:容忍 "3号"/3.0 等,清洗为 int 或 None。"""
        if v is None:
            return None
        try:
            seat = int(float(str(v).replace("号", "").strip()))
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
                seat = int(float(str(item).replace("号", "").strip()))
            except (ValueError, TypeError):
                continue
            if seat > 0 and seat not in seen:
                seen.add(seat)
                result.append(seat)
        return result or None

    @field_validator("attitudes", mode="before")
    @classmethod
    def _validate_attitudes(cls, v: Any) -> dict[int, str] | None:
        """mode=before:容忍 "3号"/3.0 键 + 中文立场词,归一化为 support/oppose/neutral。
        过滤非法座位与非正座位。返回 {seat: stance} 或 None。"""
        if v is None:
            return None
        if not isinstance(v, dict):
            return None
        normalize = {
            "support": "support", "支持": "support", "帮腔": "support", "信任": "support", "agree": "support",
            "oppose": "oppose", "反对": "oppose", "指控": "oppose", "怀疑": "oppose", "disagree": "oppose",
            "neutral": "neutral", "中立": "neutral", "无": "neutral",
        }
        result: dict[int, str] = {}
        for k, val in v.items():
            try:
                seat = int(float(str(k).replace("号", "").strip()))
            except (ValueError, TypeError):
                continue
            if seat <= 0:
                continue
            stance = normalize.get(str(val).strip().lower()) or normalize.get(str(val).strip())
            if stance is None:
                # 不认识的立场词,按字面含"反/控/疑"判 oppose,含"支/帮/信"判 support,否则 neutral
                raw = str(val).strip()
                if any(ch in raw for ch in "反控疑敌"):
                    stance = "oppose"
                elif any(ch in raw for ch in "支帮信友同"):
                    stance = "support"
                else:
                    stance = "neutral"
            result[seat] = stance
        return result or None

    @field_validator("deception", mode="before")
    @classmethod
    def _validate_deception(cls, v: Any) -> str | None:
        """mode=before:容忍中文/大小写,归一化到 omission/distortion/fabrication/misdirection/none。"""
        if v is None:
            return None
        s = str(v).strip().lower()
        norm = {
            "omission": "omission", "遗漏": "omission", "省略": "omission",
            "distortion": "distortion", "扭曲": "distortion", "曲解": "distortion",
            "fabrication": "fabrication", "捏造": "fabrication", "编造": "fabrication",
            "misdirection": "misdirection", "误导": "misdirection", "转移": "misdirection",
            "none": "none", "无": "none", "诚实": "none", "真话": "none",
        }
        return norm.get(s)

    @field_validator("suspicion")
    @classmethod
    def _clamp_suspicion(cls, v: dict[int, float] | None) -> dict[int, float] | None:
        if v is None:
            return None
        result: dict[int, float] = {}
        for k, val in (v or {}).items():
            try:
                seat = int(k)
            except (ValueError, TypeError):
                continue
            result[seat] = max(0.0, min(1.0, float(val)))
        return result

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
    # 可见证据图 + 软角色后验(非真值):由 information.py 基于该玩家可见信息生成。
    evidence_graph: dict[str, Any] = Field(default_factory=dict)
    # 本回合可执行的动作提示(引擎告知 agent 此刻能做什么)
    available_actions: list[str] = Field(default_factory=list)
    # 可选目标座位(已随机化,对抗位置偏差)
    candidate_targets: list[int] = Field(default_factory=list)
    # 投票限制目标(PK 时只允许投 PK 候选;非 PK 时为空表示可投任意活人)
    vote_targets: list[int] = Field(default_factory=list)
    # 当前是否处于 PK(平票加赛)
    in_pk: bool = False


class AgentThinking(BaseModel):
    """agent 思考摘要(流式推给前端,经整理不暴露完整隐藏推理)。

    承产品愿景:AI 思考可视化是核心卖点。但为保公平,沉浸/观战模式只显示
    经整理的摘要,上帝模式可看完整 reasoning。
    """

    seat: int
    action: str
    summary: str  # 思考摘要:verbose=False 时一两句(保公平);verbose=True 时完整 reasoning(上帝/复盘/研究用)
    suspicion_top: list[dict[str, Any]] = Field(default_factory=list)  # 怀疑最高的几个
    bid: int | None = None
    reasoning: str | None = None  # 完整推理(verbose=True 时填充,暴露分析/欺骗算计/手段)
