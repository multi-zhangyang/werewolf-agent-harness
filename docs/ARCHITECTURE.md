# 多 Agent 狼人杀系统 — 顶层架构设计

> 2026-07-03。融合两轮真实技术检索成果(Werewolf Arena / Suspicion-Agent / WOLF / ChatArena / 多agent辩论 / Avalon / BotC)与本项目产品愿景。本文是实现蓝图。

## 0. 设计哲学

三句话定调,所有后续决策服从这三句:

1. **真实对局铁律**:每个 AI 决策必须来自真实 LLM 调用,绝不伪造。失败深度重试,不 fallback 出假决策(见 [[no-fallback-design]])。
2. **引擎主控,Agent 只表达意图**:规则推进/行动校验/胜负判定由确定性 `RulesEngine` 完成。LLM 只生成 `Decision`(意图),由引擎裁决是否合法。**绝不让 LLM 当裁判判胜负**——不确定性违背铁律。
3. **状态-观察分离**:完整 `GameState` 只存在于后端。每个 seat 只能拿到 `PlayerView` 投影。前端"只渲染后端给的视图"。信息隔离是架构属性,不是 UI 过滤。

## 1. 三大永恒矛盾(游戏设计根基)

所有优秀社交推理游戏都围绕三股张力,引擎必须保留:

| 矛盾 | 体现 | 工程要求 |
|------|------|----------|
| 知情少数 vs 不知情多数 | 狼人互认,村民不知 | 角色级观察投影 |
| "知道但不能表现得太知道"(梅林/Seer 困境) | 预言家早跳信息量大但风险高 | agent 需权衡跳身份时机 |
| 软 vs 硬信息 | 查验结果是硬信息,但 BotC 证明硬信息也不可 100% 信任 | 信任模型须可被"投毒/混淆" |

**平衡护栏**(Werewolf Arena Monte Carlo 基线,必须复刻为测试):
- 无信息交换 → 村民胜率 1.2%(信息太少)
- Seer 自动揭示 + 村民盲信 → 100%(信息太确定)
- **健康区间 40%–60%**,自对弈平衡性验证

## 2. 分层架构

```
┌─────────────────────────────────────────────────────────┐
│  frontend/  (React + Vite + Tailwind v4 + shadcn/ui)     │
│  4 模式:观战/人机/上帝/复盘 · 圆桌 HUD · AI思考流 ·       │
│  欺骗/合谋/后验审计 · 昼夜主题 · 票型与人机操作提示       │
└───────────────▲──────────────────────────┬──────────────┘
        WebSocket v2 协议            REST (create/providers/config)
┌───────────────┴──────────────────────────▼──────────────┐
│  src/api/  (FastAPI)                                     │
│  server · websocket · room_manager · routes              │
│  房间生命周期 · 游戏循环 · 信息隔离广播 · 赛后复盘 · 持久化│
└───────────────▲──────────────────────────▲──────────────┘
                │                          │
┌───────────────┴────────┐    ┌────────────┴──────────────┐
│  src/game/  (确定性)    │    │  src/agent/  (LLM 决策)    │
│  rules · models · roles │    │  actor · memory · prompts  │
│  orchestrator (GM编排)  │◄───┤  schemas · information     │
│  引擎主控状态机          │ Decision(意图) │ ToM+反思+信任 │
└───────────────┬─────────┘    └────────────┬──────────────┘
                │                           │
┌───────────────▼───────────────────────────▼──────────────┐
│  src/llm/  router · models                               │
│  多 provider(openai/anthropic)· 重试退避 · json_object ·  │
│  流式 · 并发 Semaphore · 调用统计 · 绝不伪造              │
└───────────────────────────────────────────────────────────┘
```

对应 ChatArena 四层抽象(Environment→Player→Moderator→LanguageBackend),映射到本项目的 `src/game`(Environment+Moderator)、`src/agent`(Player)、`src/llm`(LanguageBackend)。

## 3. Agent 核心设计(这是"真正 agent 系统"的心脏)

### 3.1 Decision 抽象(统一意图接口)

无论 LLM 还是 Human,所有行为统一为结构化 `Decision`,引擎统一校验推进。解耦决策来源与规则执行。

```python
class Decision(BaseModel):
    action: Literal["night_kill","see","save","poison","guard","speak","vote","bid","last_words","skip"]
    target_id: str | None = None       # 行动目标(seat id)
    speech: str | None = None          # 发言内容
    bid: int | None = None             # 0-4 竞价(发言意愿)
    suspicion: dict[str, float] | None = None  # 对各 seat 的怀疑度 0-1(信任模型更新)
    reasoning: str | None = None       # 私有推理(上帝/复盘可见,不广播)
    # 清洗标记(承 no-fallback-design)
    _parse_failed: bool = False
    _skip_reason: str | None = None
```

### 3.2 Memory Stream(双记忆,来自 Werewolf Arena + Generative Agents)

每个 agent 独立维护,跨轮累积,**这是 LLM 善骗不善识破(~52% 检测率)的对抗武器**:

- **Observational Memory(观察记忆)**:GM 推送的所有可见游戏事件(死亡/放逐/发言/投票)+ 角色特权信息(Seer 查验结果、狼人队友、夜间杀人目标)。按 day/phase 索引。
- **Reflective Memory(反思记忆)**:每轮(夜/日)结束,agent 异步生成总结,提炼 key insights——"谁发言矛盾""谁跳了什么身份""谁被怀疑却没辩护"。用于后续轮次的**模式识别与矛盾检测**。

记忆注入 prompt:`Agent's memories(观察+反思) + current game state from agent's perspective + action-specific instructions`。

### 3.3 信任网络 / 纵向信任模型(来自 WOLF)

WOLF 发现:LLM 检测欺骗整体仅 ~52%,但**多轮交互提升召回率且不累积误判诚实角色**。所以:

- 每个 agent 维护 `suspicion: {seat_id: float}` 怀疑度,随轮次**平滑更新**(纵向性),非单轮重判。
- 每次发言/投票/跳身份后,agent 更新对说话者的怀疑度(私有推理产出)。
- 投票决策基于怀疑度排序 + 反思记忆中的矛盾点。
- **误报控制**:WOLF 警告会误判诚实村民为狼——prompt 须要求 agent "基于证据而非感觉",降低误报。

### 3.4 Theory of Mind(来自 Suspicion-Agent)

GPT-4 展现高阶 ToM——能推测他人心理状态并影响其行为。在狼人杀中:

- 狼人 agent 推测"村民现在怀疑谁",据此选择转移焦点或伪装。
- 预言家推测"如果我跳身份,狼人会杀谁",权衡跳身份时机(Seer 困境)。
-村民推测"这个跳预言家的是真还是狼人悍跳"。

prompt 中显式要求 agent "推断其他玩家的信息状态与意图",而非只陈述自己的怀疑。

### 3.5 三层 Prompt 架构(来自 AIwerewolf)

`Persona(人格风格) × Role(角色策略) × Strategy(情境策略)`,可独立迭代:

- **Persona**:发言风格(谨慎/激进/幽默/煽动),让 agent 有个性,也制造风格差异(已实证影响胜负——GPT-4 冗长被 Gemini 村民当可疑)。
- **Role**:角色目标 + 能力 + 信息 + 行为规范(狼人伪装策略/预言家跳身份权衡/女巫救人毒人权衡)。
- **Strategy**:当前情境下的策略(被指控如何辩护/首夜要不要跳/平票怎么办)。

### 3.6 竞价发言调度(来自 Werewolf Arena,突破性设计)

取代固定发言顺序,让 agent 自主表达发言意愿,**让 AI 对话像真人流动**:

```
竞价 0-4:
  0 = 倾听,暂不发言
  1 = 有一般想法
  2 = 有关键具体贡献
  3 = 极其迫切,必须下一个发言
  4 = 被直接点名/指控,必须回应
调度:最高出价者发言;平局时"上一轮被提及者"优先(被指控者倾向回应的自然偏置)。
辩论上限:8 轮(防后期重复);首轮保证每人有机会。
```

实测效果:多数共识在第 2-5 句达成,关键信息早传播。GPT-4 狼人竞价过高反而暴露。

### 3.7 随机化对抗 LLM 位置偏差(Werewolf Arena)

投票/夜间行动时,**随机化候选名单顺序**,对抗 LLM 倾向选首/尾项。名字从池随机选取减少名字偏差。

## 4. 当前规则引擎能力

`rules.py` / `orchestrator.py` 当前已经实现标准 6-12 人狼人杀主链路:

- setup / night / day / voting / pk / ended 阶段推进。
- 狼人夜杀、预言家查验、女巫救/毒、守卫守护、猎人开枪。
- 白天竞价发言、被点名回应优先、狼人白天 caucus、公开发言写入可见记忆。
- 投票、平票 PK、遗言、猎人追加死亡链。
- 胜负判定、死亡公告、运行中角色不泄露、赛后角色/analysis 公开。
- `requested_action` 已在 AI 与 human actor 间闭环,前端不需要暗猜夜间动作。

胜负当前按屠边/人数边界执行:狼全死→好人胜;狼数达到或超过非狼→狼人胜。自定义角色板 `deck` 暂不支持,REST 创建房间会明确 400,避免假支持。

## 5. 游戏编排器(GM)

`src/game/orchestrator.py`——确定性 GM,编排全流程:

```
SETUP → deal_roles(随机洗牌)
  └→ NIGHT(day=1) [并发:所有夜间行动 agent 同时决策]
       狼人合谋杀人(队内私聊达成一致)·守卫守护·女巫救/毒·预言家查验
       → resolve_night(守卫连守判定·女巫救毒结算·死亡公告)
       → check_winner
  └→ DAY [竞价发言循环:bid→speak,上限8轮·首轮每人机会]
       → VOTING [所有存活者投票·候选名单随机化]
       → resolve_vote(平票PK·遗言·猎人开枪)
       → check_winner
  └→ 若未结束:day+=1 → NIGHT(循环)
  └→ ENDED → 赛后复盘 game_analysis
```

- 夜间 agent 决策**并发**(asyncio.gather),日间发言**顺序**(竞价调度)。
- GM 用 RulesEngine 推进,只把"可观察事件"推入各 agent 的观察记忆,特权信息只入对应角色记忆。
- **合成投票(可选)**:每句发言后模拟投票追踪共识(Shannon 熵),不影响实际游戏、不写入记忆——用于复盘展示"共识何时达成"。

## 6. API 与 WebSocket 协议

### 6.1 REST
- `GET /api/providers` → provider 元信息
- `GET /api/config` → 当前默认模型配置(永远不返回 API key,只返回 `api_key_configured`)
- `POST /api/rooms` (body: `player_names` / `human_seats` / `model_config`) → 创建房间
- `GET /api/rooms/{id}` → 房间状态。running 阶段不泄露角色;ended 才公开角色。
- `POST /api/rooms/{id}/start` → waiting 房间开始真实对局
- `POST /api/rooms/{id}/seats/{seat}/model_config` → per-seat 模型覆盖(仅 waiting)
- `GET /api/rooms/{id}/replay` → 赛后回放。running 阶段返回 409。

### 6.2 WebSocket v2(实时事件流)
客户端连 `/ws/{room_id}?seat={seat}&mode={spectate|play|god|replay}`。服务端按 mode 下发不同投影:

- `spectate`:公开事件流(不泄露角色)
- `play`:该 seat 的 private_view + 可操作指令
- `god`:全知(所有角色/agent 怀疑度/私有推理/竞价)
- `replay`:仅赛后允许连接,按时间轴回放全部事件(含隐藏信息)

当前事件类型以 `frontend/src/lib/types.ts` 为前端镜像,主要包括:

- `snapshot`
- `room_status`
- `phase_started`
- `night_resolved`
- `speech`
- `vote_cast`
- `vote_resolved`
- `vote_incomplete`
- `last_words`
- `hunter_shot`
- `wolf_caucus` / `wolf_caucus_consensus`(god/replay)
- `trust_update` / `reflections_update`(god)
- `agent_thinking`
- `agent_decision_failed`
- `human_action_request`(对应 play seat)
- `game_ended`
- `analysis`(赛后真值与审计)
- `game_error`

**信息隔离广播**:公开事件按 room manager 的 public allowlist 下发;私有事件使用 `visibility`/`recipients` 过滤。前端永远不应收到运行中无权看到的角色真值、狼队私聊、私有 reasoning 或赛后 truth analysis。

## 7. 前端(真实 shadcn/ui 的游戏化 UI)

当前前端是 React + Vite + Tailwind v4 + shadcn/ui(radix-nova) 组件栈。开发模式由 Vite dev server 代理 `/api` 和 `/ws`;生产模式先 `npm run build`,再由 FastAPI 服务 `frontend/dist`。

- **圆桌观战台**:顶部 `GameHud`,左侧房间/玩家,中央圆桌座位环 + 实况交锋,右侧阶段/人机/审计面板。
- **真实 shadcn 组件**:业务按钮、表单、弹窗、卡片、徽章、进度条、Tabs、Accordion、Avatar 均使用 `frontend/src/components/ui` 中的真实 shadcn/Radix 组件组合;CSS 负责游戏布局和主题。
- **AI 思考流**:观战可见净化后的思考摘要;上帝/复盘可见更完整推理摘要。隐藏推理不发给普通玩家。
- **Agent 对抗可视化**:发言、投票、thinking、deception/collusion/posterior analysis 通过 WS 真实事件归约展示,不使用 mock 数据。
- **人机操作**:后端通过 `human_action_request` 明确 `requested_action`;前端发送 `night_kill/see/save/poison/guard/hunter_shot/vote` 等真实 action,不暗猜接口。
- **结束态复盘**:非 replay 观战/人机/上帝模式在 `analysis` 到达后也展示胜负海报和关键转折故事线;完整 replay 仍保留密集审计指标。
- **统一角色头像**:`RoleAvatar` 基于真实 shadcn `Avatar` + lucide glyph,圆桌、左栏和聊天消息复用同一角色视觉。
- **待完善方向**:右栏阶段信息/事件日志/胜利条件收敛、中心 ChatRoom 裁判记录面板、角色头像资产继续精修。

## 8. 评测与复盘

- **WereAlign 五维评分**:RI/SJ/DR/PS/CT + `game_quality`,作为解释层,不单独当统计显著性证明。
- **确定性轨迹指标**:`objective_metrics`,用赛后真值复算 vote accuracy、accuse precision、OSR/CT 等。
- **EvidenceGraph / posterior metrics**:`posterior_trace`、Brier/log loss/ECE、correct/wrong herding、constrained posterior。
- **欺骗/合谋审计**:`deception_audit` 与 `collusion_audit`,记录 speaker intent、审计类型、listener shift、windowed relay、pair-level misdirection。
- **多局统计**:`tests/multi_game_stats.py` 输出 JSONL、Wilson/Bootstrap CI 和 per-policy ABBA 统计。真实统计会调用模型,不能当普通单测运行。

## 9. 技术栈与铁律

- 后端:Python 3.12 / FastAPI / WebSocket / Pydantic 2
- 前端:React 18 + Vite + Tailwind v4 + shadcn/ui(radix-nova)
- LLM:按标准接口路由,`openai` = OpenAI Chat Completions,`openai_responses` = OpenAI Responses API,`anthropic` = Anthropic Messages API;模型 ID 由 `WEREWOLF_LLM_MODEL` 配置。
- 铁律:WEREWOLF_ 前缀配置绝不污染系统 env;凭据只存内存不落盘;输出 token 上限按协议映射(OpenAI Chat/Responses 的 0 不传字段,Anthropic 的 0 使用后端默认 max_tokens);绝不 fallback 伪造;宽松超时(180s)不误杀;深度重试(≥5次,指数退避带抖动)。

## 10. 实现顺序

1. LLM router(`src/llm/router.py`)— 真实调用层,agent 的地基
2. Agent 核心(`src/agent/`)— schemas/memory/prompts/actor/information
3. 规则引擎扩展(女巫/守卫/猎人/遗言/PK)
4. 游戏编排器(GM 昼夜循环+竞价发言)
5. API 层(server/websocket/room_manager/routes)
6. 前端(观战/人机/上帝/复盘 UI)
7. 测试 + 端到端真实对局验证
