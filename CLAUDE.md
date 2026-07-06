# CLAUDE.md — werewolf-mas 工作指导

> 本文件是项目的 **工作宪法**,由 2026-07-04 第三轮联网深研(deep-research workflow,103 agent / 569k token / 498 工具调用)综合报告 + 自研笔记(`docs/research-multi-agent-harness-2026-07-04.md`)+ 方向 A 端到端验证后写成。
> 每次开工先读本文件,严格遵循分层铁律与八条工作铁律。
> 最近更新:2026-07-06。

---

## 0. 一句话定位

**werewolf-mas = 一个真实 LLM 多 Agent 对抗社会的狼人杀呈现。**
不是聊天演示,不是 JSON 解析练习,不是传输层工程——核心是 **agent 之间真正的博弈、欺骗、识谎、信念交锋**。所有工作必须服务于"agent 社会的本质",而不是周边脚手架。

---

## 1. 项目目标

- **真实**:真实 LLM API 调用,真实对局,真实决策。绝不伪造/回退/回放。
- **多 Agent 对抗**:agent 之间真正对话、回应、指控、识谎(不是轮流独白)。
- **harness/agent 分层**:编排(orchestrator)是 harness,LLM(actor)是 agent 大脑,严格分离。
- **顶级呈现**:顶级游戏化 UI + Agent 可视化 + AI 思考摘要,但呈现服务于对抗,不取代对抗。
- **以研究为基**:每个 agent 社会机制设计都要有学术支撑(arXiv 论文/框架文档),区分"已实证"与"合理推测"。

---

## 2. harness / agent 分层铁律(核心,来自 deep-research)

### 2.1 定义(deep-research finding [0], high confidence, 3-0 一致)

- **agent = LLM 大脑**:推理决策引擎,通过 prompt 被赋予角色/工具/参数。
- **harness/scaffold = 大脑之外的一切**:编排循环、记忆存储机制、规划模块结构、工具 API、环境。
- 术语注意(finding caveat):"harness/scaffold" **不是论文原文**,是研究者对"operational framework / orchestration"的诠释性叠加。论文用 brain/perception/action(Xi 2309.07864)、Profile/Memory/Planning/Action(2308.11432)。**实质内容有据,术语是合理转述**。

### 2.2 werewolf-mas 的分层映射(deep-research finding [0][3],自研笔记实证)

| 学术概念 | werewolf-mas 实现 | 状态 |
|---|---|---|
| **harness/编排** | `GameOrchestratorV2`(`src/game/orchestrator.py`) | ✅ 符合 AutoGen Team 编排器模型 |
| **agent/大脑** | `AgentActor`(`src/agent/actor.py`) | ✅ 符合 AutoGen participant agent 模型 |
| Profile 模块 | `prompts.PERSONAS` + `assign_persona` | ✅ |
| Memory 模块 | `agent/memory.py`(观察/反思/trust/claims + 三因子检索) | ✅ |
| Planning 模块 | 无显式(单步决策) | ⚠️ 狼人杀单步,可弱 |
| Action 模块 | `schemas.Decision` + `RulesEngine` | ✅ |
| Team 模式 | 第0轮 RoundRobin + 后续 Selector(bid) | ✅ 混合 |

### 2.3 分层铁律(不可违反)

1. **harness 不替 agent 决策**:orchestrator 只编排轮次/调度/广播/规则裁决,绝不伪造 agent 的发言/投票/技能目标。agent 决策失败时**深度重试**(≥5 次,指数退避+抖动),不回退到规则 bot 或预设台词。
2. **agent 不越权 harness**:actor 只在被 orchestrator 调用时生成回应,不主动改游戏状态、不直接广播、不访问特权信息。
3. **编排逻辑与 agent 逻辑分离**(AutoGen v0.4 原则,finding [3]):连接池/超时/重试/调度是 harness 基础设施职责,不污染 agent 层。
4. **信息隔离是 harness 职责**:`agent/information.py` 按 visibility 过滤观察,agent 只看到它该看的(狼人不知预言家查验、好人不知狼队身份)。

### 2.4 拓扑定位(deep-research finding [1][2], high confidence)

狼人杀 = **Competitive(目标冲突) + Layered(白天广播 + 夜晚狼队子团队) + Sandbox(环境编码昼夜/讨论/投票/奖励规则)**。
白天发言 = Selector(bid 决定发言者)+ Swarm 元素(被点名 handoff 回应,方向 A 已补)。
夜间狼队 = 子团队私聊拓扑(`_werewolf_deliberation`)。
Multiagent debate(findding [2], Du 2305.14325)是白天 PK/投票辩论的架构原型。

---

## 3. 已完成的工作(本会话)

### 3.1 router 流式改造(task #26 ✅)
- 问题:非流式请求长连接超时。
- 修复:`src/llm/router.py` 改 SSE `stream:True` + `stream_options:{include_usage:True}`,帧间 60s 超时,总超时取消;`_get_stream_client` 缓存 client;`_is_retryable` 加 `__cause__` 检查认 httpx 错误可重试。

### 3.2 方向 A:bid=4 真回应 + 结构化指控(task #27 ✅,端到端验证)
- 代码:`schemas.py` 加 `reply_to`/`accuses`(mode=before validator 容忍 "3号");`prompts.py` SPEAK_SCHEMA 加字段 + speak_instruction 加对话关系提示 + speeches 渲染补元数据;`actor._sanitize_speak` 解析;`orchestrator._run_day`/`_bid_order_by_llm` 重写(被提及者且 bid≥4 最优先 → bid 降序 → 被提及者优先 → 随机,不强制 bid<4 者叫起);`_is_duplicate_speech` 加 `relax_self` 豁免反驳。
- 验证(god 模式真实对局):结构化字段端到端流转;2号指控4号→4号 bid=4 被优先叫起回应+反指控→1号加入三方辩论;0 决策失败;狼人胜。**真正的对话交锋,非轮流独白**。
- 拓扑依据:AutoGen Swarm handoff(findding [3])+ Werewolf Arena bid 竞价。

### 3.3 公网攻击堵住(task ✅)
- 8000 端口原绑 0.0.0.0 公网暴露(157.230.130.220),被路径穿越扫描。改绑 127.0.0.1(用户确认"关闭端口本地跑")。

### 3.4 第三轮多 agent harness 研究(✅)
- 自研笔记 `docs/research-multi-agent-harness-2026-07-04.md`(六来源全文实证)。
- deep-research workflow(103 agents, 569k tokens, 498 tool calls, 6818s):8 条 high-confidence findings + 5 条 caveats。完整输出 `/tmp/claude-0/-root-werewolf-mas/e9f6d96a-e218-4816-bfca-175a60ae9ad3/tasks/wz4vn6aea.output`。

### 3.5 方向 B:二阶 ToM 态度网络(task #29 ✅,端到端验证)
- 代码:`schemas.py` 加 `attitudes: dict[int,str]`(mode=before validator 容忍 "3号"/"反对"→归一化 support/oppose/neutral);`prompts.py` SPEAK_SCHEMA 加 attitudes 字段 + speak_instruction 加二阶 ToM 提示(预测对手相信你会做什么)+ 新增 `_attitude_graph` helper 融合 accuses/attitudes/vote_cast 成显式信念图 + render_observation 加"态度网络"block(带 `←你` 自标记);`actor._extract_attitudes` 解析归一化;`orchestrator` today_speeches + speech 事件 + PK 路径都带 attitudes。
- 验证(god 模式真实对局,day2 好人胜):33 发言 **100% 带 attitudes**,18 support 边 + 39 oppose 边,与方向A accuses(26)+reply_to(18)融合;0 决策失败。**二阶 ToM 涌现**:6号(狼)"5号你跟得也太快了,是不是提前知道1号要跳?你俩这默契有点怪"(识破抱团);4号/5号用态度边推断"2号带节奏"阵营小团体。
- 学术依据:S2§3.2 explicit belief graph(Li 2023 prompt 显式信念状态增强多 agent 协作)+ §3.3 Suspicion-Agent 二阶 ToM(Guo 2023)。**caveat(诚实)**:机制有学术支撑,但"在狼人杀上提升对抗质量"无 primary source 实证,需 werewolf-mas 自身对局验证(本轮已验证可工作,质量提升待 task #24/#25 量化)。

### 3.6 方向 C:狼人白天话术协同(task #30 ✅,端到端验证)
- 代码:`actor.decide_wolf_caucus`(狼队白天党团会议,返回 target_seat+strategy+reasoning);`orchestrator._werewolf_day_caucus`(复用夜间 `_werewolf_deliberation` 私聊拓扑,白天发言前1次私聊,多数决聚共识,注入每个狼人私有记忆 `wolf_caucus`/`wolf_caucus_consensus` 事件);`_run_day` 开头调用。信息隔离:wolf_caucus 事件仅对 `wolf_entries` observe,好人看不到(harness 职责)。
- 设计(用户确认):**弱协同 + 仅白天发言前1次私聊**。狼人发言仍自主走 decide_speak LLM,harness 不写发言(守 no-fallback + agent 自决)。
- 验证(god 模式真实对局,day2 好人胜):0 决策失败;**狼人协同涌现**——4号(狼)6发言5次指控2号(预言家),5号(狼)8发言3次指控2号+3次指控1号(帮腔者),两狼集中火力推预言家+打击帮腔者,正是党团会议"统一推人目标+口径"产物;**平衡未破坏**:好人仍胜(day2),证明方向B态度网络成功识别狼人抱团(4号5号互相support+集体oppose 2号)抵消了狼人协同增强。
- 学术依据:AutoGen Swarm + S2§3.3 多 agent 协作。**caveat(诚实)**:原创设计无 benchmark 先例(deep-research caveats 第5点),本轮1局验证可工作+不破平衡,胜率分布待 task #24 多局统计确认 40-60%。

### 3.7 task #25:5维对局质量自动评分(task #25 ✅,端到端验证)
- 代码:`quality.py` QUALITY_PROMPT 加方向A/B/C 结构化对话关系评分要点(回应/指控/态度/狼人协同如何细化五维)+ digest 渲染补 reply_to/accuses/attitudes 元数据(赞/反/中);`orchestrator._emit` `_speech_log` 补采 reply_to/accuses/attitudes;新增 `_dialogue_metrics` 客观统计(reply_rate/accuse_rate/attitude_rate/support_edges/oppose_edges/wolf_coordination)注入 analysis。
- 双产出:**LLM 5维评审**(RI/SJ/DR/PS/CT + game_quality,主观)+ **对话量化指标**(A/B/C 客观统计,独立于 LLM)。后者用于跨对局对比 A/B/C 落地前后提升。
- 验证(真实对局,day1 狼人胜):game_quality=0.7;对话指标 reply_rate=0.35/accuse_rate=0.65/attitude_rate=1.0/support=12/oppose=21/wolf_coordination=4.5;**评审 LLM 明确用结构化证据**——总评"狼队2、3号成功利用5号信息劣势,精准带节奏,以高协同性获胜",6号 DR=0.8"识破2、3号狼队抱团指控节奏";0 决策失败。
- 学术依据:Beyond Survival(arXiv:2510.11389)WereAlign 五维评估。

### 3.8 三因子记忆检索(deep-research finding [6] ✅,单元验证)
- 代码:`memory.py` MemoryItem 加 `importance` 字段 + `_importance_for` 启发式打分(按 kind:claim/seer_action/death=0.9, vote/speech=0.5-0.7, phase_started=0.1;含矛盾关键词加权,免 LLM 调用);`recent_observations` 重写为三因子检索 `recency×0.1 + relevance×1.0 + importance×1.0`(Generative Agents 2304.03442)。recency 按天指数衰减(0.5/天);relevance=提及当前 top-3 高怀疑座位加权。
- 收益:跨天 claim 矛盾/查验结果等硬信号不被近期 speech 噪声淹没(纯 recency 会丢)。单元验证:20条 day2 噪声 speech 中,day1 的 claim 硬信号(importance=0.9)仍排第一被检索到。
- 学术依据:Generative Agents(Park et al. UIST 2023, arXiv:2304.03442)memory stream 三因子检索。**caveat**:原论文用 LLM 打 importance 1-10,此处用启发式免调用(成本低,精度略低,可后续换 LLM 打分)。

### 3.9 OSR 客观发言重写(Beyond Survival ✅,显式两段式)
- 代码:`schemas.py` Decision 加 `objective_summary` 字段;`prompts.py` VOTE_SCHEMA 拆为两段式(objective_summary=第1段客观摘要他人发言剥离情绪话术 + thought=第2段基于摘要推理投票);`actor._sanitize_vote` 解析;`orchestrator` vote_cast 事件带 objective_summary(可审计+前端展示)。
- 收益:原 vote_instruction 只在 prompt 里"要求"客观摘要(软约束),现结构化字段**强制 LLM 真的执行摘要**(硬约束,可审计)。防御 Beyond Survival 实证的"被说服成坏选择"——LLM 易把有说服力的话当字面指令,客观摘要剥离情绪后投票。
- 学术依据:Beyond Survival(arXiv:2510.11389)RR/OSR 干预。

### 3.10 task #24:多局真实对局对抗质量验证(task #24 ✅,6局统计)
- 跑6局真实 LLM 对局(`tests/multi_game_stats.py` 直跑 orchestrator 采集 analysis),统计胜率+对话指标+5维分。
- **胜率平衡**:village 4 / werewolves 2 → village 66.7% / werewolves 33.3%,**✅ 平衡**(方向C 狼人协同没让狼人胜率爆表,方向B 态度网络成功抵消)。6局全打满 day2,0 决策失败。
- **A/B/C 对话指标基线**:平均发言23/局;reply_rate 47%(方向A 近半真回应);accuse_rate 69%(方向A 近七成含指控);**attitude_rate 100%**(方向B 稳定);support 11.8 / oppose 28.8(对抗性强);wolf_coordination 2.25(方向C 协同工作但不过分)。
- **5维分基线**:game_quality 0.76;RI 0.63(最强) > SJ 0.59 > DR/PS/CT 0.54(欺骗/劝说/反事实偏弱,后续优化方向)。
- **结论**:A/B/C 三方向+三因子记忆+OSR 全部落地后,系统对抗质量良好(0.76)、平衡在工作、0 伪造。DR/PS/CT 偏弱是下一阶段优化目标。

### 3.11 task #31:DR/PS/CT 三维提升——欺骗结构化 + 识谎线索 + 权衡显式化(task #31 ✅,端到端验证 + 诚实统计)
- 代码:`schemas.py` Decision 加 `deception: str|None` 字段 + `_validate_deception` mode=before validator(归一化 omission/distortion/fabrication/misdirection/none,容忍中文遗漏/扭曲/捏造/误导);`prompts.py` SPEAK_SCHEMA 加 deception 字段(仅狼人填,好人填none)+ speak_instruction 加【识谎线索(DR,好人侧)】(点破 fabrication/distortion/omission/misdirection 具体手段+依据)+【反事实权衡(CT)】(显式"若投X则…若投Y则…");`actor._extract_deception` 解析;`orchestrator` speech 事件 + _speech_log 保留 deception 供 god/replay/analysis,`_dialogue_metrics` 加 wolf_deception_count + wolf_deception_dist;`quality.py` digest 渲染骗[遗漏/扭曲/捏造/误导] + QUALITY_PROMPT 加 DR/PS/CT 三维细化评分要点(识谎线索/带节奏效果/反事实对比)。**2026-07-05 安全修正**:live agent 的 `today_speeches` / PK 上下文已不再携带 `deception`,避免普通 agent 看到狼人自报欺骗策略(见 §3.19)。
- 学术依据:Werewolf Arena 4 类欺骗策略(omission/distortion/fabrication/misdirection)+ WOLF 欺骗分类基准 + Beyond Survival DR 维。
- **端到端验证(单局 god 模式)**:game_quality 0.8(↑0.76基线);wolf_declaration 18 条(16 misdirection + 2 distortion);**LLM 评审引用了识谎线索**("识破5号和6号'沉默可疑'的带节奏策略""识破合谋")——DR 指南让好人侧识谎可被评审识别;3号(好人)DR 0.8 / PS 0.8;0 决策失败。
- **6局统计(诚实)**:village 50% / werewolves 50%(平衡性↑,从 67/33 变 50/50——结构化欺骗让狼人更有竞争力,符合 §4.1 平衡担忧);wolf_deception_count 平均 7.2/局(欺骗策略激活);0 决策失败。**但 DR 0.53 / PS 0.54 / CT 0.47,相比 0.54 基线无明显提升**(2局狼人 day1 速胜 game_quality 0.5/0.6 拉低均值)。
- **诚实 caveat(铁律7)**:机制端到端工作(欺骗结构化流转、狼人主动选用、好人偶尔识破被评审捕获),但"6局统计上 DR/PS/CT 从 0.54 显著提升"**未得到证实**——5维 LLM 评审方差大、6局样本小。这是 deep-research caveats 第5点"方向 B/C 落地有效性无 primary source 实证"的直接体现。需更大样本(20+局)或换更稳定的评审(规则评审对照 LLM 评审)才能定论。当前结论:机制可用、欺骗维度已激活、平衡性改善,**5维分提升留待 task #24 扩样本验证**。
- **20局扩样本验证(诚实,推翻"提升"假设)**:DR 0.50±0.27(狼伪装0.59/好人识谎0.45)/ PS 0.48±0.28 / CT 0.44±0.25 / game_quality 0.61。**相比 0.54 基线非但没提升反而略降**——20局样本更大更可信,实证证明 task#31 的欺骗结构化+识谎线索提示**未带来5维分统计提升**。方差大(±0.25-0.28)说明5维LLM评审本身不稳定。**结论:机制可用但5维分提升假设被证伪(诚实无知铁律7)**。
- **新暴露的平衡问题(需关注)**:20局 village 30% / werewolves 70%,平衡向狼人偏移(6局50/50是样本运气)。结构化欺骗+党团协同让狼人变强,好人侧识谎线索提示不足以抵消。需在好人侧加强(如三因子记忆强化矛盾检测、OSR更激进、或削弱狼人党团协同度)。0 决策失败。

### 3.12 前端 A/B/C/DR 对话元数据可视化(✅,build 通过)
- 代码:`frontend/src/lib/types.ts` speech 事件加 reply_to/accuses/attitudes/deception,vote_cast 加 objective_summary,GameAnalysis 加 dialogue_metrics + DialogueMetrics 类型;`store.ts` LogEntry 加对应字段 + speech/vote_cast pushLog 透传;`SpeechFeed.tsx` 渲染【↩ 回应 X号】【指控 X号】【态度边(赞/反/中)】【骗:遗漏/扭曲/捏造/误导】chip;`GameStatusPanel.tsx` 新增 DialogueMetricsCard(8 指标网格 + 狼人欺骗策略分布);`SeatGrid.tsx` 圆桌白天叠加 SVG 态度网络边(support 绿/oppose 红带箭头);`global.css` 加 att-edge/chip--deception/sp-dm/att-line 样式。
- 收益:方向 A/B/C/DR 产出的结构化对抗数据现在**前端可见**——玩家/god 能直观看到谁反驳谁、谁指控谁、态度网络边(圆桌上实时绘制)、狼人用了哪种欺骗策略,而非只能读自由文本。这是"暴露对抗过程"(memory `feedback-verbose-agent-process`)在 UI 层的落地。
- 验证:`npm run build` 通过(tsc + vite,43 模块,189KB)。后端 speech/vote_cast 事件已带全字段,前端零 mock 直连。
- god 模式狼人 caucus 私聊展示已落地(memory `god-mode-caucus-visualization-2026-07-05`):orchestrator emit wolf_caucus/consensus 事件(信息隔离:仅 god/replay 收,play 不收保公平),前端渲染🐺党团提案/共识。

### 3.13 task #32:好人侧平衡补强(task #32 ✅,20局验证平衡修复)
- 20局暴露 werewolves 70% 偏强。两处修复(复用而非创造,守分层铁律):
  - **削弱狼人党团**:`orchestrator._run_day` 的 `_werewolf_day_caucus()` 改为仅 day1 举办(`if day <= 1`)。首日推人目标+口径一次性注入,后续天狼人独立发言更易暴露抱团(好人态度网络可识别)。
  - **强化好人识谎**:`prompts.vote_instruction` 加【识谎检查清单(好人侧,投票前逐项过)】——对照 4 类欺骗手段(omission/distortion/fabrication/misdirection)逐项排查,谁中招谁更可能狼;并提示"2+人无理由抱团指控同一目标,这俩都高度可疑"。纯 prompt 改动,不动 harness 逻辑。
- **20局验证(v2)**:village 50% / werewolves 50%(v1 是 30%/70%)→ **平衡修复成功**。0 决策失败。对话指标:reply_rate 39%(↑v1 27%)、wolf_coordination 1.80(↓v1,党团仅day1后抱团减弱)、wolf_deception 7.3/局。
- **5维分(诚实)**:game_quality 0.61;DR 0.46(狼伪装0.50/好人识谎0.44)/ PS 0.46 / CT 0.39。**平衡修复成功但5维分绝对值仍偏低**——与 task#31 一致,5维 LLM 评审方差大(±0.22-0.25),非机制问题。绝对值偏低是评审稳定性问题,跨配置对比看相对差异更有意义。
- **结论**:好人侧平衡补强成功(30/70→50/50),守住对抗公平性。平衡担忧解除。

### 3.14 前端跨局趋势 + 思考流 verbose 展示优化(✅,build/test 通过)
- 代码:`frontend/src/lib/trends.ts` 新增真实 `analysis` 事件的轻量趋势记录(`winner/days/game_quality/RI/SJ/DR/PS/CT/dialogue_metrics`),写入浏览器 `localStorage`,按 `roomId` 去重,最多保留 40 局;`GameStatusPanel.tsx` 新增跨局趋势卡(最近 12 局胜率/平均质量/平均天数/五维均值/SVG sparkline/对话指标均值/结果点列/清空本地历史);`GameView.tsx` 传入 `roomId`;`global.css` 补趋势样式。
- 思考流优化:`ChatRoom.tsx` 的 ThinkingTimeline 新增动作筛选 + 文本检索;折叠态只显示摘要,展开态分成"摘要"与"完整推理",避免 verbose reasoning 淹没时间线;保留原有 seat/阵营筛选和 suspicion_top 展示。
- 纯前端展示层改动,不改后端协议、不改 harness/agent 决策逻辑,趋势数据来自真实赛后 `analysis` 事件,无 mock。
- 顺手修复 `global.css` 里圆桌态度网络附近残留的非法 `pointer-events` CSS 片段,消除 Vite CSS minify warning。
- 验证:`npm run build` 通过(tsc + vite,44 modules,无 CSS warning);`PYTHONPATH=. pytest -q` 通过(38 passed,3 skipped,1 warning)。首次直接 `pytest -q` 因当前 shell 未带 `PYTHONPATH=.` 导致 `ModuleNotFoundError: src`,非代码回归。

### 3.15 全局项目本质审计 + 本地绑定安全回归(✅)
- 按用户要求重新全局梳理项目本质:文档(`ARCHITECTURE.md`/研究记录/CLAUDE.md)→ 后端真实对局链路(`server/room_manager/orchestrator/rules/actor/router`)→ agent 观察/记忆/prompt/Decision → 前端 WS 事件归约与呈现 → 测试/多局统计。结论:项目本质是**用狼人杀作为可观察环境,呈现并评测真实 LLM 多 Agent 对抗社会**;狼人杀规则和 UI 都服务于观察 agent 间欺骗、识谎、信念网络、协同与反制。
- 发现安全不一致:工作宪法要求后端本地跑(127.0.0.1),但 `.env`/`.env.example`/`src/config.py` 仍默认 `0.0.0.0`。已修复为 `127.0.0.1`,不触碰任何凭据。
- 验证:`npm run build` 通过;`PYTHONPATH=. pytest -q` 通过(38 passed,3 skipped,1 warning);本地服务已在 `127.0.0.1:8000` 可访问,`/api/providers` 与 `/api/config` 正常。

### 3.16 第四轮并行调研:对抗质量瓶颈定位 + 评测路线重构(✅,research + code audit)
- 触发:用户指出需要围绕"多 agent 对抗、欺骗、多 agent 社会"继续联网调研并可启动子代理。并行 3 路:① 社交推理/狼人杀 primary sources;② LLM-as-judge 稳定性;③ 本地代码瓶颈审计。
- **核心结论**:当前瓶颈不是"再堆 prompt",而是两条证据链不足:
  1. **真实信息链断点**:投票阶段没拿到完整当天发言(`_run_voting(... today_speeches=[])`),公开 speech 也未写入所有存活 agent 的长期 memory,导致 OSR/DR/PS/CT 的输入基础不稳。
  2. **评测链不稳**:单次 LLM 绝对 5 维评分混合了 judge 偏好、长度、位置、rubric 理解、局势难度,不能继续当"统计证明主指标"。保留 WereAlign 5维作为解释层,主指标改为确定性轨迹指标 + 校准 judge + paired/ABBA + 置信区间。
- **社交推理 / 多 agent 社会 primary sources**:
  - Werewolf Arena(arXiv:2407.13943):动态发言权/bidding 是 LLM 社交推理评测核心;werewolf-mas 的 bid/reply/accuse 路线正确,但要把元数据沉淀成证据图。
  - DVM(arXiv:2501.06695):Predictor/Decider/Discussor + 目标胜率控制;启示是平衡应调 agent 能力预算(推理深度/候选发言/狼队 caucus 预算),不只调规则。
  - MultiMind(arXiv:2504.18039):ToM suspicion matrix + 搜索沟通策略;启示是从 `attitudes` 升级到稀疏二阶 ToM(`我认为X怀疑Y`)。
  - Social Deduction MARL(arXiv:2502.06060):speaking reward = 发言后他人正确信念提升;启示是 PS 应用 belief/vote/attitude shift 衡量,而非 judge 读一段话说"有说服力"。
  - WOLF(arXiv:2512.09187):逐句欺骗审计(omission/distortion/fabrication/misdirection)+ speaker 自评 + peer 检测 + 纵向怀疑;启示是 `deception` 字段要升级为审计日志,当前只证明字段存在。
  - CSP4SDG(arXiv:2511.06175)+ GRAIL/Graph-Informed Language Models(ACL 2026):结构化证据图/约束图维护 RolePosterior,LLM 负责语言理解与表达;启示是中期做 EvidenceGraph + private RolePosterior,不要把隐藏身份概率推理全塞给长 CoT。
  - Stackelberg Speaker(ACL 2026):发言是 leader-follower 说服动作;启示是后续可做候选发言 rerank,以期望回应/信念转移作为目标。
- **LLM-as-judge 稳定性 sources**:
  - MT-Bench/Chatbot Arena(arXiv:2306.05685):LLM judge 可用但有 position/verbosity/self-enhancement bias。
  - G-Eval(arXiv:2303.16634):CoT+form-filling 可提升 reference-free NLG 评分,但也暴露 LLM 文体偏好。
  - Large Language Models are not Fair Evaluators(arXiv:2305.17926):位置顺序可显著改变评测,需要 balanced position/multiple evidence/calibration。
  - FLASK(arXiv:2307.10928):粗粒度总分应拆为细粒度 skill-set + 证据。
  - Length-Controlled AlpacaEval(arXiv:2404.04475):评审需控制长度偏差。
  - Pairwise or Pointwise?(arXiv:2504.14716):pairwise 更适合版本回归但不是银弹,需 AB/BA、tie/uncertain、置信区间。
- **最低风险实施路线(先做 Phase 0/1)**:
  1. `_run_voting(today_speeches=...)`:投票时传入当天公开发言;PK 时传入主讨论 + PK 发言。
  2. 公开 speech 写入所有存活 agent memory(只写公开字段,绝不写 role truth/reasoning/wolf caucus)。
  3. 修 `quality.py` thinking digest:当前"最近2条"实际取每人最早2条,改取最近2条。
  4. 修 `wolf_coordination`:改为 day-level/狼座位覆盖率,clamp 到 [0,1],避免重复指控让指标>1。
  5. 新增 `objective_metrics`:赛后用真值计算 vote accuracy、accuse precision、attitude-vote consistency、accuse-to-vote conversion、CT marker rate 等可复算指标;只进 analysis,不进入 live prompt。
  6. 后续再做 judge v2:单维 evidence-based scoring + judge temperature=0 + AB/BA pairwise + bootstrap CI。
- **不该做**:不要让 harness 自动写发言/反驳/投票;不要把角色真值、god/replay、wolf caucus 暴露给普通 agent;不要用 localStorage 趋势或 6局均值宣称机制显著提升。

### 3.17 Phase 0/1 实施:真实证据链修复 + 确定性轨迹指标(✅,build/test 通过)
- 目标:落实 §3.16 的最低风险路线,先修"agent 看不到足够公开证据"与"评测只靠单次 LLM 绝对分"两条瓶颈,不改规则、不让 harness 代替 agent 决策。
- 后端证据链:
  - `orchestrator._run_day` 讨论结束后调用 `_run_voting(today_speeches=today_speeches)`,投票 prompt 能拿到完整当天公开发言。
  - PK 链路传 `day_speeches + pk_speeches`,PK 发言和 PK 重投票都保留日间讨论证据。
  - 新增 `_record_public_speech_memory`:每条公开 speech 写入所有存活 agent memory,只包含 seat/text/reply_to/accuses/attitudes,**不写 role truth/reasoning/deception/wolf caucus/claim 元数据**。公开 claim 仍走原有 `record_claim` 路径。
  - 删除 `_emit(vote_cast)` 对 `_vote_log` 的二次追加;现在 `_run_voting` 是唯一 vote analysis log 来源,保留 `objective_summary` 和私有 `reasoning` 供赛后分析,避免重复票污染 quality prompt。
- 后端评测链:
  - 修 `_dialogue_metrics.wolf_coordination`:改为 day-level 狼座位覆盖率,clamp 到 `[0,1]`,不再因同一只狼重复指控而出现 `4.5` 这类无界值。
  - 新增 `_objective_metrics` 并注入 `analysis.objective_metrics`:赛后用真值复算 `vote_accuracy_good/wolf`、`accuse_precision_good/wolf`、`attitude_vote_consistency`、`accuse_to_vote_conversion`、`osr_summary_rate`、`ct_marker_rate`、`seer_claim_follow_rate` 等。**仅赛后输出,绝不进入 live prompt**。
  - `score_game_quality` 调用侧把 judge temperature 固定为 `0.0`,降低同一轨迹重复评分方差;五维 WereAlign 仍保留为解释层,不再单独当主证明。
  - `quality.py` 思考摘要 bug 修复:每个 seat 真正取最近 2 条 thinking summary,而不是最早 2 条。
- 前端展示:
  - `types.ts` 新增 `ObjectiveMetrics` 与 `GameAnalysis.objective_metrics`。
  - `GameStatusPanel.tsx` 新增"客观轨迹指标(赛后真值)"卡,展示好票命中、狼票命中、好人/狼人指控精度、态度投票一致、指控转投、OSR 摘要率、CT 标记率。
  - `trends.ts` 把 objective metrics 写入本地跨局趋势,趋势卡追加好票/狼票/CT 均值。
- 测试:
  - `tests/test_orchestrator.py` 新增回归:vote log 与 vote_cast 数一致且不重复;public speech memory 不泄漏 deception/reasoning/claim;投票决策收到 today_speeches;analysis 带 objective_metrics 且 wolf_coordination 在 `[0,1]`。
  - 验证:`PYTHONPATH=. pytest -q` → `42 passed, 3 skipped`;`npm run build` → 通过(tsc + vite,44 modules)。
  - 真实 LLM smoke:`PYTHONPATH=. python tests/smoke_e2e.py` 跑完 1 局,好人 day2 胜;真实模型调用 50 次,成功 50,失败 0,重试 0,0 `agent_decision_failed`。对局中 1号预言家查杀6号、2号狼人帮腔反打、3/4号基于公开发言/投票链在 day2 锁定2号,验证证据链能支撑真实 agent 识谎。
- 诚实 caveat:本轮修的是证据链与评测链基础设施,不是宣称 DR/PS/CT 已显著提升。显著性仍需后续多局 JSONL + paired/ABBA judge v2 + bootstrap CI 验证。

### 3.18 成型化收口:事件协议去重 + 评测工具可复算化 + 复盘面板补齐(✅,build/test 通过)
- 触发:继续朝"成型项目"推进。§3.17 真实 smoke 暴露 `game_ended` 会广播两次;§4.2 要求多局统计输出 JSONL 和不确定性;前端 ReplayPanel 尚未展示 `objective_metrics`。
- 后端协议:
  - `GameOrchestratorV2` 新增 `_game_ended_emitted` + `_emit_game_ended()`,保证 `game_ended` 只广播一次;`analysis` 仍在之后作为最后复盘事件到达。
  - 新增测试 `test_game_ended_emitted_once_before_analysis`,锁住"结束事件单发且早于 analysis"。
- 评测工具:
  - 重写 `tests/multi_game_stats.py` 为真实 LLM 多局统计 CLI:输出胜率 Wilson 95% CI、dialogue/objective metrics bootstrap 95% CI、WereAlign 五维 bootstrap 95% CI,并默认写每局 JSONL(`logs/multi_game_stats.jsonl`)。
  - 纳入 `objective_metrics` 汇总:好票命中、狼票命中、好人/狼人指控精度、态度投票一致、指控转投、OSR 摘要率、CT 标记率、预言家查杀跟票率。
  - 每局摘要记录 `game_ended_events`,用于发现协议重复/缺失;统计层报告异常局数。
  - 新增 `tests/test_multi_game_stats.py`(无 LLM 单测),覆盖 Wilson CI、bootstrap CI、数值过滤、空结果、单局结果、objective metrics 过滤。修 `as_float(True/False)` 被误当 `1.0/0.0` 的边界问题。
- 前端复盘:
  - `ReplayPanel.tsx` 新增"客观轨迹指标"紧凑区块,显示 `dialogue_metrics.wolf_coordination` 和核心 `objective_metrics`。
  - `global.css` 新增 replay metric 样式,沿用现有暗色卡片与红/绿语义色。
- 并行执行:
  - 启动 2 个子代理并行:一个负责统计脚本单测,一个负责前端 ReplayPanel 补齐。主线负责后端协议与总体验证。
- 验证:
  - `PYTHONPATH=. pytest -q` → `58 passed, 3 skipped`。
  - `npm run build` → 通过(tsc + vite,44 modules)。
  - `python -m compileall -q tests/multi_game_stats.py src/game/orchestrator.py` 与 `PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10` 通过;dry-run 临时 JSONL 已删除。
  - 真实 LLM smoke:`PYTHONPATH=. python tests/smoke_e2e.py` 跑完 1 局,好人 day2 胜;真实模型调用 52 次,成功 52,失败 0,0 `agent_decision_failed`;`game_ended` 只广播/打印 1 次。过程中出现 2 次 JSON 解析失败警告,均由 actor 深度重试恢复,未触发 fallback 或伪造决策。
- 诚实 caveat:多局统计 CLI 已具备 JSONL/CI 能力,但本轮未跑新的 N>1 真实统计样本;显著性结论仍需后续实际执行多局批量验证。

### 3.19 Phase 2 最小 EvidenceGraph/RolePosterior + 第五轮联网调研(✅,test 通过)
- 触发:用户要求继续全局理解项目本质,并围绕"多 agent 对抗、欺骗、多 agent 社会"联网搜索、可开子代理并行。本轮并行 2 路子代理:① 代码隐私边界/测试审查;② 最新研究缺口扫描。主线完成最小 EvidenceGraph/RolePosterior 实装。
- **联网核验后的研究增量**:
  - Werewolf Arena(arXiv:2407.13943, https://arxiv.org/abs/2407.13943) 支持 werewolf-mas 的动态发言权/bid 路线:社会推理评测不只是"谁答对",而是 agent 在隐藏身份、欺骗、说服中如何选择何时发言。
  - CSP4SDG(arXiv:2511.06175, https://arxiv.org/abs/2511.06175)+ GRAIL/Bayesian Social Deduction with Graph-Informed Language Models(ACL 2026, https://camp-lab-purdue.github.io/bayesian-social-deduction/):身份推理应由可解释约束/图后验承担,LLM 负责语言互动;硬约束 prune + 软证据加权比把概率推理全部压给长 CoT 更稳。
  - OpenDeception(arXiv:2504.13707, https://arxiv.org/abs/2504.13707)+ Intentional Deception(arXiv:2603.07848, https://arxiv.org/abs/2603.07848):欺骗评测不能只信 speaker 自报 `deception`;成功欺骗往往是 misdirection/strategic framing,且对不同 listener 画像影响不同。后续要做外部 deception audit + listener posterior shift。
  - 多 agent debate/opinion dynamics:Can LLM Agents Really Debate?(arXiv:2511.07784, https://arxiv.org/abs/2511.07784)+ DEBATE(arXiv:2510.25110, https://arxiv.org/abs/2510.25110) 指出多 agent 辩论容易出现多数压力、过早收敛和从众;werewolf-mas 后续 PS/DR 指标应记录 belief/posterior 的逐轮变化,而不只靠 judge 读最终文本。
  - Belief update 校准:Are LLM Belief Updates Consistent with Bayes' Theorem?(arXiv:2507.17951, https://arxiv.org/abs/2507.17951) 提醒不能把 LLM `suspicion` 当校准后验;RolePosterior 应保存 evidence→delta→posterior,再用 Brier/log loss/校准曲线与真实赛后结果对照。
  - AgentSociety/Agent-Kernel/Agentopia(arXiv:2502.08691 / 2512.01610 / 2606.07513) 显示多 agent 社会模拟正走向事件化、可复现、可干预的 substrate;werewolf-mas 中期应把 EvidenceGraph 做成 event-sourced social state,支持 replay、counterfactual intervention、策略版本分组对比。
- **本轮实现(最小版,守分层铁律)**:
  - 新增 `src/agent/evidence.py`:纯确定性 helper,从单个 `AgentObservation` 的可见信息生成 `evidence_graph`:公开 `claims`、`claim_conflicts`、`attitude_edges`、`vote_edges`、`death_events`、viewer 自己可见的 `private_results`、以及软 `role_posterior/top_suspects`。不调用 LLM,不读 `GameState.players[*].role`,不改 agent 决策。
  - `AgentObservation` 新增 `evidence_graph` 字段;`information.build_observation()` 创建观察后立即构图;`attach_today_speeches()` 注入当天发言后重建图。
  - 新增 `information._sanitize_public_speech_for_agent()`:live agent 的 `today_speeches` 只保留 `seat/name/text/bid/reply_to/accuses/attitudes/claim/day/pk`;显式剔除 `deception/reasoning/wolf_caucus/role/team/teammates` 等隐藏或赛后字段。公开 `claim` 只保留 `role/checked_seat/result`。
  - `orchestrator._run_day()`/`_run_pk()` 的内部 `today_speeches` 现在携带公开 `claim` 供后续 agent 和 EvidenceGraph 使用,但不再携带 `deception`。`speech` emit payload 仍保留 `deception` 供 god/replay/analysis 与前端研究展示;普通 live prompt 不消费它。
  - `prompts.render_observation()` 新增"公开证据图 / 角色后验(非真值)"紧凑块,明确说明只是可见线索整理,不是上帝身份真相。渲染 top suspects 时引用最多 2 条证据,防 prompt 膨胀。
- **测试**:
  - 新增 `tests/test_evidence_graph.py`:覆盖公开 claim/对跳/查验冲突、attitude/vote 边、预言家私有查验只进入本人观察、posterior 不从真实身份初始化、`attach_today_speeches` 隐藏字段净化、prompt 渲染不泄漏隐藏字段。
  - `tests/test_orchestrator.py` 补 `test_live_today_speeches_do_not_leak_hidden_metadata`,锁住真实编排链路里后续 actor 收到的 `today_speeches` 不含 `deception/reasoning`,但保留公开 claim。
  - 验证:`PYTHONPATH=. pytest -q` → `65 passed, 3 skipped`;`python -m compileall -q ...` → 通过;`cd frontend && npm run build` → 通过。
  - 真实 LLM smoke:`PYTHONPATH=. python tests/smoke_e2e.py` 跑完 1 局,村民 day2 胜;真实模型调用 48 次,成功 48,失败 0,重试 3(2 次 stream 网络中断 + 1 次空 JSON 内容均由深度重试恢复),0 `agent_decision_failed`;`game_ended` 单发。
- 诚实 caveat:本轮是 **minimal visible evidence graph**,不是 CSP4SDG/GRAIL 级枚举合法身份世界;`role_posterior` 仍是启发式软分,用于让 agent prompt 有证据索引。下一步才应做 event-sourced graph + calibrated Bayesian posterior + posterior-delta metrics。

### 3.20 Posterior 轨迹指标闭环:belief shift / herding / top suspect accuracy(✅,build/test 通过)
- 触发:§4.2 已明确"posterior 轨迹指标"是下一步;§3.19 只有 live prompt 里的最小可见 EvidenceGraph,尚未把后验变化沉淀为可复算评测。继续朝"成型项目"推进,把 belief/posterior shift 从研究路线变成后端 analysis + 多局统计 + 前端展示的闭环。
- 后端:
  - `GameOrchestratorV2` 新增 `_posterior_log`:每次公开 `speech` / `pk_speech` / `vote_cast` 后,为每个存活 agent 重建其可见 `AgentObservation` + sanitized `today_speeches`,提取 compact `role_posterior/top_suspects` 快照。**Analysis-only**,不写 memory、不改 prompt、不影响 agent 决策。
  - 新增 `_posterior_metrics()` 并注入 `analysis.posterior_metrics`: `snapshot_count`、`speech_snapshot_count`、`avg_speech_posterior_shift`、`good_final_wolf_suspicion_gap`、`good_final_top_suspect_accuracy`、`herding_index`。
  - 新增 `analysis.posterior_trace`:最多保留最近 240 条紧凑后验快照,只含 day/phase/trigger/source/viewer/posterior/top_suspects,不含 `deception/reasoning` 等 hidden 字段。
  - 指标含义:
    - `avg_speech_posterior_shift`:同一 viewer 连续发言后验的平均 L1 位移,衡量发言实际撬动了多少信念。
    - `good_final_wolf_suspicion_gap`:赛后好人视角最终"狼座位平均嫌疑 - 非自己好人平均嫌疑",越高说明好人后验更会把狼排前。
    - `good_final_top_suspect_accuracy`:好人最终 top suspect 是否为真狼的比例。
    - `herding_index`:同一天好人 top suspect 集中到同一目标的比例,用于观察从众/共识收敛;它本身非好坏,要结合 accuracy 判断是正确共识还是错误抱团。
- 多局统计:
  - `tests/multi_game_stats.py` 汇总 `posterior_metrics`,输出 bootstrap CI;每局摘要行增加 `belief_gap/herding`。
  - `tests/test_multi_game_stats.py` 覆盖空结果、单局、多局 posterior metrics 打印与数值过滤。
- 前端:
  - `frontend/src/lib/types.ts` 新增 `PosteriorMetrics` / `PosteriorTraceEntry`,`GameAnalysis` 增加 `posterior_metrics/posterior_trace`。
  - `GameStatusPanel.tsx` 新增"后验轨迹指标(EvidenceGraph)"卡,展示快照、信念位移、狼嫌疑差、top 命中、从众指数;跨局趋势追加 posterior 指标均值。
  - `ReplayPanel.tsx` 的"客观轨迹指标"追加信念位移、狼嫌疑差、top 命中、从众指数和后验快照数。
  - `trends.ts` 将 posterior metrics 写入 localStorage 趋势记录。
- 测试:
  - `tests/test_orchestrator.py` 新增 `test_analysis_includes_posterior_trace_and_metrics`,锁住 analysis 输出、指标范围、speech/vote 触发快照、trace 不含 `deception/reasoning`。
  - 验证:`PYTHONPATH=. pytest -q tests/test_orchestrator.py tests/test_evidence_graph.py tests/test_multi_game_stats.py` → `33 passed`;`PYTHONPATH=. pytest -q` → `67 passed, 3 skipped`;`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10` → 通过且临时 JSONL 已删除;`cd frontend && npm run build` → 通过。
  - 真实 LLM smoke:`PYTHONPATH=. python tests/smoke_e2e.py` 跑完 1 局,村民 day2 胜;真实模型调用 50 次,成功 50,失败 0,重试 3(stream 网络中断均由深度重试恢复),0 `agent_decision_failed`;对局中 3号预言家查杀6号、2/6号狼人用"跳太急"反打,1/4号基于证据链识别带节奏并 day2 投出2号,验证 posterior trace analysis 不破坏真实对局链路。
- 诚实 caveat:这仍然是启发式 posterior 的轨迹分析,不是 calibrated Bayesian posterior。它的价值是把 persuasion/deception 的可观察影响落到可复算曲线上;后续才应做 evidence_id→likelihood_delta→posterior 的校准模型与 Brier/log loss。

### 3.21 Posterior 校准指标:Brier / log loss / ECE(✅,build/test 通过)
- 触发:§3.20 已有后验轨迹,但只有 belief shift / herding / top suspect accuracy,还不能回答"这些嫌疑概率是否校准"。本轮在不改变 live agent 决策的前提下,把最小校准评测接入后端 analysis、多局统计和前端复盘。
- 联网核验后的研究依据:
  - Werewolf Arena(arXiv:2407.13943, https://arxiv.org/abs/2407.13943) 明确把狼人杀作为 deception/deduction/persuasion 下的 LLM social deduction benchmark,并使用动态 bidding 发言;werewolf-mas 的 bid/reply/posterior trace 路线吻合。
  - Are LLM Belief Updates Consistent with Bayes' Theorem?(arXiv:2507.17951, https://arxiv.org/abs/2507.17951) 说明 LLM belief update 需要显式一致性/校准评测,不能直接把模型输出的信念当作可靠概率。
  - CSP4SDG(arXiv:2511.06175, https://arxiv.org/abs/2511.06175) 与 GRAIL/Bayesian Social Deduction with Graph-Informed Language Models(ACL 2026, https://camp-lab-purdue.github.io/bayesian-social-deduction/) 支持将隐藏身份推理外置为约束/图后验,LLM 负责语言理解与互动。
  - MultiMind(arXiv:2504.18039, https://arxiv.org/abs/2504.18039) 强调 social deduction agent 需要维护他人 suspicion/ToM 状态;OpenDeception(arXiv:2504.13707, https://arxiv.org/abs/2504.13707) 提醒 deception/trust 要从 speaker 与 listener 两侧评测,不能只信自报。
- 后端:
  - `src/game/orchestrator.py` 的 `_posterior_metrics()` 新增 `final_brier_score`、`final_log_loss`、`good_final_brier_score`、`good_final_log_loss`、`calibration_ece`、`calibration_bins`。
  - `final_*` 基于所有 viewer 的最终快照;`good_final_*` 只基于好人 viewer。每条 record 是"viewer 对某 seat 是狼的概率估计" vs 赛后真值标签。真值仅在赛后 analysis 使用,绝不进入 live prompt/memory。
  - `calibration_bins` 默认 5 桶,每桶输出 `range/count/avg_prediction/wolf_rate`;`calibration_ece` 为加权 expected calibration error。空样本返回 `None`/空数组,不伪造数值。
- 多局统计:
  - `tests/multi_game_stats.py` 将校准指标纳入 `POSTERIOR_KEYS`,输出 bootstrap 95% CI;每局摘要行增加 `brier=...`,便于批量扫描后验校准退化。
  - `tests/test_multi_game_stats.py` 覆盖单局与多局 posterior calibration 输出,并确认布尔/非法值不会污染数值汇总。
- 前端:
  - `frontend/src/lib/types.ts` 扩展 `PosteriorMetrics` 与 `CalibrationBin`,显式兼容后端 `range/avg_prediction/wolf_rate/count`。
  - `GameStatusPanel.tsx` 后验卡展示 Brier、LogLoss、好人 Brier、好人 LogLoss、ECE 与校准分桶;分桶显示估计概率、实际狼率和样本数。
  - `ReplayPanel.tsx` 在复盘指标区展示同一组校准指标;`trends.ts` 将校准指标写入 localStorage 趋势历史,用于跨局观察。
- 测试:
  - 窄测:`PYTHONPATH=. pytest -q tests/test_orchestrator.py tests/test_multi_game_stats.py` → `27 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `67 passed, 3 skipped`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10` → 通过,临时 JSONL 已删除。
  - 真实 LLM smoke:`PYTHONPATH=. python tests/smoke_e2e.py` → 好人 day2 胜;真实模型调用 51 次,成功 51,失败 0,重试 4(空 JSON/stream 中断均由深度重试恢复),进程退出码 0。
  - 凭据扫描:按用户提供密钥的唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- 诚实 caveat:这是对 §3.20 启发式 posterior 的赛后校准评测,**不是**完整 calibrated Bayesian posterior。下一步仍应做 event-sourced evidence graph + evidence_id→likelihood_delta→posterior,并分 viewer/role/phase 报告;`herding_index` 必须和 top accuracy/Brier/ECE 联合解释,高从众可能是正确共识,也可能是错误抱团。

### 3.22 Evidence provenance v1:evidence_id → posterior_delta(✅,build/test 通过)
- 触发:§3.21 已有 Brier/log loss/ECE,但校准对象仍是启发式 posterior 的最终数值,缺少"哪条可见证据导致哪次嫌疑变化"的可审计链。按 §4.2 路线先做最小 provenance,为后续 likelihood_delta / legal-world posterior 铺底。
- 后端:
  - `src/agent/evidence.py` 为可见证据生成稳定 `evidence_id`:公开 claim、claim_conflict、attitude/accuse edge、vote edge、death event、viewer 自己可见的 private seer result 都有可追溯 ID。
  - `role_posterior[*]` 新增 `posterior_deltas`:每次启发式分数变化记录 `target_seat/delta/before/after/reason/evidence_id/source_type`。原 `evidence` 文本保留,用于 prompt 的紧凑解释。
  - `evidence_graph.posterior_deltas` 汇总所有 seat 的 delta,便于赛后审计。仍只使用当前 viewer 可见信息,不读真实身份、不读 god/replay、不读 `deception/reasoning/wolf_caucus`。
  - `GameOrchestratorV2._record_posterior_snapshot()` 将每个 viewer 的 compact `posterior_deltas` 写入 `analysis.posterior_trace`,最多保留每快照前 40 条,用于复盘查看"后验为何动"。该 hook 仍是 analysis-only,不写 memory、不改 prompt、不影响 live 决策。
- 前端:
  - `frontend/src/lib/types.ts` 为 `PosteriorTraceEntry` 增加可选 `posterior_deltas`,并新增 `PosteriorDelta` 类型。当前 UI 不强行展开逐条 delta,先保证 replay/analysis 协议类型完整。
- 测试:
  - `tests/test_evidence_graph.py` 覆盖 claim/conflict/attitude/vote/private result 的 `evidence_id`,以及 posterior delta 的 `target_seat/before/after/evidence_id`。
  - `tests/test_orchestrator.py` 覆盖 `posterior_trace[*].posterior_deltas` 存在、含 `evidence_id/delta/target_seat`,并继续确认 trace 不含 `deception/reasoning`。
  - 窄测:`PYTHONPATH=. pytest -q tests/test_evidence_graph.py tests/test_orchestrator.py` → `17 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `67 passed, 3 skipped`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10` → 通过,临时 JSONL 已删除。
  - 真实 LLM smoke:`PYTHONPATH=. python tests/smoke_e2e.py` → 好人 day2 胜;真实模型调用 49 次,成功 49,失败 0,重试 0,进程退出码 0。
  - 凭据扫描:按用户提供密钥的唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- 诚实 caveat:这只是 provenance layer,不是完整 likelihood model。`delta` 仍是启发式权重,没有枚举合法身份世界,也没有学习/校准 likelihood。下一步应把这些 `evidence_id` 升级为 event-sourced edge,每条边带 `visibility/provenance/day/phase/source/confidence`,再从 heuristic delta 迁移到 constrained posterior。

### 3.23 Event-sourced EvidenceItems v1(✅,build/test 通过)
- 触发:§3.22 已有 `evidence_id→posterior_delta`,但证据仍分散在 `claims/attitude_edges/vote_edges/private_results` 等字段上。若要继续做 CSP4SDG/GRAIL 式合法身份世界约束,需要统一的 event-sourced evidence substrate。
- 后端:
  - `src/agent/evidence.py` 新增 `evidence_graph.evidence_items`:把 claim、claim_conflict、attitude/accuse、vote、death、private seer result 规范成统一证据项。
  - 每条 `evidence_item` 至少包含 `evidence_id/type/visibility/provenance/confidence`,可选 `day/phase/source_seat/target_seat/payload`。其中 `visibility` 区分 public/private,`provenance` 区分 `today_speech/public_event/private_event/derived_*`。
  - `posterior_deltas[*].evidence_id` 现在必须能在同一 graph 的 `evidence_items` 中解析;对 attitude/vote cluster 这类派生 delta,自动生成 `derived_posterior_delta` 证据项。
  - `GameOrchestratorV2._record_posterior_snapshot()` 将 compact `evidence_items` 写入 `analysis.posterior_trace`,因此赛后每个 viewer 快照都有"证据项 → delta → posterior"三段链。
  - 信息隔离不变:private seer result 只出现在对应 viewer 的 graph;普通 viewer 的 `evidence_items` 全为 public;hidden `deception/reasoning/wolf_caucus` 不进入 evidence item。
- 前端:
  - `frontend/src/lib/types.ts` 新增 `EvidenceItem` 类型,`PosteriorTraceEntry` 增加可选 `evidence_items`。当前 UI 暂不展开逐项证据,但 replay 数据协议已准备好。
- 测试:
  - `tests/test_evidence_graph.py` 覆盖统一证据项 schema、delta id 可解析、public/private visibility 边界、private seer result 只进本人 graph。
  - `tests/test_orchestrator.py` 覆盖 `posterior_trace[*].evidence_items` 存在,并且每个 snapshot 的 `posterior_deltas[*].evidence_id` 都能在同 snapshot 的 `evidence_items` 里找到。
  - 窄测:`PYTHONPATH=. pytest -q tests/test_evidence_graph.py tests/test_orchestrator.py` → `17 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `67 passed, 3 skipped`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10` → 通过,临时 JSONL 已删除。
  - 真实 LLM smoke:`PYTHONPATH=. python tests/smoke_e2e.py` → 好人 day2 胜;真实模型调用 50 次,成功 50,失败 0,重试 0,进程退出码 0。
  - 凭据扫描:按用户提供密钥的唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- 诚实 caveat:这是 event-sourced substrate 的最小版,不是完整约束求解。`confidence` 是来源置信度/结构强度的启发式标记,不是学习过的 likelihood;当前 posterior 仍是线性启发式 delta。下一步应基于 `evidence_items` 枚举/约束合法身份世界,并把 `delta` 拆成可校准 likelihood contribution。

### 3.24 Legal Team Worlds v1 + constrained posterior(✅,narrow test 通过)
- 触发:§3.23 已有统一 `evidence_items`,但 posterior 仍未显式受"公开牌组狼数 + viewer 已知事实"约束。按 §4.2 路线先做最小合法身份世界枚举,只枚举 wolf-seat team worlds,不伪装成完整角色 assignment。
- 后端:
  - `AgentObservation` 新增 `role_counts`:公开牌组分布(每种角色数量),不含任何座位身份。`information.build_observation()` 从已发牌组的角色数量生成该分布;这只暴露 deck composition,不暴露 `state.players[*].role` 的座位映射。
  - `src/agent/evidence.py` 新增 `legal_worlds`:基于 `role_counts["werewolf"]`、viewer 自己身份、狼人队友(仅狼人可见)、viewer 自己的 private seer result 枚举合法 wolf-seat worlds。
  - `legal_worlds` 输出 `wolf_count/known_wolves/known_villagers/world_count/is_contradictory/is_truncated/worlds`。6人默认局普通村民视角为 C(5,2)=10 个 world;狼人知道队友后为 1 个 world;预言家查到一狼后 world 数按约束缩小。
  - `role_posterior[*]` 新增 `constrained_werewolf_suspicion`:用当前启发式 suspicion 作为 soft likelihood,在合法 worlds 上归一化得到 constrained marginal。原 `werewolf_suspicion` 不替换,避免突然改变 live prompt/发言策略。
  - `GameOrchestratorV2._record_posterior_snapshot()` 将 `legal_worlds` 和 `constrained_posterior` 写入 `analysis.posterior_trace`,为后续 constrained Brier/ECE 对照做准备。
- 前端:
  - `frontend/src/lib/types.ts` 新增 `LegalWorlds`,并给 `PosteriorTraceEntry` 增加 `legal_worlds` / `constrained_posterior` 可选字段。UI 暂不展开,先保证协议可消费。
- 测试:
  - `tests/test_evidence_graph.py` 覆盖:村民只知道自己不是狼时 world_count=10 且其他座位 constrained marginal=0.4;预言家私有查验一狼后 world_count=4 且该 seat marginal=1.0;狼人知道队友后 world_count=1 且其他好人 constrained marginal=0。
  - `tests/test_orchestrator.py` 覆盖 `posterior_trace[*].legal_worlds` 与 `constrained_posterior` 存在。
  - 窄测:`PYTHONPATH=. pytest -q tests/test_evidence_graph.py tests/test_orchestrator.py` → `20 passed`。
- 诚实 caveat:这是 **team-world** constrained posterior,不是 full role-world posterior。它只约束哪些 seat 是狼人,尚未枚举 seer/witch/guard/hunter 等具体好人角色;`constrained_werewolf_suspicion` 的 likelihood 仍来自启发式 suspicion,不是学习过或校准过的 evidence likelihood。下一步应基于 `evidence_items` 给每类 evidence 建 likelihood contribution,并在 multi-game stats 中并排报告 heuristic vs constrained 的 Brier/log loss/ECE。

### 3.25 §3.24 硬化收口:隐私边界 + constrained metrics + smoke 可靠性(✅,build/test/smoke 通过)
- 触发:继续推进 §3.24,并按用户要求高并发开子代理审计。并行子代理覆盖三条线:① EvidenceGraph/legal_worlds 隐私与指标语义;② 前端 constrained posterior 展示完整性;③ 真实 smoke/router/API 超时风险。主线完成集成、修复和真实 smoke。
- **新增联网调研依据(2026-07-05 核验)**:
  - Werewolf Arena(arXiv:2407.13943, https://arxiv.org/abs/2407.13943):用狼人杀评测 LLM 的 deception/deduction/persuasion,并强调动态 bidding 发言权;继续支持 werewolf-mas 的 bid/reply/accuse 路线。
  - BayesBench(arXiv:2606.30850, https://arxiv.org/abs/2606.30850):多轮 evidence accumulation 下应评测 belief trajectories,不能只看 final answer;直接支持 `posterior_trace`、Brier/log loss/ECE 和 posterior shift 指标。
  - M3-BENCH(arXiv:2601.08462, https://arxiv.org/abs/2601.08462):mixed-motive games 需要 process-aware evaluation(BTA/RPA/CCA), outcome-only 会漏掉"推理强但沟通弱"和潜在机会主义;支持本项目把 dialogue/objective/posterior/quality 分层报告。
  - WOLF(arXiv:2512.09187, https://arxiv.org/abs/2512.09187):Werewolf deception 评测应区分 deception production/detection,按 omission/distortion/fabrication/misdirection 分类,并记录纵向 suspicion;支持继续保留 speaker intent,但不能把自报 deception 暴露给普通 live 客户端。
  - CSP4SDG(arXiv:2511.06175, https://arxiv.org/abs/2511.06175)+ GRAIL/Graph-Informed Language Models(ACL 2026, https://camp-lab-purdue.github.io/bayesian-social-deduction/):隐藏身份推理应由可解释约束/概率图承担,LLM 负责语言理解与互动;支持从 team-world constrained posterior 继续走向 full role-world + likelihood contribution。
- **隐私与正确性硬化**:
  - `information.build_observation()` 的 `private_events` 仅保留 `EventVisibility.PRIVATE` 且 recipient 可见的事件,不再把 public events 混入名义上的 private list。
  - `evidence._collect_private_seer_results()` 防御性忽略非 private 的 `seer_result`,避免错误 public 事件被当成某玩家硬查验。
  - `evidence._legal_team_worlds()` 只在 `my_role == "werewolf"` 时消费 `my_teammates`,非狼人 observation 即使异常带 teammates 也不会产生 known_wolves。
  - `evidence._constrained_wolf_marginals()` 在 zero legal worlds / contradictory constraints 下返回空 constrained posterior,不再伪造"所有人 0.0 狼概率"。
  - `orchestrator._emit(speech)` 改为内部 `_analysis_deception`:狼人自报 deception 仍进入 `_speech_log` 供赛后 `dialogue_metrics/quality` 使用,但实时 `speech` 事件不再广播 `deception` 给 play/spectate/god/replay WS 事件流。普通 live UI 不再看到狼人策略;赛后 analysis 仍能统计 `wolf_deception_count/dist`。
  - `avg_speech_posterior_shift` 改为按 `(viewer, day)` 分组,只比较同一天连续 speech/pk_speech 快照,避免把夜间查验/死亡造成的后验变化归因给次日发言。
- **constrained posterior 指标闭环**:
  - `orchestrator._posterior_metrics()` 并排输出 constrained calibration: `constrained_final_brier_score`、`constrained_final_log_loss`、`constrained_good_final_brier_score`、`constrained_good_final_log_loss`、`constrained_calibration_ece`、`constrained_calibration_bins`。
  - `tests/multi_game_stats.py` 将 constrained 指标纳入 `POSTERIOR_KEYS`,多局统计可 bootstrap CI 汇总 heuristic vs constrained posterior。
  - `frontend` 状态面板/复盘面板/趋势卡展示约束 Brier/LogLoss/ECE、好人约束 LogLoss、约束校准分桶和 §3.24 trace 产物(`legal_worlds/constrained_posterior/evidence_items`)。
- **smoke 可靠性硬化**:
  - `tests/smoke_e2e.py` 改为 `main() -> int`,支持 `WEREWOLF_SMOKE_TIMEOUT`(默认 900s),超时/异常/任意 `agent_decision_failed`/未 ended/无 winner 均非零退出。
  - 推荐真实 smoke 命令固定为 `timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py`,避免底层 HTTP 取消卡住时无界等待。
- **验证**:
  - 窄测:`PYTHONPATH=. pytest -q tests/test_evidence_graph.py tests/test_orchestrator.py tests/test_multi_game_stats.py` → `39 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `74 passed, 3 skipped`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10` → 通过,临时 JSONL 已删除。
  - 真实 LLM smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 村民 day2 胜;真实模型调用 50 次,成功 50,失败 0,重试 0,0 `agent_decision_failed`,进程退出码 0。对局中 3号预言家报5号金水后被刀,2/6号狼人用"查死人模糊/1号沉默"带节奏,1/4号基于公开发言链和站边链在 day2 投出6号,验证证据链与后验路线不破坏真实对抗。
- **诚实 caveat / 下一步风险**:
  - `analysis.posterior_trace` 含每个 viewer 的私有约束和赛后真身份同场输出,当前语义是赛后全量复盘/god/replay,不能误当 live-safe public trace。若将来要给普通赛后玩家分视角复盘,需拆 public-safe trace 与 god trace。
  - `final_*` posterior calibration 混合所有 viewer(包含预言家私查与狼人队友真知),解释时应主推 `good_final_*` / `constrained_good_final_*`;all-viewer 指标只能说明"所有信息集下的后验质量"。
  - Router 流式调用仍只有帧间 timeout,没有每次 LLM 调用 wall-clock 总 timeout;API `RoomManager._run_room()` 也没有房间级 timeout,异常/取消后仍可能被标成 `ended`。这不是本轮改动范围,但已确认为下一阶段 P0/P1。

### 3.26 稳定性 P0/P1:LLM 调用总时限 + 房间失败状态(✅,build/test/smoke 通过)
- 触发:§3.25 smoke 审计确认两个生产风险:① OpenAI/Anthropic 流式路径只有 SSE 帧间 `chunk_timeout`,若网关持续发送 keepalive/空 delta,单次调用可无限挂住;② API 房间后台任务无房间级 timeout,异常/取消最终也会广播 `room_status: ended`,客户端可能误判为正常结束。
- **Router per-attempt wall-clock timeout**:
  - `src/llm/router.py` 明确区分 `timeout` 与 `chunk_timeout`:前者是每次 attempt 的 wall-clock 总时限,后者仍是 SSE 帧间 idle 超时。
  - `_complete()` 对 `_call_openai/_call_anthropic` 外包 `asyncio.wait_for(..., timeout=self.timeout)`。持续 keepalive 不能绕过总时限;外层取消仍按 `CancelledError` 传播,不被吞掉。
  - `asyncio.TimeoutError` 进入 retryable 逻辑,与 httpx 瞬时网络错误一致;重试耗尽后才抛 `LLMError`,继续遵守 no-fallback。
  - 新增 `tests/test_llm_router.py`:本地 `ThreadingHTTPServer` 持续发送 SSE keepalive 但永不 `[DONE]`,验证 router 在 `timeout=0.25s` 内以 `LLMError("总超时")` 失败,而不是等到 `chunk_timeout=2s` 或无限挂住。不调用真实 LLM。
- **RoomManager 房间级 timeout / 失败状态**:
  - `Room` 状态从 `waiting/running/ended` 扩展为 `waiting/running/ended/failed/timeout/cancelled`,并记录 `end_reason/error`。
  - `RoomManager` 新增 `room_timeout`,默认从 `WEREWOLF_ROOM_TIMEOUT` 读取(默认 900s,≤0 可禁用)。
  - `_run_room()` 现在:
    - 正常完成且 `state.phase == Phase.ENDED` 且有 `winner` 才标 `ended/completed`。
    - `asyncio.TimeoutError` 标 `timeout`,广播 `game_error(reason=timeout)` 与 `room_status(status=timeout)`。
    - 普通异常标 `failed`,广播 `game_error(reason=error)` 与 `room_status(status=failed)`。
    - 主动取消标 `cancelled`,广播 `room_status(status=cancelled)` 后重新抛出 `CancelledError`,让调用方能感知取消。
    - orchestrator 缺失或提前返回未 ended/winner 标 `failed`,不再伪装正常结束。
  - `RoomManager.aclose()` 会 cancel 并 await 未完成房间任务,避免关闭时留下后台 task。
  - REST `/api/rooms`, `/api/rooms/{id}`, `/start`, `/replay` 现在返回 `end_reason/error`。
  - 前端 `RoomInfo` 与 `room_status` 类型扩展,房间页显示 failed/timeout/cancelled 标签和错误信息;store 在非正常状态写入 system log。
  - 新增 `tests/test_room_manager.py`:覆盖正常完成、超时、异常、取消四条路径,证明失败/超时/取消不会被报告为 `ended`。不调用真实 LLM。
- **验证**:
  - 窄测:`PYTHONPATH=. pytest -q tests/test_llm_router.py tests/test_room_manager.py tests/test_api.py tests/test_api_human.py tests/test_websocket_events.py` → `17 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `79 passed, 3 skipped`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10` → 通过,临时 JSONL 已删除。
  - 真实 LLM smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 村民 day2 胜;真实模型调用 50 次,成功 50,失败 0,重试 0,0 `agent_decision_failed`,进程退出码 0。新增 per-attempt wall-clock timeout 未误杀正常真实流式调用。
  - 凭据扫描:按用户提供密钥唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- **诚实 caveat / 下一步风险**:
  - Router 仍保留 `_parse_json()` 的末尾字段截断兜底;这能提升真实模型容错,但也可能把截断 JSON 当成成功 dict。下一步应引入 strict decision JSON 模式或 `parse_lossy` 元信息,让 smoke/决策 schema 能区分完整 JSON 与有损恢复。
  - 仍没有 per-decision / per-phase timeout;当前单次 LLM attempt 有总时限,房间有总时限,但 orchestrator 阶段内部 `asyncio.gather` 仍会等慢 agent 到 router/actor 重试耗尽。下一步可在 agent decision 层加 phase-specific timeout,并把超时转为 `agent_decision_failed`。

### 3.27 Strict/Lossy JSON parse 审计闭环(✅,build/test/smoke 通过)
- 触发:§3.26 确认 `_parse_json()` 的"末尾字段截断兜底"会把截断 JSON 静默重建成 dict,可能让真实 LLM 的坏输出被误判为完整决策。目标不是删除容错,而是区分"无损恢复"与"有损恢复",并让 agent 决策默认不能静默接受有损输出。
- **Router 解析分层**:
  - 新增 `JSONParseResult(data/method/recovered/lossy)`。
  - `_parse_json_result()` 标记解析来源:
    - `json`:标准 JSON,无恢复。
    - `literal`:Python/单引号字面量,无损恢复。
    - `embedded_json/embedded_literal`:从围栏或外围文本中提取完整对象,无损恢复。
    - `balanced_literal`:只补缺失右花括号等结构闭合,无损恢复。
    - `lossy_kv`:逐个提取完整 key-value、丢弃末尾截断字段,**有损恢复**。
  - `_parse_json(..., allow_lossy=False)` 默认拒绝 `lossy_kv`,抛 `LLMError("JSON 有损恢复被拒绝")`;显式 `allow_lossy=True, include_parse_metadata=True` 时返回 `_parse_lossy/_parse_recovered/_parse_method` 元信息。
  - `complete_json()` 新增 `allow_lossy/include_parse_metadata` 参数,默认仍是严格模式。
- **Agent 决策策略**:
  - `AgentActor._call_with_retry()` 前 `max_attempts-1` 次保持 strict parse:有损恢复触发 `LLMError`,进入真实 LLM 重试。
  - 最后一次 attempt 显式 `allow_lossy=True, include_parse_metadata=True`,用真实模型输出做透明兜底,避免连续截断导致整局 `agent_decision_failed`。
  - 各 sanitizer 将 `_parse_lossy` 透传到 `Decision.parse_failed=True`,因此赛后/思考流可审计"这次决策来自有损 JSON 恢复",不再静默伪装成完整解析。
  - 该策略仍守 no-fallback:没有生成假决策、没有规则 bot;兜底内容来自真实 LLM 输出,只是透明标记解析质量。
- **测试**:
  - `tests/test_llm_router.py` 覆盖:
    - 持续 SSE keepalive 仍受 wall-clock timeout 限制(§3.26 回归保留)。
    - 截断字符串字段默认被拒绝为 `有损恢复`。
    - 显式 `allow_lossy=True` 时返回 `_parse_lossy=True/_parse_method=lossy_kv`,且不包含被截断字段。
    - 缺右花括号这类无损补全可通过,标记 `_parse_recovered=True/_parse_lossy=False`。
    - `AgentActor._call_with_retry()` 第一次 strict 失败后第二次以 `allow_lossy=True` 接收结果,并让 `Decision.parse_failed=True`。
- **验证**:
  - 窄测:`PYTHONPATH=. pytest -q tests/test_llm_router.py tests/test_orchestrator.py` → `18 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `84 passed, 3 skipped`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10` → 通过,临时 JSONL 已删除。
  - 真实 LLM smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 村民 day2 胜;真实模型调用 59 次,成功 59,失败 0,重试 0,0 `agent_decision_failed`,进程退出码 0。对局中真实触发过截断 JSON:第一次 `lossy_kv` 被 strict parser 拒绝并记录 warning,随后 actor 真实重试成功,证明新链路能捕捉坏输出且不打断完整对局。
  - 凭据扫描:按用户提供密钥唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- **诚实 caveat / 下一步风险**:
  - 最后一次有损兜底仍可能产生 `speech` 缺失而转为透明沉默/skip,这是比 `agent_decision_failed` 更连续但质量较低的真实输出。后续应统计 `parse_failed` 率,把它纳入 smoke/multi-game stats,并在前端/analysis 中可见。
  - 仍没有 per-decision/per-phase timeout。下一步应在 `AgentActor` 或 orchestrator 调用点包 phase-specific `asyncio.wait_for`,把决策超时变成带 phase/action/seat 的 `agent_decision_failed` 或透明 skip,而不是只依赖 router/room 两级 timeout。

### 3.28 Parse quality metrics + 第六轮多 agent 对抗/欺骗/社会调研(✅,build/test/smoke 通过)
- 触发:用户要求“开多个子代理,高并发,加快速率”,并追问“多agent对抗、欺骗、多agent社会”等是否已有研究。本轮并行子代理覆盖:① orchestrator Decision 记录点审计;② 多局统计实现;③ 前端展示实现;④ 最新研究方向;⑤ 后端 parse metrics 实现。主线负责集成、修正语义和验证。
- **实现目标**:把 §3.27 的 `Decision.parse_failed=True` 从“actor 内部透明标记”升级为赛后可量化指标。语义严格限定为:真实 LLM 输出在最后一次 attempt 经过有损 JSON 恢复后仍被 sanitizer 构造成 `Decision`;它不是假决策、不是规则 bot、也不是 LLM 调用失败。
- **后端**:
  - `GameOrchestratorV2` 新增 `_parse_decisions` 和 `_record_consumed_decision()`,记录 `day/phase/seat/action/parse_failed/skip_reason`。该日志只用于 analysis,不广播、不进 prompt、不写 memory,不含 speech 文本、reasoning、role truth、wolf caucus 或 deception 内容。
  - 记录口径调整为“`decide_*` 成功返回并进入 orchestrator 的 `Decision`”,因此包含 day speak、bid speak(`bid=0` 也计入)、vote skip/非法目标、night skip 等解析质量分母;规则合法性仍由 objective metrics / failed events 评估。
  - `analysis["parse_metrics"]` 输出 `decision_count`、`parse_failed_count`、`parse_failed_rate`、`parse_failed_by_action`、`parse_failed_by_phase`。其中 `parse_failed_by_action` 是 action→失败次数,便于 CLI/前端轻量展示。
  - `_submit_safe()` 返回 bool 只表达夜间行动是否真正提交到 RulesEngine;skip 不再返回 True,避免女巫 save/poison 阶段把合法 skip 误当用药成功。
- **多局统计**:
  - `tests/multi_game_stats.py` 接入 `parse_metrics`,单局摘要显示 `parse_failed=<failed>/<decision_count>` 和 `parse_rate`。
  - 汇总新增 `=== Parse metrics ===`,对 `decision_count/parse_failed_count/parse_failed_rate` 输出 bootstrap 95% CI,并对 `parse_failed_by_action` 聚合 total/mean/CI。
  - `tests/test_multi_game_stats.py` 覆盖空结果、单局、多局、非法/布尔值过滤和 action 分布聚合。
- **前端**:
  - `frontend/src/lib/types.ts` 新增 `ParseMetrics` 与 `GameAnalysis.parse_metrics`。
  - `GameStatusPanel.tsx` 新增“解析质量指标”卡,展示决策数、有损解析数、失败率和 action 分布;跨局趋势追加解析失败率/有损解析均值。
  - `ReplayPanel.tsx` 在复盘客观指标区展示 parse metrics 和 action 分布;`trends.ts` 把 parse metrics 写入本地趋势。
- **测试**:
  - `tests/test_orchestrator.py` 新增 `test_analysis_includes_parse_metrics_for_lossy_decisions`:用 MockActor 人为制造一次 speak 和一次 vote 的 `parse_failed=True`,断言 analysis 计数/phase/action 正确,并确认 live `speech/vote_cast` 事件不携带 `parse_failed`。
  - 窄测:`python -m compileall -q src/game/orchestrator.py tests/test_orchestrator.py tests/multi_game_stats.py tests/test_multi_game_stats.py` → 通过。
  - 窄测:`PYTHONPATH=. pytest -q tests/test_orchestrator.py tests/test_multi_game_stats.py tests/test_llm_router.py` → `36 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `86 passed, 3 skipped`。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10` → 通过,临时 JSONL 已删除。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - 真实 LLM smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 村民 day2 胜;真实模型调用 56 次,成功 56,失败 0,0 `agent_decision_failed`,进程退出码 0。对局中多次真实触发 `lossy_kv`/空 JSON,strict parser 拒绝后 actor 重试成功,验证 parse quality metrics 接入不破坏真实对局。
  - 凭据扫描:按用户提供密钥唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- **第六轮联网/子代理调研增量(2026-07-05 核验)**:
  - Werewolf Arena(arXiv:2407.13943, https://arxiv.org/abs/2407.13943):Werewolf 明确用于评测 deception/deduction/persuasion,动态 bidding 发言权是核心机制。werewolf-mas 应把 `turn_policy` 做成可复现 ablation,输出 `bid_entropy/reply_latency/speaker_concentration`,而不是只保留直觉调度。
  - Training Language Models for Social Deduction with MARL(AAMAS 2025, https://arxiv.org/abs/2502.06060):把 listening/speaking 拆开,用消息对其他 agent 世界状态预测的影响作为 dense reward。werewolf-mas 的 PS/DR 后续应以 `belief_shift_after_speech/correct_belief_gain/misdirection_gain/speech_to_vote_conversion` 为主,少依赖 LLM judge 主观分。
  - MultiMind(arXiv:2504.18039, https://arxiv.org/abs/2504.18039):ToM 不只是“我怀疑谁”,还要表示“别人如何看我/看别人”。后续可从当前 `attitudes` 升级到稀疏 ToM matrix,并做 Jester/三阵营 research mode 压测多跳 ToM。
  - OpenDeception(arXiv:2504.13707, https://arxiv.org/abs/2504.13707):欺骗风险要从 deceptive intent 与 listener susceptibility 两侧评估。现有 `deception` 只能当 speaker intent,不可当真值;下一步应做独立 `deception_audit` 和 listener posterior shift 对齐。
  - AgentSociety(arXiv:2502.08691, https://arxiv.org/abs/2502.08691):LLM 多 agent 社会模拟正走向大规模、事件化、可复现、可干预。werewolf-mas 的中期形态应是 event-sourced social state + intervention/ablation protocol,不是临时跑几局看感觉。
  - 研究子代理还补充了 TextArena/MINDGAMES、WOLF、CSP4SDG/GRAIL、Triadic Werewolf、collusion/open-channel narrative、LLM debate/herding、adversarial behavior taxonomy 等方向。落地原则:任何“胜率/质量更强”的说法必须同时报告 JSONL、CI、错误率、parse_failed、game_ended 异常、objective/posterior 指标。
- **新的可执行路线**:
  1. `turn_policy` ablation: fixed/bid/bid+reply/bid+reply+caucus,同模型同种子多局 JSONL + CI。
  2. `deception_audit`:speaker 自报 vs 独立审计 vs peer detection vs listener posterior shift。
  3. `collusion_audit`:检测多狼公开频道叙事重叠、truth-fragment montage、support loop 和 target posterior swing;harness 只检测,不替狼写话术。
  4. debate process metrics:majority pressure、wrong consensus reversal、minority evidence survival、claim challenged rate。不要把“热闹”误判为高质量辩论。
  5. reference pool / arena protocol:固定低错误率基准 agent,候选策略与 frozen pool 对战,胜率必须和错误率、parse_failed、非法行动率一起报告。
- 诚实 caveat:`parse_failed` 可见化解决的是“输出结构质量审计”,不是 agent 社会智能本身。真实瓶颈仍是可校准后验、独立欺骗审计、合谋检测、辩论纠错和可复现实验协议。

### 3.29 Deception audit v1:speaker intent × independent audit × listener shift(✅,build/test/smoke 通过)
- 触发:§3.28 调研明确 `deception` 不能再只当狼人自报字段;OpenDeception/WOLF 路线要求把 speaker deceptive intent 与 listener susceptibility / peer detection / posterior shift 分开。本轮先做 **确定性 v1**,不额外调用 judge,不改 live agent 决策。
- **语义边界**:
  - `speech.deception` 仍只表示狼人自报的 speaker intent,不是欺骗真值。
  - `deception_audit` 是赛后 analysis-only 指标,使用赛后真值、公开 speech 元数据和 `posterior_trace` 做粗粒度审计;不进入 live prompt/memory,不广播实时事件,不含 speech 原文或 reasoning。
  - 这是 WOLF/OpenDeception 的最小工程落点,不是完整逐句语义欺骗审计。它能回答“狼人是否声明欺骗、独立规则是否也判为欺骗、听众后验是否被带偏”。
- **后端**:
  - `GameOrchestratorV2._deception_audit()` 新增 `analysis["deception_audit"]`。
  - 输出字段:
    - `wolf_speech_count`
    - `declared_deception_count`
    - `audited_deception_count`
    - `declared_vs_audited_agreement`
    - `deception_success_rate`
    - `avg_good_target_suspicion_gain`
    - `villager_false_positive_rate`
    - `successful_misdirection_count/target_good_audit_count`
    - `declared_by_type/audited_by_type`
    - `records`(最多 40 条紧凑记录:day/seat/declared/audited_types/target seats/avg gain/success flag,不含文本)
  - 审计规则 v1:
    - 非预言家声称 seer 或查验结果与真值冲突 → `fabrication`。
    - 狼人指控/反对好人 → `misdirection`。
    - 狼人围绕真预言家使用查验/跳/假/急/细节等质疑词 → `distortion`。
    - 狼人文本自称“我不是狼/我是好人/我是村民/平民”等 → `fabrication`。
  - `deception_success_rate` 只在有可测 listener posterior before/after 的 misdirection 样本上统计;若好人 viewer 对被误导目标的狼嫌疑平均上升 >0.02,记为成功误导。
  - 修 `_posterior_speech_shift_groups()` 对同一 speaker 连续发言的分组:同一 viewer 重复出现即新 speech 边界,避免把连续发言的 posterior 快照合并。
- **多局统计**:
  - `tests/multi_game_stats.py` 接入 `deception_audit`,单局摘要显示 audit 关键数值。
  - 汇总新增 `=== Deception audit ===`,对 `wolf_speech_count/declared_deception_count/audited_deception_count/declared_vs_audited_agreement/deception_success_rate/avg_good_target_suspicion_gain/villager_false_positive_rate` 输出均值 + bootstrap 95% CI。
  - `audited_by_type` 按类型聚合 total/mean/CI。
- **前端**:
  - `frontend/src/lib/types.ts` 新增 `DeceptionAudit` 与 `GameAnalysis.deception_audit`。
  - `GameStatusPanel.tsx` 新增“欺骗审计指标”卡,展示狼发言、声明欺骗、审计欺骗、声审一致、欺骗成功率、好人疑增、村民误报和审计类型 chips。
  - `ReplayPanel.tsx` 在复盘指标区展示 deception audit,foot 摘要显示狼发言/声明/审计/类型分布。
  - `trends.ts` 将 deception audit 指标与 `audited_by_type` 清洗后写入 localStorage 趋势。
- **测试**:
  - `tests/test_orchestrator.py` 新增 `test_deception_audit_compares_declared_intent_with_listener_shift`:构造狼人自报 misdirection/distortion、好人误指控、posterior shift,验证独立审计类型、声审一致、误导成功率、好人目标嫌疑增益和隐私边界。
  - `tests/test_multi_game_stats.py` 覆盖 deception audit 汇总、百分比格式、字符串数值、非法值/布尔过滤和 `audited_by_type` 聚合。
  - 窄测:`PYTHONPATH=. pytest -q tests/test_orchestrator.py tests/test_multi_game_stats.py` → `32 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `88 passed, 3 skipped`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10` → 通过,临时 JSONL 已删除。
  - 真实 LLM smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 狼人 day2 胜;真实模型调用 60 次,成功 60,失败 0,重试 0,0 `agent_decision_failed`,进程退出码 0。对局中 3号狼人被 5号预言家查杀出局后遗言悍跳预言家、反验 5号狼并给 1号金水;夜里 5号死亡后,3号遗言成功误导 1/2 号在 day2 放逐 4号好人,验证 `deception_audit` 的 speaker intent × listener shift 路线能捕捉真实多 agent 欺骗样本。
  - 凭据扫描:按用户提供密钥唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- 诚实 caveat:当前 audit 是 deterministic coarse heuristic,不是逐句事实核查或 LLM/人工标注。`distortion/misdirection/fabrication` 的判定偏保守且依赖结构化字段;`deception_success_rate` 依赖现有启发式 posterior。下一步应做 evidence-item 级 audit、peer detection、listener susceptibility by seat,并把 `records[*]` 与 `posterior_deltas/evidence_items` 对齐。

---

### 3.30 Deception audit v2:evidence refs + peer detection + listener susceptibility(✅,build/test/smoke 通过)
- 触发:§3.29 已能做 speech 级 speaker intent × independent audit × listener shift,但仍缺三件事:① 哪些 `evidence_items/posterior_deltas` 支撑该审计;② 好人是否识破说话狼人;③ 哪些听众更容易被误导。并行子代理审计还指出两个可信度风险:vote/PK 后验变化可能被错归因给 speech,以及 `posterior_trace` 截断会让 analysis 无法离线复算。
- **语义边界**:
  - v2 仍是赛后 analysis-only;不进入 live prompt/memory,不广播实时事件,不影响 agent 决策。
  - `deception_success_rate` 仍只统计有 before/after listener posterior 的可审计 misdirection,但新增覆盖率和不可审数量,不再隐藏样本缺口。
  - `records[*]` 只保存结构化 id、座位号、数值和类型分布;不保存 speech 原文、private reasoning、wolf caucus 或隐藏推理。
- **后端**:
  - `_deception_audit()` 扩展 v2 字段:
    - `misdirection_shift_coverage` / `unauditable_misdirection_count`
    - `detected_deception_count` / `peer_detection_opportunity_count` / `peer_detection_rate`
    - `avg_speaker_suspicion_gain`
    - `listener_shift_sample_count`
    - `evidence_linked_count`
    - `listener_susceptibility_by_seat`
  - `records[*]` 新增:
    - `evidence_ids`
    - `posterior_delta_ids`
    - `evidence_source_types`
    - `listener_shifts`(每个好人 viewer 的目标疑增、说话人疑增、是否被误导、是否识破)
    - `peer_detection`(是否有人识破、识破者 seats、平均说话人疑增)
  - 新增 `_speech_evidence_refs()`:把 speech audit record 对齐到同一 speech 后 listener 可见的公开 `evidence_items/posterior_deltas`,过滤 private evidence。
  - 修 `_posterior_speech_shift_groups()` baseline:所有非 speech snapshot(例如 vote)会更新 viewer/day baseline,但只有 `speech/pk_speech` 产出 speech shift,避免把 vote 或 PK 触发信息变化归因给后续 PK 发言。
  - `_posterior_metrics().avg_speech_posterior_shift` 改用同一 baseline 语义,避免指标和 audit 口径分叉。
  - `analysis["posterior_trace"]` 改为完整输出,并新增 `posterior_trace_total_count/posterior_trace_truncated/posterior_trace_dropped_count`,保证 `posterior_metrics` 和 `deception_audit` 可由输出 trace 离线复算。
- **多局统计**:
  - `tests/multi_game_stats.py` 将 v2 标量加入 `DECEPTION_AUDIT_KEYS`。
  - 单局进度行抽为 `format_game_progress_line()`,使用真实字段名输出,减少人工/脚本比对歧义。
  - 汇总新增 `listener_susceptibility_by_seat` block,按 seat 聚合误导样本、识破样本、好人目标疑增、说话人疑增、误导率、识破率,并继续使用 bootstrap 95% CI。
- **前端**:
  - `frontend/src/lib/types.ts` 扩展 `DeceptionAudit`、`records[*]`、`listener_shifts`、`peer_detection`、`listener_susceptibility_by_seat` 类型。
  - `GameStatusPanel.tsx` / `ReplayPanel.tsx` 展示识破率、误导覆盖、说话人疑增、证据关联、不可审误导和听者样本。
  - count 字段统一使用降级格式化,避免缺失字段显示 `undefined`。
  - 单局 `audited_by_type` 不再截断;跨局趋势仍保留紧凑 Top3,但标签改为 `审计Top3`。
  - `trends.ts` 将 v2 标量写入 localStorage 趋势链路。
- **测试**:
  - `tests/test_orchestrator.py` 扩展 `test_deception_audit_compares_declared_intent_with_listener_shift`:断言 evidence ids / posterior delta ids / listener shifts / peer detection / listener susceptibility / coverage 字段,并继续断言不泄漏 speech 文本、reasoning、wolf_caucus。
  - 新增 `test_deception_audit_does_not_attribute_vote_shift_to_pk_speech`:构造 speech → vote → pk_speech,验证 PK 发言 before 使用 vote 后 baseline,`deception_success_rate` 不把 vote delta 算作 speech 误导。
  - `test_analysis_includes_posterior_trace_and_metrics` 增加完整 trace 元数据断言。
  - `tests/test_multi_game_stats.py` 覆盖 v2 标量、真实字段名单局进度、listener susceptibility 聚合、非法值/布尔过滤。
  - 窄测:`PYTHONPATH=. pytest -q tests/test_orchestrator.py tests/test_multi_game_stats.py` → `35 passed`。
  - 相关隐私/证据窄测:`PYTHONPATH=. pytest -q tests/test_evidence_graph.py tests/test_websocket_events.py::test_websocket_spectator_does_not_see_private_events` → `13 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `91 passed, 3 skipped`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10` → 通过,临时 JSONL 已删除。
  - 真实 LLM smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 狼人 day2 胜;真实模型调用 45 次,成功 45,失败 0,0 `agent_decision_failed`,进程退出码 0。过程中一次 `lossy_kv` 被 strict parser 拒绝后 actor 真实重试成功,未触发 fallback。对局中 6号狼人 day1 被放逐后遗言悍跳预言家并查杀 5号,夜里 5号死亡后 3号狼人借该遗言在 day2 带偏 1号投出 4号好人,验证 v2 能覆盖“狼声明/独立审计/证据引用/listener shift/peer detection”链路。
  - 凭据扫描:按用户提供密钥唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- 诚实 caveat:当前 v2 仍是规则/结构化证据审计,不是逐句自然语言事实核查。`evidence_ids/posterior_delta_ids` 可说明“哪些可见证据项驱动了后验变化”,但还不能自动标注 omitted counterevidence 或复杂语义歪曲;下一步应做 evidence-item 级 LLM/规则混合 auditor 或 controlled intervention judge,并与 `likelihood_delta` 后验校准合并。

---

### 3.31 信息隔离硬化:thinking stream + role resource public view(✅,build/test/smoke 通过)
- 触发:§3.30 并行审计指出两个信息隔离风险:① `verbose_thinking=True` 时 WebSocket `spectate` 可能收到完整 `reasoning`,其中可能包含狼人策略/私有推理;② `GameState.public_view()` 把 `witch_antidote/witch_poison/last_guarded_seat/pending_hunter` 当公开字段下发,会污染观战/普通玩家视图和后续社会推理评估。
- **语义边界**:
  - `god`:可见完整 thinking reasoning、完整身份、信任网络、隐藏技能资源。
  - `spectate`:只可见公开事件 + thinking summary;不得看到完整 `reasoning` 或隐藏技能资源。
  - `play`:只可见公开事件 + 自己 seat 的私有事件/身份/角色资源;不得看到其他角色资源或完整 thinking stream。
- **后端**:
  - `RoomManager._broadcast()` 改为按客户端生成 mode-specific payload,不再对所有客户端复用同一 JSON 字符串。
  - 新增 `_payload_for_client()`:`thinking=True` 且非 god 时移除 `reasoning`,spectate 只收到整理摘要。
  - `GameState.public_view()` 移除 `witch_antidote/witch_poison/last_guarded_seat/pending_hunter`。
  - `GameState.private_view_for()` 新增 `role_state`:女巫只看自己的药,守卫只看自己的上一夜守护 seat,猎人只看自己是否 pending shot。
  - `RoomManager._view_for(mode="god")` 新增 `hidden_state`,god 面板仍保留全知调试/研究能力,但不污染 public snapshot。
- **前端**:
  - `frontend/src/lib/types.ts` 将隐藏技能状态从公开 `SnapshotView` 顶层移到 `role_state` / `hidden_state`。
  - `frontend/src/lib/store.ts` 和 `ChatRoom.tsx` 注释同步:god 可见完整 reasoning,spectate 只显示后端净化后的 summary。
- **测试**:
  - `tests/test_websocket_events.py` 新增 `test_spectator_thinking_stream_hides_full_reasoning`:同房间连接 spectate/god,断言 spectate thinking 没有 `reasoning`,god 保留完整 `reasoning`。
  - `tests/test_websocket_events.py` 新增 `test_websocket_snapshot_hides_hidden_role_state_except_god`:spectate snapshot 不含隐藏技能字段或 `hidden_state`,god snapshot 含 `hidden_state`。
  - `tests/test_game_views.py` 新增公开/私有视图测试:公开视图不泄漏角色资源;女巫/守卫/猎人只在自己的 `role_state` 中看到对应私有资源。
  - 窄测:`PYTHONPATH=. pytest -q tests/test_game_views.py tests/test_websocket_events.py tests/test_api.py` → `14 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `95 passed, 3 skipped`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - 真实 LLM smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 狼人 day2 胜;真实模型调用 55 次,成功 55,失败 0,0 `agent_decision_failed`,进程退出码 0。过程中多次 `lossy_kv` 被 strict parser 拒绝并重试,最后一次真实使用有损恢复结果,未触发伪造/fallback;验证本轮信息隔离改动不破坏真实 orchestrator 对局链路。
  - 凭据扫描:按用户提供密钥唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- 诚实 caveat:这轮修的是 API/视图层信息隔离,不是 agent 内部推理质量;direct orchestrator smoke 不能覆盖 WebSocket privacy,所以核心证据是新增的 API/WebSocket 单测。

---

### 3.32 Turn policy ablation + debate process metrics(✅,build/test/smoke 通过)
- 触发:§3.28 调研/Werewolf Arena 指出动态 bidding 发言权是狼人杀社会推理的关键机制;§4.2 仍把 `turn_policy` ablation + `debate process metrics` 列为后续社会机制。本轮把白天发言调度做成可复现实验轴,同时保持默认真实对局行为不变。
- **语义边界**:
  - 只控制白天发言调度和 day1 狼人 caucus 是否启用;harness 不替 agent 写发言/话术,不伪造决策,不泄漏真身份或私有推理。
  - 默认 `bid_reply_caucus` = 旧行为:首日狼人 caucus + bid 调度 + 被点名/回应优先。
  - `debate_process_metrics` 是赛后 analysis-only 指标,只用公开 speech 结构化字段和调度配置,不读 hidden reasoning,不回灌 live prompt/memory。
- **后端**:
  - 新增 `TURN_POLICIES`: `fixed_round_robin` / `bid_only` / `bid_reply` / `bid_reply_caucus`;`DEFAULT_TURN_POLICY="bid_reply_caucus"`。
  - `GameOrchestratorV2(turn_policy=...)` 校验策略;`_run_day()` 按策略切换:fixed 固定 seat 顺序且无 day caucus;bid_only 只按 bid;bid_reply 启用 mentioned/reply priority;bid_reply_caucus 保持旧行为。
  - `_emit()` 将 `bid` 写入 `_speech_log`;`_run_analysis()` 输出 `analysis["turn_policy"]` 与 `analysis["debate_process_metrics"]`。
  - `_debate_process_metrics()` 输出 `caucus_enabled/uses_bid_order/uses_reply_priority/speech_count/speaker_count/speaker_concentration/bid_entropy/avg_bid/reply_count/avg_reply_latency/claim_count/claim_challenged_rate/accuse_target_count/top_accuse_target_share/support_loop_count/opposition_loop_count`。
- **多局统计**:
  - `tests/multi_game_stats.py --turn-policy {fixed_round_robin,bid_only,bid_reply,bid_reply_caucus}` 透传到 orchestrator;JSONL/单局进度记录 `turn_policy` 和 `debate_process_metrics`。
  - 汇总新增 `turn_policy 分布` 和 `=== Debate process metrics ===` bootstrap CI;单局摘要显示 `speaker_concentration/bid_entropy/claim_challenged_rate/top_accuse_target_share`。
- **前端**:
  - `types.ts` 增加 `DebateProcessMetrics` 和 `GameAnalysis.debate_process_metrics`。
  - `GameStatusPanel` 增加"辩论过程指标"卡和跨局趋势摘要(含策略分布、发言集中、Bid熵、声明挑战、围攻占比);`ReplayPanel` 在复盘客观指标展示同一组字段;`trends.ts` 持久化 `turn_policy` 与四个过程指标。
- **测试/验证**:
  - 策略/指标单测覆盖:`test_bid_only_policy_ignores_mentioned_priority`、`test_fixed_round_robin_policy_uses_fixed_order_without_caucus`、`test_debate_process_metrics_summarize_public_debate_shape`、`test_parse_args_accepts_turn_policy`、`test_print_summary_includes_debate_process_metrics`。
  - 编译:`python -m compileall -q src/game/orchestrator.py tests/multi_game_stats.py tests/test_multi_game_stats.py` → 通过。
  - 窄测:`PYTHONPATH=. pytest -q tests/test_orchestrator.py tests/test_multi_game_stats.py` → `40 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `100 passed, 3 skipped`。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10 --turn-policy bid_only` → 通过,`=== Debate process metrics === no metrics`,临时 JSONL 已删除。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - 真实 LLM smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 好人 day2 胜;真实模型调用 52 次,成功 52,失败 0,0 `agent_decision_failed`,进程退出码 0。对局中 6号预言家查杀4号狼人,1/4号狼用"跳太急/跟得太快"叙事反打,2/3号基于公开发言链和6号死亡识别最后一狼1号,验证默认 `bid_reply_caucus` 行为未被 ablation 接口破坏。
  - 凭据扫描:按用户提供密钥唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- 诚实 caveat:本轮完成的是可配置 ablation 轴和过程指标闭环,不是已经得出哪种调度最优。真正结论需要同模型/同参数/多策略 N 局 JSONL + CI,并与胜率、parse_failed、deception_audit、posterior 校准一起解释;`speaker_concentration/bid_entropy/claim_challenged_rate/top_accuse_target_share` 描述辩论形态,不等价于辩论质量。

---

### 3.33 Collusion audit v1:公开频道狼队合谋审计(✅,build/test/smoke 通过)
- 触发:§4.2 后续社会机制列出 `collusion audit`;§3.29/§3.30 已能审计单条狼人欺骗,§3.32 已能描述辩论过程形态,但还缺"多狼是否在公开频道形成接力、互挺、共同打靶、叙事重叠并影响听众后验"的确定性审计。
- **研究依据(2026-07-05 核验)**:
  - Werewolf Arena(arXiv:2407.13943, https://arxiv.org/abs/2407.13943):狼人杀评测应关注 deception/deduction/persuasion 与动态发言权,支持把公开发言互动过程沉淀成可复算社会指标。
  - OpenDeception(arXiv:2504.13707, https://arxiv.org/abs/2504.13707):欺骗评测要区分 deceptive intent 和 listener susceptibility,支持本轮继续用后验 shift 衡量合谋话术是否带偏听众。
  - AgentSociety(arXiv:2502.08691, https://arxiv.org/abs/2502.08691):LLM 多 agent 社会模拟应事件化、可复现、可干预;支持把公开合谋痕迹做成 analysis substrate,而非只读文本凭感觉判断。
- **语义边界**:
  - `collusion_audit` 不是"读狼队私聊判断他们是否串通";它只用赛后真值识别 wolf seats,再审计这些狼在公开频道中的协同行为。
  - analysis-only,不进入 live prompt/memory,不广播实时事件,不影响 agent 决策。
  - records 只输出 seat、day、类型、数值、evidence ids、posterior delta ids;不保存 speech 原文、private reasoning、wolf caucus、隐藏事件或狼人自报欺骗文本。
- **后端**:
  - `GameOrchestratorV2._collusion_audit()` 新增 `analysis["collusion_audit"]`。
  - 主数据源复用 `_speech_log` 的公开结构字段 + `_posterior_speech_shift_groups()` + `_speech_evidence_refs()`;不重新发明后验归因,继承 vote/PK baseline 语义和 private evidence 过滤。
  - 顶层指标:
    - `wolf_speech_count`
    - `wolf_pair_count`
    - `active_wolf_pair_count`
    - `wolf_to_wolf_support_count`
    - `mutual_support_pair_count`
    - `shared_good_target_count`
    - `shared_good_target_speaker_coverage`
    - `narrative_overlap_pair_count`
    - `avg_narrative_overlap`
    - `coordinated_pressure_count`
    - `avg_shared_target_suspicion_gain`
    - `avg_colluder_suspicion_gain`
    - `evidence_linked_count`
    - `records`
  - records 类型:
    - `shared_good_target`:同日多只狼人共同 accuse/oppose 同一好人目标,按不同狼 seat 去重,避免同一狼重复刷指标。
    - `wolf_support`:狼对狼公开 support,并标记是否互挺。
    - `narrative_overlap`:同日不同狼人发言的内部文本重叠只用于计算相似度,输出仅保留 overlap 数值和座位,不输出原文/片段/ngram。
- **多局统计**:
  - `tests/multi_game_stats.py` JSONL 记录 `collusion_audit`。
  - 单局进度行新增 `shared_good_target_count/wolf_to_wolf_support_count/narrative_overlap_pair_count/avg_shared_target_suspicion_gain`。
  - 汇总新增 `=== Collusion audit ===`,对核心指标输出均值 + bootstrap 95% CI;dry-run 0局时显示 `no metrics`。
- **前端**:
  - `frontend/src/lib/types.ts` 新增 `CollusionAudit` / `CollusionAuditRecord`,并在 `GameAnalysis` 上增加 `collusion_audit?: CollusionAudit`。
  - `GameStatusPanel` 新增"合谋审计指标"卡,展示狼队配对、活跃配对、狼互挺、互挺配对、共打好人、目标覆盖、叙事重合、协同施压、疑似收益、证据与 records 数。
  - `ReplayPanel` 的客观轨迹指标区展示合谋核心字段,并增加合谋审计脚注。
  - `trends.ts` 持久化 6 个核心趋势字段:`collusion_active_wolf_pair_count/collusion_wolf_to_wolf_support_count/collusion_mutual_support_pair_count/collusion_shared_good_target_count/collusion_avg_narrative_overlap/collusion_coordinated_pressure_count`。
- **测试/验证**:
  - 新增 `test_collusion_audit_detects_public_wolf_alignment_without_raw_text`:构造两狼同日共打一个好人、互相 support、文本重叠和 posterior shift,断言合谋指标、公开 evidence ids、private evidence 过滤,并确认 records 不含发言原文、reasoning、wolf_caucus。
  - `tests/test_multi_game_stats.py` 扩展单局进度和新增 collusion summary 测试,覆盖百分比字段、非法值/布尔过滤与 CI 输出。
  - 编译:`python -m compileall -q src/game/orchestrator.py tests/multi_game_stats.py tests/test_multi_game_stats.py tests/test_orchestrator.py` → 通过。
  - 窄测:`PYTHONPATH=. pytest -q tests/test_orchestrator.py tests/test_multi_game_stats.py` → `42 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `102 passed, 3 skipped`。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10` → 通过,输出 `=== Collusion audit === no metrics`,临时 JSONL 已删除。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - 真实 LLM smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 好人 day2 胜;真实模型调用 56 次,成功 56,失败 0,重试 4,0 `agent_decision_failed`,进程退出码 0。对局中 4号预言家查杀3号狼人,3号用"跳太急/6号死得蹊跷"反打,1号狼队友公开接力同一叙事并帮腔;2/5号 day2 识别 1号帮3号带节奏后投出最后一狼,验证本轮审计对象在真实对局中自然出现且不破坏 live 决策。
  - 凭据扫描:按用户提供密钥唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- 诚实 caveat:v1 是 deterministic structural audit,不是逐句语义事实核查。`avg_narrative_overlap` 内部用文本相似度计算,但输出不含文本;它只能说明公开叙事相似/接力,不能单独证明"私下串通成功"。pair listener susceptibility + deception records 对齐已在 §3.38 的 collusion audit v2 落地;下一步应做 v3 evidence-item auditor / omitted counterevidence / likelihood_delta 对齐,并用多策略/多模型 JSONL 做 ABBA 对照。

---

### 3.34 Turn-policy batch/ABBA 实验协议(✅,build/test/smoke 通过)
- 触发:§3.32 已把 `turn_policy` 做成 ablation 轴,§3.33 caveat 要求用多策略/多模型 JSONL 做 ABBA 对照;用户要求继续调研"多 agent 对抗、欺骗、多 agent 社会、合谋"并开子代理并行。本轮把现有真实 LLM 多局工具升级为可复现实验协议,并修复子代理审计指出的 ABBA 配对风险。
- **新增调研/路线依据(2026-07-05 核验)**:
  - Werewolf Arena(arXiv:2407.13943, https://arxiv.org/abs/2407.13943):动态 bidding/发言权是狼人杀 social deduction 评测核心;支持继续做 `fixed_round_robin/bid_only/bid_reply/bid_reply_caucus` 调度对照。
  - MultiMind(arXiv:2504.18039, https://arxiv.org/abs/2504.18039):显式 ToM/suspicion matrix 对隐藏身份推理关键;支持后续从当前 `attitudes` 扩到稀疏二阶 ToM。
  - OpenDeception(arXiv:2504.13707, https://arxiv.org/abs/2504.13707)+ WOLF(arXiv:2512.09187, https://arxiv.org/abs/2512.09187):欺骗评测要分 speaker intent、observed deception、peer detection、listener susceptibility;支持 §3.29/§3.30 的 deception audit 路线继续升级。
  - CSP4SDG(arXiv:2511.06175, https://arxiv.org/abs/2511.06175)+ GRAIL/Bayesian Social Deduction with Graph-Informed Language Models(https://camp-lab-purdue.github.io/bayesian-social-deduction/):隐藏身份推理应由证据图/约束后验承担,LLM 负责语言互动;支持 §3.19-§3.25 的 EvidenceGraph/RolePosterior 主线。
  - Social Deduction MARL(arXiv:2502.06060, https://arxiv.org/abs/2502.06060)、BayesBench(arXiv:2606.30850, https://arxiv.org/abs/2606.30850)、M3-Bench(arXiv:2601.08462, https://arxiv.org/abs/2601.08462):评估应看 process-aware metrics、belief trajectory、evidence accumulation,不能只看单局胜负或单次 LLM judge。
  - 多 agent 合谋/审计方向:COLOSSEUM(arXiv:2602.15198, https://arxiv.org/abs/2602.15198) 和 Secret Collusion among AI Agents(OpenReview, https://openreview.net/forum?id=5F71Wn1vaV) 提醒多 agent 系统会出现隐蔽协同行为,支持 werewolf-mas 把 collusion audit 做成可复算指标,而不是只读文本凭感觉。
- **语义边界**:
  - 仍然只跑真实 LLM 对局;不引入 fake fallback、回放、脚本 bot 或 harness 代写发言。
  - 本轮改的是 `tests/multi_game_stats.py` 实验调度/统计层,不改变默认 `GameOrchestratorV2` 对局行为。
  - 多策略模式中 `n_games` 语义改为"每个 policy 跑几局";单策略 `--turn-policy` 仍保持兼容。
  - ABBA 配对必须有 `--seed/--experiment-seed`;没有 seed 时直接 argparse 报错,避免生成看似同 `pair_id`、实际发牌/RNG 不受控的假配对。
- **实验 CLI**:
  - 单策略兼容:`PYTHONPATH=. python tests/multi_game_stats.py 6 --turn-policy bid_reply_caucus`
  - 多策略顺序批量:`--turn-policies bid_only,bid_reply --policy-order sequential --seed 100 --experiment-id exp1`
  - ABBA 配对:`--turn-policies bid_only,bid_reply --policy-schedule abba --experiment-seed 100 --experiment-id exp-abba`
  - `--turn-policies all` 支持全部 `TURN_POLICIES`,但多策略必须显式 seed。
- **调度/元数据**:
  - 新增 `build_policy_schedule()`:
    - sequential: A1,A2,...,B1,B2,...;同一 `policy_game_idx/case_idx` 跨 policy 共享 `role_seed/actor_seed/orchestrator_seed`。
    - ABBA:仅允许两个 policy 且每 policy 局数为偶数;顺序为 A1,B1,B2,A2,每个 pair 共享 case seed。
  - `run_one_game()` 接收 `role_seed/actor_seed/orchestrator_seed/game_id/experiment_meta`,并把 `game_id` 传给 `new_game`,把 seed 分别传给发牌、actor persona RNG、orchestrator RNG。
  - JSONL 每局记录顶层与嵌套 `experiment` 元数据:`experiment_id/policy_order/policy_set/policy_alias/policy_index/policy_game_idx/policy_count/pair_id/counterbalance_order/abba_position/scheduled_total/base_seed/case_seed/role_seed/actor_seed/orchestrator_seed/game_id/player_names`。
  - 每局还记录 `roles_by_seat` 与 `router_stats_delta`,便于离线复算胜负、错误率、调用成本和延迟;这些只在赛后 JSONL 中使用,不进入 live prompt。
- **统计输出**:
  - 单局进度行新增 `experiment_id/policy_order/policy_alias/pair_id/counterbalance_order/policy_game_idx/scheduled_total/case_seed/role_seed/actor_seed/orchestrator_seed/abba_position`。
  - 总汇总新增 `experiment_id 分布`、`policy_order 分布`。
  - 多策略结果新增 `=== Turn policy grouped summaries ===`,提示总体汇总仅作诊断,优先看 per-policy 分组。
  - per-policy 分组现在覆盖 dialogue/objective/parse/debate/deception/collusion/posterior/WereAlign 关键指标,避免只看总体均值。
  - ABBA 结果新增 `=== ABBA paired deltas ===`,按 `pair_id` 计算 B-A paired delta,报告 usable pairs、incomplete pairs、seed mismatch、AB/BA order 分布以及 `village_win/failed/game_quality/reply_rate/vote_accuracy_good/good_final_wolf_suspicion_gap/deception_success_rate/shared_good_target_count` 的 delta CI。
- **子代理审计修复**:
  - Zeno 指出:此前 `seed=None` 时仍生成 `pair_id`,但 A/B 实际不会共享发牌和 RNG。已改为多策略必须显式 seed,并加测试锁住。
  - 同时把 `pair_id/counterbalance_order/abba_position/case_seed/scheduled_total` 放到顶层和进度行,降低离线脚本误读概率。
- **测试/验证**:
  - 新增/更新 `tests/test_multi_game_stats.py`:覆盖 multi-policy seed 强制、seeded sequential/ABBA schedule、invalid ABBA shape、experiment metadata、router delta、进度行元数据、per-policy 分组指标、ABBA paired delta。
  - 编译:`python -m compileall -q tests/multi_game_stats.py tests/test_multi_game_stats.py` → 通过。
  - 窄测:`PYTHONPATH=. pytest -q tests/test_multi_game_stats.py` → `31 passed`。
  - 组合窄测:`PYTHONPATH=. pytest -q tests/test_orchestrator.py tests/test_multi_game_stats.py` → `50 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `110 passed, 3 skipped`。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10 --turn-policies bid_only,bid_reply --policy-schedule abba --experiment-seed 100 --experiment-id dry-abba` → 通过,临时 JSONL 已删除。
  - no-seed guard:`PYTHONPATH=. python tests/multi_game_stats.py 0 --turn-policies bid_only,bid_reply --policy-schedule abba` → exit 2,argparse 明确提示多策略必须 `--seed/--experiment-seed`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - 真实 LLM smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 村民 day2 胜;真实模型调用 51 次,成功 51,失败 0,重试 2,0 `agent_decision_failed`,进程退出码 0。对局中 2号真预言家查杀3号狼,3/4号狼公开形成反跳和"跳太急"叙事,1/5号最终基于2号死亡与3号遗言反证投出4号,验证默认真实对局链路未被实验工具改动破坏。
  - 凭据扫描:按用户提供密钥唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- 诚实 caveat:本轮实现的是 **实验协议和配对统计能力**,不是已经得出哪种 turn_policy 最优。真正结论还需要跑 N>0 的真实 ABBA JSONL,并同时解释胜率、parse_failed、router retries、deception/collusion audit、posterior 校准和 paired delta;ABBA paired delta 仍是小样本辅助统计,不能替代充分样本与跨模型验证。

---

### 3.35 Turn-policy ABBA 真实小批量实验 + router stats 修正(✅,report/test/build/smoke 通过)
- 触发:§3.34 完成 ABBA 协议后,§4.2 下一步要求 N>0 真实实跑与解释。本轮实际跑了一个 4 局真实 LLM ABBA 小批量,并将结果固化为实验报告。
- **真实实验**:
  - 命令:
    ```bash
    timeout --foreground -s INT -k 30s 64m python -u tests/multi_game_stats.py 2 \
      --jsonl logs/turn_policy_abba_bid_reply_20260705.jsonl \
      --bootstrap-iters 500 \
      --turn-policies bid_only,bid_reply \
      --policy-schedule abba \
      --experiment-seed 20260705 \
      --experiment-id turn-policy-abba-bid-reply-20260705 \
      | tee logs/turn_policy_abba_bid_reply_20260705.out
    ```
  - 调度:`bid_only(A1)` → `bid_reply(B1)` → `bid_reply(B2)` → `bid_only(A2)`,两对 pair 分别共享 case seed。
  - 产物:
    - JSONL:`logs/turn_policy_abba_bid_reply_20260705.jsonl`
    - 控制台摘要:`logs/turn_policy_abba_bid_reply_20260705.out`
    - 报告:`docs/experiment-turn-policy-abba-2026-07-05.md`
- **结果(诚实小样本)**:
  - 4/4 村民胜,全部 day2;Wilson CI 仍很宽(`village 100%,95%CI[51.0%,100.0%]`),不能宣称平衡结论。
  - `agent_decision_failed=0`,`game_ended` 异常 0,`parse_failed_rate=0.0%`。
  - 真实 router 总量:207 calls / 207 successes / 0 failures / 1 retry / 908,853 input tokens / 73,988 output tokens。
  - per-policy:
    - `bid_only`:2/2 村民胜,106 calls,1 retry,reply_rate 40.2%,game_quality 0.75,good_final_wolf_suspicion_gap 0.34,deception_success_rate 54.1%,shared_good_target_count 2.00。
    - `bid_reply`:2/2 村民胜,101 calls,0 retries,reply_rate 46.8%,game_quality 0.80,good_final_wolf_suspicion_gap 0.30,deception_success_rate 56.2%,shared_good_target_count 1.50。
  - paired delta(`bid_reply - bid_only`,n=2):`village_win 0.00`,`failed 0.00`,`game_quality +0.05`,`reply_rate +0.07`,`vote_accuracy_good 0.00`,`good_final_wolf_suspicion_gap -0.04`,`deception_success_rate +0.02`,`shared_good_target_count -0.50`。
  - 解读:ABBA 协议工作,`bid_reply` 在这两对 pair 中确实提高 reply_rate,但胜率/投票命中不变,好人最终狼嫌疑差略低。n=2 只能说明工具可用和提供方向,不能证明机制优劣。
- **真实 smoke 补充样本**:
  - 在 router stats 修正后再次跑 `timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py`。
  - 结果:村民 day2 胜;真实模型调用 54 次,成功 54,失败 0,重试 0,`total_latency=217.214`,`avg_latency=4.022`,0 `agent_decision_failed`。
  - 关键社会现象:预言家 6 号首夜死亡,没有硬查验留存;1/3 双狼公开合谋用"4号太安静"制造焦点,2/4/5 通过重复论点和抱团带节奏识别出 3 号与 1 号,最终村民胜。这个样本说明 collusion/debate/posterior 指标在无查验硬信息局中也有价值。
- **router stats 语义修正**:
  - 问题:§3.34 的 `router_stats_delta` 对 `avg_latency` 直接做累计平均值相减,可能出现负数,不可解释。
  - 修复:`src/llm/router.py` 的 `CallStats.snapshot()` 新增累计 `total_latency`;`tests/multi_game_stats.py.router_stats_delta()` 用 `(after.total_latency-before.total_latency)/(after.calls-before.calls)` 计算窗口 `avg_latency`,不再伪造平均值差。
  - 汇总新增 `=== Router stats delta ===`;per-policy summary 和 ABBA paired delta 也输出 calls/retries/tokens/avg_latency,避免未来实验只看胜率和文本指标。
- **测试/验证**:
  - 编译:`python -m compileall -q src/llm/router.py tests/multi_game_stats.py tests/test_multi_game_stats.py` → 通过。
  - 窄测:`PYTHONPATH=. pytest -q tests/test_multi_game_stats.py tests/test_llm_router.py` → `37 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `110 passed, 3 skipped`。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10 --turn-policies bid_only,bid_reply --policy-schedule abba --experiment-seed 100 --experiment-id dry-abba` → 通过,输出 `=== Router stats delta === no metrics`,临时 JSONL 已删除。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - 真实 smoke:见上。
- 诚实 caveat:本轮是小批量实证和工具修正,不是机制显著性证明。下一步应跑更大 ABBA(至少 6-10 局/策略),再比较 `bid_reply` vs `bid_reply_caucus`,并把 router 成本/重试/parse_failed 与社会指标一起报告。

---

### 3.36 Turn-policy caucus ABBA 真实批量实验(✅,report/test/build/smoke 通过)
- 触发:§3.35 已跑 `bid_only` vs `bid_reply`,但还没隔离 day1 狼队 caucus。按 §4.2 路线继续真实实跑 `bid_reply` vs `bid_reply_caucus`,并响应用户要求继续联网调研"多 agent 对抗、欺骗、合谋、多 agent 社会"和开子代理并行。
- **真实实验**:
  - 命令:
    ```bash
    timeout --foreground -s INT -k 30s 180m python -u tests/multi_game_stats.py 6 \
      --jsonl logs/turn_policy_abba_caucus_20260705.jsonl \
      --bootstrap-iters 1000 \
      --turn-policies bid_reply,bid_reply_caucus \
      --policy-schedule abba \
      --experiment-seed 2026070502 \
      --experiment-id turn-policy-abba-caucus-20260705 \
      | tee logs/turn_policy_abba_caucus_20260705.out
    ```
  - 调度:6 个 ABBA pair,每个 pair 共享 `role_seed/actor_seed/orchestrator_seed`;总计 12 局真实 LLM 对局。
  - 产物:
    - JSONL:`logs/turn_policy_abba_caucus_20260705.jsonl`
    - 控制台摘要:`logs/turn_policy_abba_caucus_20260705.out`
    - 报告:`docs/experiment-turn-policy-caucus-abba-2026-07-05.md`
- **结果(诚实小样本)**:
  - 总体:12 局,村民 10 胜 / 狼人 2 胜;Wilson 95%CI `[55.2%,95.3%]`;全部 day2;`agent_decision_failed=0`;`game_ended` 异常 0。
  - 解析/路由:470 次决策中 1 次 `parse_failed`,失败率 0.2%;真实 router 629 calls / 629 successes / 0 failures / 1 retry / 2,460,674 input tokens / 214,013 output tokens / `total_latency=2648.76s` / `avg_latency=4.22s`。
  - per-policy:
    - `bid_reply`:5/6 村民胜,308 calls,1 retry,1,162,790 input tokens,`reply_rate=38.9%`,`deception_success_rate=56.7%`,`shared_good_target_count=0.83`,`wolf_to_wolf_support_count=0.17`,`coordinated_pressure_count=1.33`,`good_final_wolf_suspicion_gap=0.25`,`game_quality=0.72`。
    - `bid_reply_caucus`:5/6 村民胜,321 calls,0 retry,1,297,884 input tokens,`reply_rate=33.2%`,`deception_success_rate=55.7%`,`shared_good_target_count=1.00`,`wolf_to_wolf_support_count=1.33`,`coordinated_pressure_count=2.17`,`good_final_wolf_suspicion_gap=0.34`,`game_quality=0.74`。
  - paired delta(`bid_reply_caucus - bid_reply`,n=6):`village_win 0.00`,`failed 0.00`,`router_calls +2.17`,`router_tokens_in +22515.67`,`game_quality +0.03`,`reply_rate -0.06`,`vote_accuracy_good +0.03`,`good_final_wolf_suspicion_gap +0.09`,`deception_success_rate -0.01`,`shared_good_target_count +0.17`。
- **解释(谨慎)**:
  - day1 caucus 在这批样本中**没有改变胜率**:两组都是 5 村民胜 / 1 狼人胜。
  - caucus 提高了部分公开协同/成本指标:`wolf_to_wolf_support_count` 从 0.17 到 1.33,`coordinated_pressure_count` 从 1.33 到 2.17,`shared_good_target_count` 从 0.83 到 1.00,平均每局多约 2.17 次 router call 和 22.5k input tokens。
  - caucus 没有提高测得的 `deception_success_rate`,反而略低 1 个百分点;`reply_rate` 也下降约 6 个百分点。
  - `good_final_wolf_suspicion_gap` 在 caucus 下更高(+0.09)。合理推测是:弱 caucus 能制造更明显的公开协同压力,但也会把狼队站边/共打靶暴露给好人,所以最终后验可能更会指向狼。**这是本批实验推断,不是机制证明**。
- **联网/子代理调研增量(2026-07-05 核验)**:
  - Werewolf Arena(arXiv:2407.13943, https://arxiv.org/abs/2407.13943):动态 bidding / 发言权是 social deduction 评测核心,继续支持 `turn_policy` ablation。
  - Social Deduction MARL(arXiv:2502.06060, https://arxiv.org/abs/2502.06060):沟通质量应以 belief/world-state prediction 影响衡量,支持继续把 `posterior_shift/speech_to_vote` 放在主指标里。
  - OpenDeception(arXiv:2504.13707, https://arxiv.org/abs/2504.13707):欺骗要区分 deceptive intent 与 listener susceptibility,支持 speaker 自报、独立审计、listener shift 三段式。
  - MultiMind(arXiv:2504.18039, https://arxiv.org/abs/2504.18039):ToM/suspicion state 是隐藏身份推理关键,支持从 `attitudes` 升到稀疏二阶 ToM matrix。
  - GRAIL/Bayesian Social Deduction(arXiv:2506.17788, https://arxiv.org/abs/2506.17788):把隐藏身份后验外置到 graph/probabilistic model,LLM 负责语言互动,支持 EvidenceGraph/RolePosterior 主线。
  - COLOSSEUM(arXiv:2602.15198, https://arxiv.org/abs/2602.15198)+ Secret Collusion among AI Agents(OpenReview, https://openreview.net/pdf?id=bnNSQhZJ88):多 agent collusion 需要区分通信、行动与结果影响,支持 `collusion_audit` 从计数升级到 pair listener susceptibility + action/posterior impact。
- 诚实 caveat:这是 6 pair 诊断样本,不是显著性证明。真实 LLM seed 只能控制发牌/persona/orchestrator RNG,不能完全固定模型输出。后续应做四策略矩阵和跨模型/人数/策略版本分组;collusion audit v2 已在 §3.38 落地,下一步是 v3 语义审计与 likelihood_delta 对齐。

---

### 3.37 Experiment runner 成型化:resume JSONL + 扩展 ABBA delta(✅,test/build/smoke 通过)
- 触发:真实 ABBA 长跑已经开始成为项目主验证手段,但 `tests/multi_game_stats.py` 原先每次启动都会清空 JSONL,12-40 局真实 LLM 实验一旦中断只能重跑;同时 ABBA paired delta 仍只覆盖少数指标,不足以支持后续四策略矩阵和 collusion/deception 机制判断。子代理代码审查也把"可恢复实验 runner"列为最高杠杆项。
- **实现边界**:
  - 只改离线真实实验工具与单测,不改 `GameOrchestratorV2` 默认对局行为,不改 agent prompt,不改规则引擎。
  - `--resume-jsonl` 是显式开关;默认行为仍清空目标 JSONL,保证新实验产物干净。
  - resume 仅按当前 schedule 的 `game_id` 匹配已有 JSONL 行;不匹配当前 `experiment_id/game_id` 的旧行不会参与本次统计。
- **代码变更**:
  - `tests/multi_game_stats.py` 新增 `--resume-jsonl`:读取已有 JSONL,按 `game_id` 跳过当前 schedule 中已存在的对局,并把这些行纳入最终 summary;新跑出的对局继续 append。重复 `game_id` 采用 last-write-wins,符合 append-only crash recovery 语义;坏 JSON 行会 warning 后忽略。
  - 新增 `load_resume_jsonl()` / `resume_row_game_id()` helper,支持顶层 `game_id` 和嵌套 `experiment.game_id`。
  - ABBA paired delta 扩展到更完整主指标:
    - router:`calls/retries/tokens_in/tokens_out/total_latency/avg_latency`
    - parse:`parse_failed_count/parse_failed_rate`
    - dialogue/debate:`speech_count/reply_rate/accuse_rate/wolf_coordination/bid_entropy/top_accuse_target_share`
    - objective/posterior:`vote_accuracy_good/good_final_wolf_suspicion_gap/good_final_brier_score/good_final_log_loss/constrained_good_final_brier_score/constrained_good_final_log_loss/calibration_ece`
    - deception/collusion:`deception_success_rate/peer_detection_rate/villager_false_positive_rate/shared_good_target_count/wolf_to_wolf_support_count/coordinated_pressure_count/narrative_overlap_pair_count`
- **测试/验证**:
  - 编译:`python -m compileall -q tests/multi_game_stats.py tests/test_multi_game_stats.py` → 通过。
  - 窄测:`PYTHONPATH=. pytest -q tests/test_multi_game_stats.py` → `34 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `113 passed, 3 skipped`。
  - dry-run resume:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10 --turn-policies bid_only,bid_reply --policy-schedule abba --experiment-seed 100 --experiment-id dry-abba --resume-jsonl` → 通过,临时 JSONL 已删除。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - 真实 smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 狼人 day1 胜;真实模型调用 34 次,成功 34,失败 0,重试 0,`total_latency=134.364`,`avg_latency=3.952`,0 `agent_decision_failed`,进程退出码 0。过程中两次 `lossy_kv` 被 strict parser 拒绝后 actor 真实重试成功,无 fake fallback。
  - 凭据扫描:按用户提供密钥唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- 诚实 caveat:`--resume-jsonl` 只按 `game_id` 跳过已有行,不判断该行是否"健康完成"。失败/timeout/parse_failed 高的行会照样进入汇总并由 summary 暴露;如果要重跑坏行,应删掉对应 JSONL 行或换新的 `experiment_id`。这是实验工程能力升级,不是新的社会机制结论。

---

### 3.38 Collusion audit v2:pair listener susceptibility + deception alignment(✅,build/test/smoke 通过)
- 触发:§3.33 的 `collusion_audit` v1 已能检测公开频道狼互挺、共打好人、叙事重合和协同施压;§3.36 真实 ABBA 进一步显示 `bid_reply_caucus` 最明显的变化不是胜率,而是公开狼互挺和协同压力。为了把"多狼合谋"从计数推进到"对听众后验造成了什么影响",本轮落地 pair 级审计。
- **语义边界**:
  - 仍然是赛后 analysis-only。结果不进入 live prompt/memory,不广播实时事件,不影响 agent 决策。
  - 不读取狼队 caucus 私聊文本来判定合谋;只用赛后真值识别 wolf seats,再审计这些狼在公开频道中的结构化行为。
  - records / pair 明细只输出 seat、day、计数、后验增益、evidence ids、posterior delta ids 和 deception 类型分布;不保存 speech 原文、private reasoning、wolf caucus、隐藏事件。
- **后端**:
  - `GameOrchestratorV2._collusion_audit()` 新增 pair 级聚合:
    - 顶层:`pair_listener_shift_sample_count`,`avg_pair_target_suspicion_gain`,`pair_target_misdirected_rate`,`deception_linked_pair_count`,`pair_listener_susceptibility_by_pair`。
    - 每个 pair 输出:`wolf_seats/active_days/shared_good_target_count/wolf_to_wolf_support_count/mutual_support_pair_count/narrative_overlap_pair_count/coordinated_pressure_count/target_shift_sample_count/avg_target_suspicion_gain/target_misdirected_rate/colluder_shift_sample_count/avg_colluder_suspicion_gain/evidence_linked_count/deception_record_count/successful_deception_record_count/peer_detected_deception_record_count/audited_deception_types/evidence_ids/posterior_delta_ids`。
  - pair target swing 复用 `_posterior_speech_shift_groups()` 的 speech baseline 语义,只统计好人 viewer 在狼 pair 公开共打/接力后对目标好人的狼嫌疑变化。
  - deception alignment 复用 `_deception_audit_from_roles()` 的 records,按 pair 活跃 day / shared target 对齐,回答"这组公开合谋是否也伴随可审计欺骗记录"。
- **统计/前端**:
  - `tests/multi_game_stats.py` 的 `COLLUSION_AUDIT_KEYS` 加入 pair 顶层指标,单局进度行和 ABBA paired delta 同步输出 `avg_pair_target_suspicion_gain/pair_target_misdirected_rate/deception_linked_pair_count`。
  - 新增 `print_collusion_audit_block()` 和 `collusion_pair_susceptibility_values()`,多局汇总能打印 `pair_listener_susceptibility_by_pair` 的 total/mean/bootstrap CI。
  - `frontend/src/lib/types.ts` 新增 `CollusionPairSusceptibility` 类型;`GameStatusPanel`、`ReplayPanel`、`trends.ts` 展示/持久化 Pair样本、Pair疑增、Pair误导、欺骗对齐。
- **测试/验证**:
  - 单测扩展 `test_collusion_audit_detects_public_wolf_alignment_without_raw_text`:构造两狼同日共打一个好人、互相 support、叙事重叠、公开 evidence refs、posterior shift 和 wolf-declared deception,断言 pair 指标、deception 对齐、private evidence 过滤,并确认 JSON 序列化不含 speech 原文、reasoning、wolf_caucus。
  - `tests/test_multi_game_stats.py` 新增 pair susceptibility 聚合过滤测试,扩展 collusion summary 输出断言。
  - 编译:`python -m compileall -q src/game/orchestrator.py tests/multi_game_stats.py tests/test_multi_game_stats.py tests/test_orchestrator.py` → 通过。
  - 相关窄测:`PYTHONPATH=. pytest -q tests/test_orchestrator.py::test_collusion_audit_detects_public_wolf_alignment_without_raw_text tests/test_multi_game_stats.py::test_collusion_pair_susceptibility_values_filter_invalid_and_bool_values tests/test_multi_game_stats.py::test_print_summary_includes_collusion_audit_metrics` → `3 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `114 passed, 3 skipped`。
  - dry-run resume:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl logs/multi_game_stats_dryrun.jsonl --bootstrap-iters 10 --turn-policies bid_only,bid_reply --policy-schedule abba --experiment-seed 100 --experiment-id dry-abba --resume-jsonl` → 通过,临时 JSONL 已删除。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - 真实 smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 村民 day2 胜;真实模型调用 55 次,成功 55,失败 0,重试 1(stream 网络中断由 router 真实重试恢复),`total_latency=261.119`,`avg_latency=4.748`,0 `agent_decision_failed`,进程退出码 0。对局中 1号预言家连续查杀5号/3号双狼,3/5狼队使用"跳太急/帮腔太快/抱团带节奏"叙事保队友并转移焦点,正好覆盖 pair 合谋审计目标;好人最终识别并投出3号。
  - 凭据扫描:按用户提供密钥唯一片段全库扫描 → 0 命中,真实网关密钥未落盘。
- 诚实 caveat:v2 仍是结构化/规则审计,不是逐句自然语言事实核查。`pair_target_misdirected_rate` 依赖当前 EvidenceGraph posterior,不是人工标注真值;`deception_record_count` 说明 pair 合谋和 deception audit records 同场对齐,不证明私下串通意图。下一步应做 evidence-item 级 LLM/规则混合 auditor,标注 omitted counterevidence / distorted source,并与 likelihood_delta 后验校准合并。

---

### 3.39 No-fallback 边界硬化 + windowed relay / herding correctness(✅,build/test/smoke 通过)
- 触发:用户强调"一切必须真实调用,任何虚假的冒烟测试、兜底测试、兜底门户都拒绝",并要求多子代理并行。并行 3 路子代理完成:① no-fallback/真实调用链审计;② CLAUDE/docs/前端/统计缺口审计;③ 联网调研多 agent 欺骗/合谋/社会推理。主线先修真实决策边界,再落 analysis-only 指标与复盘证据链。
- **联网调研增量(已核验来源,转为项目指标)**:
  - WOLF(arXiv:2512.09187, https://arxiv.org/abs/2512.09187):逐发言欺骗审计要区分 `omission/distortion/fabrication/misdirection`,并结合 speaker intent、peer detection、suspicion trajectory。对应 werewolf-mas 的 `deception_audit.records/listener_shifts/peer_detection`。
  - Beyond Survival / WereBench & WereAlign(arXiv:2510.11389, https://arxiv.org/abs/2510.11389):胜率不足以解释社会推理,需策略/投票/身份推断对齐。对应 `objective_metrics`、WereAlign 5维和后续 strategy alignment。
  - The Traitors(arXiv:2505.12923, https://arxiv.org/abs/2505.12923):少数欺骗多数的 trust dynamics 应看 trust shift、betrayal success、collective inference quality。对应当前 posterior shift / trust graph / deception success。
  - MINDGAMES(arXiv:2605.29512, https://arxiv.org/abs/2605.29512):多 agent 竞技评估需完整轨迹、角色条件分组、错误归因,不能只报 win rate。对应 JSONL/ABBA/CI/invalid-failure 审计。
  - AvalonBench(arXiv:2310.05036, https://arxiv.org/abs/2310.05036)+ EMNLP Avalon society(https://aclanthology.org/2024.emnlp-main.7/):隐藏身份游戏应看 role-conditioned win rate、private-info leak、support/accuse/defend/coalition 社交图。对应现有 attitude/reply/accuse graph 与信息隔离测试。
  - Colosseum(arXiv:2602.15198, https://arxiv.org/abs/2602.15198)+ Double Auctions collusion(arXiv:2507.01413, https://arxiv.org/abs/2507.01413):合谋不只看文本声明,要看 action-level coordination、side-channel uplift、paper-action gap。对应本轮 windowed relay 与后续 side-channel/oversight ablation。
  - Debate truthfulness(arXiv:2402.06782, https://arxiv.org/abs/2402.06782)+ deception abilities(arXiv:2307.16513, https://arxiv.org/abs/2307.16513):辩论可能提升真相也可能只提升说服,需要 truth accuracy after debate、false-belief induction / ToM probe。对应后续 debate judge 与二阶 ToM matrix。
- **No-fallback 边界硬化**:
  - `AgentActor` 新增 `DECISION_MAX_ATTEMPTS=5` / `REFLECTION_MAX_ATTEMPTS=2` / 指数退避+抖动;`decide_night_action/decide_speak/decide_wolf_caucus/decide_vote/decide_last_words` 默认 5 次真实 LLM 尝试。最后一次仍只允许真实模型输出的透明 lossy parse,不会生成脚本决策。
  - `LLMRouter` 默认 `max_retries=5`;`config.LLM_MAX_RETRIES` 默认 5;`RoomManager()` 显式使用 `LLM_TIMEOUT/LLM_MAX_RETRIES/LLM_CONCURRENCY`;`smoke_e2e.py` 和 `multi_game_stats.py` 不再手写 3 次,统一走配置。
  - `score_game_quality()` 默认重试 5 次;失败仍返回 `None` 而不是伪造质量分。
  - `_sanitize_vote()` 移除 `obs.candidate_targets[0]` 脚本兜底:普通投票缺 `target_seat` 且无 `suspicion` 时返回透明 `SKIP(vote_target_unresolved)`;PK 投票只有 LLM 给出显式怀疑度时才在 PK 候选中修正,否则同样 SKIP。规则层可以记录无有效动作,不能替 agent 选目标。
  - 猎人开枪路径修复:原先循环先 `pop(0)` 会让 `RulesEngine.hunter_shoot()` 认为 hunter 不在 pending 中,再被 broad exception 静默吞成"不开枪"。现改为 peek,让规则函数消费 pending;`AgentDecisionError` 会 emit `agent_decision_failed(action=hunter_shot)` 后透明结算为不开枪并带 `skip_reason=hunter_decision_failed`;非预期异常不再吞掉,直接抛出让房间失败。
  - `create_room` 日志不再输出 key 片段、model、api_base,只记录 `llm_configured` 与 provider。
- **新 analysis-only 指标(不进入 live prompt/memory)**:
  - `collusion_audit` 新增窗口化狼队接力:`windowed_relay_count`,`avg_windowed_relay_latency`,`avg_relay_target_suspicion_gain`,`relay_target_misdirected_rate`;pair 明细同步输出这 4 项。判定只看公开频道 K=4 条发言内的"同指好人/跟随互挺",不读取 wolf caucus 文本。
  - `posterior_metrics` 新增 `herding_event_count`,`correct_herding_rate`,`wrong_herding_rate`,把原 `herding_index` 拆成正确共识与错误从众,避免把好人共同锁狼和被带偏共识混在一起。
  - `tests/multi_game_stats.py` 将新增指标纳入 summary、per-game progress 和 ABBA paired delta;ABBA delta 同时补齐 `declared_deception_count/audited_deception_count/misdirection_shift_coverage/listener_shift_sample_count/deception_evidence_linked_count`。
- **前端复盘证据链**:
  - `types.ts`/`trends.ts`/`GameStatusPanel`/`ReplayPanel` 接入 windowed relay 与 correct/wrong herding。
  - `ReplayPanel` 新增"审计证据链":只展示 day/seat/type/target/evidence id 数量/posterior delta 数量/listener shift 数量/relay latency 等结构化字段,不展示 raw speech、reasoning、wolf caucus。
- **测试/验证**:
  - 新增/扩展单测:真实决策入口默认 5 次;反思预算单独低;投票缺目标不暗选第一个候选;PK 无有效目标不暗选;RoomManager 使用 LLM runtime config;猎人决策失败 emit `agent_decision_failed`;collusion audit windowed relay;posterior correct/wrong herding;multi-game stats 新指标过滤 bool/NaN。
  - 编译:`python -m compileall -q src/agent/actor.py src/game/orchestrator.py src/api/room_manager.py src/api/server.py src/agent/quality.py tests/...` → 通过。
  - 窄测:`PYTHONPATH=. pytest -q tests/test_llm_router.py tests/test_room_manager.py tests/test_orchestrator.py tests/test_multi_game_stats.py` → `70 passed`;补充统计/actor 窄测 → `50 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `120 passed, 3 skipped`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - dry-run:`PYTHONPATH=. python tests/multi_game_stats.py 0 --jsonl /tmp/multi_game_stats_dryrun.jsonl --bootstrap-iters 10 --turn-policies bid_only,bid_reply --policy-schedule abba --experiment-seed 100 --experiment-id dry-abba --resume-jsonl` → 通过。
  - 真实 LLM smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 村民 day2 胜;真实模型调用 56 次,成功 56,失败 0,router 重试 0,`total_latency=235.511`,`avg_latency=4.206`,0 `agent_decision_failed`,进程退出码 0。过程中多次 `lossy_kv` 被 strict parser 拒绝,actor 走 `attempt=1/5...4/5` 后真实重试;一次第 5 次透明有损恢复并标记 parse_failed,未触发伪造/fallback。
  - 凭据扫描:按用户提供的密钥唯一片段全库扫描(排除 node_modules/dist) → 0 命中;密钥片段未写入文档。
- 诚实 caveat:windowed relay 仍是结构化行为审计,不是意图证明;correct/wrong herding 依赖当前 EvidenceGraph posterior 和赛后真值,只用于评估,不应用于 live agent。下一步若做 Colosseum 风格合谋 regret/side-channel 实验,必须仍保持真实 LLM 调用和信息隔离。

---

### 3.40 `/api/config` 凭据片段零泄漏(✅,build/test/smoke 通过)
- 触发:继续复核安全/no-fallback 铁律时发现 `/api/config` 虽然不返回完整 key,但仍会把默认 key 的前 6 位拼 `***` 下发给前端。用户已明确要求不泄露真实调用凭据,所以"只露前缀"也不合格。
- **后端**:
  - `src/api/server.py.get_config()` 改为 `api_key` 永远返回空字符串,新增 `api_key_configured: bool` 供前端判断"后端已配置,留空沿用"。
  - 这只改配置展示协议,不改 `DEFAULT_MODEL_CONFIG` 内存使用,不影响真实 LLM 调用。
- **前端**:
  - `frontend/src/lib/types.ts` 的 `ModelConfigDTO` 加 `api_key_configured?: boolean`。
  - `LobbyView` 的 API Key placeholder 改用 `defaultCfg.api_key_configured`,浏览器不再拿到任何 key 内容或片段。
- **测试/验证**:
  - `tests/test_api.py::test_config_hides_api_key` 改为 patch 一个假 secret,断言响应 `api_key == ""`, `api_key_configured is True`,且 JSON 中不含完整 secret 或 secret 前缀。
  - `PYTHONPATH=. pytest -q tests/test_api.py` → `8 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `120 passed, 3 skipped`。
  - `cd frontend && npm run build` → 通过(tsc + vite,44 modules)。
  - 真实 LLM smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 村民 day2 胜;真实模型调用 51 次,成功 51,失败 0,router 重试 0,`total_latency=235.568`,`avg_latency=4.619`,0 `agent_decision_failed`,进程退出码 0。
  - 凭据扫描:按用户提供的密钥唯一片段全库扫描(排除 node_modules/dist) → 0 命中;密钥片段未写入文档。
- 诚实 caveat:前端创建房间时用户手工输入的 per-room key 仍会通过 HTTPS/本地 API 请求体进入后端内存,这是功能需要;但后端默认 key 不再通过 `/api/config` 反向暴露给浏览器。

---

### 3.41 Orchestrator per-decision timeout + 真实 shadcn/ui 前端基线(✅,build/test/browser/smoke 通过)
- 触发:用户要求继续当前稳定性工作,同时明确前端 UI 必须使用真实 shadcn/ui 组件库,不得手写假 shadcn。并行 2 路子代理只读审计:① 前端真实栈/shadcn 缺口;② orchestrator 决策 timeout 调用点。主线完成实现、集成和验证。
- **后端稳定性 / no-fallback**:
  - `src/config.py` 新增 `AGENT_DECISION_TIMEOUT`(默认 240s)与 `AGENT_DECISION_TIMEOUT_BY_PHASE`,支持 `WEREWOLF_AGENT_DECISION_TIMEOUT_{NIGHT,DAY,VOTING,PK,LAST_WORDS,HUNTER,REFLECTION}`。
  - `GameOrchestratorV2` 新增 `decision_timeout/decision_timeouts` 构造参数与 `_with_decision_timeout()`。所有真实 `actor.decide_*` 调用点(夜间行动、狼人夜杀提案、白天 caucus、白天发言、bid/speak、投票、PK 发言、遗言、猎人开枪、reflection)都在 harness 层包 wall-clock timeout。
  - 超时只转成 `AgentDecisionError`,复用既有透明失败路径:不补默认发言、不补默认票、不补默认夜间目标、不替猎人选枪。投票超时会导致 `vote_incomplete`;猎人超时/失败是透明不开枪并带 `skip_reason`,不是假目标。
  - `RoomManager` 显式把 `AGENT_DECISION_TIMEOUT` 传给 orchestrator;`hunter_shot` 失败事件补 `phase: hunter` 和 `reason`。
  - `GameOrchestratorV2._decision_failure_metrics()` 进入赛后 `analysis.decision_failure_metrics`,聚合 `failure_count/timeout_count/by_phase/by_action/by_seat/records`。它只记录 seat/day/phase/action/reason 截断/timeout,不记录 speech 原文、private reasoning、wolf caucus 或角色真值;超时不计入 `parse_metrics`。
- **后端测试**:
  - `tests/test_orchestrator.py` 新增发言超时和投票超时回归:断言 emit `agent_decision_failed`,不出现超时 seat 的 speech/vote_cast,不提交假票,并能统计 timeout metrics。
  - 全量测试当前基线更新为 `PYTHONPATH=. pytest -q` → `122 passed, 3 skipped`。
- **真实 shadcn/ui 接入**:
  - 用真实 CLI 初始化:`npx shadcn@latest init -t vite -b radix --preset nova -y`。CLI preflight 先拒绝缺 Tailwind/alias,补齐后才成功;没有手写假组件。
  - 前端基础设施:
    - 安装真实依赖:`tailwindcss`, `@tailwindcss/vite`, `radix-ui`, `lucide-react`, `class-variance-authority`, `clsx`, `tailwind-merge`, `tw-animate-css`, `@fontsource-variable/geist` 等。
    - 新增真实 `components.json`(`style: radix-nova`, `baseColor: neutral`, `iconLibrary: lucide`)。
    - `vite.config.ts` 接入 `@tailwindcss/vite` 与 `@` alias;`tsconfig.json` 配置 `baseUrl/paths`;`main.tsx` 注入 `.dark` 和 `TooltipProvider`。
    - shadcn CLI 真实生成 `src/components/ui/{button,card,input,label,select,checkbox,dialog,alert,badge,tabs,scroll-area,separator,progress,tooltip,textarea}.tsx` 与 `src/lib/utils.ts`。
  - 业务 UI 迁移:
    - `LobbyView` 改用 shadcn `Card/Button/Input/Select/Checkbox/Label/Alert/Tooltip`,并保留"只发送非空覆盖字段,空 API key 不覆盖默认"语义。
    - `ModelConfigModal` 改用 shadcn `Dialog/Input/Select/Checkbox/Label/Button`,保留"留空继承"语义。
    - `RoomView` 改用 shadcn `Card/Button/Badge/Alert/Tooltip`,模式入口与座位配置按钮不再是原生 button。
    - `App/ChatRoom/HumanActionPanel/GameStatusPanel/RoomSidebar/PhaseHud/GodPanel/VotePanel/SeatGrid/ReplayPanel` 的业务按钮、输入、textarea 和 root card 容器迁到真实 shadcn 组件。
    - 扫描验证:`rg "<button|<input|<select|<textarea|<details|<div className=\"card"` 在业务源码 0 命中;仅 shadcn 自身 `ui/input.tsx` / `ui/textarea.tsx` 内部存在原生 input/textarea,这是组件库实现而非假 UI。
- **浏览器/接口验证**:
  - `agent-browser` 真实打开 `http://127.0.0.1:5173/`,大厅可访问树显示 Button/Input/Select/Checkbox;人数切换真实更新座位数;截图保存 `/tmp/werewolf-shadcn-lobby.png`。
  - 启动/复用本地后端 `127.0.0.1:8000`,真实点击"创建房间"进入 RoomView;打开单座模型配置 shadcn Dialog,截图保存 `/tmp/werewolf-shadcn-room.png`。
  - 浏览器检查时发现 8000 上旧后端进程仍返回 masked `api_key` 片段;源码已是 §3.40 安全版本,重启旧进程后重新验证 `/api/config` → `api_key:""`, `api_key_configured:true`。这次问题是旧进程未重启,不是源码回归,但已作为安全回归流程记录。
- **验证**:
  - 编译:`python -m compileall -q src/game/orchestrator.py src/config.py src/api/room_manager.py tests/test_orchestrator.py` → 通过。
  - Orchestrator 窄测:`PYTHONPATH=. pytest -q tests/test_orchestrator.py` → `22 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `122 passed, 3 skipped`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,1915 modules)。
  - 真实 LLM smoke:`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 村民 day2 胜;真实模型调用 51 次,成功 51,失败 0,router 重试 0,`total_latency=333.713`,`avg_latency=6.543`,0 `agent_decision_failed`,进程退出码 0。过程中真实触发一次有损 JSON 严格拒绝并由 actor 重试恢复,未触发伪造/fallback。
  - 凭据扫描:按用户提供密钥唯一片段扫描(仅输出文件名,排除 node_modules/dist/pyc) → 0 命中;密钥片段未写入代码或文档。
- 诚实 caveat:本轮完成的是 shadcn 真实组件库基线和主要业务控件迁移,不是最终视觉设计定稿;旧全局 CSS 仍大量承担游戏化布局/颜色/面板密度。后续若继续"前端搞好",应基于 shadcn 组件继续做视觉密度、移动端布局、指标面板信息层级和更完整的设计系统收敛,但不得回退到假组件或原生控件。

---

### 3.42 普通异常透明审计 + shadcn 细化修正(✅,build/test/browser/smoke 通过)
- 触发:用户继续要求开子代理加速,并强调"一切必须真实调用、不得假冒烟/假兜底"。本轮并行 2 路只读子代理:① orchestrator `asyncio.gather(..., return_exceptions=True)` 普通异常审计;② 前端 shadcn 剩余风险审计。主线负责后端 no-fallback 硬化、实际 UI 迁移和验证。
- **后端 no-fallback 硬化**:
  - 新增 `GameOrchestratorV2._agent_decision_failure_event()`:统一构造 `agent_decision_failed` 事件,带 `seat/phase/action/error_type/reason/timeout/timeout_seconds`。`AgentDecisionError` 保留 actor 层已净化原因;普通 `Exception` 只公开 `RuntimeError during phase/action` 这类类型/位置,原始异常详情只进服务端日志,避免把 provider/request/secret 细节带到前端或 analysis。
  - `_with_decision_timeout()` 给超时错误挂 `timeout=True` 与 `timeout_seconds`,赛后指标无需靠字符串猜。
  - 所有 `gather(return_exceptions=True)` 结果消费点补普通 `Exception` 分支:夜间角色行动、狼人夜杀提案、女巫救/毒、白天狼队 caucus、bid/speak 收集、投票、reflection。行为统一为**透明失败并跳过该 agent 本次输出**,不补发言、不补票、不补夜间目标、不生成党团共识。
  - 白天狼队 caucus 失败从 `_failed_events` 缓冲改为即时 `_emit`,避免"白天失败等下一夜才 flush"或日间结束时永久丢失。
  - 顺序 actor 调用也补普通异常审计:首轮/固定顺序发言、PK 发言、遗言、猎人开枪。遗言失败只出失败事件并移出遗言队列,不生成"(无遗言)"假文本;猎人决策异常按既有透明语义不开枪并带 `skip_reason=hunter_decision_failed`,规则引擎自身异常仍抛出,不把内部 bug 当 agent 失败吞掉。
  - `decision_failure_metrics` 新增 `by_error_type`,records 记录 `error_type`。仍不记录 speech 原文、private reasoning、wolf caucus 或角色真值。
- **后端测试**:
  - `tests/test_orchestrator.py` 新增普通异常矩阵:夜间预言家行动 `RuntimeError` 不提交行动且 emit;白天 wolf caucus `ValueError` 即时 emit 且不进 `_failed_events`;bid/speak 普通异常不进入排序、不留下 `_pending_speak_decision`;投票普通异常 emit 后保留 `vote_incomplete`;reflection 普通异常不隐藏其他 seat 的 reflection update。
  - 当时全量测试基线:`PYTHONPATH=. pytest -q` → `127 passed, 3 skipped`。
- **真实 shadcn/ui 细化**:
  - 两个前端子代理一致确认:真实 `components.json`/shadcn CLI 组件存在,业务层已大量使用真实 `Button/Input/Card/Dialog/Select/Tabs/Badge/Progress` 等;剩余最高风险是 `VotePanel` 的 `<div onClick>` 伪按钮、投票条/质量条等自绘 progress、泛用 chip/badge。
  - `VotePanel` 的可点击票型行改为真实 shadcn `Button variant="ghost"` + `aria-pressed`,内部票数条改为真实 shadcn `Progress`;CSS 只负责业务布局/颜色,不再自造 button/progress 结构。
  - 实际挂载在页面里的 `ChatRoom` 投票 tab 自绘 `votetab__bar-wrap/votetab__bar` 改为真实 shadcn `Progress`,保留现有紧凑投票布局。
  - `Button`/`Badge` 真实 shadcn 封装补 `React.forwardRef`,修复 React 18 + Radix `TooltipTrigger asChild` 的真实浏览器警告"Function components cannot be given refs"。这是兼容性修正,不改变组件 API,也不是自造假组件。
  - `types.ts` 的实时 `agent_decision_failed` 与 `DecisionFailureMetrics.records` 补 `error_type` 类型。
- **浏览器/真实调用验证**:
  - `agent-browser` 新 session 打开 `http://127.0.0.1:5173/`;控制台仅有 Vite/React DevTools 信息,不再出现 Radix ref 警告。
  - 浏览器 `scrollintoview + click` 真实点击"创建房间"成功进入 RoomView(房间 `ed8a8e2ddbc4`);说明 shadcn Button 可访问树与点击路径可用。此前不滚动直接点离屏 ref 不触发是 browser 坐标问题,非前端逻辑问题。
  - 通过真实后端接口启动并用浏览器观战两局 UI 房间:`5708135c29cb` 与 `ed8a8e2ddbc4` 均完整结束,村民 day2 胜,各 1 个 `game_ended` + 1 个 `analysis` + 0 个 `agent_decision_failed`。这两局使用的是当时已运行的 uvicorn 进程,用于验证 UI/WS/真实调用链;后端新异常审计逻辑由当前源码 pytest + smoke 验证。
  - `/api/config` 复查仍返回 `api_key:""` 与 `api_key_configured:true`,不下发 key 片段。
  - 凭据片段扫描(排除 `frontend/node_modules`, `frontend/dist`, `__pycache__`, `*.pyc`) → 0 命中。
- **验证**:
  - 编译:`python -m py_compile src/game/orchestrator.py tests/test_orchestrator.py` → 通过。
  - Orchestrator 窄测:`PYTHONPATH=. pytest -q tests/test_orchestrator.py -q` → 通过。
  - 全测:`PYTHONPATH=. pytest -q` → `127 passed, 3 skipped`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,1916 modules)。
  - 真实 LLM smoke(当前源码):`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 狼人 day2 胜;真实模型调用 38 次,成功 38,失败 0,router 重试 0,`total_latency=124.146`,`avg_latency=3.267`,进程退出码 0。过程中 5号 reflection 第一次 JSON 用中文引号被 strict parser 拒绝,actor 按真实重试恢复;day1 投票 4/5 不完整时系统透明跳过结算进入夜晚,没有补假票或伪造投票。
- 诚实 caveat:本轮修复了普通异常透明审计,但当时还不是全局 phase deadline;该缺口已在 §3.43 继续补齐。前端仍有泛用 chip/badge、质量条、后验条、ScrollArea 等低中优先级 shadcn 收敛项,但不能用假组件解决。

### 3.43 共享 phase deadline + ChatRoom 真 TabsContent + shadcn Badge 收敛(✅,build/test/browser 通过)
- 触发:继续按用户要求"前后端都好好做",并响应"开子代理加速"。先关闭已完成旧子代理释放额度,再并行 2 路只读子代理:① phase deadline 配置链路审计;② 当前前端 shadcn 剩余点审计。主线修后端入口、做真实浏览器验证和小范围 UI 收敛。
- **后端 phase deadline / no-fallback**:
  - `src/config.py` 已有 `AGENT_PHASE_DEADLINE` 与 `AGENT_PHASE_DEADLINE_BY_PHASE`,支持 `WEREWOLF_AGENT_PHASE_DEADLINE_{NIGHT,DAY,VOTING,PK,LAST_WORDS,HUNTER,REFLECTION}`;默认 `0` 表示关闭。
  - `GameOrchestratorV2` 已支持 `phase_deadline/phase_deadlines`、`_start_phase_deadline()`、`_phase_deadline_error()` 与 `_with_decision_timeout(..., phase_deadline=...)`。同一阶段共享 wall-clock budget;预算耗尽后尚未开始的 seat 会关闭未启动 awaitable 并产生 `AgentDecisionError(error_type=PhaseDeadlineExceeded, phase_deadline_exhausted=True)`,由既有透明失败路径 emit `agent_decision_failed`。
  - phase deadline 已接到夜间角色行动、狼队夜杀/白天 caucus、女巫救/毒、白天发言与 bid/speak、投票、PK、遗言、猎人、reflection。行为仍是失败透明记录并跳过本次真实输出,不补发言、不补票、不补夜间目标。
  - 本轮修复 API 入口缺口:`RoomManager.start_game()` 现在同时传 `phase_deadline=AGENT_PHASE_DEADLINE` 和 `phase_deadlines=AGENT_PHASE_DEADLINE_BY_PHASE`。此前只传全局值会让 `WEREWOLF_AGENT_PHASE_DEADLINE_DAY` 这类 per-phase env 在真实房间路径被吞掉。
- **后端测试**:
  - `tests/test_orchestrator.py::test_day_phase_deadline_marks_not_started_seats_without_fake_speech` 覆盖 day 共享 deadline:第一个 actor 启动后 deadline during decision,其余存活 seat before decision start;所有 seat 都有 `agent_decision_failed`, `error_type=PhaseDeadlineExceeded`,且未开始 seat 的 `decide_speak` 没被调用,没有伪造 speech。
  - `tests/test_room_manager.py::test_start_game_passes_per_phase_deadlines_to_orchestrator` 新增 API 链路回归:patch per-phase map 后启动房间,断言 orchestrator 保留 `day=0.25`、`voting=0.5` 覆盖值,不调用真实 LLM。
  - 窄测:`python -m py_compile src/api/room_manager.py tests/test_room_manager.py` → 通过;`PYTHONPATH=. pytest -q tests/test_room_manager.py tests/test_orchestrator.py::test_day_phase_deadline_marks_not_started_seats_without_fake_speech -q` → 通过。
- **真实 shadcn/ui 继续收敛**:
  - `ChatRoom` 的实际挂载 tab 结构迁到真实 `TabsContent`,5 个内容区(`chat/thinking/vote/record/identity`)都由 `Tabs` 包裹,不再靠外部手写条件渲染。CSS 改用 Radix/shadcn active data attribute;`.chatroom__body[hidden]` 明确隐藏非 active content。
  - 子代理复核确认 `VotePanel` 与 `ChatRoom` 当前已用真实 `Button/Progress/TabsContent`,本轮不重复改同一块。
  - 按最低风险路线把 `GameStatusPanel` 的泛用审计/校准 chip 从 `<span className="chip chip--deception">` 迁到真实 shadcn `Badge variant="outline"`,保留业务 class 只做颜色/密度。涉及 parse failure、decision failure、deception audit type、posterior calibration bins、posterior trace summary 等展示点。
- **浏览器/接口验证**:
  - `agent-browser` 新 session 真实打开 `http://127.0.0.1:5173/`,通过真实 UI 点击"创建房间",进入房间 `9a6d4bf70296`,再点击"观战模式"进入 `GameView`。
  - 可访问树显示真实 Radix tab 语义:`tablist "聊天思考流投票记录身份"`、5 个 `tab`、当前 `tabpanel`。点击"投票"/"记录"/"身份"均能切换 selected tab 与 visible panel。
  - DOM eval 验证 `document.querySelectorAll('[data-slot="tabs-content"]').length === 5`;非 active content 为 `hidden/display:none`,active content 为 `display:block`;API 请求无 4xx/5xx。
  - `/api/config` 复查仍返回 `api_key:""` 与 `api_key_configured:true`,不下发 key 片段。
  - 凭据片段扫描(排除 `frontend/node_modules`, `frontend/dist`, `__pycache__`, `*.pyc`) → 0 命中。
- **验证**:
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,1916 modules)。
  - 全测:`PYTHONPATH=. pytest -q` → `129 passed, 3 skipped`。
  - 真实 LLM smoke(当前源码):`timeout --foreground -s INT -k 30s 16m python -u tests/smoke_e2e.py` → 村民 day2 胜;真实模型调用 52 次,成功 52,失败 0,router 重试 0,`total_latency=220.284`,`avg_latency=4.236`,进程退出码 0。过程中 actor 层两次有损 JSON 被 strict parser 拒绝后按真实重试恢复;没有补假发言、假投票或假夜间目标。
- 诚实 caveat:phase deadline 是"每个 orchestrator 子阶段/收集点共享预算",不是把整天 day+voting+pk 合并成一个大预算;当前 `_run_voting()` 每次投票收集(含 PK 重投)单独开启 voting deadline,遗言/猎人追加死亡链也会重新开启对应子阶段 deadline。前端仍有 `SpeechFeed` 泛用 chip、质量/趋势/后验条和部分滚动区可继续 shadcn 化,但不得新建假 `Chip/ProgressBar`。

### 3.44 参考图 UI 成型 + P0 安全边界修复(✅,build/test/browser 通过)
- 触发:用户提供当前目录图片 `c3f65f3f-0275-410a-a2ed-e6cd5f7ecc7f.png`,要求按图中深色狼人杀圆桌 HUD 风格美化 UI,并再次强调必须使用真实 shadcn/ui 组件库,不得手搓假组件。同时后端子代理审计发现 P0 静态文件路径穿越与 replay/REST 信息边界缺口,本轮先修安全再做浏览器 UI 验证。
- **参考图设计读取**:
  - 图片风格:深色全屏仪表盘、顶部 HUD、左侧游戏进度/身份分布/存活人数、中央圆桌座位环 + 中央记录面板、右侧阶段信息/事件日志/胜利条件。
  - 本项目落地选择:不做营销页,不造假组件;在现有三栏应用中把中栏改成"圆桌态势主面板 + 下方实况交锋",左/右栏保持密集但降噪的工具面板。6/8/12 人圆桌半径响应式,避免座位裁切/文字重叠。
- **真实 shadcn/ui 前端实现**:
  - `GameView` 接入 `SeatGrid`,中栏从单一聊天流升级为 `SeatGrid + ChatRoom`,打开游戏第一屏即可看到 agent 圆桌对抗态势。
  - `SeatGrid` 继续使用真实 shadcn `Card`,并把圆桌身份标签迁到真实 `Badge variant="outline"`;CSS 只做业务视觉,不自造 Badge。
  - `RoomSidebar` 把房间头、阶段状态、玩家列表外层迁到真实 `Card`;房间模式/连接状态/身份标签迁到真实 `Badge`;阶段进度条迁到真实 `Progress`,删除手写 `roomsb__progress-bar` DOM 与动画。
  - `SpeechFeed` 的日期/狼人欺骗标记迁到真实 `Badge`;`ReplayPanel` 后验世界/证据 chips 迁到真实 `Badge`;`ChatRoom` 的身份 pill 迁到真实 `Badge`。
  - `GameStatusPanel` 顶部核心卡改用真实 `CardHeader/CardTitle/CardContent`,赛后分析顺序调整为欺骗审计/合谋审计/后验轨迹优先,再显示质量/解析/客观指标/LLM 统计,避免"指标墙"盖过 agent 对抗。
  - 全局视觉改为更接近参考图的深色 HUD:8px 圆角、半透明深蓝卡片、细边框、少量蓝/红/金强调、中央圆桌环形纹理与玩家卡。
- **后端安全/隐私修复**:
  - `src/api/server.py` 的 SPA fallback 新增 `resolve()+relative_to()` 约束,静态文件只能从 `frontend/dist` 下发;编码路径 `/%2e%2e/%2e%2e/.env` 现在只回 SPA index,不返回 dist 外文件。
  - 普通 `GET /api/rooms/{id}` 在 running 阶段不再返回 `players[*].role`;只有 ended/replay 语义才公开角色。
  - `/api/rooms/{id}/replay` 改为仅 `room.status == "ended"` 可访问;running 中返回 409。WS `mode=replay` 也仅赛后允许,否则关闭 4409,避免 running 中补看 `wolf_caucus`/thinking/roles。
  - `RoomManager._should_receive()` 修 play 私有事件:按 `recipients` 中的 player.id/seat 下发给对应 play seat,spectate 和其他 seat 收不到。
  - `RoomManager._run_room()` 普通未处理异常的公开 `room.error/game_error` 改为 `RuntimeError during game loop` 这类脱敏消息,原始异常只进日志。
  - `GameOrchestratorV2._agent_decision_failure_event()` 对非 timeout/deadline 的 `AgentDecisionError` 也做公开脱敏,避免 provider body/原始模型输出/隐藏片段进入 `agent_decision_failed.reason`;timeout/phase deadline 仍保留结构化时间信息。
  - `build_actors()` 改为 `model_config.merge(seat_override)`,单座模型配置只覆盖显式字段,空字段继承房间默认 `api_base/api_key`。`ModelConfig.merge()` 改用 `model_fields_set`,避免未显式填写的 `temperature/use_json_format` 被 Pydantic 默认值误覆盖。`POST /api/rooms` 允许 `temperature=0` 显式覆盖。
- **测试/验证**:
  - 新增/更新测试覆盖:静态路径穿越不泄露临时 sentinel、running 房间详情不泄露角色、running replay REST 返回 409、WS replay running 拒绝 4409、play 私有事件正反过滤、per-seat config 继承默认 key/base、公开 `AgentDecisionError` 脱敏、room error 脱敏、`temperature=0` 协议一致。
  - 编译:`python -m py_compile src/api/server.py src/api/room_manager.py src/game/orchestrator.py src/llm/models.py tests/...` → 通过。
  - 安全窄测:`PYTHONPATH=. pytest -q tests/test_api.py tests/test_room_manager.py tests/test_websocket_events.py tests/test_orchestrator.py::test_agent_decision_error_public_reason_is_sanitized -q` → `26 passed`。
  - 全测:`PYTHONPATH=. pytest -q` → `139 passed, 3 skipped`。
  - 前端:`cd frontend && npm run build` → 通过(tsc + vite,1917 modules)。
  - 浏览器(agent-browser):真实打开 `http://127.0.0.1:5173/`,真实 UI 创建房间 `46c07c0a0176`,进入观战;DOM 验证当前游戏页有真实 shadcn `Card/Badge/Progress/TabsContent` 挂载(`cards=8,badges=16,progress=1,tabs=5`);截图 `/tmp/werewolf-refstyle-game-waiting-final.png`。
  - 后端当前源码重启后 `/api/config` 仍返回 `api_key:""` 与 `api_key_configured:true`;静态穿越哨兵请求不返回 `.env`;凭据片段扫描(排除 `frontend/node_modules`, `frontend/dist`, `__pycache__`, `*.pyc`) → 0 命中。
- 诚实 caveat:本轮浏览器验证使用 waiting 房间验证 UI/组件/安全边界,没有再跑完整真实 LLM smoke;真实 LLM 链路上轮 §3.43 已跑通过,本轮后端改动由全量 pytest 和安全窄测覆盖。该 caveat 已在 §3.45 通过新 UI 后的真实 LLM smoke 补齐;仍建议继续补赛后 ended 页面浏览器截图,尤其是欺骗/合谋/后验卡的视觉层级。

### 3.45 shadcn 圆桌观战台定稿 + UI 开局闭环 + 人机 action 协议修复(✅,build/test/browser/smoke 通过)
- 触发:继续按用户要求"前后端都好好做",并强调前端必须真实使用 shadcn/ui 组件库、参考当前目录图片完成暗色圆桌 HUD、网页端必须能正常开始/观看 agent 对抗。并行子代理覆盖 UI/shadcn 合规、参考图移动端审计、前后端协议可用性;主线完成实现、验证和文档更新。
- **真实 shadcn/ui 继续收敛**:
  - 通过真实 CLI 安装 `accordion` 与 `avatar`:`cd frontend && npx shadcn@latest add accordion avatar --yes`;`npx shadcn@latest info --json` 确认已安装组件包含 `accordion/avatar/badge/button/card/checkbox/dialog/input/label/progress/scroll-area/select/separator/tabs/textarea/tooltip`。
  - `GameView` 新增顶部 `GameHud`,使用真实 shadcn `Badge/Button` 和 lucide `Clipboard/LogOut`,展示房间号、复制、day/phase/winner、连接状态、存活人数、模式与离开按钮。
  - `SeatGrid` 使用真实 `CardHeader/CardContent/CardTitle`、`Avatar/AvatarFallback/AvatarBadge`、`Badge`;座位环半径拆成 `radiusX/radiusY`,6 人局 `radiusY=43`,避免底部座位裁切。
  - `ChatRoom` 的思考流迁到真实 `Accordion/AccordionItem/AccordionTrigger/AccordionContent`,不再手写展开状态;已有真实 `Tabs/TabsContent/Button/Input/Progress/Badge` 保持。
  - `ReplayPanel` 的 root/empty state、角色摘要、座位结果、agent 总结和 `ReplayMetric` 继续收敛到真实 `Card/CardHeader/CardContent/CardTitle/CardDescription/Badge/Progress`。
  - `global.css` 按参考图补暗色 HUD、中央圆桌舞台、桌面 overlay 聊天区、移动端自然流布局、`room__startbar`、顶部栏移动端截断和 `spin` 动画。CSS 只负责布局/主题,不新造假 shadcn 组件。
- **前后端开局与人机协议闭环**:
  - `RoomView` 新增真实 "开始真实对局" 按钮,调用 `POST /api/rooms/{roomId}/start`,waiting 房间不再只能从外部接口启动。
  - `RoomInfo` 与后端 `create/get/start/replay` 响应都包含 `human_seats`,前端人机入口使用首个真实 human seat,不再硬编码 1 号。
  - `App.enterGame()` 进入游戏时派发 `{type:"__context__", mySeat, mode}`;`store.snapshot` 也会从 `view.self.seat` 回填 `state.mySeat`,修复 play 模式 `mySeat` 长期为 `null` 的问题。
  - 人类夜间/猎人动作不再发送模糊 `night_action`;`GameStatusPanel.HumanActionHint` 根据后端请求上下文发送真实协议动作:`night_kill/see/save/poison/guard/hunter_shot/vote`。女巫救人使用 `context.killed_seat`,目标过滤按动作区分。
  - `AgentActor.decide_night_action()` 新增 `requested_action` 与 `human_context`,人类 actor 请求中明确包含 `phase/day/role/requested_action/context`;解析层接受 `kill -> night_kill`、`hunter_shot -> NIGHT_KILL` 等后端合法动作。
  - `GameOrchestratorV2` 在夜间角色行动、狼人夜杀、女巫救/毒、猎人开枪处向 actor 传明确 `requested_action`,避免前端暗猜接口。
- **浏览器与真实调用验证**:
  - 前端构建:`cd frontend && npm run build` → 通过(tsc + Vite,1919 modules)。
  - Python 全测:`PYTHONPATH=. pytest -q` → `139 passed, 3 skipped`。
  - 编译:`python -m py_compile src/api/server.py src/agent/actor.py src/game/orchestrator.py` → 通过。
  - 真实 LLM smoke:`PYTHONPATH=. python tests/smoke_e2e.py` → 完整跑完;村民胜;真实模型调用 `59`,成功 `59`,失败 `0`,router 重试 `0`,`total_tokens_in=260931`,`total_tokens_out=22108`,`avg_latency=4.476`,事件 `80`,思考摘要 `22`,0 `agent_decision_failed`。对局中出现真实欺骗/反制:1号/5号狼用"6号发言太急"叙事带节奏,6号反抓"我还没正式发言过,你们怎么知道我急?",day1 PK 出 5 号狼,day2 好人抓 5 号遗言保 1 号矛盾并放逐 1 号狼。
  - agent-browser 真实打开 Vite 页面,从大厅点击创建房间进入 waiting 房间,再进入观战;桌面截图 `/tmp/werewolf-ui-1365x768-final.png`,移动端截图 `/tmp/werewolf-ui-mobile.png`。桌面中心舞台高度约 `588`,聊天 overlay 高度约 `415`,6 个座位无裁切;移动端圆桌/聊天自然纵向滚动,不再塌缩。
  - 安全复查:`/api/config` 返回 `api_key:""` 与 `api_key_configured:true`;编码穿越 `/%2e%2e/%2e%2e/.env` 不返回 `.env`;用户提供密钥片段扫描(排除 `frontend/node_modules`, `frontend/dist`, `__pycache__`, `*.pyc`) → 0 命中。
- 诚实 caveat:本轮 CLI 真实 smoke 已证明新代码链路不破坏真实 LLM 对局;浏览器已验证新 UI 创建/进入/布局,但还没用浏览器从新 "开始真实对局" 按钮一路等待到 ended 并截图赛后复盘。下一步应补 ended 浏览器截图和回放/欺骗/合谋/后验面板层级收敛。

### 3.46 requested_action 真闭环 + 开源启动文档 + 主屏对抗元数据(✅,build/test/browser/smoke 通过)
- 触发:继续按用户要求把项目做成型,并根据并行子代理审计修复 P0/P1:AI 猎人 `requested_action="hunter_shot"` 被夜间清洗吞成 `SKIP`;human 夜间请求仍可能要求前端重复传 action;README/ARCHITECTURE 仍描述旧无构建前端;waiting 观战页可能卡住;主聊天没有直接展示 reply/accuse/attitude/objective summary。
- **后端协议/no-fallback 修复**:
  - `AgentActor.decide_night_action()` 现在把 `requested_action` 传入 observation `available_actions`、prompt 和清洗函数;prompt 对 `hunter_shot/save/poison/guard/see/night_kill` 给出具体动作指令,不再让猎人收到"本夜无行动"这类矛盾提示。
  - `_sanitize_night(... requested_action=...)` 可按明确动作清洗真实 LLM 输出。`hunter_shot` 映射为 `AgentAction.NIGHT_KILL`,让猎人真实目标可落地;目标非法仍透明 `SKIP(night_target_unresolved)`,不补假目标。
  - `_sanitize_witch(... requested_action=...)` 按 save/poison 阶段约束输出:save 阶段收到 poison-only 形状时返回 `SKIP(requested_action_mismatch)`,不静默把真实意图吞掉;poison 阶段同理。
  - human actor 夜间等待请求的 `action_type` 改为具体 `requested_action` 而不是统一 `night_action`;前端只提交 `{target_seat}` 时后端也能按请求动作解析,不再要求暗传重复字段。
  - human action `target_seat` 非法字符串改为透明 `SKIP(human_action_invalid_target)`,不把前端输入错误混成未处理异常。
  - `POST /api/rooms` 校验 `human_seats` 必须在 `1..player_count`;`deck` 字段暂不支持时明确 400,避免"接口接受但实际忽略"的假支持。
  - `RoomManager.start_game()` 在设置 `room.status="running"` 后立即广播 `room_status: running`;前端 `phase_started` 也会把 waiting 推到 running,修复观战 HUD 启动按钮在已开局后残留。
  - `frontend/dist` 缺失时 FastAPI 不再回退返回未编译的 `frontend/index.html`;改为 503 明确提示使用 Vite dev 或先 `npm run build`,避免干净克隆用户打开坏 TSX 入口。
- **前端/shadcn 与开源可用性**:
  - `GameHud` 的 "开始真实对局" 按钮在 `status=waiting && phase=setup` 才显示;从 waiting 观战页可直接真实启动对局,进入 night/running 后按钮消失。
  - `RoomSidebar` 和 `ChatRoom` 消息头像迁到真实 shadcn `Avatar/AvatarFallback/AvatarBadge`,保留业务色彩但不再手写 avatar 结构。
  - `ChatRoom` 移除未实现的"举报/设置/表情/图片"入口,避免开源用户看到假功能按钮。
  - 主聊天气泡新增真实 `Badge` 元数据:回应谁、指控谁、支持/反对/观察谁;投票系统消息展示 `objectiveSummary` 客观摘要。这样用户在主屏即可看到"2号质疑/4号反打/6号识谎/投票摘要"链路,不必只看赛后指标墙。
  - README 更新为 React/Vite/shadcn 真实启动矩阵:开发模式后端 8000 + Vite 5173;生产/演示模式先 `npm run build` 再访问 8000;测试命令改为 `PYTHONPATH=. pytest -q`;真实 smoke 明确会调用模型。
  - `docs/ARCHITECTURE.md` 同步技术栈、REST 路径(`/api/rooms/...`)、生产构建方式和 shadcn 前端现状;`.gitignore` 补 `frontend/node_modules/`, `frontend/dist/`, `*.tsbuildinfo`。
- **测试/验证**:
  - 新增/更新单测:REST `human_seats` create/get/start/replay 保留且 running 不泄露角色;非法 human seats 400;start 后 human actor 映射;unsupported `deck` 400;dist 缺失返回 503 setup page;human requested night action 可省略 duplicate action 字段;human invalid target 透明 skip;AI hunter requested_action 落地真实目标;witch save requested_action 拒绝 poison-only shape;orchestrator 向 seer/guard/wolves/witch/hunter 传具体 requested_action;start_game 广播 running status。
  - 编译:`python -m py_compile src/api/server.py src/api/room_manager.py src/agent/actor.py src/agent/prompts.py tests/...` → 通过。
  - 前端:`cd frontend && npm run build` → 通过(tsc + Vite,1919 modules)。
  - shadcn 真实信息:`cd frontend && npx shadcn@latest info --json` → framework `Vite`, Tailwind `v4`, style `radix-nova`, installed components 含 `accordion/avatar/badge/button/card/progress/tabs/tooltip` 等。
  - Python 全测:`PYTHONPATH=. pytest -q` → `159 passed, 3 skipped`。
  - 真实 LLM smoke:`PYTHONPATH=. python tests/smoke_e2e.py` → 完整跑完;村民胜;真实模型调用 `53`,成功 `53`,失败 `0`,router 重试 `0`,`total_tokens_in=225422`,`total_tokens_out=18669`,`avg_latency=4.272`,事件 `73`,思考摘要 `23`,0 `agent_decision_failed`。对局中 5号预言家查杀4号狼,2/4号狼用"跳太快/带节奏"叙事反打,6号根据投票与发言矛盾识破2号模糊焦点,day2 放逐2号狼。
  - agent-browser 真实生产构建验证:`http://127.0.0.1:8000` 打开大厅,真实点击创建房间 `0f5da661c640`,进入 waiting 观战页,HUD 显示"开始真实对局";点击后后端日志确认 `POST /api/rooms/0f5da661c640/start 200`,页面进入 `第 1 天 · 夜晚阶段`,启动按钮消失。截图:`/tmp/werewolf-prod-waiting-hud.png`, `/tmp/werewolf-prod-running-hud.png`。
  - 安全复查:`/api/config` 返回 `api_key:""` 与 `api_key_configured:true`;凭据片段扫描(排除 `frontend/node_modules`, `frontend/dist`, `__pycache__`, `*.pyc`) → 0 命中;浏览器/uvicorn 会话验证后已停止,无后台服务继续消耗真实调用。
- 诚实 caveat:本轮补齐了 UI 启动闭环、主屏对抗元数据、协议 no-fallback 边界和开源启动文档,但还不是最终开源发布完成态。仍需做 ended 浏览器截图/胜负海报、ReplayPanel 顶部故事线、更多 ScrollArea/指标折叠、LICENSE/CONTRIBUTING/SECURITY/Docker/版本锁等发布资产。

### 3.47 ended 胜负海报 + 关键转折故事线 + 开源协作骨架(✅,build/test/browser 真对局通过)
- 触发:用户继续要求"读当前目录照片按那种 UI 设计,必须真实 shadcn UI,开子代理加速,前后端都好好做,保证能欣赏 agent 对抗"。本轮并行 3 路只读子代理:① 结束态/Replay UX;② 参考图 + shadcn UI 审计;③ 开源发布/协议文档审计。主线负责实现、验证和文档收口。
- **联网核验增量(2026-07-06)**:
  - Werewolf Arena(arXiv:2407.13943, https://arxiv.org/abs/2407.13943):继续支撑 werewolf-mas 的动态发言权/bid + 社交推理评测路线,其摘要明确把狼人杀用于 deception/deduction/persuasion 场景,并使用 bidding 模拟真实讨论中"何时发言"的策略选择。
  - OpenDeception(arXiv:2504.13707, https://arxiv.org/abs/2504.13707):支撑赛后 deception audit / trust susceptibility 思路,不能只看 speaker 自报,要做 intent + listener susceptibility 的联合评估。
  - MultiMind(arXiv:2504.18039, https://arxiv.org/abs/2504.18039):支撑 ToM/suspicion levels 和策略选择,与本项目 EvidenceGraph/posterior trace/attitudes 的方向一致。
  - Can LLM Agents Really Debate?(arXiv:2511.07784, https://arxiv.org/abs/2511.07784):提醒多 agent debate 需要过程级分析,多数压力会抑制独立纠错;werewolf-mas 的关键转折故事线/后验轨迹展示应服务于过程审计,而不是只显示最终胜负。
- **结束态/复盘 UI**:
  - `ReplayPanel` 新增 `compact` 模式和赛后 hero:展示"真实对局结算"、胜负阵营、天数、幸存者、出局线、决策失败/有损解析计数。全部来自 `state.analysis` / `state.winner`,不造假数据。
  - 新增 `buildStoryPoints()` / `ReplayStory`:从真实 `analysis.seats`、`state.log`、`deception_audit.records`、`collusion_audit.records`、`posterior_trace` 推导"关键转折故事线",例如 D1/D2 谁出局、最终结算、欺骗审计、窗口接力、最终嫌疑排序。只摘要真实字段,不编造剧情。
  - `ReplayPanel` 的密集指标、审计证据链、后验快照、Agent 终态信任改用真实 shadcn `Accordion`;审计/后验/Agent 长列表源码接入真实 shadcn `ScrollArea`;审计 chip 从裸 `span` 改为真实 shadcn `Badge`。
  - `GameView` 在 `state.status === "ended" && state.analysis && mode !== "replay"` 时也展示 `ReplayPanel compact`,修复非 replay 观战/上帝/人机结束后只能从日志里扫结果的问题。现在 ended 主界面即可欣赏胜负海报 + 转折线,replay 模式保留完整研究面板。
  - `global.css` 新增复盘暗色 HUD 样式:参考图方向的深色结算卡、阵营色边、幸存/出局 badges、故事卡、折叠指标区。CSS 只做布局/视觉,交互组件继续用真实 shadcn `Card/Badge/Accordion/ScrollArea/Progress`。
- **开源协作骨架**:
  - 新增 `.nvmrc`(`20`) 与 `.python-version`(`3.12`)。
  - 新增 `Makefile`: `install/install-py/install-ui/dev-api/dev-ui/build-ui/test/test-py/test-ui/smoke-real/stats-dryrun`。
  - 新增 `CONTRIBUTING.md`:本地开发、真实 LLM smoke 会花钱、no-fallback 规则、shadcn 真组件要求、协议变更同步要求、不要提交 secrets/logs。
  - 新增 `SECURITY.md`:本地绑定/公网部署警告、漏洞报告、凭据不下发、运行中信息隔离、赛后 analysis 真值边界。
  - 新增 `docs/PROTOCOL.md`:REST、WS mode、事件 union、visibility、人类 action 协议、赛后 analysis 字段与"不要喂回 live agent"约束。
  - `README.md` 补 Makefile、`docs/PROTOCOL.md`、`CONTRIBUTING.md`、`SECURITY.md`、多局统计入口和"许可证待确认"说明。
  - `.gitignore` 补 `session_*.zip`。
  - 未创建 `LICENSE`:许可证是法律/治理选择,需要用户确认 MIT / Apache-2.0 / AGPL-3.0 / 暂不授权复用,不能替用户暗猜。
- **真实 shadcn 验证**:
  - `cd frontend && npx shadcn@latest info --json` 返回 framework `Vite`, Tailwind `v4`, style `radix-nova`, base `radix`, icon `lucide`, installed components 含 `accordion/avatar/badge/button/card/progress/scroll-area/tabs/tooltip` 等。
  - 浏览器 ended DOM 验证新增结算区: `hasSettlement=true`, `hasStory=true`, `hasWinner=true`,真实 shadcn DOM `cards=32`, `badges=83`, `accordions=2`, `accordionItems=5`。截图: `/tmp/werewolf-ended-summary-3-47.png`。
  - 诚实 caveat:源码已接入 `ScrollArea` 且 build 通过,但本轮浏览器尝试展开审计证据链时没有拿到 `.replaypanel__audit-groups` DOM,因此"ScrollArea 浏览器挂载"不记为已验证;已验证的是 ended 摘要、hero/story 和 Accordion/Card/Badge shadcn 挂载。
- **真实浏览器 + 真实模型对局验证**:
  - 启动生产构建后端 `python -m src.api.server`。
  - `agent-browser` 打开 `http://127.0.0.1:8000`;`/api/config` 返回 `api_key:""` 与 `api_key_configured:true`,不下发 key。
  - 真实 UI 创建房间 `76793d5b9ac7`。注意前两次点击因按钮在视口下方未触发 POST,未计为验证;执行 `scrollintoview @e32` 后真实点击,后端日志确认 `POST /api/rooms 200`。
  - 真实点击 "开始真实对局",后端日志确认 `POST /api/rooms/76793d5b9ac7/start 200`,并持续出现真实 OpenAI Chat Completions 兼容接口 `HTTP/1.1 200 OK`。
  - 对局完整结束:REST `GET /api/rooms/76793d5b9ac7` 返回 `status:"ended"`, `end_reason:"completed"`, `winner:"village"`, `phase:"ended"`, `day:2`;幸存 4号/6号 villager;1号/3号 werewolf 出局,5号 seer/2号 villager 出局。
  - 浏览器观战页显示 `第 2 天 · 好人胜`;新增结算文本包含"真实对局结算 / 好人阵营获胜 / 决策失败 0 / 有损解析 0 / 关键转折故事线"。故事线真实提取到: D1 1号狼放逐 + 5号预言家狼刀; D2 2号民狼刀 + 3号狼放逐; 3号欺骗审计; 1/3 窗口接力; 6号最终嫌疑排序。
  - 期间 actor 严格 JSON 解析拒绝了数次有损恢复并真实重试,未伪造发言/投票/夜间行动。
  - 验证后 `agent-browser close`,后端 `Ctrl+C` 停止;`pgrep -af "uvicorn|src.api.server|vite|npm run dev"` 只剩 pgrep 自身,无后台服务继续消耗真实调用。
- **测试/构建/安全**:
  - 前端:`cd frontend && npm run build` → 通过(tsc + Vite,1920 modules)。
  - Python 全测:`PYTHONPATH=. pytest -q` → `159 passed, 3 skipped in 69.24s`。
  - 编译:`python -m py_compile src/api/server.py src/api/room_manager.py src/agent/actor.py src/agent/prompts.py src/game/orchestrator.py` → 通过。
  - 凭据片段扫描(排除 `frontend/node_modules`, `frontend/dist`, `__pycache__`, `*.pyc`) → 0 命中。
- 诚实 caveat:本轮完成 ended 主界面摘要、真实 UI 完整对局验证和开源协作骨架,但还不是最终开源发布:仍需用户确认 `LICENSE`;`docs/ARCHITECTURE.md` 仍偏旧,应后续按 `docs/PROTOCOL.md` 和当前 EvidenceGraph/审计/后验体系重写;右栏信息架构和角色头像资产仍可继续按参考图精修。

### 3.48 统一角色头像资产 + 架构文档当前化(✅,build/test/browser waiting 验证通过)
- 触发:继续按 §3.47 caveat 和参考图审计推进前端成型度。参考图的第一视觉锚点是统一角色头像/号码角标/阵营边光;当前 `SeatGrid`、`RoomSidebar`、`ChatRoom` 仍各自拼 shadcn `Avatar` + emoji/座位号 fallback,风格不统一。
- **真实 shadcn 角色头像封装**:
  - 新增 `frontend/src/components/RoleAvatar.tsx`:基于真实 shadcn `Avatar/AvatarFallback/AvatarBadge`,内部用 lucide glyph 作为统一角色符号。不是手写假头像控件。
  - 角色映射:狼人 `PawPrint`,预言家 `Eye`,女巫 `FlaskConical`,守卫 `Shield`,猎人 `Crosshair`,医生 `HeartPulse`,村民 `Bot`,未知身份 `CircleQuestionMark`。
  - `RoleAvatar` 支持 `role/team/seat/alive/reveal/size/className/badgeClassName`,并统一输出 `.role-avatar--werewolf/seer/witch/guard/hunter/doctor/villager/hidden` 阵营和角色色。
  - `SeatGrid` 改用 `RoleAvatar` 渲染圆桌座位头像,并用 `roleLabel()` 替代 emoji 角色标签。
  - `RoomSidebar` 玩家列表改用同一个 `RoleAvatar`,保留 alive badge 和运行中身份隐藏逻辑。
  - `ChatRoom` 消息头像改用 `RoleAvatar`;普通观战仍只显示未知/座位号,本人/死人/赛后才按既有规则 reveal 身份,不改变信息隔离。
  - `global.css` 新增统一角色头像视觉:暗色浮雕、座位号角标、alive 光点、角色/阵营渐变边光;圆桌/左栏/聊天复用同一视觉基线。
- **协议/架构文档对齐**:
  - `frontend/src/lib/types.ts` 把 `room_status.status` 从宽泛 `string` 收紧为 `RoomStatus`,并给 `game_error` 补 `reason?: string | null`,与后端事件更一致。
  - `docs/ARCHITECTURE.md` 更新仍过期的 §4/§6/§7/§8/§9:
    - "规则引擎扩展"改为"当前规则引擎能力",明确 Witch/Guard/Hunter/Last Words/PK/公开记忆/requested_action 已实现。
    - WebSocket 事件名从旧 `roles_dealt/phase_changed/night_deaths/player_exiled` 更新为当前 `snapshot/room_status/phase_started/night_resolved/speech/vote_cast/vote_resolved/vote_incomplete/...`。
    - 信息隔离描述改为当前事实:公开事件走 room manager allowlist,私有事件走 `visibility/recipients`;运行中不泄露角色真值、wolf caucus、private reasoning、赛后 truth analysis。
    - 前端章节补 ended 复盘和 `RoleAvatar`;评测章节补 `objective_metrics/posterior_metrics/deception_audit/collusion_audit/multi_game_stats.py JSONL/CI`。
    - LLM 描述改成标准接口路径: `openai` Chat Completions、`openai_responses` Responses API、`anthropic` Messages API,模型由 `WEREWOLF_LLM_MODEL` 配置。
- **验证**:
  - 前端:`cd frontend && npm run build` → 通过(tsc + Vite,1921 modules)。
  - Python 全测:`PYTHONPATH=. pytest -q` → `159 passed, 3 skipped in 70.08s`。
  - shadcn 真实信息:`cd frontend && npx shadcn@latest info --json` → framework `Vite`, Tailwind `v4`, style `radix-nova`, base `radix`, icon `lucide`, components 含 `avatar/badge/card/accordion/scroll-area/...`。
  - agent-browser waiting 页验证(不启动真实对局,因此不触发模型费用):生产构建页面真实创建房间 `8465e335c18b`,进入观战 waiting 页;DOM 统计 `roleAvatars=12`, `shadcnAvatars=12`, `fallbackCount=12`, `seatGridAvatars=6`, `sidebarAvatars=6`, `hiddenAvatars=12`,页面状态 `第 0 天 · 准备阶段`。截图:`/tmp/werewolf-role-avatar-waiting-3-48.png`。
  - 验证后 `agent-browser close`,后端 `Ctrl+C` 停止;`pgrep -af "uvicorn|src.api.server|vite|npm run dev"` 只剩 pgrep 自身。
  - 凭据片段扫描(排除 `frontend/node_modules`, `frontend/dist`, `__pycache__`, `*.pyc`) → 0 命中。
- 诚实 caveat:本轮浏览器验证只覆盖 waiting 页角色头像挂载,没有启动真实 LLM 对局;真实 LLM 全链路沿用 §3.47 的完整 UI 对局验证。下一步视觉优先级仍是右栏"阶段信息/事件日志/胜利条件"重排和 ChatRoom 裁判记录面板,以及用户确认 `LICENSE`。

---

### 3.49 右栏信息架构 + ChatRoom 裁判记录面板(✅,build/test/browser waiting 验证通过)
- 触发:继续 §3.48 caveat 中的前端成型优先级,补齐参考图右栏"阶段信息 / 事件日志 / 胜利条件"信息架构,并把 ChatRoom 的记录 tab 从原始日志列表升级成可扫描的裁判记录面板。
- **前端实现**:
  - `ChatRoom.RecordTab` 新增"裁判记录"摘要区,按真实 `state.log` 分类统计裁判/发言/投票/风险,并提供 `裁判/发言/投票/思考/风险/全部` 筛选。记录行用真实 shadcn `Badge/Button/ScrollArea`,保留 day、事件类型、actor、目标、reply/bid/OSR/claim 等结构化元数据,不使用 mock。
  - `GameStatusPanel` 新增 `PhaseBriefCard` / `RecentEventsCard` / `VictoryConditionCard`:展示当前阶段、房间状态、存活/票数、当前焦点、最近 6 条公开事件、胜利条件与赛后胜方。运行中非 god 视角不计算/显示隐藏狼人存活数,只显示"运行中隐藏阵营人数"。
  - `global.css` 补裁判记录、阶段摘要、事件日志、胜利条件样式,沿用现有暗色 HUD、8px 内圆角、shadcn 组件和蓝/金/红/绿语义色。
- **边界**:
  - 纯展示层改动;不改后端协议、不改 harness/agent 决策、不新增假数据。
  - 最近事件只读前端已收到的 `state.log`;信息隔离仍由后端 WS mode 和 reducer 输入决定。
- **验证**:
  - 前端:`cd frontend && npm run build` → 通过(tsc + Vite,1921 modules)。
  - Python 全测:`PYTHONPATH=. pytest -q` → `159 passed, 3 skipped`。
  - 本地开发服务启动:后端 `http://127.0.0.1:8000`,Vite `http://127.0.0.1:5173/`;`/api/config` 仍返回 `api_key:""` 与 `api_key_configured:true`。
  - agent-browser waiting 页验证(不启动真实对局,不触发模型费用):真实 UI 创建房间 `b67e54cb53ee` 并进入观战;DOM 统计 `phaseBrief=1/recentEvents=1/victory=1`;点击"记录"tab 后 `recordSummary=1`,筛选按钮为 `裁判/发言/投票/思考/风险/全部`。截图:`/tmp/werewolf-record-rightpanel-3-49.png`。
- 诚实 caveat:本轮只验证 waiting 页挂载和 DOM,没有启动真实 LLM 对局;真实 LLM 全链路沿用 §3.47/§3.48 的验证。下一步前端优先级是趋势历史导入/导出 JSON、按模型/人数/策略版本/experiment_id 分组对比,以及用户确认 `LICENSE`。

---

## 4. 下一步规划(按优先级 + 学术支撑强度)

### 4.1 前端(独立线,首要)
- §3.12 已落地 A/B/C/DR 对话元数据可视化(chip + 圆桌态度网络边 + DialogueMetricsCard + god 模式狼人党团私聊展示)。
- §3.14 已落地跨局趋势图 + 思考流 verbose 展示优化;§3.18 已补 ReplayPanel 客观轨迹指标;§3.32 已接入 turn_policy/debate process 单局卡、复盘和趋势;§3.33 已接入 collusion audit 单局卡、复盘和趋势;§3.34 已把离线多局工具升级为多策略/ABBA JSONL 协议;§3.35/§3.36 已产出两个真实 ABBA 报告;§3.37 已补可恢复长跑 runner;§3.38 已展示 pair 级合谋影响;§3.39 已补审计证据链、windowed relay、correct/wrong herding;§3.41 已接入真实 shadcn/ui + Tailwind v4 + radix-nova 组件基线;§3.42 已修 Radix ref 兼容并把实际投票条迁到真实 `Progress`;§3.43 已把 `ChatRoom` 迁到真实 `TabsContent` 并把 `GameStatusPanel` 泛用指标 chip 迁到真实 `Badge`;§3.44 已按参考图把中栏升级为圆桌态势主面板,并继续迁移 `RoomSidebar/SpeechFeed/ReplayPanel/SeatGrid/ChatRoom` 的 Card/Badge/Progress;§3.45 已补顶部 `GameHud`、真实 `Accordion/Avatar`、桌面 overlay、移动端防塌缩、UI 开局按钮、人机 `human_seats/mySeat/requested_action` 协议闭环,并在新 UI 代码后跑过真实 LLM smoke;§3.46 已修 requested_action 真闭环、waiting 观战页 HUD 启动闭环、主聊天 reply/accuse/attitude/objective summary 元数据、真实 Avatar 收敛和开源启动文档;§3.47 已补 ended 胜负海报/关键转折故事线、非 replay 结束态复盘、真实 UI 完整对局截图、CONTRIBUTING/SECURITY/PROTOCOL/Makefile/版本文件;§3.48 已统一 `RoleAvatar` 角色头像资产并更新 ARCHITECTURE 当前协议/规则/评测描述;§3.49 已补右栏"阶段信息/事件日志/胜利条件"与 ChatRoom 裁判记录面板。下一步:前端应继续趋势历史导入/导出 JSON、按模型/人数/策略版本/experiment_id 分组对比,并继续按参考图压实右栏/复盘的信息层级,不得造假组件。

### 4.2 中长期(deep-research finding [6] Generative Agents,可选精度升级)
- **短期 Phase 0/1(已完成,见 §3.17)**:真实信息链(投票上下文 + 公开发言进 memory) + objective_metrics + 修评审输入 bug + 修 wolf_coordination。
- **中期 EvidenceGraph/RolePosterior v2**:§3.19 已落最小可见证据图,§3.20 已落最小后验轨迹指标,§3.22 已落 `evidence_id→posterior_delta` provenance,§3.23 已落统一 `evidence_items` substrate,§3.24 已落 team-world constrained posterior。下一步基于 CSP4SDG/GRAIL 从 team-world 扩展到 full role-world,并用公开证据/claim/vote/death/态度边约束合法身份世界。LLM 负责表达/欺骗/说服,不是独自承担概率推理。
- **posterior 校准评测**:§3.21 已完成最小 Brier/log loss/ECE/校准分桶闭环,§3.22 已补可追溯 delta,§3.24 已提供 constrained posterior,§3.25 已并排接入 constrained Brier/log loss/ECE 与多局统计/前端展示。下一步是实现 evidence_id→likelihood_delta→posterior,按 viewer/role/phase 分层评估正确共识、错误从众、独立纠错与过度自信。
- **欺骗/合谋审计 v3**:§3.29 已落 deterministic `deception_audit` v1(speaker intent × independent audit × listener shift),§3.30 已落 evidence refs + peer detection + listener susceptibility by seat,§3.38 已落 pair 级 `collusion_audit` v2,§3.39 已落 windowed relay 与审计证据链展示。下一步做 evidence-item 级 LLM/规则混合 auditor,标注 `observed_deception_type/distorted_source/omitted_counterevidence/targeted_listener/listener_shift`,并与 `likelihood_delta` 后验校准合并;再做 Colosseum 风格 side-channel/collusion regret ablation。
- **评测体系 v2**:§3.18 已把多局统计升级为 JSONL + CI,§3.34 已补多策略/ABBA schedule、per-policy summary 和 paired delta,§3.35 已跑首个真实 ABBA 并补 router stats delta,§3.37 已补 `--resume-jsonl` 和扩展 ABBA delta。下一步是 evidence-based single-dimension judge + AB/BA pairwise + bootstrap CI + controlled intervention meta-eval(删/换/打乱关键证据看 judge 是否按预期变分),并把多模型/人数/策略版本/experiment_id 作为分组维度。
- **稳定性 P0/P1**:§3.26 已完成 `LLMRouter` per-attempt wall-clock 总 timeout、`RoomManager._run_room()` 房间级 timeout、failed/timeout/cancelled 状态和前端/API 状态展示;§3.27 已完成 strict/lossy JSON parse 标记与最后一次透明有损兜底;§3.28 已完成 `parse_failed` 率统计/展示/多局聚合;§3.39 已把玩家真实决策默认重试、router/config/API/smoke/multi-game runner 对齐到 ≥5,并移除投票/猎人隐性脚本兜底;§3.41 已完成 orchestrator per-decision timeout 和 `decision_failure_metrics`;§3.42 已补所有并发/顺序 actor 普通异常透明审计与 `by_error_type`;§3.43 已完成共享 phase deadline、未开始 seat 透明 `PhaseDeadlineExceeded` 审计,并修 API 入口 per-phase env 传参。下一步是让 smoke 可选要求赛后 quality judge 成功,并按阶段/角色继续压测 timeout 与 room timeout 组合。
- **后续社会机制**:`turn_policy` 四策略矩阵(`fixed_round_robin/bid_only/bid_reply/bid_reply_caucus`)与跨模型/人数/策略版本分组、collusion audit v3(窗口化接力 + omitted counterevidence + evidence-item auditor + likelihood_delta 对齐)、reference-pool arena protocol、多效用角色/Jester stress test、稀疏二阶 ToM matrix、Stackelberg candidate utterance rerank、DVM-style balance controller。

---

## 5. 八条工作铁律(用户确认,来自 memory `feedback-eight-principles`)

1. **查阅而非暗猜**:不懂上网搜/读论文/读代码,不凭空假设。
2. **寻求确认而非模糊执行**:业务/呈现决策问用户,技术实现自己定。
3. **人类确认业务呈现**:UI/玩法呈现层等用户拍板。
4. **复用而非创造**:优先复用现有函数/模式,不重写(方向 A 复用 `_extract_int`/`_clamp_suspicion`/`_bid_order_by_llm`)。
5. **主动测试**:改完跑 pytest + 真实对局验证,不靠猜。
6. **遵循架构**:严守 harness/agent 分层,不越权。
7. **诚实无知**:假设要实证验证(httpx client 复用 bug 被复现证伪,诚实承认),不编造。
8. **谨慎重构**:加字段/调调度键可以,不重写规则引擎/信息隔离层。

---

## 6. 安全 / no-fallback 铁律(来自 memory `no-fallback-design` + `multi-provider-model-config`)

- **凭据**:仅用 `WEREWOLF_` 前缀环境变量,**绝不用** `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/`ANTHROPIC_BASE_URL`。凭据仅存内存,绝不持久化到代码/记忆/git。
- **配置展示**:`/api/config` 只返回 `api_key_configured` 布尔状态,`api_key` 永远为空字符串;不得把 key 前缀/后缀/掩码片段下发给前端或写入日志。
- **不伪造**:绝不回退/伪造决策;失败深度重试(≥5 次,指数退避+抖动);宽松超时不误杀。
- **API 调用**:按标准接口适配: OpenAI Chat Completions、OpenAI Responses API、Anthropic Messages API。输出 token 上限按协议映射,不写供应商/模型特殊分支。
- **结构化输出**:OpenAI Chat 使用 `response_format`,OpenAI Responses 使用 `text.format`,Anthropic Messages 不发送 OpenAI 专用字段;JSON 解析失败只走透明重试/失败,不伪造决策。
- **网络**:后端绑 127.0.0.1,不公网暴露。
- **前端验证**:UI 改动必须 `npm run build`;涉及页面/交互时用真实浏览器(agent-browser)做 DOM/可访问树/截图/关键点击验证,不得只看控制台或假截图。
- **子代理**:可在用户明确要求时使用 Codex 子代理并行做调研/审计/窄范围实现;子代理不能替代主线对信息隔离、no-fallback、真实 LLM 验证和最终集成负责。

---

## 7. 关键文件地图

```
src/
├── llm/router.py          # SSE 流式 LLM 调用,多 provider 分流,深度重试
├── agent/
│   ├── actor.py           # agent 大脑:decide_speak/vote/skill, _sanitize_speak 解析 reply_to/accuses
│   ├── schemas.py         # Decision(reply_to/accuses/attitudes/deception), AgentObservation.evidence_graph
│   ├── prompts.py         # SYSTEM_BASE/PERSONAS/SPEAK_SCHEMA, ToM/attitude/evidence graph 渲染
│   ├── memory.py          # AgentMemory: observations/reflections/trust/claims + 三因子检索
│   ├── evidence.py        # 可见 EvidenceGraph + 软 RolePosterior(不读真身份,不替 agent 决策)
│   └── information.py     # 信息隔离:按 visibility 过滤观察(harness 职责)
├── game/
│   ├── orchestrator.py    # harness 编排器:_run_day/_bid_order_by_llm/turn_policy/_debate_process_metrics/_collusion_audit
│   ├── rules.py           # RulesEngine 规则裁决,vote_cast 落库
│   ├── state.py / models.py / roles.py / eval.py
└── api/
    ├── server.py          # FastAPI + WS,绑 127.0.0.1
    └── room_manager.py

tests/
├── verify_direction_a.py  # 方向 A 端到端验证(god 模式 WS 采集 reply_to/accuses)
├── smoke_e2e.py / multi_game_stats.py / test_*.py  # 当前 159 passed,3 skipped
docs/
├── research-multi-agent-harness-2026-07-04.md  # 第三轮深研笔记
```

---

## 8. 工作流(每次改动的默认动作)

1. 读 CLAUDE.md(本文) + 相关 memory。
2. 改代码:遵循分层铁律,复用现有函数,加字段/调调度而非重写。
3. `PYTHONPATH=. pytest -q`(当前 159 passed,3 skipped 不回归)。
4. 真实对局验证(`verify_direction_a.py` / `smoke_e2e.py` god 模式)。
5. 0 决策失败(no-fallback 守恒)。
6. 更新 memory + 本 CLAUDE.md 的"已完成"章节。

---

## 9. 当前阻塞 / open question(诚实)

- **WOLF/MultiMind/Beyond Survival/DVM/Werewolf Arena 等方向已有多轮联网核验与工程落地引用**,但仍缺系统复现实验/跨模型多局验证;不能把论文方向等同于 werewolf-mas 的统计显著提升。
- **方向 B/C 落地有效性无 primary source 实证**(deep-research caveats 第5点)。werewolf-mas 已做单局/6局/20局自身验证,但仍只是本项目小样本实证;跨模型/人数/策略版本仍需扩样本。
- **verify_direction_a.py 统计部分未打印**(WS 在 game_ended 后断,小 bug,不影响验证结论)。
- **AvalonBench 的 LLM 落败基于早期 ChatGPT**(caveat 第4点),新模型可能缩小差距——但"prompt-driven ReAct 不足以应对对抗社会推理"的架构启示依然成立。
