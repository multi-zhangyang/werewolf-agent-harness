# AI 狼人杀研究报告:游戏设计 · UI · 多agent对抗架构

> 检索日期 2026-07-03。来源均为真实抓取:GitHub API、DuckDuckGo、arXiv HTML、维基百科、项目 README。

---

## 一、最好的狼人杀:机制与平衡设计

### 1.1 核心张力 —— 三大永恒矛盾
所有优秀社交推理游戏都围绕三个矛盾设计,**你的引擎必须保留这三股张力**:

1. **知情少数 vs 不知情多数**:邪恶方知道队友,好人不知道。好人靠"能力角色"(预言家/医生)弥补信息劣势。
2. **"知道但不能表现得太知道"**(梅林困境):掌握关键信息者若过早暴露,会被反杀。这是 Avalon 刺杀机制与狼人杀预言家跳身份的核心张力。
3. **软信息 vs 硬信息**:硬信息(机制验证,如查验结果)与软信息(行为读人、逻辑矛盾)分层。**好设计刻意让硬信息不可 100% 信任**(BotC 的 Drunk/Poisoned 机制)。

### 1.2 Avalon 的信息不对称设计(教科书级)
来自《抵抗组织:阿瓦隆》,六个角色构成多层信息结构,直接可借鉴到你的"板子"设计:

| 角色 | 阵营 | 知道什么 | 设计意图 |
|------|------|----------|----------|
| 梅林 Merlin | 好 | 看到大部分邪恶玩家(除莫德雷德) | 好人的信息王牌,但暴露即死 |
| 帕西瓦尔 Percival | 好 | 知道两人中"一个是梅林一个是莫甘娜"但不知谁是谁 | 创造识别博弈 |
| 莫甘娜 Morgana | 邪 | 在帕西瓦尔眼中伪装成梅林 | 保护真梅林 |
| 莫德雷德 Mordred | 邪 | **对梅林隐身** | 平衡梅林信息优势 |
| 奥伯伦 Oberon | 邪 | 其他邪不知其身份 | 制造邪恶方内部信息断裂 |
| 刺客 Assassin | 邪 | 末局刺杀梅林 | 终极悬念 |

**关键启发**:每一方都有部分信息但无人拥有完整信息。`莫德雷德隐身`+`奥伯伦制造内部断裂`这种"在优势方内部再挖盲区"的设计,比单纯"狼人互认"更耐玩。

### 1.3 Blood on the Clocktower 的三大创新
BotC(2018,Steven Medway)解决了传统狼人杀的两个痛点:

1. **死亡不淘汰(ghost vote)**:死者失去能力与投票权,但仍可发言、**保留最后一票**。→ 解决"被淘汰玩家坐冷板凳"问题。对 AI 对局意义:死者 agent 仍可作为"信息源"参与讨论,提高博弈密度。
2. **Storyteller(主持人)有裁量权**:可"在限度内弯折规则"以平衡和制造戏剧性。→ AI 版本中可体现为一个**有平衡倾向的 GM agent**,而非纯机械规则。
3. **"registers as"(登记为)机制**:能力检查的是"登记身份"而非"真实身份"。Recluse 可能登记为邪,Spy 可能登记为好。→ **任何信息都不是 100% 可信**,强制玩家综合多方证据。

### 1.4 平衡性量化基线(来自 Werewolf Arena 蒙特卡洛模拟)
论文用 Monte Carlo 给出两条极端基线,**你应复刻这两个测试**作为平衡性护栏:

- 无信息交换时,村民胜率仅 **1.2%**(信息太少→纯随机→狼人碾压)
- 预言家自动揭示 + 村民盲信时,村民胜率 **100%**(信息太确定→无博弈)
- **健康的平衡区间:任一阵营胜率 40%–60%**

经验法则:邪恶方约占总人数 1/3(向上取整);8 人局常用 1预言家+1医生+2狼人+4村民。

### 1.5 第一手优势补偿
邪恶方开局即知全局,好人需追赶。补偿手段:给好人侧能力角色(查验/保护),并把**首轮信息获取设计得渐进**(如预言家每晚只能查一人),让好人随轮次积累信息。

---

## 二、顶级 UI 设计

### 2.1 信息安全(最关键,你的项目铁律之一)
社交推理游戏的 UI 泄露信息 = 游戏报废。必须遵守:

- **绝不渲染玩家无权看到的元素**:隐藏角色信息不要靠 CSS 隐藏,要靠后端不下发。前端"只渲染后端给的视图"。
- **动画/加载状态必须角色无关**:不要让加载时长、动画时机暴露"某能力角色正在行动"。
- **spectator 视图与玩家视图严格分离**:旁观者绝对看不到私密信息。
- 防止 DOM/网络包可被 devtools 检视出隐藏信息 → 后端投影式隔离(见 AIwerewolf 架构 §4.3)。

### 2.2 布局与组件
- **环形/网格座位**:镜像真实座次,存活/死亡用灰化+斜杠+图标多重表达(兼顾色盲)。
- **当前发言者**:高亮 + 计时环 + 边框光晕。
- **角色卡**:角落常驻,点击展开;颜色编码(红=狼/蓝=民/金=特殊)+ 图标双重编码。
- **阶段指示器**:昼夜用全局主题切换(色调/背景/光照),配阶段进度条。
- **夜间非行动玩家**:显示"闭眼/等待"屏,防信息泄露。
- **行动历史日志**:可滚动面板,回顾已发生事件。
- **投票**:一键选人 + 确认;实时或结算后显示票数;倒计时。

### 2.3 沟通与反馈
- 聊天面板白天突出、夜间可折叠;支持私聊/队聊(狼队)用差异化背景。
- **微动画**强化戏剧性:死亡揭晓、票数结算、阶段切换。
- 触控目标 ≥ 44×44px;移动优先布局;支持横竖屏。
- **AI 思考可视化**(你的产品愿景):agent 思考摘要流式展示,让玩家"看见 AI 的推理"——这是区别于传统狼人杀游戏的核心卖点。

### 2.4 值得研究的参考实现
- **Town of Salem / Town of Salem 2**:确立了在线社交推理 UI 范式。
- **Among Us**:会议/投票 UI 的极简大众化。
- **Blood on the Clocktower 官方 app**:复杂角色管理与主持人工具。
- **Goose Goose Duck**:移动端优化。

---

## 三、多 agent 对抗系统架构

### 3.1 ChatArena 的四层抽象(可直接借鉴的分层)
来自 Farama Foundation 的开源多agent语言博弈框架(已 deprecated 但架构极具参考价值):

```
Arena        ── 主循环 + UI/存储/配置加载
  └─ Environment ── 游戏状态 + 规则 + 观察生成(get_observation)
       ├─ Player 1 (角色A + 策略, stateless: observation→action)
       ├─ Player 2 (角色B + 策略)
       └─ Moderator / 硬编码规则 ── 信息分发 + 裁决
Language Backend ── 语言智能来源(多 provider: OpenAI/Anthropic/Cohere/HF)
```

**对你的 `src/game` 的映射**:
- `Environment` → 你的规则引擎(状态机:天黑→天亮→投票)
- `Player` → 你的 agent(stateless 策略函数:观察→Decision)
- `Language Backend` → 你的 `src/llm`(多 provider 路由)
- `Moderator vs 硬编码规则` → **混合范式**:确定性逻辑(谁被杀/被救)用硬编码;发言理解/引导用 LLM。**不要用 LLM 当裁判判定胜负**(不确定性太高,违背你"绝不伪造决策"铁律)。

### 3.2 状态-观察分离(信息不对称的工程实现)
ChatArena 最值得借鉴的一点:**游戏状态对玩家不可直接见,玩家只看到 `get_observation` 生成的自然语言局部信息**。

- 环境维护完整状态(谁是什么角色、谁被杀)
- 每个玩家通过 `get_observation` 获得不同视图:狼人看到队友,平民看不到角色信息
- 夜晚行动时,环境只向相关角色暴露行动选项

**你的项目对应**:这正是 AIwerewolf 的 `GameState → PlayerView 投影` 设计(§4.3)。你应确保 `src/game` 的完整真相与下发到 agent/前端的视图严格分离。

### 3.3 多 agent 辩论架构(用于白天推理)
来自 Du et al. 2023 "Improving Factuality and Reasoning through Multiagent Debate"(arXiv 2305.14325)。机制:

1. **独立初始化**:N 个 LLM 实例各自独立生成初始答案+推理
2. **多轮辩论(R 轮)**:每轮看到其他实例上一轮的答案+推理,据此修正自己
3. **收敛**:R 轮后趋同(多数投票或末轮共识)

**迁移到狼人杀白天**:
```
Round 0: 每个 agent 私下推理"谁可能是狼,为什么"
Round 1..R: 每个 agent 收到上一轮所有公开发言,
            更新怀疑度 + 生成公开指控/辩护
Final: 投票;多轮交换使狼人难以维持矛盾的供词
```
核心价值:**对抗性自我修正** —— 狼人必须反驳他人提出的矛盾证据,村民可跨轮次抓住逻辑不一致。这天然映射到"村民揭穿狼人供词矛盾"。

### 3.4 AIwerewolf 的工程架构(技术栈与你高度一致,强烈推荐参考)
GitHub `wxhfy/AIwerewolf`,技术栈 FastAPI/WebSocket + Next.js + PostgreSQL/Redis,与你的项目几乎一致。其设计可直接借鉴:

| 设计点 | 做法 | 借鉴价值 |
|--------|------|----------|
| **Decision 抽象** | 所有 agent 行为(LLM/Human)统一为结构化 `Decision`,引擎统一校验推进 | 解耦决策来源与规则执行 |
| **三层 Prompt** | Persona(人格风格)× Role(角色策略)× Strategy(检索的历史经验) | 可独立迭代 |
| **投影式信息隔离** | `GameState`→`PlayerView`+public snapshot,前端只渲染后端给的视图 | 架构上杜绝泄露 |
| **引擎主控** | Agent 只提交 Decision,状态推进/校验/结算由引擎完成 | Agent 无法作弊改状态 |
| **多维评分(Track B)** | 逐决策打分:发言/投票/技能/时机/影响力;证据链 GameEvent→Decision→Review | 复盘与迭代 |
| **策略生命周期(Track C)** | candidate→active→deprecated 状态机;赛后自动门禁+批处理治理 | 避免策略池膨胀 |

**自进化策略系统**尤其值得关注:从高价值对局片段抽取策略知识 → 质量门禁晋级为 active → 下局检索注入 agent prompt。默认检索策略 `same_role_all_mbti`,在 374 条 active 策略上 P@3=1.0、nDCG@5=0.9885。这给你"AI 越打越强"的闭环。

---

## 四、LLM 欺骗与推理的关键研究发现

### 4.1 Werewolf Arena 的竞价发言系统(发言调度突破)
来自 Google `werewolf_arena`(arXiv 2407.13943)。**取代固定发言顺序**,让 agent 自主表达发言意愿:

- **0-4 五级竞价**:0=倾听 / 1=一般想法 / 2=关键具体贡献 / 3=极其迫切 / 4=被点名必须回应
- 最高出价者发言;平局时**上一轮被提及者**优先(自然"被指控者倾向回应"偏置)
- 效果:多数共识在第 2-5 句达成,关键信息早传播

**配套机制**:
- 每个 agent 独立 **memory stream**:观察记忆(事件+角色特权信息)+ 反思记忆(每轮总结,用于模式识别/矛盾检测)
- **合成投票**:每句话后模拟投票(不影响实际游戏、不写入记忆)追踪共识演变,用 Shannon 熵衡量不确定性下降
- **随机化候选名单顺序**对抗 LLM 位置偏差(倾向选首/尾项)

### 4.2 WOLF 基准的欺骗发现(NeurIPS 2025)
来自 WOLF: Werewolf-based Observations for LLM Deception and Falsehoods。7,320 条发言、100 轮游戏:

- **生成欺骗**:狼人在 **31% 的轮次**产生欺骗性发言(非全部,有策略选择性)
- **检测欺骗**:同伴侦测精确率 **71-73%**,但整体准确率仅 **~52%**(接近随机)→ **LLM 善骗不善识破**
- 怀疑度随轮次:对狼人从 ~52% 升到 60%+,对村民/医生稳定 44-46%
- **延长交互提升对说谎者的召回率,且不累积误判诚实角色** → 多轮博弈是关键
- **误报问题**:会错误判定诚实村民为欺骗者 → 检测 agent 需专门控制误报

**四类欺骗分类**(可作 agent 欺骗决策框架与检测信号):omission(遗漏)、distortion(扭曲)、fabrication(捏造)、misdirection(误导)

**纵向信任建模**:单轮判断不够,agent 需维护随时间演化的信任/怀疑模型。

### 4.3 模型风格差异影响胜负(Werewolf Arena 实测)
- GPT-4 狼人擅长**间接、看似协作的话术**:播种怀疑、诉诸共识、制造不信任
- Gemini 1.5 Pro 倾向**短促、情绪化、幽默讽刺**
- GPT-4 的冗长间接风格**对 Gemini 村民适得其反**——村民把啰嗦当可疑信号
- **启示**:跨模型对战需考虑风格匹配;你的多 provider 配置([[multi-provider-model-config]])天然适合做风格多样化的 agent 阵营

### 4.4 Seer 困境(预言家张力点)
早期揭示信息量大但风险高(被狼杀),延迟揭示更安全但信息价值递减。**自建系统必须确保这一张力存在**,它是好人侧策略深度的来源。

### 4.5 其他可参考的 LLM 狼人杀研究
- **Werewolf Among Us**(CU Boulder):人类 vs LLM 对比,163 场单轮人类标注 + 19 场多轮 LLM;发现 "LLM agents secure faster, more decisive wins"
- **Language Agents with RL for Werewolf**(arXiv 2310.18940):把狼人杀视为"混合合作-竞争多agent测试床",结合 RL
- **MaKTO-Werewolf**:Multi-agent KTO 强化策略交互训练
- **MultiMind**(ACM):引入多模态(面部表情/语调)推理,突破纯文本局限
- **WOLF 批评现有评估**:把欺骗简化为静态分类,忽略交互性/对抗性/纵向性 → 你的评测应做多轮动态评估

---

## 五、对你的 werewolf-mas 项目的具体建议

基于你已有结构(`src/game` `src/llm` `src/api` `frontend` `configs` `tests`)与产品愿景:

### 5.1 引擎层(`src/game`)
1. **状态-观察分离**:完整 `GameState` 只存在于后端;为每个 seat 生成 `PlayerView` 投影。狼人视图含队友,平民视图无角色信息。
2. **Decision 抽象**:无论 LLM 还是 Human,统一结构化 `Decision`,引擎校验推进。Agent 不能直接改状态。
3. **硬编码规则 + LLM 引导**:确定性逻辑(杀人/救人/查验/胜负)硬编码;不要用 LLM 当裁判判胜负(违背 [[no-fallback-design]] 真实对局铁律)。
4. **平衡性护栏**:复刻 Werewolf Arena 的两个蒙特卡洛基线(无信息 1.2% / 盲信 100%),把胜率锁在 40-60%。

### 5.2 Agent/LLM 层(`src/llm`)
1. **三层 Prompt**:Persona × Role × Strategy,可独立迭代。
2. **Memory stream**:观察记忆 + 每轮反思记忆,支撑矛盾检测。
3. **纵向信任模型**:每个 agent 维护对其他玩家的怀疑度,随轮次平滑更新(WOLF 的纵向性)。
4. **竞价发言调度**:0-4 级竞价替代固定顺序,被指控者优先(让 AI 对话像真人流动)。
5. **随机化候选顺序**:投票/行动时打乱名单,对抗 LLM 位置偏差。
6. **多 provider 风格多样化**:用 [[multi-provider-model-config]] 让不同 seat 用不同模型,制造风格差异(已实证影响胜负)。

### 5.3 前端(`frontend`)
1. **后端投影式信息隔离**:前端只渲染 `PlayerView`,绝不靠 CSS 隐藏角色信息。
2. **AI 思考可视化**(你的核心卖点):流式展示 agent 思考摘要 + 竞价/怀疑度仪表盘。
3. **昼夜全局主题切换** + 阶段进度条 + 死亡多重编码(灰化+图标,兼顾色盲)。
4. **环形座位**镜像座次,当前发言者高亮+计时环。
5. **移动优先**,触控 ≥44px,横竖屏适配。

### 5.4 评测与进化(新增方向)
1. **逐决策多维评分**(借鉴 AIwerewolf Track B):发言/投票/技能/时机/影响力,带证据链。
2. **策略知识闭环**(借鉴 Track C):candidate→active→deprecated,赛后自动门禁,下局检索注入。
3. **多轮动态欺骗评测**(WOLF 启示):不要静态分类,测交互性/对抗性/纵向性;分离"欺骗生成"与"欺骗检测"能力。

### 5.5 待研究的真实代码实现(可深挖)
高参考价值的开源仓库(按相关度):
- `wxhfy/AIwerewolf` — 技术栈最接近,工程架构首选参考
- `JuneQQQ/deepwolf` — "explainable human copilot" 可借鉴思考可视化
- `Muqian-Sun/ai-werewolf-agent-teams` — 字节评测+复盘平台,信息隔离+trace logging
- `google/werewolf_arena` — 竞价发言系统 + 全套 prompt 模板开源
- `ReneeYe/MaKTO-Werewolf` — RL 策略训练

---

## 来源清单(均真实抓取)

**论文/学术**
- Multiagent Debate — https://arxiv.org/abs/2305.14325
- AutoGen — https://arxiv.org/abs/2308.08155
- Werewolf Arena — https://arxiv.org/html/2407.13943v1
- WOLF (NeurIPS 2025) — https://neurips.cc/virtual/2025/128050
- Language Agents with RL for Werewolf — https://arxiv.org/html/2310.18940v2
- Werewolf Among Us — https://cuboulder-ds.github.io/CSCI-5423-Final/
- MultiMind (ACM) — https://dl.acm.org/doi/10.1145/3746027.3755752

**框架/项目**
- ChatArena — https://github.com/chatarena/chatarena
- AIwerewolf — https://github.com/wxhfy/AIwerewolf
- deepwolf — https://github.com/JuneQQQ/deepwolf
- google/werewolf_arena — https://github.com/google/werewolf_arena
- MaKTO-Werewolf — https://github.com/ReneeYe/MaKTO-Werewolf

**游戏设计**
- Avalon — https://en.wikipedia.org/wiki/The_Resistance:_Avalon
- Blood on the Clocktower — BotC 官方/Pandemonium Institute
