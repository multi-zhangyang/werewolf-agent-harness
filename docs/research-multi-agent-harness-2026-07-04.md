# 多 Agent 对抗与社会模拟:harness / agent 框架研究

> 2026-07-04 第三轮联网深研。聚焦"多 agent 对抗 + 多 agent 社会的 harness/scaffold 架构",
> 以狼人杀为呈现载体。承接前两轮狼人杀专项研究(research-round2)。
> 本文区分【实证】(论文/文档原文)与【推测】(我据此推断),并明确落地映射。

## 来源(均已联网获取全文)

- **S1** A Survey on LLM-Based Autonomous Agents (arXiv:2308.11432, Front. Comput. Sci. 2025) — 全文 23k 字
- **S2** A Survey on LLM-Based Social Agents in Game-Theoretic Scenarios (arXiv:2412.03920, TMLR 2025) — 全文 18k 字
- **S3** Generative Agents: Interactive Simulacra of Human Behavior (Park et al., UIST 2023, arXiv:2304.03442) — 全文 20k 字
- **S4** AutoGen v0.4 官方文档 (microsoft.github.io/autogen) — Teams / GroupChat 章节
- **S5** Prompt Engineering Guide — LLM Agents 架构页 (promptingguide.ai)
- 前两轮已读:Werewolf Arena(2407.13943)/ WOLF(2512.09187)/ Beyond Survival(2510.11389)/ MultiMind(2504.18039)/ DVM(2501.06695)

---

## 一、harness / scaffold / agent 的定义与边界

### 1.1 agent 架构四模块(统一框架)【S1 实证 Fig.2】

Survey(S1)提出 LLM agent 的统一架构 = **Profile + Memory + Planning + Action** 四模块:

- **Profile**:agent 的身份(人口学/性格/社会关系)。决定"你是谁"。
- **Memory**:短时(上下文窗口)+ 长时(外部存储);操作 = 读/写/反思。
- **Planning**:有反馈(环境反馈/自我反思,如 ReAct/Reflexion)或无反馈(单路/多路推理,如 CoT/ToT)。
- **Action**:动作空间(工具 + 自身知识)+ 动作影响(环境/新动作/内部状态)。

### 1.2 harness vs agent 的边界【S4/S5 实证】

- **agent = LLM 大脑**:推理决策引擎,通过 prompt 被赋予角色/工具/参数。【S5】
- **harness/scaffold = 大脑之外的一切**:编排循环、记忆存储机制、规划模块结构、工具 API、环境。【S5】
- **AutoGen(S4)的关键分离**:Team 对象(RoundRobin/Selector/Swarm)是**编排器**,管轮次顺序/终止条件/状态持久化/消息广播;participant agents 只负责"轮到自己时生成回应"。**编排逻辑与 agent 逻辑分离**。【S4】

### 1.3 落地映射到 werewolf-mas【推测,待验证】

| 学术概念 | werewolf-mas 现状 | 对应 |
|---|---|---|
| Profile 模块 | `prompts.PERSONAS` + `assign_persona` | ✅ 已有 |
| Memory 模块 | `agent/memory.py`(观察/反思/trust/claims) | ✅ 已有,缺 recency/importance 检索 |
| Planning 模块 | 无显式(决策即规划) | ⚠️ 狼人杀单步决策,规划可弱 |
| Action 模块 | `schemas.Decision` + `RulesEngine` | ✅ 已有 |
| **harness/编排** | `GameOrchestratorV2` | ✅ 符合 S4 的 Team 编排器模型 |
| **agent/大脑** | `AgentActor` | ✅ 符合 S4 的 participant agent 模型 |
| Team 模式 | 第0轮 RoundRobin + 后续 Selector(bid) | ✅ 混合模式,符合 S4 |

**结论**:werewolf-mas 的 harness/agent 分层**符合学术定义**(已在 reference-multiagent-harness-definition 记忆确认,本轮再实证)。问题不在分层,在**agent 间通信与二阶建模**不够(见下)。

---

## 二、多 agent 拓扑与狼人杀定位

### 2.1 四种 team 模式【S4 实证】

AutoGen v0.4 的 team 预设,按"谁决定下一个发言者"区分:

- **RoundRobinGroupChat**:固定轮转,每人广播给所有人。
- **SelectorGroupChat**:LLM 动态选下一个发言者。
- **Swarm**:agent 用 `HandoffMessage` 显式把控制权交给另一个 agent(无中央调度)。
- **MagenticOneGroupChat**:专门求解开放式 web/文件任务。

### 2.2 LangGraph 四架构【WebSearch 训练知识,未直接实证】

- **Supervisor**:中央 supervisor 路由任务给 worker agents。
- **Hierarchical**:多层 supervisor。
- **Network/Multi-agent**:agent 间自由通信。
- **Swarm/Handoff**:agent 动态移交控制(共享 state)。

### 2.3 狼人杀的拓扑定位【推测】

狼人杀是 **信息不对称的辩论-轮转 + 子团队(狼队)私聊** 混合拓扑:
- 白天:Selector 模式(bid 决定发言者)+ Swarm 元素(被点名者应被 handoff 回应)。
- 夜间:狼队是 Hierarchical/Network 子拓扑(私聊合谋),其他角色是独立 Supervisor-less 行动。
- **werewolf-mas 当前:白天 RoundRobin(第0轮)+ Selector(bid),缺 Swarm 的 handoff 回应机制** = 这正是方向 A 要补的。

---

## 三、多 agent 社会模拟的关键机制

### 3.1 记忆流 + 反思【S3 实证,Generative Agents】

Generative Agents 三件套:
- **Memory Stream**:agent 经历的自然语言记录,每条含创建时间 + 最近访问时间。
- **检索 = recency × 0.1 + relevance × 1.0 + importance × 1.0**(recency 指数衰减 ~0.99/小时;importance LLM 打分 1-10;relevance embedding 相似度)。【S3】
- **Reflection**:当近期记忆 importance 累积超阈值(~150),触发反思:① 识别要反思的问题 ② 检索相关记忆 ③ LLM 抽取洞察(如"Klaus 专注研究")④ 反思写回 memory stream,可被再反思(层次抽象)。
- **Planning**:日级计划 → 递归细化为小时/分钟级 → 反应式重规划。

### 3.2 落地映射【推测】

werewolf-mas 的 `AgentMemory` 有 observations/reflections/trust/claims,但:
- ❌ **缺三因子检索**:当前 `recent_observations(limit=30)` 只取最近 N 条,无 relevance/importance 加权。狼人杀轮次短,recency 主导尚可,但跨天 claim 矛盾检测可受益于 relevance 检索。
- ✅ **反思已有**:`reflect()` 每轮生成 insight 写入 reflections。
- ⚠️ **importance 缺失**:没有对记忆条目打重要度分。中长期可补。

### 3.3 Belief Module(信念模块)【S2 实证,§3.2】

S2 把 Social Agent 拆成 **Preference + Belief + Reasoning** 三模块。Belief Module 三问题:
1. **agent 是否有内部信念?** — 内部表征(线性 probe)+ 外部行为(Gandhi 2024: Forward/Backward Belief,仅 GPT-4 类人)。
2. **如何增强信念建模?** — **显式图表示**:
   - Sclar 2023:嵌套信念状态的图,让模型从每个角色视角回答。
   - Kassner 2023:**belief graph**(系统信念 + 推断关系,可解释)。
   - Li 2023:**prompt 工程显式表示信念状态**,增强多 agent 协作。
3. **能否修正信念?** — Fan 2023:LLM 信念修正不成熟;**Xu 2023:LLM 的正确事实信念易被修辞/重复操纵** → 印证 Beyond Survival 的 OSR 防御必要性。

### 3.4 二阶 ToM 与高阶推理【S2 实证,§3.3】

- **Suspicion-Agent**(Guo 2023):**second-order ToM** — 不仅预测对手会做什么(一阶),还预测"对手相信我会做什么"(二阶)。【S2】
- **ReCon**(Wang 2023):一阶 + 二阶视角转换,识破并反击虚假信息。
- **K-Level-Reasoning**(Zhang 2024d):高阶 ToM 推理。
- ToM = 把信念/意图/欲望/情绪/知识归因给自己和他人(Premack & Woodruff 1978)。

### 3.5 落地映射:方向 B 的学术依据【推测→已验证可行】

werewolf-mas 当前:
- `_observable_tom_signals`:从投票提取"谁投谁"(一阶:谁怀疑谁)。
- `SYSTEM_BASE` 第2条要求 ToM,但**无二阶结构**。

**方向 B(二阶 ToM 态度网络)的学术依据**:
- S2 §3.3 的 Suspicion-Agent 二阶 ToM = werewolf-mas 应让 agent 建模"对手相信我会做什么"。
- S2 §3.2 的 **explicit belief graph**(Sclar/Kassner/Li)= werewolf-mas 的 `attitude_edges` 结构化态度网络。**用 prompt 工程显式表示信念状态(Li 2023)是已被验证的增强多 agent 协作的方法**。
- 这是方向 B 的直接学术支撑,不是凭空设计。

---

## 四、狼人杀 benchmark 架构对比【S2 §2 + 前两轮】

S2 把狼人杀归为 **social deduction / hidden role** 类(§2, line 578),Werewolf Arena 被引用为统一研究框架(line 591)。

### 4.1 各 benchmark 的 harness 编排(前两轮已读,本轮交叉验证)

| benchmark | harness 编排 | agent 间通信 | 二阶 ToM | 欺骗结构化 |
|---|---|---|---|---|
| Werewolf Arena | 竞价发言(bid 0-4) | 公开广播 | 隐式 | 4 策略(omission/distortion/fabrication/misdirection) |
| WOLF | LangGraph 状态机 | 公开 + 投票 | 无 | 4 分类欺骗基准 |
| MultiMind | ToM 怀疑矩阵 + MCTS | 公开 | ✅ ToM 矩阵 | 隐式 |
| DVM | Predictor/Decider/Discussor | 公开 + RL | 隐式 | 胜率约束 |
| Beyond Survival | 标准 + RR/OSR 干预 | 公开 | 隐式 | 5 维评估 |

### 4.2 关键发现(本轮再确认)

- **Werewolf Arena 的 bid=4 + 被提及者优先**(S2 line 591 + 前轮):werewolf-mas 方向 A 正在补这个。
- **MultiMind 的 ToM 怀疑矩阵**:werewolf-mas 方向 B 的 `attitude_edges` 是其简化版。
- **没有任何 benchmark 把狼人白天话术协同做成显式私聊通道** — werewolf-mas 方向 C 是原创设计(有风险但符合 S2 §3.3 多 agent 协作方向)。

---

## 五、可落地到 werewolf-mas 的设计杠杆(按优先级)

### 【已实证支撑,高优先级】

1. **方向 A:bid=4 真回应 + 结构化 accuses/reply_to**(已实施,task #27)
   - 依据:S2/Werewolf Arena 竞价发言 + AutoGen Swarm handoff(S4)。
   - 状态:代码已改完,pytest 31 绿,待真实对局验证(被 httpx client 复用 bug 阻塞)。

2. **方向 B:二阶 ToM 态度网络**(下轮)
   - 依据:S2 §3.2 explicit belief graph(Sclar/Kassner/Li 2023)+ §3.3 Suspicion-Agent 二阶 ToM。
   - 落地:agent 发言产出 attitudes(support/oppose/neutral),orchestrator 聚合成 `attitude_edges` 注入 observation。让 agent 建模"谁和谁抱团"。
   - **学术支撑强**:Li 2023 已验证"prompt 工程显式信念状态增强多 agent 协作"。

3. **方向 C:狼人白天话术协同**(再下轮)
   - 依据:S4 Swarm + S2 §3.3 多 agent 协作;复用 werewolf-mas 夜间 `_werewolf_deliberation` 私聊模式。
   - 风险:无直接 benchmark 先例,需验证不破坏平衡(胜率 40-60%)。

### 【已实证支撑,中优先级】

4. **三因子记忆检索**(Generative Agents S3):给 `AgentMemory` 加 importance 打分 + relevance 检索,替换纯 recency 的 `recent_observations`。中长期收益(跨天 claim 矛盾)。

5. **OSR 客观发言重写**(Beyond Survival,前轮已记):投票前让 agent 先客观摘要他人发言再投。已在 `vote_instruction` 有提示,可强化为显式两段式。

### 【推测,低优先级】

6. **K-Level-Reasoning**(Zhang 2024d):高阶 ToM。狼人杀轮次短,二阶足够,三阶以上收益边际递减且 prompt 膨胀。

---

## 六、对当前阻塞的诊断呼应

用户指出"httpx client 复用是 harness 设计缺陷"——这符合 S4 的**编排逻辑与 agent 逻辑分离**原则:连接池管理是 harness 基础设施职责。当前 `_get_stream_client` 缓存复用导致连接池污染,是 harness 层 bug,不是 agent 层。修复方向(待复现确认):每次 LLM 调用新建独立 client,或隔离流式连接池。

---

## 引用

- [S1] A Survey on LLM-Based Autonomous Agents (arXiv:2308.11432)
- [S2] A Survey on LLM-Based Social Agents in Game-Theoretic Scenarios (arXiv:2412.03920, TMLR 2025)
- [S3] Generative Agents: Interactive Simulacra of Human Behavior (Park et al., UIST 2023, arXiv:2304.03442)
- [S4] AutoGen v0.4 文档 — Teams (microsoft.github.io/autogen)
- [S5] Prompt Engineering Guide — LLM Agents (promptingguide.ai/research/llm-agents)
- 前两轮:Werewolf Arena (arXiv:2407.13943), WOLF (arXiv:2512.09187), Beyond Survival (arXiv:2510.11389), MultiMind (arXiv:2504.18039), DVM (arXiv:2501.06695)

关联记忆:[[research-round2-2026-07-04]] [[research-implementation-2026-07-04]] [[reference-multiagent-harness-definition]] [[feedback-eight-principles]] [[no-fallback-design]]
