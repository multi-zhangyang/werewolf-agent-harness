# 狼人杀多智能体对抗框架

[![CI](https://github.com/multi-zhangyang/werewolf-agent-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/multi-zhangyang/werewolf-agent-harness/actions/workflows/ci.yml)

这是一个面向隐藏身份博弈、多人对抗实验和可审计模型评估的 Agent Harness。
它不是让一个聊天机器人扮演多个角色，而是为每个玩家创建独立、持续存在、拥有
私有记忆和工具调用边界的 Agent，并由确定性环境负责规则、权限、结算与证据记录。

狼人杀与 Cipher Council 是当前的
精确版本 environment；产品核心不是聊天室，也不是把若干角色提示词包装成聊天机器人。

## 当前状态

- 后端单元、集成、协议、隔离、artifact 和对抗场景测试均纳入 GitHub Actions。
- 前端在 Node.js 20 下执行锁定依赖安装、TypeScript 检查和 Vite 生产构建。
- 生产 Agent 只通过真实 LLM 或真人输入做决定，不存在脚本化发言、投票或行动回退。
- 支持 OpenAI-compatible Chat Completions、OpenAI Responses 和 Anthropic Messages
  三种标准协议，不按供应商、网关或模型名称增加隐藏分支。
- 真实模型 smoke artifact 必须通过请求/终态、工具调用、环境消费和凭据扫描验证；
  CI 本身不会使用或请求任何真实 API key。

Harness 负责构造每个席位可见的 observation、声明合法动作、调度 deadline、执行规则并记录事实。LLM 或真人负责做决定。生产路径不会用脚本补发言、自动改投、伪造模型响应、二次模型润色或完成后再切片冒充流式输出。

生产架构严格是一名玩家一个持续存在的 `AgentActor`。每个座位拥有独立的
事实记忆、主观角色信念、对手模型、策略/伪装计划、公开承诺账本和 RNG；共享
`LLMRouter` 只复用无会话的网络传输与预算，不是共享大脑。构造器会拒绝 Actor
或 Memory 复用、缺席位以及 seat/name/role 错绑。

## 决策边界

每个生产决策都经过同一条链路：

```text
Environment
  → ActionRequest(observation, legal_actions, deadline)
  → DecisionRuntime
  → AgentProtocol.decide(request)
  → DecisionEnvelope | linked response failure
  → protocol validation
  → RulesEngine
  → immutable Transcript
```

关键语义：

- 一个已接受的 `ActionRequest` 对应且只对应一个终态：`DecisionEnvelope`、一条关联同一
  `request_id` 的结构化失败记录，或明确的 room/run cancellation 记录。
- `DecisionEnvelope.request_id` 和席位必须与请求匹配。
- 公开发言直接取自同一个 envelope 的 `decision.speech`，不会再调用模型改写。
- `SKIP` 是明确决定。系统只记录 skip/resolution，不会替 Agent 说“（沉默）”“（倾听）”或“（无遗言）”。
- 非法目标会保留在原 envelope 中并产生 `decision_envelope_rejected`；harness
  不会偷偷改成另一个目标或伪装成 `SKIP`。
- 模型调用失败、真人超时和 phase deadline 都会产生可审计失败，不会触发
  scripted fallback，也不会伪装成 `SKIP`。
- `thought`/`reasoning` 是私有 trace，不会进入公共发言。
- 工具循环的初始请求和 `get_legal_actions` 都携带本局 `visible_seats` /
  `alive_seats`。所有含座位的工具 schema 按当前 roster 或精确
  `LegalAction.target_seats` 生成，幽灵座位会先在参数校验层被拒绝；环境仍是动作
  合法性的最终裁决者。
- 常见的“先读上下文再决策”路径优先调用一次 `read_turn_context`：它返回有界的
  座位私有快照，合并精确合法动作/目标、私有事实、近期公开事件/投票/声明、主观
  belief/策略状态和本座承诺。需要更新多项认知时优先用一次
  `update_private_state` 原子提交；细粒度读取/更新工具仍可用于快照未覆盖的细节。
- Agent 的 `private_state` 会跨回合保存主观 belief、候选/选中策略、二阶视角和
  伪装计划；它只属于该座位，不是环境真值或独立审计评分。
- `read_public_events` 默认返回 12 条、最多 24 条；长期窗口只保留发言和遗言，
  并去掉当前 events/today speeches 已经表达的重复项。`get_beliefs` 不夹带公开
  commitments，后者只能通过 `get_commitments` 读取。
- 公开 claim 可以故意与私有 belief 不同。RulesEngine 只校验字段和动作，不会
  因真实身份不符而纠正狼人假跳预言家等合法欺骗。
- 狼队夜间先各自产生 exact team-private 消息，再让每只狼看到所有消息后独立
  提交最终击杀票；不存在替狼队统一思考的中央 Agent。
- 模型自报的怀疑、欺骗、态度或总结不是独立真值，本项目不把它们包装成审计指标。

当前生产 Agent adapter 有三类：

- `AgentActor`：通过真实 LLM API 生成结构化 decision。
- `CoreToolActor`：供环境中立 Core 协议使用；它把该请求的每个 `ActionOption` 编译为
  一个终结函数，要求模型恰好调用一个，不以普通 chat 文本代替动作。每个实例只绑定一个
  `actor_id`、lock、模型配置、私有有界回合记忆和管理员 trace sink。该记忆只回放此
  Actor 此前获准看到的 observation 与自己已经提交的动作，不保存 provider reasoning，
  也不会进入其他 Actor 的 prompt 或公共 transcript。
- Human seat：通过同一个 `ActionRequest` / `DecisionEnvelope` 边界提交真人输入。

测试可以使用本地 mock，但生产 runner 没有 scripted/replay Agent 工厂。

## 快速开始

要求 Python 3.12+ 和 Node.js 20+。后端支持三种标准协议：

- `openai`：OpenAI-compatible Chat Completions
- `openai_responses`：OpenAI Responses
- `anthropic`：Anthropic Messages

安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

只使用 `WEREWOLF_*` 配置；项目不会回退读取进程里的 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`：

```bash
export WEREWOLF_LLM_PROVIDER=openai
export WEREWOLF_LLM_API_BASE=https://your-compatible-gateway.example/v1
export WEREWOLF_LLM_API_KEY=replace-locally
export WEREWOLF_LLM_MODEL=your-model-id
```

结构化输出有三种通用模式：

- `WEREWOLF_LLM_RESPONSE_FORMAT` 留空且
  `WEREWOLF_LLM_USE_JSON_FORMAT=false`：不发送 provider `response_format`，仍要求
  模型返回普通 JSON，并由 Router 解析。这是兼容性最宽的模式。
- `WEREWOLF_LLM_USE_JSON_FORMAT=true`：发送旧式 `{"type":"json_object"}`。
- 显式设置 `WEREWOLF_LLM_RESPONSE_FORMAT`：该值优先于旧开关。标准
  `json_schema` descriptor 会在 Chat Completions 中作为 `response_format` 发送，
  在 Responses API 中映射为等价的 `text.format`。

完整 Werewolf 对局包含发言、投票、狼队协商、遗言和多种夜间动作，当前配置的
`response_format` 会全局应用到这些请求。因此完整对局推荐使用通用 JSON object：

```bash
export WEREWOLF_LLM_RESPONSE_FORMAT='{"type":"json_object"}'
```

`json_schema` 适合所有调用共享同一响应形状的独立 Harness 集成。只允许
`target_seat` 的 strict schema **不适合完整 Werewolf 对局**：它会按设计拒绝
必须返回 `speech`、`team_message`、`bid` 和 `private_state` 的请求。若要在完整
对局使用 strict schema，应先实现并记录 request-specific schema，而不是配置一个
只覆盖某类动作的全局 schema。

配置只接受标准 `json_object` 或 `json_schema` descriptor。非 JSON 值、未知字段、
非法名称和不符合 JSON Schema Draft 2020-12 元语法的 schema 会在发起模型调用前
被拒绝。schema、reasoning 和 thinking 的安全化配置会写入 RunSpec provenance，
因此这些请求参数变化会改变 `run_spec_hash`，但 API key 不会进入 manifest。

也可以复制 `.env.example` 为本地 `.env`。不要提交 `.env`，不要把真实 key 放进命令历史、文档、截图、artifact 或 issue。

### Web runtime

开发模式使用两个进程：

```bash
# 终端 1
source .venv/bin/activate
python -m src.api.server

# 终端 2
cd frontend
npm run dev
```

访问 `http://localhost:5173`。Vite 将 `/api` 和 `/ws` 代理到后端。

生产式本地运行：

```bash
cd frontend && npm run build && cd ..
python -m src.api.server
```

访问 `http://localhost:8000`。前端是 Harness Console：展示 environment 事件、请求/响应配对、合法动作、规则结算、失败和授权范围内的私有 reasoning；它不是 chat transcript UI。

### 离线真实 Harness run

```bash
python -m src.harness.cli \
  --seed 100 \
  --runs 1 \
  --turn-policies fixed_round_robin \
  --artifact-root artifacts
```

`--runs` 表示**每个调度策略**的运行数；总局数是 `--runs × 策略数`。比较两个策略时可传 `--turn-policies fixed_round_robin,bid_reply`。`--policy-order abba` 要求恰好两个策略且 `--runs` 为偶数；例如 `--runs 2` 会调度 4 局，顺序为 `fixed_round_robin, bid_reply, bid_reply, fixed_round_robin`。同一配对内共享 role/actor/orchestrator 三类 seed，配对之间使用不同 seed。

真实 provider 验证保持同一套 CLI 和标准 OpenAI-compatible Router，不为某个
endpoint 或模型增加分支。验证时可把 `WEREWOLF_LLM_CONCURRENCY=1` 作为低负载
配置，并在完成后运行 `python -m src.harness.smoke <run-dir>`；verifier 会从磁盘
重新核对 manifest、summary、transcript 的完整性和请求/响应配对。最新 r12 完整
六人验证以 `status=completed` 结束：69 次 provider 调用全部成功、0 次失败/重试，
31 个 request/terminal-response 配对和 69 个 tool call/result 配对均完整，31 个
决策均被环境消费；其中 29 个规则动作被接受，另外 2 个狼刀提议按 plurality 记为
`not_selected`，不是丢失请求。运行时解析了六个互不复用的 `seat:1` 至 `seat:6`
Actor。该运行证明真实 provider 门禁可通过；发布证据仍须把脱敏 artifact 附到精确
代码 revision。外部模型的非确定性不属于复现保证。

可选的 `--seat-permutation cyclic` 会按 case 循环移动玩家身份；同一个 policy
配对使用完全相同的 seat permutation，下一 case 再移动一席。程序化调用可在
`ExperimentSpec.metadata` 中设置 `seat_permutation_mode="cyclic"`。每个具体
`RunSpec` 都记录 1-based `seat_permutation`、`seat_rotation`、`permutation_id`
和原始 `source_player_names`；默认 `fixed` 不增加这些字段，因此旧 schedule 和
summary resume 保持兼容。传入的 `seat_model_configs` 以 source seat 为身份，
会用同一 permutation 同步移动到实际 physical seat；resolved `seat_models` 和
真实调用使用同一映射，因此不同模型的 cross-play 不会只轮换显示名称。

角色布局和 persona 是独立的正式实验控制。`--role-layout-mode fixed` 固定同一
物理座位角色布局；`counterbalanced` 只在完整控制块结束后切换到下一个可复现
布局。`--persona-mode fixed|randomized|counterbalanced` 使用带版本的 persona
catalog 并把逐座位 prompt 文本、source/physical seat 映射、独立 seed 和 assignment
hash 写入 RunSpec。显式 persona 在创建 Actor 后逐席绑定，但仍严格保持一个座位一个
独立 Actor/Memory/PrivateAgentState/RNG。

座位循环和 persona counterbalance 使用完整笛卡尔交叉，不走同步“对角线”。例如
6 个座位、6 个 persona profile 且启用 `--seat-permutation cyclic` 时，每个角色
布局、每种 turn policy 都需要 36 个 case 才是完整控制块；不足 36 会 fail-closed。
座位是快轴、persona 是慢轴，同一 case 的所有 policy 共享完全相同的角色、身份、
persona 和模型映射。

批量结果中的 `strategy_evaluation` 只汇总 environment truth、已接受的
`decision_consumed` trace 和确定性的失败记录，不使用模型自评或赛后生成文本。
它提供 `overall`、`by_turn_policy`、`by_role`、`by_seat`、`by_persona` 和
`by_role_layout` 视图。decision
failure rate、wolf council coverage 和 wolf vote agreement 都同时保存分子与
分母；belief Brier 使用 `belief_brier_sum / belief_observation_count` 跨局加权，
不会对各局均值做简单平均。false role claim rate 与 false seer result 分开；
seer contradiction 只报告可复算 count。新生成的行始终是 run-summary v4，失败
或无策略分析的行也不会冒充旧 v3；旧 v3 JSONL 仍可恢复。

summary JSONL 是恢复 checkpoint/cache，不是独立的评估证据。它里面的 digest 和
provenance 标记可以被一起改写，因此 JSONL 重载后只进入普通事实总计，不进入
operational、strategy、deception 或 comparative evaluation。发布级复算应使用已提交
artifact 目录调用 `load_verified_run_summary(run_dir)`，由 verifier 从三件套重新
构建 transcript 并重新推导 summary；结果对象上的 attestation 不会写入 JSON。

`operational_evaluation` 按 overall/policy 汇总 provider/parse failure 分子分母、
latency、token cost、公开 accuse 与后续投票目标的时序对齐，以及 transcript
visibility audit。后者只把公开投影中的 hidden marker 违规计为私有信息泄漏；
私有 reasoning 的正常存在不是“泄漏”。provider failure 以 calls 为分母，
parse/lossy/incomplete 以 structured responses 为分母；投票对齐是可观察关联，
不声称因果影响。

每局 `analysis.decision_trace_metrics` 还从同一份 trace 重算 Agent 工具循环成本：
model generation、tool call/result、成功/失败、失败 code/tool、有失败的请求数，
单请求最大 generation/tool-call 数，以及 provider 输入历史压缩次数、涉及请求数、
单次最多压缩的完整工具组、压缩前/后的字符峰值和未能满足目标上限的次数。summary
只保留这些可复算计数和安全约束代码，不复制工具参数、错误正文或私有 reasoning；
可以直接发现“最终决策成功但中间反复调用失败工具”或上下文持续放大的退化路径。

完整 `AgentSession.messages` 和 admin 审计 trace 不会被历史压缩改写。压缩只作用于
下一次发给 provider 的副本，并且只把较旧、完整配对的 assistant tool-call + tool-result
组替换成有界摘要，保留最近完整工具组。admin capability 保护的 trace 还可展示经过
递归脱敏和长度/深度限制的 tool arguments，因为它们常是工具模型唯一可见的结构化
计划；这些参数不会进入其他 Agent 的观察或任何公共事件流。

新生成的 summary 还包含 `deception_metrics`，跨局结果提供独立的
`deception_evaluation`。它只从公开、已接受且与 environment truth 冲突的身份声明
或查验结果生成 signal，再比较对手在 signal 前最近一次和 signal 后首次记录的
`belief_state_after`。正的 `deception_direction_shift` 表示观察者的狼人概率朝假声明
希望的方向移动。重复声明落在同一观察窗口时只计一次；无法同时找到前后 checkpoint
的 signal 保留为 unpaired。该指标是可重算的时序关联，期间其他公开信息也可能影响
信念，绝不解释为谎言造成的因果效果。

#### Werewolf 规则集与牌组

`werewolf.classic@1` plugin 当前只实现一个精确规则集：
`ruleset_id=classic.v1`。这不是多变体系统；任何其他 ruleset ID 都会在创建
session、构建 Agent 或调用模型前 fail-closed。Werewolf `RunSpec` 会记录
`ruleset_id`，legacy wrapper 生成的 `CoreRunSpec.environment_config` 也携带同一
字段；legacy/core manifest 分别内嵌对应的完整 spec，因此规则集属于可哈希的
run provenance。

自定义 `role_deck` 必须与玩家数一致，至少包含一名狼人和一名非狼人，并且只
能使用已有完整执行路径的角色。`werewolf` 与 `villager` 可以重复；`seer`、
`doctor`、`witch`、`guard`、`hunter` 在 `classic.v1` 中各最多一张。相同校验
同时位于 RunSpec/plugin 边界和 `RulesEngine.deal_roles` 最终防线。原有 6–12
人默认牌组分布保持不变，仍不会默认加入 Doctor。

#### RunSpec 迁移与 Actor provenance

canonical `CoreRunSpec` 使用精确 schema `agent-harness.run-spec.v1`，并通过
`actors` 记录 credential-free Agent 绑定：`default_model`、按 environment
`actor_id` 索引的 `model_overrides`，以及排序去重的 `human_actor_ids`。这些字段
中的模型配置只保存安全 manifest，human 列表只保存 Actor ID；API key、
Authorization 或其他凭据都不能进入 spec，真人 Actor 也不能同时拥有模型
override。

`load_core_run_spec()` 只接受精确的 Core v1，或通过唯一的
`legacy_werewolf_run_to_core()` 映射迁移 `werewolf.harness.spec.v3`。缺失或未知
`schema_version` 会 fail-closed。迁移把 legacy seat 配置规范化为 `seat:<n>` Actor
ID，并在 Core metadata 中记录 `source_schema_version` 与 `legacy_spec_hash`。
legacy hash 只是来源证据，Core spec 会计算自己的不同 hash；离线 Werewolf
wrapper 已复用这条映射，不再维护第二份字段转换逻辑。

`werewolf.classic@1` plugin 不再私自持有第二套 Actor 解析路径。每个物理座位都以
canonical `seat:<n>` ID 调用 `EnvironmentRunContext.resolve_agent()`，由 Core
run-scoped `AgentRegistry` 保证同一 ID 始终返回同一对象、同一对象不能跨席复用。
plugin 同时以 `CoreRunSpec.actors` 校验完整执行绑定：声明不能指向牌桌外座位，
每席必须由默认/覆盖模型或 human binding 覆盖；resolved human/model 类型必须与
`human_actor_ids` 一致，模型 Actor 的安全 manifest 必须与对应 default/override
manifest 完全一致。缺席、错席、类型错配或 provenance 错配都会在对局执行前失败。

#### Cipher Council 环境

`council.cipher@1` 与 `council.cipher@2` 是不依赖 Werewolf 模块的生产 Core
environment 版本。两者都使用 `council:<n>` Actor ID、`roles` 与 `order` 两个显式
seed，并要求 5--10 名参与者。隐藏的 Cipher 阵营在公开讨论、提名和公开投票中可以误导
他人；通过的队伍再同时提交秘密任务承诺，Council 只能提交 `support`，Cipher 可以提交
`support` 或 `sabotage`。

v1 保持这套基线流程。v2 在每次公开提案尝试前增加一次 Cipher 阵营私密策略协商：每个
Cipher 都由其自己的 Actor 同时收到 `send_cipher_strategy_message` 工具请求；同一轮请求只
能读取此前轮次已经投递的协商消息，不能读取本轮同行尚未完成的消息。所有本轮 decision
终结后，只有实际提交的消息才会以私有 `council_cipher_message` event 交给本局全部 Cipher
收件人；Council、公开投影和其他 Agent 都看不到它。跳过或失败只代表该成员本轮没有消息，
不会合成策略、不会产生公共失败事件，也不存在代表阵营作决定的中心 Actor。

授权的上帝视角可以看到这种环境私有 event；持有 room admin capability 的人类 God Console
还可通过独立、长度受限且脱敏的 admin trace 查看模型 reasoning。它不进入 WebSocket 游戏
事件、replay event 或任何 Agent observation，因此不会成为其他 Agent 的信息源。

环境不会替任何 Agent 生成选择：跳过/缺失发言不产生发言；跳过/缺失提名使本次提案失败；
跳过/缺失投票只计为 `absent`，绝不伪造成反对票；缺失秘密承诺会公开任务作废并以
`status=incomplete` 终结，而不是猜测 support 或 sabotage。角色分配、个人秘密承诺，以及
v2 的 Cipher 协商消息都是带精确 `recipients` 的私有事件，公开事件必须显式标记
`visibility="public"`。

环境中立的真实运行入口是 `run_core_llm_environment()`。它按 environment 请求的
`actor_id` 惰性创建独立 `CoreToolActor`，只共享无状态 Router 传输层，并把
`router_stats_delta` 和 `model_calls` 写入 generic result metrics。它使用标准
`openai`、`openai_responses` 或 `anthropic` 路径，不根据 endpoint 或模型名分支；在
OpenAI 路径中，`ModelConfig.max_tokens=0` 仍表示不发送输出 token 上限字段。每个实际
model Actor 在创建前还必须精确匹配 `CoreRunSpec.actors` 中该 actor 的脱敏
default/override manifest；缺失、无效或不匹配会在第一条 provider 请求前失败，因此
artifact 的模型 provenance 不能与真实调用脱节。

#### 通用 Core 真实运行

`src.harness.core_cli` 是 environment-neutral 的单次真实运行入口。它只接受精确
`agent-harness.run-spec.v1` JSON、一个显式的受信本地 `module:attribute` plugin
引用，以及运行时 `WEREWOLF_LLM_*` 配置；spec 中必须已经包含与运行时模型精确匹配
的 credential-free `ActorSpec` manifest，不能把 key 写进 spec。命令会注册并核对
plugin 的精确 `environment.id@version`，再用每个 actor 独立的 `CoreToolActor` 运行，
最后只打印简短安全报告并写入三文件 artifact：

```bash
PYTHONPATH=. python -m src.harness.core_cli \
  --spec runs/cipher-v2.json \
  --plugin src.environments.cipher_council:CipherCouncilV2EnvironmentPlugin \
  --artifact-root artifacts \
  --verify-smoke
```

等价的 Make 入口是：

```bash
make harness-core-real \
  CORE_SPEC=runs/cipher-v2.json \
  CORE_PLUGIN=src.environments.cipher_council:CipherCouncilV2EnvironmentPlugin
```

`--plugin` 会执行本地 Python 代码，只能指向你信任的模块。此命令刻意只接纳
`agent-harness.decision.v1` 的 Core tool contract；历史 Werewolf plugin 使用自己的
legacy actor contract，仍应通过 `src.harness.cli` 运行。这个边界会在任何 provider
请求之前拒绝，避免把错误 actor 类型伪装成失败的模型调用。

使用 `--resume --summary-jsonl <path>` 时，只有 `run_id` 和完整有效
`run_spec_hash` 都匹配的行才会恢复；规则集、牌组、策略、seed、超时或安全化
模型配置变化都会拒绝旧行，避免把不同实验混为一谈。未声明 `ruleset_id` 的旧
输入会规范化为 `classic.v1`，但 resolved spec 现在把该字段纳入 hash，因此不
含该 provenance 的旧 summary 行会被 strict resume 拒绝。

每局只写三种 artifact：

```text
artifacts/<run-id>/
  manifest.json
  summary.json
  transcript.jsonl
```

同一个 `write_run_artifacts()` 入口支持 legacy `HarnessRunResult + RunSpec`
和 environment-neutral `EnvironmentRunResult + CoreRunSpec`。前者写入
`agent-harness.manifest.v2`，后者写入独立版本的
`agent-harness.core-manifest.v1`；两种格式不会通过大量可选字段混成一个
schema。manifest 保存安全配置和 provenance，不保存 API key；
`summary.json` 只包含执行结果和 Router 统计事实。`transcript.jsonl` 保存有序
environment event 和 decision trace，私有 reasoning 只存在于授权的同一
`DecisionEnvelope` trace，不会复制成伪“思考流”。

三个文件采用原子替换，manifest 最后写入并记录 summary/transcript 的
SHA-256 与字节数。`verify_run_artifacts()` 按 manifest 的精确版本分流，未知
schema 会被拒绝；它还验证严格文件集合、普通文件约束、run/spec identity，
以及每条 JSONL 的 transcript schema、run ID、连续序号和 payload hash。
Core manifest 另存经过脱敏的 transcript metadata 和 kind counts，因此可以从
磁盘行重新构造 Transcript、核对 counts 并独立重算 stable digest。writer 会
拒绝预先存在的 run-directory symlink，verifier 也拒绝 artifact 文件 symlink。

真实模型 run 完成后，用同一目录运行 credential-free smoke verifier：

```bash
PYTHONPATH=. python -m src.harness.smoke artifacts/<run-id>
# 或
make harness-verify SMOKE_RUN_DIR=artifacts/<run-id>
```

它先验证三文件完整性，再要求 `status=completed`、非零 Router call count、
每个 `agent_request` 恰好一个终态、至少一个带 `model_call_id` 的有效
`agent_response` 被 `decision_consumed` 引用，并拒绝未脱敏凭据。报告不包含
prompt、response 或 call ID 原文；离线 fixture 通过不等于真实模型门禁通过，
必须另附真实 provider 生成的 artifact。

`load_verified_run_summary()` 是 summary 评估的唯一离线重建入口；旧 manifest 若没有
transcript metadata/counts 仍可做兼容完整性验证，但不会恢复派生评估信任。当前
manifest 没有外部签名，完整目录被同时替换时只能证明内部自洽，不能证明 artifact
来源；需要来源认证时应在部署层增加签名、HMAC 或 append-only digest 锚点。

## 真人席位

Web room runtime 支持真人席位；无人值守的离线 runner 明确拒绝 interactive human seats。创建房间时声明：

```bash
curl -X POST http://localhost:8000/api/rooms \
  -H 'Content-Type: application/json' \
  -d '{"player_names":["A","B","C","D","E","F"],"human_seats":[1]}'
```

创建响应会返回 room admin token 和对应 seat token。开始、trace、赛后 replay 需要 admin token；真人 `play` WebSocket 需要 seat token。详细协议见 [docs/PROTOCOL.md](docs/PROTOCOL.md)。

Interactive migration 的 Phase B 已落地：`room.id`、`GameState.id`、legacy
`RunSpec.run_id`、canonical `CoreRunSpec.run_id`、`Transcript.run_id` 和后续
`ActionRequest.run_id` 使用同一 identity；每个 `seat:<n>` 都经过 run-scoped
`AgentRegistry` 解析并校验 human/model provenance。发牌、两种 spec、唯一 transcript、
逐席 Actor、共享 `DecisionRuntime` 和 room-owned plugin session 先在 detached state 上
构造，全部成功并持久化后才一次性发布为 `running`。后台房间 task 只调用
`run_prepared_environment_run`；Core 独占 run timeout、session/runtime close、bounded
cleanup 和环境结果生命周期，`RoomManager` 只把 Core 结果投影到 REST/WebSocket 房间
状态，并继续拥有 capability、delivery 和持久化边界。`EnvironmentRunEvidence` 的 full
sinks 把房间 source history、同一个 Transcript 和 delivery cursor 作为一次可回滚
持久化提交，成功后才向 WebSocket queue 暴露消息，不产生第二套证据。Core 的
completed/incomplete Harness 终态与校验后的房间终态、replay `room_status` 同次落盘；
Core 新记录恢复时还会强制校验 canonical/legacy hash、两份 spec 的共享语义、逐类
`source_idx` 顺序和 event/decision 共用的严格递增 `trace_seq` 时间线。

交互式 API 当前只支持**单 worker**。`RoomManager` 在进程内拥有 active room、
对局 task/lock、WebSocket client queue、投影后的 delivery stream/cursor、capability
状态及 SQLite snapshot 写入协调；持久化只提供单 owner 重启恢复，并不提供多进程
一致性。启动多个 API worker 会形成相互独立的 room owner，可能拆分 REST/WS
路由、广播和并发 mutation。多 worker/多主机部署必须先增加共享 room ownership、
锁与状态、跨进程 pub/sub、sticky routing/reconnect 语义和分布式限流/费用账本。

## 验证

```bash
source .venv/bin/activate
PYTHONPATH=. python -m pytest -q

cd frontend
npx tsc -b --pretty false
npm run build
```

`pytest` 不调用外部模型。真实联调必须通过 harness CLI 或 Web room runtime，并检查 Router call count 和 transcript 中的 `agent_request` / `agent_response`，不能仅凭 UI 看起来像 AI 就宣称完成了模型验证。

## 目录

```text
src/agent/       LLM/Human Agent adapter、prompt、纯事实记忆、Decision schema
src/game/        狼人杀 environment、状态、规则和 orchestrator
src/environments/通用 Harness 的内建 environment plugin adapters
src/harness/     中立协议、registry、generic runner、legacy adapter、transcript、artifact
src/llm/         三种标准协议 Router、有限重试和调用统计
src/api/         room runtime、REST、WebSocket、权限与投影
frontend/        React Harness Console
tests/           规则、协议、隔离、API、runner 和前端 reducer 回归
docs/            架构、协议和真实参考资料
```

## 当前限制

- 已有精确版本的 environment registry、通用 runner 和两个生产 environment ID（三个
  plugin version）：`werewolf.classic@1`、`council.cipher@1` 与 `council.cipher@2`。
  前者已接入交互式 API；Cipher Council v1/v2 目前由通用 Core runner/离线 artifact
  路径执行，尚未作为房间创建选项暴露在交互式 API 中。
- Interactive Phase B 已统一 canonical `CoreRunSpec`、逐席 `AgentRegistry`、room-owned
  plugin session、共享 `DecisionRuntime`、唯一 transcript 和 Core execution lifecycle；
  legacy Werewolf `RunSpec` 仅作为现有 API/provenance 兼容视图。项目仍不能宣称 100%，
  因为当前 revision 还需要附带新的真实模型 artifact，且 multi-worker、交互式多环境
  选择与独立心理/欺骗真值评估仍是明确边界。
- Werewolf ruleset 目前只有精确的 `classic.v1`；未知版本会被拒绝，但尚未实现
  第二种规则变体或规则 DSL。
- 离线 run 的可复现性限于角色、调度和本地随机源。外部模型本身、网关负载和 provider 更新可能非确定。
- replay 是 ended room 的 transcript 只读投影，不会重放 Agent 或重新驱动 RulesEngine。
- 项目没有独立的欺骗识别、心理状态真值、质量裁判或校准器。
- 交互式 API 是单 worker runtime；SQLite persistence 不会把进程内 room owner、
  task/lock 或 WebSocket delivery 变成分布式协调。开发服务器也不是可直接暴露
  公网的多租户服务，部署限制见 [SECURITY.md](SECURITY.md)。

## 文档

- [架构](docs/ARCHITECTURE.md)
- [Agent / REST / WebSocket 协议](docs/PROTOCOL.md)
- [贡献规则](CONTRIBUTING.md)
- [研究背景与引用边界](docs/REFERENCES.md)
- [质量门禁与完成证据](docs/QUALITY_GATES.md)

当前尚未选择开源许可证；在添加 `LICENSE` 前不要假定获得了复制或再分发授权。
