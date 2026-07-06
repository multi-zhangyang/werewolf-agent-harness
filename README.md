# 狼烟 · Werewolf MAS

多 Agent 狼人杀后端 + React/Vite/shadcn 前端 SPA。每个 AI 决策来自真实 LLM 调用,规则推进/胜负判定由确定性引擎完成。

## 快速开始

### 环境要求

- Python 3.12+
- Node.js 20+ 或 22+
- 一个标准 LLM API 配置,通过 `WEREWOLF_*` 环境变量提供
  (`openai` Chat Completions、`openai_responses` Responses API 或 `anthropic` Messages API)

```bash
# 1. 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置模型(复制 .env.example 为 .env 并填写 WEREWOLF_* 变量)
cp .env.example .env
# 编辑 .env: WEREWOLF_LLM_API_BASE / WEREWOLF_LLM_API_KEY / WEREWOLF_LLM_MODEL

# 3. 安装前端依赖
cd frontend
npm install
cd ..
```

### 开发模式

开发模式需要两个进程。Vite 会把 `/api` 和 `/ws` 代理到后端。

```bash
# 终端 1:后端
python -m src.api.server

# 终端 2:前端
cd frontend
npm run dev
```

打开浏览器访问 `http://localhost:5173`。

网页路径:创建房间 → 点击 **开始真实对局** → 观战/上帝视角查看 agent 对抗。若先进入 waiting 观战页,顶部 HUD 也提供 **开始真实对局** 按钮。

### 生产/单进程演示模式

先构建前端,再由 FastAPI 服务 `frontend/dist`。

```bash
cd frontend
npm run build
cd ..
python -m src.api.server
```

打开浏览器访问 `http://localhost:8000`。

### 人机混合

创建房间时通过 API 指定 `human_seats: [1]` 即可让 1 号座位由真人操作。
前端切换到 **人机对局** 并连接该座位,轮到操作时会高亮提示;超时未操作则透明 `SKIP`。

```bash
curl -X POST http://localhost:8000/api/rooms \
  -H 'Content-Type: application/json' \
  -d '{"player_names":["A","B","C","D","E","F"],"human_seats":[1]}'
```

## 运行测试

```bash
source .venv/bin/activate
PYTHONPATH=. pytest -q

cd frontend
npm run build
```

也可以使用 Makefile:

```bash
make test
```

真实 LLM smoke 会实际调用模型,需要有效 `WEREWOLF_*` 配置,会产生耗时和费用:

```bash
PYTHONPATH=. python tests/smoke_e2e.py
# 或
make smoke-real
```

## 项目结构

```
frontend/          React + Vite + Tailwind v4 + shadcn/ui 前端
src/
  agent/           Agent 决策、记忆、Prompt、清洗
  api/             FastAPI REST + WebSocket + 房间管理
  game/            规则引擎、状态机、编排器、角色
  llm/             多 provider LLM 路由(真实调用,绝不伪造)
tests/             pytest 测试 + 真实 LLM smoke / 多局统计工具
docs/              架构设计 ARCHITECTURE.md
```

## 协作与协议

- `CONTRIBUTING.md` — 本地开发、测试要求、no-fallback 贡献规则
- `SECURITY.md` — 凭据、信息隔离和公网部署注意事项
- `docs/PROTOCOL.md` — REST / WebSocket / 赛后 analysis 协议说明
- `Makefile` — 常用安装、测试、构建、真实 smoke 命令

许可证尚未选择。正式开源前请在 MIT / Apache-2.0 / AGPL-3.0 / 暂不授权复用之间确认一种,再添加 `LICENSE`。

## 核心原则

- **真实对局**:每个 AI 决策必须来自真实 LLM 调用,失败走深度重试,绝不伪造。
- **引擎主控**:胜负/行动合法性由 `RulesEngine` 判定,LLM 只表达意图。
- **状态-观察分离**:完整 `GameState` 只存在于后端,每个 seat 只能拿到 `PlayerView` 投影。

## 主要入口

- `python -m src.api.server` — 启动 Web 服务
- `cd frontend && npm run dev` — 前端开发服务器
- `cd frontend && npm run build` — 构建生产前端
- `PYTHONPATH=. python tests/smoke_e2e.py` — 跑一局完整 AI 对局(真实 LLM,耗时较长)
- `PYTHONPATH=. python tests/multi_game_stats.py N --jsonl logs/run.jsonl` — 跑 N 局真实多局统计并输出 JSONL/CI
- `python tests/monte_carlo.py` — 随机基线平衡性测试
- `python tests/monte_carlo_seer.py` — 盲信预言家基线测试
