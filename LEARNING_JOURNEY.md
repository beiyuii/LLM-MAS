# 从零搭建本地多Agent系统 — 学习手册

> **项目名称**：LLM-MAS (LLM-based Multi-Agent System)
> **学习者**：北煜
> **学习周期**：2026年4月
> **技术栈**：Python 3.10+ · Anthropic SDK · MiniMax API · YAML · JSON
> **状态**：Phase 1-2 完成（Step 1-5），Phase 3 规划中（Step 6-8）

---

## 这个项目在学什么

这个项目的目标是**从零理解并搭建一套本地化的多智能体协作系统**（LLM-based Multi-Agent System）。

在学术界，这套体系被称为 **LLM-MAS**；在工业界，它有多种名字：Agent Orchestration（LangChain 生态）、AI Crew（CrewAI）、Swarm（OpenAI）、Agentic AI（Gartner）。核心思想是：让多个具备不同专长的 AI 智能体（Agent）各自承担不同的任务，由用户（或系统本身）在它们之间调度和协作，形成一个「虚拟团队」。

**我选择不用任何现成框架（如 CrewAI、LangGraph、AutoGen），而是从最底层的 API 调用开始，一步步把每个模块亲手搭出来。** 目的不是为了造轮子，而是为了彻底理解每一层的原理——就像学 Spring Boot 之前先理解 Servlet 一样。

---

## 最终完成的形态

完成全部 8 个 Step 后，系统将具备以下能力：

```
┌─────────────────────────────────────────────────────┐
│                    你（总指挥）                        │
│              说一句话 → 系统自动分配给对应 Agent        │
└──────────────────────┬──────────────────────────────┘
                       │ Router（自动路由）
        ┌──────────────┼──────────────────┐
        ▼              ▼                  ▼
   ┌─────────┐   ┌─────────┐       ┌─────────┐
   │ 研究员   │   │ 开发者   │  ...  │ 文案     │
   │ T=0.3   │   │ T=0.4   │       │ T=0.7   │
   ├─────────┤   ├─────────┤       ├─────────┤
   │ 独立人设 │   │ 独立人设 │       │ 独立人设 │
   │ 独立记忆 │   │ 独立记忆 │       │ 独立记忆 │
   │ 独立工具 │   │ 独立工具 │       │ 独立工具 │
   └────┬────┘   └────┬────┘       └────┬────┘
        │              │                  │
        └──────────────┼──────────────────┘
                       ▼
              ┌────────────────┐
              │  shared/       │
              │  共享白板       │
              │  知识库 · 任务  │
              └────────────────┘
```

每个 Agent 是一个「虚拟员工」，拥有：
- **人设（Soul）**：system prompt，决定它是谁、擅长什么、怎么说话
- **记忆（Memory）**：独立的对话历史，关机重启不丢失
- **工具（Tools）**：可调用的函数，让它从「只会说话」变成「能动手做事」
- **配置（Config）**：YAML 文件定义一切，新增 Agent 不需要改代码

---

## 学习路线图：8 个台阶

```
Phase 1: 单兵作战              Phase 2: 组建团队              Phase 3: 协作增强
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Step 1: 最小Agent │    │ Step 4: 配置化    │    │ Step 7: 共享记忆  │
│ Step 2: 加上记忆  │    │ Step 5: 路由器    │    │ Step 8: 任务规划  │
│ Step 3: 挂载工具  │    │ Step 6: 聊天界面  │    │                  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
      ✅ 已完成                ✅ 5/6 完成              📋 规划中
```

---

## Step 1：最小 Agent — 一个函数就是一个 Agent

### 学到了什么

Agent 的本质极其简单：**接收输入 → 带着人设调用 LLM → 返回输出**。

整个聊天系统的数据模型就是一个 `messages` 数组，每条消息有 `role`（system / user / assistant）和 `content`。LLM 本身没有「记忆」——它之所以「记得」你说过什么，是因为程序每次都把完整的对话历史发给它。

### 核心认知

- `system` 消息是 Agent 的灵魂，决定了它的行为方式
- `messages` 数组不断增长，每轮对话多两条（user + assistant）
- 对话超过上下文窗口后，前面的消息会被截断 → Agent「忘记」
- OpenAI 和 Anthropic 的 API 接口格式不同：OpenAI 把 system 放在 messages 里，Anthropic 把它作为独立参数

### 代码产出

一个 Python 终端对话程序，通过 Anthropic 兼容协议调用 MiniMax API，实现持续多轮对话。

### 我的思考

> Anthropic API 和 OpenAI API 处理 system prompt 的方式不一样，需要一个 `split_system_and_dialog` 函数做格式转换。这让我意识到：如果以后要支持多个模型提供商，中间一定需要一个适配层。

---

## Step 2：加上记忆 — 让 Agent 记住你

### 学到了什么

记忆系统要解决两个问题：**持久化**（程序关了数据不丢）和 **隔离**（多个 Agent 各自独立）。

最简单的方案：每个 Agent 一个文件夹，里面存 `system_prompt.md`（人设）和 `conversation.json`（对话历史）。人设和对话历史分开存，因为它们的生命周期不同——人设是长期稳定的，对话历史每轮都在变。

### 核心认知

- `system_prompt.md` 作为「单一事实来源（Single Source of Truth）」—— 每次启动从文件重新读取，可以用任何编辑器修改，不需要通过程序命令
- 每轮对话后立即写入文件（毫秒级，不影响性能），防止程序崩溃时丢失数据
- 切换 Agent 时必须先保存当前状态再加载新状态，顺序不能反——否则 A 的数据会存到 B 的文件里

### 代码产出

`agent_memory.py` 模块，支持按 Agent 目录读写人设和对话历史，`/agent` 命令切换 Agent 并自动保存/加载。

### 我的思考

> 关于记忆系统的更完整设计，我想到了三文件模型：
> - **Soul（灵魂）**：Agent 的认知偏向和基础人设
> - **User（用户画像）**：我的行为标签、语气偏好、意图模式
> - **History（历史）**：任务状态、标签、todolist、最近 N 轮原文
>
> 这套设计跟学术界的 CoALA 框架（语义记忆 / 用户画像 / 情景记忆）几乎一致。但在 MVP 阶段先做最简版本，验证可行后再升级。**先让它跑起来，再让它跑得好。**

---

## Step 3：挂载工具 — 让 Agent 能「做事」

### 学到了什么

Tool Use（函数调用）的本质：**LLM 不执行工具，它只做决策**。LLM 判断需要什么工具、传什么参数，你的代码真正执行，结果喂回 LLM，它再基于结果生成最终回答。

整个过程至少需要**两次 API 调用**：第一次 LLM 返回 `tool_use` 块，你执行工具；第二次把 `tool_result` 发回去，LLM 给出最终回答。如果 LLM 需要多个工具，它可以在一次回复里输出多个 `tool_use` 块。

这个循环就是 **ReAct 循环**（Reason-Act）：推理 → 行动 → 观察 → 再推理，直到任务完成。

### 核心认知

- 工具定义 = 名字 + 描述 + 参数 JSON Schema，跟 MCP 的 Skill 几乎一模一样
- 工具描述的精准度**直接决定**调用的准确率——描述太宽，Agent 对什么问题都调用工具；描述太窄，该调的时候不调
- 安全沙箱很重要：`read_file` 工具必须限制在项目目录内，防止通过 `../` 读取系统文件

### 代码产出

`llm_tools.py` 模块，包含 7 个工具（`get_current_time`、`read_file`、`list_files`、`grep_in_file`、`get_runtime_env`、`read_file_range`、`text_file_statistics`），以及完整的 ReAct 循环实现。

### 我的思考

> Tool Use 跟 MCP 的关系：Tool Use 是底层机制（LLM 怎么调用工具），MCP 是标准化协议（工具怎么描述、怎么发现、怎么连接）。先理解底层，后面接 MCP 就知道它到底帮你省了什么。

---

## Step 4：多 Agent 配置化 — 零代码扩展

### 学到了什么

当系统里有多个 Agent 时，如果每加一个 Agent 都要改代码（硬编码 ID、显示名、工具列表），维护成本会随 Agent 数量线性增长。解决方案是**配置和代码分离**——跟 Spring Boot 的 `application.yml` 是同一个思路。

每个 Agent 的所有信息都在一个 `config.yaml` 文件里：名称、描述、模型、温度、工具列表、人设。程序启动时扫描 `agents/` 目录自动发现所有 Agent。

### 核心认知

- YAML 比 JSON 更适合人类编辑：支持注释、可读性高、缩进代替大括号
- `description` 字段不是给人看的装饰，是给 Router 看的决策依据
- 工具注册表模式（`TOOL_REGISTRY`）：所有工具集中注册，每个 Agent 通过名称列表按需引用
- 程序不应因某个 Agent 的配置文件损坏而整体崩溃

### 代码产出

`agent_config.py` 模块，含 `AgentConfig` 数据类、自动发现、配置加载/保存、交互式创建向导。每个 Agent 文件夹包含 `config.yaml` + `system_prompt.md` + `conversation.json`。

### 目录结构

```
agents/
├── researcher/
│   ├── config.yaml           # 角色/模型/工具/人设
│   ├── system_prompt.md      # 人设（与 config 同步）
│   └── conversation.json     # 对话历史
├── developer/
│   └── ...
├── writer/
│   └── ...
└── product_manager/          # 纯配置创建，零代码改动
    └── ...
```

### 我的思考

> 配置化最大的好处不是「方便加 Agent」，而是**让调优手段从「改代码」变成「改文字」**。调一个 Agent 的行为，只需要修改它的 config.yaml 里的 persona 或 description，不需要碰任何 Python 文件。这是工程思维和产品思维的交汇点。

---

## Step 5：路由器（Router）— 自动任务分配

### 学到了什么

Router 的实现出人意料地简单：**再调用一次 LLM**。把所有 Agent 的 name 和 description 拼成一段提示，加上用户的输入，问 LLM「应该交给谁」，它返回一个 Agent 名称。

Router 有自己的配置文件 `router.yaml`，独立于任何 Agent，使用低温度（0.1）和少量 token（50）来保证分类的确定性和成本效率。

### 核心认知

- Router 的准确率取决于每个 Agent 的 `description` 写得多好——description 就是 Router 的「决策依据」
- 双模式设计：`specified`（手动指定 Agent）和 `auto`（每轮自动路由），用户可以根据场景选择
- 对于模糊任务（如「帮我写一篇关于 AI 框架的博客」），Router 可能分给研究员也可能分给文案——这不是 Router 的 bug，而是需要任务拆解（Planner）来解决的问题

### 代码产出

`router.py` 模块，含 Router 配置加载、system prompt 构造、输出解析（含多级容错）、路由调用。`router.yaml` 配置文件。

### 我的思考

> Router 本质上是一个**分类器**，而不是规划器。它回答的问题是「这句话归谁管」，不是「这个任务应该怎么拆」。后者是 Step 8（Planner）的职责。
>
> 关于优化：可以在 Router 前面加一层关键词匹配作为「快速通道」，命中规则的直接分发，不走 LLM。但 MVP 阶段先不做，先验证 LLM Router 的准确率够不够用。

---

## Step 6-8：待完成的增强

### Step 6：聊天界面（Streamlit）

将终端交互升级为可视化界面——左侧 Agent 列表，右边对话窗口，就像一个「员工管理面板」。纯 UI 工作，不涉及新的架构概念。

### Step 7：共享记忆 + 向量数据库

实现 Agent 间的信息共享和语义检索：

- **共享白板**（`shared/`）：所有 Agent 可读写的公共空间
- **向量数据库**（ChromaDB）：语义级别的记忆检索，支持「我们上次讨论的那个 XXX 结论是什么」这类模糊查询
- **三文件记忆模型**：Soul（人设）+ User（用户画像）+ History（任务状态）

### Step 8：任务规划器（Planner）

从 Level 1（Router，单任务分发）升级到 Level 2（Planner，多步编排）：

- 用户给一个复合任务 → 系统拆解为子任务 → 按依赖关系逐步分发 → 汇总结果
- 例如：「帮我写一篇 AI 框架的博客」→ 研究员调研 → 结果传给文案 → 文案写作 → 返回用户

---

## 关键设计理念

### 1. 先让它跑起来，再让它跑得好

每一步都产出一个**可运行的系统**，而不是搭到最后才能看到结果。Step 1 完成后就能对话，Step 2 完成后就有记忆，Step 3 完成后就能用工具。立刻有正反馈。

### 2. 配置和代码分离

Agent 的行为由 YAML 配置决定，不由 Python 代码决定。调优 Agent = 改文字，不是改代码。这是系统可维护性的基础。

### 3. 理解 > 手写

在 AI 辅助编程的时代，「能不能手写每一行代码」不是关键能力。关键能力是：**能读懂**（看到代码知道它在做什么）、**能判断**（知道它写得对不对）、**能指挥**（知道应该让 AI 写什么）。每一步完成后通过「为什么」的问题来验证理解。

### 4. 每个模块都有对应的学术概念

| 我搭的模块 | 学术名称 | 框架中的对应 |
|-----------|---------|------------|
| system_prompt.md | Persona / Profile | CrewAI 的 role + goal + backstory |
| conversation.json | Episodic Memory | CoALA 的情景记忆 |
| llm_tools.py + ReAct 循环 | Tool Use / Function Calling | MCP 的 Tools 原语 |
| config.yaml | Agent Definition | AgentScope 的声明式配置 |
| router.py | Task Router / Classifier | CrewAI 的 Hierarchical Process |
| shared/ (规划中) | Blackboard Architecture | LbMAS 的共享黑板 |

---

## 技术栈

| 层级 | 技术选择 | 选择理由 |
|------|---------|---------|
| 语言 | Python 3.10+ | AI 生态最成熟 |
| LLM 调用 | Anthropic SDK → MiniMax 网关 | Anthropic 协议兼容性好 |
| 配置 | YAML + Pydantic-style dataclass | 人类可编辑、有注释、结构化 |
| 持久化 | JSON 文件 | 最轻量、无依赖、可直接查看 |
| 环境管理 | Conda + pip | 跨平台一致性 |
| 版本控制 | Git | 标准工程实践 |

---

## 如何运行

### 环境搭建

```bash
# 克隆项目
git clone <repo-url>
cd LLM-MAS/step-1

# 创建环境
conda env create -f environment.yml
conda activate llm-mas-step1

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入 MINIMAX_API_KEY
```

### 启动对话

```bash
# 指定模式：手动选择 Agent
python llm_client.py --mode specified --agent researcher

# 自动模式：Router 根据你的话自动分配 Agent
python llm_client.py --mode auto
```

### 常用命令

| 命令 | 说明 |
|------|------|
| `/agents` | 列出所有已配置的 Agent |
| `/agent <id>` | 切换到指定 Agent（指定模式） |
| `/clear` | 清空当前 Agent 的对话历史 |
| `/tools` | 查看当前 Agent 的工具列表 |
| `new agent` | 交互式创建新 Agent |
| `quit` | 退出程序 |

### 新增 Agent（零代码）

创建 `agents/<名称>/config.yaml`，重启程序即可使用：

```yaml
name: designer
display_name: 设计师
description: UI/UX 设计思路、交互逻辑、视觉规范
model: MiniMax-M2.7
temperature: 0.6
max_tokens: 4096
tools:
  - read_file
  - list_files
  - get_current_time
persona: |
  你是一个资深的 UI/UX 设计师，擅长交互设计和视觉规范...
```

---

## 参考资源

### 学术论文

- [A Survey on LLM-based Multi-Agent System](https://arxiv.org/abs/2412.17481) — LLM-MAS 综述（IJCAI 2024）
- [Exploring Advanced LLM Multi-Agent Systems Based on Blackboard Architecture](https://arxiv.org/abs/2507.01701) — 黑板架构在 LLM-MAS 中的应用
- [Towards a science of scaling agent systems](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/) — Google Research 关于多 Agent 系统扩展的实证研究

### 开源框架（参考但未使用）

- [CrewAI](https://github.com/crewAIInc/crewAI) — 角色定义最直观的多 Agent 框架
- [AgentScope](https://github.com/agentscope-ai/agentscope) — 阿里巴巴出品，中文生态友好
- [OpenAI Swarm](https://github.com/openai/swarm) — 极简教学框架，理解 Handoff 概念
- [PicoAgents](https://github.com/victordibia/designing-multiagent-systems) — 从零搭建多 Agent 的教学代码

### 协议标准

- [MCP（Model Context Protocol）](https://modelcontextprotocol.io/) — Anthropic 提出的 Agent-工具通信协议
- [A2A（Agent-to-Agent Protocol）](https://a2a-protocol.org/) — Google 提出的 Agent 间通信协议

---

## License

MIT
