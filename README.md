# LLM-MAS：从零搭建本地多Agent系统

一个**从底层 API 调用开始**，不依赖任何框架，逐步搭建完整多智能体协作系统的学习项目。

## 这是什么

这不是一个开源框架，而是一份**带代码的学习手册**。通过 8 个递进的步骤，从「一个能聊天的函数」搭到「能自动分配任务、有独立记忆和工具的多 Agent 系统」。每一步都有可运行的代码、设计决策的思考过程、以及验证理解的自测题。

## 当前进度

| Step | 主题 | 状态 | 核心能力 |
|------|------|------|---------|
| 1 | 最小 Agent | ✅ | API 调用 + 多轮对话 |
| 2 | 持久化记忆 | ✅ | JSON 存储 + Agent 隔离 |
| 3 | 工具调用 | ✅ | Tool Use + ReAct 循环 |
| 4 | YAML 配置化 | ✅ | 零代码新增 Agent |
| 5 | 自动路由 | ✅ | LLM Router 任务分发 |
| 6 | 聊天界面 | 📋 | Streamlit 可视化 |
| 7 | 共享记忆 | 📋 | 向量数据库 + Agent 间通信 |
| 8 | 任务规划 | 📋 | 多步编排 + Planner |

## 快速开始

```bash
cd step-1
conda env create -f environment.yml
conda activate llm-mas-step1
cp .env.example .env   # 填入 MINIMAX_API_KEY

# 指定模式
python llm_client.py --agent researcher

# 自动路由模式
python llm_client.py --mode auto
```

## 项目结构

```
step-1/
├── llm_client.py         # 主入口：对话循环 + Agent 管理
├── agent_config.py       # YAML 配置加载
├── agent_memory.py       # 记忆持久化
├── llm_tools.py          # 工具注册表 + 执行
├── router.py             # LLM Router 自动分发
├── router.yaml           # Router 配置
├── agents/               # 每个 Agent 一个文件夹
│   ├── researcher/       #   config.yaml + system_prompt.md + conversation.json
│   ├── developer/
│   ├── writer/
│   └── product_manager/
└── .env                  # API Key（git ignored）
```

## 详细学习记录

完整的学习过程、每一步的思考和设计决策，见 [LEARNING_JOURNEY.md](./LEARNING_JOURNEY.md)。

## 技术栈

Python 3.10+ · Anthropic SDK · MiniMax API · YAML · JSON

## License

MIT
