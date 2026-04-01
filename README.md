# LLM-MAS 学习笔记与实验代码

本仓库用于 **LLM 多智能体系统（LLM-MAS）** 相关学习与实验。课程/练习中主要使用 **Claude** 作为教学与辅助编程环境。

## 内容说明

- **`step-1/`**：通过 MiniMax **Anthropic 兼容** API 调用大模型；支持终端多轮对话、多 Agent 角色（研究员 / 开发者 / 文案作者），并将 **system 提示（soul）** 与 **对话历史** 分别存于 `agents/<角色>/system_prompt.md` 与 `conversation.json`。

## 环境准备

### Conda（推荐）

```bash
cd step-1
conda env create -f environment.yml
conda activate llm-mas-step1
```

### 仅 pip

```bash
cd step-1
pip install -r requirements.txt
```

## 配置与运行

1. 在环境中设置 MiniMax API Key（勿提交到仓库）：

   ```bash
   export MINIMAX_API_KEY='你的密钥'
   ```

2. 启动终端对话（默认 Agent：研究员 `researcher`）：

   ```bash
   cd step-1
   python llm_client.py
   ```

3. 指定角色，例如文案作者：

   ```bash
   python llm_client.py --agent writer
   ```

4. 运行中可用 `/agent developer` 等切换角色（会自动保存当前会话）。

更多指令见运行时的终端提示（如 `/clear`、`xgsys` 修改 system 等）。

## 目录结构（节选）

```
step-1/
├── llm_client.py
├── agent_memory.py
├── environment.yml
├── requirements.txt
└── agents/
    ├── researcher/
    ├── developer/
    └── writer/
        ├── system_prompt.md
        └── conversation.json
```

## 声明

- 请勿将 **API Key**、私密对话写入公开仓库；本仓库 `.gitignore` 已忽略常见密钥文件。
- 第三方 API 使用须遵守相应服务条款。
