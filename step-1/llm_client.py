"""
使用 Anthropic 兼容协议调用 MiniMax API（非 OpenAI 兼容）。
支持按 Agent 目录持久化 system_prompt.md 与 conversation.json。
"""
import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import anthropic

import agent_memory

# MiniMax Anthropic 兼容网关地址，可通过环境变量覆盖
MINIMAX_ANTHROPIC_BASE_URL: str = os.getenv(
    "MINIMAX_API_BASE",
    "https://api.minimaxi.com/anthropic",
)

# API Key 仅从环境变量读取，请勿在代码中硬编码
MINIMAX_API_KEY: Optional[str] = os.getenv("MINIMAX_API_KEY")

# Anthropic 客户端（MiniMax 网关）
client: anthropic.Anthropic = anthropic.Anthropic(
    api_key=MINIMAX_API_KEY or "",
    base_url=MINIMAX_ANTHROPIC_BASE_URL,
)

# 退出持续对话时识别的命令（小写比较）
EXIT_COMMANDS: Set[str] = frozenset({"quit", "exit", "q", ":q", "bye"})

# 终端展示用名称
AGENT_DISPLAY_NAMES: Dict[str, str] = {
    "researcher": "研究员",
    "developer": "开发者",
    "writer": "文案作者",
}

# 当前绑定的 Agent；None 表示未初始化（不应在终端模式下出现）
current_agent_id: Optional[str] = None

# 对话历史：OpenAI 风格；由 init_agent / switch_agent 加载
messages: List[Dict[str, str]] = []


def init_agent(agent_id: str) -> None:
    """
    从磁盘加载指定 Agent 的 system 与 conversation，绑定为当前会话。

    Args:
        agent_id: researcher / developer / writer
    """
    global messages, current_agent_id
    current_agent_id = agent_id
    messages = agent_memory.load_messages(agent_id)


def switch_agent(new_agent_id: str) -> None:
    """
    切换 Agent：先持久化当前会话，再加载目标 Agent 的记忆。

    Args:
        new_agent_id: 目标 Agent 标识
    """
    global messages, current_agent_id
    if current_agent_id:
        agent_memory.save_messages(current_agent_id, messages)
    current_agent_id = new_agent_id
    messages = agent_memory.load_messages(new_agent_id)


def persist_agent_state() -> None:
    """将当前 messages 写入当前 Agent 的 conversation.json。"""
    if current_agent_id:
        agent_memory.save_messages(current_agent_id, messages)


def split_system_and_dialog(openai_style_messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    将 OpenAI 风格消息拆分为 Anthropic 的 system 与 messages。

    Args:
        openai_style_messages: 含 system/user/assistant 的消息列表

    Returns:
        (system 文本或 None, Anthropic messages 列表)
    """
    system_parts: List[str] = []
    dialog: List[Dict[str, Any]] = []
    for item in openai_style_messages:
        role = item.get("role")
        content = item.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            dialog.append({"role": role, "content": content})
    system_text: Optional[str] = "\n\n".join(system_parts) if system_parts else None
    return system_text, dialog


def extract_text_from_message(message: anthropic.types.Message) -> str:
    """
    从 Anthropic 响应中拼接文本块（兼容含 thinking 等多段 content 的模型）。

    Args:
        message: messages.create 的返回对象

    Returns:
        助手回复的纯文本
    """
    parts: List[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def chat(new_user_message: str, model: str = "MiniMax-M2.7") -> str:
    """
    发送用户消息并获取 AI 回复。

    Args:
        new_user_message: 用户的新消息
        model: MiniMax 模型名（Anthropic 兼容路径下使用，如 MiniMax-M2.7）

    Returns:
        AI 的回复内容
    """
    if not MINIMAX_API_KEY:
        raise ValueError("请设置环境变量 MINIMAX_API_KEY")

    messages.append(
        {
            "role": "user",
            "content": new_user_message,
        }
    )

    system_text, anthropic_messages = split_system_and_dialog(messages)

    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": 1000,
        "temperature": 0.7,
        "messages": anthropic_messages,
    }
    if system_text:
        kwargs["system"] = system_text

    response = client.messages.create(**kwargs)

    assistant_message = extract_text_from_message(response)

    messages.append(
        {
            "role": "assistant",
            "content": assistant_message,
        }
    )

    persist_agent_state()
    return assistant_message


def get_messages() -> List[Dict[str, str]]:
    """获取当前对话历史。"""
    return messages


def clear_messages() -> None:
    """清空对话历史，保留 system prompt（来自当前 messages 首条 system）。"""
    global messages
    system_prompt = (
        messages[0]
        if messages and messages[0]["role"] == "system"
        else {
            "role": "system",
            "content": agent_memory.DEFAULT_SYSTEM_FALLBACK,
        }
    )
    messages = [system_prompt]
    persist_agent_state()


def set_system_prompt(new_content: str) -> None:
    """
    修改当前会话的 system：覆盖首条 system，并写回 system_prompt.md 与 conversation.json。

    Args:
        new_content: 新的 system 文本（不可为纯空白）
    """
    global messages
    if not new_content.strip():
        raise ValueError("system 内容不能为空")
    if messages and messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": new_content}
    else:
        messages.insert(0, {"role": "system", "content": new_content})
    if current_agent_id:
        agent_memory.write_system_prompt_md(current_agent_id, new_content)
    persist_agent_state()


def run_terminal_chat(model: str = "MiniMax-M2.7", agent_id: str = "researcher") -> None:
    """
    在终端中持续多轮对话，直到用户输入退出命令或 EOF。

    Args:
        model: MiniMax 模型名
        agent_id: 启动时加载的 Agent
    """
    if not MINIMAX_API_KEY:
        print("错误：请设置环境变量 MINIMAX_API_KEY 后再运行。", file=sys.stderr)
        sys.exit(1)

    init_agent(agent_id)
    label = AGENT_DISPLAY_NAMES.get(agent_id, agent_id)

    print("=== MiniMax 终端对话（Anthropic 兼容）===")
    print(f"当前 Agent: {agent_id}（{label}）| 数据目录: agents/{agent_id}/")
    print("输入内容后回车发送；quit / exit / q / bye 结束；/clear 清空对话；")
    print("/agent researcher|developer|writer 切换角色并自动保存当前会话；")
    print("xgsys <新提示词> 修改 system（会写入 system_prompt.md）；单独 xgsys 下一行输入。")
    print("-" * 48)

    while True:
        try:
            user_line = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            if current_agent_id:
                agent_memory.save_messages(current_agent_id, messages)
            break

        if not user_line:
            continue

        lowered = user_line.lower()
        if lowered in EXIT_COMMANDS:
            print("再见。")
            if current_agent_id:
                agent_memory.save_messages(current_agent_id, messages)
            break

        if lowered in ("/clear", "/reset"):
            clear_messages()
            print("[系统] 已清空对话历史（保留 system 提示）。")
            continue

        if lowered.startswith("/agent"):
            parts = user_line.split()
            if len(parts) < 2:
                print("[系统] 用法: /agent researcher | /agent developer | /agent writer")
                continue
            target = parts[1].strip().lower()
            if target not in agent_memory.VALID_AGENT_IDS:
                print(f"[系统] 未知 Agent: {target}")
                continue
            if target == current_agent_id:
                print(f"[系统] 已在 {target}，无需切换。")
                continue
            switch_agent(target)
            nl = AGENT_DISPLAY_NAMES.get(target, target)
            print(f"[系统] 已切换到 {target}（{nl}），已加载其记忆。")
            continue

        if lowered.startswith("xgsys"):
            tail = user_line[len("xgsys") :].lstrip()
            if tail:
                try:
                    set_system_prompt(tail)
                    print("[系统] 已更新 system prompt（已写入 system_prompt.md）。")
                except ValueError as err:
                    print(f"[系统] {err}")
                continue
            print("请输入新的 system 内容（单行），留空取消：")
            try:
                next_line = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[系统] 已取消。")
                continue
            if not next_line:
                print("[系统] 已取消。")
                continue
            try:
                set_system_prompt(next_line)
                print("[系统] 已更新 system prompt（已写入 system_prompt.md）。")
            except ValueError as err:
                print(f"[系统] {err}")
            continue

        try:
            reply = chat(user_line, model=model)
            print(f"\n助手: {reply}")
        except Exception as exc:
            print(f"\n请求失败: {exc}", file=sys.stderr)
            if messages and messages[-1].get("role") == "user":
                messages.pop()
            persist_agent_state()


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        含 agent、model 的命名空间
    """
    parser = argparse.ArgumentParser(description="MiniMax Anthropic 兼容终端对话（多 Agent 记忆）")
    parser.add_argument(
        "--agent",
        "-a",
        default="researcher",
        choices=sorted(agent_memory.VALID_AGENT_IDS),
        help="启动时使用的 Agent：researcher=研究员, developer=开发者, writer=文案作者",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="MiniMax-M2.7",
        help="MiniMax 模型名",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_terminal_chat(model=args.model, agent_id=args.agent)
