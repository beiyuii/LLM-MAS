"""
使用 Anthropic 兼容协议调用 MiniMax API（非 OpenAI 兼容）。
支持按 Agent 目录持久化 system_prompt.md 与 conversation.json。
支持 Anthropic Tool Use：时间查询、读取文件。
环境变量优先从与本文件同目录的 .env 加载（不覆盖已存在的环境变量）。
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import anthropic
from anthropic.types import TextBlock, ToolUseBlock
from dotenv import load_dotenv

import agent_memory
import llm_tools

# 在读取 os.getenv 之前加载与本文件同目录的 .env（不覆盖已存在的环境变量）
load_dotenv(Path(__file__).resolve().parent / ".env")

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

# 对话历史：system 为 str；user/assistant 的 content 可为 str 或 Anthropic 内容块 list（工具调用）
messages: List[Dict[str, Any]] = []


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


def split_system_and_dialog(openai_style_messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    将内部消息拆分为 Anthropic 的 system 与 messages（支持字符串或多段 content）。

    Args:
        openai_style_messages: 含 system/user/assistant 的消息列表

    Returns:
        (system 文本或 None, Anthropic messages 列表)
    """
    system_parts: List[str] = []
    dialog: List[Dict[str, Any]] = []
    for item in openai_style_messages:
        role = item.get("role")
        content = item.get("content")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            else:
                system_parts.append(str(content))
        elif role in ("user", "assistant"):
            dialog.append(
                {
                    "role": role,
                    "content": llm_tools.normalize_message_content_for_api(content),
                }
            )
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
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


def run_chat_turn_with_tools(model: str) -> str:
    """
    在已追加本轮 user 文本消息后，循环请求模型直至得到最终文本（含多轮 tool_use）。

    Args:
        model: MiniMax 模型名

    Returns:
        最后一轮助手可见文本（纯文本块拼接）
    """
    max_tool_rounds = 16
    final_text = ""
    for _ in range(max_tool_rounds):
        system_text, anthropic_messages = split_system_and_dialog(messages)
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "temperature": 0.7,
            "messages": anthropic_messages,
            "tools": llm_tools.TOOL_DEFINITIONS,
        }
        if system_text:
            kwargs["system"] = system_text

        response = client.messages.create(**kwargs)
        has_tool_use = any(isinstance(b, ToolUseBlock) for b in response.content)

        if has_tool_use:
            ass_content = llm_tools.content_blocks_to_serializable(response.content)
            messages.append({"role": "assistant", "content": ass_content})

            tool_result_blocks: List[Dict[str, Any]] = []
            for block in response.content:
                if not isinstance(block, ToolUseBlock):
                    continue
                tool_name = getattr(block, "name", "") or ""
                tool_id = getattr(block, "id", "") or ""
                tool_input = getattr(block, "input", None) or {}
                print(f"[工具] {tool_name} 已触发 | 参数: {json.dumps(tool_input, ensure_ascii=False)}")
                result_str = llm_tools.execute_tool(tool_name, tool_input)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_str,
                    }
                )

            if not tool_result_blocks:
                if messages and messages[-1].get("role") == "assistant":
                    messages.pop()
                raise RuntimeError("模型返回 tool_use 但未包含可执行的工具块，请重试或检查模型是否兼容 Tool Use")

            messages.append({"role": "user", "content": tool_result_blocks})
            persist_agent_state()
            continue

        final_text = extract_text_from_message(response)
        messages.append({"role": "assistant", "content": final_text})
        persist_agent_state()
        return final_text

    raise RuntimeError("工具调用轮数超过上限，请简化请求或检查模型是否支持 Tool Use")


def chat(new_user_message: str, model: str = "MiniMax-M2.7") -> str:
    """
    发送用户消息并获取 AI 回复（启用 Tool Use 时可能多轮请求）。

    Args:
        new_user_message: 用户的新消息
        model: MiniMax 模型名（Anthropic 兼容路径下使用，如 MiniMax-M2.7）

    Returns:
        AI 的回复内容（最终轮文本）
    """
    if not MINIMAX_API_KEY:
        raise ValueError("请设置环境变量 MINIMAX_API_KEY")

    messages.append(
        {
            "role": "user",
            "content": new_user_message,
        }
    )

    return run_chat_turn_with_tools(model)


def get_messages() -> List[Dict[str, Any]]:
    """获取当前对话历史。"""
    return messages


def print_slash_help() -> None:
    """在终端打印斜杠指令说明与内置工具列表。"""
    print("【斜杠指令】")
    print("  / 或 /tools   — 显示本帮助与可用工具")
    print("  /help         — 同上")
    print("  /clear        — 清空对话（保留 system）")
    print("  /agent <id>   — 切换 researcher / developer / writer")
    print("  xgsys         — 修改 system（见启动说明）")
    print("")
    print("【内置工具】(Anthropic Tool Use)")
    for spec in llm_tools.TOOL_DEFINITIONS:
        name = spec.get("name", "")
        desc = spec.get("description", "").strip().replace("\n", " ")
        if len(desc) > 120:
            desc = desc[:117] + "..."
        print(f"  • {name}: {desc}")


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

    print("=== MiniMax 终端对话（Anthropic 兼容 + Tool Use）===")
    print(f"当前 Agent: {agent_id}（{label}）| 数据目录: agents/{agent_id}/")
    print("输入内容后回车发送；quit / exit / q / bye 结束；输入 / 或 /tools 查看工具列表；")
    print("/clear 清空对话；/agent researcher|developer|writer 切换角色；")
    print("xgsys <新提示词> 修改 system（会写入 system_prompt.md）；模型触发工具时会显示 [工具] 行。")
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

        if lowered in ("/", "/tools", "/help"):
            print_slash_help()
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
