"""
使用 Anthropic 兼容协议调用 MiniMax API（非 OpenAI 兼容）。
Agent 由 agents/<id>/config.yaml 配置（人设、模型、温度、工具等）。
支持 Anthropic Tool Use；环境变量从 .env 加载。
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

import agent_config
import agent_memory
import llm_tools
import router as router_module

load_dotenv(Path(__file__).resolve().parent / ".env")

MINIMAX_ANTHROPIC_BASE_URL: str = os.getenv(
    "MINIMAX_API_BASE",
    "https://api.minimaxi.com/anthropic",
)

MINIMAX_API_KEY: Optional[str] = os.getenv("MINIMAX_API_KEY")

client: anthropic.Anthropic = anthropic.Anthropic(
    api_key=MINIMAX_API_KEY or "",
    base_url=MINIMAX_ANTHROPIC_BASE_URL,
)

EXIT_COMMANDS: Set[str] = frozenset({"quit", "exit", "q", ":q", "bye"})

current_agent_id: Optional[str] = None
current_agent_config: Optional[agent_config.AgentConfig] = None

messages: List[Dict[str, Any]] = []


def init_agent(agent_id: str) -> None:
    """
    从磁盘加载指定 Agent 的 config 与 conversation。

    Args:
        agent_id: 目录名（须含 config.yaml）
    """
    global messages, current_agent_id, current_agent_config
    current_agent_id = agent_id
    current_agent_config = agent_config.load_agent_config(agent_id)
    messages = agent_memory.load_messages(agent_id)


def switch_agent(new_agent_id: str) -> None:
    """
    切换 Agent：先持久化当前会话，再加载目标 Agent。

    Args:
        new_agent_id: 目标目录名
    """
    global messages, current_agent_id, current_agent_config
    if current_agent_id:
        agent_memory.save_messages(current_agent_id, messages)
    current_agent_id = new_agent_id
    current_agent_config = agent_config.load_agent_config(new_agent_id)
    messages = agent_memory.load_messages(new_agent_id)


def persist_agent_state() -> None:
    """将当前 messages 写入 conversation.json。"""
    if current_agent_id:
        agent_memory.save_messages(current_agent_id, messages)


def split_system_and_dialog(openai_style_messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """将内部消息拆分为 Anthropic 的 system 与 messages。"""
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
    """从响应中拼接可见文本块。"""
    parts: List[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


def run_chat_turn_with_tools() -> str:
    """
    在已追加本轮 user 文本后，按当前 Agent 配置的模型与工具循环请求直至得到最终文本。

    Returns:
        最后一轮助手可见文本
    """
    if not current_agent_config:
        raise RuntimeError("未加载 Agent 配置")
    cfg = current_agent_config
    tool_defs = llm_tools.get_tool_definitions_for_names(cfg.tools)

    max_tool_rounds = 16
    for _ in range(max_tool_rounds):
        system_text, anthropic_messages = split_system_and_dialog(messages)
        kwargs: Dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "messages": anthropic_messages,
        }
        if tool_defs:
            kwargs["tools"] = tool_defs
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
                raise RuntimeError("模型返回 tool_use 但未包含可执行的工具块")

            messages.append({"role": "user", "content": tool_result_blocks})
            persist_agent_state()
            continue

        final_text = extract_text_from_message(response)
        messages.append({"role": "assistant", "content": final_text})
        persist_agent_state()
        return final_text

    raise RuntimeError("工具调用轮数超过上限")


def chat(new_user_message: str) -> str:
    """
    发送用户消息并获取回复（模型/温度/工具由当前 Agent 的 config.yaml 决定）。

    Args:
        new_user_message: 用户的新消息

    Returns:
        助手最终文本
    """
    if not MINIMAX_API_KEY:
        raise ValueError("请设置环境变量 MINIMAX_API_KEY")

    messages.append({"role": "user", "content": new_user_message})
    return run_chat_turn_with_tools()


def get_messages() -> List[Dict[str, Any]]:
    """获取当前对话历史。"""
    return messages


def print_slash_help(chat_mode: str = "specified") -> None:
    """打印斜杠指令与当前 Agent 可用工具。"""
    print("【斜杠指令】")
    print("  / 或 /tools   — 本帮助")
    print("  /agents       — 列出所有已配置的 Agent")
    print("  /clear        — 清空对话（保留人设）")
    if chat_mode == "specified":
        print("  /agent <id>   — 切换 Agent")
    else:
        print("  （自动模式下不可用）/agent — 请改用 --mode specified 以手动指定 Agent")
    print("  new agent     — 交互式新建 Agent（目录名、显示名、描述、人设）")
    print("")
    if current_agent_config:
        print(f"【当前 Agent】{current_agent_config.name}（{current_agent_config.display_name}）")
        print(f"  模型: {current_agent_config.model}  temperature: {current_agent_config.temperature}  max_tokens: {current_agent_config.max_tokens}")
        print("  已启用工具:")
        for spec in llm_tools.get_tool_definitions_for_names(current_agent_config.tools):
            name = spec.get("name", "")
            desc = (spec.get("description") or "").strip().replace("\n", " ")
            if len(desc) > 100:
                desc = desc[:97] + "..."
            print(f"    • {name}: {desc}")
    else:
        print("【当前 Agent】未加载")


def clear_messages() -> None:
    """清空对话历史，保留 system（人设来自 config）。"""
    global messages
    system_prompt = (
        messages[0]
        if messages and messages[0]["role"] == "system"
        else {"role": "system", "content": agent_memory.DEFAULT_SYSTEM_FALLBACK}
    )
    messages = [system_prompt]
    persist_agent_state()


def run_new_agent_wizard() -> None:
    """交互式新建 Agent：写入 config.yaml、system_prompt.md、conversation.json。"""
    print("[系统] 新建 Agent（任一步输入留空则取消）")
    try:
        slug = input("目录名（英文小写+下划线，如 product_manager）: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[系统] 已取消。")
        return
    if not slug:
        print("[系统] 已取消。")
        return
    try:
        dn = input("显示名称（中文）: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[系统] 已取消。")
        return
    if not dn:
        print("[系统] 已取消。")
        return
    try:
        desc = input("一句话描述（Router 可用）: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[系统] 已取消。")
        return
    print("请输入人设（多行）；单独一行输入 END 结束：")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print("\n[系统] 已取消。")
            return
        if line.strip() == "END":
            break
        lines.append(line)
    persona = "\n".join(lines).strip()
    if not persona:
        print("[系统] 人设不能为空，已取消。")
        return
    try:
        aid = agent_config.create_agent_from_inputs(slug, dn, desc, persona)
        print(f"[系统] 已创建 agents/{aid}/（模型/温度/工具见 config.yaml 默认值）。输入 /agent {aid} 切换。")
    except ValueError as err:
        print(f"[系统] {err}")


def run_terminal_chat(agent_id: str, mode: str = "specified") -> None:
    """
    终端多轮对话。

    Args:
        agent_id: 启动时加载的 Agent 目录名（自动模式下仅作首屏占位，首轮起由 Router 分配）
        mode: specified=固定或 /agent 切换；auto=每轮由 Router 根据用户输入选择 Agent
    """
    if not MINIMAX_API_KEY:
        print("错误：请设置环境变量 MINIMAX_API_KEY 或在 .env 中配置。", file=sys.stderr)
        sys.exit(1)

    router_cfg = router_module.load_router_config()
    init_agent(agent_id)
    assert current_agent_config is not None
    cfg = current_agent_config

    print("=== MiniMax 终端对话（config.yaml + Tool Use + Router）===")
    if mode == "auto":
        print(f"模式: 自动 — 每轮由 Router 选择 Agent（router.yaml: T={router_cfg.temperature}, max_tokens={router_cfg.max_tokens}）")
    else:
        print("模式: 指定 — 使用当前 Agent，可用 /agent 切换")
    print(f"当前 Agent: {cfg.name}（{cfg.display_name}）| 配置: agents/{cfg.name}/config.yaml")
    print("输入 / 或 /tools 查看帮助；/agents 列出全部 Agent；new agent 新建角色；")
    print("人设请在 agents/<id>/config.yaml 中编辑 persona；模型触发工具时会打印 [工具] 行。")
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
            print("[系统] 已清空对话历史（保留人设）。")
            continue

        if lowered in ("/", "/tools", "/help"):
            print_slash_help(mode)
            continue

        if mode == "auto" and lowered.startswith("/agent"):
            print(
                "[系统] 自动模式下每轮由 Router 分配 Agent，不能使用 /agent。"
                " 若需手动指定，请使用: python llm_client.py --mode specified --agent <id>"
            )
            continue

        if lowered == "/agents" or lowered == "/list":
            ids = agent_config.discover_agent_ids()
            print("[系统] 已配置的 Agent:")
            for i in ids:
                try:
                    c = agent_config.load_agent_config(i)
                    print(f"  • {c.name} — {c.display_name}  {c.description}")
                except OSError:
                    print(f"  • {i} （配置读取失败）")
            continue

        if lowered.startswith("/agent"):
            parts = user_line.split()
            if len(parts) < 2:
                print("[系统] 用法: /agent <id>   可用: " + ", ".join(agent_config.discover_agent_ids()))
                continue
            target = parts[1].strip().lower()
            if target not in agent_config.discover_agent_ids():
                print(f"[系统] 未知 Agent: {target}")
                continue
            if target == current_agent_id:
                print(f"[系统] 已在 {target}，无需切换。")
                continue
            switch_agent(target)
            nl = current_agent_config.display_name if current_agent_config else target
            print(f"[系统] 已切换到 {target}（{nl}），已加载其记忆。")
            continue

        if lowered == "new agent":
            run_new_agent_wizard()
            continue

        try:
            if mode == "auto":
                try:
                    chosen = router_module.route_user_message(user_line, client, router_cfg)
                except Exception as rexc:
                    print(f"[Router] 调用失败，使用当前 Agent: {rexc}", file=sys.stderr)
                    chosen = current_agent_id or agent_id
                if chosen != current_agent_id:
                    switch_agent(chosen)
                # 自动模式：无论是否切换，都打印 Router 选型，便于对照
                nl = current_agent_config.display_name if current_agent_config else chosen
                print(f"[Router] 本轮 Agent: {chosen}（{nl}）")
            reply = chat(user_line)
            if current_agent_config:
                print(f"\n[执行 Agent] {current_agent_config.name} · {current_agent_config.display_name}")
            print(f"助手: {reply}")
        except Exception as exc:
            print(f"\n请求失败: {exc}", file=sys.stderr)
            if messages and messages[-1].get("role") == "user":
                messages.pop()
            persist_agent_state()


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    discovered = agent_config.discover_agent_ids()
    if not discovered:
        print("错误：agents/ 下未找到任何 config.yaml。", file=sys.stderr)
        sys.exit(1)
    default_agent = "researcher" if "researcher" in discovered else discovered[0]
    parser = argparse.ArgumentParser(description="MiniMax Anthropic 兼容终端对话（YAML 配置 Agent + Router）")
    parser.add_argument(
        "--mode",
        choices=["specified", "auto"],
        default="specified",
        help="specified=手动指定/切换 Agent；auto=每轮由 Router 根据输入自动选择 Agent",
    )
    parser.add_argument(
        "--agent",
        "-a",
        default=default_agent,
        choices=sorted(discovered),
        help="启动时加载的 Agent；自动模式下仅作首屏占位，首轮对话起由 Router 分配",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_terminal_chat(agent_id=args.agent, mode=args.mode)
