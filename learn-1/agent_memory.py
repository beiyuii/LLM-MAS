"""
按 Agent 目录读写 system_prompt.md（可选同步）、conversation.json（messages 序列化）。
人设优先来自 config.yaml 的 persona。
"""
import json
from pathlib import Path
from typing import Any, Dict, List

import agent_config

# step-1 目录下的 agents 根路径
AGENTS_ROOT: Path = Path(__file__).resolve().parent / "agents"

# 当人设为空时的占位文案
DEFAULT_SYSTEM_FALLBACK: str = "你是一个 helpful 的 AI 助手。"


def get_agent_dir(agent_id: str) -> Path:
    """
    返回某个 Agent 的数据目录路径。

    Args:
        agent_id: 目录名（须已存在 config.yaml）

    Returns:
        目录 Path

    Raises:
        ValueError: 非法 agent_id
    """
    agent_config.validate_agent_id(agent_id)
    return AGENTS_ROOT / agent_id


def read_system_prompt_md(agent_id: str) -> str:
    """
    读取 agents/<id>/system_prompt.md 全文（config 无人设时的回退）。

    Args:
        agent_id: Agent 标识

    Returns:
        文件文本；不存在或为空时返回默认占位
    """
    path = get_agent_dir(agent_id) / "system_prompt.md"
    if not path.is_file():
        return DEFAULT_SYSTEM_FALLBACK
    text = path.read_text(encoding="utf-8").strip()
    return text if text else DEFAULT_SYSTEM_FALLBACK


def get_system_prompt_text(agent_id: str) -> str:
    """
    获取用于 system 的人设文本：优先 config.yaml 的 persona，否则 system_prompt.md。

    Args:
        agent_id: Agent 标识

    Returns:
        system 字符串
    """
    try:
        cfg = agent_config.load_agent_config(agent_id)
        if cfg.persona.strip():
            return cfg.persona.strip()
    except (OSError, ValueError, FileNotFoundError):
        pass
    return read_system_prompt_md(agent_id)


def write_system_prompt_md(agent_id: str, content: str) -> None:
    """
    将 system 写回 system_prompt.md（与 config 人设同步时一并写入）。

    Args:
        agent_id: Agent 标识
        content: 新的 system 全文
    """
    agent_dir = get_agent_dir(agent_id)
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "system_prompt.md"
    path.write_text(content.strip() + "\n", encoding="utf-8")


def load_messages(agent_id: str) -> List[Dict[str, Any]]:
    """
    组装会话 messages：system 以人设为准，user/assistant 来自 conversation.json。

    Args:
        agent_id: Agent 标识

    Returns:
        messages 列表（首条为 system）；content 可为 str 或 Anthropic 内容块 list（工具调用后）
    """
    system_text = get_system_prompt_text(agent_id)
    base: List[Dict[str, Any]] = [{"role": "system", "content": system_text}]

    conv_path = get_agent_dir(agent_id) / "conversation.json"
    if not conv_path.is_file():
        return base

    raw = json.loads(conv_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return base

    tail: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "")
        if role in ("user", "assistant"):
            if isinstance(content, (str, list)):
                tail.append({"role": role, "content": content})
            else:
                tail.append({"role": role, "content": str(content)})

    return base + tail


def save_messages(agent_id: str, messages: List[Dict[str, Any]]) -> None:
    """
    将当前 messages 完整序列化写入 conversation.json（含 system）。

    Args:
        agent_id: Agent 标识
        messages: 完整消息列表（content 可为 str 或 list）
    """
    agent_dir = get_agent_dir(agent_id)
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "conversation.json"
    serializable: List[Dict[str, Any]] = []
    for m in messages:
        serializable.append({"role": m.get("role", ""), "content": m.get("content", "")})
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
