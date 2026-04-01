"""
按 Agent 目录读写 system_prompt.md（soul）与 conversation.json（messages 序列化）。
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# 与 agents/ 下子目录名一致
VALID_AGENT_IDS: Set[str] = frozenset({"researcher", "developer", "writer"})

# step-1 目录下的 agents 根路径
AGENTS_ROOT: Path = Path(__file__).resolve().parent / "agents"

# 当 system_prompt.md 为空时的占位文案
DEFAULT_SYSTEM_FALLBACK: str = "你是一个 helpful 的 AI 助手。"


def get_agent_dir(agent_id: str) -> Path:
    """
    返回某个 Agent 的数据目录路径。

    Args:
        agent_id: researcher / developer / writer

    Returns:
        目录 Path

    Raises:
        ValueError: 非法 agent_id
    """
    if agent_id not in VALID_AGENT_IDS:
        raise ValueError(f"未知 agent_id: {agent_id}，可选: {sorted(VALID_AGENT_IDS)}")
    return AGENTS_ROOT / agent_id


def read_system_prompt_md(agent_id: str) -> str:
    """
    读取 agents/<id>/system_prompt.md 全文作为 soul。

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


def write_system_prompt_md(agent_id: str, content: str) -> None:
    """
    将 system 写回 system_prompt.md。

    Args:
        agent_id: Agent 标识
        content: 新的 system 全文
    """
    agent_dir = get_agent_dir(agent_id)
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "system_prompt.md"
    path.write_text(content.strip() + "\n", encoding="utf-8")


def load_messages(agent_id: str) -> List[Dict[str, str]]:
    """
    组装会话 messages：system 以 system_prompt.md 为准，user/assistant 来自 conversation.json。

    Args:
        agent_id: Agent 标识

    Returns:
        OpenAI 风格的 messages 列表（首条为 system）
    """
    system_text = read_system_prompt_md(agent_id)
    base: List[Dict[str, str]] = [{"role": "system", "content": system_text}]

    conv_path = get_agent_dir(agent_id) / "conversation.json"
    if not conv_path.is_file():
        return base

    raw = json.loads(conv_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return base

    tail: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        if role in ("user", "assistant"):
            tail.append({"role": role, "content": content})

    return base + tail


def save_messages(agent_id: str, messages: List[Dict[str, str]]) -> None:
    """
    将当前 messages 完整序列化写入 conversation.json（含 system）。

    Args:
        agent_id: Agent 标识
        messages: 完整消息列表
    """
    agent_dir = get_agent_dir(agent_id)
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "conversation.json"
    serializable: List[Dict[str, Any]] = []
    for m in messages:
        serializable.append({"role": m.get("role", ""), "content": m.get("content", "")})
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
