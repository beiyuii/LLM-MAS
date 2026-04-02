"""
Anthropic Tool Use：工具定义与本地执行（时间查询、读文件）。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

from anthropic.types import TextBlock, ToolUseBlock

# step-1 目录作为读文件沙箱根目录
STEP1_ROOT: Path = Path(__file__).resolve().parent

# 单次读取文件最大字节数，防止误读超大文件
READ_FILE_MAX_BYTES: int = 512 * 1024

# 供 messages.create 传入的 Anthropic 风格工具列表
TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "get_current_time",
        "description": (
            "获取当前准确的日期与时间。返回本机系统时区下的 ISO 8601 时间、时区名与星期。"
            "当用户询问现在几点、今天日期、当前时间时使用。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": (
            "读取本地文本文件内容，使用 UTF-8 解码。"
            "路径相对于项目 step-1 目录，或使用该目录内的相对路径；禁止跳出该目录。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对 step-1 的文件路径，例如 agents/researcher/system_prompt.md",
                }
            },
            "required": ["path"],
        },
    },
]


def tool_get_current_time() -> str:
    """
    返回当前本机时区的准确时间信息（JSON 字符串）。

    Returns:
        含 iso、timezone、weekday_zh 等字段的 JSON 字符串
    """
    now = datetime.now().astimezone()
    payload = {
        "iso8601": now.isoformat(timespec="seconds"),
        "timezone": str(now.tzinfo) if now.tzinfo else "local",
        "weekday_zh": ("周一", "周二", "周三", "周四", "周五", "周六", "周日")[now.weekday()],
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
    }
    return json.dumps(payload, ensure_ascii=False)


def tool_read_file(rel_path: str) -> str:
    """
    在 step-1 目录内读取文本文件。

    Args:
        rel_path: 相对路径或位于 step-1 下的路径

    Returns:
        文件正文；失败时返回错误说明字符串
    """
    raw = (rel_path or "").strip()
    if not raw:
        return "错误：path 不能为空"
    candidate = (STEP1_ROOT / raw).resolve()
    try:
        candidate.relative_to(STEP1_ROOT)
    except ValueError:
        return "错误：路径必须位于 step-1 目录内（禁止 ../ 逃逸）"
    if not candidate.is_file():
        return f"错误：文件不存在 — {candidate}"
    size = candidate.stat().st_size
    if size > READ_FILE_MAX_BYTES:
        return f"错误：文件过大（{size} 字节），上限 {READ_FILE_MAX_BYTES} 字节"
    try:
        text = candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "错误：无法按 UTF-8 解码，请使用文本文件"
    except OSError as exc:
        return f"错误：读取失败 — {exc}"
    return text


def execute_tool(name: str, tool_input: Any) -> str:
    """
    根据工具名执行并返回字符串结果（供 tool_result.content）。

    Args:
        name: get_current_time / read_file
        tool_input: 模型传入的参数（通常为 dict）

    Returns:
        工具输出字符串
    """
    if name == "get_current_time":
        return tool_get_current_time()
    if name == "read_file":
        params = tool_input if isinstance(tool_input, dict) else {}
        path_val = params.get("path", "")
        return tool_read_file(str(path_val))
    return f"错误：未知工具 {name}"


def content_blocks_to_serializable(content: Any) -> Union[str, List[Dict[str, Any]]]:
    """
    将 SDK 返回的 content 转为可写入 JSON 的结构。

    Args:
        content: Message.content 或已是 list/dict

    Returns:
        字符串，或 content 块列表
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    serializable: List[Dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextBlock):
            serializable.append({"type": "text", "text": block.text or ""})
            continue
        if isinstance(block, ToolUseBlock):
            serializable.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input or {}}
            )
            continue
        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        if btype == "text":
            text = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text", "")
            serializable.append({"type": "text", "text": text or ""})
        elif btype == "tool_use":
            tid = getattr(block, "id", None) if not isinstance(block, dict) else block.get("id")
            tname = getattr(block, "name", None) if not isinstance(block, dict) else block.get("name")
            tin = getattr(block, "input", None) if not isinstance(block, dict) else block.get("input")
            serializable.append({"type": "tool_use", "id": tid, "name": tname, "input": tin or {}})
        elif btype == "thinking":
            # 部分模型带 thinking 块，原样序列化便于回放
            thinking = getattr(block, "thinking", "") if not isinstance(block, dict) else block.get("thinking", "")
            serializable.append({"type": "thinking", "thinking": thinking})
    return serializable if serializable else ""


def normalize_message_content_for_api(content: Any) -> Union[str, List[Dict[str, Any]]]:
    """
    将持久化后的 content 转为 API 可接受的格式。

    Args:
        content: str 或已保存的块列表

    Returns:
        传给 Anthropic messages 的 content
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content
    return str(content)
