"""
Anthropic Tool Use：工具定义与本地执行（时间、读文件、列目录、检索、环境、按行读、文本统计等）。
"""
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from anthropic.types import TextBlock, ToolUseBlock

# step-1 目录作为读文件沙箱根目录
STEP1_ROOT: Path = Path(__file__).resolve().parent

# 单次读取文件最大字节数，防止误读超大文件
READ_FILE_MAX_BYTES: int = 512 * 1024

# 全部内置工具定义（按名称索引；Agent 配置通过名称子集启用）
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_current_time": {
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
    "read_file": {
        "name": "read_file",
        "description": (
            "读取本地文本文件内容，使用 UTF-8 解码。"
            "路径相对于项目 step-1 目录；禁止跳出该目录。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对 step-1 的文件路径，例如 agents/researcher/config.yaml",
                }
            },
            "required": ["path"],
        },
    },
    "list_files": {
        "name": "list_files",
        "description": (
            "列出 step-1 目录下某子目录中的文件与文件夹（相对路径），"
            "用于浏览项目结构；默认列出当前目录。会跳过隐藏目录与 __pycache__。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "相对 step-1 的目录路径，默认 .",
                },
                "max_entries": {
                    "type": "integer",
                    "description": "最多列出的条目数，默认 300",
                },
            },
            "required": [],
        },
    },
    "grep_in_file": {
        "name": "grep_in_file",
        "description": (
            "在 step-1 内某文本文件中搜索关键字或正则，返回匹配行号与内容摘要。"
            "适合调研、核对引用、定位某段文字在文件中的位置。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对 step-1 的文件路径"},
                "pattern": {"type": "string", "description": "搜索模式"},
                "literal": {
                    "type": "boolean",
                    "description": "为 true 时按字面子串匹配；为 false 时按 Python 正则（慎用复杂模式）",
                },
                "max_matches": {"type": "integer", "description": "最多返回条数，默认 80"},
            },
            "required": ["path", "pattern"],
        },
    },
    "get_runtime_env": {
        "name": "get_runtime_env",
        "description": (
            "获取当前 Python 解释器、操作系统、工作区根路径（step-1）等运行环境信息，"
            "便于排查依赖与路径问题。"
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "read_file_range": {
        "name": "read_file_range",
        "description": (
            "按行号读取 step-1 内文本文件的片段（行号从 1 开始，含首尾）。"
            "适合查看代码或日志的局部，避免一次性读入过大文件。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对 step-1 的文件路径"},
                "start_line": {"type": "integer", "description": "起始行号（≥1）"},
                "end_line": {"type": "integer", "description": "结束行号（含）"},
            },
            "required": ["path", "start_line", "end_line"],
        },
    },
    "text_file_statistics": {
        "name": "text_file_statistics",
        "description": (
            "统计 step-1 内某 UTF-8 文本文件的字数、字符数、行数、段落数等，"
            "用于估算篇幅、阅读时长与排版。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对 step-1 的文件路径"},
            },
            "required": ["path"],
        },
    },
}

# 兼容旧代码：全部工具列表
TOOL_DEFINITIONS: List[Dict[str, Any]] = list(TOOL_REGISTRY.values())


def get_tool_definitions_for_names(names: List[str]) -> List[Dict[str, Any]]:
    """
    按配置中的工具名列表，返回对应 Anthropic 工具定义（顺序与配置一致，去重）。

    Args:
        names: 工具名列表，如 ['read_file','list_files']

    Returns:
        tools 参数可用的定义列表
    """
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for raw in names:
        key = str(raw).strip()
        if not key or key in seen:
            continue
        spec = TOOL_REGISTRY.get(key)
        if spec:
            seen.add(key)
            out.append(spec)
    return out


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


def tool_list_files(directory: str = ".", max_entries: int = 300) -> str:
    """
    列出 step-1 下某目录内的文件与文件夹（相对路径），条目数有上限。

    Args:
        directory: 相对 step-1 的目录
        max_entries: 最大条目数

    Returns:
        JSON 字符串，含 count 与 paths
    """
    rel = (directory or ".").strip() or "."
    base = (STEP1_ROOT / rel).resolve()
    try:
        base.relative_to(STEP1_ROOT)
    except ValueError:
        return json.dumps({"error": "目录必须位于 step-1 内（禁止 ../ 逃逸）"}, ensure_ascii=False)
    if not base.exists():
        return json.dumps({"error": f"路径不存在: {base}"}, ensure_ascii=False)
    if not base.is_dir():
        return json.dumps({"error": f"不是目录: {base}"}, ensure_ascii=False)

    paths: List[str] = []
    for p in sorted(base.rglob("*")):
        if len(paths) >= max(1, min(max_entries, 5000)):
            break
        try:
            rel_p = p.relative_to(STEP1_ROOT)
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel_p.parts):
            continue
        if "__pycache__" in rel_p.parts:
            continue
        suffix = "/" if p.is_dir() else ""
        paths.append(str(rel_p).replace("\\", "/") + suffix)

    payload = {"count": len(paths), "paths": paths, "root": str(STEP1_ROOT)}
    if len(paths) >= max_entries:
        payload["truncated"] = True
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


def _read_text_under_step1(rel_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    读取 step-1 内文本；成功返回 (text, None)，失败返回 (None, error_msg)。
    """
    raw = (rel_path or "").strip()
    if not raw:
        return None, "错误：path 不能为空"
    candidate = (STEP1_ROOT / raw).resolve()
    try:
        candidate.relative_to(STEP1_ROOT)
    except ValueError:
        return None, "错误：路径必须位于 step-1 目录内"
    if not candidate.is_file():
        return None, f"错误：文件不存在 — {candidate}"
    size = candidate.stat().st_size
    if size > READ_FILE_MAX_BYTES:
        return None, f"错误：文件过大（{size} 字节），上限 {READ_FILE_MAX_BYTES} 字节"
    try:
        text = candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None, "错误：无法按 UTF-8 解码"
    except OSError as exc:
        return None, f"错误：读取失败 — {exc}"
    return text, None


def tool_grep_in_file(
    rel_path: str,
    pattern: str,
    literal: bool = True,
    max_matches: int = 80,
) -> str:
    """
    在文件内搜索 pattern，返回 JSON（匹配列表，含行号）。
    """
    pat = (pattern or "")[:500]
    if not pat:
        return json.dumps({"error": "pattern 不能为空"}, ensure_ascii=False)

    text, err = _read_text_under_step1(rel_path)
    if err:
        return json.dumps({"error": err}, ensure_ascii=False)

    lines = text.splitlines()
    matches: List[Dict[str, Any]] = []
    max_matches = max(1, min(max_matches, 500))
    regex: Any = None
    if not literal:
        try:
            regex = re.compile(pat)
        except re.error as exc:
            return json.dumps({"error": f"正则无效: {exc}"}, ensure_ascii=False)

    for i, line in enumerate(lines, start=1):
        if len(matches) >= max_matches:
            break
        ok = pat in line if literal else (regex.search(line) is not None)
        if ok:
            preview = line if len(line) <= 500 else line[:497] + "..."
            matches.append({"line": i, "text": preview})

    return json.dumps(
        {
            "path": rel_path,
            "pattern": pat,
            "literal": literal,
            "match_count": len(matches),
            "truncated": len(matches) >= max_matches,
            "matches": matches,
        },
        ensure_ascii=False,
    )


def tool_get_runtime_env() -> str:
    """返回 Python/OS/工作区等信息。"""
    payload = {
        "python": sys.version.split()[0],
        "full_version": sys.version,
        "platform": sys.platform,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "step1_root": str(STEP1_ROOT),
    }
    return json.dumps(payload, ensure_ascii=False)


def tool_read_file_range(rel_path: str, start_line: int, end_line: int) -> str:
    """
    按 1-based 行号读取闭区间 [start_line, end_line]。
    """
    text, err = _read_text_under_step1(rel_path)
    if err:
        return json.dumps({"error": err}, ensure_ascii=False)

    lines = text.splitlines()
    n = len(lines)
    s = int(start_line)
    e = int(end_line)
    if s > e:
        return json.dumps({"error": "start_line 不能大于 end_line"}, ensure_ascii=False)
    s = max(1, s)
    e = max(s, e)
    if n == 0:
        return json.dumps({"path": rel_path, "start_line": s, "end_line": e, "total_lines": 0, "content": ""}, ensure_ascii=False)
    if s > n:
        return json.dumps({"error": f"起始行 {s} 超出文件行数 {n}"}, ensure_ascii=False)
    e = min(e, n)
    span = e - s + 1
    if span > 800:
        return json.dumps({"error": "单次最多读取 800 行，请缩小范围"}, ensure_ascii=False)

    chunk = lines[s - 1 : e]
    numbered = [f"{i + s:6d}|{lines[i + s - 1]}" for i in range(len(chunk))]
    body = "\n".join(numbered)
    return json.dumps(
        {
            "path": rel_path,
            "start_line": s,
            "end_line": e,
            "total_lines": n,
            "content": body,
        },
        ensure_ascii=False,
    )


def tool_text_file_statistics(rel_path: str) -> str:
    """统计文本文件篇幅指标。"""
    text, err = _read_text_under_step1(rel_path)
    if err:
        return json.dumps({"error": err}, ensure_ascii=False)

    lines = text.splitlines()
    line_count = len(lines)
    char_count = len(text)
    char_no_space = len(re.sub(r"\s+", "", text))
    words = len(text.split())
    paras = [p for p in text.split("\n\n") if p.strip()]
    para_count = len(paras)
    # 粗略阅读分钟（中文约 400 字/分钟，英文词约 200 wpm 混合取经验值）
    est_min = max(1, round(words / 200)) if words else 0

    return json.dumps(
        {
            "path": rel_path,
            "lines": line_count,
            "characters": char_count,
            "characters_no_whitespace": char_no_space,
            "words_tokens_whitespace_split": words,
            "paragraphs_double_newline": para_count,
            "estimated_reading_minutes_rough": est_min,
        },
        ensure_ascii=False,
    )


def execute_tool(name: str, tool_input: Any) -> str:
    """
    根据工具名执行并返回字符串结果（供 tool_result.content）。

    Args:
        name: 已注册工具名
        tool_input: 模型传入的参数（通常为 dict）

    Returns:
        工具输出字符串
    """
    params = tool_input if isinstance(tool_input, dict) else {}

    if name == "get_current_time":
        return tool_get_current_time()
    if name == "read_file":
        return tool_read_file(str(params.get("path", "")))
    if name == "list_files":
        d = str(params.get("directory", "."))
        try:
            me = int(params.get("max_entries", 300))
        except (TypeError, ValueError):
            me = 300
        return tool_list_files(d, max(1, min(me, 5000)))
    if name == "grep_in_file":
        literal = bool(params.get("literal", True))
        try:
            mm = int(params.get("max_matches", 80))
        except (TypeError, ValueError):
            mm = 80
        return tool_grep_in_file(
            str(params.get("path", "")),
            str(params.get("pattern", "")),
            literal=literal,
            max_matches=mm,
        )
    if name == "get_runtime_env":
        return tool_get_runtime_env()
    if name == "read_file_range":
        try:
            s = int(params.get("start_line", 1))
            e = int(params.get("end_line", 1))
        except (TypeError, ValueError):
            return json.dumps({"error": "start_line/end_line 须为整数"}, ensure_ascii=False)
        return tool_read_file_range(str(params.get("path", "")), s, e)
    if name == "text_file_statistics":
        return tool_text_file_statistics(str(params.get("path", "")))
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
