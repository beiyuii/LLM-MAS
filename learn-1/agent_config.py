"""
从 agents/<id>/config.yaml 加载 Agent 配置（人设、模型、工具等）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

# 与 agent_memory.AGENTS_ROOT 一致
AGENTS_ROOT: Path = Path(__file__).resolve().parent / "agents"

CONFIG_FILENAME: str = "config.yaml"

# 新建 Agent 时除人设外的默认项（可被 config 覆盖）
DEFAULT_NEW_AGENT: Dict[str, Any] = {
    "model": "MiniMax-M2.7",
    "temperature": 0.7,
    "max_tokens": 4096,
    "tools": ["read_file", "list_files", "get_current_time"],
}

# 目录名（slug）合法字符
SLUG_PATTERN: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass
class AgentConfig:
    """单个 Agent 的配置（来自 YAML）。"""

    name: str
    display_name: str
    description: str
    model: str
    temperature: float
    max_tokens: int
    tools: List[str]
    persona: str
    raw_path: Optional[Path] = None


def discover_agent_ids() -> List[str]:
    """
    扫描 agents/ 下含 config.yaml 的子目录名。

    Returns:
        已排序的 agent id 列表
    """
    if not AGENTS_ROOT.is_dir():
        return []
    ids: List[str] = []
    for child in AGENTS_ROOT.iterdir():
        if child.is_dir() and (child / CONFIG_FILENAME).is_file():
            ids.append(child.name)
    return sorted(ids)


def validate_agent_id(agent_id: str, known: Optional[Set[str]] = None) -> None:
    """
    校验 agent_id 是否存在于已发现列表。

    Args:
        agent_id: 目录名
        known: 已知集合；为 None 时使用 discover_agent_ids

    Raises:
        ValueError: 未知 id
    """
    pool = known if known is not None else set(discover_agent_ids())
    if agent_id not in pool:
        raise ValueError(f"未知 agent_id: {agent_id}，可选: {sorted(pool)}")


def _parse_config_dict(data: Dict[str, Any], path: Path) -> AgentConfig:
    """将 YAML dict 转为 AgentConfig。"""
    name = str(data.get("name", path.parent.name)).strip()
    display_name = str(data.get("display_name", name)).strip()
    description = str(data.get("description", "")).strip()
    model = str(data.get("model", DEFAULT_NEW_AGENT["model"])).strip()
    temperature = float(data.get("temperature", DEFAULT_NEW_AGENT["temperature"]))
    max_tokens = int(data.get("max_tokens", DEFAULT_NEW_AGENT["max_tokens"]))
    tools_raw = data.get("tools", DEFAULT_NEW_AGENT["tools"])
    if not isinstance(tools_raw, list):
        tools_raw = list(DEFAULT_NEW_AGENT["tools"])
    tools = [str(t).strip() for t in tools_raw if str(t).strip() and not str(t).strip().startswith("#")]
    persona = data.get("persona", "")
    if persona is None:
        persona = ""
    if not isinstance(persona, str):
        persona = str(persona)
    persona = persona.strip()
    return AgentConfig(
        name=name or path.parent.name,
        display_name=display_name or name,
        description=description,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        persona=persona,
        raw_path=path,
    )


def load_agent_config(agent_id: str) -> AgentConfig:
    """
    加载 agents/<agent_id>/config.yaml。

    Args:
        agent_id: 目录名

    Returns:
        AgentConfig

    Raises:
        FileNotFoundError: 缺少配置文件
        ValueError: 配置不合法
    """
    path = AGENTS_ROOT / agent_id / CONFIG_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"缺少配置文件: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.yaml 根节点必须为 mapping: {path}")
    return _parse_config_dict(data, path)


def save_agent_config(agent_id: str, cfg: AgentConfig) -> None:
    """
    将配置写回 agents/<agent_id>/config.yaml（覆盖）。

    Args:
        agent_id: 目录名
        cfg: 配置对象
    """
    agent_dir = AGENTS_ROOT / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / CONFIG_FILENAME
    payload: Dict[str, Any] = {
        "name": cfg.name,
        "display_name": cfg.display_name,
        "description": cfg.description,
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "tools": cfg.tools,
        "persona": cfg.persona,
    }
    header = (
        "# === 基本信息 ===\n"
        "# name / display_name / description（description 可供 Router 使用）\n"
        "# === 模型配置 ===\n"
        "# === 工具配置 ===\n"
        "# === 人设（system / persona）===\n"
        "\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(
            payload,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )


def update_persona_and_sync_files(agent_id: str, persona: str) -> None:
    """
    更新人设：写回 config.yaml 的 persona，并同步 system_prompt.md。

    Args:
        agent_id: 目录名
        persona: 新人设全文
    """
    cfg = load_agent_config(agent_id)
    cfg.persona = persona.strip()
    save_agent_config(agent_id, cfg)
    # 同步一份 md 便于人工编辑 diff
    md_path = AGENTS_ROOT / agent_id / "system_prompt.md"
    md_path.write_text(cfg.persona + "\n", encoding="utf-8")


def create_agent_from_inputs(
    slug: str,
    display_name: str,
    description: str,
    persona: str,
) -> str:
    """
    在 agents/<slug>/ 下创建 config.yaml、system_prompt.md、conversation.json。

    Args:
        slug: 目录名（英文小写+下划线）
        display_name: 展示名
        description: 一句话描述
        persona: 人设全文

    Returns:
        创建的 agent_id（slug）

    Raises:
        ValueError: slug 非法或已存在
    """
    slug = slug.strip().lower()
    if not SLUG_PATTERN.match(slug):
        raise ValueError(
            "目录名须匹配 ^[a-z][a-z0-9_]{0,63}$（小写字母开头，仅小写、数字、下划线）"
        )
    target = AGENTS_ROOT / slug
    if target.exists():
        raise ValueError(f"已存在 agents/{slug}/，请换目录名或先删除该目录")

    persona = persona.strip()
    if not persona:
        raise ValueError("人设不能为空")

    cfg = AgentConfig(
        name=slug,
        display_name=display_name.strip() or slug,
        description=description.strip(),
        model=str(DEFAULT_NEW_AGENT["model"]),
        temperature=float(DEFAULT_NEW_AGENT["temperature"]),
        max_tokens=int(DEFAULT_NEW_AGENT["max_tokens"]),
        tools=list(DEFAULT_NEW_AGENT["tools"]),
        persona=persona,
    )
    target.mkdir(parents=True)
    save_agent_config(slug, cfg)
    (target / "system_prompt.md").write_text(persona + "\n", encoding="utf-8")
    (target / "conversation.json").write_text("[]\n", encoding="utf-8")
    return slug
