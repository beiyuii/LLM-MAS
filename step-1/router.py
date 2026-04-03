"""
Router：在自动模式下调用大模型，根据用户输入从已注册 Agent 中选择一个 id。
配置见 router.yaml（低温度、少 token）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from anthropic.types import TextBlock

import agent_config

STEP1_ROOT: Path = Path(__file__).resolve().parent
ROUTER_CONFIG_PATH: Path = STEP1_ROOT / "router.yaml"


@dataclass
class RouterConfig:
    """Router 调用参数。"""

    model: str
    temperature: float
    max_tokens: int


def load_router_config() -> RouterConfig:
    """
    读取 router.yaml；缺省文件时使用内置默认值。

    Returns:
        RouterConfig
    """
    defaults: Dict[str, Any] = {
        "model": "MiniMax-M2.7",
        "temperature": 0.1,
        "max_tokens": 50,
    }
    if ROUTER_CONFIG_PATH.is_file():
        with ROUTER_CONFIG_PATH.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            defaults.update({k: data[k] for k in ("model", "temperature", "max_tokens") if k in data})
    return RouterConfig(
        model=str(defaults["model"]),
        temperature=float(defaults["temperature"]),
        max_tokens=int(defaults["max_tokens"]),
    )


def build_router_system_prompt(valid_ids: List[str]) -> str:
    """
    构造 Router 的 system 提示：枚举各 Agent 的 id、显示名与描述。

    Args:
        valid_ids: 已排序的 agent 目录名列表

    Returns:
        system 文本
    """
    lines: List[str] = [
        "你是多智能体系统中的「路由」模块。你的唯一任务：根据用户的一句话任务描述，从下列 Agent 中选出最合适的一个。",
        "",
        "【输出规则】只输出该 Agent 的 id（即下列列表中的英文目录名，小写），不要输出任何其他字符、空格、换行、标点、引号或解释。",
        "",
        "【可选 Agent】",
    ]
    for aid in valid_ids:
        try:
            cfg = agent_config.load_agent_config(aid)
            lines.append(f"- id: {cfg.name} | 名称: {cfg.display_name} | 说明: {cfg.description}")
        except OSError:
            lines.append(f"- id: {aid} （配置读取失败，请勿选）")
    lines.extend(
        [
            "",
            "若任务与以上均不完全匹配，选择最接近的一项；若完全无法判断，输出列表中的第一个 id。",
        ]
    )
    return "\n".join(lines)


def parse_router_output(raw: str, valid_ids: Set[str], fallback: str) -> str:
    """
    从模型输出中解析出合法的 agent id。

    Args:
        raw: 模型原文
        valid_ids: 合法 id 集合
        fallback: 解析失败时的默认值

    Returns:
        agent id
    """
    text = (raw or "").strip()
    if not text:
        return fallback
    t0 = text.splitlines()[0].strip()
    if t0 in valid_ids:
        return t0
    # 首行按空白/标点切分，取第一个合法 token
    for token in re.split(r"[\s\.,;，。:：\"'（）()]+", t0):
        tok = token.strip().lower()
        if tok in valid_ids:
            return tok
    # 子串匹配（优先长 id，避免短名误匹配）
    lower = text.lower()
    for aid in sorted(valid_ids, key=len, reverse=True):
        if aid in lower:
            return aid
    return fallback


def route_user_message(
    user_text: str,
    client: Any,
    router_cfg: RouterConfig,
) -> str:
    """
    调用大模型做一次路由，返回选中的 agent id。

    Args:
        user_text: 用户本轮自然语言输入
        client: anthropic.Anthropic 客户端
        router_cfg: Router 参数

    Returns:
        选中的 agent 目录名
    """
    ids = agent_config.discover_agent_ids()
    valid = set(ids)
    if not ids:
        raise RuntimeError("未找到任何 Agent（agents/*/config.yaml）")
    fallback = "researcher" if "researcher" in valid else ids[0]

    system = build_router_system_prompt(ids)
    kwargs: Dict[str, Any] = {
        "model": router_cfg.model,
        "max_tokens": router_cfg.max_tokens,
        "temperature": router_cfg.temperature,
        "system": system,
        "messages": [{"role": "user", "content": user_text.strip()}],
    }
    response = client.messages.create(**kwargs)
    raw = ""
    for block in response.content:
        if isinstance(block, TextBlock):
            raw += block.text or ""
        elif getattr(block, "type", None) == "text":
            raw += getattr(block, "text", "") or ""
    chosen = parse_router_output(raw, valid, fallback)
    return chosen
