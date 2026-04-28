import os
from typing import Dict


def apply_inline_env_overrides(overrides: Dict[str, object], priority: str = "code") -> Dict[str, str]:
    """
    允许在脚本内部直接覆盖环境变量，便于调试。

    priority:
    - "code"：脚本内配置优先，直接覆盖终端同名环境变量
    - "terminal"：若终端已设置同名环境变量，则保留终端值
    """
    normalized_priority = str(priority or "code").strip().lower()
    applied: Dict[str, str] = {}
    for raw_key, raw_value in (overrides or {}).items():
        if raw_key is None or raw_value is None:
            continue
        key = str(raw_key).strip()
        if not key:
            continue
        value = str(raw_value)
        if normalized_priority == "terminal" and key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied


def format_inline_env_overrides(applied: Dict[str, str]) -> str:
    if not applied:
        return "inline_env_overrides=none"
    items = [key + "=" + value for key, value in sorted(applied.items())]
    return "inline_env_overrides=" + ", ".join(items)
