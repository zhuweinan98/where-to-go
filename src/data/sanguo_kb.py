"""三国主题知识库：从外置 JSON 加载（默认同仓库 ``data/sanguo_places.json``）。

可通过环境变量 ``SANGUO_KB_JSON`` 指定绝对或相对路径，便于多套数据或单测临时文件。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_json_path() -> Path:
    return _project_root() / "data" / "sanguo_places.json"


def _load_sanguo_places() -> list[dict[str, Any]]:
    raw = os.getenv("SANGUO_KB_JSON", "").strip()
    path = Path(raw).expanduser() if raw else _default_json_path()
    if not path.is_file():
        raise FileNotFoundError(
            "三国知识库 JSON 不存在: "
            f"{path.resolve()}（默认应为仓库 data/sanguo_places.json；"
            "或设置环境变量 SANGUO_KB_JSON 指向自定义文件）"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"三国知识库 JSON 根类型须为数组: {path}")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"三国知识库 JSON 第 {i} 条不是对象: {path}")
        out.append(dict(item))
    return out


SANGUO_PLACES: list[dict[str, Any]] = _load_sanguo_places()
