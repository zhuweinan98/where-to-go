"""城市单一注册表：从项目根 ``config/cities.json`` 加载。

- 话里匹配城名、和风静态 LocationID、无 API 时的 Mock 天气回退，均来自同一份列表。
- 新增城市：只改 JSON；``MOCK_PLACES`` 里的 ``city`` 须出现在该列表中。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
_CITIES_JSON = _ROOT / "config" / "cities.json"


def _load_registry() -> tuple[dict[str, str], dict[str, dict[str, Any]], tuple[str, ...]]:
    raw = json.loads(_CITIES_JSON.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("config/cities.json 须为 JSON 数组")

    location: dict[str, str] = {}
    mock_weather: dict[str, dict[str, Any]] = {}
    names: list[str] = []
    seen: set[str] = set()

    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"cities.json[{i}] 须为对象")
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"cities.json[{i}] 缺少非空 name")
        if name in seen:
            raise ValueError(f"cities.json 中城市名重复：{name!r}")
        seen.add(name)
        names.append(name)

        qid = entry.get("qweather_id")
        if qid is not None and str(qid).strip():
            location[name] = str(qid).strip()

        mw = entry.get("mock_weather")
        if mw is not None:
            if not isinstance(mw, dict):
                raise ValueError(f"cities.json[{i}] mock_weather 须为对象或 null")
            mock_weather[name] = dict(mw)

    longest_first = tuple(sorted(names, key=len, reverse=True))
    return location, mock_weather, longest_first


CITY_TO_LOCATION_ID, MOCK_WEATHER, CITY_NAMES_LONGEST_FIRST = _load_registry()
ALL_CITY_NAMES: frozenset[str] = frozenset(CITY_NAMES_LONGEST_FIRST)
