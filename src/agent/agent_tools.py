"""Agent 可调用的工具实现（与 LangChain 解耦，便于单测与日后换实现）。

list_places 预留 query 供后续 RAG；当前仅返回全量 mock 列表。
"""

from __future__ import annotations

import json
from typing import Any

from src.data.mock import places_for_city
from src.data.weather_source import get_weather_for_city


def get_weather_json(city: str) -> str:
    """返回当前城市天气 JSON 字符串（和风或 Mock）。"""
    c = (city or "").strip() or "上海"
    w: dict[str, Any] = get_weather_for_city(c)
    return json.dumps(w, ensure_ascii=False)


def list_places_json(city: str, query: str = "") -> str:
    """返回该城景点列表 JSON。query 预留给语义检索，当前未使用。"""
    c = (city or "").strip() or "上海"
    _ = (query or "").strip()  # RAG 阶段使用
    places: list[dict[str, Any]] = places_for_city(c)
    return json.dumps(places, ensure_ascii=False)
