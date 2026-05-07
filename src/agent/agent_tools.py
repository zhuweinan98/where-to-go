"""Agent 可调用的工具实现（与 LangChain 解耦，便于单测与日后换实现）。

list_places 在 query 非空时走本地轻量 RAG 检索（top-k）。
"""

from __future__ import annotations

import json
from typing import Any

from src.data.mock import places_for_city
from src.data.rag_retriever import (
    consume_rag_debug_logs,
    search_places,
    search_sanguo_places,
)
from src.data.sanguoyanyi_retriever import search_romance_excerpts
from src.data.weather_source import get_weather_for_city


def get_weather_json(city: str) -> str:
    """返回当前城市天气 JSON 字符串（和风或 Mock）。"""
    c = (city or "").strip() or "上海"
    w: dict[str, Any] = get_weather_for_city(c)
    return json.dumps(w, ensure_ascii=False)


def list_places_json(city: str, query: str = "") -> str:
    """返回该城景点列表 JSON。query 非空时返回检索命中的 top-k。"""
    c = (city or "").strip() or "上海"
    q = (query or "").strip()
    places: list[dict[str, Any]]
    if q:
        places = search_places(c, q, top_k=3)
    else:
        places = places_for_city(c)
    return json.dumps(places, ensure_ascii=False)


def search_sanguo_places_json(query: str) -> str:
    """检索三国主题知识库：places 为结构化景点；romance_excerpts 为《演义》词法摘录（chunk_id+回目+节选）。"""
    places, _, romance = search_sanguo_places_with_meta(query)
    return json.dumps(
        {"places": places, "romance_excerpts": romance},
        ensure_ascii=False,
    )


def search_sanguo_places_with_meta(
    query: str,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """检索三国知识库；并同路径拉取演义 jsonl 摘录。返回 (places, rag_debug_logs, romance_excerpts)。"""
    q = (query or "").strip()
    places = search_sanguo_places(q, top_k=3)
    logs = consume_rag_debug_logs()
    if places:
        names = [str(p.get("name", "")) for p in places]
        romance = search_romance_excerpts(q, names, top_k=2)
        logs.extend(consume_rag_debug_logs())
    else:
        romance = []
    return places, logs, romance
