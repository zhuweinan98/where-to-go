"""工具实现单测（不加载 LangChain、不请求大模型）。"""

import json

from src.agent.agent_tools import get_weather_json, list_places_json


def test_get_weather_json_shape():
    raw = get_weather_json("上海")
    data = json.loads(raw)
    assert "weather" in data
    assert "temp" in data


def test_list_places_json_contains_known_spot():
    raw = list_places_json("上海", "")
    data = json.loads(raw)
    names = {p["name"] for p in data}
    assert "顾村公园" in names or "西岸美术馆" in names


def test_list_places_query_reserved_no_crash():
    raw = list_places_json("北京", "博物馆")
    data = json.loads(raw)
    assert isinstance(data, list)
