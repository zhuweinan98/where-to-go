"""工具实现单测（不加载 LangChain、不请求大模型）。"""

import json

from src.agent.agent_tools import (
    get_weather_json,
    list_places_json,
    search_sanguo_places_json,
)


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
    assert len(data) >= 1
    names = {p["name"] for p in data}
    assert "中国国家博物馆" in names or "河北博物院" in names


def test_list_places_query_adds_retrieval_score():
    raw = list_places_json("上海", "美术馆")
    data = json.loads(raw)
    assert isinstance(data, list)
    assert "_score" in data[0]
    assert "_score_type" in data[0]


def test_search_sanguo_places_json_hits_chibi():
    raw = search_sanguo_places_json("赤壁 曹操")
    data = json.loads(raw)
    assert isinstance(data, dict)
    assert "places" in data and "romance_excerpts" in data
    assert isinstance(data["romance_excerpts"], list)
    names = {p["name"] for p in data["places"]}
    assert "赤壁古战场" in names
