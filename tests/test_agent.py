"""Agent 单元测试：直接调用 chat()，不启动 HTTP 服务。"""

from src.agent.bot import chat


def test_greeting():
    r = chat("你好")
    assert "今天去哪玩" in r
    assert "天气" in r or "推荐" in r


def test_weather_shanghai():
    r = chat("今天天气怎么样", city="上海")
    assert "上海" in r
    assert "°C" in r


def test_recommend():
    r = chat("推荐去哪里玩", city="上海")
    assert "推荐" in r
    assert "顾村" in r or "西岸" in r


def test_rain_indoor_guangzhou():
    r = chat("推荐", city="广州")
    assert "美术馆" in r


def test_city_in_message_overrides_form():
    """话里带城市名时覆盖表单默认（如仍选上海但说北京）。"""
    r = chat("今天天气怎么样", city="上海")
    assert "上海" in r
    r2 = chat("北京今天天气怎么样", city="上海")
    assert r2.startswith("今天北京")
    assert "°C" in r2


def test_recommend_shenyang():
    r = chat("推荐去哪里玩", city="沈阳")
    assert "推荐" in r
    assert "沈阳故宫" in r or "张氏帅府" in r
