"""Agent 单元测试：直接调用 chat()，不启动 HTTP 服务。"""

from src.agent.bot import chat


def test_off_mode_hint_on_greeting():
    r = chat("你好")
    assert "LLM_MODE=off" in r


def test_off_mode_hint_on_weather():
    r = chat("今天天气怎么样", city="上海")
    assert "LLM_MODE=off" in r


def test_off_mode_hint_on_recommend():
    r = chat("推荐去哪里玩", city="上海")
    assert "LLM_MODE=off" in r


def test_off_mode_hint_on_arbitrary_text():
    r = chat("北京今天天气怎么样", city="上海")
    assert "LLM_MODE=off" in r
