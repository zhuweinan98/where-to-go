"""Agent 核心：规则式对话（Week 1）。

职责：读用户话与城市，结合 Mock 数据生成一句回复。
作用：终端与 Web 共用同一套逻辑（import chat）。

默认仅规则引擎（单测依赖）；LLM_MODE=ollama 走本地 Ollama，LLM_MODE=openai 走云端 OpenAI 兼容 API。
天气/推荐规则在愿意改为纯模型行为之前保留，勿删。
"""

import os

from dotenv import load_dotenv

load_dotenv()

from src.agent import llm as llm_mod
from src.data.mock import MOCK_PLACES, MOCK_WEATHER
from src.data.weather_source import CITY_TO_LOCATION_ID, get_weather_for_city

# TODO: 未命中白名单时，从 user_text 主动解析城市（如和风 Geo 模糊搜、或小模型 NER），并与 form_city 约定优先级。
# TODO: 将城名集合迁到配置文件/官方城市列表，减少手写维护（_EXTRA_CITY_NAMES + MOCK 并集）。

# 话里出现、且可与和风 Geo 对齐的常见城名（与静态表合并，按长名优先匹配）
_EXTRA_CITY_NAMES = frozenset(
    {
        "深圳",
        "杭州",
        "成都",
        "重庆",
        "天津",
        "南京",
        "武汉",
        "西安",
        "苏州",
        "郑州",
        "长沙",
        "沈阳",
        "青岛",
        "宁波",
        "厦门",
        "无锡",
        "合肥",
        "福州",
        "济南",
        "昆明",
        "哈尔滨",
        "长春",
        "石家庄",
        "南宁",
        "贵阳",
        "太原",
        "乌鲁木齐",
        "拉萨",
        "海口",
        "兰州",
        "银川",
        "西宁",
    }
)


def _effective_city(user_text: str, form_city: str) -> str:
    """用户话里若明确提到城市，优先于表单 city（解决「下拉仍是上海但说北京」）。"""
    fb = (form_city or "上海").strip() or "上海"
    names = sorted(
        set(MOCK_WEATHER) | set(CITY_TO_LOCATION_ID) | _EXTRA_CITY_NAMES,
        key=len,
        reverse=True,
    )
    hits: list[tuple[int, str]] = []
    for name in names:
        pos = user_text.find(name)
        if pos >= 0:
            hits.append((pos, name))
    if not hits:
        return fb  # TODO: 此处仅回退表单城市；白名单外地名需接上条 TODO
    hits.sort(key=lambda x: x[0])
    return hits[0][1]


def chat(user_input: str, city: str = "上海") -> str:
    """Agent 主入口：输入用户原话与城市，返回助手回复文本。"""
    text = user_input.strip()
    form_city = (city or "上海").strip() or "上海"
    c = _effective_city(text, form_city)
    if llm_mod.llm_enabled():
        weather = get_weather_for_city(c)
        system = llm_mod.build_system_prompt(c, weather, MOCK_PLACES)
        try:
            out = llm_mod.complete(system, text, city=c)
            return out if out.strip() else "（模型没有返回文字，请重试或缩短问题。）"
        except Exception as e:
            return f"模型暂时不可用：{e}"

    if "天气" in text:
        weather = get_weather_for_city(c)
        return f"今天{c}{weather['weather']}，{weather['temp']}°C"

    if "推荐" in text or "去哪" in text:
        w = get_weather_for_city(c)
        is_rain = "雨" in text or "雨" in str(w.get("weather", ""))
        if is_rain:
            places = [p for p in MOCK_PLACES if p["type"] == "美术馆"]
        else:
            places = MOCK_PLACES[:2]
        reply = "推荐去处：\n"
        for p in places:
            reply += f"📍 {p['name']} - {p['reason']} (¥{p['price']})\n"
        return reply.rstrip()

    if "你好" in text:
        return "你好！我是今天去哪玩助手，问我天气或推荐去哪里玩～"

    return "我还在学习中，可以问我：\n- 上海天气\n- 推荐去哪里玩"


def main() -> None:
    """命令行入口：循环读输入，调用 chat，打印回复；quit/exit/q 退出。"""
    while True:
        user = input("你：")
        if user.strip().lower() in ("quit", "exit", "q"):
            break
        if llm_mod.llm_enabled() and user.strip():
            if os.getenv("LLM_MODE", "off").strip().lower() == "openai":
                print("(正在请求云端模型，请稍候…)", flush=True)
            else:
                print("(正在请求本地 Ollama，首次可能较慢，请稍候…)", flush=True)
        reply = chat(user)
        print(f"助手：{reply}")


if __name__ == "__main__":
    main()
