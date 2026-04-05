"""Agent 核心：规则式对话（Week 1）。

职责：读用户话与城市，结合 Mock 数据生成一句回复。
作用：终端与 Web 共用同一套逻辑（import chat）。

默认仅规则引擎（单测依赖）；LLM_MODE=ollama 时走本地 Ollama。天气/推荐规则在本地 LLM
测通、愿意改为纯模型行为之前保留，勿删。
"""

from dotenv import load_dotenv

load_dotenv()

from src.agent import llm as llm_mod
from src.data.mock import MOCK_PLACES, MOCK_WEATHER


def chat(user_input: str, city: str = "上海") -> str:
    """Agent 主入口：输入用户原话与城市，返回助手回复文本。"""
    text = user_input.strip()
    if llm_mod.cloud_mode_requested():
        return (
            "云端大模型已预留环境变量位（OPENAI_*），尚未接入；"
            "请先用 LLM_MODE=ollama 接本地，或改回 off 使用规则回复。"
        )
    if llm_mod.llm_enabled():
        c = (city or "上海").strip() or "上海"
        weather = MOCK_WEATHER.get(c, {"weather": "未知", "temp": 0})
        system = llm_mod.build_system_prompt(c, weather, MOCK_PLACES)
        try:
            out = llm_mod.complete(system, text)
            return out if out.strip() else "（模型没有返回文字，请重试或缩短问题。）"
        except Exception as e:
            return f"模型暂时不可用：{e}"

    if "天气" in text:
        weather = MOCK_WEATHER.get(city, {"weather": "未知", "temp": 0})
        return f"今天{city}{weather['weather']}，{weather['temp']}°C"

    if "推荐" in text or "去哪" in text:
        w = MOCK_WEATHER.get(city, {})
        is_rain = "雨" in text or w.get("weather") == "雨"
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
            print("(正在请求本地 Ollama，首次可能需几十秒到数分钟，请稍候…)", flush=True)
        reply = chat(user)
        print(f"助手：{reply}")


if __name__ == "__main__":
    main()
