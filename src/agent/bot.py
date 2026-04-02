"""Agent 核心：规则式对话（Week 1）。

职责：读用户话与城市，结合 Mock 数据生成一句回复。
作用：终端与 Web 共用同一套逻辑（import chat）。
"""

from src.data.mock import MOCK_PLACES, MOCK_WEATHER


def chat(user_input: str, city: str = "上海") -> str:
    """Agent 主入口：输入用户原话与城市，返回助手回复文本。"""
    text = user_input.strip()
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
        reply = chat(user)
        print(f"助手：{reply}")


if __name__ == "__main__":
    main()
