"""Week 1 假数据：地点与天气。

作用：不接真实 API 时，让推荐/天气回复有稳定、可测的内容。
职责：只定义常量；业务判断在 agent.bot 中完成。
"""

# 推荐用地点列表：名称、类型、理由、票价、适宜天气（供后续扩展）
MOCK_PLACES = [
    {
        "name": "顾村公园",
        "type": "公园",
        "reason": "樱花季最佳时期",
        "price": 20,
        "suitable_weather": ["晴", "多云"],
    },
    {
        "name": "西岸美术馆",
        "type": "美术馆",
        "reason": "室内场所，新展印象派",
        "price": 100,
        "suitable_weather": ["晴", "雨", "多云"],
    },
    {
        "name": "外滩",
        "type": "景点",
        "reason": "夜景很美，适合散步",
        "price": 0,
        "suitable_weather": ["晴", "多云"],
    },
]

# 城市 → 当日 Mock 天气（天气文案 + 气温）
MOCK_WEATHER = {
    "上海": {"weather": "晴", "temp": 25},
    "北京": {"weather": "多云", "temp": 20},
    "广州": {"weather": "雨", "temp": 28},
}
