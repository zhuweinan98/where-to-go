"""Week 1 假数据：地点列表。

天气 Mock 回退与城名全集见 ``config/cities.json``（``cities_config``）。景点的 ``city`` 须在 JSON 中注册。
职责：``MOCK_PLACES`` + ``places_for_city``；业务判断在 agent.bot 中完成。
"""

from __future__ import annotations

from typing import Any

from src.data.cities_config import ALL_CITY_NAMES, MOCK_WEATHER

# 推荐用地点：须含 ``city``，与当前对话城市对齐后传入规则 / LLM。
# 票价为约示性人民币（实地以景区公示为准）。
MOCK_PLACES: list[dict[str, Any]] = [
    # ---------- 上海 ----------
    {
        "city": "上海",
        "name": "顾村公园",
        "type": "公园",
        "reason": "春季樱花季人气高，适合晴天户外散步",
        "price": 20,
        "suitable_weather": ["晴", "多云"],
    },
    {
        "city": "上海",
        "name": "西岸美术馆",
        "type": "美术馆",
        "reason": "滨江西岸，室内看展，雨天也合适",
        "price": 100,
        "suitable_weather": ["晴", "雨", "多云"],
    },
    {
        "city": "上海",
        "name": "外滩",
        "type": "景点",
        "reason": "黄浦江畔经典天际线，夜景尤佳",
        "price": 0,
        "suitable_weather": ["晴", "多云"],
    },
    # ---------- 北京 ----------
    {
        "city": "北京",
        "name": "故宫博物院",
        "type": "景点",
        "reason": "明清皇宫与世界遗产，需提前预约",
        "price": 60,
        "suitable_weather": ["晴", "多云"],
    },
    {
        "city": "北京",
        "name": "颐和园",
        "type": "景点",
        "reason": "皇家园林，昆明湖与长廊适合半日游览",
        "price": 30,
        "suitable_weather": ["晴", "多云"],
    },
    {
        "city": "北京",
        "name": "中国国家博物馆",
        "type": "博物馆",
        "reason": "免费预约，展陈丰富，雨天室内首选之一",
        "price": 0,
        "suitable_weather": ["晴", "雨", "多云"],
    },
    # ---------- 广州 ----------
    {
        "city": "广州",
        "name": "陈家祠",
        "type": "景点",
        "reason": "岭南祠堂建筑与木雕砖雕精华",
        "price": 10,
        "suitable_weather": ["晴", "多云"],
    },
    {
        "city": "广州",
        "name": "沙面岛",
        "type": "景点",
        "reason": "欧陆历史街区，沿江散步拍照",
        "price": 0,
        "suitable_weather": ["晴", "多云"],
    },
    {
        "city": "广州",
        "name": "广东美术馆",
        "type": "美术馆",
        "reason": "二沙岛临珠江，常设当代与主题展",
        "price": 0,
        "suitable_weather": ["晴", "雨", "多云"],
    },
    {
        "city": "广州",
        "name": "广州塔",
        "type": "景点",
        "reason": "城市地标，登塔可俯瞰珠江新城",
        "price": 150,
        "suitable_weather": ["晴", "多云"],
    },
    # ---------- 沈阳 ----------
    {
        "city": "沈阳",
        "name": "保利云禧韩毓家",
        "type": "休闲",
        "reason": "熟人小聚、休闲落脚点",
        "price": 0,
        "suitable_weather": ["晴", "雨", "多云"],
    },
    # ---------- 石家庄 ----------
    {
        "city": "石家庄",
        "name": "正定古城·隆兴寺",
        "type": "景点",
        "reason": "千年古刹与宋代木构，正定古城核心",
        "price": 50,
        "suitable_weather": ["晴", "多云"],
    },
    {
        "city": "石家庄",
        "name": "赵州桥",
        "type": "景点",
        "reason": "隋代敞肩石拱桥，中国古代桥梁标志",
        "price": 40,
        "suitable_weather": ["晴", "多云"],
    },
    {
        "city": "石家庄",
        "name": "西柏坡纪念馆",
        "type": "景点",
        "reason": "红色旅游经典，革命旧址与陈列馆",
        "price": 0,
        "suitable_weather": ["晴", "雨", "多云"],
    },
    {
        "city": "石家庄",
        "name": "河北博物院",
        "type": "博物馆",
        "reason": "满城汉墓与燕赵文化常设展，室内避雨",
        "price": 0,
        "suitable_weather": ["晴", "雨", "多云"],
    },
    # ---------- 盘锦 ----------
    {
        "city": "盘锦",
        "name": "白鹭郡戚建北家",
        "type": "休闲",
        "reason": "熟人小聚、休闲落脚点",
        "price": 0,
        "suitable_weather": ["晴", "雨", "多云"],
    },
]


def _validate_place_cities() -> None:
    for p in MOCK_PLACES:
        c = str(p.get("city", ""))
        if c and c not in ALL_CITY_NAMES:
            raise ValueError(
                f"MOCK_PLACES 中的城市 {c!r} 未在 config/cities.json 注册，请先补充城市列表"
            )


_validate_place_cities()


def places_for_city(city: str) -> list[dict[str, Any]]:
    """返回该城景点列表；未单独收录的城市暂以上海数据兜底（与早期全局列表行为一致）。"""
    c = (city or "上海").strip() or "上海"
    found = [p for p in MOCK_PLACES if p.get("city") == c]
    if found:
        return found
    return [p for p in MOCK_PLACES if p.get("city") == "上海"]
