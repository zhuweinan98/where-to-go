"""天气数据：配置和风 QWeather 时用实时接口，否则回退 ``cities_config.MOCK_WEATHER``（来自 ``config/cities.json``）。

认证（二选一，均需 ``QWEATHER_HOST``）：

- **API KEY**：``QWEATHER_KEY``；``QWEATHER_AUTH`` = ``auto``（默认）时先 ``X-QW-Api-Key``，401 再试 URL ``key=``。
- **JWT**：控制台用 Ed25519 签发的 Token，填入 ``QWEATHER_BEARER_TOKEN``（``Authorization: Bearer …``），与 API KEY 互斥时优先 Bearer。

401 且两种方式都试过仍失败：Key 错误、Token 过期、或 Host 与凭据不属于同一项目。开启 ``QWEATHER_DEBUG=1`` 会打印错误响应片段。
文档：https://dev.qweather.com/docs/configuration/authentication/
"""

from __future__ import annotations

import os
import traceback
from typing import Any

import httpx

from src.data.cities_config import CITY_TO_LOCATION_ID, MOCK_WEATHER


def _weather_debug() -> bool:
    return os.getenv("QWEATHER_DEBUG", "").strip().lower() in ("1", "true", "yes")


def _dlog(msg: str) -> None:
    """QWEATHER_DEBUG 时直接打到 stdout，避免 uvicorn 默认不展示第三方 logger.info。"""
    print(f"[qweather] {msg}", flush=True)

# CITY_TO_LOCATION_ID 由 config/cities.json 加载；其余城市走 /geo/v2/city/lookup 动态解析并缓存
_geo_id_cache: dict[str, str] = {}


def _api_base() -> str:
    raw = os.getenv("QWEATHER_HOST", "").strip().rstrip("/")
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        return raw
    return f"https://{raw}"


def _qweather_credential_ok() -> bool:
    return bool(os.getenv("QWEATHER_KEY", "").strip()) or bool(
        os.getenv("QWEATHER_BEARER_TOKEN", "").strip()
    )


def qweather_enabled() -> bool:
    return bool(_api_base()) and _qweather_credential_ok()


def _qweather_auth_mode() -> str:
    m = (os.getenv("QWEATHER_AUTH") or "auto").strip().lower()
    return m if m in ("auto", "header", "query") else "auto"


def _qweather_request(client: httpx.Client, path: str, params: dict[str, str]) -> httpx.Response:
    """发起 GET。优先 JWT Bearer；否则 API KEY（auto 时 Header 401 再试 query key）。"""
    base = _api_base()
    url = f"{base}{path}"
    bearer = os.getenv("QWEATHER_BEARER_TOKEN", "").strip()
    gzip_only = {"Accept-Encoding": "gzip"}

    if bearer:
        return client.get(
            url,
            params=params,
            headers={**gzip_only, "Authorization": f"Bearer {bearer}"},
        )

    key = os.getenv("QWEATHER_KEY", "").strip()
    if not key:
        raise ValueError("已启用和风 Host 但未设置 QWEATHER_KEY 或 QWEATHER_BEARER_TOKEN")

    mode = _qweather_auth_mode()

    def with_header() -> httpx.Response:
        return client.get(
            url,
            params=params,
            headers={**gzip_only, "X-QW-Api-Key": key},
        )

    def with_query() -> httpx.Response:
        return client.get(url, params={**params, "key": key}, headers=gzip_only)

    if mode == "query":
        return with_query()
    if mode == "header":
        return with_header()
    r = with_header()
    if r.status_code == 401:
        if _weather_debug():
            _dlog("和风返回 401（X-QW-Api-Key），改用 URL 参数 key= 重试")
        rq = with_query()
        if rq.status_code == 401 and _weather_debug():
            _dlog(f"仍 401，响应片段: {rq.text[:400]!r}")
        return rq
    return r


def _log_http_error_if_debug(r: httpx.Response) -> None:
    if _weather_debug() and r.status_code >= 400:
        _dlog(f"HTTP {r.status_code} 响应片段: {r.text[:400]!r}")


def _resolve_location_id(city: str) -> str | None:
    """返回和风 LocationID：先查静态表与进程内缓存，再请求 Geo 城市搜索。"""
    c = city.strip()
    if not c:
        return None
    if c in CITY_TO_LOCATION_ID:
        return CITY_TO_LOCATION_ID[c]
    if c in _geo_id_cache:
        return _geo_id_cache[c]
    if not qweather_enabled():
        return None
    timeout = float(os.getenv("QWEATHER_TIMEOUT", "10").strip() or "10")
    with httpx.Client(timeout=timeout) as client:
        r = _qweather_request(
            client,
            "/geo/v2/city/lookup",
            {"location": c, "number": "1", "range": "cn"},
        )
        _log_http_error_if_debug(r)
        r.raise_for_status()
        data = r.json()
    if str(data.get("code")) != "200":
        return None
    locs = data.get("location") or []
    if not locs:
        return None
    lid = locs[0].get("id")
    if not lid:
        return None
    sid = str(lid)
    _geo_id_cache[c] = sid
    return sid


def _fetch_qweather_now(city: str) -> dict[str, Any]:
    loc = _resolve_location_id(city)
    if not loc:
        raise ValueError(f"无法解析城市和风 LocationID：{city!r}")
    timeout = float(os.getenv("QWEATHER_TIMEOUT", "10").strip() or "10")
    with httpx.Client(timeout=timeout) as client:
        r = _qweather_request(client, "/v7/weather/now", {"location": loc})
        _log_http_error_if_debug(r)
        r.raise_for_status()
        data = r.json()
    code = str(data.get("code", ""))
    if code != "200":
        raise RuntimeError(f"和风返回 code={code!r}")
    now = data.get("now") or {}
    text = str(now.get("text") or "未知").strip() or "未知"
    raw_temp = now.get("temp", "0")
    try:
        temp = int(float(raw_temp))
    except (TypeError, ValueError):
        temp = 0
    return {"weather": text, "temp": temp}


def get_weather_for_city(city: str) -> dict[str, Any]:
    """与 MOCK_WEATHER 单条结构一致：weather（文案）、temp（整数 ℃）。"""
    c = (city or "").strip() or "上海"
    if not qweather_enabled():
        if _weather_debug():
            _dlog(
                f"未走和风 API（需 QWEATHER_HOST 且 QWEATHER_KEY 或 QWEATHER_BEARER_TOKEN），城市={c!r} → Mock"
            )
        return MOCK_WEATHER.get(c, {"weather": "未知", "temp": 0})
    try:
        out = _fetch_qweather_now(c)
        if _weather_debug():
            _dlog(f"已走和风 API，城市={c!r} → {out}")
        return out
    except Exception as e:
        if _weather_debug():
            _dlog(f"和风 API 失败，城市={c!r}，已回退 Mock。原因：{e!r}")
            traceback.print_exc()
        return MOCK_WEATHER.get(c, {"weather": "未知", "temp": 0})
