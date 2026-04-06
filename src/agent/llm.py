"""本地 Ollama 与云端 OpenAI 兼容 API（同一 Chat Completions 协议）。

LLM_MODE：off | ollama | openai
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


def _llm_debug(msg: str) -> None:
    """与 QWEATHER_DEBUG 一致：打到 stdout，便于 uvicorn / 终端直接看到。"""
    if os.getenv("LLM_DEBUG", "").strip().lower() in ("1", "true", "yes"):
        print(f"[llm] {msg}", flush=True)


def _mode() -> str:
    return os.getenv("LLM_MODE", "off").strip().lower()


def llm_enabled() -> bool:
    """走大模型（本地 Ollama 或云端）。"""
    return _mode() in ("ollama", "openai")


def _normalize_v1_base(url: str) -> str:
    u = url.rstrip("/")
    return u if u.endswith("/v1") else f"{u}/v1"


def _client_and_model() -> tuple[OpenAI, str]:
    m = _mode()
    if m == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
        model = os.getenv("OLLAMA_MODEL", "").strip()
        if not model:
            raise ValueError("已设置 LLM_MODE=ollama，请在环境变量中配置 OLLAMA_MODEL（例如 qwen3:4b）")
        api_key = os.getenv("OLLAMA_API_KEY", "ollama").strip() or "ollama"
        timeout_sec = float(os.getenv("OLLAMA_HTTP_TIMEOUT", "600").strip() or "600")
        client = OpenAI(
            base_url=_normalize_v1_base(base),
            api_key=api_key,
            timeout=timeout_sec,
        )
        return client, model

    if m == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("已设置 LLM_MODE=openai，请在环境变量中配置 OPENAI_API_KEY")
        raw_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").strip()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        timeout_sec = float(os.getenv("OPENAI_HTTP_TIMEOUT", "120").strip() or "120")
        client = OpenAI(
            base_url=_normalize_v1_base(raw_base),
            api_key=key,
            timeout=timeout_sec,
        )
        return client, model

    raise ValueError(f"未知的 LLM_MODE：{_mode()!r}，应为 off / ollama / openai")


def build_system_prompt(city: str, weather: dict[str, Any], places: list[dict[str, Any]]) -> str:
    allowed = [str(p.get("name", "")).strip() for p in places if str(p.get("name", "")).strip()]
    allowed_line = "、".join(allowed) if allowed else "（暂无）"
    return (
        "你是「今天去哪玩」助手。回答简洁、口语化，使用简体中文。\n"
        f"用户当前选择的城市：{city}。\n"
        "以下为应用内唯一可信数据源，不得推荐、暗示或编造其中不存在的景点名称：\n"
        f"weather: {json.dumps(weather, ensure_ascii=False)}\n"
        f"places: {json.dumps(places, ensure_ascii=False)}\n"
        f"允许出现的景点名（须与 places 中 name 一致，不可改写为别称）：{allowed_line}\n"
        "若问天气，只依据 weather；若问推荐或去哪，只能从上述 name 中选一处或多处，并结合天气说明理由。"
        "禁止补充列表外的商场、网红店或其它城市景点。"
    )


def complete(system: str, user: str, *, city: str | None = None) -> str:
    """调用当前配置的模型，返回助手正文。

    city 仅用于 LLM_DEBUG 日志（与 [qweather] 风格一致）；不传则日志里不写城市。
    """
    mode = _mode()
    client, model = _client_and_model()
    ollama_v1 = _normalize_v1_base(
        os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    )
    if mode == "openai":
        route = "已走云端大模型（OpenAI 兼容 API）"
    else:
        route = "已走本地模型（Ollama）"
    city_part = f"，城市={city!r}" if city else ""
    _llm_debug(f"{route}{city_part}，model={model!r} → base_url={client.base_url}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except APIConnectionError as e:
        if mode == "ollama":
            raise RuntimeError(
                "连不上 Ollama。请确认：1) 本机已打开 Ollama；"
                "2) curl http://127.0.0.1:11434/api/tags 有返回；"
                "3) OLLAMA_BASE_URL 使用 http 而非 https；"
                "4) 代理环境下对 127.0.0.1 设置 NO_PROXY。"
                f" 当前基址：{ollama_v1}。详情：{e}"
            ) from e
        raise RuntimeError(
            "连不上云端 API。请检查 OPENAI_BASE_URL、OPENAI_API_KEY、网络与代理。"
            f" 详情：{e}"
        ) from e
    except APITimeoutError as e:
        if mode == "ollama":
            raise RuntimeError(
                "Ollama 请求超时。可尝试增大 OLLAMA_HTTP_TIMEOUT，或减少并发、换更小模型。"
            ) from e
        raise RuntimeError(
            "云端模型请求超时。可尝试增大 OPENAI_HTTP_TIMEOUT。"
        ) from e
    except APIStatusError as e:
        sc = getattr(e, "status_code", None)
        raw = str(e).lower()
        if mode == "openai" and (sc == 402 or "insufficient balance" in raw):
            raise RuntimeError(
                "云端账户余额不足（402）：请到 DeepSeek（或你使用的 API 平台）控制台充值或查看账单；"
                "试用赠金用完也会出现此提示。代码本身无需修改。"
            ) from e
        if mode == "openai" and sc == 401:
            raise RuntimeError(
                "云端 API 鉴权失败（401）：请检查 OPENAI_API_KEY 是否正确、密钥是否仍有效。"
            ) from e
        if mode == "openai" and sc == 429:
            raise RuntimeError(
                "云端 API 触发限流（429）：请稍后再试。"
            ) from e
        brief = getattr(e, "message", None) or str(e)
        raise RuntimeError(f"模型服务返回错误（HTTP {sc}）：{brief}") from e
    usage = getattr(resp, "usage", None)
    if usage is not None:
        _llm_debug(
            "response "
            f"prompt_tokens={usage.prompt_tokens} "
            f"completion_tokens={usage.completion_tokens} "
            f"total_tokens={usage.total_tokens}"
        )
    else:
        _llm_debug("response ok (payload 无 usage 字段)")
    msg = resp.choices[0].message.content
    return (msg or "").strip()
