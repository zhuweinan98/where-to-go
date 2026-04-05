"""本地 Ollama（OpenAI 兼容 /v1）与云端占位说明。

当前仅实现 LLM_MODE=ollama。云端走 OPENAI_* 的接入预留为环境变量与下方注释，测通本地后再接云。"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


def _mode() -> str:
    return os.getenv("LLM_MODE", "off").strip().lower()


def llm_enabled() -> bool:
    """仅本地 Ollama 为已接入能力。"""
    return _mode() == "ollama"


def cloud_mode_requested() -> bool:
    """用户显式选了云端模式（尚未实现调用，用于提示）。"""
    return _mode() == "openai"


def _normalize_v1_base(url: str) -> str:
    u = url.rstrip("/")
    return u if u.endswith("/v1") else f"{u}/v1"


def _client_and_model() -> tuple[OpenAI, str]:
    m = _mode()
    if m == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
        model = os.getenv("OLLAMA_MODEL", "").strip()
        if not model:
            raise ValueError("已设置 LLM_MODE=ollama，请在环境变量中配置 OLLAMA_MODEL（例如 llama3.2）")
        api_key = os.getenv("OLLAMA_API_KEY", "ollama").strip() or "ollama"
        # 本地首次加载权重、CPU 推理可能很慢，默认客户端超时容易误判为「无回复」
        timeout_sec = float(os.getenv("OLLAMA_HTTP_TIMEOUT", "600").strip() or "600")
        client = OpenAI(
            base_url=_normalize_v1_base(base),
            api_key=api_key,
            timeout=timeout_sec,
        )
        return client, model

    # 云端预留：在此读取 OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL，
    # 与 ollama 分支相同方式构造 OpenAI(base_url=..., api_key=...)，测通后再删掉本注释并实现。
    raise ValueError(f"未知的 LLM_MODE：{_mode()!r}，当前支持 off / ollama；openai 为预留尚未接入")


def build_system_prompt(city: str, weather: dict[str, Any], places: list[dict[str, Any]]) -> str:
    return (
        "你是「今天去哪玩」助手。回答简洁、口语化，使用简体中文。\n"
        f"用户当前选择的城市：{city}。\n"
        "以下 JSON 是应用内的模拟数据，仅作参考，不要编造列表外地点：\n"
        f"weather: {json.dumps(weather, ensure_ascii=False)}\n"
        f"places: {json.dumps(places, ensure_ascii=False)}\n"
        "若问天气，结合 weather；若问推荐或去哪，结合天气从 places 里挑选并说明理由。"
    )


def complete(system: str, user: str) -> str:
    """调用当前配置的模型，返回助手正文。"""
    client, model = _client_and_model()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    msg = resp.choices[0].message.content
    return (msg or "").strip()
