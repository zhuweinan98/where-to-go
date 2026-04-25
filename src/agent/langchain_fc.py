"""LangChain + Function Calling：由模型决定是否调用天气 / 景点工具。

仅在 ``LLM_MODE`` 为 ``ollama`` / ``openai`` 且 ``LLM_CLIENT=langchain``（默认）时由 ``bot.chat`` 调用。
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.agent.agent_tools import get_weather_json, list_places_json
from src.agent.llm import _llm_debug


def _fc_system_prompt(city: str) -> str:
    return (
        "你是「今天去哪玩」助手。回答简洁、口语化，使用简体中文。\n"
        f"界面默认城市：{city}。若用户话里明确提到其他城市，以该城市为准调用工具。\n"
        "你必须通过工具获取天气与景点数据：\n"
        "- 用户问天气、气温、下雨等 → 调用 get_weather\n"
        "- 用户问推荐、去哪玩、有什么地方 → 先 list_places 再结合 get_weather 做建议\n"
        "禁止编造工具结果中不存在的景点名称；景点名须与 list_places 返回的 name 字段完全一致。"
    )


@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气（实况失败时回退 Mock）。城市名须为应用已注册中文名，如 上海、北京。"""
    return get_weather_json(city)


@tool
def list_places(city: str, query: str = "") -> str:
    """列出应用内该城市的去处（名称、类型、理由、票价等）。query 预留给语义检索，当前可传空字符串。"""
    return list_places_json(city, query)


_TOOLS = [get_weather, list_places]
_TOOL_MAP = {t.name: t for t in _TOOLS}
_MAX_TOOL_ROUNDS = 8


def _clip_text(s: Any, limit: int = 180) -> str:
    txt = str(s)
    if len(txt) <= limit:
        return txt
    return f"{txt[:limit]}...(len={len(txt)})"


def _normalize_v1_base(url: str) -> str:
    u = url.rstrip("/")
    return u if u.endswith("/v1") else f"{u}/v1"


def _get_chat_model_client_info() -> dict[str, Any]:
    mode = os.getenv("LLM_MODE", "off").strip().lower()
    if mode == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
        model = os.getenv("OLLAMA_MODEL", "").strip()
        if not model:
            raise ValueError(
                "已设置 LLM_MODE=ollama，请在环境变量中配置 OLLAMA_MODEL（例如 qwen3:4b）"
            )
        api_key = os.getenv("OLLAMA_API_KEY", "ollama").strip() or "ollama"
        timeout = float(os.getenv("OLLAMA_HTTP_TIMEOUT", "600").strip() or "600")
        return {
            "mode": mode,
            "model": model,
            "api_key": api_key,
            "base_url": _normalize_v1_base(base),
            "timeout": timeout,
        }
    if mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "已设置 LLM_MODE=openai，请在环境变量中配置 OPENAI_API_KEY"
            )
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").strip()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        timeout = float(os.getenv("OPENAI_HTTP_TIMEOUT", "120").strip() or "120")
        return {
            "mode": mode,
            "model": model,
            "api_key": api_key,
            "base_url": _normalize_v1_base(base),
            "timeout": timeout,
        }
    raise ValueError(f"未知的 LLM_MODE：{mode!r}，应为 off / ollama / openai")


def _tool_call_field(tc: Any, key: str) -> str:
    if isinstance(tc, dict):
        v = tc.get(key)
        return str(v) if v is not None else ""
    v = getattr(tc, key, None)
    return str(v) if v is not None else ""


def _tool_call_args(tc: Any) -> dict[str, Any]:
    if isinstance(tc, dict):
        raw = tc.get("args")
        return raw if isinstance(raw, dict) else {}
    raw = getattr(tc, "args", None)
    return raw if isinstance(raw, dict) else {}


def chat_with_tools(user_text: str, city: str) -> str:
    """多轮 tool 循环，返回助手最终正文。"""
    info = _get_chat_model_client_info()
    mode = info["mode"]
    ollama_v1 = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    ollama_v1 = ollama_v1.rstrip("/")
    if not ollama_v1.endswith("/v1"):
        ollama_v1 = f"{ollama_v1}/v1"

    _llm_debug(
        f"已走 LangChain FC，默认城市={city!r}（用户文本可覆盖），model={info['model']!r} → base_url={info['base_url']}"
    )

    llm = ChatOpenAI(
        model=info["model"],
        api_key=info["api_key"],
        base_url=info["base_url"],
        timeout=info["timeout"],
    )
    llm_tools = llm.bind_tools(_TOOLS)
    messages: list[Any] = [
        SystemMessage(content=_fc_system_prompt(city)),
        HumanMessage(content=user_text.strip()),
    ]

    try:
        for i in range(_MAX_TOOL_ROUNDS):
            ai: AIMessage = llm_tools.invoke(messages)
            messages.append(ai)
            _llm_debug(
                f"[fc] round={i + 1} assistant_content={_clip_text(ai.content)}"
            )
            _llm_debug(
                f"[fc] round={i + 1} 模型返回 tool_calls 数量={len(ai.tool_calls or [])}（>0 表示将进入工具执行）"
            )
            if not ai.tool_calls:
                out = (ai.content or "").strip()
                _llm_debug(f"[fc] round={i + 1} final={_clip_text(out)}")
                return out if out else "（模型没有返回文字，请重试或缩短问题。）"
            for tc in ai.tool_calls:
                name = _tool_call_field(tc, "name")
                tid = _tool_call_field(tc, "id")
                args = _tool_call_args(tc)
                _llm_debug(f"[fc] tool_call name={name!r} id={tid!r} args={args}")
                tool_fn = _TOOL_MAP.get(name)
                if tool_fn is None:
                    payload = (
                        f'{{"error": "未知工具 {name!r}，仅允许: {list(_TOOL_MAP)}"}}'
                    )
                    _llm_debug(
                        f"[fc] tool_error name={name!r} error={_clip_text(payload)}"
                    )
                    messages.append(ToolMessage(content=payload, tool_call_id=tid))
                    continue
                try:
                    result = tool_fn.invoke(args)
                    _llm_debug(
                        f"[fc] tool_result name={name!r} result={_clip_text(result)}"
                    )
                except Exception as e:
                    result = f'{{"error": "工具执行失败: {e}"}}'
                    _llm_debug(
                        f"[fc] tool_exception name={name!r} error={_clip_text(result)}"
                    )
                messages.append(ToolMessage(content=str(result), tool_call_id=tid))
        return "（工具调用轮数过多，请简化问题后重试。）"
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
        raise RuntimeError("云端模型请求超时。可尝试增大 OPENAI_HTTP_TIMEOUT。") from e
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
            raise RuntimeError("云端 API 触发限流（429）：请稍后再试。") from e
        brief = getattr(e, "message", None) or str(e)
        raise RuntimeError(f"模型服务返回错误（HTTP {sc}）：{brief}") from e
