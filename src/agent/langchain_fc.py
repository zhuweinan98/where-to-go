"""LangChain + Function Calling：由模型决定是否调用天气 / 景点工具。

仅在 ``LLM_MODE`` 为 ``ollama`` / ``openai`` 时由 ``bot.chat`` 调用。
"""

from __future__ import annotations

import os
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.agent.agent_tools import (
    get_weather_json,
    list_places_json,
    search_sanguo_places_json,
    search_sanguo_places_with_meta,
)
from src.data.rag_retriever import consume_rag_debug_logs


def _llm_debug(msg: str) -> None:
    if os.getenv("LLM_DEBUG", "").strip().lower() in ("1", "true", "yes"):
        print(f"[llm] {msg}", flush=True)


def _fc_system_prompt(city: str) -> str:
    return (
        "你是「今天去哪玩」助手。回答简洁、口语化，使用简体中文。\n"
        f"界面默认城市：{city}。若用户话里明确提到其他城市，以该城市为准调用工具。\n"
        "你必须通过工具获取天气与景点数据：\n"
        "- 用户问天气、气温、下雨等 → 调用 get_weather\n"
        "- 用户问推荐、去哪玩、有什么地方 → 先 list_places 再结合 get_weather 做建议\n"
        "- 若已提供「三国检索结果」上下文，优先基于该上下文回答，不要忽略\n"
        "- 若无三国检索上下文，且用户问三国、蜀汉、曹操、赤壁等历史主题地点 → 调用 search_sanguo_places\n"
        "- search_sanguo_places 返回 JSON 含 places（结构化景点）与 romance_excerpts（演义节选）；"
        "推荐理由须基于 places 字段；与 places 冲突时以 places 为准。\n"
        "【硬性·三国/演义】满足任一即须遵守：用户话含三国相关词；或本对话已注入三国知识库 System；或你已调用 search_sanguo_places 并收到 tool 结果。\n"
        "1）若 romance_excerpts 非空，或 System 内已有「[演义摘录」段落：最终回复正文必须含恰好一行 "
        "【演义原文】「……」——引号内须为 excerpt 正文中连续原字照抄 30～120 个汉字（禁止意译改写后再套引号），"
        "同一行末尾括号内写明该条 chapter_title 与 chunk_id（与 JSON/System 中一致）。\n"
        "2）若 romance_excerpts 为空且 System 中亦无演义摘录块：必须写一行 "
        "【演义原文】本轮未检索到节选，推荐仅据 places。\n"
        "3）推荐结论以 places 为准；「演义原文」行不得与 places 地名矛盾，矛盾时仍以 places 为准，但仍须完成(1)或(2)。\n"
        "禁止编造工具结果中不存在的景点名称；景点名须与 list_places 返回的 name 字段完全一致。"
    )


def _is_sanguo_query(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    keywords = (
        "三国",
        "蜀汉",
        "曹操",
        "刘备",
        "诸葛亮",
        "赤壁",
        "武侯祠",
        "白帝城",
        "关羽",
        "张飞",
        "司马",
        "魏国",
        "吴国",
    )
    return any(k in t for k in keywords)


def _build_sanguo_context(
    user_text: str,
) -> tuple[str, list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    items, rag_logs, romance = search_sanguo_places_with_meta(user_text)
    if not isinstance(items, list):
        return "", [], rag_logs, romance if isinstance(romance, list) else []
    picked = items[:3]
    if not picked:
        return "", [], rag_logs, romance if isinstance(romance, list) else []
    lines = []
    for idx, x in enumerate(picked, start=1):
        lines.append(
            f"{idx}. {x.get('name', '')}（{x.get('province', '')}{x.get('modern_city', '')}）"
            f" | 史实: {x.get('era_role', '')}"
            f" | 标签: {','.join(x.get('tags', []))}"
            f" | 游玩建议: {x.get('visit_hint', '')}"
            f" | score={x.get('_score', 0)}"
        )
    ctx = "以下是三国知识库检索结果（仅可基于这些信息回答）：\n" + "\n".join(lines)
    if not romance and picked:
        ctx += (
            "\n\n【硬性】本轮未附上《演义》节选（词法未命中或 jsonl 不可用）。"
            "最终用户可见回复须含一行：【演义原文】本轮未检索到节选，推荐仅据 places。"
        )
    if romance:
        rlines = []
        for i, r in enumerate(romance, start=1):
            ct = str(r.get("chapter_title", "") or "")
            cid = str(r.get("chunk_id", "") or "")
            ex = str(r.get("excerpt", "") or "")
            rlines.append(f"[演义摘录{i}] {ct}（{cid}）\n{ex}")
        ctx += (
            "\n\n以下为《三国演义》原文节选（词法检索 jsonl）。"
            "地点与游玩信息以「三国知识库」上表为准；"
            "你须在最终用户回复中按 fc_system 硬性规定写出「【演义原文】」行，从下列摘录中照抄 30～120 字：\n\n"
            + "\n\n".join(rlines)
        )
    return ctx, picked, rag_logs, romance


def _rag_diag(engine: str, emb_model: str) -> dict[str, str]:
    configured_mode = os.getenv("RAG_EMBEDDING_MODE", "auto").strip().lower() or "auto"
    configured_model = os.getenv("RAG_EMBEDDING_MODEL", "").strip()
    if engine == "semantic":
        return {
            "status": "semantic_ok",
            "configured_mode": configured_mode,
            "configured_model": configured_model,
            "hint": "语义检索已生效。",
        }
    if configured_mode == "off":
        hint = "已显式关闭 embedding（RAG_EMBEDDING_MODE=off），当前仅词法检索。"
    elif emb_model:
        hint = "已回退到词法检索。请查看后端 [rag] 日志中的 embedding 失败原因。"
    else:
        hint = (
            "已回退到词法检索。若使用 Ollama，请先 pull embedding 模型并配置 "
            "RAG_EMBEDDING_MODEL。"
        )
    return {
        "status": "lexical_fallback",
        "configured_mode": configured_mode,
        "configured_model": configured_model,
        "hint": hint,
    }


@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气（实况失败时回退 Mock）。城市名须为应用已注册中文名，如 上海、北京。"""
    return get_weather_json(city)


@tool
def list_places(city: str, query: str = "") -> str:
    """列出应用内该城市的去处（名称、类型、理由、票价等）。当用户有偏好关键词时请填写 query 以检索 top-k。"""
    return list_places_json(city, query)


@tool
def search_sanguo_places(query: str) -> str:
    """检索三国历史主题地点：返回 JSON（places + romance_excerpts）。收到后须在回复中按系统硬性规定写出「【演义原文】」行。"""
    return search_sanguo_places_json(query)


_TOOLS = [get_weather, list_places, search_sanguo_places]
_TOOL_MAP = {t.name: t for t in _TOOLS}
_MAX_TOOL_ROUNDS = 8
# debug_trace 里 SystemMessage 全文上限（避免极端配置撑爆 JSON）
_SYSTEM_FULL_TEXT_TRACE_CAP = 65536


def _clip_text(s: Any, limit: int = 180) -> str:
    txt = str(s)
    if len(txt) <= limit:
        return txt
    return f"{txt[:limit]}...(len={len(txt)})"


def _lc_message_role(msg: Any) -> str:
    t = getattr(msg, "type", None)
    if isinstance(t, str) and t:
        return t
    name = type(msg).__name__
    if name.endswith("Message"):
        return name[: -len("Message")].lower()
    return name.lower()


def _system_message_blocks(messages: list[Any]) -> list[dict[str, Any]]:
    """当前 messages 里按出现顺序提取 SystemMessage，供每轮 invoke 前可观测。"""
    sys_i = 0
    blocks: list[dict[str, Any]] = []
    for m in messages:
        if _lc_message_role(m) != "system":
            continue
        sys_i += 1
        body = str(getattr(m, "content", "") or "")
        if sys_i == 1:
            slot = "fc_system"
        elif sys_i == 2:
            slot = "sanguo_rag_system"
        else:
            slot = f"system_extra_{sys_i}"
        full = (
            body
            if len(body) <= _SYSTEM_FULL_TEXT_TRACE_CAP
            else _clip_text(body, _SYSTEM_FULL_TEXT_TRACE_CAP)
        )
        blocks.append(
            {
                "slot": slot,
                "chars": len(body),
                "preview": _clip_text(body, 3200),
                "full_text": full,
            }
        )
    return blocks


def _emit_fc_invoke_system_snapshot(
    round_no: int,
    messages: list[Any],
    trace: list[dict[str, Any]] | None,
    trace_hook: Callable[[dict[str, Any]], None] | None,
) -> None:
    blocks = _system_message_blocks(messages)
    roles = ",".join(_lc_message_role(m) for m in messages)
    _add_trace(
        trace,
        trace_hook,
        "fc_invoke_system_messages",
        round=round_no,
        total_messages=len(messages),
        roles=roles,
        system_blocks=blocks,
    )
    if not blocks:
        _llm_debug(
            f"✅ invoke 前：round={round_no} system_blocks=0 total_messages={len(messages)} "
            f"roles={roles}（本轮无 SystemMessage，罕见）"
        )
        return
    _llm_debug(
        f"✅ invoke 前 SystemMessage 栈：round={round_no} system_blocks={len(blocks)} "
        f"total_messages={len(messages)} roles={roles}"
    )
    for b in blocks:
        _llm_debug(
            f"  · slot={b['slot']} chars={b['chars']} preview={b['preview']}"
        )


def _add_trace(
    trace: list[dict[str, Any]] | None,
    trace_hook: Callable[[dict[str, Any]], None] | None,
    event: str,
    **data: Any,
) -> None:
    item = {"event": event, **data}
    if trace is not None:
        trace.append(item)
    if trace_hook is not None:
        trace_hook(item)


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


def chat_with_tools(
    user_text: str,
    city: str,
    *,
    trace: list[dict[str, Any]] | None = None,
    trace_hook: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """多轮 tool 循环，返回助手最终正文。"""
    info = _get_chat_model_client_info()
    mode = info["mode"]
    ollama_v1 = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    ollama_v1 = ollama_v1.rstrip("/")
    if not ollama_v1.endswith("/v1"):
        ollama_v1 = f"{ollama_v1}/v1"

    _llm_debug(
        "✅ FC 启动（LangChain + bind_tools）："
        f"mode={info['mode']} model={info['model']} default_city={city!r} "
        f"base_url={info['base_url']} max_tool_rounds={_MAX_TOOL_ROUNDS}"
    )
    _add_trace(
        trace,
        trace_hook,
        "fc_start",
        default_city=city,
        mode=info["mode"],
        model=info["model"],
        base_url=info["base_url"],
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
    ]
    text = user_text.strip()
    if _is_sanguo_query(text):
        rag_context, rag_hits, rag_logs, romance_hits = _build_sanguo_context(text)
        if rag_context:
            engine = str(rag_hits[0].get("_retrieval_engine", "")) if rag_hits else ""
            emb_model = str(rag_hits[0].get("_embedding_model", "")) if rag_hits else ""
            diag = _rag_diag(engine, emb_model)
            hit_names = "、".join(str(x.get("name", "")) for x in rag_hits)
            rom_trace = [
                {
                    "chunk_id": str(x.get("chunk_id", "")),
                    "chapter_title": str(x.get("chapter_title", "")),
                    "excerpt_len": len(str(x.get("excerpt", "") or "")),
                }
                for x in (romance_hits or [])
            ]
            _llm_debug(
                "✅ 三国预检索命中（将注入 SystemMessage）："
                f"top_k={len(rag_hits)} route={engine or 'unknown'} "
                f"embedding_model={emb_model or '-'} hits={hit_names} "
                f"romance_excerpts={len(rom_trace)}"
            )
            _add_trace(
                trace,
                trace_hook,
                "rag_retrieved",
                source="sanguo_kb",
                query=text,
                top_k=len(rag_hits),
                retrieval_engine=engine or "unknown",
                embedding_model=emb_model or "",
                rag_status=diag["status"],
                rag_config_mode=diag["configured_mode"],
                rag_config_model=diag["configured_model"],
                rag_hint=diag["hint"],
                rag_debug_logs=rag_logs,
                hits=rag_hits,
                romance_hits=rom_trace,
            )
            messages.append(SystemMessage(content=rag_context))
        else:
            _llm_debug(
                "ℹ️ 三国预检索：无可用上下文（hits=0），继续通用 FC（仅系统 Prompt + 用户话）"
            )
            _add_trace(
                trace,
                trace_hook,
                "rag_retrieved",
                source="sanguo_kb",
                query=text,
                top_k=0,
                rag_debug_logs=rag_logs,
                hits=[],
                romance_hits=[],
            )
    else:
        _add_trace(
            trace,
            trace_hook,
            "rag_skipped",
            source="sanguo_kb",
            reason="query_not_sanguo",
        )
    messages.append(HumanMessage(content=text))

    try:
        for i in range(_MAX_TOOL_ROUNDS):
            _emit_fc_invoke_system_snapshot(i + 1, messages, trace, trace_hook)
            ai: AIMessage = llm_tools.invoke(messages)
            messages.append(ai)
            _add_trace(
                trace,
                trace_hook,
                "assistant_round",
                round=i + 1,
                assistant_content=_clip_text(ai.content),
                tool_calls_count=len(ai.tool_calls or []),
            )
            n_tc = len(ai.tool_calls or [])
            _llm_debug(
                f"⏳ LLM round={i + 1}：assistant 正文片段={_clip_text(ai.content)} "
                f"tool_calls={n_tc}（Function Calling，>0 将执行工具）"
            )
            if not ai.tool_calls:
                out = (ai.content or "").strip()
                _llm_debug(
                    f"✅ LLM round={i + 1} 结束：无 tool_calls，final_answer={_clip_text(out)}"
                )
                _add_trace(
                    trace,
                    trace_hook,
                    "final_answer",
                    round=i + 1,
                    content=_clip_text(out),
                )
                return out if out else "（模型没有返回文字，请重试或缩短问题。）"
            for tc in ai.tool_calls:
                name = _tool_call_field(tc, "name")
                tid = _tool_call_field(tc, "id")
                args = _tool_call_args(tc)
                _add_trace(
                    trace,
                    trace_hook,
                    "tool_call",
                    round=i + 1,
                    name=name,
                    id=tid,
                    args=args,
                )
                _llm_debug(
                    f"⏳ tool_call：round={i + 1} name={name} id={tid} args={args}"
                )
                tool_fn = _TOOL_MAP.get(name)
                if tool_fn is None:
                    payload = (
                        f'{{"error": "未知工具 {name!r}，仅允许: {list(_TOOL_MAP)}"}}'
                    )
                    _llm_debug(
                        f"❌ tool_error：round={i + 1} name={name} error={_clip_text(payload)}"
                    )
                    _add_trace(
                        trace,
                        trace_hook,
                        "tool_error",
                        round=i + 1,
                        name=name,
                        error=_clip_text(payload),
                    )
                    messages.append(ToolMessage(content=payload, tool_call_id=tid))
                    continue
                try:
                    result = tool_fn.invoke(args)
                    rag_logs: list[str] = []
                    if name in ("list_places", "search_sanguo_places"):
                        rag_logs = consume_rag_debug_logs()
                    _llm_debug(
                        f"✅ tool_result：round={i + 1} name={name} "
                        f"bytes={len(str(result))} preview={_clip_text(result, 240)}"
                    )
                    tr_tool: dict[str, Any] = {
                        "round": i + 1,
                        "name": name,
                        # 调试面板需完整 JSON 才能解析 TopN；与 LLM 上下文截断分离
                        "result": _clip_text(result, limit=16000),
                    }
                    if rag_logs:
                        tr_tool["rag_debug_logs"] = rag_logs
                    _add_trace(trace, trace_hook, "tool_result", **tr_tool)
                except Exception as e:
                    if name in ("list_places", "search_sanguo_places"):
                        consume_rag_debug_logs()
                    result = f'{{"error": "工具执行失败: {e}"}}'
                    _llm_debug(
                        f"❌ tool_exception：round={i + 1} name={name} error={_clip_text(result)}"
                    )
                    _add_trace(
                        trace,
                        trace_hook,
                        "tool_exception",
                        round=i + 1,
                        name=name,
                        error=_clip_text(result),
                    )
                messages.append(ToolMessage(content=str(result), tool_call_id=tid))
        _add_trace(trace, trace_hook, "max_rounds_reached", max_rounds=_MAX_TOOL_ROUNDS)
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
