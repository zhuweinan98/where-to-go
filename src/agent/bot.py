"""Agent 核心：规则式对话（Week 1）。

职责：读用户话与城市，结合 Mock 数据生成一句回复。
作用：终端与 Web 共用同一套逻辑（import chat）。

默认仅规则引擎（单测依赖）；LLM_MODE=ollama / openai 时走 LangChain Function Calling。

LLM_MODE=off 时仅返回兜底提示。
"""

import os
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()


def _llm_mode() -> str:
    return os.getenv("LLM_MODE", "off").strip().lower()


def _llm_enabled() -> bool:
    return _llm_mode() in ("ollama", "openai")


def _append_trace(
    debug_trace: list[dict[str, Any]] | None,
    trace_hook: Callable[[dict[str, Any]], None] | None,
    event: str,
    **data: Any,
) -> None:
    item = {"event": event, **data}
    if debug_trace is not None:
        debug_trace.append(item)
    if trace_hook is not None:
        trace_hook(item)


def chat(
    user_input: str,
    city: str = "上海",
    *,
    debug_trace: list[dict[str, Any]] | None = None,
    trace_hook: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Agent 主入口：输入用户原话与城市，返回助手回复文本。"""
    text = user_input.strip()
    c = (city or "上海").strip() or "上海"
    _append_trace(debug_trace, trace_hook, "city_resolved", effective_city=c)
    if _llm_enabled():
        _append_trace(debug_trace, trace_hook, "llm_route", client_kind="langchain")
        try:
            from src.agent.langchain_fc import chat_with_tools

            out = chat_with_tools(text, c, trace=debug_trace, trace_hook=trace_hook)
            return out if out.strip() else "（模型没有返回文字，请重试或缩短问题。）"
        except Exception as e:
            _append_trace(debug_trace, trace_hook, "llm_error", error=str(e))
            return f"模型暂时不可用：{e}"
    _append_trace(debug_trace, trace_hook, "rule_off")
    return (
        "当前为规则关闭模式（LLM_MODE=off）。\n"
        "请将 LLM_MODE 设为 ollama 或 openai 后再试。"
    )


def main() -> None:
    """命令行入口：循环读输入，调用 chat，打印回复；quit/exit/q 退出。"""
    while True:
        user = input("你：")
        if user.strip().lower() in ("quit", "exit", "q"):
            break
        if _llm_enabled() and user.strip():
            if _llm_mode() == "openai":
                print("(正在请求云端模型，请稍候…)", flush=True)
            else:
                print("(正在请求本地 Ollama，首次可能较慢，请稍候…)", flush=True)
        reply = chat(user)
        print(f"助手：{reply}")


if __name__ == "__main__":
    main()
