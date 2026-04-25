"""FastAPI 应用：HTTP 与页面。

职责：
  - GET / ：返回聊天页（Jinja2 模板 index.html）
  - POST /chat ：解析表单，调用 agent.chat，返回 JSON { "reply": "..." }
作用：把浏览器/接口与 bot 里的 Agent 逻辑连接起来。
"""

import asyncio
import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from starlette.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

from src.agent.bot import chat


def _llm_loading_hint() -> str:
    """首页「思考中」文案：与 LLM_MODE 一致（bot 已 load_dotenv）。"""
    m = os.getenv("LLM_MODE", "off").strip().lower()
    if m == "ollama":
        return "助手：思考中…（本地模型可能较慢，请稍等）"
    if m == "openai":
        return "助手：思考中…（正在请求云端模型，请稍等）"
    return "助手：思考中…"


def _llm_debug_enabled() -> bool:
    return os.getenv("LLM_DEBUG", "").strip().lower() in ("1", "true", "yes")


_ROOT = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(_ROOT / "templates"))

# FastAPI 应用实例：uvicorn 加载的就是这个 app
app = FastAPI(title="今天去哪玩")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """首页：渲染模板，给用户可操作的聊天界面。

    可选查询参数 message、city：Starlette 会按 UTF-8 正确解码百分号编码（与 uvicorn
    访问日志里显示的原始 %XX 不同，日志一般为未解码的 path）。
    """
    hint = _llm_loading_hint()
    qp = request.query_params
    initial_message = (qp.get("message") or "").strip()
    initial_city = (qp.get("city") or "").strip()
    return templates.TemplateResponse(
        request,
        "index.html",
        context={
            "llm_loading_hint_json": json.dumps(hint, ensure_ascii=False),
            "initial_message_json": json.dumps(initial_message, ensure_ascii=False),
            "initial_city_json": json.dumps(initial_city, ensure_ascii=False),
            "llm_debug_enabled_json": json.dumps(_llm_debug_enabled()),
        },
    )


@app.post("/chat")
async def chat_endpoint(message: str = Form(...), city: str = Form("上海")) -> dict:
    """聊天接口：接收表单里的 message、city，返回 reply 与可选调试轨迹。"""
    c = (city or "上海").strip() or "上海"
    debug_on = _llm_debug_enabled()
    # chat() 内含同步 HTTP，放到线程里避免长时间阻塞事件循环（否则看起来像「卡住无响应」）
    debug_trace: list[dict] | None = [] if debug_on else None
    reply = await asyncio.to_thread(chat, message, c, debug_trace=debug_trace)
    payload = {"reply": reply}
    if debug_trace is not None:
        payload["debug_trace"] = debug_trace
    return payload


def _sse_event(event: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


@app.get("/chat/stream")
async def chat_stream(message: str, city: str = "上海") -> StreamingResponse:
    """聊天流式接口：实时推送调试轨迹，最后推送最终回复。"""
    if not _llm_debug_enabled():
        # 调试关闭时避免前端误走 SSE；直接返回错误事件
        def disabled_gen():
            yield _sse_event("error", {"message": "LLM_DEBUG 未开启，流式调试不可用"})

        return StreamingResponse(disabled_gen(), media_type="text/event-stream")

    c = (city or "上海").strip() or "上海"
    q: queue.Queue[tuple[str, dict]] = queue.Queue()
    done = threading.Event()

    def trace_hook(item: dict) -> None:
        q.put(("trace", item))

    def worker() -> None:
        try:
            reply = chat(message, c, trace_hook=trace_hook)
            q.put(("final", {"reply": reply}))
        except Exception as e:
            q.put(("error", {"message": str(e)}))
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        yield _sse_event("start", {"ok": True})
        while not (done.is_set() and q.empty()):
            try:
                event, payload = q.get(timeout=0.5)
                yield _sse_event(event, payload)
            except queue.Empty:
                yield ": ping\n\n"
        yield _sse_event("end", {"ok": True})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
