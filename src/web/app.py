"""FastAPI 应用：HTTP 与页面。

职责：
  - GET / ：返回聊天页（Jinja2 模板 index.html）
  - POST /chat ：解析表单，调用 agent.chat，返回 JSON { "reply": "..." }
作用：把浏览器/接口与 bot 里的 Agent 逻辑连接起来。
"""

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
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
        },
    )


@app.post("/chat")
async def chat_endpoint(message: str = Form(...), city: str = Form("上海")) -> dict:
    """聊天接口：接收表单里的 message、city，交给 chat()，JSON 返回 reply。"""
    c = (city or "上海").strip() or "上海"
    # chat() 内含同步 HTTP，放到线程里避免长时间阻塞事件循环（否则看起来像「卡住无响应」）
    reply = await asyncio.to_thread(chat, message, c)
    return {"reply": reply}
