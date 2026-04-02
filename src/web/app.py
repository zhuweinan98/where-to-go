"""FastAPI 应用：HTTP 与页面。

职责：
  - GET / ：返回聊天页（Jinja2 模板 index.html）
  - POST /chat ：解析表单，调用 agent.chat，返回 JSON { "reply": "..." }
作用：把浏览器/接口与 bot 里的 Agent 逻辑连接起来。
"""

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.agent.bot import chat

_ROOT = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(_ROOT / "templates"))

# FastAPI 应用实例：uvicorn 加载的就是这个 app
app = FastAPI(title="今天去哪玩")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """首页：渲染模板，给用户可操作的聊天界面。"""
    # Starlette 1.x：第一个参数必须是 request，第二个是模板文件名
    return templates.TemplateResponse(request, "index.html")


@app.post("/chat")
async def chat_endpoint(message: str = Form(...), city: str = Form("上海")) -> dict:
    """聊天接口：接收表单里的 message、city，交给 chat()，JSON 返回 reply。"""
    c = (city or "上海").strip() or "上海"
    reply = chat(message, city=c)
    return {"reply": reply}
