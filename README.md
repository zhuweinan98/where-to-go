# where-to-go

今天去哪玩 — Week 1 可交互 Agent（规则 + Mock 数据），FastAPI 网页。

架构与分层说明见 [docs/技术方案.md](docs/技术方案.md)。

## 环境

- Python 3.10+（推荐用 Homebrew 的 `python3.14` 建 venv）

```bash
cd /path/to/where-to-go
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 终端对话（Agent）

```bash
python -m src.agent.bot
```

输入 `quit` 退出。

## Web 本地

```bash
uvicorn src.web.app:app --host 0.0.0.0 --port 8000
```

浏览器打开 <http://127.0.0.1:8000>。

## 测试

**自动化（改代码后建议先跑）：**

```bash
source .venv/bin/activate
pytest
```

**手动：**

1. 启动 Web（见上一节），浏览器里试：`你好`、`今天天气怎么样`、`推荐去哪里玩`；城市选「广州」再点「推荐」可测雨天室内推荐。
2. 或终端运行 `python -m src.agent.bot`，用同样句子对话；输入 `quit` 退出。

**可选（需先起 uvicorn）：**

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "message=今天天气怎么样&city=上海"
```

## Railway 部署

1. 仓库推送到 GitHub，在 Railway 里 **New Project → Deploy from GitHub** 选中本仓库。
2. **Start Command**：`uvicorn src.web.app:app --host 0.0.0.0 --port $PORT`  
   （若已识别 `Procfile` 中的 `web` 进程，可与 Dashboard 配置二选一。）
3. 生成公网域名后访问根路径即可聊天。
