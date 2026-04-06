# where-to-go

今天去哪玩 — Week 1 可交互 Agent（规则 + Mock 数据），FastAPI 网页。

架构、Mock 与 **本地 Ollama** 的完整说明见 [docs/技术方案.md](docs/技术方案.md)（伙伴接手建议先读该文档）。

## 环境

- Python 3.10+（推荐用 Homebrew 的 `python3.14` 建 venv）

```bash
cd /path/to/where-to-go
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 城市列表（`config/cities.json`）

**一份 JSON** 同时驱动：用户话里匹配城名、和风静态 LocationID（可选填）、无 API 时的 Mock 天气回退。新增城市只改此文件；`src/data/mock.py` 里景点的 `city` 必须出现在列表中。

## 本地 Ollama（可选）

1. 安装并打开 [Ollama](https://ollama.com/)，拉模型：`ollama pull qwen3:4b`（模型名需与 `OLLAMA_MODEL` 一致，可自选其它已拉取的模型）。
2. **`.env` 已加入 `.gitignore`**，每位开发者本地自建：复制 `cp .env.example .env`，按文件内说明填写 `LLM_MODE=ollama` 与 `OLLAMA_MODEL`。
3. **单测 / CI**：使用 `LLM_MODE=off` 或不设置，避免依赖本机 Ollama。
4. **云端大模型（Railway 等）**：`LLM_MODE=openai`，并配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`（可选）、`OPENAI_MODEL`（可选）；与 OpenAI 兼容的国内 API 同样可用。
5. 细节见 [docs/技术方案.md](docs/技术方案.md) 中的「大模型」一节。

## 和风天气（可选）

在 `.env` 中配置 `QWEATHER_HOST` 与 `QWEATHER_KEY`（控制台 **API KEY** 凭据）后，天气与「推荐」分支会按实况判断晴雨；未配置或请求失败时使用 Mock（失败时**不会**在界面报错，所以容易误以为没调 API）。可在 `.env` 加 **`QWEATHER_DEBUG=1`**，重启后在 **uvicorn 终端** 会打印以 `[qweather]` 开头的行（直接 stdout，不依赖 logging 配置）。说明见 [docs/技术方案.md](docs/技术方案.md) 中的「和风天气」一节。

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
4. **Variables（环境变量）**：在服务的 **Variables** 里添加（**不要**把密钥写进仓库；Railway 里填的值等同于本机 `.env`）：

   | 变量 | 说明 |
   |------|------|
   | `LLM_MODE` | 线上建议 `openai`（容器内无法跑 Ollama）。仅用规则 + Mock 时设 `off`。 |
   | `OPENAI_API_KEY` | 必填（当 `openai` 时）：DeepSeek / OpenAI 等平台的密钥。 |
   | `OPENAI_BASE_URL` | DeepSeek 示例：`https://api.deepseek.com`；官方 OpenAI 可省略（默认 `https://api.openai.com`）。 |
   | `OPENAI_MODEL` | DeepSeek 示例：`deepseek-chat`；官方示例：`gpt-4o-mini`。 |
   | `OPENAI_HTTP_TIMEOUT` | 可选，秒，默认 `120`。 |
   | `LLM_DEBUG` | 可选：设 `1` 可在 **Deploy Logs** 里看到 `[llm]` 调试行。 |
   | `QWEATHER_HOST` / `QWEATHER_KEY` | 可选：与本地相同，不配则天气回退 Mock。 |
   | `QWEATHER_DEBUG` | 可选：设 `1` 在日志里打印 `[qweather]`。 |

   Railway 会自动注入 **`PORT`**，无需手写；也勿在变量里提交 `.env` 文件内容到 Git。
