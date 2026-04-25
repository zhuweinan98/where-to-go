"""单测环境：先于 src.agent.bot 的 load_dotenv 生效，强制走规则引擎。"""

import os

os.environ["LLM_MODE"] = "off"
# 避免本机 .env 经 load_dotenv 写入后，实况天气与单测期望的雨天室内分支不一致
# （dotenv 默认不覆盖已在环境中的键，故置空字符串以压住 .env 里的和风配置）
os.environ["QWEATHER_HOST"] = ""
os.environ["QWEATHER_KEY"] = ""
os.environ["QWEATHER_BEARER_TOKEN"] = ""
os.environ.pop("QWEATHER_DEBUG", None)
