"""单测环境：先于 src.agent.bot 的 load_dotenv 生效，强制走规则引擎。"""

import os

os.environ["LLM_MODE"] = "off"
