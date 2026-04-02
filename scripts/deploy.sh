#!/usr/bin/env bash
set -euo pipefail
# Railway: connect repo in dashboard, set start command:
#   uvicorn src.web.app:app --host 0.0.0.0 --port $PORT
echo "Deploy via Railway UI: New Project → Deploy from GitHub → select repo."
