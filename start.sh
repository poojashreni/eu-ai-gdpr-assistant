#!/usr/bin/env bash
# Starts the API in the background, waits until it is up, then runs the UI in the foreground.
set -euo pipefail

uvicorn assistant.api:app --host 127.0.0.1 --port 8000 &

python - <<'PY'
import sys, time, urllib.request
for _ in range(120):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        sys.exit(0)
    except Exception:
        time.sleep(1)
print("API did not become healthy in 120 s", file=sys.stderr)
sys.exit(1)
PY

exec streamlit run app/streamlit_app.py \
    --server.port 7860 --server.address 0.0.0.0 --server.headless true \
    --browser.gatherUsageStats false
