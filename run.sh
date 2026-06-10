#!/bin/bash
# ── Quick start script for local development (no Docker) ─────────────────────
#
# Prerequisites (one-time):
#   pip install -r requirements.txt
#
# Step 1 – Train a model:
#   python -m service.train_service \
#       --data-path ../K4/syslog_dev \
#       --model-version default \
#       --embedder all-MiniLM-L6-v2 \
#       --detector gmm --k 5
#
# Step 2 – Run the service:
#   python -c "from service.api import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"
#
# Or with hot-reload:
#   uvicorn service.api:app --reload --host 0.0.0.0 --port 8000

set -e
export PYTHONPATH="$(dirname "$0"):${PYTHONPATH}"

echo "=== K4 Service ==="
echo "Available commands:"
echo "  ./run.sh train    – train a model (edit DATA_PATH below first)"
echo "  ./run.sh serve    – start the FastAPI server (dev mode)"
echo "  ./run.sh test     – run pytest"
echo "  ./run.sh curl     – smoke test the running service"
