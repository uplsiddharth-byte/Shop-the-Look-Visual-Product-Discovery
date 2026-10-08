#!/usr/bin/env bash
# Start the Shop the Look server. HOST/PORT/DEVICE are optional (DEVICE=cpu|cuda|mps; default: auto-detect).
set -euo pipefail
cd "$(dirname "$0")/app"
PY="${PYTHON:-../.venv/bin/python}"; [ -x "$PY" ] || PY="$(command -v python3)"
exec "$PY" -m uvicorn server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
