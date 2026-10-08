#!/usr/bin/env bash
set -e
python app/fetch_data.py                      # fetch data/ from the private dataset repo if it is not there yet
cd app && exec python -m uvicorn server:app --host 0.0.0.0 --port "${PORT:-7860}"
