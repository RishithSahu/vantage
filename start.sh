#!/bin/sh
# Starts VehicleGPT (macOS / Linux). Uses ./.venv when present.
cd "$(dirname "$0")" || exit 1
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python3"; fi
exec "$PY" vehiclegpt.py run "$@"
