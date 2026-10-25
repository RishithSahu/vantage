#!/bin/sh
# One-time setup (macOS / Linux): private environment + requirements, then the self-check.
cd "$(dirname "$0")" || exit 1
python3 -m venv .venv && .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -r requirements.txt || { echo "install failed - see TROUBLESHOOTING.md"; exit 1; }
.venv/bin/python vehiclegpt.py doctor
echo "Start with ./start.sh"
