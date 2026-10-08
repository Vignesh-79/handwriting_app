#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -d .venv ] || (python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv)
source .venv/bin/activate
pip install -r requirements.txt
python app.py
