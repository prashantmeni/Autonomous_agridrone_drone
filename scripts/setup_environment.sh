#!/usr/bin/env bash
set -e
python3 -m venv .venv || true
. .venv/bin/activate 2>/dev/null || true
pip install -e ".[dev]"
echo "env ready"
