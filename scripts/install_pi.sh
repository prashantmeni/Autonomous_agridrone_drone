#!/usr/bin/env bash
set -e
echo "== AgriDrone Pi install =="
python3 --version
pip install -e ".[dev]" || pip install -e .
cp -n .env.example .env || true
python3 scripts/check_mavlink.py || true
echo "Install done. Edit .env then: sudo systemctl enable --now drone.service"
