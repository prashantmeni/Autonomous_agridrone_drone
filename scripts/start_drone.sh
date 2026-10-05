#!/usr/bin/env bash
set -e
CONFIG=${1:-config/production.yaml}
exec python -m drone.main --config "$CONFIG"
