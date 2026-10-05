#!/usr/bin/env bash
set -e
echo "Start PX4 SITL first (e.g. make px4_sitl gz_x500), then:"
echo "  python -m drone.main --config config/simulation.yaml"
exec python -m drone.main --config config/simulation.yaml
