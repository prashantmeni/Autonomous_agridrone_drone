# Installation

This page describes how to install the project and run the Python service locally.

## Purpose

Use these steps before you start the app, bring up the dashboard, or test MAVLink connectivity.

## Prerequisites

- Python 3.11+
- Git
- Optional: Node.js and npm for the dashboard
- Optional: Pixhawk hardware or PX4 SITL

## Install steps

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

The package is defined in `pyproject.toml` and includes the app dependencies and dev tooling.

## Project dependencies

The Python dependencies in `pyproject.toml` include:

- `fastapi`
- `uvicorn[standard]`
- `pydantic` and `pydantic-settings`
- `pyyaml`
- `pymavlink`
- `numpy`
- `shapely`
- `pyserial`
- `psutil`
- `websockets`
- `opencv-python-headless`

## Dashboard install

For the frontend dashboard in `dashboard/`:

```bash
cd dashboard
npm install
npm run dev
```

## Environment setup

```bash
cp .env.example .env
```

Then review the values in `.env`, especially:

```dotenv
MAVLINK_CONNECTION=/dev/ttyACM0
MAVLINK_BAUD=115200
API_PORT=8000
DRONE_CONFIG=config/production.yaml
```

## Validate install

```bash
python -m drone.main --config config/development.yaml
```

This should start the app with the development configuration. If you are in a hardware-less environment, use SITL instead.

## Related docs

- [raspberry-pi-setup.md](raspberry-pi-setup.md)
- [mavlink.md](mavlink.md)
- [troubleshooting.md](troubleshooting.md)
