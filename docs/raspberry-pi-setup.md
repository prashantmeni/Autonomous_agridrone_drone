# Raspberry Pi Setup

This page covers the Raspberry Pi companion-computer setup for the project.

## Purpose

The Pi is responsible for running the Python application, exposing the API, recording telemetry, and coordinating mission logic. The project expects it to be connected to a Pixhawk/PX4 controller over MAVLink.

## Prerequisites

- Raspberry Pi OS (64-bit recommended)
- Network access for package installation
- Python 3.11+
- Git
- Access to the repo

## Step-by-step

### 1. Update the Pi

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git
```

### 2. Clone and install the project

```bash
git clone https://github.com/prashantmeni/Autonomous_agridrone_drone.git
cd Autonomous_agridrone_drone
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. Prepare environment variables

```bash
cp .env.example .env
nano .env
```

Recommended settings for a physical Pi:

```dotenv
MAVLINK_CONNECTION=/dev/ttyACM0
MAVLINK_BAUD=115200
DRONE_CONFIG=config/production.yaml
API_HOST=0.0.0.0
API_PORT=8000
```

### 4. Verify MAVLink path

```bash
python3 scripts/check_mavlink.py
```

This script checks whether the configured local device or UDP/TCP target exists and whether a MAVLink heartbeat is seen.

### 5. Run the app

```bash
python -m drone.main --config config/production.yaml
```

### 6. Start via systemd

The repo includes service files in `systemd/`:

```bash
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now drone.service
```

## Safety note

The Pi should not be treated as a substitute for PX4 flight safety. It is a monitoring and orchestration layer.

## Related docs

- [installation.md](installation.md)
- [pixhawk-setup.md](pixhawk-setup.md)
- [mavlink.md](mavlink.md)
