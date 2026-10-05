# Troubleshooting

This page focuses on the practical checks that match the project’s scripts and runtime structure.

## Purpose

Use these checks when the app, MAVLink link, or dashboard does not respond as expected.

## Basic commands

### MAVLink health

```bash
python3 scripts/check_mavlink.py
```

### API health

```bash
curl -sf http://localhost:8000/api/health
```

### Serial device check

```bash
ls -l /dev/ttyACM*
dmesg | grep ttyACM
```

### Local service start

```bash
python -m drone.main --config config/development.yaml
```

## Common issues

### No heartbeat on MAVLink

Possible causes:

- wrong serial device path
- wrong baud rate
- PX4 not running or not connected
- SITL not running on `udp://127.0.0.1:14540`

### API not responding

Check:

- Python process is running
- port 8000 is not blocked
- environment file is valid

### Dashboard is blank or stale

Check:

- backend is reachable
- websocket endpoint `ws://localhost:8000/ws/telemetry` is available
- `npm run dev` is running in `dashboard/`

### Mission fails to start

Check:

- the mission ID exists
- health checks are passing
- no critical safety gate is active
- the drone is not in a blocked state

## Safety reminder

Do not continue autonomous operations on uncertain health. If the system cannot verify the link or safety state, stop and investigate.

## Related docs

- [mavlink.md](mavlink.md)
- [safety.md](safety.md)
- [deployment.md](deployment.md)
