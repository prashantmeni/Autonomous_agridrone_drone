# Hardware

This project assumes a Raspberry Pi companion computer paired with a Pixhawk/PX4 flight controller. The Pi runs the Python application and user-facing services; the controller runs the flight stack.

## Purpose

Use this page to confirm the hardware assumptions before you risk any real system bring-up. The project is not designed to replace the Pixhawk with Pi-only control.

## Recommended hardware

| Component | Typical role | Notes |
| --- | --- | --- |
| Raspberry Pi 4 or 5 | Companion computer | Runs Python app, API, telemetry, mission logic |
| Pixhawk-compatible controller | Flight controller | Runs PX4; handles stabilization and failsafes |
| GPS receiver | Positioning data | Used by PX4 and the Pi for health/mission context |
| USB camera | Vision input | Optional but relevant to landing and detection workflows |
| Telemetry link | MAVLink access | Usually `/dev/ttyACM0` or `udp://127.0.0.1:14540` for SITL |
| Power system | Safe flight power | Must match airframe requirements |

## Physical layout

```text
USB camera --> Raspberry Pi
Pixhawk telemetry / serial --> Raspberry Pi
Raspberry Pi <----> MAVLink <----> Pixhawk
Pixhawk --> ESCs / motors / sensors
```

## Configuration references

The project uses configuration files in `config/`:

- `config/default.yaml`
- `config/production.yaml`
- `config/development.yaml`
- `config/simulation.yaml`

These define the MAVLink connection, telemetry rate, safety thresholds, and feature flags.

## Safety note

A missing camera, missing obstacle sensor, or unavailable model should not be hidden behind a “ready” claim. Use degraded status reporting instead.

Examples from the project safety model:

- `CAMERA_UNAVAILABLE`
- `OBSTACLE_SENSOR_UNAVAILABLE`
- `MODEL_NOT_AVAILABLE`
- `DEGRADED`

## Related docs

- [raspberry-pi-setup.md](raspberry-pi-setup.md)
- [pixhawk-setup.md](pixhawk-setup.md)
- [mavlink.md](mavlink.md)
- [safety.md](safety.md)
