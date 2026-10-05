# MAVLink

The project communicates with a Pixhawk/PX4 flight controller through MAVLink. The connection is configured in `config/*.yaml` and can be overridden by environment variables.

## Purpose

This document is the authoritative reference for the MAVLink transport used by the project.

## Actual settings in the repo

The default and environment-specific config files show the following patterns:

```yaml
# config/default.yaml
mavlink:
  connection: ${MAVLINK_CONNECTION:/dev/ttyACM0}
  baud: 115200
  heartbeat_timeout_s: 5.0
  reconnect_delay_s: 2.0
```

```yaml
# config/simulation.yaml
mavlink:
  connection: udp://127.0.0.1:14540
  baud: 115200
```

## Supported connection types

The repo explicitly expects a local serial device in production and a UDP SITL target in simulation:

- Serial: `/dev/ttyACM0`
- SITL UDP: `udp://127.0.0.1:14540`

## Environment variables

```dotenv
MAVLINK_CONNECTION=/dev/ttyACM0
MAVLINK_BAUD=115200
```

Or for simulation:

```bash
export MAVLINK_CONNECTION=udp://127.0.0.1:14540
export MAVLINK_BAUD=115200
```

## Runtime verification

```bash
python3 scripts/check_mavlink.py
```

Expected behavior:

- `FOUND /dev/ttyACM0` or `FOUND udp://127.0.0.1:14540`
- `HEARTBEAT OK` when the link is healthy
- non-zero exit if the heartbeat does not arrive in time

## Safety note

If MAVLink heartbeat fails, the app cannot confidently treat the drone as operational. Mission execution and takeoff flow should be treated as blocked unless the controller is actually reachable.

## Related docs

- [pixhawk-setup.md](pixhawk-setup.md)
- [flight-modes.md](flight-modes.md)
- [telemetry.md](telemetry.md)
