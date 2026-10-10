# Telemetry

Telemetry is a core part of the project. The drone app records and broadcasts telemetry while it monitors the MAVLink stream and other runtime health data.

## Purpose

This page documents the actual telemetry flow used by the project.

## Components

The project has telemetry components in `src/drone/telemetry/`:

- `logger.py`
- `metrics.py`
- `publisher.py`
- `recorder.py`

The app also has a MAVLink telemetry parser in `src/drone/mavlink/telemetry.py`.

## API endpoints

```bash
GET /api/drone/status
GET /api/drone/telemetry
GET /api/health
GET /api/logs
```

## Websocket stream

```bash
ws://localhost:8000/ws/telemetry
```

The websocket server is defined in `src/drone/api/websocket.py`.

## Data flow

```text
MAVLink messages -> TelemetryStore -> recorder + websocket publisher -> dashboard + API
```

## Example curl

```bash
curl http://localhost:8000/api/drone/telemetry
curl http://localhost:8000/api/health
```

## Expected output

The API returns JSON telemetry or health data. The health endpoint calls `evaluate()` from `src/drone/mavlink/health.py` and merges the results with the snapshot from `TelemetryStore`.

## Safety note

Telemetry is useful for diagnosis, but it is not the same as flight readiness. The project requires actual health checks and controller connectivity before it should treat the system as operational.

## Related docs

- [mavlink.md](mavlink.md)
- [flight-modes.md](flight-modes.md)
- [troubleshooting.md](troubleshooting.md)
