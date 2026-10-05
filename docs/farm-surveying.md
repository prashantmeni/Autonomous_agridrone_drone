# Mission Planning

Mission planning is implemented in the Python app and exposed through the FastAPI endpoints in `src/drone/api/server.py`.

## Purpose

This document describes how mission data is created, validated, and started in practice.

## API endpoints

```bash
GET /api/missions
POST /api/missions
GET /api/missions/{mid}
POST /api/missions/{mid}/start
POST /api/missions/{mid}/abort
```

## Example mission creation

```bash
curl -X POST http://localhost:8000/api/missions \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "field-scan-01",
    "waypoints": [
      {"lat": 12.345, "lon": 98.765, "alt": 15.0},
      {"lat": 12.351, "lon": 98.770, "alt": 15.0}
    ],
    "takeoff": {"altitude_m": 15.0},
    "survey": {"speed_mps": 4.0}
  }'
```

## Expected response

```json
{"id": "<uuid>"}
```

## Mission validation

The configuration file `config/default.yaml` sets mission constraints such as:

```yaml
mission:
  max_altitude_m: 30.0
  max_speed_mps: 8.0
  default_takeoff_alt_m: 15.0
```

The app uses these values as operational limits during mission behavior and validation. These are hard constraints for the repository, not marketing placeholders.

## Mission start

```bash
curl -X POST http://localhost:8000/api/missions/<id>/start
```

If the mission is not found or the drone is blocked by a safety or health gate, the API responds with an error.

## Failure behavior

Mission execution should fail in a controlled way when:

- the drone is not healthy
- a preflight check fails
- MAVLink connection is lost
- the mission is invalid
- the safety rules reject the action

## Related docs

- [flight-modes.md](flight-modes.md)
- [boundary-mapping.md](boundary-mapping.md)
- [farm-surveying.md](farm-surveying.md)
- [safety.md](safety.md)
