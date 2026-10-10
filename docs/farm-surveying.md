# Farm Surveying

Farm surveying is the project’s operational workflow for scanning a field boundary and generating a survey path based on the field polygon.

## Purpose

This feature is implemented via the survey generation API and the mapping modules in the project.

## Survey API

```bash
POST /api/survey/generate
```

This endpoint accepts a boundary GeoJSON structure and returns waypoints or generated coverage paths.

## Example request

```bash
curl -X POST http://localhost:8000/api/survey/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "boundary_geojson": {
      "type": "Polygon",
      "coordinates": [[
        [77.0, 12.0],
        [77.01, 12.0],
        [77.01, 12.01],
        [77.0, 12.01],
        [77.0, 12.0]
      ]]
    },
    "altitude_m": 15.0,
    "speed_mps": 4.0,
    "overlap": 0.2
  }'
```

## Expected response

The response includes generated waypoint information and related coverage metadata such as altitude, speed, overlap, and waypoint list.

## Project modules involved

- `src/drone/mapping/survey_planner.py`
- `src/drone/mapping/boundary_mapper.py`
- `src/drone/autonomy/geofence.py`

## Operational note

Survey generation is a planning layer, not flight certification. A generated path should still be checked against real health, geofence constraints, and flight safety rules before takeoff.

## Related docs

- [boundary-mapping.md](boundary-mapping.md)
- [mission-planning.md](mission-planning.md)
- [safety.md](safety.md)
