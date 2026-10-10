# Boundary Mapping

This page explains how the project stores and validates field boundaries for missions and survey planning.

## Purpose

Boundaries are represented in GeoJSON. The application validates the input and stores it in the database for later mission planning and surveying.

## API

```bash
POST /api/boundaries
POST /api/boundaries/import-kml
```

## Example boundary creation

```bash
curl -X POST http://localhost:8000/api/boundaries \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "field-a",
    "geojson": {
      "type": "Polygon",
      "coordinates": [[
        [77.0, 12.0],
        [77.01, 12.0],
        [77.01, 12.01],
        [77.0, 12.01],
        [77.0, 12.0]
      ]]
    }
  }'
```

## Validation behavior

The app calls `validate_geojson(...)` before accepting a boundary. Invalid geometry is rejected with a validation error.

## File samples

Example boundary and mission files exist in:

- `data/boundaries/example.geojson`
- `data/missions/example.yaml`

## KML import

The project supports KML import via:

```bash
curl -X POST http://localhost:8000/api/boundaries/import-kml \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "field-kml",
    "kml": "<kml>...</kml>"
  }'
```

## Operational note

Boundary data is a planning artifact. It should be validated with actual geofence logic and safety restrictions before autonomous execution.

## Related docs

- [farm-surveying.md](farm-surveying.md)
- [mission-planning.md](mission-planning.md)
- [safety.md](safety.md)
