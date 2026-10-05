# Plant Disease Detection

This project includes logic for crop disease detection, but it is model-dependent and should only be treated as active when the model and sensor input are available.

## Purpose

The detection layer can classify crops and diseases and store detections with coordinates for later review.

## Configuration

```yaml
disease_detection:
  enabled: false
  model_path: ""
  confidence_threshold: 0.6
  inference_interval_s: 2.0
```

## API endpoint

```bash
GET /api/detections
```

It returns rows such as:

```json
[
  {
    "crop": "tomato",
    "disease": "leaf_spot",
    "confidence": 0.89,
    "latitude": 12.345,
    "longitude": 98.765,
    "timestamp": "2026-10-02T12:00:00"
  }
]
```

## Failure behavior

If no model is loaded or no input is available, the correct behavior is not to pretend the system is ready. The project expects:

- `MODEL_NOT_AVAILABLE`
- `DEGRADED`
- user-visible health warnings

## Related docs

- [telemetry.md](telemetry.md)
- [safety.md](safety.md)
- [deployment.md](deployment.md)
