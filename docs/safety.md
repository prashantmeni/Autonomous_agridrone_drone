# Safety

Safety is a strict design requirement in this repository. The rule is simple and explicit:

`PX4 safety > Pi autonomy > UI state`

## Purpose

This page defines how the project treats health, degraded operation, and flight readiness.

## Safety hierarchy

| Priority | Source | Meaning |
| --- | --- | --- |
| 1 | PX4 controller | Real flight safety and failsafes |
| 2 | Pi app | Mission and health rules |
| 3 | Dashboard/UI | Monitoring and control surface |

## Required behavior

- A real health check is required before claiming flight readiness.
- A missing camera, missing obstacle sensor, or missing model should be reported as degraded or unavailable.
- The dashboard cannot override PX4 flight safety.
- If the MAVLink heartbeat is absent, treat the system as not operational.

## Example degraded states

- `CAMERA_UNAVAILABLE`
- `OBSTACLE_SENSOR_UNAVAILABLE`
- `MODEL_NOT_AVAILABLE`
- `DEGRADED`

## Operational principle

The project should prefer safer abort or hold states over continuing autonomous behavior when there is uncertainty.

## Related docs

- [flight-modes.md](flight-modes.md)
- [precision-landing.md](precision-landing.md)
- [obstacle-avoidance.md](obstacle-avoidance.md)
- [plant-disease-detection.md](plant-disease-detection.md)
