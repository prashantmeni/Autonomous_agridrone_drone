# Precision Landing

Precision landing is a feature area in the project that depends on vision and landing behavior. It is enabled in the configuration and should be treated as a real capability requiring validation.

## Purpose

The system can support precision landing by detecting a fiducial or landing marker and adjusting the descent path before touchdown.

## Configuration

```yaml
precision_landing:
  enabled: false
  marker_type: aruco
  max_descent_speed_mps: 0.8
  marker_lost_timeout_s: 2.0
```

In simulation, the config may enable it:

```yaml
precision_landing:
  enabled: true
```

## Operational behavior

The app expects a landing marker pipeline to provide visual guidance to reduce touchdown error. If the marker is lost, the timeout logic should trigger a safe fallback rather than silently continuing.

## Failure and degraded behavior

If the marker is not seen or the camera is unavailable:

- the landing path must degrade gracefully
- the project should report degraded or unavailable state
- a fallback to a safer landing or abort may be required

This is consistent with the repository’s safety-first policy.

## Related docs

- [obstacle-avoidance.md](obstacle-avoidance.md)
- [safety.md](safety.md)
- [flight-modes.md](flight-modes.md)
