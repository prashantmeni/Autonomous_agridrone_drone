# Obstacle Avoidance

Obstacle avoidance is a configurable feature in the project. It should be treated as a real implementation area, not as a guarantee of autonomous safety without sensor validation.

## Purpose

The project can evaluate obstacle distance and modify flight behavior using thresholds from configuration.

## Configuration

```yaml
obstacle_avoidance:
  enabled: false
  warning_distance_m: 5.0
  critical_distance_m: 2.0
  emergency_distance_m: 1.0
```

## Degraded and unavailable behavior

If no obstacle sensor is connected or the sensor is unavailable, the system must report a degraded or unavailable state and avoid claiming safe autonomous operation.

Examples:

- `OBSTACLE_SENSOR_UNAVAILABLE`
- `DEGRADED`

## Safety implications

When obstacle distance reaches warning thresholds, flight logic should slow down or alter trajectory. At critical thresholds, safety logic should favor abort or avoidance.

## Related docs

- [safety.md](safety.md)
- [flight-modes.md](flight-modes.md)
- [precision-landing.md](precision-landing.md)
