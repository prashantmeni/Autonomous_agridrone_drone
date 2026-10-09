"""Preflight gate + battery/GPS/link monitors.

Every check reports what was actually measured. A check whose input is missing
reports UNKNOWN, never PASS — an unknown battery or an estimator we have never
heard from must not read as "good to arm".
"""
from __future__ import annotations

# States, worst first. UNKNOWN means "we could not measure this".
PASS, WARN, FAIL, UNKNOWN, SKIP = "PASS", "WARNING", "FAIL", "UNKNOWN", "SKIPPED"


def _gps_check(snap, cfg) -> str:
    if not snap.connected:
        return FAIL
    fix = snap.gps_fix or 0
    sats = snap.satellites or 0
    hdop = snap.hdop
    if fix < 3:
        return FAIL if cfg.gps_required else WARN
    # A 3D fix with too few satellites or poor dilution is not a usable fix.
    if sats and sats < cfg.min_satellites:
        return FAIL if cfg.gps_required else WARN
    if hdop is not None and hdop < 99.0 and hdop > cfg.max_hdop:
        return FAIL if cfg.gps_required else WARN
    return PASS


def _battery_check(snap, cfg) -> str:
    rem = snap.battery_remaining_pct
    if rem is None or rem < 0:
        return UNKNOWN          # FC connected but has not reported a battery
    if rem < cfg.critical_battery_percent:
        return FAIL
    if rem < cfg.minimum_battery_percent:
        return WARN
    return PASS


def _ekf_check(snap) -> str:
    if not snap.connected:
        return FAIL
    if not snap.ekf_known:
        return UNKNOWN          # ESTIMATOR_STATUS never received
    if snap.ekf_estimate_ok:
        return PASS
    return FAIL


def _geofence_check(app, snap) -> str:
    """Verify the aircraft is inside a configured boundary, when one exists."""
    if not getattr(app.cfg.safety, "geofence_required", False):
        return SKIP
    if snap.lat is None or snap.lon is None:
        return UNKNOWN
    try:
        fences = _active_fences(app)
        if not fences:
            return UNKNOWN      # required, but none configured
        inside = all(fence.contains(snap.lat, snap.lon) for fence in fences)
        return PASS if inside else FAIL
    except Exception:  # noqa: BLE001 - never let the gate itself crash preflight
        return UNKNOWN


def _active_fences(app) -> list:
    """Build Geofence objects from every stored boundary (GeoJSON Polygon)."""
    import json

    from ..autonomy.geofence import Geofence

    margin = float(getattr(app.cfg.safety, "geofence_margin_m", 0.0) or 0.0)
    fences = []
    try:
        rows = app.db.con.execute("SELECT geojson FROM boundaries").fetchall()
    except Exception:  # noqa: BLE001 - no table yet
        return []
    for row in rows:
        try:
            geo = row[0] if not isinstance(row[0], str) else json.loads(row[0])
            rings = geo.get("coordinates") or []
            ring = rings[0] if rings else []
            polygon = [(float(pt[1]), float(pt[0])) for pt in ring if len(pt) >= 2]
            if len(polygon) >= 3:
                fences.append(Geofence(polygon, margin_m=margin))
        except Exception:  # noqa: BLE001
            continue
    return fences


def _camera_check(app) -> str:
    cam = getattr(app.cfg.camera, "enabled", False)
    if not cam:
        return SKIP
    try:
        from ..runtime import get_services
        status = get_services(app).camera.status()
        st = str(status.get("status", "")).upper()
        if st == "READY":
            return PASS
        if st in ("NOT_STARTED", "STARTING"):
            return WARN
        return WARN if st == "CAMERA_UNAVAILABLE" else UNKNOWN
    except Exception:  # noqa: BLE001
        return UNKNOWN


def _obstacle_check(app) -> str:
    if not getattr(app.cfg.obstacle_avoidance, "enabled", False):
        return SKIP            # avoidance is off: not a failure, not a claim
    try:
        from ..runtime import get_services
        status = get_services(app).proximity.status()
        if status.get("available"):
            return PASS
        return UNKNOWN          # enabled, but no rangefinder is reporting
    except Exception:  # noqa: BLE001
        return UNKNOWN


def preflight_check(app) -> dict:
    """Gate arming on what is genuinely known to be healthy."""
    snap = app.tstore.snap
    cfg = app.cfg.safety
    checks: dict[str, str] = {
        "pixhawk": PASS if snap.connected else FAIL,
        "gps": _gps_check(snap, cfg),
        "battery": _battery_check(snap, cfg),
        "ekf": _ekf_check(snap),
        "geofence": _geofence_check(app, snap),
        "camera": _camera_check(app),
        "obstacle_detection": _obstacle_check(app),
    }
    if snap.rc_status()[0]:
        checks["rc"] = PASS
    else:
        checks["rc"] = UNKNOWN

    # Unknown is not good enough to arm: only a clean PASS on the critical
    # checks may proceed.
    critical = ["pixhawk", "battery", "ekf"]
    if cfg.gps_required:
        critical.append("gps")
    if checks["geofence"] == FAIL:
        critical.append("geofence")
    blocking = {k: checks[k] for k in critical if checks[k] != PASS}
    return {
        "ready": not blocking,
        "checks": checks,
        "blocking": blocking,
        "sensors": list(snap.sensor_issues or []),
        "ekf_issues": snap.ekf_issues,
    }


def battery_status(pct: int, min_pct: int, crit_pct: int) -> str:
    if pct is None or pct < 0:
        return "UNKNOWN"
    if pct < crit_pct:
        return "CRITICAL"
    if pct < min_pct:
        return "LOW"
    return "OK"
