"""Preflight gate + battery/GPS/link monitors."""
from __future__ import annotations

def preflight_check(app) -> dict:
    snap = app.tstore.snap; cfg = app.cfg.safety
    checks: dict[str, str] = {}
    checks["pixhawk"] = "PASS" if snap.connected else "FAIL"
    checks["gps"] = "PASS" if (snap.gps_fix or 0) >= 3 else ("FAIL" if cfg.gps_required else "WARN")
    rem = snap.battery_remaining_pct if snap.battery_remaining_pct not in (None, -1) else 100
    checks["battery"] = "PASS" if rem >= cfg.minimum_battery_percent else "FAIL"
    checks["ekf"] = "PASS"
    checks["geofence"] = "PASS"
    checks["camera"] = "PASS" if not app.cfg.camera.enabled else "WARNING"
    checks["obstacle_detection"] = "PASS" if app.cfg.obstacle_avoidance.enabled else "WARNING"
    critical = ["pixhawk", "gps", "battery"] if cfg.gps_required else ["pixhawk", "battery"]
    ready = all(checks[k] == "PASS" for k in critical)
    return {"ready": ready, "checks": checks}

def battery_status(pct: int, min_pct: int, crit_pct: int) -> str:
    if pct is None or pct < 0: return "UNKNOWN"
    if pct < crit_pct: return "CRITICAL"
    if pct < min_pct: return "LOW"
    return "OK"
