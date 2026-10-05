"""Flight-health evaluation. Never reports READY without real data."""
from __future__ import annotations
import time

def evaluate(snap, cfg, pi_stats: dict) -> dict:
    checks: dict[str, str] = {}
    checks["pixhawk"] = "PASS" if snap.connected else "FAIL"
    gps_ok = (snap.gps_fix or 0) >= 3 and snap.satellites >= cfg.min_satellites
    checks["gps"] = "PASS" if gps_ok else ("FAIL" if cfg.gps_required else "WARN")
    checks["battery"] = "PASS" if snap.battery_remaining_pct is not None and snap.battery_remaining_pct >= cfg.minimum_battery_percent else "FAIL"
    checks["ekf"] = "UNKNOWN"  # extended when EKF_STATUS_REPORT parsed
    checks["link"] = "PASS" if time.time() - snap.updated_at < 5 else "FAIL"
    rc_ok, rc_age = snap.rc_status()
    if rc_ok:
        checks["rc"] = "PASS"
    elif rc_age < 0 and not snap.rc_receiver_ok:
        checks["rc"] = "UNKNOWN"  # no RC stream seen yet (MAVLink-control setups have none)
    else:
        checks["rc"] = "WARN"  # receiver seen before but signal stale
    checks["pi_cpu"] = "PASS" if pi_stats.get("cpu_pct", 0) < 90 else "FAIL"
    checks["pi_temp"] = "PASS" if pi_stats.get("temp_c", 0) < 75 else "FAIL"
    checks["pi_disk"] = "PASS" if pi_stats.get("disk_free_pct", 100) > 10 else "FAIL"
    required = [checks["pixhawk"], checks["battery"]]
    if cfg.gps_required: required.append(checks["gps"])
    status = "READY" if all(v == "PASS" for v in required) else "DEGRADED"
    if checks["pixhawk"] == "FAIL": status = "FAULT"
    return {"status": status, "checks": checks}
