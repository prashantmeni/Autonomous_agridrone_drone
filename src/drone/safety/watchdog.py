"""Watchdog: detects crash/MAVLink loss/overload. Never commands aircraft; PX4 failsafes authoritative."""
from __future__ import annotations
import time, logging
log = logging.getLogger("drone.watchdog")
def check(app) -> dict:
    from .telemetry.metrics import pi_stats
    issues = []
    s = app.tstore.snap
    if not s.connected: issues.append("MAVLINK_LOSS")
    if time.time() - s.updated_at > 10: issues.append("TELEMETRY_STALE")
    p = pi_stats()
    if p["cpu_pct"] > 95: issues.append("CPU_OVERLOAD")
    if p["temp_c"] > 80: issues.append("PI_OVERHEAT")
    if p["disk_free_pct"] < 5: issues.append("DISK_EXHAUSTION")
    return {"ok": not issues, "issues": issues, "pi": p}
