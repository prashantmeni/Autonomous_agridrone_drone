"""Failsafe evaluator: battery / link / GPS -> recommended action. PX4 remains authoritative."""
from __future__ import annotations

def evaluate_failsafe(snap, cfg) -> dict:
    actions: list[str] = []
    rem = snap.battery_remaining_pct if snap.battery_remaining_pct is not None else -1
    if 0 <= rem < cfg.critical_battery_percent: actions.append("EMERGENCY_LAND")
    elif 0 <= rem < cfg.minimum_battery_percent: actions.append("RETURN_HOME")
    if cfg.gps_required and (snap.gps_fix or 0) < 3: actions.append("HOLD_POSITION")
    if not snap.connected: actions.append("WAIT_RECONNECT")
    severity = "OK" if not actions else ("CRITICAL" if "EMERGENCY_LAND" in actions else "WARNING")
    return {"severity": severity, "actions": actions}
