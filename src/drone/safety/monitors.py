from __future__ import annotations
class GpsMonitor:
    def __init__(self, min_sats=8, max_hdop=1.5): self.min_sats, self.max_hdop = min_sats, max_hdop
    def check(self, snap):
        ok = (snap.gps_fix or 0) >= 3 and (snap.satellites or 0) >= self.min_sats
        state = "OK" if ok else ("LOW_GPS" if snap.connected else "NO_GPS")
        return {"state": state, "fix": snap.gps_fix, "sats": snap.satellites, "hdop": snap.hdop}
class LinkMonitor:
    def check(self, snap):
        import time
        age = time.time() - snap.updated_at
        return {"state": "OK" if snap.connected and age < 5 else "MAVLINK_RECONNECTING", "age_s": age}
class CpuMonitor:
    def check(self):
        from ..telemetry.metrics import pi_stats
        s = pi_stats()
        issues = []
        if s["cpu_pct"] > 90: issues.append("CPU_OVERLOAD")
        if s["temp_c"] > 75: issues.append("PI_OVERHEAT")
        if s["disk_free_pct"] < 10: issues.append("DISK_LOW")
        return {"stats": s, "issues": issues}
class TemperatureMonitor(CpuMonitor): pass
class EmergencyHandler:
    def __init__(self, app): self.app = app
    async def emergency(self, reason: str):
        from ..core.state import FlightState
        self.app.fsm.force(FlightState.EMERGENCY, reason)
        return {"emergency": True, "reason": reason}
