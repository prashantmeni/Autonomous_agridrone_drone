from __future__ import annotations
from .flight_safety import battery_status
class BatteryMonitor:
    def __init__(self, min_pct=25, crit_pct=15): self.min, self.crit = min_pct, crit_pct
    def check(self, pct): return {"pct": pct, "state": battery_status(pct, self.min, self.crit)}
