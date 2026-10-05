"""Geofence polygon check (ray casting). Pure logic, fully tested."""
from __future__ import annotations

def point_in_polygon(lat: float, lon: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon)
    if n < 3: return False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]; yj, xj = polygon[j]
        if ((xi > lon) != (xj > lon)) and (lat < (yj - yi) * (lon - xi) / ((xj - xi) or 1e-12) + yi):
            inside = not inside
        j = i
    return inside

class Geofence:
    def __init__(self, polygon: list[tuple[float, float]] | None = None, margin_m: float = 5.0):
        self.polygon = polygon or []
        self.margin_m = margin_m

    def is_loaded(self) -> bool: return len(self.polygon) >= 3

    def contains(self, lat: float, lon: float) -> bool:
        if not self.is_loaded(): return True  # no geofence = permit (logged by caller)
        return point_in_polygon(lat, lon, self.polygon)

    def validate_mission(self, waypoints: list[dict]) -> tuple[bool, list[int]]:
        bad = [i for i, w in enumerate(waypoints) if not self.contains(w["lat"], w["lon"])]
        return (len(bad) == 0, bad)
