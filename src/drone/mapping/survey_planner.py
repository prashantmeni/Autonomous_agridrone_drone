"""Boundary + geospatial + survey coverage planner."""
from __future__ import annotations
import math
from .geospatial import area_m2, perimeter_m

def validate_geojson(gj: dict) -> tuple[bool, list[str]]:
    errs = []
    if gj.get("type") != "Polygon": errs.append("only Polygon supported")
    try:
        ring = gj["coordinates"][0]
        if len(ring) < 4: errs.append("polygon ring needs >=4 positions")
        for p in ring:
            if len(p) != 2: errs.append(f"bad position {p}")
    except Exception as e: errs.append(str(e))
    return (not errs, errs)

class BoundaryMapper:
    @staticmethod
    def from_points(points: list[tuple[float,float]]) -> dict:
        ring = [[lon, lat] for lat, lon in points]
        if ring and ring[0] != ring[-1]: ring.append(ring[0])
        return {"type": "Polygon", "coordinates": [ring]}

    @staticmethod
    def stats(gj: dict) -> dict:
        ring = gj["coordinates"][0]
        poly = [(p[1], p[0]) for p in ring]
        return {"area_m2": area_m2(poly), "perimeter_m": perimeter_m(poly)}

class SurveyPlanner:
    """Boustrophedon coverage over bounding box of polygon (safe, geofence-clipped by caller)."""
    def __init__(self, altitude_m=20.0, speed_mps=5.0, side_overlap=0.7, fov_deg=78.0):
        self.alt = altitude_m; self.speed = speed_mps
        self.overlap = side_overlap; self.fov = fov_deg

    def footprint_m(self) -> float:
        return 2 * self.alt * math.tan(math.radians(self.fov/2)) * (1 - self.overlap)

    def generate(self, boundary_gj: dict) -> dict:
        ring = boundary_gj["coordinates"][0]
        lats = [p[1] for p in ring]; lons = [p[0] for p in ring]
        min_lat, max_lat, min_lon, max_lon = min(lats), max(lats), min(lons), max(lons)
        step_deg = max(self.footprint_m() / 111320.0, 1e-6)
        lines = []
        lat = min_lat
        flip = False
        nlon = max(int((max_lon - min_lon) / max(step_deg, 1e-9)), 1)
        while lat <= max_lat:
            line = [(lat, min_lon), (lat, max_lon)]
            lines.append(line[::-1] if flip else line)
            flip = not flip
            lat += step_deg
        wps = [{"lat": la, "lon": lo, "alt": self.alt} for line in lines for la, lo in line]
        from ..navigation.coordinate_transform import haversine_m
        dist = sum(haversine_m(wps[i]["lat"], wps[i]["lon"], wps[i+1]["lat"], wps[i+1]["lon"]) for i in range(len(wps)-1)) if len(wps) > 1 else 0
        return {"waypoints": wps, "lines": lines, "distance_m": dist,
                "eta_s": dist / max(self.speed, 0.1), "footprint_m": self.footprint_m()}
