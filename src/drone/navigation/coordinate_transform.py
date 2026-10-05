"""Haversine + equirectangular helpers."""
from __future__ import annotations
import math
R = 6371000.0

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def offset_latlon(lat, lon, dx_m, dy_m) -> tuple[float, float]:
    return lat + (dy_m / R) * 180 / math.pi, lon + (dx_m / R) * 180 / math.pi / max(0.2, math.cos(math.radians(lat)))

def polygon_area_m2(poly: list[tuple[float,float]]) -> float:
    if len(poly) < 3: return 0.0
    lat0 = sum(p[0] for p in poly)/len(poly)
    pts = [((lon-math.radians(0))*0, 0) for _, lon in poly]  # placeholder avoided below
    # equirectangular projection around mean latitude
    import math as m
    xs = [(lon - poly[0][1]) * 111320 * m.cos(m.radians(lat0)) for _, lon in poly]
    ys = [(lat - poly[0][0]) * 110540 for lat, _ in poly]
    a = 0.0
    for i in range(len(xs)):
        x1, y1 = xs[i], ys[i]; x2, y2 = xs[(i+1) % len(xs)], ys[(i+1) % len(ys)]
        a += x1*y2 - x2*y1
    return abs(a)/2
