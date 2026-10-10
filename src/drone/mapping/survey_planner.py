"""Boundary + geospatial + survey coverage planner.

Coverage is computed by intersecting each survey scanline with the polygon, so a
concave or L-shaped field is flown on the field itself rather than on its
bounding box (which is how the previous version behaved: waypoints over empty
ground, wasted flight time and unnecessary crop-spray exposure off-target).
"""
from __future__ import annotations

import math

from .geospatial import area_m2, perimeter_m


def validate_geojson(gj: dict) -> tuple[bool, list[str]]:
    errs = []
    if gj.get("type") != "Polygon":
        errs.append("only Polygon supported")
    try:
        ring = gj["coordinates"][0]
        if len(ring) < 4:
            errs.append("polygon ring needs >=4 positions")
        for p in ring:
            if len(p) != 2:
                errs.append(f"bad position {p}")
    except Exception as e:
        errs.append(str(e))
    return (not errs, errs)


class BoundaryMapper:
    @staticmethod
    def from_points(points: list[tuple[float, float]]) -> dict:
        ring = [[lon, lat] for lat, lon in points]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        return {"type": "Polygon", "coordinates": [ring]}

    @staticmethod
    def stats(gj: dict) -> dict:
        ring = gj["coordinates"][0]
        poly = [(p[1], p[0]) for p in ring]
        return {"area_m2": area_m2(poly), "perimeter_m": perimeter_m(poly)}


def _scanline_spans(poly: list[tuple[float, float]], lat: float) -> list[tuple[float, float]]:
    """Longitude spans where the horizontal line at ``lat`` crosses the polygon.

    Returns one (lon_min, lon_max) pair per contiguous crossing, so concave
    shapes keep their gaps instead of being filled in.
    """
    xs: list[float] = []
    n = len(poly)
    for i in range(n):
        y1, x1 = poly[i]
        y2, x2 = poly[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            t = (lat - y1) / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))
    xs.sort()
    spans = [(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]
    return [s for s in spans if s[1] - s[0] > 1e-9]


def _order_path(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Nearest-neighbour ordering, so the drone does not fly back across the field."""
    if len(points) < 3:
        return list(points)
    remaining = list(points)
    ordered = [remaining.pop(0)]
    while remaining:
        last = ordered[-1]
        nxt = min(remaining, key=lambda p: (p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2)
        ordered.append(nxt)
        remaining.remove(nxt)
    return ordered


class SurveyPlanner:
    """Boustrophedon coverage clipped to the polygon itself."""

    def __init__(self, altitude_m=20.0, speed_mps=5.0, side_overlap=0.7, fov_deg=78.0):
        self.alt = altitude_m
        self.speed = speed_mps
        self.overlap = side_overlap
        self.fov = fov_deg

    def footprint_m(self) -> float:
        return 2 * self.alt * math.tan(math.radians(self.fov / 2)) * (1 - self.overlap)

    def generate(self, boundary_gj: dict) -> dict:
        from ..navigation.coordinate_transform import haversine_m

        ring = boundary_gj["coordinates"][0]
        poly = [(p[1], p[0]) for p in ring]
        step_deg = max(self.footprint_m() / 111320.0, 1e-6)
        min_lat = min(p[0] for p in poly)
        max_lat = max(p[0] for p in poly)

        lines: list[list[tuple[float, float]]] = []
        # Waypoints sit slightly inside the boundary: a point exactly on an edge
        # is ambiguous for containment tests (ray casting returns False), so the
        # survey would immediately look like it is outside its own geofence.
        inset = min(step_deg * 0.05, 1e-5)
        lat = min_lat + step_deg / 2.0
        flip = False
        while lat < max_lat:
            spans = _scanline_spans(poly, lat)
            for lon_a, lon_b in spans:
                a, b = lon_a + inset, lon_b - inset
                if b - a <= 1e-9:
                    continue
                seg = [(lat, a), (lat, b)]
                lines.append(seg[::-1] if flip else seg)
            flip = not flip
            lat += step_deg
        if not lines:
            # Degenerate: field narrower than one footprint. Cover its centroid
            # so the operator still gets a usable single-point survey.
            clat = sum(p[0] for p in poly) / len(poly)
            clon = sum(p[1] for p in poly) / len(poly)
            lines = [[(clat, clon)]]

        # Order the segments so the path is continuous instead of alternating
        # blindly (a concave field has disconnected spans on the same latitude).
        ordered: list[tuple[float, float]] = []
        for seg in lines:
            ordered.extend(seg if not ordered else _order_path([ordered[-1]] + list(seg))[1:])

        wps = [{"lat": la, "lon": lo, "alt": self.alt} for la, lo in ordered]
        dist = 0.0
        for i in range(len(wps) - 1):
            dist += haversine_m(wps[i]["lat"], wps[i]["lon"],
                                wps[i + 1]["lat"], wps[i + 1]["lon"])
        return {
            "waypoints": wps,
            "lines": lines,
            "distance_m": dist,
            "eta_s": dist / max(self.speed, 0.1),
            "footprint_m": self.footprint_m(),
            "covered_area_m2": sum(
                haversine_m(a[0], a[1], b[0], b[1]) * max(self.footprint_m(), 0.0)
                for line in lines for a, b in zip(line, line[1:])
            ),
        }