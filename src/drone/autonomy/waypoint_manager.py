"""Mission validation: altitude/speed/geofence/mandatory fields."""
from __future__ import annotations

def validate_mission(body: dict, max_alt: float, max_speed: float) -> tuple[bool, list[str]]:
    errs: list[str] = []
    if not isinstance(body, dict): return False, ["mission must be an object"]
    if "waypoints" in body:
        for i, w in enumerate(body["waypoints"]):
            if "lat" not in w or "lon" not in w: errs.append(f"wp[{i}] missing lat/lon")
            if float(w.get("alt", 0)) > max_alt: errs.append(f"wp[{i}] exceeds max_altitude {max_alt}")
    to = body.get("takeoff", {})
    if to and float(to.get("altitude_m", 0)) > max_alt: errs.append("takeoff exceeds max_altitude")
    sp = float(body.get("survey", {}).get("speed_mps", 0) or 0)
    if sp > max_speed: errs.append(f"survey speed {sp} exceeds max {max_speed}")
    return (len(errs) == 0, errs)
