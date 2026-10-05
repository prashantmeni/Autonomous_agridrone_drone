from __future__ import annotations
def to_mavlink_mission(wps: list[dict]) -> list[dict]:
    out = []
    for i, w in enumerate(wps):
        cmd = 16
        if i == 0: cmd = 16
        out.append({"lat": w["lat"], "lon": w["lon"], "alt": w.get("alt", 20), "command": cmd})
    return out
