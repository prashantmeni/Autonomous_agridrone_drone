"""KML <coordinates>lon,lat[,alt] → GeoJSON Polygon."""
from __future__ import annotations
import re
def kml_to_geojson(kml: str) -> dict:
    m = re.search(r"<coordinates>(.*?)</coordinates>", kml, re.S | re.I)
    if not m: raise ValueError("no <coordinates> in KML")
    ring = []
    for tok in m.group(1).strip().split():
        parts = tok.split(",")
        ring.append([float(parts[0]), float(parts[1])])
    if ring and ring[0] != ring[-1]: ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}
