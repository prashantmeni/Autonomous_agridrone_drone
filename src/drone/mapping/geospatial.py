"""Area/perimeter on equirectangular projection."""
from __future__ import annotations
import math
def area_m2(poly):
    if len(poly) < 3: return 0.0
    lat0 = sum(p[0] for p in poly)/len(poly)
    xs = [(lon-poly[0][1])*111320*math.cos(math.radians(lat0)) for _,lon in poly]
    ys = [(lat-poly[0][0])*110540 for lat,_ in poly]
    return abs(sum(xs[i]*ys[(i+1)%len(xs)]-xs[(i+1)%len(xs)]*ys[i] for i in range(len(xs))))/2
def perimeter_m(poly):
    from ..navigation.coordinate_transform import haversine_m
    return sum(haversine_m(poly[i][0],poly[i][1],poly[(i+1)%len(poly)][0],poly[(i+1)%len(poly)][1]) for i in range(len(poly))) if len(poly)>1 else 0.0
