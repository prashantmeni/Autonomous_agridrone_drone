from __future__ import annotations
from .coordinate_transform import haversine_m
def distance_to(a: tuple[float,float], b: tuple[float,float]) -> float:
    return haversine_m(a[0], a[1], b[0], b[1])
