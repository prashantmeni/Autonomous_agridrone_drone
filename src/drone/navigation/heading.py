from __future__ import annotations
def normalize(h: float) -> float: return h % 360.0
def diff(a: float, b: float) -> float: return (b - a + 180) % 360 - 180
