"""GPS helpers."""
from __future__ import annotations
def fix_quality(fix_type: int) -> str:
    return {0:"NO_FIX",1:"NO_FIX",2:"2D",3:"3D",4:"DGPS",5:"RTK_FLOAT",6:"RTK_FIXED"}.get(fix_type,"UNKNOWN")
def is_valid(snap, min_sats: int = 8) -> bool:
    return (snap.gps_fix or 0) >= 3 and (snap.satellites or 0) >= min_sats
