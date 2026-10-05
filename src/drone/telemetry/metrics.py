"""Raspberry Pi stats via psutil. Degraded gracefully when sensors missing."""
from __future__ import annotations
import shutil
try: import psutil
except ImportError: psutil = None

def pi_stats() -> dict:
    out = {"cpu_pct": 0.0, "ram_pct": 0.0, "disk_free_pct": 100.0, "temp_c": 0.0}
    try:
        if psutil:
            out["cpu_pct"] = psutil.cpu_percent(interval=0.1)
            out["ram_pct"] = psutil.virtual_memory().percent
            out["disk_free_pct"] = 100.0 - psutil.disk_usage("/").percent
        import glob
        for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            out["temp_c"] = int(open(p).read().strip()) / 1000.0
            break
    except Exception:
        pass
    return out
