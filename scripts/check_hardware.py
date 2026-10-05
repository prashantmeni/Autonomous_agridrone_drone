#!/usr/bin/env python3
import sys; sys.path.insert(0, "src")
from drone.telemetry.metrics import pi_stats
from drone.core.config import load_config
import os
print("AUTONOMOUS SMART FARMING DRONE"); print("-"*32)
s = pi_stats()
print(f"CPU {s['cpu_pct']:.0f}%  Temp {s['temp_c']:.1f}C  DiskFree {s['disk_free_pct']:.0f}%")
print(f"MAVLINK {os.environ.get('MAVLINK_CONNECTION','/dev/ttyACM0')}")
print("NOTE: CAMERA / OBSTACLE / DISEASE report UNVAILABLE until hardware validated")
