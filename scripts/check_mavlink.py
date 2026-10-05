#!/usr/bin/env python3
"""Check Pixhawk serial + heartbeat. Honest output, no faking."""
import os, sys
dev = os.environ.get("MAVLINK_CONNECTION", "/dev/ttyACM0")
print(f"Checking {dev} ...")
exists = os.path.exists(dev) or dev.startswith(("udp:", "tcp:"))
print(("FOUND " if exists else "MISSING ") + dev)
if not exists: sys.exit(2)
try:
    sys.path.insert(0, "src")
    from drone.mavlink.connection import MavlinkConnection
    c = MavlinkConnection(dev, int(os.environ.get("MAVLINK_BAUD", "115200")))
    c.connect()
    ok = c.wait_heartbeat(5.0)
    print("HEARTBEAT " + ("OK" if ok else "TIMEOUT"))
    sys.exit(0 if ok else 1)
except Exception as e:
    print(f"ERROR {e}"); sys.exit(1)
