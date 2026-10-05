"""Mission upload/download via MAVLink mission protocol."""
from __future__ import annotations
import logging, time
log = logging.getLogger("drone.mavlink.mission")

def upload_waypoints(conn, waypoints: list[dict], timeout: float = 30.0) -> bool:
    """waypoints: [{lat, lon, alt, command(16=WP,22=takeoff,21=land), hold_s}]"""
    m = conn.master
    if m is None or not conn.connected:
        log.error("upload rejected: not connected"); return False
    from pymavlink import mavutil
    n = len(waypoints)
    m.mav.mission_count_send(conn.target_system, conn.target_component, n)
    t0 = time.time(); idx = 0
    while idx < n and time.time() - t0 < timeout:
        msg = m.recv_match(type=["MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK"], blocking=True, timeout=2)
        if msg is None: continue
        t = msg.get_type()
        if t == "MISSION_ACK":
            return msg.type == 0
        seq = msg.seq
        wp = waypoints[seq]
        cmd = wp.get("command", 16)
        m.mav.mission_item_int_send(conn.target_system, conn.target_component, seq,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, cmd, 0, 1,
            wp.get("hold_s", 0), 0, 0, 0,
            int(wp["lat"] * 1e7), int(wp["lon"] * 1e7), float(wp["alt"]))
        idx += 1
    return False

def start_mission(conn) -> bool:
    from .commands import set_mode
    return set_mode(conn, "AUTO")
