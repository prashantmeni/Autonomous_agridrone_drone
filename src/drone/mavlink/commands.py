"""High-level vehicle commands via MAVLink. PX4 executes control loops."""
from __future__ import annotations
import logging, threading, time
log = logging.getLogger("drone.mavlink.cmd")

_ACK_RESULT = {0: "ACCEPTED", 1: "TEMPORARILY_REJECTED", 2: "DENIED",
               3: "UNSUPPORTED", 4: "FAILED", 5: "IN_PROGRESS", 6: "CANCELLED"}

def wait_for(conn, want_type: str, predicate, timeout: float = 5.0):
    """Wait for a specific MAVLink message without racing the pollers.

    The telemetry and maintain loops drain the link continuously with
    recv_match(blocking=False) and discard messages they don't use, so a
    blocking recv wait would almost never see its ACK. pymavlink message
    hooks fire from whichever thread pulls the message off the wire, so
    nothing is lost. Returns (message, last_statustext_seen, timed_out).
    """
    master = conn.master
    if master is None:
        return None, None, True
    found: dict = {}
    texts: list[str] = []
    done = threading.Event()

    def hook(*args):
        m = args[-1]  # pymavlink calls hook(self, msg)
        t = m.get_type()
        if t == want_type and predicate(m):
            found["msg"] = m
            done.set()
        elif t == "STATUSTEXT":
            try:
                texts.append(str(m.text).rstrip("\x00"))
            except Exception:
                pass

    hooks = getattr(master, "message_hooks", None)
    if hooks is None:  # ancient pymavlink: fall back to plain recv wait
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = master.recv_match(type=want_type, blocking=True, timeout=0.5)
            if m is not None and predicate(m):
                return m, None, False
        return None, None, True
    hooks.append(hook)
    try:
        done.wait(timeout)
    finally:
        try:
            hooks.remove(hook)
        except ValueError:
            pass
    return found.get("msg"), (texts[-1] if texts else None), "msg" not in found

def _recent_px4_text(conn, since: float) -> str | None:
    """Latest PX4 STATUSTEXT recorded after `since` (captured globally by connection)."""
    buf = getattr(conn, "px4_status", None)
    if not buf:
        return None
    for ts, text in reversed(buf):
        if ts >= since:
            return text
    return None

def _cmd(conn, command: int, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0, timeout=5.0) -> tuple[bool, str]:
    if conn.master is None or not conn.connected:
        log.error("command rejected: MAVLink not connected")
        return False, "MAVLink not connected"
    t0 = time.time()
    conn.master.mav.command_long_send(conn.target_system, conn.target_component,
        command, 0, p1, p2, p3, p4, p5, p6, p7)
    ack, text, _ = wait_for(conn, "COMMAND_ACK", lambda m: m.command == command, timeout)
    if ack is None:
        detail = f"no ACK from autopilot in {timeout:.0f}s"
        if text:
            detail += f" (last status: {text})"
        log.error(f"CMD {command} timeout; {detail}")
        return False, detail
    if ack.result == 0:
        log.info(f"CMD {command} -> ACCEPTED")
        return True, "ACCEPTED"
    result = _ACK_RESULT.get(ack.result, f"result={ack.result}")
    if not text:
        text = _recent_px4_text(conn, t0)
    detail = result if not text else f"{result}: {text}"
    log.warning(f"CMD {command} -> {detail}")
    return False, detail

def arm(conn, force: bool = False) -> tuple[bool, str]:
    return _cmd(conn, 400, 1, 21196 if force else 0)

def disarm(conn) -> tuple[bool, str]:
    return _cmd(conn, 400, 0)

def takeoff(conn, alt_m: float) -> tuple[bool, str]:
    return _cmd(conn, 22, 0, 0, 0, 0, 0, 0, alt_m)

def rtl(conn) -> bool:
    return _cmd(conn, 20)[0]

def land(conn) -> bool:
    return _cmd(conn, 21)[0]

def set_mode(conn, mode: str) -> bool:
    from pymavlink import mavutil
    if conn.master is None or not conn.connected:
        log.error("mode rejected: MAVLink not connected"); return False
    # PX4 custom-mode layout (inverse of our telemetry decoder):
    #   custom_mode = (sub << 24) | (main << 16)
    main_modes = {"MANUAL": 1, "ALTCTL": 2, "POSCTL": 3, "AUTO": 4,
                  "ACRO": 5, "OFFBOARD": 6, "STABILIZED": 7, "RATTITUDE": 8}
    sub_modes = {"READY": 1, "TAKEOFF": 2, "LOITER": 3, "MISSION": 4, "RTL": 5,
                 "LAND": 6, "RTGS": 7, "FOLLOW_TARGET": 8, "PRECLAND": 9}
    aliases = {"STABILIZED": (7, 0), "STABILIZE": (7, 0),
               "ALTCTL": (2, 0), "ALTITUDE": (2, 0),
               "POSCTL": (3, 0), "POSITION": (3, 0),
               "HOLD": (4, 3), "LOITER": (4, 3),
               "AUTO": (4, 4), "MISSION": (4, 4),
               "RTL": (4, 5), "LAND": (4, 6), "TAKEOFF": (4, 2), "READY": (4, 1),
               "ACRO": (5, 0), "OFFBOARD": (6, 0), "MANUAL": (1, 0),
               "RATTITUDE": (8, 0), "FOLLOW_TARGET": (4, 8), "PRECLAND": (4, 9)}
    key = (mode or "").strip().upper().replace(" ", "").replace("-", "")
    custom = None
    if "/" in key:
        main_name, sub_name = key.split("/", 1)
        main = main_modes.get(main_name)
        sub = sub_modes.get(sub_name)
        if main is not None and sub is not None:
            custom = (sub << 24) | (main << 16)
    elif key in aliases:
        main, sub = aliases[key]
        custom = (sub << 24) | (main << 16)
    if custom is None:
        mapping = conn.master.mode_mapping()  # non-PX4 autopilots
        if not mapping or mode not in mapping:
            log.error(f"unknown mode {mode}"); return False
        custom = mapping[mode]
    conn.master.mav.set_mode_send(conn.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, custom)
    log.info(f"SET_MODE {mode} -> custom_mode=0x{custom:08X}")
    return True

def goto(conn, lat: float, lon: float, alt_m: float) -> None:
    conn.master.mav.set_position_target_global_int_send(
        0, conn.target_system, conn.target_component, 0b0000111111111000,
        int(lat * 1e7), int(lon * 1e7), alt_m, 0, 0, 0, 0, 0, 0)
