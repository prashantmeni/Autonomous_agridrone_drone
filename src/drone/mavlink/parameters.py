"""Parameter read/write helpers (hook-based, race-free against the pollers)."""
import logging
from .commands import wait_for
log = logging.getLogger("drone.mavlink.param")

def _match(name: str):
    return lambda p: p.param_id.strip("\x00") == name

def read_param(conn, name: str, timeout=5.0):
    if conn.master is None: return None
    # The request is sent from inside wait_for, once the PARAM_VALUE hook is
    # already installed; sending first would drop an immediate reply.
    m, _, _ = wait_for(
        conn, "PARAM_VALUE", _match(name), timeout,
        send=lambda: conn.master.mav.param_request_read_send(
            conn.target_system, conn.target_component, name.encode()[:16], -1))
    return m.param_value if m is not None else None

def set_param(conn, name: str, value: float, timeout=5.0) -> bool:
    from pymavlink import mavutil
    if conn.master is None: return False
    # Same ordering rule as read_param: hook first, then send.
    m, _, _ = wait_for(
        conn, "PARAM_VALUE", _match(name), timeout,
        send=lambda: conn.master.mav.param_set_send(
            conn.target_system, conn.target_component,
            name.encode()[:16], float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32))
    ok = m is not None
    log.info(f"SET_PARAM {name}={value} -> {'ack ok' if ok else 'timeout'}")
    return ok