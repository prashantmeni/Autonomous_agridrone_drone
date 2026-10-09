"""Mission upload/download via the MAVLink mission protocol.

Every wait here goes through a ``message_hooks`` callback rather than
``recv_match(blocking=True)``. The telemetry and maintain loops drain the link
with ``recv_match(blocking=False)`` and discard anything they do not use, so a
blocking recv would race them and usually never see the FC's replies — which is
exactly why the previous version of this module could not work.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("drone.mavlink.mission")

# MISSION_ACK.type values; only ACCEPTED means the FC took the upload.
ACK_ACCEPTED = 0
ACK_REASONS = {
    1: "generic error",
    2: "unsupported frame",
    3: "unsupported command",
    4: "no space in mission storage",
    5: "invalid parameter",
    6: "invalid param 1",
    7: "invalid param 2",
    8: "invalid param 3",
    9: "invalid param 4",
    10: "invalid sequence",
    11: "denied",
    12: "operation cancelled",
}


def build_items(waypoints: list[dict]) -> list[dict]:
    """Normalise mission waypoints into MISSION_ITEM_INT fields.

    Accepts the loose dicts the planner produces and fills in the MAVLink
    command ids and frames. ``command`` defaults to MAV_CMD_NAV_WAYPOINT.
    """
    items: list[dict] = []
    for w in waypoints:
        items.append(
            {
                "seq": len(items),
                "lat": float(w["lat"]),
                "lon": float(w["lon"]),
                "alt": float(w.get("alt", w.get("altitude_m", 0)) or 0),
                "command": int(w.get("command", 16)),
                "hold_s": float(w.get("hold_s", 0) or 0),
            }
        )
    return items


def _send_item(conn, m, item: dict) -> None:
    from pymavlink import mavutil

    m.mav.mission_item_int_send(
        conn.target_system, conn.target_component, item["seq"],
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, item["command"], 0, 1,
        item["hold_s"], 0.0, 0.0, 0.0, float("nan"),
        int(item["lat"] * 1e7), int(item["lon"] * 1e7), item["alt"],
    )


def clear_mission(conn, timeout: float = 8.0) -> tuple[bool, str]:
    """Erase the mission list on the flight controller."""
    master = conn.master
    if master is None:
        return False, "MAVLink not connected"
    done = threading.Event()
    result: dict = {}

    def hook(*args):
        m = args[-1]
        if m.get_type() == "MISSION_ACK":
            result["type"] = int(m.type)
            done.set()

    hooks = getattr(master, "message_hooks", None)
    if hooks is not None:
        hooks.append(hook)
    try:
        master.mav.mission_clear_all_send(conn.target_system, conn.target_component)
        if hooks is None:
            deadline = time.time() + timeout
            while time.time() < deadline:
                m = master.recv_match(type="MISSION_ACK", blocking=True, timeout=0.5)
                if m is not None:
                    result["type"] = int(m.type)
                    break
        else:
            done.wait(timeout)
    except Exception as e:
        return False, f"clear failed: {e}"
    finally:
        if hooks is not None:
            try:
                hooks.remove(hook)
            except ValueError:
                pass

    if "type" not in result:
        return False, "no MISSION_ACK from flight controller"
    if result["type"] != ACK_ACCEPTED:
        return False, ACK_REASONS.get(result["type"], f"rejected ({result['type']})")
    return True, "cleared"


def upload_mission(conn, waypoints: list[dict], timeout: float = 30.0) -> tuple[bool, str]:
    """Upload a waypoint list and wait for the FC to acknowledge it.

    Returns ``(ok, detail)``. The flight controller is the authority: without a
    successful MISSION_ACK nothing here claims the mission was accepted.
    """
    master = conn.master
    if master is None or not conn.connected:
        return False, "MAVLink not connected"
    if not waypoints:
        return False, "mission has no items"

    items = build_items(waypoints)
    done = threading.Event()
    state: dict = {"ack": None, "sent": set(), "requested": 0}

    def hook(*args):
        m = args[-1]
        mtype = m.get_type()
        if mtype in ("MISSION_REQUEST", "MISSION_REQUEST_INT"):
            seq = int(m.seq)
            state["requested"] += 1
            if 0 <= seq < len(items):
                _send_item(conn, master, items[seq])
                state["sent"].add(seq)
        elif mtype == "MISSION_ACK":
            state["ack"] = int(m.type)
            done.set()

    hooks = getattr(master, "message_hooks", None)
    if hooks is not None:
        hooks.append(hook)
    try:
        master.mav.mission_count_send(conn.target_system, conn.target_component, len(items))
        if hooks is None:
            deadline = time.time() + timeout
            while time.time() < deadline and state["ack"] is None:
                m = master.recv_match(
                    type=["MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK"],
                    blocking=True, timeout=1.0)
                if m is None:
                    continue
                t = m.get_type()
                if t == "MISSION_ACK":
                    state["ack"] = int(m.type)
                else:
                    seq = int(m.seq)
                    if 0 <= seq < len(items):
                        _send_item(conn, master, items[seq])
                        state["sent"].add(seq)
        else:
            done.wait(timeout)
    except Exception as e:
        return False, f"upload failed: {e}"
    finally:
        if hooks is not None:
            try:
                hooks.remove(hook)
            except ValueError:
                pass

    if state["ack"] is None:
        return False, f"no MISSION_ACK within {timeout:.0f}s (sent {len(state['sent'])}/{len(items)})"
    if state["ack"] != ACK_ACCEPTED:
        return False, ACK_REASONS.get(state["ack"], f"rejected ({state['ack']})")
    log.info("uploaded %d mission items to FC", len(items))
    return True, f"accepted {len(items)} items"


def set_mission_current(conn, seq: int, timeout: float = 8.0) -> tuple[bool, str]:
    """Jump the FC to a mission item (used to resume and to skip ahead)."""
    master = conn.master
    if master is None:
        return False, "MAVLink not connected"
    done = threading.Event()
    state: dict = {}

    def hook(*args):
        m = args[-1]
        if m.get_type() == "MISSION_ACK":
            state["type"] = int(m.type)
            done.set()

    hooks = getattr(master, "message_hooks", None)
    if hooks is not None:
        hooks.append(hook)
    try:
        master.mav.mission_set_current_send(conn.target_system, conn.target_component, int(seq))
        if hooks is not None:
            done.wait(timeout)
        else:
            deadline = time.time() + timeout
            while time.time() < deadline:
                m = master.recv_match(type="MISSION_ACK", blocking=True, timeout=0.5)
                if m is not None:
                    state["type"] = int(m.type)
                    break
    except Exception as e:
        return False, f"set current failed: {e}"
    finally:
        if hooks is not None:
            try:
                hooks.remove(hook)
            except ValueError:
                pass

    if "type" not in state:
        return False, "no MISSION_ACK"
    if state["type"] != ACK_ACCEPTED:
        return False, ACK_REASONS.get(state["type"], f"rejected ({state['type']})")
    return True, f"current={seq}"


def request_mission_list(conn, timeout: float = 8.0) -> list[dict]:
    """Download the FC's mission list (used to verify what is actually on board)."""
    master = conn.master
    if master is None:
        return []
    collected: dict[int, dict] = {}
    done = threading.Event()

    def hook(*args):
        m = args[-1]
        t = m.get_type()
        if t == "MISSION_ITEM_INT" or t == "MISSION_ITEM":
            collected[int(m.seq)] = {
                "seq": int(m.seq), "lat": m.x / 1e7, "lon": m.y / 1e7,
                "alt": float(m.z), "command": int(m.command),
            }
        elif t == "MISSION_ACK":
            done.set()

    hooks = getattr(master, "message_hooks", None)
    if hooks is not None:
        hooks.append(hook)
    try:
        master.mav.mission_request_list_send(conn.target_system, conn.target_component)
        if hooks is None:
            deadline = time.time() + timeout
            while time.time() < deadline:
                m = master.recv_match(
                    type=["MISSION_ITEM", "MISSION_ITEM_INT", "MISSION_ACK"],
                    blocking=True, timeout=0.5)
                if m is None:
                    continue
                if m.get_type() == "MISSION_ACK":
                    break
                collected[int(m.seq)] = {
                    "seq": int(m.seq), "lat": m.x / 1e7, "lon": m.y / 1e7,
                    "alt": float(m.z), "command": int(m.command),
                }
        else:
            done.wait(timeout)
    finally:
        if hooks is not None:
            try:
                hooks.remove(hook)
            except ValueError:
                pass
    return [collected[k] for k in sorted(collected)]


def start_mission(conn) -> tuple[bool, str]:
    """Put the FC into AUTO so it begins executing the uploaded mission."""
    from .commands import set_mode
    return (True, "AUTO") if set_mode(conn, "AUTO") else (False, "FC refused AUTO mode")


def pause_mission(conn) -> tuple[bool, str]:
    from .commands import set_mode
    return (True, "HOLD") if set_mode(conn, "HOLD") else (False, "FC refused HOLD mode")


def resume_mission(conn) -> tuple[bool, str]:
    from .commands import set_mode
    return (True, "AUTO") if set_mode(conn, "AUTO") else (False, "FC refused AUTO mode")


def upload_waypoints(conn, waypoints: list[dict], timeout: float = 30.0) -> bool:
    """Backwards-compatible wrapper over :func:`upload_mission`."""
    ok, _ = upload_mission(conn, waypoints, timeout=timeout)
    return ok
