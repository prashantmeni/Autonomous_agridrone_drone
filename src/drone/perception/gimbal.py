"""Pan-tilt servo (camera gimbal) control.

Sends MAV_CMD_DO_MOUNT_CONTROL to the autopilot when a live MAVLink link
exists; otherwise reports SERVO_CONTROL_UNAVAILABLE honestly while still
remembering the requested angles for the operator UI.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("drone.perception.gimbal")

PAN_MIN, PAN_MAX = -90.0, 90.0
TILT_MIN, TILT_MAX = -90.0, 0.0

MAV_CMD_DO_MOUNT_CONTROL = 205
MAV_MOUNT_MODE_MAVLINK_TARGETING = 0


class PanTiltController:
    def __init__(self):
        self._lock = threading.Lock()
        self.pan_deg = 0.0        # -90 (left) .. +90 (right)
        self.tilt_deg = -90.0     # -90 (nadir/down) .. 0 (horizon)
        self.last_sent = False
        self.last_error: str | None = None

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(v)))

    def set(self, pan: float | None = None, tilt: float | None = None, conn=None) -> dict:
        with self._lock:
            if pan is not None:
                self.pan_deg = self._clamp(pan, PAN_MIN, PAN_MAX)
            if tilt is not None:
                self.tilt_deg = self._clamp(tilt, TILT_MIN, TILT_MAX)
            pan, tilt = self.pan_deg, self.tilt_deg

        sent, err = False, None
        if conn is not None and getattr(conn, "master", None) is not None and conn.connected:
            try:
                conn.master.mav.command_long_send(
                    conn.target_system,
                    conn.target_component,
                    MAV_CMD_DO_MOUNT_CONTROL,
                    0,
                    tilt,          # param1: pitch
                    0.0,           # param2: roll
                    pan,           # param3: yaw
                    0.0, 0.0, 0.0,
                    MAV_MOUNT_MODE_MAVLINK_TARGETING,
                )
                sent = True
                err = None
            except Exception as e:
                err = str(e)
                log.warning("gimbal command failed: %s", e)
        else:
            err = "MAVLINK_NOT_CONNECTED"

        with self._lock:
            self.last_sent = sent
            self.last_error = err

        return self.status()

    def status(self) -> dict:
        with self._lock:
            return {
                "pan_deg": round(self.pan_deg, 1),
                "tilt_deg": round(self.tilt_deg, 1),
                "pan_min": PAN_MIN,
                "pan_max": PAN_MAX,
                "tilt_min": TILT_MIN,
                "tilt_max": TILT_MAX,
                "control": "mavlink" if self.last_sent else "unavailable",
                "available": self.last_sent,
                "last_error": self.last_error,
            }
