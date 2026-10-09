"""Cached telemetry snapshot parsed from MAVLink messages."""
from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict

# MAV_SYS_STATUS_SENSOR bits (mavlink common.xml — official values)
_SYS_BITS = [
    (0x00000001, "3D_GYRO"), (0x00000002, "3D_ACCEL"), (0x00000004, "3D_MAG"),
    (0x00000008, "ABSOLUTE_PRESSURE"), (0x00000010, "DIFFERENTIAL_PRESSURE"),
    (0x00000020, "GPS"), (0x00000040, "OPTICAL_FLOW"), (0x00000080, "VISION_POSITION"),
    (0x00000100, "LASER_POSITION"), (0x00000200, "EXTERNAL_GROUND_TRUTH"),
    (0x00000400, "ANGULAR_RATE_CONTROL"), (0x00000800, "ATTITUDE_STABILIZATION"),
    (0x00001000, "YAW_POSITION"), (0x00002000, "Z_ALTITUDE_CONTROL"),
    (0x00004000, "XY_POSITION_CONTROL"), (0x00008000, "MOTOR_OUTPUTS"),
    (0x00010000, "RC_RECEIVER"), (0x00020000, "3D_GYRO2"), (0x00040000, "3D_ACCEL2"),
    (0x00080000, "3D_MAG2"), (0x00100000, "GEOFENCE"), (0x00200000, "AHRS"),
    (0x00400000, "TERRAIN"), (0x01000000, "LOGGING"), (0x02000000, "BATTERY"),
    (0x04000000, "PROXIMITY"), (0x10000000, "PREARM_CHECK"), (0x20000000, "OBSTACLE_AVOIDANCE"),
    (0x40000000, "PROPULSION"),
]

# MAV_ESTIMATOR_STATUS_FLAGS (common.xml). The arithmetic bits are what make an
# EKF estimate untrustworthy for position/velocity control; the GPS bits describe
# the GPS solution rather than the filter itself.
_EKF_FLAGS = [
    (0x01, "GPS_GLITCH"), (0x02, "ACCEL_ERROR"), (0x04, "VELOCITY_ERROR"),
    (0x08, "POS_ERROR"), (0x10, "AZ_ERROR"), (0x20, "OTHER_ERROR"),
    (0x40, "GPS_RESET"), (0x80, "GPS_DIVERGENCE"), (0x100, "GPS_NOFIX"),
]
_EKF_ARITHMETIC_BITS = 0x02 | 0x04 | 0x08 | 0x10 | 0x20


@dataclass
class TelemetrySnapshot:
    connected: bool = False
    armed: bool = False
    mode: str = "UNKNOWN"
    lat: float | None = None
    lon: float | None = None
    alt_m: float = 0.0
    relative_alt_m: float = 0.0
    alt_baro_m: float = 0.0
    ground_speed_mps: float = 0.0
    heading_deg: float = 0.0
    battery_voltage_v: float = 0.0
    battery_current_a: float = 0.0
    battery_remaining_pct: int = -1
    gps_fix: int = 0
    satellites: int = 0
    hdop: float = 99.9
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    sensor_issues: list = field(default_factory=list)
    # EKF health from ESTIMATOR_STATUS. ekf_flags is -1 until the FC has sent
    # the message at least once, which is what lets preflight report UNKNOWN
    # instead of guessing PASS.
    ekf_flags: int = -1
    ekf_vel_ratio: float = 0.0
    ekf_hpos_ratio: float = 0.0
    # RC / handheld transmitter link (from RC_CHANNELS or SYS_STATUS RC_RECEIVER bit)
    rc_last_seen: float = 0.0
    rc_rssi: int = -1
    rc_channels: int = 0
    rc_receiver_ok: bool = False
    updated_at: float = field(default_factory=time.time)

    RC_MAX_AGE_S = 3.0

    def rc_status(self) -> tuple[bool, float]:
        """(transmitter_connected, age_seconds). age -1 = never seen."""
        age = (time.time() - self.rc_last_seen) if self.rc_last_seen else -1.0
        fresh = 0.0 <= age <= self.RC_MAX_AGE_S
        return (fresh or self.rc_receiver_ok), age

    @property
    def ekf_known(self) -> bool:
        """False until the FC has reported ESTIMATOR_STATUS at least once."""
        return self.ekf_flags >= 0

    @property
    def ekf_issues(self) -> list[str]:
        if not self.ekf_known:
            return []
        return [name for bit, name in _EKF_FLAGS if self.ekf_flags & bit]

    @property
    def ekf_estimate_ok(self) -> bool:
        """True only when the estimator reports no arithmetic error flags."""
        return self.ekf_known and not (self.ekf_flags & _EKF_ARITHMETIC_BITS)

    def to_dict(self):
        d = asdict(self)
        d["rc_connected"], d["rc_age_s"] = self.rc_status()
        return d

class TelemetryStore:
    def __init__(self): self.snap = TelemetrySnapshot()
    def update_from_msg(self, msg) -> None:
        from math import degrees
        t = msg.get_type(); s = self.snap
        try:
            if t == "HEARTBEAT":
                s.connected = True
                s.armed = bool(msg.base_mode & 128)
                if getattr(msg, "autopilot", None) == 12:  # MAV_AUTOPILOT_PX4
                    main = (msg.custom_mode >> 16) & 0xFF
                    sub = (msg.custom_mode >> 24) & 0xFF
                    mains = {1: "MANUAL", 2: "ALTCTL", 3: "POSCTL", 4: "AUTO", 5: "ACRO",
                             6: "OFFBOARD", 7: "STABILIZED", 8: "RATTITUDE"}
                    subs = {1: "READY", 2: "TAKEOFF", 3: "LOITER", 4: "MISSION", 5: "RTL",
                            6: "LAND", 7: "RTGS", 8: "FOLLOW_TARGET", 9: "PRECLAND"}
                    name = mains.get(main, f"MAIN_{main}")
                    if main == 4 and sub in subs:
                        name = f"AUTO/{subs[sub]}"
                    s.mode = name
                else:
                    modes = {4: "GUIDED", 3: "AUTO", 5: "LOITER", 6: "RTL", 9: "LAND", 2: "STABILIZE"}
                    s.mode = modes.get(msg.custom_mode, f"MODE_{msg.custom_mode}")
            elif t == "GLOBAL_POSITION_INT":
                s.lat, s.lon = msg.lat / 1e7, msg.lon / 1e7
                s.alt_m, s.relative_alt_m = msg.alt / 1000.0, msg.relative_alt / 1000.0
                s.heading_deg = msg.hdg / 100.0 if msg.hdg != 65535 else s.heading_deg
            elif t == "VFR_HUD":
                s.ground_speed_mps = msg.groundspeed
                s.alt_baro_m = msg.alt
            elif t == "SYS_STATUS":
                s.battery_voltage_v = msg.voltage_battery / 1000.0
                s.battery_current_a = msg.current_battery / 100.0
                s.battery_remaining_pct = msg.battery_remaining
                # sensors enabled but not healthy -> why prearm/arming fails
                try:
                    enabled = int(getattr(msg, "onboard_control_sensors_enabled", 0) or 0)
                    health = int(getattr(msg, "onboard_control_sensors_health", 0) or 0)
                    sick = [name for bit, name in _SYS_BITS if (enabled & bit) and not (health & bit)]
                    if sick != s.sensor_issues:
                        s.sensor_issues = sick
                    s.rc_receiver_ok = bool(enabled & 0x00010000) and bool(health & 0x00010000)
                except Exception:
                    pass
            elif t == "RC_CHANNELS":
                # PX4 only publishes input_rc when receiver frames arrive, so a fresh
                # message with chancount > 0 == transmitter link alive.
                # rssi: 0-100 valid, 255 = unknown (keep last valid).
                chans = int(getattr(msg, "chancount", 0) or 0)
                rssi = int(getattr(msg, "rssi", 255) or 255)
                if chans > 0:
                    s.rc_last_seen = time.time()
                    s.rc_channels = chans
                    if rssi != 255:
                        s.rc_rssi = rssi
            elif t == "RC_CHANNELS_RAW":
                rssi = int(getattr(msg, "rssi", 255) or 255)
                s.rc_last_seen = time.time()
                s.rc_channels = s.rc_channels or 8
                if rssi != 255:
                    s.rc_rssi = rssi
            elif t == "GPS_RAW_INT":
                s.gps_fix, s.satellites = msg.fix_type, msg.satellites_visible
                s.hdop = msg.eph / 100.0 if msg.eph != 65535 else 99.9
            elif t == "ATTITUDE":
                s.roll_deg, s.pitch_deg = degrees(msg.roll), degrees(msg.pitch)
                s.yaw_deg = degrees(msg.yaw)
            elif t == "HIGHRES_IMU":
                s.accel_x, s.accel_y, s.accel_z = msg.xacc, msg.yacc, msg.zacc
            elif t == "ESTIMATOR_STATUS":
                s.ekf_flags = int(getattr(msg, "flags", 0) or 0)
                s.ekf_vel_ratio = float(getattr(msg, "vel_ratio", 0.0) or 0.0)
                s.ekf_hpos_ratio = float(getattr(msg, "hpos_ratio", 0.0) or 0.0)
            elif t in ("SCALED_IMU", "SCALED_IMU2", "RAW_IMU"):
                g = 9.80665 / 1000.0  # milli-g -> m/s^2
                s.accel_x, s.accel_y, s.accel_z = msg.xacc * g, msg.yacc * g, msg.zacc * g
        finally:
            s.updated_at = time.time()
