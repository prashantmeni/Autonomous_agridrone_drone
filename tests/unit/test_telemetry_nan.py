"""PX4 sends NaN/sentinel values for unknown telemetry; none may reach JSON.

The flight controller uses NaN and UINT16_MAX sentinels (GPS dilution, heading
before a fix, several attitude fields). Letting those into the snapshot made
FastAPI raise "Out of range float values are not JSON compliant: nan", which
turned /api/drone/status into a 500 the moment the FC connected.
"""
import math
import sys
from math import degrees
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from drone.mavlink.telemetry import TelemetryStore, _finite  # noqa: E402


def msg(mtype, **f):
    m = SimpleNamespace(_type=mtype, **f)
    m.get_type = lambda: mtype
    return m


def test_finite_coerces_bad_values():
    assert _finite(float("nan")) == 0.0
    assert _finite(float("inf")) == 0.0
    assert _finite(None) == 0.0
    assert _finite("abc") == 0.0
    assert _finite(3.5) == 3.5
    assert _finite(None, 99.9) == 99.9


def test_global_position_nan_becomes_none():
    st = TelemetryStore()
    st.update_from_msg(msg("GLOBAL_POSITION_INT", lat=float("nan"), lon=1.0,
                            alt=float("nan"), relative_alt=2500, hdg=65535))
    snap = st.snap
    assert snap.lat is None
    assert snap.lon == pytest.approx(1.0 / 1e7)
    assert math.isfinite(snap.alt_m)
    assert snap.relative_alt_m == pytest.approx(2.5)


def test_attitude_nan_is_zeroed():
    st = TelemetryStore()
    st.update_from_msg(msg("ATTITUDE", roll=float("nan"), pitch=0.1, yaw=0.2))
    assert st.snap.roll_deg == 0.0
    assert st.snap.pitch_deg == pytest.approx(degrees(0.1))
    assert math.isfinite(st.snap.yaw_deg)


def test_vfr_hud_nan_is_zeroed():
    st = TelemetryStore()
    st.update_from_msg(msg("VFR_HUD", groundspeed=float("nan"), alt=float("inf")))
    assert st.snap.ground_speed_mps == 0.0
    assert st.snap.alt_baro_m == 0.0


def test_gps_raw_nan_dilution_uses_default():
    st = TelemetryStore()
    st.update_from_msg(msg("GPS_RAW_INT", fix_type=3, satellites_visible=11,
                           eph=float("nan")))
    assert st.snap.hdop == pytest.approx(99.9)
    assert math.isfinite(st.snap.hdop)


def test_snapshot_is_json_serialisable_after_nan_input():
    import json
    st = TelemetryStore()
    st.update_from_msg(msg("ATTITUDE", roll=float("nan"), pitch=float("nan"), yaw=float("nan")))
    st.update_from_msg(msg("VFR_HUD", groundspeed=float("nan"), alt=float("nan")))
    st.update_from_msg(msg("HIGHRES_IMU", xacc=float("nan"), yacc=1.0, zacc=9.8))
    # allow_nan=False is what FastAPI's encoder enforces
    text = json.dumps(st.snap.to_dict(), allow_nan=False)
    assert "NaN" not in text