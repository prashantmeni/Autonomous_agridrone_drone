"""Wireless induction charging dock status.

Docking is derived from real flight state (on ground + disarmed). Charging
itself is detected from a sustained battery-voltage rise while docked — no
sensor readings are invented. Detection is deliberately strict:

* the first 10 s after touching down are ignored (battery voltage rebounds
  after the motors stop — that is load release, not charging),
* then a robust median-of-halves trend over a 30 s window must exceed
  RISE_V, so random battery noise cannot masquerade as charging.

`source` always says how the conclusion was reached so the UI can stay honest.
"""
from __future__ import annotations

import threading
import time
from collections import deque


class WirelessChargingStation:
    WINDOW_S = 30.0
    SETTLE_S = 10.0     # ignore load-rebound right after landing
    MIN_SAMPLES = 5
    RISE_V = 0.02       # sustained rise required — resting noise stays below this

    def __init__(self, history_s: float = 65.0):
        self._lock = threading.Lock()
        self._samples: deque[tuple[float, float]] = deque(maxlen=int(history_s * 2))  # (ts, volts)
        self.pad_id = "PAD_01"
        self._dock_since: float | None = None

    @staticmethod
    def _median(vals: list[float]) -> float:
        m = len(vals)
        return vals[m // 2] if m % 2 else (vals[m // 2 - 1] + vals[m // 2]) / 2.0

    def _trend(self) -> tuple[float | None, int]:
        """Robust (median of second half - median of first half) over the window."""
        now = time.time()
        with self._lock:
            vals = [v for ts, v in self._samples if now - ts <= self.WINDOW_S and v > 0]
        n = len(vals)
        if n < self.MIN_SAMPLES:
            return None, n
        half = n // 2
        first = sorted(vals[:half])
        second = sorted(vals[-half:])
        return self._median(second) - self._median(first), n

    def status(self, app) -> dict:
        snap = app.tstore.snap
        from ..core.state import FlightState

        docked = (
            not snap.armed
            and (snap.relative_alt_m or 0.0) < 0.25
            and app.fsm.state in (FlightState.LANDED, FlightState.DISARMED)
        )
        auto = True
        try:
            from ..core.behaviour import load_behaviour
            rules = load_behaviour()
            auto = rules.auto_charge_on_landing
            self.pad_id = rules.landing_pad_id
        except Exception:
            pass

        now = time.time()
        if not docked:
            self._dock_since = None
            with self._lock:
                self._samples.clear()
        else:
            if self._dock_since is None:
                self._dock_since = now
            # skip the settle period so post-motor voltage rebound reads as neither
            if now - self._dock_since >= self.SETTLE_S and snap.battery_voltage_v > 0:
                with self._lock:
                    self._samples.append((now, snap.battery_voltage_v))

        delta_v, n = self._trend()
        charging = bool(
            docked and auto and delta_v is not None and delta_v >= self.RISE_V
        )

        if not docked:
            source = "not_docked"
        elif not auto:
            source = "auto_charge_disabled"
        elif charging:
            source = "battery_voltage_trend"
        elif n < self.MIN_SAMPLES:
            source = "insufficient_data"
        else:
            source = "no_charge_rise_detected"

        return {
            "docked": docked,
            "charging": charging,
            "auto_charge_on_landing": auto,
            "source": source,
            "voltage_v": round(snap.battery_voltage_v, 2) if snap.battery_voltage_v else None,
            "battery_pct": snap.battery_remaining_pct if snap.battery_remaining_pct >= 0 else None,
            "voltage_trend_mv": int(delta_v * 1000) if delta_v is not None else None,
            "samples": n,
            "pad_id": self.pad_id,
        }
