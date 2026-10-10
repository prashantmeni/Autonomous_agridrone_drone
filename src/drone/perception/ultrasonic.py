"""Ultrasonic rangefinder support (HC-SR04 class sensors).

The sensor is driven over two GPIO lines: a 10 microsecond TRIG pulse, then the
width of the ECHO pulse, which converts to distance at ~58 us/cm (343 m/s round
trip at 20 C).

Raw HC-SR04 readings are noisy and occasionally absurd (a 6 m reading from a
wall 1 m away), so every reading is the median of several samples and implausible
values are rejected rather than averaged in. Without that, a single spike can
command a spurious avoidance manoeuvre.

The GPIO layer is abstracted so the driver can be exercised without hardware:
:class:`GpioBackend` is implemented by :class:`RpiGpioBackend` on a Pi and by a
plain double in the tests.
"""
from __future__ import annotations

import logging
import statistics
import time
from typing import Protocol

log = logging.getLogger("drone.perception.ultrasonic")

# Speed of sound round trip: 343 m/s -> 58 us per cm.
US_PER_CM = 58.0
# Beyond this the reading is a reflection artefact, not a surface.
MAX_RANGE_M = 4.0
MIN_RANGE_M = 0.02
# HC-SR04 needs time between measurements or the previous echo is re-detected.
SETTLE_S = 0.06


class GpioBackend(Protocol):
    """Minimal GPIO surface needed to drive an HC-SR04."""

    def setup_output(self, pin: int) -> None: ...
    def setup_input(self, pin: int) -> None: ...
    def write(self, pin: int, value: bool) -> None: ...
    def pulse_width_us(self, pin: int, timeout_s: float = 0.04) -> float: ...
    def cleanup(self) -> None: ...


class RpiGpioBackend:
    """Real GPIO backend. Uses rpi-lgpio, then libgpiod, then RPi.GPIO.

    Raises RuntimeError when no GPIO library is available so the caller can
    report "no sensor" honestly instead of pretending to measure.
    """

    def __init__(self):
        self._impl = None
        self._name = ""
        try:
            import RPi.GPIO as gpio  # type: ignore
            self._impl, self._name = gpio, "RPi.GPIO"
        except Exception:
            try:
                from rpi_lgpio import lgpio  # type: ignore
                lgpio.get_gpio()
                self._impl, self._name = lgpio, "rpi-lgpio"
            except Exception:
                try:
                    import gpiod  # type: ignore
                    self._impl, self._name = gpiod, "gpiod"
                except Exception as e:
                    raise RuntimeError(
                        "no GPIO backend available (install rpi-lgpio or python3-rpi.gpio)"
                    ) from e

    def setup_output(self, pin: int) -> None:
        if self._name == "RPi.GPIO":
            self._impl.setup(pin, self._impl.OUT)
        elif self._name == "rpi-lgpio":
            self._impl.setup_output(pin)
        else:
            line = self._impl.get_line(pin)
            self._impl.line_request(line, self._impl.LINE_REQ_DIR_OUT, 0)

    def setup_input(self, pin: int) -> None:
        if self._name == "RPi.GPIO":
            self._impl.setup(pin, self._impl.IN)
        elif self._name == "rpi-lgpio":
            self._impl.setup_input(pin)
        else:
            line = self._impl.get_line(pin)
            self._impl.line_request(line, self._impl.LINE_REQ_DIR_IN, 0)

    def write(self, pin: int, value: bool) -> None:
        if self._name == "RPi.GPIO":
            self._impl.output(pin, self._impl.HIGH if value else self._impl.LOW)
        else:
            self._impl.output(pin, 1 if value else 0)

    def pulse_width_us(self, pin: int, timeout_s: float = 0.04) -> float:
        if self._name == "RPi.GPIO":
            deadline = time.monotonic() + timeout_s
            while not self._impl.input(pin):
                if time.monotonic() > deadline:
                    return 0.0
                time.sleep(0.000001)
            start = time.monotonic()
            while self._impl.input(pin):
                if time.monotonic() - start > timeout_s:
                    return 0.0
            return (time.monotonic() - start) * 1e6
        if self._name == "rpi-lgpio":
            return float(self._impl.pulse_in(pin, timeout_s)) or 0.0
        line = self._impl.get_line(pin)
        return float(self._impl.edge_event(line, self._impl.Edge.FALLING, timeout_s)) or 0.0

    def cleanup(self) -> None:
        try:
            if self._name == "RPi.GPIO" or self._name == "rpi-lgpio":
                self._impl.cleanup()
        except Exception:  # noqa: BLE001
            pass


class SimulatedUltrasonic:
    """Deterministic backend for tests and for bench demos without hardware."""

    def __init__(self, distance_m: float = 1.0):
        self.distance_m = distance_m
        self.pins: dict[int, str] = {}
        self.writes: list[tuple[int, bool]] = []

    def setup_output(self, pin: int) -> None:
        self.pins[pin] = "out"

    def setup_input(self, pin: int) -> None:
        self.pins[pin] = "in"

    def write(self, pin: int, value: bool) -> None:
        self.writes.append((pin, value))

    def pulse_width_us(self, pin: int, timeout_s: float = 0.04) -> float:
        return self.distance_m * 100.0 * US_PER_CM

    def cleanup(self) -> None:
        self.pins.clear()


class HcSr04:
    """One HC-SR04 on a trigger/echo pin pair."""

    def __init__(self, trig_pin: int, echo_pin: int, backend: GpioBackend | None = None,
                 samples: int = 3, name: str = "ultrasonic"):
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.name = name
        self.samples = max(1, int(samples))
        self.backend = backend if backend is not None else RpiGpioBackend()
        self.backend.setup_output(trig_pin)
        self.backend.setup_input(echo_pin)
        self.last_error: str | None = None

    def measure_m(self) -> float | None:
        """Median-filtered distance in metres, or None when nothing is valid."""
        readings: list[float] = []
        for _ in range(self.samples):
            # 10 us trigger pulse, then wait for the echo.
            self.backend.write(self.trig_pin, False)
            time.sleep(0.000001)
            self.backend.write(self.trig_pin, True)
            time.sleep(0.00001)
            self.backend.write(self.trig_pin, False)
            width_us = self.backend.pulse_width_us(self.echo_pin, timeout_s=0.04)
            if width_us and width_us > 0:
                metres = (width_us / US_PER_CM) / 100.0
                if MIN_RANGE_M <= metres <= MAX_RANGE_M:
                    readings.append(metres)
            time.sleep(SETTLE_S)
        if not readings:
            self.last_error = "no echo"
            return None
        self.last_error = None
        # Median kills the single-sample outliers HC-SR04 is known for.
        return statistics.median(readings)

    def get_distance_m(self) -> float | None:
        """Sensor-rig interface, shared with the serial rangefinder driver."""
        return self.measure_m()

    def close(self) -> None:
        try:
            self.backend.cleanup()
        except Exception:  # noqa: BLE001
            pass