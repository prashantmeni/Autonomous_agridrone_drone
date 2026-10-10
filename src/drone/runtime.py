"""Lazily-created runtime services shared by API endpoints.

Everything attaches to the DroneApp instance on first use so tests and
`--no-api` runs never open camera devices or serial ports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .autonomy.precision_landing import PrecisionLandingTask
from .perception.charging import WirelessChargingStation
from .perception.gimbal import PanTiltController
from .perception.proximity import ProximityRig
from .perception.stream import CameraService


@dataclass
class Services:
    camera: CameraService
    gimbal: PanTiltController
    charging: WirelessChargingStation
    proximity: ProximityRig
    precision: PrecisionLandingTask
    started: bool = field(default=False)

    def start(self) -> None:
        if not self.started:
            self.camera.start()
            self.started = True


def get_services(app) -> Services:
    svc = getattr(app, "_services", None)
    if svc is None:
        cfg = app.cfg
        # Rangefinders come from config (obstacle_avoidance.sensors). This used
        # to be a hardcoded empty dict, which made every distance-sensing driver
        # in the repo unreachable no matter what hardware was fitted.
        sensor_ports = {}
        try:
            from .core.behaviour import load_behaviour
            load_behaviour()  # ensure file exists with defaults
        except Exception:
            pass
        svc = Services(
            camera=CameraService(cfg.camera),
            gimbal=PanTiltController(),
            charging=WirelessChargingStation(),
            proximity=ProximityRig(cfg.obstacle_avoidance, sensor_ports),
            precision=PrecisionLandingTask(),
        )
        app._services = svc
    return svc
