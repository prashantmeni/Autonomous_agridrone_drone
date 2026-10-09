"""App lifecycle: wires config, db, mavlink, telemetry loop, api."""
from __future__ import annotations
import asyncio, logging
from functools import partial
from ..core.config import AppConfig
from ..mavlink.connection import MavlinkConnection
from ..mavlink.telemetry import TelemetryStore
from ..telemetry.recorder import TelemetryRecorder
from ..telemetry import publisher as pub
from ..db.store import Database
from ..core.state import FlightStateMachine

log = logging.getLogger("drone.lifecycle")

class DroneApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.db = Database(cfg.database.path)
        self.conn = MavlinkConnection(cfg.mavlink.connection, cfg.mavlink.baud,
            cfg.mavlink.heartbeat_timeout_s, cfg.mavlink.reconnect_delay_s)
        self.tstore = TelemetryStore()
        self.recorder = TelemetryRecorder(cfg.telemetry.log_dir, cfg.telemetry.max_log_mb)
        self.fsm = FlightStateMachine()
        self._tasks: list[asyncio.Task] = []

    async def start(self):
        await asyncio.to_thread(self.conn.connect)
        await asyncio.to_thread(self.conn.wait_heartbeat, 8.0)
        self._tasks = [asyncio.create_task(self.conn.maintain()), asyncio.create_task(self._telemetry_loop())]
        if self.cfg.disease_detection.enabled:
            from ..perception.crop_monitor import CropAIMonitor
            monitor = CropAIMonitor(self)
            self._tasks.append(asyncio.create_task(monitor.run()))
            log.info("crop monitor started (interval=%.1fs model=%s)",
                     self.cfg.disease_detection.inference_interval_s,
                     self.cfg.disease_detection.model_path)
        log.info(f"DroneApp started env={self.cfg.environment} connected={self.conn.connected}")

    async def _telemetry_loop(self):
        rate = max(1.0, self.cfg.telemetry.rate_hz)
        while True:
            try:
                if self.conn.master is not None:
                    # blocking must be passed as a keyword: recv_match's first
                    # positional parameter is `type`, so passing a dict here
                    # asks for messages whose type is a dict, matches nothing,
                    # and leaves telemetry permanently empty.
                    msg = await asyncio.to_thread(
                        partial(self.conn.master.recv_match, blocking=False))
                    if msg is not None:
                        self.tstore.update_from_msg(msg)
                snap = self.tstore.snap.to_dict()
                snap["fsm_state"] = self.fsm.state.value
                self.recorder.record(snap)
                await pub.publisher.broadcast({"type": "telemetry", "data": snap})
            except Exception as e:
                log.error(f"telemetry loop: {e}")
            await asyncio.sleep(1.0 / rate)

    async def stop(self):
        for t in self._tasks: t.cancel()
