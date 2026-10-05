"""MAVLink connection with reconnect. Supports serial/udp/tcp. Never hard-codes device."""
from __future__ import annotations
import asyncio, logging, time
from collections import deque
from typing import Any

log = logging.getLogger("drone.mavlink")

class MavlinkConnection:
    def __init__(self, connection_string: str, baud: int = 115200,
                 heartbeat_timeout_s: float = 5.0, reconnect_delay_s: float = 2.0):
        self.connection_string = connection_string
        self.baud = baud
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.reconnect_delay_s = reconnect_delay_s
        self._master: Any = None
        self.connected = False
        self.last_heartbeat = 0.0
        self._cmd_seq = 0
        self.px4_status: deque = deque(maxlen=20)  # (ts, text) — PX4 STATUSTEXT history

    def connect(self) -> bool:
        from pymavlink import mavutil
        s = self.connection_string
        try:
            if s.startswith(("udp:", "tcp:")):
                self._master = mavutil.mavlink_connection(s)
            else:
                self._master = mavutil.mavlink_connection(s, baud=self.baud)
            self._install_status_logger()
            log.info(f"MAVLink connecting: {s}")
            return True
        except Exception as e:
            log.error(f"MAVLink connect failed {s}: {e}")
            return False

    def _install_status_logger(self):
        """Permanently log every PX4 STATUSTEXT (prearm denials, estimator warnings...).

        Installed on the raw connection so it fires no matter which loop drains
        the message off the wire, and survives outside command wait windows.
        """
        master = self._master
        hooks = getattr(master, "message_hooks", None) if master is not None else None
        if hooks is None:
            return

        def hook(*args):
            m = args[-1]  # pymavlink calls hook(self, msg)
            try:
                if m.get_type() != "STATUSTEXT":
                    return
                text = str(m.text).rstrip("\x00")
            except Exception:
                return
            if not text:
                return
            self.px4_status.append((time.time(), text))
            log.warning(f"PX4: {text}")

        hooks.append(hook)

    def wait_heartbeat(self, timeout: float = 10.0) -> bool:
        try:
            m = self._master.wait_heartbeat(timeout=timeout)
            if m is not None:
                self.connected = True
                self.last_heartbeat = time.time()
                return True
        except Exception as e:
            log.error(f"heartbeat wait failed: {e}")
        return False

    async def maintain(self):
        """Background reconnect loop. Never issues duplicate dangerous commands."""
        while True:
            if self._master is None:
                self.connect()
                await asyncio.sleep(self.reconnect_delay_s)
                continue
            try:
                msg = self._master.recv_match(blocking=False)
                if msg is not None and msg.get_type() == "HEARTBEAT":
                    self.last_heartbeat = time.time()
                    if not self.connected:
                        self.connected = True
                        log.info("MAVLink heartbeat restored - resynchronizing telemetry")
                if self.connected and time.time() - self.last_heartbeat > self.heartbeat_timeout_s:
                    self.connected = False
                    log.warning("MAVLINK_RECONNECTING: heartbeat lost")
            except Exception as e:
                self.connected = False
                log.error(f"MAVLink loop error: {e}")
            await asyncio.sleep(0.2)

    @property
    def master(self): return self._master
    @property
    def target_system(self): return getattr(self._master, "target_system", 1) if self._master else 1
    @property
    def target_component(self): return getattr(self._master, "target_component", 1) if self._master else 1
