"""Structured logging with flight_id context."""
from __future__ import annotations
import logging, sys, json
from logging.handlers import RotatingFileHandler
from pathlib import Path

_current_flight = {"id": "no-flight"}

def set_flight_id(fid: str): _current_flight["id"] = fid

class FlightFilter(logging.Filter):
    def filter(self, record):
        record.flight_id = _current_flight["id"]
        return True

def setup_logging(level: str = "INFO", log_dir: str = "data/logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("drone")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        fmt = logging.Formatter('%(asctime)s %(levelname)s [%(flight_id)s] %(name)s %(message)s')
        sh = logging.StreamHandler(sys.stdout)
        sh.addFilter(FlightFilter()); sh.setFormatter(fmt)
        fh = RotatingFileHandler(f"{log_dir}/drone.log", maxBytes=5_000_000, backupCount=5)
        fh.addFilter(FlightFilter()); fh.setFormatter(fmt)
        root.addHandler(sh); root.addHandler(fh)
    return root

def event_log(logger: logging.Logger, level: str, component: str, event: str, **kw):
    payload = {"component": component, "event": event, **kw}
    getattr(logger, level.lower())(json.dumps(payload))
