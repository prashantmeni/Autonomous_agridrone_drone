"""JSONL telemetry recorder (rotation by size)."""
from __future__ import annotations
import json, time
from pathlib import Path

class TelemetryRecorder:
    def __init__(self, log_dir: str = "data/logs", max_mb: float = 50.0):
        self.dir = Path(log_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_mb * 1e6)
        self._f = None; self._path = None
        self._open_new()

    def _open_new(self):
        self._path = self.dir / f"telemetry_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self._f = open(self._path, "a")

    def record(self, snap: dict):
        line = json.dumps({"ts": time.time(), **snap}) + "\n"
        if self._path.stat().st_size + len(line) > self.max_bytes:
            self._f.close(); self._open_new()
        self._f.write(line); self._f.flush()
