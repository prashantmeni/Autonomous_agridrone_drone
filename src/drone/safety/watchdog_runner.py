"""Watchdog runner: polls safety.watchdog.check, logs, never commands aircraft."""
from __future__ import annotations
import asyncio, logging
async def run(app, interval_s: float = 5.0):
    from .watchdog import check
    log = logging.getLogger("drone.watchdog")
    while True:
        r = check(app)
        if not r["ok"]:
            log.warning(f"WATCHDOG issues={r['issues']} (PX4 failsafes authoritative; no auto-restart of flight)")
            try: app.db.log_event("WARNING", "watchdog", "ISSUES", {"issues": r["issues"]})
            except Exception: pass
        await asyncio.sleep(interval_s)
