"""The telemetry loop must actually feed the telemetry store.

Regression guard: recv_match's first positional parameter is ``type``, so
passing ``{"blocking": False}`` positionally asks for messages whose type is a
dict. Nothing matches, the store never updates, and health reports the flight
controller as absent while it is streaming perfectly.
"""
import asyncio
from types import SimpleNamespace

from drone.core.lifecycle import DroneApp


class FakeMaster:
    """Records how recv_match was called and yields one message."""

    def __init__(self, msg):
        self.msg = msg
        self.calls = []

    def recv_match(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.msg


def _app(master):
    app = DroneApp.__new__(DroneApp)          # bypass __init__ and its I/O
    updated = []

    store = SimpleNamespace(
        snap=SimpleNamespace(to_dict=lambda: {"connected": True, "lat": 1.0}),
        update_from_msg=lambda m: updated.append(m),
    )
    app.conn = SimpleNamespace(master=master, connected=True)
    app.tstore = store
    app.fsm = SimpleNamespace(state=SimpleNamespace(value="DISARMED"))
    app.recorder = SimpleNamespace(record=lambda snap: None)
    app.cfg = SimpleNamespace(telemetry=SimpleNamespace(rate_hz=1000))
    return app, updated


def test_recv_match_is_called_with_blocking_as_keyword():
    """The dict form must not reappear: it silently matches nothing."""
    msg = object()
    master = FakeMaster(msg)
    app, updated = _app(master)

    async def run():
        task = asyncio.create_task(DroneApp._telemetry_loop(app))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())

    assert master.calls, "telemetry loop never called recv_match"
    for args, kwargs in master.calls:
        assert not args, f"recv_match must not take positional args, got {args}"
        assert kwargs.get("blocking") is False, f"blocking must be a keyword, got {kwargs}"
    assert updated, "telemetry store was never updated"


def test_store_receives_each_message():
    msg = object()
    master = FakeMaster(msg)
    app, updated = _app(master)

    async def run():
        task = asyncio.create_task(DroneApp._telemetry_loop(app))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())
    assert msg in updated
