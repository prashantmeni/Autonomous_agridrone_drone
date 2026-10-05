"""RTL helper."""
from __future__ import annotations
async def return_to_home(app) -> dict:
    from ..mavlink import commands as C
    import asyncio
    app.fsm.force(app.fsm.state.__class__.RETURN_HOME, "RTH")
    ok = await asyncio.to_thread(C.rtl, app.conn)
    return {"rtl": ok}
