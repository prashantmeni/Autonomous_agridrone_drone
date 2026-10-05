from __future__ import annotations
from fastapi import APIRouter, WebSocket
import asyncio
router = APIRouter()

@router.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    from ..telemetry.publisher import publisher
    await ws.accept()
    q = publisher.subscribe()
    try:
        while True:
            msg = await q.get()
            await ws.send_json(msg)
    except Exception:
        pass
    finally:
        publisher.unsubscribe(q)

@router.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "events", "note": "subscribe /ws/telemetry for live stream"})
    await ws.close()

@router.websocket("/ws/health")
async def ws_health(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"status": "ok"})
    await ws.close()

@router.websocket("/ws/mission")
async def ws_mission(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"status": "mission channel ready"})
    await ws.close()
