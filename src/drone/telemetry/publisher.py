"""Broadcast hub for WebSocket telemetry."""
from __future__ import annotations
import asyncio
class Publisher:
    def __init__(self): self._queues: set[asyncio.Queue] = set()
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._queues.add(q); return q
    def unsubscribe(self, q): self._queues.discard(q)
    async def broadcast(self, msg: dict):
        for q in list(self._queues):
            try: q.put_nowait(msg)
            except asyncio.QueueFull:
                try: q.get_nowait(); q.put_nowait(msg)
                except Exception: pass
publisher = Publisher()
