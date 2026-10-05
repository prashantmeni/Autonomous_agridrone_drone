"""Lightweight async event bus."""
from __future__ import annotations
import asyncio
from collections import defaultdict
from typing import Callable

class EventBus:
    def __init__(self):
        self._subs: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, topic: str, fn: Callable):
        self._subs[topic].append(fn)

    async def publish(self, topic: str, payload: dict | None = None):
        for fn in list(self._subs.get(topic, [])) + list(self._subs.get("*", [])):
            if asyncio.iscoroutinefunction(fn):
                await fn(topic, payload or {})
            else:
                fn(topic, payload or {})

bus = EventBus()
