# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Live fan-out to the browsers.

The API does not produce gate events; the worker does. This module keeps the
open sockets, forwards what Redis delivers, and notices dead clients without
waiting for the next event.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from fastapi import WebSocket

from backend.adapters.outbound.bus.redis_subscriber import relay

HEARTBEAT_SECONDS = 30


@dataclass
class _Client:
    websocket: WebSocket
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send(self, message: dict[str, object]) -> None:
        # Broadcasts and heartbeats can overlap; one writer at a time per socket.
        async with self.lock:
            await self.websocket.send_json(message)


class ConnectionManager:
    def __init__(self) -> None:
        self._clients: list[_Client] = []

    @property
    def count(self) -> int:
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> _Client:
        await websocket.accept()
        client = _Client(websocket)
        self._clients.append(client)
        return client

    def disconnect(self, client: _Client) -> None:
        if client in self._clients:
            self._clients.remove(client)

    async def broadcast(self, message: dict[str, object]) -> None:
        dead: list[_Client] = []
        for client in list(self._clients):
            try:
                await client.send(message)
            except Exception:
                dead.append(client)
        for client in dead:
            self.disconnect(client)


manager = ConnectionManager()


async def run_websocket(websocket: WebSocket) -> None:
    """Hold one browser connection open, with heartbeats and clean teardown."""
    client = await manager.connect(websocket)
    gone = asyncio.Event()

    async def heartbeat() -> None:
        while not gone.is_set():
            try:
                await asyncio.wait_for(gone.wait(), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                try:
                    await client.send({
                        "type": "ping",
                        "timestamp": datetime.now(UTC).isoformat(),
                    })
                except Exception:
                    gone.set()

    async def watch() -> None:
        try:
            while True:
                await websocket.receive_text()
        except Exception:
            gone.set()

    tasks = [asyncio.create_task(heartbeat()), asyncio.create_task(watch())]
    try:
        await gone.wait()
    finally:
        for task in tasks:
            task.cancel()
        manager.disconnect(client)


async def relay_events(url: str, channel: str) -> None:
    """Subscribe to the worker's channel and broadcast to every client."""
    await relay(url, manager.broadcast, channel)
