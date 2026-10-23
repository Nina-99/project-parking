"""WebSocket endpoint for live occupancy events.

The worker publishes to Redis; this socket keeps the client connection open and
sends a heartbeat. Message types: ``vehicle_entry``, ``vehicle_exit``,
``occupancy_update``, ``ping``.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    """Keeps the open sockets and broadcasts to them."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        """Send a JSON message to all connected clients."""
        dead: list[WebSocket] = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead.append(connection)
        for conn in dead:
            self.disconnect(conn)


manager = ConnectionManager()


@router.websocket("/occupancy")
async def occupancy_websocket(websocket: WebSocket):
    """Hold the connection open and send a heartbeat every 30 s."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.send_json({"type": "ping", "timestamp": str(datetime.now())})
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
