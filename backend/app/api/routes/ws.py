"""
WebSocket endpoint for real-time occupancy updates.

Frontend se conecta acá y recibe eventos en vivo:
  - vehicle_entry
  - vehicle_exit
  - occupancy_update
"""
from __future__ import annotations

import json
import asyncio
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""

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
    """
    WebSocket endpoint for real-time parking occupancy.

    Usage from frontend:
      const ws = new WebSocket("ws://localhost:8000/api/v1/ws/occupancy");
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // data.type: "vehicle_entry" | "vehicle_exit" | "occupancy_update" | "ping"
      };
    """
    await manager.connect(websocket)
    try:
        # Send heartbeat every 30s
        while True:
            await websocket.send_json({"type": "ping", "timestamp": str(datetime.now())})
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
