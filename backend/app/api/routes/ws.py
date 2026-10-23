"""WebSocket endpoint for live occupancy updates.

Message types: ``vehicle_entry``, ``vehicle_exit``, ``vehicle_exit_unmatched``,
``occupancy_update`` and ``ping``. The socket is mounted under ``/api/v1/ws``,
which is what the frontend and the reverse proxy must use.
"""
from __future__ import annotations

from fastapi import APIRouter, WebSocket

from backend.adapters.inbound.http.realtime import run_websocket

router = APIRouter()


@router.websocket("/occupancy")
async def occupancy_websocket(websocket: WebSocket) -> None:
    await run_websocket(websocket)
