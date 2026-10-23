"""The socket is part of the API: no valid cookie, no connection."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket

from backend.adapters.inbound.http.realtime import run_websocket
from backend.app.api.auth import verify_token
from backend.app.core.config import settings

router = APIRouter()


@router.websocket("/occupancy")
async def occupancy_websocket(websocket: WebSocket) -> None:
    token = websocket.cookies.get(settings.AUTH_COOKIE_NAME)
    try:
        verify_token(token)
    except HTTPException:
        await websocket.close(code=1008)
        return
    await run_websocket(websocket)
