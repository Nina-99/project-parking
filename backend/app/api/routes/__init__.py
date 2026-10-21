"""Mount all API routers."""
from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.routes.entry import router as entry_router
from backend.app.api.routes.exit import router as exit_router
from backend.app.api.routes.stats import router as stats_router
from backend.app.api.routes.vehicles import router as vehicles_router
from backend.app.api.routes.ws import router as ws_router

router = APIRouter()
router.include_router(entry_router, prefix="/entry", tags=["entry"])
router.include_router(exit_router, prefix="/exit", tags=["exit"])
router.include_router(stats_router, prefix="/stats", tags=["stats"])
router.include_router(vehicles_router, prefix="/vehicles", tags=["vehicles"])
router.include_router(ws_router, prefix="/ws", tags=["websocket"])
