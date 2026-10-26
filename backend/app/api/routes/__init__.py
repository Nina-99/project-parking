# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""API router: one prefix per resource.

Ingestion is not an HTTP endpoint. Frames arrive through ``capture.py`` and
Celery, and both end in ``application.register_observation``. Adding an upload
route back would create a second write path with different semantics, which is
exactly what the old ``/entry`` and ``/exit`` routes did before they were
removed.
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.routes.cameras import router as cameras_router
from backend.app.api.routes.sessions import router as sessions_router
from backend.app.api.routes.stats import router as stats_router
from backend.app.api.routes.vehicles import router as vehicles_router
from backend.app.api.routes.ws import router as ws_router

router = APIRouter()
router.include_router(cameras_router, prefix="/cameras", tags=["cameras"])
router.include_router(sessions_router, prefix="/sessions", tags=["sessions"])
router.include_router(stats_router, prefix="/stats", tags=["stats"])
router.include_router(vehicles_router, prefix="/vehicles", tags=["vehicles"])
router.include_router(ws_router, prefix="/ws", tags=["websocket"])
