# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""FastAPI entry point."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.adapters.inbound.http.realtime import relay_events
from backend.app.api.auth import require_operator
from backend.app.api.auth import router as auth_router
from backend.app.api.errors import install_error_handlers
from backend.app.api.health import router as health_router
from backend.app.api.routes import router as api_router
from backend.app.core.config import settings
from backend.app.core.database import engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Start the live relay. The schema is Alembic's job, not the app's."""
    # The worker publishes gate events on Redis; without this task the dashboard
    # stays frozen until the next poll. It reconnects on its own.
    relay_task = asyncio.create_task(
        relay_events(settings.CELERY_BROKER_URL, settings.EVENTS_CHANNEL)
    )

    yield

    relay_task.cancel()
    with suppress(asyncio.CancelledError):
        await relay_task
    await engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    contact={"name": "Nina"},
    license_info={"name": "Proprietary"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count"],
)

install_error_handlers(app)

app.include_router(health_router)
app.include_router(auth_router, prefix="/api/v1")
app.include_router(
    api_router,
    prefix="/api/v1",
    dependencies=[Depends(require_operator)],
)
