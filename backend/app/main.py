"""FastAPI entry point."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.adapters.inbound.http.realtime import relay_events
from backend.app.api.routes import router as api_router
from backend.app.core.config import settings
from backend.app.core.database import Base, engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create the tables on startup; dev shortcut until Alembic is wired in."""
    import backend.app.models  # noqa: F401  (registers the metadata)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

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
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.VERSION}
