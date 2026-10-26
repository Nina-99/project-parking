# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Liveness and readiness probes.

Liveness answers "is the process up". Readiness answers "can it serve": Postgres
reachable and Redis reachable. A process that can reach neither should be
restarted or removed from rotation, not reported healthy.
"""
from __future__ import annotations

import asyncio

import redis.asyncio as redis
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from backend.app.core.config import settings
from backend.app.core.database import engine

router = APIRouter(tags=["health"])

PROBE_TIMEOUT_SECONDS = 2.0


async def database_ok() -> bool:
    try:
        async with engine.connect() as connection:
            await asyncio.wait_for(
                connection.execute(text("SELECT 1")),
                timeout=PROBE_TIMEOUT_SECONDS,
            )
        return True
    except Exception:
        return False


async def redis_ok() -> bool:
    client = redis.Redis.from_url(
        settings.CELERY_BROKER_URL,
        socket_connect_timeout=1,
        socket_timeout=1,
    )
    try:
        return bool(await asyncio.wait_for(client.ping(), timeout=PROBE_TIMEOUT_SECONDS))
    except Exception:
        return False
    finally:
        await client.aclose()


@router.get("/health")
async def health() -> dict[str, str]:
    """Backwards-compatible liveness: the process answered."""
    return {"status": "ok", "version": settings.VERSION}


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready")
async def ready() -> JSONResponse:
    database, cache = await asyncio.gather(database_ok(), redis_ok())
    healthy = database and cache
    return JSONResponse(
        {
            "status": "ready" if healthy else "degraded",
            "database": database,
            "redis": cache,
        },
        status_code=200 if healthy else 503,
    )
