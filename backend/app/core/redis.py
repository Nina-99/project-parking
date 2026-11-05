# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""
Redis connection manager for Celery result backend and pub/sub.
"""
from __future__ import annotations

import redis.asyncio as aioredis

from backend.app.core.config import settings

redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Get or create the Redis connection."""
    global redis_client
    if redis_client is None:
        redis_client = await aioredis.from_url(
            settings.CELERY_BROKER_URL,
            decode_responses=True,
        )
    return redis_client


async def close_redis() -> None:
    """Close Redis connection gracefully."""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
