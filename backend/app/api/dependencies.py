"""Dependencies shared by the write endpoints.

The factory is injected rather than the unit of work itself: the unit of work is
opened inside the thread that uses it, because a SQLAlchemy Session belongs to
the thread and event loop that created it.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import redis.asyncio as redis

from backend.adapters.outbound.bus.redis_publisher import RedisEventPublisher
from backend.adapters.outbound.cache.rate_limit import RedisRateLimiter
from backend.adapters.outbound.db.session import get_session_factory
from backend.adapters.outbound.db.unit_of_work import SqlAlchemyUnitOfWork
from backend.adapters.outbound.time import SystemClock
from backend.app.core.config import settings


def get_uow_factory() -> Callable[[], SqlAlchemyUnitOfWork]:
    return lambda: SqlAlchemyUnitOfWork(get_session_factory())


def get_publisher() -> RedisEventPublisher:
    return RedisEventPublisher(settings.CELERY_BROKER_URL, settings.EVENTS_CHANNEL)


def get_clock() -> SystemClock:
    return SystemClock()


@lru_cache
def get_rate_limiter() -> RedisRateLimiter:
    """One client for the process; FastAPI calls this per request."""
    return RedisRateLimiter(
        redis.Redis.from_url(
            settings.CELERY_BROKER_URL,
            socket_connect_timeout=1,
            socket_timeout=1,
        ),
        limit=settings.LOGIN_RATE_LIMIT,
        window_seconds=settings.LOGIN_RATE_WINDOW_SECONDS,
    )
