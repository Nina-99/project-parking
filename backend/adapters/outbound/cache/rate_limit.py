"""Fixed-window login rate limit in Redis.

Fail-open on purpose: a broker outage must not lock the operator out of the
system they are supposed to monitor. The window is per client IP.
"""
from __future__ import annotations

import redis.asyncio as redis


class RedisRateLimiter:
    def __init__(self, client: redis.Redis, limit: int, window_seconds: int) -> None:
        self._client = client
        self._limit = limit
        self._window = window_seconds

    async def allow(self, key: str) -> bool:
        if self._limit <= 0:
            return True
        try:
            count = await self._client.incr(key)
            if count == 1:
                await self._client.expire(key, self._window)
            return bool(count <= self._limit)
        except Exception:
            return True
