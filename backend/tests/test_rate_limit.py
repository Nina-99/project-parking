# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The login limiter counts per key and fails open on a broker outage."""
from __future__ import annotations

import asyncio

from backend.adapters.outbound.cache.rate_limit import RedisRateLimiter


class _Client:
    def __init__(self, counts: list[int], fails: bool = False) -> None:
        self._counts = iter(counts)
        self.fails = fails
        self.expired: list[tuple[str, int]] = []

    async def incr(self, _key: str) -> int:
        if self.fails:
            raise ConnectionError("broker down")
        return next(self._counts)

    async def expire(self, key: str, window: int) -> None:
        if self.fails:
            raise ConnectionError("broker down")
        self.expired.append((key, window))


def test_the_first_attempt_starts_the_window() -> None:
    client = _Client([1])
    limiter = RedisRateLimiter(client, limit=2, window_seconds=300)

    assert asyncio.run(limiter.allow("rate:login:1.2.3.4")) is True
    assert client.expired == [("rate:login:1.2.3.4", 300)]


def test_the_limit_is_enforced_inside_the_window() -> None:
    limiter = RedisRateLimiter(_Client([2, 3]), limit=2, window_seconds=300)

    assert asyncio.run(limiter.allow("k")) is True
    assert asyncio.run(limiter.allow("k")) is False


def test_a_broker_outage_does_not_lock_the_operator_out() -> None:
    limiter = RedisRateLimiter(_Client([1], fails=True), limit=1, window_seconds=300)

    assert asyncio.run(limiter.allow("k")) is True


def test_a_zero_limit_disables_the_check() -> None:
    limiter = RedisRateLimiter(_Client([]), limit=0, window_seconds=300)

    assert asyncio.run(limiter.allow("k")) is True
