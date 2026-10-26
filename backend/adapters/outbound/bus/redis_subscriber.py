# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Async Redis subscription for the API process.

The worker is the only producer of gate events. The API subscribes once at
startup and fans the messages out to the browsers over WebSocket. The
subscription reconnects on its own: Redis restarting must not leave the
dashboard silently frozen.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

import redis.asyncio as redis

from backend.adapters.outbound.bus.redis_publisher import EVENTS_CHANNEL

logger = logging.getLogger(__name__)


async def stream_events(
    url: str,
    channel: str = EVENTS_CHANNEL,
) -> AsyncIterator[dict[str, object]]:
    """Yield the payloads published on ``channel`` until cancelled."""
    client = redis.Redis.from_url(
        url,
        socket_connect_timeout=2,
        health_check_interval=30,
    )
    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(channel)
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            try:
                payload = json.loads(message["data"])
            except (TypeError, ValueError):
                logger.warning("dropping a non-JSON event on %s", channel)
                continue
            if isinstance(payload, dict):
                yield payload
    finally:
        await pubsub.aclose()
        await client.aclose()


async def relay(
    url: str,
    broadcast: Callable[[dict[str, object]], Awaitable[None]],
    channel: str = EVENTS_CHANNEL,
    retry_seconds: float = 2.0,
) -> None:
    """Forward every event to ``broadcast`` and reconnect after a failure."""
    while True:
        try:
            async for event in stream_events(url, channel):
                await broadcast(event)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # broker down, DNS, auth
            logger.warning("event relay lost the broker (%s); retrying", error)
            await asyncio.sleep(retry_seconds)
