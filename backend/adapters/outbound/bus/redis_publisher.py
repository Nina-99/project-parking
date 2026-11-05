# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Redis implementation of the event publisher.

A broker outage must not undo a committed row, so a failed publish returns False
and the caller keeps the data. The dashboard will catch up on the next event.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import redis

EVENTS_CHANNEL = "parking:events"


class RedisEventPublisher:
    def __init__(self, url: str, channel: str = EVENTS_CHANNEL) -> None:
        self._url = url
        self._channel = channel

    def publish(self, payload: Mapping[str, object]) -> bool:
        client = None
        try:
            client = redis.Redis.from_url(
                self._url,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.publish(self._channel, json.dumps(dict(payload)))
            return True
        except Exception:
            return False
        finally:
            if client is not None:
                client.close()
