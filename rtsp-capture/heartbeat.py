# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Heartbeat for one capture process.

The API answers "is this camera alive?" from the last beat, so a dead process
needs no shutdown hook: its key expires. A broker outage must never stop the
camera from sampling, so every write is best-effort.
"""
from __future__ import annotations

from datetime import UTC, datetime

import redis

KEY_PREFIX = "camera:"


class CameraHeartbeat:
    def __init__(
        self,
        client: object,
        camera_id: str,
        gate: str,
        ttl_seconds: int = 30,
    ) -> None:
        self._client = client
        self.camera_id = camera_id
        self.gate = gate
        self.ttl_seconds = ttl_seconds
        self.key = f"{KEY_PREFIX}{camera_id}"

    def beat(self, dropped_frames: int = 0, now: datetime | None = None) -> bool:
        stamp = (now or datetime.now(UTC)).isoformat()
        try:
            self._client.hset(  # type: ignore[attr-defined]
                self.key,
                mapping={
                    "camera_id": self.camera_id,
                    "gate": self.gate,
                    "last_seen": stamp,
                    "dropped_frames": str(dropped_frames),
                },
            )
            self._client.expire(self.key, self.ttl_seconds)  # type: ignore[attr-defined]
            return True
        except Exception:
            return False


def build_heartbeat(camera_id: str, gate: str) -> CameraHeartbeat:
    from backend.app.core.config import settings

    client = redis.Redis.from_url(
        settings.CELERY_BROKER_URL,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    return CameraHeartbeat(client, camera_id, gate)
