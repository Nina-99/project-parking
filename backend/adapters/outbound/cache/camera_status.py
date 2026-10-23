"""Camera heartbeats live in Redis, not in Postgres.

They change every few seconds and are worthless after a restart; a table would
add writes and migrations for state that expires on its own.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import redis.asyncio as redis

KEY_PREFIX = "camera:"


@dataclass(frozen=True)
class CameraReading:
    camera_id: str
    gate: str | None
    last_seen: datetime | None
    dropped_frames: int


def _text(value: object) -> str | None:
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def _reading(key: object, fields: dict[object, object]) -> CameraReading:
    key_text = _text(key) or ""
    camera_id = key_text.removeprefix(KEY_PREFIX)
    raw_seen = _text(fields.get(b"last_seen") or fields.get("last_seen"))
    try:
        last_seen = datetime.fromisoformat(raw_seen) if raw_seen else None
    except ValueError:
        last_seen = None
    raw_dropped = _text(fields.get(b"dropped_frames") or fields.get("dropped_frames"))
    return CameraReading(
        camera_id=camera_id,
        gate=_text(fields.get(b"gate") or fields.get("gate")),
        last_seen=last_seen,
        dropped_frames=int(raw_dropped) if raw_dropped else 0,
    )


class RedisCameraStatus:
    def __init__(self, url: str) -> None:
        self._url = url

    async def list(self) -> list[CameraReading]:
        client = redis.Redis.from_url(
            self._url,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        try:
            readings = []
            async for key in client.scan_iter(f"{KEY_PREFIX}*"):
                readings.append(_reading(key, await client.hgetall(key)))
            return sorted(readings, key=lambda reading: reading.camera_id)
        finally:
            await client.aclose()
