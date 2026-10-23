"""The heartbeat is best-effort: a broker outage must not stop the camera."""
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

RTSP_DIR = Path(__file__).resolve().parents[2] / "rtsp-capture"
if str(RTSP_DIR) not in sys.path:
    sys.path.insert(0, str(RTSP_DIR))

from heartbeat import CameraHeartbeat  # noqa: E402

NOW = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)


class _Client:
    def __init__(self, fails: bool = False) -> None:
        self.fails = fails
        self.hashes: list[tuple[str, dict]] = []
        self.expirations: list[tuple[str, int]] = []

    def hset(self, key: str, mapping: dict) -> None:
        if self.fails:
            raise ConnectionError("broker down")
        self.hashes.append((key, mapping))

    def expire(self, key: str, ttl: int) -> None:
        if self.fails:
            raise ConnectionError("broker down")
        self.expirations.append((key, ttl))


def test_a_beat_records_who_is_alive_and_expires_on_its_own() -> None:
    client = _Client()
    heartbeat = CameraHeartbeat(client, "gate-1", "entry", ttl_seconds=30)

    assert heartbeat.beat(dropped_frames=3, now=NOW) is True

    key, mapping = client.hashes[0]
    assert key == "camera:gate-1"
    assert mapping["gate"] == "entry"
    assert mapping["last_seen"] == NOW.isoformat()
    assert mapping["dropped_frames"] == "3"
    assert client.expirations == [("camera:gate-1", 30)]


def test_a_broker_failure_is_reported_but_never_raised() -> None:
    heartbeat = CameraHeartbeat(_Client(fails=True), "gate-1", "entry")

    assert heartbeat.beat() is False
