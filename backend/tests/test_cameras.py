# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Camera liveness: a heartbeat that stops is the only offline signal."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from backend.adapters.outbound.cache.camera_status import CameraReading
from backend.app.api.auth import require_operator
from backend.app.api.routes.cameras import get_camera_status
from backend.app.core.database import get_session
from backend.app.main import app
from backend.domain.cameras import is_online

NOW = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)


def test_a_recent_heartbeat_is_online() -> None:
    assert is_online(NOW - timedelta(seconds=5), NOW, 20)


def test_an_old_heartbeat_is_offline() -> None:
    assert not is_online(NOW - timedelta(seconds=30), NOW, 20)


def test_no_heartbeat_is_offline() -> None:
    assert not is_online(None, NOW, 20)


def test_a_non_positive_window_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        is_online(NOW, NOW, 0)


class _Status:
    def __init__(self, readings: list[CameraReading] | None = None, fails: bool = False):
        self._readings = readings or []
        self._fails = fails

    async def list(self) -> list[CameraReading]:
        if self._fails:
            raise ConnectionError("broker down")
        return self._readings


class _EmptyResult:
    def __iter__(self):
        return iter(())


class _FakeSession:
    """Only what this route uses: the list of registered cameras.

    The registry merge is what is under test here, not SQL: a unit test with a
    real database belongs in the integration suite.
    """

    def __init__(self, cameras: list[object] | None = None) -> None:
        self._cameras = cameras or []

    async def scalars(self, _statement: object) -> _EmptyResult:
        return _Result(self._cameras)


class _Result:
    def __init__(self, items: list[object]) -> None:
        self._items = items

    def __iter__(self):
        return iter(self._items)


def _client(status: _Status, cameras: list[object] | None = None) -> AsyncClient:
    app.dependency_overrides[require_operator] = lambda: "tester"
    app.dependency_overrides[get_camera_status] = lambda: status
    app.dependency_overrides[get_session] = lambda: _FakeSession(cameras)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_the_route_reports_each_camera_and_its_state() -> None:
    now = datetime.now(UTC)
    status = _Status([
        CameraReading("gate-1", "entry", now, dropped_frames=2),
        CameraReading("gate-2", "exit", now - timedelta(hours=1), dropped_frames=0),
    ])
    async with _client(status) as client:
        response = await client.get("/api/v1/cameras")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["redis_available"] is True
    assert body["cameras"][0]["camera_id"] == "gate-1"
    assert body["cameras"][0]["online"] is True
    assert body["cameras"][0]["dropped_frames"] == 2
    assert body["cameras"][1]["online"] is False


async def test_a_redis_outage_hides_the_cameras_but_not_the_endpoint() -> None:
    async with _client(_Status(fails=True)) as client:
        response = await client.get("/api/v1/cameras")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"redis_available": False, "cameras": []}
