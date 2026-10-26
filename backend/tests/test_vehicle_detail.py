# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""One vehicle with its sessions, for the incident detail view."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient

from backend.app.api.auth import require_operator
from backend.app.core.database import get_session
from backend.app.main import app
from backend.app.models.parking_session import SessionStatus

NOW = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)


class _Result:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def scalars(self) -> _Result:
        return self

    def all(self) -> list:
        return self._rows


class _Session:
    def __init__(self, vehicle, sessions: list) -> None:
        self._vehicle = vehicle
        self._sessions = sessions

    async def get(self, _model, _ident):
        return self._vehicle

    async def execute(self, _statement):
        return _Result(self._sessions)


def _vehicle() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        license_plate="1234ABC",
        vehicle_type="car",
        vehicle_type_confidence=0.9,
        color="plata",
        color_confidence=0.8,
        entry_image_path="/tmp/crop.jpg",
        created_at=NOW,
    )


def _session() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        entry_gate="gate-1",
        exit_gate=None,
        entry_time=NOW,
        exit_time=None,
        duration_seconds=None,
        status=SessionStatus.ACTIVE,
        close_reason=None,
        closed_by=None,
    )


@asynccontextmanager
async def _client(vehicle, sessions: list):
    app.dependency_overrides[get_session] = lambda: _Session(vehicle, sessions)
    app.dependency_overrides[require_operator] = lambda: "tester"
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


async def test_the_detail_returns_the_vehicle_and_its_sessions() -> None:
    vehicle = _vehicle()
    async with _client(vehicle, [_session()]) as client:
        response = await client.get(f"/api/v1/vehicles/{vehicle.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["license_plate"] == "1234ABC"
    assert body["image_url"] == f"/api/v1/vehicles/{vehicle.id}/image"
    assert body["sessions"][0]["status"] == "active"


async def test_an_unknown_vehicle_is_404() -> None:
    async with _client(None, []) as client:
        response = await client.get(f"/api/v1/vehicles/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "vehicle not found"}
