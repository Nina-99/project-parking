# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The validation queue route: what it returns and what it records.

No database: the session dependency is a recorder, so the test also asserts the
query shape that makes the queue possible (one row per unreviewed vehicle).
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.api.auth import require_operator
from backend.app.api.dependencies import get_clock, get_uow_factory
from backend.app.api.routes.validations import operator_only
from backend.app.core.database import get_session
from backend.app.main import app
from backend.tests.fakes import FakeUnitOfWork, FixedClock

NOW = datetime(2026, 5, 4, 10, 0, tzinfo=UTC)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Recorder:
    def __init__(self, rows):
        self.rows = rows
        self.statements: list = []

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result(self.rows)


def _vehicle(plate: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        license_plate=plate,
        license_plate_confidence=0.9,
        vehicle_type="car",
        vehicle_type_confidence=0.8,
        color="plata",
        color_confidence=0.7,
        entry_image_path="/entries/crop.jpg",
        plate_characters=None,
        created_at=NOW,
    )


@pytest.fixture
def recorder() -> _Recorder:
    return _Recorder([])


@pytest.fixture
def uow() -> FakeUnitOfWork:
    return FakeUnitOfWork()


@pytest.fixture
async def client(recorder: _Recorder, uow: FakeUnitOfWork):
    app.dependency_overrides[require_operator] = lambda: "ana"
    app.dependency_overrides[operator_only] = lambda: ("ana", "operator")
    app.dependency_overrides[get_session] = lambda: recorder
    app.dependency_overrides[get_uow_factory] = lambda: (lambda: uow)
    app.dependency_overrides[get_clock] = lambda: FixedClock()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_pending_events_carry_direction_and_whitelist_state(
    client: AsyncClient, recorder: _Recorder,
) -> None:
    vehicle = _vehicle("ABC123")
    session_id = uuid.uuid4()
    recorder.rows = [(vehicle, "entry", "gate-1", session_id, NOW, True)]

    response = await client.get("/api/v1/validations/pending")

    assert response.status_code == 200
    [item] = response.json()
    assert item["vehicle_id"] == str(vehicle.id)
    assert item["direction"] == "entry"
    assert item["camera_id"] == "gate-1"
    assert item["authorized"] is True
    assert item["session_id"] == str(session_id)
    assert item["image_url"] == f"/api/v1/vehicles/{vehicle.id}/image"


async def test_a_vehicle_without_a_plate_has_no_authorization_verdict(
    client: AsyncClient, recorder: _Recorder,
) -> None:
    recorder.rows = [(_vehicle(None), "exit", "gate-2", None, NOW, False)]

    response = await client.get("/api/v1/validations/pending")

    assert response.json()[0]["authorized"] is None, (
        "unknown and unauthorized are different answers"
    )


async def test_confirming_returns_the_audited_outcome(
    client: AsyncClient, uow: FakeUnitOfWork,
) -> None:
    vehicle = _vehicle("ABC123")
    uow.vehicles.plates[str(vehicle.id)] = "ABC123"

    response = await client.post(
        f"/api/v1/validations/{vehicle.id}/decision",
        json={"decision": "confirmed"},
    )

    assert response.status_code == 200
    assert response.json()["operator"] == "ana"
    assert uow.validations.added[0]["decision"] == "confirmed"


async def test_a_plate_correction_reaches_the_vehicle_store(
    client: AsyncClient, uow: FakeUnitOfWork,
) -> None:
    vehicle = _vehicle("ABD123")
    uow.vehicles.plates[str(vehicle.id)] = "ABD123"

    response = await client.post(
        f"/api/v1/validations/{vehicle.id}/decision",
        json={"decision": "plate_corrected", "corrected_plate": "ABC-123"},
    )

    assert response.status_code == 200
    assert uow.vehicles.corrected == [(str(vehicle.id), "ABC123")]
    assert response.json()["corrected_plate"] == "ABC123"


async def test_a_decision_outside_the_domain_is_422(client: AsyncClient) -> None:
    response = await client.post(
        f"/api/v1/validations/{uuid.uuid4()}/decision",
        json={"decision": "approved"},
    )

    assert response.status_code == 422


async def test_a_missing_vehicle_is_404(client: AsyncClient, uow: FakeUnitOfWork) -> None:
    vehicle = _vehicle("ABC123")
    uow.vehicles.missing.add(str(vehicle.id))

    response = await client.post(
        f"/api/v1/validations/{vehicle.id}/decision",
        json={"decision": "dismissed"},
    )

    assert response.status_code == 404
