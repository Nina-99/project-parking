# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The incident-search route: filters, colour expansion and ordering.

No database here. The session dependency is replaced with a recorder, so the
test asserts what the route *asks* the database for as well as what it returns --
the colour expansion only matters if it reaches the SQL.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.api.auth import require_operator
from backend.app.core.database import get_session
from backend.app.main import app


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def scalar(self):
        return len(self._rows)


class _Recorder:
    def __init__(self, rows):
        self.rows = rows
        self.statements: list = []

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result(self.rows)


def _vehicle(color: str, minute: int, plate: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        license_plate=plate,
        vehicle_type="car",
        color=color,
        color_confidence=0.7,
        entry_image_path=None,
        created_at=datetime(2026, 5, 4, 10, minute, tzinfo=UTC),
    )


@pytest.fixture
def recorder():
    return _Recorder([])


@pytest.fixture
async def client(recorder):
    app.dependency_overrides[get_session] = lambda: recorder
    app.dependency_overrides[require_operator] = lambda: "tester"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_the_requested_colour_is_expanded_before_it_reaches_sql(
    client: AsyncClient, recorder: _Recorder,
) -> None:
    response = await client.get(
        "/api/v1/vehicles/search",
        params={"from": "2026-05-04T09:00:00Z", "to": "2026-05-04T11:00:00Z",
                "color": "plata"},
    )

    assert response.status_code == 200
    values = list(recorder.statements[0].compile().params.values())
    colour_param = next(value for value in values if isinstance(value, list))
    assert colour_param == ["plata", "gris", "blanco"], "the measured neighbour must reach the query"
    assert 0.3 in values, "the confidence floor must reach the query"


async def test_exact_matches_outrank_neighbours(client: AsyncClient, recorder: _Recorder) -> None:
    recorder.rows = [_vehicle("gris", 0, "NEIGHBOUR"), _vehicle("plata", 40, "EXACT")]

    response = await client.get(
        "/api/v1/vehicles/search",
        params={"from": "2026-05-04T09:00:00Z", "to": "2026-05-04T11:00:00Z",
                "color": "plata", "at": "2026-05-04T10:00:00Z"},
    )

    hits = response.json()
    assert [hit["license_plate"] for hit in hits] == ["EXACT", "NEIGHBOUR"]
    assert [hit["match"] for hit in hits] == ["exact", "neighbour"]


async def test_without_a_colour_nothing_is_filtered_by_colour(
    client: AsyncClient, recorder: _Recorder,
) -> None:
    recorder.rows = [_vehicle("azul", 1, "A")]

    response = await client.get(
        "/api/v1/vehicles/search",
        params={"from": "2026-05-04T09:00:00Z", "to": "2026-05-04T11:00:00Z"},
    )

    assert response.status_code == 200
    assert "azul" not in recorder.statements[0].compile().params.values()
    assert response.json()[0]["match"] == "exact"


async def test_an_inverted_window_is_rejected(client: AsyncClient) -> None:
    response = await client.get(
        "/api/v1/vehicles/search",
        params={"from": "2026-05-04T11:00:00Z", "to": "2026-05-04T09:00:00Z"},
    )

    assert response.status_code == 422


async def test_the_total_and_the_offset_reach_the_response(
    client: AsyncClient, recorder: _Recorder,
) -> None:
    recorder.rows = [_vehicle("plata", 0, "A")]

    response = await client.get(
        "/api/v1/vehicles/search",
        params={
            "from": "2026-05-04T09:00:00Z",
            "to": "2026-05-04T11:00:00Z",
            "limit": 5,
            "offset": 10,
        },
    )

    assert response.headers["x-total-count"] == "1"
    assert "OFFSET" in str(recorder.statements[-1]).upper()
