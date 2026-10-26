# SPDX-FileCopyrightText: 2026 Nina
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The manual close endpoint: audit in the response, 409 on a double close."""
from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.api.auth import require_operator
from backend.app.api.dependencies import get_clock, get_publisher, get_uow_factory
from backend.app.main import app
from backend.domain.errors import SessionNotActiveError, SessionNotFoundError
from backend.tests.fakes import FakePublisher, FakeUnitOfWork, FixedClock


@pytest.fixture
def uow() -> FakeUnitOfWork:
    return FakeUnitOfWork()


@pytest.fixture
async def client(uow: FakeUnitOfWork):
    app.dependency_overrides[require_operator] = lambda: "ana"
    app.dependency_overrides[get_uow_factory] = lambda: (lambda: uow)
    app.dependency_overrides[get_publisher] = lambda: FakePublisher()
    app.dependency_overrides[get_clock] = lambda: FixedClock()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_a_close_returns_the_audited_outcome(
    client: AsyncClient, uow: FakeUnitOfWork,
) -> None:
    session_id = uuid.uuid4()

    response = await client.post(
        f"/api/v1/sessions/{session_id}/close",
        json={"reason": "left without a read"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "session_id": str(session_id),
        "duration_seconds": 3600,
        "reason": "left without a read",
        "operator": "ana",
    }
    assert uow.sessions.closed[0]["operator"] == "ana"


async def test_an_empty_reason_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        f"/api/v1/sessions/{uuid.uuid4()}/close",
        json={"reason": ""},
    )

    assert response.status_code == 422


async def test_an_unknown_session_is_404(
    client: AsyncClient, uow: FakeUnitOfWork,
) -> None:
    uow.sessions.failure = SessionNotFoundError("nope")

    response = await client.post(
        f"/api/v1/sessions/{uuid.uuid4()}/close",
        json={"reason": "manual"},
    )

    assert response.status_code == 404


async def test_closing_twice_is_409(
    client: AsyncClient, uow: FakeUnitOfWork,
) -> None:
    uow.sessions.failure = SessionNotActiveError("already closed")

    response = await client.post(
        f"/api/v1/sessions/{uuid.uuid4()}/close",
        json={"reason": "manual"},
    )

    assert response.status_code == 409


async def test_an_anonymous_close_is_rejected() -> None:
    app.dependency_overrides.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            f"/api/v1/sessions/{uuid.uuid4()}/close",
            json={"reason": "manual"},
        )

    assert response.status_code == 401
