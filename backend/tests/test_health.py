"""Liveness, readiness and read-only routes, with no database needed."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.api import health
from backend.app.api.auth import require_operator
from backend.app.core.database import get_session
from backend.app.main import app


class _EmptyResult:
    def all(self) -> list:
        return []

    def scalar(self) -> int:
        return 0


class _EmptySession:
    async def execute(self, _statement: object) -> _EmptyResult:
        return _EmptyResult()


@pytest.fixture
async def client():
    app.dependency_overrides[get_session] = lambda: _EmptySession()
    app.dependency_overrides[require_operator] = lambda: "tester"
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_health_endpoint(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


async def test_liveness_does_not_touch_a_dependency(client: AsyncClient) -> None:
    response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


async def test_readiness_is_200_when_both_dependencies_answer(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def ok() -> bool:
        return True

    monkeypatch.setattr(health, "database_ok", ok)
    monkeypatch.setattr(health, "redis_ok", ok)

    response = await client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": True, "redis": True}


async def test_readiness_is_503_when_the_broker_is_down(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def up() -> bool:
        return True

    async def down() -> bool:
        return False

    monkeypatch.setattr(health, "database_ok", up)
    monkeypatch.setattr(health, "redis_ok", down)

    response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": True, "redis": False}


async def test_active_sessions_empty(client: AsyncClient) -> None:
    response = await client.get("/api/v1/stats/active")

    assert response.status_code == 200
    assert response.json() == []


async def test_occupancy(client: AsyncClient) -> None:
    response = await client.get("/api/v1/stats/occupancy")

    assert response.status_code == 200
    assert response.json()["active_sessions"] == 0
