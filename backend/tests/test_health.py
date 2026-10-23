"""Health check and empty-database responses."""
from __future__ import annotations

from httpx import AsyncClient, ASGITransport
import pytest

from backend.app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_active_sessions_empty(client: AsyncClient):
    response = await client.get("/api/v1/stats/active")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_occupancy(client: AsyncClient):
    response = await client.get("/api/v1/stats/occupancy")
    assert response.status_code == 200
    data = response.json()
    assert "active_sessions" in data
