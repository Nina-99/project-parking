"""History keeps its list body; the total travels in a header."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.api.auth import require_operator
from backend.app.core.database import get_session
from backend.app.main import app


class _Result:
    def __init__(self, rows: list, total: int) -> None:
        self._rows = rows
        self._total = total

    def scalar(self) -> int:
        return self._total

    def all(self) -> list:
        return self._rows


class _Session:
    def __init__(self, total: int) -> None:
        self._total = total

    async def execute(self, _statement):
        return _Result([], self._total)


@pytest.fixture
async def client():
    app.dependency_overrides[get_session] = lambda: _Session(42)
    app.dependency_overrides[require_operator] = lambda: "tester"
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_history_reports_the_total_before_pagination(client: AsyncClient) -> None:
    response = await client.get("/api/v1/stats/history?limit=10&offset=0")

    assert response.status_code == 200
    assert response.json() == []
    assert response.headers["x-total-count"] == "42"
