# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Model health is computed from the inference rows, not from a benchmark.

Admin only: the panel diagnoses the detector, it does not describe the lot.
"""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.api.auth import issue_token, require_operator
from backend.app.api.routes.stats import admin_only
from backend.app.core.config import settings
from backend.app.core.database import get_session
from backend.app.main import app


class _Result:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def all(self) -> list:
        return self._rows


class _Recorder:
    def __init__(self, rows: list) -> None:
        self.rows = rows
        self.statements: list = []

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result(self.rows)


@pytest.fixture
async def client_factory():
    clients = []

    def build(rows: list) -> tuple[AsyncClient, _Recorder]:
        recorder = _Recorder(rows)
        app.dependency_overrides[get_session] = lambda: recorder
        app.dependency_overrides[require_operator] = lambda: "tester"
        app.dependency_overrides[admin_only] = lambda: ("tester", "admin")
        transport = ASGITransport(app=app)
        client = AsyncClient(transport=transport, base_url="http://test")
        clients.append(client)
        return client, recorder

    try:
        yield build
    finally:
        for client in clients:
            await client.aclose()
        app.dependency_overrides.clear()


async def test_the_model_panel_returns_the_aggregates(client_factory) -> None:
    client, recorder = client_factory([("yolo26x", 10, 0.8, 45.0, 90.0)])

    response = await client.get("/api/v1/stats/model?days=7")

    assert response.status_code == 200
    assert response.json() == {
        "window_days": 7,
        "models": [{
            "model_name": "yolo26x",
            "events": 10,
            "avg_confidence": 0.8,
            "p50_latency_ms": 45.0,
            "p95_latency_ms": 90.0,
        }],
        "total_events": 10,
    }
    assert "percentile_cont" in str(recorder.statements[0]).lower()


async def test_an_empty_window_is_an_empty_list(client_factory) -> None:
    client, _ = client_factory([])

    response = await client.get("/api/v1/stats/model")

    assert response.status_code == 200
    assert response.json()["models"] == []
    assert response.json()["total_events"] == 0


@pytest.mark.parametrize("role", ["operator", "owner"])
async def test_only_an_admin_reads_the_inference_telemetry(
    client_factory, role: str,
) -> None:
    client, _ = client_factory([])
    app.dependency_overrides.pop(admin_only, None)
    client.cookies.set(settings.AUTH_COOKIE_NAME, issue_token("alguien", role))

    response = await client.get("/api/v1/stats/model")

    assert response.status_code == 403
