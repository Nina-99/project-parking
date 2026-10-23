"""The read API is private: a signed cookie or nothing."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.core.config import settings
from backend.app.main import app


@pytest.fixture(autouse=True)
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "OPERATOR_USER", "tester")
    monkeypatch.setattr(settings, "OPERATOR_PASSWORD", "s3cret")
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-with-at-least-32-bytes!!")


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def login(client: AsyncClient, password: str = "s3cret"):
    return await client.post(
        "/api/v1/auth/login",
        json={"username": "tester", "password": password},
    )


async def test_login_sets_an_http_only_cookie(client: AsyncClient) -> None:
    response = await login(client)

    assert response.status_code == 200
    assert response.json() == {"username": "tester"}
    assert settings.AUTH_COOKIE_NAME in response.cookies
    assert "httponly" in response.headers["set-cookie"].lower()


async def test_a_wrong_password_is_rejected(client: AsyncClient) -> None:
    response = await login(client, password="wrong")

    assert response.status_code == 401
    assert settings.AUTH_COOKIE_NAME not in response.cookies


async def test_an_unknown_user_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/auth/login",
        json={"username": "nobody", "password": "s3cret"},
    )

    assert response.status_code == 401


async def test_me_requires_a_session(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/auth/me")).status_code == 401


async def test_me_returns_the_logged_in_operator(client: AsyncClient) -> None:
    await login(client)

    response = await client.get("/api/v1/auth/me")

    assert response.status_code == 200
    assert response.json() == {"username": "tester"}


async def test_a_protected_route_rejects_anonymous_requests(client: AsyncClient) -> None:
    response = await client.get("/api/v1/stats/occupancy")

    assert response.status_code == 401


async def test_logout_clears_the_session(client: AsyncClient) -> None:
    await login(client)

    assert (await client.post("/api/v1/auth/logout")).status_code == 204
    assert (await client.get("/api/v1/auth/me")).status_code == 401


async def test_a_tampered_token_is_rejected(client: AsyncClient) -> None:
    await login(client)

    client.cookies.set(settings.AUTH_COOKIE_NAME, "not-a-token")
    assert (await client.get("/api/v1/auth/me")).status_code == 401
