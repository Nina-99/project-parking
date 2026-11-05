# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The read API is private: a signed cookie or nothing.

Credentials come from the ``users`` table, so these are integration tests: they
need the throwaway PostgreSQL and its migrations.
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from backend.app.api.dependencies import get_rate_limiter
from backend.app.core.config import settings
from backend.app.core.security import hash_password
from backend.app.main import app
from backend.app.models.user import User, UserRole

pytestmark = pytest.mark.integration


class _AllowAll:
    async def allow(self, _key: str) -> bool:
        return True


@pytest.fixture(autouse=True)
def credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-with-at-least-32-bytes!!")
    app.dependency_overrides[get_rate_limiter] = lambda: _AllowAll()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client(api):
    http, factory = api
    async with factory() as session:
        session.add(User(
            username="tester",
            password_hash=hash_password("s3cret"),
            role=UserRole.ADMIN.value,
        ))
        await session.commit()
    yield http


async def login(client: AsyncClient, password: str = "s3cret"):
    return await client.post(
        "/api/v1/auth/login",
        json={"username": "tester", "password": password},
    )


async def test_login_sets_an_http_only_cookie(client: AsyncClient) -> None:
    response = await login(client)

    assert response.status_code == 200
    assert response.json() == {"username": "tester", "role": "admin"}
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


async def test_an_inactive_user_cannot_log_in(client: AsyncClient, api) -> None:
    _, factory = api
    async with factory() as session:
        user = await session.scalar(select(User).where(User.username == "tester"))
        user.is_active = False
        await session.commit()

    assert (await login(client)).status_code == 401


async def test_me_requires_a_session(client: AsyncClient) -> None:
    assert (await client.get("/api/v1/auth/me")).status_code == 401


async def test_me_returns_the_logged_in_operator(client: AsyncClient) -> None:
    await login(client)

    response = await client.get("/api/v1/auth/me")

    assert response.status_code == 200
    assert response.json() == {"username": "tester", "role": "admin"}


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


async def test_the_login_is_rate_limited(client: AsyncClient) -> None:
    class _Deny:
        async def allow(self, _key: str) -> bool:
            return False

    app.dependency_overrides[get_rate_limiter] = lambda: _Deny()

    response = await login(client)

    assert response.status_code == 429
    assert response.headers["retry-after"] == str(settings.LOGIN_RATE_WINDOW_SECONDS)
