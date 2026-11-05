# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Users, roles and the invariants that keep the lot administrable.

Admin only, enforced on the server: the UI is not the only client.
"""
from __future__ import annotations

import pytest
from sqlalchemy import func, select

from backend.app.api.auth import seed_first_admin
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
def _no_rate_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "SECRET_KEY", "test-secret-with-at-least-32-bytes!!")
    app.dependency_overrides[get_rate_limiter] = lambda: _AllowAll()
    yield
    app.dependency_overrides.clear()


async def _seed(factory, username: str, role: UserRole, password: str = "s3cret") -> None:
    async with factory() as session:
        session.add(User(username=username, password_hash=hash_password(password),
                         role=role.value))
        await session.commit()


async def _login(client, username: str, password: str = "s3cret"):
    response = await client.post(
        "/api/v1/auth/login", json={"username": username, "password": password},
    )
    assert response.status_code == 200, response.text
    return response


async def test_an_admin_creates_a_user_and_that_user_can_enter(api) -> None:
    client, factory = api
    await _seed(factory, "root", UserRole.ADMIN)
    await _login(client, "root")

    created = await client.post("/api/v1/users", json={
        "username": "juan", "password": "unaClaveLarga", "role": "operator",
    })
    assert created.status_code == 201, created.text
    assert created.json()["role"] == "operator"
    assert "password" not in created.json()

    client.cookies.clear()
    await _login(client, "juan", "unaClaveLarga")
    me = await client.get("/api/v1/auth/me")
    assert me.json() == {"username": "juan", "role": "operator"}


@pytest.mark.parametrize("role", [UserRole.OPERATOR, UserRole.OWNER])
async def test_non_admins_cannot_manage_users(api, role: UserRole) -> None:
    client, factory = api
    await _seed(factory, "someone", role)
    await _login(client, "someone")

    assert (await client.get("/api/v1/users")).status_code == 403
    assert (await client.post("/api/v1/users", json={
        "username": "otro", "password": "unaClaveLarga", "role": "owner",
    })).status_code == 403


async def test_an_admin_cannot_delete_itself(api) -> None:
    client, factory = api
    await _seed(factory, "root", UserRole.ADMIN)
    await _login(client, "root")
    listed = (await client.get("/api/v1/users")).json()
    my_id = next(u["id"] for u in listed if u["username"] == "root")

    response = await client.delete(f"/api/v1/users/{my_id}")

    assert response.status_code == 409


async def test_the_last_admin_cannot_be_demoted_or_deactivated(api) -> None:
    client, factory = api
    await _seed(factory, "root", UserRole.ADMIN)
    await _login(client, "root")
    listed = (await client.get("/api/v1/users")).json()
    my_id = next(u["id"] for u in listed if u["username"] == "root")

    demote = await client.patch(f"/api/v1/users/{my_id}", json={"role": "owner"})
    disable = await client.patch(f"/api/v1/users/{my_id}", json={"is_active": False})

    assert demote.status_code == 409
    assert disable.status_code == 409


async def test_a_second_admin_allows_demoting_the_first(api) -> None:
    client, factory = api
    await _seed(factory, "root", UserRole.ADMIN)
    await _seed(factory, "root2", UserRole.ADMIN)
    await _login(client, "root")
    listed = (await client.get("/api/v1/users")).json()
    other_id = next(u["id"] for u in listed if u["username"] == "root2")

    response = await client.patch(f"/api/v1/users/{other_id}", json={"role": "operator"})

    assert response.status_code == 200
    assert response.json()["role"] == "operator"


async def test_the_first_admin_is_seeded_from_the_environment(api, monkeypatch) -> None:
    _, factory = api
    monkeypatch.setattr(settings, "OPERATOR_USER", "primero")
    monkeypatch.setattr(settings, "OPERATOR_PASSWORD", "claveDeArranque")

    async with factory() as session:
        await seed_first_admin(session)
        again = await session.scalar(select(User).where(User.username == "primero"))
    assert again is not None
    assert again.role == UserRole.ADMIN.value

    # idempotent: a second call must not create a duplicate nor fail
    async with factory() as session:
        await seed_first_admin(session)
        count = await session.scalar(select(func.count()).select_from(User))
    assert count == 1


async def test_a_password_shorter_than_eight_characters_is_rejected(api) -> None:
    client, factory = api
    await _seed(factory, "root", UserRole.ADMIN)
    await _login(client, "root")

    response = await client.post("/api/v1/users", json={
        "username": "corto", "password": "1234", "role": "operator",
    })

    assert response.status_code == 422
