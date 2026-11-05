# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.api.auth import require_operator
from backend.app.api.routes.whitelist import admin_only, can_read
from backend.app.core.database import get_session
from backend.app.main import app
from backend.app.models.authorized_vehicle import AuthorizedVehicle
from backend.app.models.user import UserRole


class _FakeScalarResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items

    def scalar_one_or_none(self):
        return self._items[0] if self._items else None


class _FakeDbSession:
    def __init__(self):
        self.store: dict[uuid.UUID, AuthorizedVehicle] = {}

    async def execute(self, _statement):
        # Return all matching in store
        return _FakeScalarResult(list(self.store.values()))

    async def get(self, _model, entity_id):
        return self.store.get(entity_id)

    def add(self, entity):
        if not hasattr(entity, "id") or not entity.id:
            entity.id = uuid.uuid4()
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        if not getattr(entity, "created_at", None):
            entity.created_at = now
        if not getattr(entity, "updated_at", None):
            entity.updated_at = now
        self.store[entity.id] = entity

    async def commit(self):
        pass

    async def refresh(self, entity):
        pass

    async def rollback(self):
        pass

    async def delete(self, entity):
        if entity.id in self.store:
            del self.store[entity.id]


@pytest.fixture
async def whitelist_client():
    fake_db = _FakeDbSession()
    # Add one initial record
    init_v = AuthorizedVehicle(
        id=uuid.uuid4(),
        license_plate="1234ABC",
        owner_name="Ing. Juan Pérez",
        vehicle_type="Sedán",
        color="Blanco",
        notes="Docente FNI",
        is_active=True,
    )
    fake_db.add(init_v)

    app.dependency_overrides[get_session] = lambda: fake_db
    app.dependency_overrides[require_operator] = lambda: "admin_user"
    app.dependency_overrides[can_read] = lambda: ("admin_user", UserRole.ADMIN)
    app.dependency_overrides[admin_only] = lambda: ("admin_user", UserRole.ADMIN)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac, fake_db
    app.dependency_overrides.clear()


async def test_list_and_check_authorized_vehicle(whitelist_client) -> None:
    client, _db = whitelist_client

    # 1. List
    resp = await client.get("/api/v1/whitelist")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["license_plate"] == "1234ABC"

    # 2. Check plate
    resp_chk = await client.get("/api/v1/whitelist/check/1234abc")
    assert resp_chk.status_code == 200
    chk_data = resp_chk.json()
    assert chk_data["authorized"] is True
    assert chk_data["vehicle"]["owner_name"] == "Ing. Juan Pérez"


async def test_create_and_delete_authorized_vehicle(whitelist_client) -> None:
    client, _db = whitelist_client

    create_payload = {
        "license_plate": "5678XYZ",
        "owner_name": "Lic. María Gómez",
        "vehicle_type": "Camioneta",
        "color": "Rojo",
        "notes": "Administrativo",
        "is_active": True,
    }
    resp = await client.post("/api/v1/whitelist", json=create_payload)
    assert resp.status_code == 201
    created = resp.json()
    assert created["license_plate"] == "5678XYZ"

    # Delete
    veh_id = created["id"]
    resp_del = await client.delete(f"/api/v1/whitelist/{veh_id}")
    assert resp_del.status_code == 204
