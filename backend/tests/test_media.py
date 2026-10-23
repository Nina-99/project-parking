"""Crops are served by id, and only from inside the image root.

The operator needs the thumbnail next to the candidate; the browser must never
receive a filesystem path, and a bad row must not become a file read.
"""
from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from httpx import ASGITransport, AsyncClient

from backend.app.api.routes.vehicles import resolve_image_path
from backend.app.core.config import settings
from backend.app.core.database import get_session
from backend.app.main import app


def _crop(root: Path, name: str = "crop.jpg") -> Path:
    path = root / "gate-1" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"jpeg-bytes")
    return path


def _vehicle(image: str | None) -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4(), entry_image_path=image)


@asynccontextmanager
async def _client(vehicle: SimpleNamespace | None):
    class _Session:
        async def get(self, _model: object, _ident: object) -> SimpleNamespace | None:
            return vehicle

    app.dependency_overrides[get_session] = lambda: _Session()
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def test_a_crop_inside_the_root_is_served(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ENTRY_IMAGE_DIR", str(tmp_path))
    path = _crop(tmp_path)

    async def scenario() -> None:
        async with _client(_vehicle(str(path))) as client:
            response = await client.get(f"/api/v1/vehicles/{uuid.uuid4()}/image")

            assert response.status_code == 200
            assert response.content == b"jpeg-bytes"
            assert response.headers["content-type"].startswith("image/jpeg")

    asyncio.run(scenario())


def test_a_path_outside_the_root_is_refused(tmp_path: Path) -> None:
    outside = _crop(tmp_path / "outside")

    assert resolve_image_path(str(outside), root=tmp_path / "inside") is None


def test_traversal_out_of_the_root_is_refused(tmp_path: Path) -> None:
    outside = _crop(tmp_path.parent)

    assert resolve_image_path(str(outside), root=tmp_path) is None


def test_a_missing_file_is_not_an_image(tmp_path: Path) -> None:
    assert resolve_image_path(str(tmp_path / "gate-1" / "gone.jpg"), root=tmp_path) is None
    assert resolve_image_path(None, root=tmp_path) is None


def test_an_unknown_vehicle_is_404() -> None:
    async def scenario() -> None:
        async with _client(None) as client:
            response = await client.get(f"/api/v1/vehicles/{uuid.uuid4()}/image")

            assert response.status_code == 404

    asyncio.run(scenario())


def test_a_vehicle_without_a_crop_is_404(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "ENTRY_IMAGE_DIR", str(tmp_path))

    async def scenario() -> None:
        async with _client(_vehicle(None)) as client:
            response = await client.get(f"/api/v1/vehicles/{uuid.uuid4()}/image")

            assert response.status_code == 404

    asyncio.run(scenario())
