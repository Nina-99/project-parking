# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Camera health for the operator: who is reporting and when.

The UI needs to answer "is the exit camera alive?" before trusting an empty
occupancy number. A Redis outage does not hide the vehicles; it only hides the
camera state, so this endpoint degrades instead of failing.

Two sources, one answer. The ``cameras`` table says which cameras *should* be
reporting; Redis says which ones *are*. Listing only Redis made a camera that
never started indistinguishable from a system with no cameras at all, which is
how an empty dashboard gets mistaken for a broken one.
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.adapters.outbound.cache.camera_status import RedisCameraStatus
from backend.app.api.auth import require_role
from backend.app.core.config import settings
from backend.app.core.database import get_session
from backend.app.models.camera import Camera
from backend.app.models.user import UserRole
from backend.domain.cameras import is_online

router = APIRouter()

GATES = ("entry", "exit")


class CameraResponse(BaseModel):
    camera_id: str
    gate: str | None
    label: str | None = None
    last_seen: datetime | None
    online: bool
    dropped_frames: int
    registered: bool = True


class CamerasResponse(BaseModel):
    redis_available: bool
    cameras: list[CameraResponse]


class CameraCreate(BaseModel):
    camera_id: str = Field(min_length=1, max_length=64)
    gate: str = Field(default="entry")
    label: str | None = Field(default=None, max_length=120)
    rtsp_url: str | None = None
    is_active: bool = True


class CameraUpdate(BaseModel):
    gate: str | None = None
    label: str | None = Field(default=None, max_length=120)
    rtsp_url: str | None = None
    is_active: bool | None = None


def get_camera_status() -> RedisCameraStatus:
    return RedisCameraStatus(settings.CELERY_BROKER_URL)


admin_only = require_role(UserRole.ADMIN)


def _validate_gate(gate: str) -> None:
    if gate not in GATES:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"gate must be one of {list(GATES)}")


@router.get("", response_model=CamerasResponse)
async def list_cameras(
    status_endpoint: RedisCameraStatus = Depends(get_camera_status),
    session: AsyncSession = Depends(get_session),
) -> CamerasResponse:
    try:
        readings = await status_endpoint.list()
        available = True
    except Exception:
        readings = []
        available = False

    registered = list(await session.scalars(
        select(Camera).where(Camera.is_active.is_(True)).order_by(Camera.camera_id)
    ))
    now = datetime.now(UTC)

    cameras: list[CameraResponse] = []
    seen: set[str] = set()

    # Registered first: a configured camera appears even when it has never sent a
    # heartbeat, which is precisely the case the old endpoint could not express.
    for camera in registered:
        reading = next((r for r in readings if r.camera_id == camera.camera_id), None)
        seen.add(camera.camera_id)
        cameras.append(CameraResponse(
            camera_id=camera.camera_id,
            gate=camera.gate,
            label=camera.label,
            last_seen=reading.last_seen if reading else None,
            online=is_online(reading.last_seen, now, settings.CAMERA_OFFLINE_AFTER_SECONDS)
            if reading else False,
            dropped_frames=reading.dropped_frames if reading else 0,
            registered=True,
        ))

    # Then anyone reporting that nobody registered: the capturer is running with a
    # config the database does not know about. Showing it beats hiding it.
    for reading in readings:
        if reading.camera_id in seen:
            continue
        cameras.append(CameraResponse(
            camera_id=reading.camera_id,
            gate=reading.gate,
            last_seen=reading.last_seen,
            online=is_online(reading.last_seen, now, settings.CAMERA_OFFLINE_AFTER_SECONDS),
            dropped_frames=reading.dropped_frames,
            registered=False,
        ))

    return CamerasResponse(redis_available=available, cameras=cameras)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_camera(
    body: CameraCreate,
    _: tuple[str, str] = Depends(admin_only),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    _validate_gate(body.gate)
    exists = await session.scalar(
        select(Camera).where(Camera.camera_id == body.camera_id)
    )
    if exists is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="camera_id already exists")

    camera = Camera(
        camera_id=body.camera_id,
        gate=body.gate,
        label=body.label,
        rtsp_url=body.rtsp_url,
        is_active=body.is_active,
    )
    session.add(camera)
    await session.flush()
    return {"id": str(camera.id), "camera_id": camera.camera_id}


@router.patch("/{camera_id}")
async def update_camera(
    camera_id: str,
    body: CameraUpdate,
    _: tuple[str, str] = Depends(admin_only),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    camera = await session.scalar(select(Camera).where(Camera.camera_id == camera_id))
    if camera is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    if body.gate is not None:
        _validate_gate(body.gate)
        camera.gate = body.gate
    if body.label is not None:
        camera.label = body.label
    if body.rtsp_url is not None:
        camera.rtsp_url = body.rtsp_url
    if body.is_active is not None:
        camera.is_active = body.is_active
    await session.flush()
    return {"id": str(camera.id), "camera_id": camera.camera_id}


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    camera_id: str,
    _: tuple[str, str] = Depends(admin_only),
    session: AsyncSession = Depends(get_session),
) -> None:
    camera = await session.scalar(select(Camera).where(Camera.camera_id == camera_id))
    if camera is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="camera not found")
    await session.delete(camera)
