"""Camera health for the operator: who is reporting and when.

The UI needs to answer "is the exit camera alive?" before trusting an empty
occupancy number. A Redis outage does not hide the vehicles; it only hides the
camera state, so this endpoint degrades instead of failing.
"""
from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.adapters.outbound.cache.camera_status import RedisCameraStatus
from backend.app.core.config import settings
from backend.domain.cameras import is_online

router = APIRouter()


class CameraResponse(BaseModel):
    camera_id: str
    gate: str | None
    last_seen: datetime | None
    online: bool
    dropped_frames: int


class CamerasResponse(BaseModel):
    redis_available: bool
    cameras: list[CameraResponse]


def get_camera_status() -> RedisCameraStatus:
    return RedisCameraStatus(settings.CELERY_BROKER_URL)


@router.get("", response_model=CamerasResponse)
async def list_cameras(
    status: RedisCameraStatus = Depends(get_camera_status),
) -> CamerasResponse:
    try:
        readings = await status.list()
        available = True
    except Exception:
        readings = []
        available = False

    now = datetime.now(UTC)
    return CamerasResponse(
        redis_available=available,
        cameras=[
            CameraResponse(
                camera_id=reading.camera_id,
                gate=reading.gate,
                last_seen=reading.last_seen,
                online=is_online(
                    reading.last_seen, now, settings.CAMERA_OFFLINE_AFTER_SECONDS,
                ),
                dropped_frames=reading.dropped_frames,
            )
            for reading in readings
        ],
    )
