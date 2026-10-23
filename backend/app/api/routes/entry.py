"""Entry gate endpoint: open a session and hand the image to the worker."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_session
from backend.app.models.parking_session import ParkingSession, SessionStatus
from backend.app.models.vehicle import Vehicle
from backend.app.schemas.parking import EntryResponse

router = APIRouter()


@router.post("", response_model=EntryResponse, status_code=201)
async def register_entry(
    gate_id: str,
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """Open a session for this gate and return it before inference finishes."""
    image_bytes = await image.read()

    vehicle = Vehicle()
    session.add(vehicle)
    await session.flush()

    parking_session = ParkingSession(
        vehicle_id=vehicle.id,
        entry_gate=gate_id,
        status=SessionStatus.ACTIVE,
    )
    session.add(parking_session)
    await session.flush()

    # TODO: dispatch detection for this uploaded image. The live path is wired
    # through rtsp-capture/capture.py -> detection.process_frame; this upload
    # path still needs a task that fills in the vehicle the route just created.
    _ = image_bytes

    return EntryResponse(
        session_id=parking_session.id,
        vehicle_id=vehicle.id,
        status="registered",
    )


@router.post("/image", status_code=201)
async def register_entry_with_base64(
    gate_id: str,
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """Same as the main endpoint, kept for clients that post a raw upload."""
    return await register_entry(gate_id, image, session)
