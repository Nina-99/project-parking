"""
Entry gate endpoint.

Receives vehicle image → dispatches Celery task → returns session info.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select
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
    """
    Register a vehicle entry.

    - Receives the image from the gate camera
    - Creates a pending ParkingSession
    - Dispatches a Celery task for YOLO inference + OCR
    - Returns the session ID immediately
    """
    # Read image bytes
    image_bytes = await image.read()

    # Create vehicle placeholder
    vehicle = Vehicle()
    session.add(vehicle)
    await session.flush()

    # Create parking session
    parking_session = ParkingSession(
        vehicle_id=vehicle.id,
        entry_gate=gate_id,
        status=SessionStatus.ACTIVE,
    )
    session.add(parking_session)
    await session.flush()

    # TODO: Dispatch Celery task for async detection
    # from backend.worker.tasks.detection import process_entry
    # process_entry.delay(str(parking_session.id), image_bytes)

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
    """Alternative entry endpoint — receives raw image upload."""
    return await register_entry(gate_id, image, session)
