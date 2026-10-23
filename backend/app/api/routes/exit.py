"""Exit gate endpoint."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_session
from backend.app.models.parking_session import ParkingSession, SessionStatus
from backend.app.schemas.parking import ExitResponse

router = APIRouter()


@router.post("", response_model=ExitResponse)
async def register_exit(
    gate_id: str,
    session_id: str | None = None,
    image: UploadFile | None = File(None),
    db_session: AsyncSession = Depends(get_session),
):
    """Close the given session, or refuse when no session was provided."""
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required until ReID is implemented",
        )

    result = await db_session.execute(
        select(ParkingSession).where(
            ParkingSession.id == uuid.UUID(session_id),
            ParkingSession.status == SessionStatus.ACTIVE,
        ),
    )
    parking_session = result.scalar_one_or_none()
    if not parking_session:
        raise HTTPException(status_code=404, detail="Active session not found")

    now = datetime.now(timezone.utc)
    parking_session.exit_time = now
    parking_session.exit_gate = gate_id
    parking_session.status = SessionStatus.COMPLETED

    if parking_session.entry_time:
        delta = now - parking_session.entry_time
        parking_session.duration_seconds = int(delta.total_seconds())

    await db_session.flush()

    # TODO: dispatch Celery task for exit image processing

    return ExitResponse(
        session_id=parking_session.id,
        vehicle_id=parking_session.vehicle_id,
        entry_time=parking_session.entry_time,
        exit_time=parking_session.exit_time,
        duration_minutes=(
            round(parking_session.duration_seconds / 60, 1)
            if parking_session.duration_seconds else None
        ),
        status="completed",
    )
