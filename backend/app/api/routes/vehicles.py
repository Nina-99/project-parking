"""Vehicle lookup for incident investigation.

The scenario this serves: an operator remembers "a silver vehicle entered around
10:00" and needs the candidate list, not a yes/no answer. So the endpoint takes a
time window and an optional colour, expands the colour to its measured confusion
neighbourhood, and orders by proximity to the remembered instant.

Two decisions worth keeping in mind:

* Colour is a *soft* filter. A row is only returned when
  ``color_confidence >= min_confidence``, because a low-confidence label is noise
  whichever colour it names.
* With 1-7 vehicles per 10 minutes at the gate, the time window already does most
  of the narrowing. Colour only earns its keep on windows of hours or a full day.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.database import get_session
from backend.app.models.parking_session import ParkingSession
from backend.app.models.vehicle import Vehicle
from backend.app.schemas.vehicle import (
    VehicleDetail,
    VehicleSearchHit,
    VehicleSession,
)
from backend.domain.colours import expand_color
from backend.domain.errors import VehicleNotFoundError
from backend.domain.incident import rank_hits

router = APIRouter()


def resolve_image_path(stored: str | None, root: Path | None = None) -> Path | None:
    """A crop path is served only when it lives inside the image root.

    The stored value comes from the database, but a database is not a trust
    boundary: a bad row must not turn into an arbitrary file read.
    """
    if not stored:
        return None
    base = (root or Path(settings.ENTRY_IMAGE_DIR)).resolve()
    candidate = Path(stored).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        return None
    return candidate



@router.get("/search", response_model=list[VehicleSearchHit])
async def search_vehicles(
    response: Response,
    start: datetime = Query(alias="from", description="Window start, inclusive."),
    end: datetime = Query(alias="to", description="Window end, inclusive."),
    color: str | None = Query(None, description="Requested colour, e.g. 'plata'."),
    vehicle_type: str | None = Query(None, alias="type", description="Body type filter."),
    min_confidence: float = Query(
        0.3, ge=0.0, le=1.0, description="Minimum colour confidence to trust the label.",
    ),
    at: datetime | None = Query(
        None, description="Remembered instant; results are ordered by distance to it.",
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db_session: AsyncSession = Depends(get_session),
) -> list[VehicleSearchHit]:
    """Candidate vehicles inside a time window, best match first.

    The total before pagination travels in ``X-Total-Count``: the body stays a
    plain list, so a client that only wants the page does not change.
    """
    if start > end:
        raise HTTPException(status_code=422, detail="`from` must not be after `to`")

    base = select(Vehicle).where(
        Vehicle.created_at >= start,
        Vehicle.created_at <= end,
    )
    if vehicle_type:
        base = base.where(Vehicle.vehicle_type == vehicle_type)
    requested = expand_color(color)
    if requested:
        base = base.where(
            Vehicle.color.in_(requested),
            Vehicle.color_confidence >= min_confidence,
        )

    total = (await db_session.execute(
        select(func.count()).select_from(base.subquery()),
    )).scalar() or 0
    response.headers["X-Total-Count"] = str(total)

    stmt = base.order_by(Vehicle.created_at.desc()).offset(offset).limit(limit)

    rows = list((await db_session.execute(stmt)).scalars().all())
    ranked = rank_hits(rows, color, at)

    return [
        VehicleSearchHit(
            id=row.id,
            license_plate=row.license_plate,
            vehicle_type=row.vehicle_type,
            color=row.color,
            color_confidence=row.color_confidence,
            image_url=(
                f"/api/v1/vehicles/{row.id}/image" if row.entry_image_path else None
            ),
            created_at=row.created_at,
            match="exact" if not color or row.color == color else "neighbour",
        )
        for row in ranked
    ]


@router.get("/{vehicle_id}/image", response_class=FileResponse)
async def vehicle_image(
    vehicle_id: uuid.UUID,
    db_session: AsyncSession = Depends(get_session),
) -> FileResponse:
    """The crop for one observation. Paths never leave the server."""
    vehicle = await db_session.get(Vehicle, vehicle_id)
    if vehicle is None:
        raise HTTPException(status_code=404, detail="vehicle not found")
    path = resolve_image_path(vehicle.entry_image_path)
    if path is None:
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.get("/{vehicle_id}", response_model=VehicleDetail)
async def get_vehicle(
    vehicle_id: uuid.UUID,
    db_session: AsyncSession = Depends(get_session),
) -> VehicleDetail:
    """One vehicle with its sessions, for the incident detail view."""
    vehicle = await db_session.get(Vehicle, vehicle_id)
    if vehicle is None:
        raise VehicleNotFoundError(str(vehicle_id))

    result = await db_session.execute(
        select(ParkingSession)
        .where(ParkingSession.vehicle_id == vehicle_id)
        .order_by(ParkingSession.entry_time.desc()),
    )
    sessions = list(result.scalars().all())

    return VehicleDetail(
        id=vehicle.id,
        license_plate=vehicle.license_plate,
        vehicle_type=vehicle.vehicle_type,
        vehicle_type_confidence=vehicle.vehicle_type_confidence,
        color=vehicle.color,
        color_confidence=vehicle.color_confidence,
        image_url=(
            f"/api/v1/vehicles/{vehicle.id}/image"
            if vehicle.entry_image_path else None
        ),
        created_at=vehicle.created_at,
        sessions=[
            VehicleSession(
                id=session.id,
                entry_gate=session.entry_gate,
                exit_gate=session.exit_gate,
                entry_time=session.entry_time,
                exit_time=session.exit_time,
                duration_seconds=session.duration_seconds,
                status=str(session.status),
                close_reason=session.close_reason,
                closed_by=session.closed_by,
            )
            for session in sessions
        ],
    )
