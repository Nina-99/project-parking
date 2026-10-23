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

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_session
from backend.app.models.vehicle import Vehicle
from backend.app.schemas.vehicle import VehicleSearchHit
from backend.domain.colours import expand_color
from backend.domain.incident import rank_hits

router = APIRouter()


@router.get("/search", response_model=list[VehicleSearchHit])
async def search_vehicles(
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
    db_session: AsyncSession = Depends(get_session),
) -> list[VehicleSearchHit]:
    """Candidate vehicles inside a time window, best match first."""
    if start > end:
        raise HTTPException(status_code=422, detail="`from` must not be after `to`")

    stmt = select(Vehicle).where(
        Vehicle.created_at >= start,
        Vehicle.created_at <= end,
    )
    if vehicle_type:
        stmt = stmt.where(Vehicle.vehicle_type == vehicle_type)
    requested = expand_color(color)
    if requested:
        stmt = stmt.where(
            Vehicle.color.in_(requested),
            Vehicle.color_confidence >= min_confidence,
        )
    stmt = stmt.order_by(Vehicle.created_at.desc()).limit(limit)

    rows = list((await db_session.execute(stmt)).scalars().all())
    ranked = rank_hits(rows, color, at)

    return [
        VehicleSearchHit(
            id=row.id,
            license_plate=row.license_plate,
            vehicle_type=row.vehicle_type,
            color=row.color,
            color_confidence=row.color_confidence,
            entry_image_path=row.entry_image_path,
            created_at=row.created_at,
            match="exact" if not color or row.color == color else "neighbour",
        )
        for row in ranked
    ]
