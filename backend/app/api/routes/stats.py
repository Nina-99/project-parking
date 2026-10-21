"""
Statistics and reports endpoints.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_session
from backend.app.models.parking_session import ParkingSession, SessionStatus
from backend.app.models.vehicle import Vehicle
from backend.app.schemas.parking import ActiveSessionResponse, SessionResponse

router = APIRouter()


@router.get("/active", response_model=list[ActiveSessionResponse])
async def get_active_sessions(
    db_session: AsyncSession = Depends(get_session),
):
    """List all vehicles currently in the parking lot."""
    result = await db_session.execute(
        select(ParkingSession, Vehicle)
        .join(Vehicle, ParkingSession.vehicle_id == Vehicle.id)
        .where(ParkingSession.status == SessionStatus.ACTIVE),
    )
    rows = result.all()
    return [
        ActiveSessionResponse(
            id=ps.id,
            vehicle_id=ps.vehicle_id,
            entry_time=ps.entry_time,
            entry_gate=ps.entry_gate,
            license_plate=v.license_plate,
            vehicle_type=v.vehicle_type,
            color=v.color,
        )
        for ps, v in rows
    ]


@router.get("/history", response_model=list[SessionResponse])
async def get_history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db_session: AsyncSession = Depends(get_session),
):
    """List completed parking sessions (paginated)."""
    result = await db_session.execute(
        select(ParkingSession, Vehicle)
        .join(Vehicle, ParkingSession.vehicle_id == Vehicle.id)
        .where(ParkingSession.status == SessionStatus.COMPLETED)
        .order_by(ParkingSession.exit_time.desc())
        .offset(offset)
        .limit(limit),
    )
    rows = result.all()
    return [
        SessionResponse(
            id=ps.id,
            vehicle_id=ps.vehicle_id,
            entry_gate=ps.entry_gate,
            exit_gate=ps.exit_gate,
            entry_time=ps.entry_time,
            exit_time=ps.exit_time,
            duration_seconds=ps.duration_seconds,
            status=ps.status.value,
            license_plate=v.license_plate,
            vehicle_type=v.vehicle_type,
            color=v.color,
        )
        for ps, v in rows
    ]


@router.get("/occupancy")
async def get_occupancy(
    db_session: AsyncSession = Depends(get_session),
):
    """Current occupancy count."""
    result = await db_session.execute(
        select(func.count(ParkingSession.id)).where(
            ParkingSession.status == SessionStatus.ACTIVE,
        ),
    )
    active_count = result.scalar() or 0
    return {"active_sessions": active_count, "timestamp": datetime.now()}


@router.get("/daily")
async def get_daily_stats(
    days: int = Query(7, ge=1, le=90),
    db_session: AsyncSession = Depends(get_session),
):
    """Aggregated stats for the last N days."""
    since = datetime.now() - timedelta(days=days)
    result = await db_session.execute(
        select(
            func.date(ParkingSession.entry_time).label("date"),
            func.count(ParkingSession.id).label("total"),
            func.avg(ParkingSession.duration_seconds).label("avg_duration"),
        )
        .where(ParkingSession.entry_time >= since)
        .group_by(func.date(ParkingSession.entry_time))
        .order_by(func.date(ParkingSession.entry_time)),
    )
    rows = result.all()
    return [
        {
            "date": str(row.date),
            "total_vehicles": row.total,
            "avg_duration_minutes": round(row.avg_duration / 60, 1) if row.avg_duration else None,
        }
        for row in rows
    ]
