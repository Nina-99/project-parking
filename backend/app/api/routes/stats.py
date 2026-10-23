"""Read-only aggregates over the stored vehicles and sessions."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.database import get_session
from backend.app.models.detection_event import DetectionEvent
from backend.app.models.parking_session import ParkingSession, SessionStatus
from backend.app.models.vehicle import Vehicle
from backend.app.schemas.parking import ActiveSessionResponse, SessionResponse
from backend.domain.colours import (
    MIN_REPORT_CONFIDENCE,
    REPORTABLE_COLORS,
    colour_counts,
    report_bucket,
)
from backend.domain.occupancy import occupancy_percent
from backend.domain.session import age_minutes, is_stale

router = APIRouter()


@router.get("/active", response_model=list[ActiveSessionResponse])
async def get_active_sessions(
    db_session: AsyncSession = Depends(get_session),
) -> list[ActiveSessionResponse]:
    """List all vehicles currently in the parking lot."""
    result = await db_session.execute(
        select(ParkingSession, Vehicle)
        .join(Vehicle, ParkingSession.vehicle_id == Vehicle.id)
        .where(ParkingSession.status == SessionStatus.ACTIVE),
    )
    rows = result.all()
    now = datetime.now(UTC)
    return [
        ActiveSessionResponse(
            id=ps.id,
            vehicle_id=ps.vehicle_id,
            entry_time=ps.entry_time,
            entry_gate=ps.entry_gate,
            license_plate=v.license_plate,
            vehicle_type=v.vehicle_type,
            color=v.color,
            stale=is_stale(ps.entry_time, now, settings.STALE_SESSION_HOURS),
            age_minutes=age_minutes(ps.entry_time, now),
        )
        for ps, v in rows
    ]


@router.get("/history", response_model=list[SessionResponse])
async def get_history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db_session: AsyncSession = Depends(get_session),
) -> list[SessionResponse]:
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
            close_reason=ps.close_reason,
            closed_by=ps.closed_by,
        )
        for ps, v in rows
    ]


@router.get("/occupancy")
async def get_occupancy(
    db_session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    """Current occupancy count."""
    result = await db_session.execute(
        select(func.count(ParkingSession.id)).where(
            ParkingSession.status == SessionStatus.ACTIVE,
        ),
    )
    active_count = result.scalar() or 0
    return {
        "active_sessions": active_count,
        "capacity": settings.PARKING_CAPACITY,
        "percent": occupancy_percent(active_count, settings.PARKING_CAPACITY),
        "timestamp": datetime.now(UTC),
    }


@router.get("/daily")
async def get_daily_stats(
    days: int = Query(7, ge=1, le=90),
    db_session: AsyncSession = Depends(get_session),
) -> list[dict[str, object]]:
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


@router.get("/model")
async def get_model_stats(
    days: int = Query(7, ge=1, le=90),
    db_session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    """Latency and confidence of the detector, per model version.

    Telemetry is written one row per confirmed event, so these numbers describe
    the inference that actually produced the vehicles in the database, not a
    benchmark run.
    """
    since = datetime.now(UTC) - timedelta(days=days)
    latency = DetectionEvent.inference_latency_ms
    result = await db_session.execute(
        select(
            DetectionEvent.model_name,
            func.count(DetectionEvent.id),
            func.avg(DetectionEvent.confidence),
            func.percentile_cont(0.5).within_group(latency),
            func.percentile_cont(0.95).within_group(latency),
        )
        .where(DetectionEvent.frame_timestamp >= since)
        .group_by(DetectionEvent.model_name),
    )
    rows = result.all()

    return {
        "window_days": days,
        "models": [
            {
                "model_name": name,
                "events": count,
                "avg_confidence": round(float(avg), 3) if avg is not None else None,
                "p50_latency_ms": round(float(p50), 1) if p50 is not None else None,
                "p95_latency_ms": round(float(p95), 1) if p95 is not None else None,
            }
            for name, count, avg, p50, p95 in rows
        ],
        "total_events": sum(row[1] for row in rows),
    }


@router.get("/colors")
async def get_colour_distribution(
    days: int = Query(30, ge=1, le=365),
    min_confidence: float = Query(
        MIN_REPORT_CONFIDENCE, ge=0.0, le=1.0,
        description="Below this colour confidence the label is treated as unknown.",
    ),
    db_session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    """Colour distribution, split by vehicle type.

    Only the six colours with enough labelled support are reported individually;
    the rest are aggregated into ``minority`` for display, never as a class.
    Rows with a low-confidence colour are excluded rather than guessed: a wrong
    count is worse than a smaller one when the number is going into a report.

    Splitting by type is not cosmetic: 41% of the black and 85% of the yellow
    crops are motorcycles, so a single distribution describes neither cars nor
    bikes.
    """
    since = datetime.now(UTC) - timedelta(days=days)
    result = await db_session.execute(
        select(Vehicle.vehicle_type, Vehicle.color, func.count(Vehicle.id))
        .where(
            Vehicle.created_at >= since,
            Vehicle.color_confidence >= min_confidence,
        )
        .group_by(Vehicle.vehicle_type, Vehicle.color),
    )

    by_type: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    for vehicle_type, color, count in result.all():
        bucket = report_bucket(color)
        key = vehicle_type or "unknown"
        per_type = by_type.setdefault(key, {})
        per_type[bucket] = per_type.get(bucket, 0) + count
        totals[bucket] = totals.get(bucket, 0) + count

    return {
        "window_days": days,
        "min_confidence": min_confidence,
        "reportable_colors": list(REPORTABLE_COLORS),
        "by_type": {t: colour_counts(c) for t, c in sorted(by_type.items())},
        "totals": colour_counts(totals),
        "total_vehicles": sum(totals.values()),
    }
