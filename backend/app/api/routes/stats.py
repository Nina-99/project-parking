# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Read-only aggregates over the stored vehicles and sessions."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import require_role
from backend.app.core.config import settings
from backend.app.core.database import get_session
from backend.app.models.detection_event import DetectionEvent
from backend.app.models.parking_session import ParkingSession, SessionStatus
from backend.app.models.user import UserRole
from backend.app.models.validation_decision import ValidationDecision
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
from backend.domain.validation import DECISIONS

router = APIRouter()

# Inference telemetry is diagnostic, not operational: the owner reads the lot,
# the admin reads whether the detector is degrading.
admin_only = require_role(UserRole.ADMIN)


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
    response: Response,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    gate: str | None = Query(None, description="Filter by entry or exit gate"),
    from_date: datetime | None = Query(None, description="Start date filter (entry_time >= from_date)"),
    to_date: datetime | None = Query(None, description="End date filter (entry_time <= to_date)"),
    search: str | None = Query(None, description="Search query for license_plate, vehicle_type or color"),
    status: str | None = Query(None, description="Filter by status: active, completed, cancelled"),
    db_session: AsyncSession = Depends(get_session),
) -> list[SessionResponse]:
    """List parking sessions with optional filtering and pagination.

    The total matching records before pagination travels in ``X-Total-Count``.
    """
    conditions = []

    if status:
        try:
            target_status = SessionStatus(status)
            conditions.append(ParkingSession.status == target_status)
        except ValueError:
            pass
    else:
        # Default behavior: completed sessions
        conditions.append(ParkingSession.status == SessionStatus.COMPLETED)

    if gate:
        conditions.append(
            (ParkingSession.entry_gate == gate) | (ParkingSession.exit_gate == gate)
        )

    if from_date:
        conditions.append(ParkingSession.entry_time >= from_date)

    if to_date:
        conditions.append(ParkingSession.entry_time <= to_date)

    if search and search.strip():
        term = f"%{search.strip()}%"
        conditions.append(
            Vehicle.license_plate.ilike(term)
            | Vehicle.vehicle_type.ilike(term)
            | Vehicle.color.ilike(term)
        )

    count_stmt = (
        select(func.count(ParkingSession.id))
        .join(Vehicle, ParkingSession.vehicle_id == Vehicle.id)
        .where(*conditions)
    )
    total = (await db_session.execute(count_stmt)).scalar() or 0
    response.headers["X-Total-Count"] = str(total)

    order_col = (
        ParkingSession.exit_time.desc()
        if not status or status == SessionStatus.COMPLETED.value
        else ParkingSession.entry_time.desc()
    )

    query_stmt = (
        select(ParkingSession, Vehicle)
        .join(Vehicle, ParkingSession.vehicle_id == Vehicle.id)
        .where(*conditions)
        .order_by(order_col)
        .offset(offset)
        .limit(limit)
    )
    result = await db_session.execute(query_stmt)
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
    """Aggregated stats for the last N days.

    The day bucket is the site's local day. A vehicle entering at 21:00 in La Paz
    is a movement of that day; grouping in UTC would move it to the next one.
    """
    since = datetime.now(UTC) - timedelta(days=days)
    local_day = func.date(
        func.timezone(settings.REPORT_TIMEZONE, ParkingSession.entry_time)
    )
    result = await db_session.execute(
        select(
            local_day.label("date"),
            func.count(ParkingSession.id).label("total"),
            func.avg(ParkingSession.duration_seconds).label("avg_duration"),
        )
        .where(ParkingSession.entry_time >= since)
        .group_by(local_day)
        .order_by(local_day),
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


@router.get("/hourly")
async def get_hourly_stats(
    days: int = Query(7, ge=1, le=90),
    db_session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    """Entries by hour of day, over the window.

    RF-12 asks for the temporal distribution of entries. The daily aggregate
    answers "how many", not "when": a lot with the same daily total can peak at
    07:00 or at 18:00, and the staffing decision is different. Hours with no
    entries are returned as zero so the chart has a fixed 24-slot axis.

    The hour is the site's local hour. The database stores timestamptz, so the
    conversion happens in SQL with ``REPORT_TIMEZONE``; grouping the raw UTC
    column would put the evening peak four hours late in La Paz.
    """
    since = datetime.now(UTC) - timedelta(days=days)
    local_hour = func.extract(
        "hour",
        func.timezone(settings.REPORT_TIMEZONE, ParkingSession.entry_time),
    )
    result = await db_session.execute(
        select(local_hour.label("hour"), func.count(ParkingSession.id))
        .where(ParkingSession.entry_time >= since)
        .group_by(local_hour)
        .order_by(local_hour),
    )
    counts = {int(row[0]): int(row[1]) for row in result.all()}
    return {
        "window_days": days,
        "timezone": settings.REPORT_TIMEZONE,
        "hours": [
            {"hour": value, "vehicles": counts.get(value, 0)}
            for value in range(24)
        ],
    }


@router.get("/model", dependencies=[Depends(admin_only)])
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


@router.get("/quality")
async def get_quality_stats(
    days: int = Query(7, ge=1, le=90),
    db_session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    """How often the operator had to disagree with the machine.

    This is the operational complement of the frozen-set evaluation: it measures
    the system that is actually running, not the one that was benchmarked. Two
    rates matter and neither is an accuracy claim:

    * ``plate_read_rate`` is the share of registered events that carried a plate
      at all. It says how much of the lot is identifiable without a human.
    * ``correction_rate`` is the share of *reviewed* events whose plate the
      operator had to fix. It is the honest measure of the OCR gap in operation,
      because it only counts events someone actually looked at.

    ``unmatched_exits`` counts exits that could not be paired with an open
    session; those are the ones that move a vehicle out of the lot only after a
    person closes it.
    """
    since = datetime.now(UTC) - timedelta(days=days)

    events_total = await db_session.scalar(
        select(func.count(Vehicle.id)).where(Vehicle.created_at >= since)
    ) or 0
    events_with_plate = await db_session.scalar(
        select(func.count(Vehicle.id)).where(
            Vehicle.created_at >= since,
            Vehicle.license_plate.is_not(None),
        )
    ) or 0
    unmatched_exits = await db_session.scalar(
        select(func.count(DetectionEvent.id)).where(
            DetectionEvent.frame_timestamp >= since,
            DetectionEvent.direction == "exit",
            DetectionEvent.session_id.is_(None),
        )
    ) or 0

    decision_rows = await db_session.execute(
        select(ValidationDecision.decision, func.count(ValidationDecision.id))
        .where(ValidationDecision.created_at >= since)
        .group_by(ValidationDecision.decision)
    )
    by_decision = {decision: 0 for decision in DECISIONS}
    for decision, count in decision_rows.all():
        if decision in by_decision:
            by_decision[decision] = count

    reviewed = sum(by_decision.values())

    operator_rows = await db_session.execute(
        select(
            ValidationDecision.operator,
            func.count(ValidationDecision.id),
            func.sum(case(
                (ValidationDecision.decision == "plate_corrected", 1),
                else_=0,
            )),
        )
        .where(ValidationDecision.created_at >= since)
        .group_by(ValidationDecision.operator)
        .order_by(func.count(ValidationDecision.id).desc())
    )

    return {
        "window_days": days,
        "events_total": events_total,
        "events_with_plate": events_with_plate,
        "plate_read_rate": (
            round(events_with_plate / events_total, 3) if events_total else None
        ),
        "unmatched_exits": unmatched_exits,
        "decisions_total": reviewed,
        "by_decision": by_decision,
        "correction_rate": (
            round(by_decision["plate_corrected"] / reviewed, 3) if reviewed else None
        ),
        "operators": [
            {
                "operator": operator,
                "decisions": count,
                "corrected": int(corrected or 0),
            }
            for operator, count, corrected in operator_rows.all()
        ],
    }
