# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The operator validation queue.

The worker confirms a gate event; the operator confirms the event is right. This
endpoint answers "what happened recently that nobody has reviewed yet?" and the
decision endpoint records the answer. Both halves of RF-10 live here: the system
classifies, the person is accountable for the classification.

The queue is server-backed on purpose. A socket-only queue would lose the events
that arrived while the panel was closed, and an operator who confirms a plate
would leave no trace of having done so.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from backend.adapters.outbound.db.unit_of_work import SqlAlchemyUnitOfWork
from backend.adapters.outbound.storage.clips import resolve_clip_path
from backend.adapters.outbound.time import SystemClock
from backend.app.api.auth import require_role
from backend.app.api.dependencies import get_clock, get_uow_factory
from backend.app.core.config import settings
from backend.app.core.database import get_session
from backend.app.models.authorized_vehicle import AuthorizedVehicle
from backend.app.models.detection_event import DetectionEvent
from backend.app.models.user import UserRole
from backend.app.models.validation_decision import ValidationDecision
from backend.app.models.vehicle import Vehicle
from backend.app.schemas.vehicle import PlateCharacter
from backend.application.decide_validation import (
    ValidationOutcome,
    record_validation,
)
from backend.domain.validation import DECISIONS

router = APIRouter()

operator_only = require_role(UserRole.OPERATOR, UserRole.ADMIN)

# Strips separators on both sides of the comparison: the registry stores what
# the administrator typed, the OCR stores its own punctuation.
_NORMALIZED_PLATE = func.regexp_replace(
    func.upper(AuthorizedVehicle.license_plate), "[^A-Z0-9]", "", "g",
)


class PendingValidation(BaseModel):
    vehicle_id: uuid.UUID
    camera_id: str | None
    direction: str | None
    observed_at: datetime | None
    created_at: datetime
    license_plate: str | None
    plate_confidence: float | None
    vehicle_type: str | None
    vehicle_type_confidence: float | None
    color: str | None
    color_confidence: float | None
    image_url: str | None
    clip_url: str | None = None
    plate_characters: list[PlateCharacter] | None = None
    authorized: bool | None
    session_id: uuid.UUID | None


class DecisionRequest(BaseModel):
    decision: Literal["confirmed", "plate_corrected", "dismissed", "session_closed"]
    corrected_plate: str | None = Field(default=None, max_length=20)
    session_id: uuid.UUID | None = None
    notes: str | None = Field(default=None, max_length=500)


class DecisionResponse(BaseModel):
    vehicle_id: uuid.UUID
    decision: str
    original_plate: str | None
    corrected_plate: str | None
    operator: str


@router.get("/pending", response_model=list[PendingValidation])
async def pending_validations(
    hours: float = Query(8.0, ge=0.5, le=168.0),
    limit: int = Query(50, ge=1, le=200),
    db_session: AsyncSession = Depends(get_session),
) -> list[PendingValidation]:
    """Gate events inside the window that no operator has decided on yet.

    The newest detection per vehicle supplies the direction and camera: an entry
    and an unmatched exit look identical on the vehicle row alone.
    """
    since = datetime.now(UTC) - timedelta(hours=hours)

    ranked = (
        select(
            DetectionEvent.vehicle_id.label("vehicle_id"),
            DetectionEvent.direction.label("direction"),
            DetectionEvent.camera_id.label("camera_id"),
            DetectionEvent.session_id.label("session_id"),
            DetectionEvent.frame_timestamp.label("observed_at"),
            func.row_number().over(
                partition_by=DetectionEvent.vehicle_id,
                order_by=DetectionEvent.frame_timestamp.desc(),
            ).label("rank"),
        )
        .where(DetectionEvent.vehicle_id.is_not(None))
        .subquery()
    )

    authorized = (
        select(AuthorizedVehicle.id).where(
            func.upper(Vehicle.license_plate) == _NORMALIZED_PLATE,
            AuthorizedVehicle.is_active.is_(True),
        ).exists()
    )

    stmt = (
        select(
            Vehicle,
            ranked.c.direction,
            ranked.c.camera_id,
            ranked.c.session_id,
            ranked.c.observed_at,
            authorized.label("authorized"),
        )
        .outerjoin(
            ranked,
            (ranked.c.vehicle_id == Vehicle.id) & (ranked.c.rank == 1),
        )
        .outerjoin(ValidationDecision, ValidationDecision.vehicle_id == Vehicle.id)
        .where(
            ValidationDecision.id.is_(None),
            Vehicle.created_at >= since,
        )
        .order_by(Vehicle.created_at.desc())
        .limit(limit)
    )

    rows = (await db_session.execute(stmt)).all()
    return [
        PendingValidation(
            vehicle_id=vehicle.id,
            camera_id=camera_id,
            direction=direction,
            observed_at=observed_at,
            created_at=vehicle.created_at,
            license_plate=vehicle.license_plate,
            plate_confidence=vehicle.license_plate_confidence,
            vehicle_type=vehicle.vehicle_type,
            vehicle_type_confidence=vehicle.vehicle_type_confidence,
            color=vehicle.color,
            color_confidence=vehicle.color_confidence,
            image_url=(
                f"/api/v1/vehicles/{vehicle.id}/image"
                if vehicle.entry_image_path else None
            ),
            clip_url=(
                f"/api/v1/vehicles/{vehicle.id}/clip"
                if resolve_clip_path(
                    str(vehicle.id), media_root=Path(settings.ENTRY_IMAGE_DIR),
                ) else None
            ),
            plate_characters=vehicle.plate_characters,
            authorized=None if vehicle.license_plate is None else bool(is_authorized),
            session_id=session_id,
        )
        for vehicle, direction, camera_id, session_id, observed_at, is_authorized in rows
    ]


@router.post("/{vehicle_id}/decision", response_model=DecisionResponse)
async def decide(
    vehicle_id: uuid.UUID,
    body: DecisionRequest,
    actor: tuple[str, str] = Depends(operator_only),
    uow_factory: Callable[[], SqlAlchemyUnitOfWork] = Depends(get_uow_factory),
    clock: SystemClock = Depends(get_clock),
) -> DecisionResponse:
    """Record what the operator decided. The audit row is append-only."""

    def run() -> ValidationOutcome:
        with uow_factory() as uow:
            return record_validation(
                vehicle_id=str(vehicle_id),
                decision=body.decision,
                operator=actor[0],
                uow=uow,
                clock=clock,
                corrected_plate=body.corrected_plate,
                session_id=str(body.session_id) if body.session_id else None,
                notes=body.notes,
            )

    outcome = await run_in_threadpool(run)

    return DecisionResponse(
        vehicle_id=uuid.UUID(outcome.vehicle_id),
        decision=outcome.decision,
        original_plate=outcome.original_plate,
        corrected_plate=outcome.corrected_plate,
        operator=outcome.operator,
    )


__all__ = ["router", "DECISIONS"]
