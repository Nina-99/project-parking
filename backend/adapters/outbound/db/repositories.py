"""SQLAlchemy stores behind the application ports.

The ORM models stay here. The application receives domain values and identifiers,
never a mapped row, so a schema change does not reach the policy code.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.parking_session import ParkingSession, SessionStatus
from backend.app.models.vehicle import Vehicle
from backend.domain.errors import SessionNotActiveError, SessionNotFoundError
from backend.domain.matching import OpenSession, normalize_plate
from backend.domain.vehicle import Plate, VehicleAttributes


class SqlAlchemyVehicleStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self,
        attributes: VehicleAttributes,
        plate: Plate | None,
        image_path: str | None,
    ) -> str:
        row = Vehicle(
            license_plate=normalize_plate(plate.text) if plate else None,
            license_plate_confidence=plate.confidence if plate else None,
            vehicle_type=attributes.vehicle_type,
            vehicle_type_confidence=attributes.vehicle_type_confidence,
            color=attributes.color,
            color_confidence=attributes.color_confidence,
            entry_image_path=image_path,
        )
        self._session.add(row)
        self._session.flush()
        return str(row.id)


class SqlAlchemySessionStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def open(self, vehicle_id: str, gate: str, at: datetime) -> str:
        row = ParkingSession(
            vehicle_id=uuid.UUID(vehicle_id),
            entry_gate=gate,
            entry_time=at,
            status=SessionStatus.ACTIVE,
        )
        self._session.add(row)
        self._session.flush()
        return str(row.id)

    def active(self) -> list[OpenSession]:
        rows = self._session.execute(
            select(ParkingSession, Vehicle)
            .join(Vehicle, ParkingSession.vehicle_id == Vehicle.id)
            .where(ParkingSession.status == SessionStatus.ACTIVE)
        ).all()
        return [
            OpenSession(
                session_id=str(parking.id),
                vehicle_id=str(vehicle.id),
                entry_time=parking.entry_time,
                license_plate=vehicle.license_plate,
                plate_confidence=vehicle.license_plate_confidence,
            )
            for parking, vehicle in rows
        ]

    def close(
        self,
        session_id: str,
        at: datetime,
        *,
        gate: str | None = None,
        reason: str | None = None,
        operator: str | None = None,
    ) -> int:
        row = self._session.get(ParkingSession, uuid.UUID(session_id))
        if row is None:
            raise SessionNotFoundError(session_id)
        if row.status is not SessionStatus.ACTIVE:
            raise SessionNotActiveError(session_id)

        row.exit_gate = gate
        row.exit_time = at
        duration = 0
        if row.entry_time is not None:
            duration = max(0, int((at - row.entry_time).total_seconds()))
        row.duration_seconds = duration
        row.status = SessionStatus.COMPLETED
        row.close_reason = reason
        row.closed_by = operator
        self._session.flush()
        return duration
