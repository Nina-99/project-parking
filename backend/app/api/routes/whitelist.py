# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""CRUD routes for authorized vehicles (Padrón de vehículos habilitados - RF-10)."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import require_role
from backend.app.core.database import get_session
from backend.app.models.authorized_vehicle import AuthorizedVehicle
from backend.app.models.user import UserRole
from backend.app.schemas.authorized_vehicle import (
    AuthorizedVehicleCreate,
    AuthorizedVehicleResponse,
    AuthorizedVehicleUpdate,
)

router = APIRouter()

# Operators and Admins can read whitelist, only Admin can write
can_read = require_role(UserRole.ADMIN, UserRole.OPERATOR)
admin_only = require_role(UserRole.ADMIN)


@router.get("", response_model=list[AuthorizedVehicleResponse])
async def list_authorized_vehicles(
    search: str | None = Query(None, description="Search by plate or owner"),
    active_only: bool = Query(False, description="Filter only active vehicles"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _actor: tuple[str, str] = Depends(can_read),
    db_session: AsyncSession = Depends(get_session),
) -> list[AuthorizedVehicleResponse]:
    """List all registered authorized vehicles."""
    query = select(AuthorizedVehicle)
    if active_only:
        query = query.where(AuthorizedVehicle.is_active.is_(True))
    if search and search.strip():
        term = f"%{search.strip()}%"
        query = query.where(
            AuthorizedVehicle.license_plate.ilike(term)
            | AuthorizedVehicle.owner_name.ilike(term)
        )

    query = query.order_by(AuthorizedVehicle.license_plate.asc()).offset(offset).limit(limit)
    result = await db_session.execute(query)
    return [row[0] if isinstance(row, tuple) else row for row in result.all()]


@router.post("", response_model=AuthorizedVehicleResponse, status_code=status.HTTP_201_CREATED)
async def create_authorized_vehicle(
    payload: AuthorizedVehicleCreate,
    _actor: tuple[str, str] = Depends(admin_only),
    db_session: AsyncSession = Depends(get_session),
) -> AuthorizedVehicleResponse:
    """Register a new vehicle in the authorized registry."""
    normalized_plate = payload.license_plate.strip().upper()
    vehicle = AuthorizedVehicle(
        license_plate=normalized_plate,
        owner_name=payload.owner_name,
        vehicle_type=payload.vehicle_type,
        color=payload.color,
        notes=payload.notes,
        is_active=payload.is_active,
    )
    db_session.add(vehicle)
    try:
        await db_session.commit()
        await db_session.refresh(vehicle)
    except IntegrityError as error:
        await db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"La placa '{normalized_plate}' ya se encuentra registrada en el padrón.",
        ) from error
    return vehicle


@router.get("/check/{license_plate}", response_model=dict[str, object])
async def check_authorization(
    license_plate: str,
    _actor: tuple[str, str] = Depends(can_read),
    db_session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    """Check if a license plate is authorized."""
    normalized_plate = license_plate.strip().upper()
    query = select(AuthorizedVehicle).where(
        func.upper(AuthorizedVehicle.license_plate) == normalized_plate
    )
    result = await db_session.execute(query)
    vehicle = result.scalar_one_or_none()

    if vehicle and vehicle.is_active:
        return {
            "authorized": True,
            "vehicle": AuthorizedVehicleResponse.model_validate(vehicle),
        }
    return {
        "authorized": False,
        "vehicle": AuthorizedVehicleResponse.model_validate(vehicle) if vehicle else None,
    }


@router.patch("/{vehicle_id}", response_model=AuthorizedVehicleResponse)
async def update_authorized_vehicle(
    vehicle_id: uuid.UUID,
    payload: AuthorizedVehicleUpdate,
    _actor: tuple[str, str] = Depends(admin_only),
    db_session: AsyncSession = Depends(get_session),
) -> AuthorizedVehicleResponse:
    """Update authorized vehicle attributes or active status."""
    vehicle = await db_session.get(AuthorizedVehicle, vehicle_id)
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vehículo autorizado no encontrado.",
        )

    if payload.owner_name is not None:
        vehicle.owner_name = payload.owner_name
    if payload.vehicle_type is not None:
        vehicle.vehicle_type = payload.vehicle_type
    if payload.color is not None:
        vehicle.color = payload.color
    if payload.notes is not None:
        vehicle.notes = payload.notes
    if payload.is_active is not None:
        vehicle.is_active = payload.is_active

    await db_session.commit()
    await db_session.refresh(vehicle)
    return vehicle


@router.delete("/{vehicle_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_authorized_vehicle(
    vehicle_id: uuid.UUID,
    _actor: tuple[str, str] = Depends(admin_only),
    db_session: AsyncSession = Depends(get_session),
) -> None:
    """Delete a vehicle from the authorized registry."""
    vehicle = await db_session.get(AuthorizedVehicle, vehicle_id)
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vehículo autorizado no encontrado.",
        )
    await db_session.delete(vehicle)
    await db_session.commit()
