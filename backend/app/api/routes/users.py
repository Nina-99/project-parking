# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Who may enter the system, and with which role.

Admin only. Three invariants are enforced here rather than in the UI, because the
UI is not the only client:

* an admin cannot delete itself (locking yourself out by accident is permanent);
* the last active admin cannot be deleted or demoted (a lot with no admin cannot
  create its next one);
* a password is never returned, and never stored in clear.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.auth import require_role
from backend.app.core.database import get_session
from backend.app.core.security import hash_password
from backend.app.models.user import ROLE_VALUES, User, UserRole

router = APIRouter()

MIN_PASSWORD_LENGTH = 8


class UserResponse(BaseModel):
    id: uuid.UUID
    username: str
    role: str
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None

    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=128)
    role: UserRole = UserRole.OPERATOR

    @field_validator("username")
    @classmethod
    def _normalise(cls, value: str) -> str:
        value = value.strip()
        if not value.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError("username may contain letters, digits, dot, dash and underscore")
        return value


class UserUpdate(BaseModel):
    role: UserRole | None = None
    is_active: bool | None = None
    password: str | None = Field(default=None, min_length=MIN_PASSWORD_LENGTH, max_length=128)


admin_only = require_role(UserRole.ADMIN)


async def _active_admins(session: AsyncSession) -> int:
    return await session.scalar(
        select(func.count()).select_from(User).where(
            User.role == UserRole.ADMIN.value, User.is_active.is_(True),
        )
    ) or 0


@router.get("", response_model=list[UserResponse])
async def list_users(
    _: tuple[str, str] = Depends(admin_only),
    session: AsyncSession = Depends(get_session),
) -> list[User]:
    rows = await session.scalars(select(User).order_by(User.username))
    return list(rows)


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate,
    _: tuple[str, str] = Depends(admin_only),
    session: AsyncSession = Depends(get_session),
) -> User:
    exists = await session.scalar(select(User).where(User.username == body.username))
    if exists is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="username already exists")

    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        role=body.role.value,
    )
    session.add(user)
    await session.flush()
    return user


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: uuid.UUID,
    body: UserUpdate,
    _actor: tuple[str, str] = Depends(admin_only),
    session: AsyncSession = Depends(get_session),
) -> User:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")

    losing_admin = (
        user.role == UserRole.ADMIN.value
        and (body.role is not None and body.role is not UserRole.ADMIN
             or body.is_active is False)
    )
    if losing_admin and await _active_admins(session) <= 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="the last active admin cannot be demoted or deactivated",
        )

    if body.role is not None:
        user.role = body.role.value
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.password is not None:
        user.password_hash = hash_password(body.password)
    await session.flush()
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: uuid.UUID,
    actor: tuple[str, str] = Depends(admin_only),
    session: AsyncSession = Depends(get_session),
) -> None:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")
    if user.username == actor[0]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="you cannot delete your own account")
    if user.role == UserRole.ADMIN.value and user.is_active and await _active_admins(session) <= 1:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="the last active admin cannot be deleted")
    await session.delete(user)


@router.get("/roles", response_model=list[str])
async def list_roles(_: tuple[str, str] = Depends(admin_only)) -> list[str]:
    """The role vocabulary, so the UI does not hardcode it."""
    return list(ROLE_VALUES)
