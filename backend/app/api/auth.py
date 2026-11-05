# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Operator session: a signed cookie, no server-side session store.

Plates are personal data, so the read API is not public. The token is a JWT in
an httpOnly cookie: the browser never sees it from JavaScript and the WebSocket
handshake carries it automatically.

Credentials live in the ``users`` table, one row per person, each with a role.
The environment only bootstraps the first administrator: an operator of a parking
lot is a person who can be replaced, not a constant in a config file.
"""
from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.adapters.outbound.cache.rate_limit import RedisRateLimiter
from backend.app.api.dependencies import get_rate_limiter
from backend.app.core.config import settings
from backend.app.core.database import get_session
from backend.app.core.security import hash_password, verify_password
from backend.app.models.user import User, UserRole

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class OperatorResponse(BaseModel):
    username: str
    role: str = UserRole.OPERATOR.value


# A real hash of a random password: verifying it costs the same as a real user,
# so a wrong username cannot be told apart from a wrong password by timing.
_DUMMY_HASH = hash_password(secrets.token_hex(16))


def issue_token(username: str, role: str = UserRole.OPERATOR.value) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": username,
        "role": role,
        "iat": now,
        "exp": now + timedelta(seconds=settings.AUTH_TOKEN_TTL_SECONDS),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


def verify_token(token: str | None) -> str:
    """The username in a valid token, or 401."""
    return verify_claims(token)[0]


def verify_claims(token: str | None) -> tuple[str, str]:
    """(username, role) from a valid token, or 401.

    A token minted before roles existed has no ``role`` claim. It is treated as an
    operator: the narrowest role that can still do the job, never admin.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="not authenticated",
        )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid session",
        ) from error
    username = payload.get("sub")
    if not isinstance(username, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid session",
        )
    role = payload.get("role")
    if role not in {member.value for member in UserRole}:
        role = UserRole.OPERATOR.value
    return username, role


def require_operator(
    session: str | None = Cookie(default=None, alias=settings.AUTH_COOKIE_NAME),
) -> str:
    """Dependency for every route that exposes stored vehicle data."""
    return verify_token(session)


def require_role(*allowed: UserRole):
    """Dependency factory: the caller's role must be in ``allowed``.

    Reads the same cookie as ``require_operator``, so it can be used on a single
    endpoint inside a router already protected by that dependency.
    """

    def dependency(
        session: str | None = Cookie(default=None, alias=settings.AUTH_COOKIE_NAME),
    ) -> tuple[str, str]:
        username, role = verify_claims(session)
        if role not in {item.value for item in allowed}:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="your role cannot perform this action",
            )
        return username, role

    return dependency


async def seed_first_admin(session: AsyncSession) -> None:
    """Create the first admin from the environment when the table is empty.

    This is a bootstrap, not a fallback: once one user exists the environment
    password is never consulted again, so rotating it cannot resurrect access.
    """
    try:
        count = await session.scalar(select(func.count()).select_from(User))
    except Exception:  # table missing (migrations not run yet) or DB down
        return
    if count:
        return

    session.add(User(
        username=settings.OPERATOR_USER,
        password_hash=hash_password(settings.OPERATOR_PASSWORD),
        role=UserRole.ADMIN.value,
    ))
    await session.commit()


async def login_rate_limit(
    request: Request,
    limiter: RedisRateLimiter = Depends(get_rate_limiter),
) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if not await limiter.allow(f"rate:login:{client_ip}"):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="too many login attempts",
            headers={"Retry-After": str(settings.LOGIN_RATE_WINDOW_SECONDS)},
        )


@router.post(
    "/login",
    response_model=OperatorResponse,
    dependencies=[Depends(login_rate_limit)],
)
async def login(
    body: LoginRequest,
    response: Response,
    session: AsyncSession = Depends(get_session),
) -> OperatorResponse:
    user = await session.scalar(select(User).where(User.username == body.username))

    # Verify a dummy hash when the user does not exist so the response time does
    # not reveal which usernames are real.
    stored = user.password_hash if user is not None else _DUMMY_HASH
    password_ok = verify_password(body.password, stored)
    if user is None or not user.is_active or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
        )

    user.last_login_at = datetime.now(UTC)

    response.set_cookie(
        settings.AUTH_COOKIE_NAME,
        issue_token(user.username, user.role),
        max_age=settings.AUTH_TOKEN_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=settings.AUTH_COOKIE_SECURE,
        path="/",
    )
    return OperatorResponse(username=user.username, role=user.role)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response) -> None:
    response.delete_cookie(settings.AUTH_COOKIE_NAME, path="/")


@router.get("/me", response_model=OperatorResponse)
def me(
    session: str | None = Cookie(default=None, alias=settings.AUTH_COOKIE_NAME),
) -> OperatorResponse:
    username, role = verify_claims(session)
    return OperatorResponse(username=username, role=role)
