"""Operator session: a signed cookie, no server-side session store.

Plates are personal data, so the read API is not public. The token is a JWT in
an httpOnly cookie: the browser never sees it from JavaScript and the WebSocket
handshake carries it automatically.

This is the minimum that is honest. The credential lives in the environment and
is compared in constant time; a deployment behind TLS should set
``AUTH_COOKIE_SECURE=true`` and replace ``OPERATOR_PASSWORD`` with a rotated
secret.
"""
from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from pydantic import BaseModel

from backend.app.core.config import settings

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class OperatorResponse(BaseModel):
    username: str


def issue_token(username: str) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + timedelta(seconds=settings.AUTH_TOKEN_TTL_SECONDS),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


def verify_token(token: str | None) -> str:
    """The username in a valid token, or 401."""
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
    return username


def require_operator(
    session: str | None = Cookie(default=None, alias=settings.AUTH_COOKIE_NAME),
) -> str:
    """Dependency for every route that exposes stored vehicle data."""
    return verify_token(session)


@router.post("/login", response_model=OperatorResponse)
def login(body: LoginRequest, response: Response) -> OperatorResponse:
    # Compare both values without short-circuiting: a timing difference would
    # tell an attacker whether the username exists.
    user_ok = secrets.compare_digest(body.username, settings.OPERATOR_USER)
    password_ok = secrets.compare_digest(body.password, settings.OPERATOR_PASSWORD)
    if not (user_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
        )

    response.set_cookie(
        settings.AUTH_COOKIE_NAME,
        issue_token(body.username),
        max_age=settings.AUTH_TOKEN_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=settings.AUTH_COOKIE_SECURE,
        path="/",
    )
    return OperatorResponse(username=body.username)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response) -> None:
    response.delete_cookie(settings.AUTH_COOKIE_NAME, path="/")


@router.get("/me", response_model=OperatorResponse)
def me(username: str = Depends(require_operator)) -> OperatorResponse:
    return OperatorResponse(username=username)
