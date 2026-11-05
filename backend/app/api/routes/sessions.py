# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Manual session close.

The operator sees a vehicle leave that the model could not prove, and closes the
session. The reason and the operator travel with the row: an audited correction
is worth more than a silent one.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.adapters.outbound.bus.redis_publisher import RedisEventPublisher
from backend.adapters.outbound.db.unit_of_work import SqlAlchemyUnitOfWork
from backend.adapters.outbound.time import SystemClock
from backend.app.api.auth import require_role
from backend.app.api.dependencies import get_clock, get_publisher, get_uow_factory
from backend.app.models.user import UserRole
from backend.application.close_session import CloseOutcome, close_session

router = APIRouter()

# Closing a session changes the occupancy count: only the people who run the
# lot may do it. An owner reads reports; it does not correct the record.
operator_only = require_role(UserRole.OPERATOR, UserRole.ADMIN)


class CloseRequest(BaseModel):
    reason: str = Field("manual", min_length=1, max_length=50)


class CloseResponse(BaseModel):
    session_id: uuid.UUID
    duration_seconds: int
    reason: str
    operator: str


@router.post("/{session_id}/close", response_model=CloseResponse)
async def close(
    session_id: uuid.UUID,
    body: CloseRequest,
    actor: tuple[str, str] = Depends(operator_only),
    uow_factory: Callable[[], SqlAlchemyUnitOfWork] = Depends(get_uow_factory),
    publisher: RedisEventPublisher = Depends(get_publisher),
    clock: SystemClock = Depends(get_clock),
) -> CloseResponse:
    """Close an active session by hand. Idempotence is not granted: closing an
    already closed session is a 409, because rewriting its exit is worse than
    telling the operator it was already done."""

    def run() -> CloseOutcome:
        with uow_factory() as uow:
            return close_session(
                session_id=str(session_id),
                operator=actor[0],
                reason=body.reason,
                uow=uow,
                publisher=publisher,
                clock=clock,
            )

    outcome = await run_in_threadpool(run)

    return CloseResponse(
        session_id=uuid.UUID(outcome.session_id),
        duration_seconds=outcome.duration_seconds,
        reason=outcome.reason,
        operator=outcome.operator,
    )
