"""One transaction per use case.

The commit belongs here, not to the HTTP dependency: a use case knows when its
work is complete, and a response must not be built on a transaction that can
still fail.
"""
from __future__ import annotations

from collections.abc import Callable
from types import TracebackType

from sqlalchemy.orm import Session

from backend.adapters.outbound.db.repositories import (
    SqlAlchemySessionStore,
    SqlAlchemyVehicleStore,
)
from backend.domain.ports import SessionStore, VehicleStore


class SqlAlchemyUnitOfWork:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory
        self._session: Session | None = None

    def __enter__(self) -> SqlAlchemyUnitOfWork:
        self._session = self._session_factory()
        # Annotated as the port, not the concrete store: a mutable attribute is
        # invariant in a Protocol, so the concrete type would not satisfy it.
        self.vehicles: VehicleStore = SqlAlchemyVehicleStore(self._session)
        self.sessions: SessionStore = SqlAlchemySessionStore(self._session)
        return self

    def commit(self) -> None:
        if self._session is None:
            raise RuntimeError("unit of work used outside its context")
        self._session.commit()

    def rollback(self) -> None:
        if self._session is not None:
            self._session.rollback()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._session is None:
            return
        if exc_type is not None:
            self._session.rollback()
        self._session.close()
        self._session = None
