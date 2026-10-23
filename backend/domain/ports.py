"""The ports the application needs, and nothing else.

Each protocol here exists because something below it needs to be replaced in a
test or lives in a different process. If a new protocol cannot answer "what
invariant does this protect?", it does not belong in this file.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from backend.domain.matching import OpenSession
from backend.domain.vehicle import VehicleAttributes


class VehicleStore(Protocol):
    def add(self, attributes: VehicleAttributes, image_path: str | None) -> str: ...


class SessionStore(Protocol):
    def open(self, vehicle_id: str, gate: str, at: datetime) -> str: ...

    def active(self) -> list[OpenSession]: ...

    def close(self, session_id: str, at: datetime, gate: str) -> None: ...


class EventPublisher(Protocol):
    def publish(self, payload: Mapping[str, object]) -> bool: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class UnitOfWork(Protocol):
    """One transaction, with the stores that share it."""

    vehicles: VehicleStore
    sessions: SessionStore

    def commit(self) -> None: ...
