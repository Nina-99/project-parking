# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Synchronous engine for writes outside the request event loop.

The API reads with async SQLAlchemy. Writes go through the application's sync
unit of work, executed in a worker thread: at 1-7 vehicles per ten minutes there
is nothing to gain from a second repository implementation, and one write path is
worth more than an async write path that nobody exercises.

The engine is built on first use, not at import: importing a module must not open
a pool in processes that only wanted its names.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.DATABASE_URL_SYNC, pool_pre_ping=True, future=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _session_factory


@contextmanager
def sync_session() -> Iterator[Session]:
    """One transaction per call: commit on success, roll back on anything else."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
