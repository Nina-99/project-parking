"""Synchronous database access for the Celery worker.

The API is async; Celery tasks are not. The worker therefore uses the sync driver
configured as ``DATABASE_URL_SYNC``.

The engine is built on first use, not at import: importing a task module must not
require a database driver to be installed, and it must not open a pool in
processes that only wanted the task names.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import settings

_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.DATABASE_URL_SYNC, pool_pre_ping=True, future=True)
    return _engine


def get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _session_factory


@contextmanager
def sync_session() -> Iterator[Session]:
    """One transaction per task: commit on success, roll back on anything else."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
