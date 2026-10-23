"""A throwaway PostgreSQL database for integration tests.

Skipped when no server is reachable, so the normal suite stays green on a
laptop. The database name is derived from the configured one and must end in
``_test``: a misconfigured URL must never be able to wipe real data.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from backend.app.core.config import settings

TEST_SUFFIX = "_test"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")
_SOFTWARE_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def postgres_test_url():
    base = make_url(settings.DATABASE_URL_SYNC)
    if base.get_backend_name() != "postgresql":
        pytest.skip("integration tests need PostgreSQL")

    name = f"{base.database}{TEST_SUFFIX}"
    assert name.endswith(TEST_SUFFIX) and _IDENTIFIER.match(name)

    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": name},
            ).scalar()
            if not exists:
                connection.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as error:  # no server, no permission, wrong host
        pytest.skip(f"postgres not available: {error}")
    finally:
        admin.dispose()

    url = base.set(database=name).render_as_string(hide_password=False)
    config = Config(str(_SOFTWARE_ROOT / "backend" / "alembic.ini"))
    config.set_main_option("script_location", str(_SOFTWARE_ROOT / "backend" / "alembic"))
    config.attributes["sqlalchemy_url"] = url
    command.upgrade(config, "head")

    yield url

    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    finally:
        admin.dispose()


@pytest.fixture
def db(postgres_test_url, monkeypatch):
    """A clean schema per test, bound to the app's sync session factory."""
    import backend.adapters.outbound.db.session as db_session

    monkeypatch.setattr(settings, "DATABASE_URL_SYNC", postgres_test_url)
    db_session._engine = None
    db_session._session_factory = None

    engine = db_session.get_engine()
    with engine.begin() as connection:
        connection.execute(text(
            "TRUNCATE detection_events, parking_sessions, vehicles "
            "RESTART IDENTITY CASCADE",
        ))

    yield db_session.get_session_factory()

    if db_session._engine is not None:
        db_session._engine.dispose()
    db_session._engine = None
    db_session._session_factory = None
