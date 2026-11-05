# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

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
            "TRUNCATE detection_events, parking_sessions, vehicles, "
            "authorized_vehicles "
            "RESTART IDENTITY CASCADE",
        ))

    yield db_session.get_session_factory()

    if db_session._engine is not None:
        db_session._engine.dispose()
    db_session._engine = None
    db_session._session_factory = None


@pytest.fixture
async def api(postgres_test_url):
    """HTTP client and async session factory bound to the throwaway database.

    The API's own engine is built at import time from the configured URL, which
    in a container setup points at the compose network. Overriding ``get_session``
    is what lets these tests run against the throwaway database from the host.
    """
    from httpx import ASGITransport, AsyncClient
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from backend.app.core.database import get_session
    from backend.app.main import app

    url = make_url(postgres_test_url).set(drivername="postgresql+asyncpg")
    engine = create_async_engine(
        url.render_as_string(hide_password=False), poolclass=NullPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _session():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_session] = _session

    async with engine.begin() as connection:
        await connection.execute(text("TRUNCATE users, cameras RESTART IDENTITY CASCADE"))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, factory

    app.dependency_overrides.pop(get_session, None)
    await engine.dispose()
