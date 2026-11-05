# SPDX-FileCopyrightText: 2026 Nina Marco <marconina999@gmail.com>
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Alembic migration environment."""
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

import backend.app.models  # noqa: F401  (registers the metadata)
from backend.app.core.config import settings
from backend.app.core.database import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """The synchronous driver is psycopg 3, not psycopg2.

    The async URL cannot be handed to Alembic as-is, and stripping ``+asyncpg``
    leaves a bare ``postgresql://`` that SQLAlchemy resolves to psycopg2 -- which
    is not a dependency. Prefer the explicit sync URL and normalise either form.
    """
    override = config.attributes.get("sqlalchemy_url")
    url = str(override or settings.DATABASE_URL_SYNC)
    return url.replace("+asyncpg", "+psycopg")


def run_migrations_offline() -> None:
    url = _database_url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    url = _database_url()
    connectable = engine_from_config(
        {"sqlalchemy.url": url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
