"""Alembic environment configuration for async SQLAlchemy + SQLModel.

Supports:
- Async engine (asyncpg driver)
- SQLModel metadata autogenerate
- pgvector type rendering
- Environment variable driven DATABASE_URL
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import all models so Alembic can detect schema changes via autogenerate
from app.config import get_settings
from app.models import (  # noqa: F401  # ensure models are registered
    Case,
    Document,
    DocumentChunk,
    Hearing,
    Party,
    User,
)
from sqlmodel import SQLModel

# Alembic Config object — access to values in alembic.ini
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# SQLModel metadata for autogenerate
target_metadata = SQLModel.metadata

# Override sqlalchemy.url from environment (never hardcode credentials)
settings = get_settings()
config.set_main_option(
    "sqlalchemy.url",
    settings.database_url.get_secret_value(),
)


def include_object(object, name, type_, reflected, compare_to):  # type: ignore[override]
    """Filter objects for autogenerate.

    Exclude PostGIS/pgvector system tables from migration detection.
    """
    if type_ == "table" and name in {"spatial_ref_sys"}:
        return False
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generate SQL script without DB connection).

    Useful for generating migration scripts to review before applying.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Execute migrations on an active connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        # Render pgvector type in autogenerate
        user_module_prefix="pgvector.sqlalchemy.",
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # No pool needed for migration runs
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using async engine."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()