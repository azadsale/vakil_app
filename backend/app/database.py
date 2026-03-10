"""Async SQLAlchemy engine and session factory.

Provides:
- AsyncEngine (shared, pool-managed)
- AsyncSessionLocal (session factory)
- get_db() — FastAPI dependency for request-scoped sessions
- set_rls_user_id() — sets app.current_user_id for PostgreSQL RLS isolation
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from app.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# Engine
# Use NullPool for test environments to avoid connection leaks across tests.
# ---------------------------------------------------------------------------
_engine_kwargs: dict = {
    "echo": settings.db_echo,
    "pool_size": settings.db_pool_size,
    "max_overflow": settings.db_max_overflow,
    "pool_timeout": settings.db_pool_timeout,
    "pool_pre_ping": True,  # reconnect on stale connections
}

engine = create_async_engine(
    settings.database_url.get_secret_value(),
    **_engine_kwargs,
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# RLS Helper
# ---------------------------------------------------------------------------
async def set_rls_user_id(session: AsyncSession, user_id: str) -> None:
    """Set the PostgreSQL session variable used by RLS policies.

    Must be called at the start of every request-scoped session after
    authentication, before any DML operations.

    Args:
        session: Active async DB session.
        user_id: UUID string of the authenticated lawyer.
    """
    await session.execute(
        # Use parameterized set_config to prevent SQL injection
        # set_config(parameter, value, is_local=true) — local to transaction
        __import__("sqlalchemy").text(
            "SELECT set_config('app.current_user_id', :uid, true)"
        ),
        {"uid": user_id},
    )


# ---------------------------------------------------------------------------
# FastAPI Dependency
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a request-scoped async DB session.

    Usage:
        @router.get("/")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Startup / Shutdown helpers (used in app lifespan)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan_db():
    """Context manager for engine lifecycle in tests or standalone scripts."""
    try:
        yield engine
    finally:
        await engine.dispose()


async def create_all_tables() -> None:
    """Create all tables from SQLModel metadata.

    For use in tests only — production uses Alembic migrations.
    """
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def drop_all_tables() -> None:
    """Drop all tables — test teardown only."""
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)