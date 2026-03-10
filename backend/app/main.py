"""FastAPI application factory for Legal-CoPilot backend.

Entrypoint: uvicorn app.main:app
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import engine
from app.utils.logging import configure_logging, get_logger

settings = get_settings()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    Startup:
        - Configure structured logging
        - Verify DB connectivity
    Shutdown:
        - Dispose DB connection pool
    """
    # Startup
    configure_logging(settings.log_level)
    logger.info(
        "startup",
        environment=settings.environment,
        service="vakil-backend",
    )

    # Verify DB is reachable (fail fast on misconfiguration)
    try:
        async with engine.connect() as conn:
            from sqlalchemy import text
            await conn.execute(text("SELECT 1"))
        logger.info("db_connected")
    except Exception as exc:
        logger.error("db_connection_failed", error=str(exc))
        raise

    yield

    # Shutdown
    await engine.dispose()
    logger.info("shutdown", service="vakil-backend")


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    application = FastAPI(
        title="Legal-CoPilot API",
        description=(
            "Case management and AI-assisted legal research platform "
            "for Maharashtra property lawyers."
        ),
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # -------------------------------------------------------------------------
    # CORS — restrict to frontend origin in production
    # -------------------------------------------------------------------------
    allowed_origins = (
        ["http://localhost:3000"]
        if not settings.is_production
        else ["https://vakil.yourdomain.com"]  # Update before going live
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,  # Required for HttpOnly cookie auth
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # -------------------------------------------------------------------------
    # Routers (v1) — registered here as services are built
    # -------------------------------------------------------------------------
    # from app.api.v1.router import api_router
    # application.include_router(api_router, prefix="/api/v1")

    # -------------------------------------------------------------------------
    # Health check endpoint — used by docker-compose healthcheck
    # -------------------------------------------------------------------------
    @application.get("/health", tags=["ops"], include_in_schema=False)
    async def health_check() -> JSONResponse:
        """Basic liveness probe."""
        return JSONResponse({"status": "ok", "service": "vakil-backend"})

    return application


app = create_app()