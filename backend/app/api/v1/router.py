"""API v1 router — aggregates all route modules."""

from fastapi import APIRouter

from app.api.v1 import admin, drafting

api_router = APIRouter()

api_router.include_router(
    drafting.router,
    prefix="/drafting",
    tags=["drafting"],
)
api_router.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"],
)