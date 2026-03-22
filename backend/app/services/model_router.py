"""Quota-aware model router — selects the best Gemini model based on remaining daily quota.

Before processing a document, the router:
  1. Pre-scans the input (page count, request type)
  2. Calculates how many API requests will be needed
  3. Checks Redis for how many requests have been used today per model
  4. Selects the highest-quality model that has enough remaining quota
  5. Falls back to lower-quality models if the preferred one is exhausted

Usage:
    from app.services.model_router import select_model, track_usage

    # Before starting OCR on a 41-page PDF:
    model = await select_model(task="vision_ocr", required_requests=9)

    # After each successful API call:
    await track_usage(model)

    # For a single LLM call (fact extraction, draft generation):
    model = await select_model(task="text_llm", required_requests=1)

Redis keys:
    gemini_quota:{model}:{YYYY-MM-DD} → integer counter, TTL 25 hours
"""

import datetime
from dataclasses import dataclass

import redis.asyncio as aioredis  # type: ignore[import]

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Model tier definitions — ordered from highest quality to lowest
# ---------------------------------------------------------------------------

@dataclass
class ModelTier:
    """A Gemini model with its free-tier quota limits."""
    model: str
    daily_limit: int   # max requests per day (free tier)
    rpm: int           # max requests per minute (free tier)
    quality: str       # human label for logging
    supports_vision: bool  # can process images


# Ordered: best quality first → fallback last
_MODEL_TIERS: list[ModelTier] = [
    ModelTier(
        model="gemini-2.5-flash",
        daily_limit=20,
        rpm=2,
        quality="best",
        supports_vision=True,
    ),
    ModelTier(
        model="gemini-2.0-flash",
        daily_limit=1500,
        rpm=15,
        quality="good",
        supports_vision=True,
    ),
    ModelTier(
        model="gemini-2.0-flash-lite",
        daily_limit=1500,
        rpm=30,
        quality="basic",
        supports_vision=True,
    ),
]


class QuotaExhaustedError(Exception):
    """Raised when all model tiers have exceeded their daily quota."""


def _redis_key(model: str) -> str:
    """Redis key for daily usage counter: gemini_quota:{model}:{date}."""
    today = datetime.date.today().isoformat()
    return f"gemini_quota:{model}:{today}"


async def _get_redis() -> aioredis.Redis:
    """Get a Redis connection."""
    return aioredis.from_url(settings.redis_url, decode_responses=True)


async def get_usage(model: str) -> int:
    """Get the number of requests used today for a specific model."""
    try:
        r = await _get_redis()
        val = await r.get(_redis_key(model))
        await r.aclose()
        return int(val) if val else 0
    except Exception:
        return 0  # If Redis is down, assume 0 (optimistic)


async def track_usage(model: str, count: int = 1) -> int:
    """Increment the daily usage counter for a model.

    Args:
        model: Gemini model name (e.g. "gemini-2.0-flash").
        count: Number of requests to add (default 1).

    Returns:
        New total usage count for today.
    """
    try:
        r = await _get_redis()
        key = _redis_key(model)
        new_count = await r.incrby(key, count)
        # Set TTL to 25 hours (ensures it expires even with timezone drift)
        await r.expire(key, 90_000)
        await r.aclose()
        return new_count
    except Exception as exc:
        logger.warning("quota_track_failed", model=model, error=str(exc))
        return -1


async def select_model(
    task: str = "text_llm",
    required_requests: int = 1,
    require_vision: bool = False,
) -> str:
    """Select the best available Gemini model based on remaining daily quota.

    Strategy:
      - Try highest-quality model first
      - If remaining quota < required_requests, skip to next tier
      - If all tiers exhausted, raise QuotaExhaustedError

    Args:
        task: Description for logging (e.g. "vision_ocr", "fact_extraction").
        required_requests: Number of API calls this job will make.
        require_vision: If True, only consider models that support vision.

    Returns:
        Model name string (e.g. "gemini-2.0-flash").

    Raises:
        QuotaExhaustedError: When no model has enough remaining quota.
    """
    tiers = _MODEL_TIERS
    if require_vision:
        tiers = [t for t in tiers if t.supports_vision]

    selection_log = []

    for tier in tiers:
        used = await get_usage(tier.model)
        remaining = tier.daily_limit - used

        selection_log.append({
            "model": tier.model,
            "quality": tier.quality,
            "used": used,
            "daily_limit": tier.daily_limit,
            "remaining": remaining,
            "sufficient": remaining >= required_requests,
        })

        if remaining >= required_requests:
            logger.info(
                "model_selected",
                task=task,
                model=tier.model,
                quality=tier.quality,
                required=required_requests,
                used_today=used,
                remaining=remaining,
                daily_limit=tier.daily_limit,
            )
            return tier.model

    # No model has enough quota
    logger.error(
        "all_models_exhausted",
        task=task,
        required=required_requests,
        tiers=selection_log,
    )
    raise QuotaExhaustedError(
        f"All Gemini models exhausted for today. "
        f"Need {required_requests} requests but no model has enough quota remaining. "
        f"Tier status: {selection_log}"
    )


async def get_all_quotas() -> list[dict]:
    """Return current quota status for all model tiers (for health/admin endpoints).

    Returns:
        List of dicts with model, daily_limit, used, remaining, quality.
    """
    result = []
    for tier in _MODEL_TIERS:
        used = await get_usage(tier.model)
        result.append({
            "model": tier.model,
            "quality": tier.quality,
            "daily_limit": tier.daily_limit,
            "rpm": tier.rpm,
            "used_today": used,
            "remaining": tier.daily_limit - used,
            "supports_vision": tier.supports_vision,
        })
    return result
