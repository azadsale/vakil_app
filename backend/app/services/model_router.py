"""Quota-aware model router with multi-key rotation.

Before processing a document, the router:
  1. Pre-scans the input (page count, request type)
  2. Calculates how many API requests will be needed
  3. Checks Redis for how many requests have been used today per model per key
  4. Selects the best (model, API key) combination that has enough quota
  5. Falls back to lower-quality models or other keys if the preferred one is exhausted

Multi-key support:
    Set GEMINI_API_KEY=key1,key2,key3 in .env
    Each key gets its own 20 req/day quota for gemini-2.5-flash.
    3 keys = 60 req/day — enough for ~10 full document uploads.

Usage:
    from app.services.model_router import select_model, track_usage

    # Before starting OCR on a 41-page PDF:
    model, api_key = await select_model(task="vision_ocr", required_requests=9)

    # After each successful API call:
    await track_usage(model, api_key_index=0)

    # For a single LLM call (fact extraction, draft generation):
    model, api_key = await select_model(task="text_llm", required_requests=1)

Redis keys:
    gemini_quota:{model}:key{index}:{YYYY-MM-DD} → integer counter, TTL 25 hours
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
    daily_limit: int   # max requests per day (free tier) PER API KEY
    rpm: int           # max requests per minute (free tier)
    quality: str       # human label for logging
    supports_vision: bool  # can process images


# Ordered: best quality first → fallback last
# NOTE: gemini-2.0-flash and gemini-2.0-flash-lite return limit=0 for some
# free-tier API keys. Only include models confirmed to work with the key.
# The router will auto-skip any model that returns 429 and mark it exhausted.
_MODEL_TIERS: list[ModelTier] = [
    ModelTier(
        model="gemini-2.5-flash",
        daily_limit=20,    # confirmed working — 20 req/day free PER KEY
        rpm=10,            # 10 RPM for 2.5-flash
        quality="best",
        supports_vision=True,
    ),
    ModelTier(
        model="gemini-2.0-flash",
        daily_limit=1500,  # may be 0 for some API keys
        rpm=15,
        quality="good",
        supports_vision=True,
    ),
    ModelTier(
        model="gemini-2.0-flash-lite",
        daily_limit=1500,  # may be 0 for some API keys
        rpm=30,
        quality="basic",
        supports_vision=True,
    ),
]


class QuotaExhaustedError(Exception):
    """Raised when all model tiers (across all keys) have exceeded their daily quota."""


def _redis_key(model: str, key_index: int = 0) -> str:
    """Redis key for daily usage counter: gemini_quota:{model}:key{index}:{date}."""
    today = datetime.date.today().isoformat()
    return f"gemini_quota:{model}:key{key_index}:{today}"


# Keep backward compat — old keys without key index
def _redis_key_legacy(model: str) -> str:
    """Legacy Redis key (no key index) for backward compatibility."""
    today = datetime.date.today().isoformat()
    return f"gemini_quota:{model}:{today}"


async def _get_redis() -> aioredis.Redis:
    """Get a Redis connection."""
    return aioredis.from_url(settings.redis_url, decode_responses=True)


def _get_api_keys() -> list[str]:
    """Get list of Gemini API keys from config."""
    keys = settings.gemini_api_keys
    if not keys:
        # Fallback to single key
        single = settings.gemini_api_key.get_secret_value()
        return [single] if single else []
    return keys


async def get_usage(model: str, key_index: int = 0) -> int:
    """Get the number of requests used today for a specific model and key."""
    try:
        r = await _get_redis()
        # Try new key format first, fall back to legacy
        val = await r.get(_redis_key(model, key_index))
        if val is None and key_index == 0:
            # Check legacy key (for backward compat with existing counters)
            val = await r.get(_redis_key_legacy(model))
        await r.aclose()
        return int(val) if val else 0
    except Exception:
        return 0  # If Redis is down, assume 0 (optimistic)


async def track_usage(model: str, count: int = 1, key_index: int = 0) -> int:
    """Increment the daily usage counter for a model + key combination.

    Args:
        model: Gemini model name (e.g. "gemini-2.0-flash").
        count: Number of requests to add (default 1).
        key_index: Which API key was used (0-based index).

    Returns:
        New total usage count for today.
    """
    try:
        r = await _get_redis()
        key = _redis_key(model, key_index)
        new_count = await r.incrby(key, count)
        # Set TTL to 25 hours (ensures it expires even with timezone drift)
        await r.expire(key, 90_000)
        # Also update legacy key for backward compat with quota endpoint
        legacy = _redis_key_legacy(model)
        await r.incrby(legacy, count)
        await r.expire(legacy, 90_000)
        await r.aclose()
        return new_count
    except Exception as exc:
        logger.warning("quota_track_failed", model=model, key_index=key_index, error=str(exc))
        return -1


async def mark_key_exhausted(model: str, key_index: int = 0) -> None:
    """Mark a specific key as exhausted for a model (set counter to daily limit)."""
    try:
        r = await _get_redis()
        key = _redis_key(model, key_index)
        limit = next((t.daily_limit for t in _MODEL_TIERS if t.model == model), 9999)
        await r.setex(key, 90_000, str(limit))
        await r.aclose()
        logger.info("key_marked_exhausted", model=model, key_index=key_index, limit=limit)
    except Exception as exc:
        logger.warning("mark_exhausted_failed", model=model, key_index=key_index, error=str(exc))


async def select_model(
    task: str = "text_llm",
    required_requests: int = 1,
    require_vision: bool = False,
) -> tuple[str, str, int]:
    """Select the best available (model, API key) based on remaining daily quota.

    Strategy:
      - For each model tier (best→worst), try each API key
      - If remaining quota < required_requests, skip to next key/tier
      - If all combinations exhausted, raise QuotaExhaustedError

    Args:
        task: Description for logging (e.g. "vision_ocr", "fact_extraction").
        required_requests: Number of API calls this job will make.
        require_vision: If True, only consider models that support vision.

    Returns:
        Tuple of (model_name, api_key, key_index).

    Raises:
        QuotaExhaustedError: When no model/key has enough remaining quota.
    """
    api_keys = _get_api_keys()
    if not api_keys:
        raise QuotaExhaustedError("No Gemini API keys configured")

    tiers = _MODEL_TIERS
    if require_vision:
        tiers = [t for t in tiers if t.supports_vision]

    selection_log = []

    for tier in tiers:
        for key_idx, api_key in enumerate(api_keys):
            used = await get_usage(tier.model, key_idx)
            remaining = tier.daily_limit - used

            selection_log.append({
                "model": tier.model,
                "quality": tier.quality,
                "key_index": key_idx,
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
                    key_index=key_idx,
                    total_keys=len(api_keys),
                    required=required_requests,
                    used_today=used,
                    remaining=remaining,
                    daily_limit=tier.daily_limit,
                )
                return tier.model, api_key, key_idx

    # No model/key has enough quota
    logger.error(
        "all_models_exhausted",
        task=task,
        required=required_requests,
        total_keys=len(api_keys),
        tiers=selection_log,
    )
    raise QuotaExhaustedError(
        f"All Gemini models exhausted for today (across {len(api_keys)} API key(s)). "
        f"Need {required_requests} requests but no model/key has enough quota remaining. "
        f"Tier status: {selection_log}"
    )


async def get_all_quotas() -> list[dict]:
    """Return current quota status for all model tiers (for health/admin endpoints).

    Returns:
        List of dicts with model, daily_limit, used, remaining, quality.
    """
    api_keys = _get_api_keys()
    n_keys = len(api_keys) if api_keys else 1

    result = []
    for tier in _MODEL_TIERS:
        # Sum usage across all keys for this model
        total_used = 0
        total_limit = tier.daily_limit * n_keys
        key_details = []
        for key_idx in range(n_keys):
            used = await get_usage(tier.model, key_idx)
            total_used += used
            key_details.append({
                "key_index": key_idx,
                "used": used,
                "remaining": tier.daily_limit - used,
            })

        result.append({
            "model": tier.model,
            "quality": tier.quality,
            "daily_limit_per_key": tier.daily_limit,
            "total_daily_limit": total_limit,
            "rpm": tier.rpm,
            "used_today": total_used,
            "remaining": total_limit - total_used,
            "supports_vision": tier.supports_vision,
            "num_api_keys": n_keys,
            "keys": key_details if n_keys > 1 else None,
        })
    return result
