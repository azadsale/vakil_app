"""Unified LLM service — auto-selects best available provider.

Priority:
    1. Google Gemini 2.0 Flash (FREE, 1M tokens/min) — if GEMINI_API_KEY is set
    2. Groq llama-3.3-70b (FREE, 12K tokens/min) — fallback

Uses the new google-genai SDK (google-generativeai is deprecated).

Usage:
    from app.services.llm_service import call_llm

    response = await call_llm(
        system_prompt="You are a legal drafter...",
        user_prompt="Draft a petition for...",
        temperature=0.3,
        max_tokens=8192,
        json_mode=False,
    )
"""

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Maximum input characters sent to any LLM to avoid token limit errors.
# Gemini 2.5-flash: 1M token context — 200,000 chars is well within limits.
# Groq free tier: ~12,000 tokens/min — keep at 12,000 chars as safety fallback.
# We set different limits per provider in call_llm().
_MAX_INPUT_CHARS_GEMINI = 200_000   # Gemini handles this easily
_MAX_INPUT_CHARS_GROQ   = 12_000    # Groq free tier is strict


class LLMError(Exception):
    """Raised when all LLM providers fail."""


def _truncate(text: str, max_chars: int = _MAX_INPUT_CHARS_GROQ) -> tuple[str, bool]:
    """Truncate text to max_chars. Returns (text, was_truncated)."""
    if len(text) <= max_chars:
        return text, False
    truncated = text[:max_chars]
    # Try to break at a sentence boundary
    last_stop = max(truncated.rfind("।"), truncated.rfind("."), truncated.rfind("\n"))
    if last_stop > max_chars * 0.8:
        truncated = truncated[:last_stop + 1]
    return truncated + "\n\n[... statement truncated for processing ...]", True


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    json_mode: bool = False,
) -> str:
    """Call the best available LLM and return the response text.

    Automatically truncates oversized inputs to prevent token limit errors.
    Tries Gemini first (if key set), then Groq.

    Args:
        system_prompt: System/role instructions.
        user_prompt: The actual task content (transcript, facts, etc.)
        temperature: Creativity (0.0 = deterministic, 1.0 = creative).
        max_tokens: Max output tokens.
        json_mode: Request JSON-only output.

    Returns:
        Response text from the LLM.

    Raises:
        LLMError: If all providers fail.
    """
    gemini_key = settings.gemini_api_key.get_secret_value()
    groq_key = settings.groq_api_key.get_secret_value()

    if gemini_key:
        # Use model router to pick the best model with enough quota
        try:
            from app.services.model_router import select_model, track_usage
            selected_model = await select_model(task="text_llm", required_requests=1)
        except Exception:
            selected_model = settings.gemini_model  # fallback to config default

        # Gemini has 1M token context — only truncate at a very high limit
        user_prompt, was_truncated = _truncate(user_prompt, _MAX_INPUT_CHARS_GEMINI)
        if was_truncated:
            logger.warning("llm_input_truncated_gemini", truncated_to=_MAX_INPUT_CHARS_GEMINI)
        try:
            result = await _call_gemini(
                system_prompt, user_prompt, temperature, max_tokens, json_mode,
                model_override=selected_model,
            )
            await track_usage(selected_model, count=1)
            return result
        except Exception as exc:
            logger.warning("gemini_failed_trying_groq", error=str(exc))
            if groq_key:
                # Groq has strict limits — re-truncate for it
                user_prompt_groq, _ = _truncate(user_prompt, _MAX_INPUT_CHARS_GROQ)
                return await _call_groq(system_prompt, user_prompt_groq, temperature, max_tokens, json_mode)
            raise LLMError(f"Gemini failed and no Groq key configured: {exc}") from exc

    if groq_key:
        user_prompt, was_truncated = _truncate(user_prompt, _MAX_INPUT_CHARS_GROQ)
        if was_truncated:
            logger.warning("llm_input_truncated_groq", truncated_to=_MAX_INPUT_CHARS_GROQ)
        return await _call_groq(system_prompt, user_prompt, temperature, max_tokens, json_mode)

    raise LLMError(
        "No LLM API key configured. Set GEMINI_API_KEY (recommended, free) "
        "in your .env file. Get a key at https://aistudio.google.com"
    )


async def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    model_override: str | None = None,
) -> str:
    """Call Google Gemini via the new google-genai SDK.

    Safety filters are disabled — legal DV content (violence descriptions, incident
    details) must pass through without being blocked.

    Args:
        model_override: If set, use this model instead of the config default.
                       Typically set by the quota-aware model_router.
    """
    from google import genai  # type: ignore[import]
    from google.genai import types  # type: ignore[import]

    use_model = model_override or settings.gemini_model
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    # Disable safety filters — legal case content about violence must not be blocked
    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
        safety_settings=safety_settings,
        response_mime_type="application/json" if json_mode else "text/plain",
    )

    logger.info("llm_call_gemini", model=use_model, json_mode=json_mode)

    response = await client.aio.models.generate_content(
        model=use_model,
        contents=user_prompt,
        config=config,
    )

    # Log if response was empty (usually means safety filter blocked it)
    text = response.text or ""
    if not text.strip():
        finish_reason = None
        try:
            finish_reason = response.candidates[0].finish_reason if response.candidates else None
        except Exception:
            pass
        logger.warning(
            "gemini_empty_response",
            model=use_model,
            json_mode=json_mode,
            finish_reason=str(finish_reason),
        )

    return text


async def _call_groq(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> str:
    """Call Groq LLM API."""
    from groq import AsyncGroq  # type: ignore[import]

    client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())

    kwargs: dict = {
        "model": settings.groq_llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    logger.info("llm_call_groq", model=settings.groq_llm_model, json_mode=json_mode)

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def active_provider() -> str:
    """Return which LLM provider will be used (for health/status endpoints)."""
    if settings.gemini_api_key.get_secret_value():
        return f"gemini ({settings.gemini_model})"
    if settings.groq_api_key.get_secret_value():
        return f"groq ({settings.groq_llm_model})"
    return "none — no API key configured"
