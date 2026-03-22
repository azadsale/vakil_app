"""Unified LLM service — auto-selects best available provider.

Priority:
    1. Google Gemini Flash (FREE, 1M tokens/min) — if GEMINI_API_KEY is set
    2. Groq llama-3.3-70b (FREE, 12K tokens/min) — fallback

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
# ~10,000 chars ≈ 2,500 tokens — leaves plenty of room for system prompt + output.
_MAX_INPUT_CHARS = 12_000


class LLMError(Exception):
    """Raised when all LLM providers fail."""


def _truncate(text: str, max_chars: int = _MAX_INPUT_CHARS) -> tuple[str, bool]:
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
        json_mode: Request JSON-only output (Groq) / JSON hint (Gemini).

    Returns:
        Response text from the LLM.

    Raises:
        LLMError: If all providers fail.
    """
    # Truncate user prompt if too large
    user_prompt, was_truncated = _truncate(user_prompt)
    if was_truncated:
        logger.warning(
            "llm_input_truncated",
            original_chars=len(user_prompt),
            truncated_to=_MAX_INPUT_CHARS,
        )

    gemini_key = settings.gemini_api_key.get_secret_value()
    groq_key = settings.groq_api_key.get_secret_value()

    if gemini_key:
        try:
            return await _call_gemini(system_prompt, user_prompt, temperature, max_tokens, json_mode)
        except Exception as exc:
            logger.warning("gemini_failed_trying_groq", error=str(exc))
            if groq_key:
                return await _call_groq(system_prompt, user_prompt, temperature, max_tokens, json_mode)
            raise LLMError(f"Gemini failed and no Groq key configured: {exc}") from exc

    if groq_key:
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
) -> str:
    """Call Google Gemini Flash API."""
    import google.generativeai as genai  # type: ignore[import]

    genai.configure(api_key=settings.gemini_api_key.get_secret_value())

    generation_config = genai.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="application/json" if json_mode else "text/plain",
    )

    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=system_prompt,
        generation_config=generation_config,
    )

    logger.info("llm_call_gemini", model=settings.gemini_model, json_mode=json_mode)

    response = await model.generate_content_async(user_prompt)
    return response.text


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
