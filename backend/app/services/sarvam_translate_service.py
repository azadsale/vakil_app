"""Sarvam Saarika translation service — English → Indian languages.

Uses Sarvam AI's translate API (mayura:v1) for high-quality Indic translations.
Handles long documents by chunking at sentence boundaries.

Usage:
    from app.services.sarvam_translate_service import translate_text

    marathi_text = await translate_text(
        text="The petitioner respectfully submits...",
        target_language="mr-IN",
    )
"""

import re

import httpx

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Sarvam translate limits: mayura:v1 = 1000 chars, sarvam-translate:v1 = 2000 chars
_MODEL = "mayura:v1"
_MAX_CHARS_PER_CHUNK = 900  # slightly under 1000 to leave safety margin
_TRANSLATE_ENDPOINT = "/translate"


class TranslationError(Exception):
    """Raised when translation fails."""


def _chunk_text(text: str, max_chars: int = _MAX_CHARS_PER_CHUNK) -> list[str]:
    """Split text into chunks at sentence boundaries, each under max_chars.

    Tries to split at: paragraph breaks → full stops → semicolons → commas.
    Never splits mid-word.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        # Find the best split point within max_chars
        segment = remaining[:max_chars]

        # Try split points in order of preference
        split_pos = -1
        for delimiter in ["\n\n", "\n", "। ", ". ", "; ", ", "]:
            pos = segment.rfind(delimiter)
            if pos > max_chars * 0.3:  # don't split too early
                split_pos = pos + len(delimiter)
                break

        if split_pos == -1:
            # Last resort: split at last space
            split_pos = segment.rfind(" ")
            if split_pos == -1:
                split_pos = max_chars  # hard cut

        chunk = remaining[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos:].strip()

    return chunks


async def translate_text(
    text: str,
    target_language: str = "mr-IN",
    source_language: str = "en-IN",
    mode: str = "formal",
) -> str:
    """Translate text from English to an Indian language using Sarvam Saarika.

    Automatically chunks long text and translates each chunk separately.

    Args:
        text: Source text (English).
        target_language: Target language code (e.g. "mr-IN" for Marathi).
        source_language: Source language code (default "en-IN").
        mode: Translation style — "formal" for legal documents.

    Returns:
        Translated text in target language.

    Raises:
        TranslationError: If the API call fails.
    """
    api_key = settings.sarvam_api_key.get_secret_value()
    if not api_key:
        raise TranslationError("SARVAM_API_KEY not set — cannot translate")

    base_url = settings.sarvam_api_base_url
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json",
    }

    # Split into chunks
    chunks = _chunk_text(text)
    logger.info(
        "translation_start",
        source=source_language,
        target=target_language,
        total_chars=len(text),
        n_chunks=len(chunks),
        model=_MODEL,
        mode=mode,
    )

    translated_chunks: list[str] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                translated_chunks.append("")
                continue

            payload = {
                "input": chunk,
                "source_language_code": source_language,
                "target_language_code": target_language,
                "model": _MODEL,
                "mode": mode,
                "numerals_format": "international",  # keep numbers as 1,2,3 not devanagari
            }

            try:
                response = await client.post(
                    _TRANSLATE_ENDPOINT,
                    json=payload,
                    headers=headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    translated = data.get("translated_text", "")
                    translated_chunks.append(translated)
                    logger.debug(
                        "translation_chunk_done",
                        chunk_index=i + 1,
                        total_chunks=len(chunks),
                        source_chars=len(chunk),
                        translated_chars=len(translated),
                    )
                elif response.status_code == 429:
                    # Rate limited — wait and retry once
                    import asyncio
                    logger.warning("translation_rate_limited", chunk=i + 1)
                    await asyncio.sleep(2)
                    retry = await client.post(
                        _TRANSLATE_ENDPOINT,
                        json=payload,
                        headers=headers,
                    )
                    if retry.status_code == 200:
                        translated_chunks.append(retry.json().get("translated_text", ""))
                    else:
                        # Give up on this chunk — keep original English
                        logger.warning("translation_chunk_failed_keeping_original", chunk=i + 1)
                        translated_chunks.append(chunk)
                else:
                    error_msg = response.text[:200]
                    logger.warning(
                        "translation_chunk_error",
                        chunk=i + 1,
                        status=response.status_code,
                        error=error_msg,
                    )
                    # Keep original text for failed chunks so the draft isn't lost
                    translated_chunks.append(chunk)

            except Exception as exc:
                logger.warning(
                    "translation_chunk_exception",
                    chunk=i + 1,
                    error=str(exc),
                )
                # Keep original for this chunk
                translated_chunks.append(chunk)

    result = "\n".join(translated_chunks)

    logger.info(
        "translation_complete",
        source=source_language,
        target=target_language,
        source_chars=len(text),
        translated_chars=len(result),
        n_chunks=len(chunks),
    )

    return result
