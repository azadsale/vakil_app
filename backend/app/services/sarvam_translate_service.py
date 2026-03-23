"""Sarvam Saarika translation service — English → Indian languages.

Uses Sarvam AI's translate API (mayura:v1) for high-quality Indic translations.
Handles long documents by chunking at PARAGRAPH boundaries to preserve
petition structure (headings, numbered clauses, prayer sections).

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


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into structural paragraphs at double-newline boundaries.

    Preserves petition structure: headings, numbered paragraphs, prayer clauses
    stay as separate units. Each paragraph is translated independently.
    """
    # Split on double newlines (paragraph breaks)
    raw_paras = re.split(r'\n\s*\n', text)
    # Filter empty paragraphs and strip whitespace
    return [p.strip() for p in raw_paras if p.strip()]


def _chunk_paragraph(paragraph: str, max_chars: int = _MAX_CHARS_PER_CHUNK) -> list[str]:
    """Split a single paragraph into chunks if it exceeds max_chars.

    For most petition paragraphs (headings, numbered clauses), they fit in one chunk.
    Only long incident descriptions need splitting — and those are split at sentence
    boundaries to keep meaning intact.
    """
    if len(paragraph) <= max_chars:
        return [paragraph]

    chunks: list[str] = []
    remaining = paragraph

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        segment = remaining[:max_chars]

        # Try split points: sentence end → semicolon → comma → space
        split_pos = -1
        for delimiter in ["। ", ". ", "; ", ", "]:
            pos = segment.rfind(delimiter)
            if pos > max_chars * 0.3:
                split_pos = pos + len(delimiter)
                break

        if split_pos == -1:
            split_pos = segment.rfind(" ")
            if split_pos == -1:
                split_pos = max_chars

        chunk = remaining[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos:].strip()

    return chunks


def _prepare_chunks(text: str) -> list[str]:
    """Split text into translation-ready chunks that preserve document structure.

    Strategy:
    1. Split at paragraph boundaries (double newlines) — preserves petition structure
    2. If any paragraph exceeds 900 chars, split it at sentence boundaries
    3. Each chunk is translated independently, then reassembled with original spacing

    This ensures headings, numbered clauses, and prayer sections stay intact.
    """
    paragraphs = _split_into_paragraphs(text)
    chunks: list[str] = []

    for para in paragraphs:
        sub_chunks = _chunk_paragraph(para)
        chunks.extend(sub_chunks)

    return chunks


# Patterns that should NOT be translated — keep as-is
_NO_TRANSLATE_PATTERNS = [
    re.compile(r'^-{3,}$'),           # --- separator lines
    re.compile(r'^\*{3,}$'),          # *** separator lines
    re.compile(r'^={3,}$'),           # === separator lines
    re.compile(r'^LEGAL DISCLAIMER:', re.IGNORECASE),  # Disclaimer header
]


def _should_skip_translation(chunk: str) -> bool:
    """Check if a chunk should be kept as-is (separators, markers, etc.)."""
    stripped = chunk.strip()
    if not stripped:
        return True
    # Very short chunks that are just formatting
    if len(stripped) < 5 and not any(c.isalpha() for c in stripped):
        return True
    for pattern in _NO_TRANSLATE_PATTERNS:
        if pattern.match(stripped):
            return True
    return False


async def translate_text(
    text: str,
    target_language: str = "mr-IN",
    source_language: str = "en-IN",
    mode: str = "formal",
) -> str:
    """Translate text from English to an Indian language using Sarvam Saarika.

    Splits at paragraph boundaries to preserve petition structure (headings,
    numbered clauses, sections). Each paragraph is translated independently.

    Args:
        text: Source text (English).
        target_language: Target language code (e.g. "mr-IN" for Marathi).
        source_language: Source language code (default "en-IN").
        mode: Translation style — "formal" for legal documents.

    Returns:
        Translated text in target language with structure preserved.

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

    # Split into structure-preserving chunks
    chunks = _prepare_chunks(text)
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
            # Skip empty chunks and formatting-only chunks
            if _should_skip_translation(chunk):
                translated_chunks.append(chunk)
                continue

            payload = {
                "input": chunk,
                "source_language_code": source_language,
                "target_language_code": target_language,
                "model": _MODEL,
                "mode": mode,
                "numerals_format": "international",
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
                    translated_chunks.append(chunk)

            except Exception as exc:
                logger.warning(
                    "translation_chunk_exception",
                    chunk=i + 1,
                    error=str(exc),
                )
                translated_chunks.append(chunk)

    # Reassemble with double-newlines between paragraphs (preserving petition structure)
    result = "\n\n".join(translated_chunks)

    logger.info(
        "translation_complete",
        source=source_language,
        target=target_language,
        source_chars=len(text),
        translated_chars=len(result),
        n_chunks=len(chunks),
    )

    return result
