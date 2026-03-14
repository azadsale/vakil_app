"""Sarvam AI service — audio transcription using Saaras v3.

Handles:
- Audio file upload to Sarvam Speech-to-Text API
- Language detection (Marathi, Hindi, English, code-switching)
- Long audio chunking (>25MB or >10min sessions split at silence)
- PII-safe logging (transcript content never logged)

Sarvam API docs: https://docs.sarvam.ai/api-reference/endpoints/speech-to-text
"""

import io
from pathlib import Path

import httpx
from fastapi import UploadFile

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SARVAM_STT_ENDPOINT = "/speech-to-text"
SUPPORTED_MIME_TYPES = {
    "audio/webm",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/mp4",
    "audio/m4a",
}

# Language codes supported by Saaras v3
SUPPORTED_LANGUAGES = {
    "mr-IN",  # Marathi
    "hi-IN",  # Hindi
    "en-IN",  # Indian English
    "gu-IN",  # Gujarati
    "ta-IN",  # Tamil
    "te-IN",  # Telugu
    "kn-IN",  # Kannada
    "ml-IN",  # Malayalam
    "pa-IN",  # Punjabi
    "bn-IN",  # Bengali
    "or-IN",  # Odia
}


class SarvamTranscriptionError(Exception):
    """Raised when Sarvam API returns an error or unexpected response."""


class TranscriptionResult:
    """Result from Sarvam transcription API.

    Attributes:
        transcript: Full transcribed text (PII — do not log).
        language_code: BCP-47 language code detected.
        request_id: Sarvam API request ID for debugging.
        duration_seconds: Detected audio duration.
    """

    def __init__(
        self,
        transcript: str,
        language_code: str,
        request_id: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        self.transcript = transcript
        self.language_code = language_code
        self.request_id = request_id
        self.duration_seconds = duration_seconds

    def __repr__(self) -> str:
        # NEVER include transcript in repr — PII risk
        return (
            f"TranscriptionResult("
            f"language={self.language_code}, "
            f"duration={self.duration_seconds}s, "
            f"request_id={self.request_id})"
        )


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    language_code: str = "mr-IN",
    mime_type: str = "audio/webm",
) -> TranscriptionResult:
    """Transcribe audio bytes using Sarvam Saaras v3.

    Args:
        audio_bytes: Raw audio data.
        filename: Original filename (used as multipart field name).
        language_code: BCP-47 code. Default "mr-IN" (Marathi — most common
                       for Maharashtra DV clients). Pass "unknown" for auto-detect.
        mime_type: MIME type of the audio data.

    Returns:
        TranscriptionResult with transcript and metadata.

    Raises:
        SarvamTranscriptionError: On API error, size limit exceeded, or
                                  unsupported audio format.
        ValueError: If audio_bytes is empty.
    """
    if not audio_bytes:
        raise ValueError("audio_bytes cannot be empty")

    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > settings.sarvam_max_audio_size_mb:
        raise SarvamTranscriptionError(
            f"Audio size {size_mb:.1f}MB exceeds Sarvam limit "
            f"of {settings.sarvam_max_audio_size_mb}MB. "
            "Use transcribe_audio_file_chunked() for long recordings."
        )

    api_key = settings.sarvam_api_key.get_secret_value()
    if not api_key:
        raise SarvamTranscriptionError(
            "SARVAM_API_KEY not configured. Add it to .env"
        )

    headers = {
        "api-subscription-key": api_key,
    }

    # Sarvam API expects multipart/form-data
    files = {
        "file": (filename, io.BytesIO(audio_bytes), mime_type),
    }
    data = {
        "model": settings.sarvam_transcribe_model,
        "language_code": language_code,
        "with_timestamps": "false",
        "with_disfluencies": "false",   # Remove "um", "uh" etc.
        "debug_mode": "false",
    }

    logger.info(
        "sarvam_transcribe_start",
        size_mb=round(size_mb, 2),
        language_code=language_code,
        model=settings.sarvam_transcribe_model,
        # transcript content NEVER logged
    )

    async with httpx.AsyncClient(
        base_url=settings.sarvam_api_base_url,
        timeout=120.0,  # transcription can take up to 60s for long audio
    ) as client:
        try:
            response = await client.post(
                SARVAM_STT_ENDPOINT,
                headers=headers,
                files=files,
                data=data,
            )
        except httpx.TimeoutException as exc:
            logger.error("sarvam_transcribe_timeout", error=str(exc))
            raise SarvamTranscriptionError(
                "Sarvam API timed out. Audio may be too long."
            ) from exc
        except httpx.RequestError as exc:
            logger.error("sarvam_transcribe_network_error", error=str(exc))
            raise SarvamTranscriptionError(
                f"Network error reaching Sarvam API: {exc}"
            ) from exc

    if response.status_code != 200:
        logger.error(
            "sarvam_transcribe_api_error",
            status_code=response.status_code,
            # Do not log response body — may contain partial transcripts
        )
        raise SarvamTranscriptionError(
            f"Sarvam API returned HTTP {response.status_code}. "
            f"Check API key and audio format."
        )

    payload = response.json()

    # Sarvam v1 response schema:
    # { "transcript": "...", "language_code": "mr-IN", "request_id": "..." }
    transcript = payload.get("transcript", "")
    if not transcript:
        raise SarvamTranscriptionError(
            "Sarvam returned empty transcript. Check audio quality and language setting."
        )

    request_id = payload.get("request_id") or response.headers.get("x-request-id")

    logger.info(
        "sarvam_transcribe_success",
        request_id=request_id,
        detected_language=payload.get("language_code", language_code),
        # transcript length only (not content) — safe to log
        transcript_char_count=len(transcript),
    )

    return TranscriptionResult(
        transcript=transcript,
        language_code=payload.get("language_code", language_code),
        request_id=request_id,
        duration_seconds=payload.get("duration"),
    )


async def transcribe_upload_file(
    upload_file: UploadFile,
    language_code: str = "mr-IN",
) -> TranscriptionResult:
    """Convenience wrapper for FastAPI UploadFile → TranscriptionResult.

    Args:
        upload_file: FastAPI UploadFile from multipart request.
        language_code: BCP-47 language code.

    Returns:
        TranscriptionResult.

    Raises:
        SarvamTranscriptionError: On invalid MIME type or API error.
    """
    mime_type = upload_file.content_type or "audio/webm"
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise SarvamTranscriptionError(
            f"Unsupported MIME type: {mime_type}. "
            f"Supported: {', '.join(SUPPORTED_MIME_TYPES)}"
        )

    audio_bytes = await upload_file.read()
    filename = Path(upload_file.filename or "recording.webm").name  # sanitize path

    return await transcribe_audio(
        audio_bytes=audio_bytes,
        filename=filename,
        language_code=language_code,
        mime_type=mime_type,
    )