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

# pydub is used for webm → wav conversion (browser MediaRecorder outputs webm)
# ffmpeg must be installed in the container (added to Dockerfile)
try:
    from pydub import AudioSegment
    _PYDUB_AVAILABLE = True
except ImportError:
    _PYDUB_AVAILABLE = False

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SARVAM_STT_ENDPOINT = "/speech-to-text"
SUPPORTED_MIME_TYPES = {
    "audio/webm",
    "audio/webm;codecs=opus",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/ogg;codecs=opus",
    "audio/mp4",
    "audio/m4a",
    "audio/aac",
}

# MIME types that need conversion to WAV before sending to Sarvam
# (Sarvam saaras:v3 does not accept WebM or AAC containers)
_NEEDS_CONVERSION = {
    "audio/webm",
    "audio/webm;codecs=opus",
    "audio/aac",
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


# Sarvam real-time STT limit: 30 seconds. We use 25s chunks with 1s overlap.
_SARVAM_MAX_DURATION_MS = 25_000   # 25 seconds per chunk (5s buffer)
_CHUNK_OVERLAP_MS = 500            # 500ms overlap to avoid cutting words


def _load_audio_segment(audio_bytes: bytes, mime_type: str) -> "AudioSegment":
    """Load raw audio bytes into a pydub AudioSegment normalised to 16kHz/mono/16-bit.

    Args:
        audio_bytes: Raw audio data.
        mime_type: MIME type of the source audio.

    Returns:
        Normalised AudioSegment ready for Sarvam.

    Raises:
        SarvamTranscriptionError: If pydub/ffmpeg unavailable or conversion fails.
    """
    if not _PYDUB_AVAILABLE:
        raise SarvamTranscriptionError(
            "pydub is not installed. Cannot convert audio. Install pydub."
        )
    try:
        fmt = "webm" if "webm" in mime_type else mime_type.split("/")[-1].split(";")[0]
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        # Sarvam saaras:v3 requires 16kHz / mono / 16-bit PCM
        return audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    except Exception as exc:
        logger.error("sarvam_audio_load_failed", error=str(exc))
        raise SarvamTranscriptionError(
            f"Audio load/conversion failed: {exc}. "
            "Ensure ffmpeg is installed in the container."
        ) from exc


def _export_segment_to_wav(segment: "AudioSegment") -> bytes:
    """Export an AudioSegment to WAV bytes."""
    buf = io.BytesIO()
    segment.export(buf, format="wav")
    return buf.getvalue()


async def _call_sarvam_stt(
    wav_bytes: bytes,
    filename: str,
    language_code: str,
    api_key: str,
) -> str:
    """Make a single Sarvam STT API call and return the transcript text.

    Args:
        wav_bytes: 16kHz/mono/16-bit WAV data (≤25 seconds).
        filename: Filename to send in the multipart form.
        language_code: BCP-47 language code.
        api_key: Sarvam API subscription key.

    Returns:
        Transcript string (may be empty for silent audio).

    Raises:
        SarvamTranscriptionError: On API error.
    """
    headers = {"api-subscription-key": api_key}
    files = {"file": (filename, io.BytesIO(wav_bytes), "audio/wav")}
    data = {
        "model": settings.sarvam_transcribe_model,
        "language_code": language_code,
        "with_timestamps": "false",
        "with_disfluencies": "false",
        "debug_mode": "false",
    }

    async with httpx.AsyncClient(
        base_url=settings.sarvam_api_base_url,
        timeout=60.0,
    ) as client:
        try:
            response = await client.post(
                SARVAM_STT_ENDPOINT,
                headers=headers,
                files=files,
                data=data,
            )
        except httpx.TimeoutException as exc:
            raise SarvamTranscriptionError("Sarvam API timed out for chunk.") from exc
        except httpx.RequestError as exc:
            raise SarvamTranscriptionError(
                f"Network error reaching Sarvam API: {exc}"
            ) from exc

    if response.status_code != 200:
        try:
            err_body = response.json()
            err_obj = err_body.get("error", err_body)
            err_code = err_obj.get("code", "")
            err_detail = err_obj.get("message") or err_body.get("detail") or str(err_body)
        except Exception:
            err_code = ""
            err_detail = response.text[:300]

        logger.error(
            "sarvam_chunk_api_error",
            status_code=response.status_code,
            sarvam_error_code=err_code,
            sarvam_error=err_detail,
        )
        if response.status_code == 429 or "quota" in err_detail.lower():
            raise SarvamTranscriptionError(
                "Sarvam API quota exhausted. Top up at https://console.sarvam.ai → Billing."
            )
        raise SarvamTranscriptionError(
            f"Sarvam API error {response.status_code} [{err_code}]: {err_detail}"
        )

    payload = response.json()
    return payload.get("transcript", "")


async def _transcribe_in_chunks(
    audio: "AudioSegment",
    filename_stem: str,
    language_code: str,
    api_key: str,
) -> tuple[str, int]:
    """Split audio into ≤25-second chunks and transcribe each one.

    Args:
        audio: Full normalised AudioSegment.
        filename_stem: Base filename (without extension).
        language_code: BCP-47 language code.
        api_key: Sarvam subscription key.

    Returns:
        Tuple of (full_transcript, num_chunks_processed).
    """
    duration_ms = len(audio)
    num_chunks = (duration_ms + _SARVAM_MAX_DURATION_MS - 1) // _SARVAM_MAX_DURATION_MS

    logger.info(
        "sarvam_chunked_start",
        duration_seconds=round(duration_ms / 1000, 1),
        num_chunks=num_chunks,
    )

    transcripts: list[str] = []
    start_ms = 0

    for i in range(num_chunks):
        end_ms = min(start_ms + _SARVAM_MAX_DURATION_MS, duration_ms)
        chunk = audio[start_ms:end_ms]
        wav_bytes = _export_segment_to_wav(chunk)

        chunk_filename = f"{filename_stem}_chunk{i:02d}.wav"
        chunk_duration_s = round(len(chunk) / 1000, 1)

        logger.info(
            "sarvam_chunk_sending",
            chunk=i + 1,
            total=num_chunks,
            duration_s=chunk_duration_s,
            size_kb=round(len(wav_bytes) / 1024, 1),
        )

        transcript = await _call_sarvam_stt(wav_bytes, chunk_filename, language_code, api_key)
        if transcript:
            transcripts.append(transcript.strip())

        # Next chunk starts at end of this one (no overlap to avoid repetition)
        start_ms = end_ms

    full_transcript = " ".join(transcripts)
    logger.info(
        "sarvam_chunked_complete",
        chunks_processed=num_chunks,
        total_chars=len(full_transcript),
    )
    return full_transcript, num_chunks


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    language_code: str = "mr-IN",
    mime_type: str = "audio/webm",
) -> TranscriptionResult:
    """Transcribe audio bytes using Sarvam Saaras v3.

    Automatically chunks audio longer than 25 seconds (Sarvam's 30s hard limit).

    Args:
        audio_bytes: Raw audio data.
        filename: Original filename.
        language_code: BCP-47 code. Default "mr-IN" (Marathi).
        mime_type: MIME type of the audio data.

    Returns:
        TranscriptionResult with concatenated transcript.

    Raises:
        SarvamTranscriptionError: On API error or unsupported format.
        ValueError: If audio_bytes is empty.
    """
    if not audio_bytes:
        raise ValueError("audio_bytes cannot be empty")

    api_key = settings.sarvam_api_key.get_secret_value()
    if not api_key:
        raise SarvamTranscriptionError(
            "SARVAM_API_KEY not configured. Add it to .env"
        )

    # Always load through pydub so we can:
    #  a) convert format if needed, b) check duration, c) chunk if >25s
    needs_convert = (
        mime_type.lower() in _NEEDS_CONVERSION
        or mime_type.split(";")[0].strip().lower() in _NEEDS_CONVERSION
    )

    if _PYDUB_AVAILABLE and (needs_convert or True):
        # Use pydub for ALL audio — gives us duration + normalisation
        audio = _load_audio_segment(audio_bytes, mime_type)
        duration_ms = len(audio)
        duration_s = duration_ms / 1000

        logger.info(
            "sarvam_audio_loaded",
            duration_s=round(duration_s, 1),
            source_mime=mime_type,
            source_size_kb=round(len(audio_bytes) / 1024, 1),
        )

        filename_stem = Path(filename).stem
        size_mb = len(audio_bytes) / (1024 * 1024)

        if size_mb > settings.sarvam_max_audio_size_mb:
            raise SarvamTranscriptionError(
                f"Audio file {size_mb:.1f}MB exceeds max allowed "
                f"{settings.sarvam_max_audio_size_mb}MB."
            )

        logger.info(
            "sarvam_transcribe_start",
            duration_s=round(duration_s, 1),
            language_code=language_code,
            model=settings.sarvam_transcribe_model,
            chunked=duration_ms > _SARVAM_MAX_DURATION_MS,
        )

        if duration_ms > _SARVAM_MAX_DURATION_MS:
            # Audio is longer than 25 seconds — split and transcribe in chunks
            transcript, num_chunks = await _transcribe_in_chunks(
                audio, filename_stem, language_code, api_key
            )
        else:
            # Short audio — single call
            wav_bytes = _export_segment_to_wav(audio)
            transcript = await _call_sarvam_stt(
                wav_bytes, filename_stem + ".wav", language_code, api_key
            )
            num_chunks = 1

        if not transcript:
            raise SarvamTranscriptionError(
                "Sarvam returned empty transcript. Check audio quality and language setting."
            )

        logger.info(
            "sarvam_transcribe_success",
            language_code=language_code,
            transcript_char_count=len(transcript),
            chunks=num_chunks,
        )

        return TranscriptionResult(
            transcript=transcript,
            language_code=language_code,
            duration_seconds=duration_s,
        )

    # Fallback: send raw bytes directly (pydub not available)
    size_mb = len(audio_bytes) / (1024 * 1024)
    transcript = await _call_sarvam_stt(audio_bytes, filename, language_code, api_key)
    if not transcript:
        raise SarvamTranscriptionError(
            "Sarvam returned empty transcript. Check audio quality."
        )
    return TranscriptionResult(transcript=transcript, language_code=language_code)


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
    # Normalise before checking (browser adds codec params e.g. "audio/webm;codecs=opus")
    mime_base = mime_type.split(";")[0].strip().lower()
    if mime_base not in {m.split(";")[0].strip() for m in SUPPORTED_MIME_TYPES}:
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