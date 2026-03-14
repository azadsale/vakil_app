"""ClientStatement model — audio recording + transcript from Sarvam AI.

One statement per client visit. The audio is stored in MinIO (encrypted).
The transcript is stored here for RAG/fact-extraction processing.
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlmodel import Field, Relationship, SQLModel


class StatementLanguage(str, Enum):
    """Detected language of the client's statement."""

    MARATHI = "mr-IN"
    HINDI = "hi-IN"
    ENGLISH = "en-IN"
    CODE_SWITCH = "mixed"  # Common in Maharashtra — Marathi+Hindi+English blend


class StatementStatus(str, Enum):
    """Processing lifecycle of the statement."""

    AUDIO_UPLOADED = "audio_uploaded"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    FACTS_EXTRACTED = "facts_extracted"
    DRAFT_GENERATED = "draft_generated"
    FAILED = "failed"


class ClientStatement(SQLModel, table=True):
    """Audio recording and transcript of a client's statement.

    Flow:
        1. Lawyer/intern records client via browser mic or uploads audio file.
        2. Audio stored in MinIO (AES-256 encrypted).
        3. Sarvam Saaras v3 transcribes to text.
        4. Fact extraction service converts transcript → ChronologyOfEvents JSON.

    Attributes:
        id: UUID primary key.
        case_id: FK → cases.id (nullable — statement can be pre-case intake).
        user_id: FK → users.id (the lawyer who owns this).
        audio_storage_path: MinIO object key for encrypted audio file.
        audio_encryption_iv: AES-256-GCM IV for the audio blob.
        audio_duration_seconds: Duration of recording.
        audio_mime_type: MIME type (audio/webm, audio/wav, audio/mp3).
        language: Detected or declared language.
        transcript_raw: Raw Sarvam API output — verbatim (NOT logged, PII).
        transcript_clean: Post-processed, punctuated version.
        sarvam_request_id: Sarvam API response ID for debugging/billing.
        status: Processing lifecycle status.
        created_at: Row creation timestamp.
        updated_at: Last modification timestamp.
    """

    __tablename__ = "client_statements"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
    )
    case_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="cases.id",
        index=True,
        description="Nullable — can be created before case is filed",
    )
    user_id: uuid.UUID = Field(foreign_key="users.id", nullable=False, index=True)
    audio_storage_path: str | None = Field(
        default=None,
        max_length=1000,
        description="MinIO object key — encrypted audio",
    )
    audio_encryption_iv: str | None = Field(
        default=None,
        max_length=64,
        description="Hex-encoded AES-256-GCM IV",
    )
    audio_duration_seconds: float | None = Field(default=None)
    audio_mime_type: str = Field(default="audio/webm", max_length=50)
    language: StatementLanguage = Field(default=StatementLanguage.MARATHI)
    transcript_raw: str | None = Field(
        default=None,
        description="Verbatim Sarvam output — PII, never log",
    )
    transcript_clean: str | None = Field(
        default=None,
        description="Cleaned/punctuated transcript — PII, never log",
    )
    sarvam_request_id: str | None = Field(default=None, max_length=100)
    status: StatementStatus = Field(
        default=StatementStatus.AUDIO_UPLOADED, index=True
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    # Relationships
    draft_petitions: list["DraftPetition"] = Relationship(  # type: ignore[name-defined]  # noqa: F821
        back_populates="statement"
    )