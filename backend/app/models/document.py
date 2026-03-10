"""Document model — uploaded legal documents with OCR and encryption metadata."""

import uuid
from datetime import datetime
from enum import Enum

from sqlmodel import Field, Relationship, SQLModel


class DocumentLanguage(str, Enum):
    """Primary language of the document."""

    ENGLISH = "en"
    MARATHI = "mr"
    HINDI = "hi"
    BILINGUAL_EN_MR = "en_mr"   # Common for Maharashtra govt documents


class OCRStatus(str, Enum):
    """OCR processing lifecycle status."""

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"          # e.g., document is already a text PDF


class DocumentType(str, Enum):
    """Legal document category — important for OCR model selection."""

    SEVEN_TWELVE_EXTRACT = "7_12_extract"     # 7/12 utara — land record
    FERFAR = "ferfar"                          # Mutation entry
    INDEX_TWO = "index_2"                      # Index II — property transaction history
    SALE_DEED = "sale_deed"
    AGREEMENT_FOR_SALE = "agreement_for_sale"
    COURT_ORDER = "court_order"
    VAKALATNAMA = "vakalatnama"
    LEGAL_NOTICE = "legal_notice"
    AFFIDAVIT = "affidavit"
    SURVEY_PLAN = "survey_plan"
    NA_ORDER = "na_order"                      # Non-agricultural land use order
    OTHER = "other"


class Document(SQLModel, table=True):
    """An uploaded legal document associated with a case.

    The actual file is stored in MinIO/S3 (AES-256-GCM encrypted).
    Only the storage path and encryption IV are stored in the DB.

    Attributes:
        id: UUID primary key.
        case_id: FK → cases.id.
        original_filename: Original upload filename (sanitized, not logged).
        document_type: Legal document category.
        language: Primary language of document.
        storage_path: Encrypted path in MinIO bucket.
        encryption_iv: AES-256-GCM initialization vector (hex-encoded).
        mime_type: MIME type of the uploaded file.
        file_size_bytes: File size for storage tracking.
        ocr_status: Current OCR processing status.
        ocr_provider: Which OCR provider was used.
        page_count: Number of pages in the document.
        uploaded_by: FK → users.id (uploader, may differ from case owner).
        created_at: Row creation timestamp.
        updated_at: Last modification timestamp.
    """

    __tablename__ = "documents"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
    )
    case_id: uuid.UUID = Field(foreign_key="cases.id", nullable=False, index=True)
    original_filename: str = Field(
        max_length=500,
        description="Sanitized filename — never log raw user-supplied filename",
    )
    document_type: DocumentType = Field(default=DocumentType.OTHER)
    language: DocumentLanguage = Field(default=DocumentLanguage.ENGLISH)
    storage_path: str = Field(
        max_length=1000,
        description="MinIO object key — combine with bucket from config",
    )
    encryption_iv: str = Field(
        max_length=64,
        description="Hex-encoded AES-256-GCM IV for this document",
    )
    mime_type: str = Field(max_length=100)
    file_size_bytes: int | None = Field(default=None)
    ocr_status: OCRStatus = Field(default=OCRStatus.PENDING, index=True)
    ocr_provider: str | None = Field(
        default=None,
        max_length=50,
        description="azure | google | none",
    )
    page_count: int | None = Field(default=None)
    uploaded_by: uuid.UUID = Field(foreign_key="users.id", nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    # Relationships
    case: "Case" = Relationship(back_populates="documents")  # type: ignore[name-defined]  # noqa: F821
    chunks: list["DocumentChunk"] = Relationship(back_populates="document")  # type: ignore[name-defined]  # noqa: F821