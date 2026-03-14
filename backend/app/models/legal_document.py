"""LegalDocument model — uploaded statutes (DV Act PDF, etc.) for RAG grounding.

Distinct from the case Document model — these are reference law documents,
not client-uploaded evidence. Indexed once into pgvector for all cases.
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlmodel import Field, Relationship, SQLModel


class LegalDocumentType(str, Enum):
    """Category of legal statute or reference document."""

    DV_ACT = "dv_act_2005"                           # Protection of Women from DV Act 2005
    IPC = "ipc"                                       # Indian Penal Code (legacy)
    BNS = "bns"                                       # Bharatiya Nyaya Sanhita 2023
    MLRC = "mlrc"                                     # Maharashtra Land Revenue Code
    REGISTRATION_ACT = "registration_act"
    TRANSFER_OF_PROPERTY = "transfer_of_property_act"
    CPC = "cpc"                                       # Code of Civil Procedure
    OTHER = "other"


class IndexingStatus(str, Enum):
    """RAG indexing lifecycle status."""

    UPLOADED = "uploaded"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
    STALE = "stale"   # Re-index needed (embedding model changed)


class LegalDocument(SQLModel, table=True):
    """A legal statute or reference document indexed for RAG.

    Attributes:
        id: UUID primary key.
        title: Full title of the statute (e.g. "Protection of Women from Domestic Violence Act, 2005").
        short_name: Short reference (e.g. "DV Act 2005").
        doc_type: LegalDocumentType enum.
        storage_path: MinIO object key for the PDF.
        encryption_iv: AES-256-GCM IV.
        total_pages: Total pages in the document.
        total_chunks: Number of chunks indexed into pgvector.
        indexing_status: Current indexing lifecycle status.
        embed_model: Embedding model used for chunks (detect stale index on upgrade).
        language: Primary language of the document.
        jurisdiction: Applicable jurisdiction (India / Maharashtra).
        effective_date: Date the statute came into force.
        uploaded_by: FK → users.id.
        created_at: Row creation timestamp.
        updated_at: Last modification timestamp.
    """

    __tablename__ = "legal_documents"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
    )
    title: str = Field(max_length=500, index=True)
    short_name: str = Field(max_length=100, index=True)
    doc_type: LegalDocumentType = Field(default=LegalDocumentType.DV_ACT, index=True)
    storage_path: str = Field(max_length=1000)
    encryption_iv: str = Field(max_length=64)
    total_pages: int | None = Field(default=None)
    total_chunks: int | None = Field(default=None)
    indexing_status: IndexingStatus = Field(
        default=IndexingStatus.UPLOADED, index=True
    )
    embed_model: str = Field(default="text-embedding-ada-002", max_length=100)
    language: str = Field(default="en", max_length=10)
    jurisdiction: str = Field(default="India", max_length=100)
    effective_date: str | None = Field(
        default=None,
        max_length=20,
        description="ISO date string e.g. 2006-10-26",
    )
    uploaded_by: uuid.UUID = Field(foreign_key="users.id", nullable=False)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)