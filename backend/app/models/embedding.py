"""DocumentChunk model — text chunks with pgvector embeddings for RAG.

This table is the backbone of the RAG pipeline:
  Document → OCR → chunk text → embed → store here → LlamaIndex queries this.

The ``source_citation`` field is critical for anti-hallucination:
  Every chunk must carry a reference to its legal source.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, Relationship, SQLModel

# Embedding dimension — must match the model used:
# OpenAI text-embedding-ada-002 → 1536
# OpenAI text-embedding-3-small → 1536
# OpenAI text-embedding-3-large → 3072
EMBEDDING_DIM = 1536


class DocumentChunk(SQLModel, table=True):
    """A text chunk extracted from a document, with its vector embedding.

    Each chunk is a semantically coherent passage (typically 256-512 tokens)
    derived from OCR output or native PDF text extraction.

    Attributes:
        id: UUID primary key.
        document_id: FK → documents.id.
        chunk_index: Sequential order of this chunk within the document.
        content: Raw text of the chunk (legally sensitive — not logged).
        embedding: pgvector float array of dimension EMBEDDING_DIM.
        source_citation: Human-readable legal citation for anti-hallucination.
            Format: "Document: <filename>, Page <n>, Section <x>"
            or for statutes: "Section 44, Maharashtra Land Revenue Code, 1966"
        token_count: Approximate token count of this chunk.
        language: Language of this chunk (en/mr/hi).
        embed_model: Name of embedding model used (for re-indexing detection).
        created_at: Row creation timestamp.
    """

    __tablename__ = "document_chunks"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
    )
    document_id: uuid.UUID = Field(
        foreign_key="documents.id", nullable=False, index=True
    )
    chunk_index: int = Field(nullable=False)
    content: str = Field(
        description="Chunk text — legally sensitive, never log",
    )
    # pgvector column — SQLAlchemy Column wrapping needed for custom types
    embedding: list[float] = Field(
        default=None,
        sa_column=Column(Vector(EMBEDDING_DIM), nullable=True),
    )
    source_citation: str = Field(
        max_length=1000,
        description="Mandatory citation for RAG anti-hallucination grounding",
    )
    token_count: int | None = Field(default=None)
    language: str = Field(default="en", max_length=10)
    embed_model: str = Field(
        default="text-embedding-ada-002",
        max_length=100,
        description="Embedding model used — detect stale embeddings on model upgrade",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    # Relationships
    document: "Document" = Relationship(back_populates="chunks")  # type: ignore[name-defined]  # noqa: F821