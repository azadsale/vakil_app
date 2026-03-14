"""LawyerTemplate model — past DV petition drafts for few-shot style matching.

These are the "shots" in the few-shot prompting strategy.
The lawyer's approved drafts are embedded and stored here.
pgvector similarity search retrieves the top-K most relevant templates
for each new case based on the ChronologyOfEvents facts.
"""

import uuid
from datetime import datetime
from enum import Enum

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, SQLModel

TEMPLATE_EMBEDDING_DIM = 384   # sentence-transformers/all-MiniLM-L6-v2 (local, free)


class TemplateType(str, Enum):
    """Type of legal petition template.

    Values MUST match the PostgreSQL templatetype enum exactly (case-sensitive).
    """

    DV_PETITION = "DV_PETITION"                   # Sec 12 DV Act — main application
    DV_PROTECTION_ORDER = "DV_PROTECTION_ORDER"   # Sec 18 relief
    DV_RESIDENCE_ORDER = "DV_RESIDENCE_ORDER"     # Sec 19 relief
    DV_MONETARY_RELIEF = "DV_MONETARY_RELIEF"     # Sec 20 relief
    DV_CUSTODY_ORDER = "DV_CUSTODY_ORDER"         # Sec 21 relief
    DV_COMPENSATION = "DV_COMPENSATION"           # Sec 22 relief
    GENERAL = "GENERAL"


class LawyerTemplate(SQLModel, table=True):
    """A past petition draft used as few-shot style reference.

    Workflow:
        1. Lawyer uploads past petition (PDF or text).
        2. Text is extracted and cleaned.
        3. Embedding generated via text-embedding-ada-002.
        4. Stored here with metadata.
        5. At draft-time: top-3 similar templates retrieved via cosine similarity
           and injected as "shots" into the generation prompt.

    Attributes:
        id: UUID primary key.
        user_id: FK → users.id (each lawyer has their own template pool).
        title: Descriptive title for admin reference.
        template_type: Category of petition.
        content: Full text of the past petition (legally sensitive — not logged).
        embedding: pgvector embedding of content for similarity search.
        embed_model: Embedding model used.
        usage_count: Times this template has been used (for relevance scoring).
        is_active: Soft-disable without deleting.
        source_draft_id: FK → draft_petitions.id if promoted from an approved draft.
        created_at: Row creation timestamp.
        updated_at: Last modification timestamp.
    """

    __tablename__ = "lawyer_templates"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
    )
    user_id: uuid.UUID = Field(foreign_key="users.id", nullable=False, index=True)
    title: str = Field(max_length=500)
    template_type: TemplateType = Field(default=TemplateType.DV_PETITION, index=True)
    content: str = Field(
        description="Full petition text — PII present, never log"
    )
    embedding: list[float] = Field(
        default=None,
        sa_column=Column(Vector(TEMPLATE_EMBEDDING_DIM), nullable=True),
        description="Semantic embedding for similarity search",
    )
    embed_model: str = Field(default="text-embedding-ada-002", max_length=100)
    usage_count: int = Field(default=0)
    is_active: bool = Field(default=True)
    source_draft_id: uuid.UUID | None = Field(
        default=None,
        foreign_key="draft_petitions.id",
        description="Set when promoted from an approved DraftPetition",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)