"""DraftPetition model — AI-generated DV petition with version tracking.

Stores the generated petition text, the facts JSON used, and lawyer feedback
for continuous improvement of the drafting pipeline.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel


class DraftStatus(str, Enum):
    """Lifecycle status of a petition draft."""

    GENERATING = "generating"
    DRAFT = "draft"              # AI-generated, not yet reviewed
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"        # Lawyer approved — eligible as template
    REJECTED = "rejected"
    ARCHIVED = "archived"


class DraftPetition(SQLModel, table=True):
    """An AI-generated DV petition draft.

    Version tracking: multiple drafts can exist per statement.
    When a draft is approved, it can be added to LawyerTemplate
    pool for future style matching.

    Attributes:
        id: UUID primary key.
        case_id: FK → cases.id.
        statement_id: FK → client_statements.id.
        user_id: FK → users.id (owning lawyer).
        version: Draft version number (starts at 1, increments on regeneration).
        facts_json: ChronologyOfEvents JSON used to generate this draft.
        legal_sections_used: List of DV Act section references injected (anti-hallucination).
        template_ids_used: List of LawyerTemplate IDs used as few-shot examples.
        draft_text: The full generated petition text (legally sensitive — not logged).
        disclaimer: Mandatory legal disclaimer appended to draft.
        status: DraftStatus lifecycle.
        lawyer_feedback: Free-text feedback from lawyer for model improvement.
        generation_model: LLM model used (e.g. gpt-4o) for audit trail.
        generation_prompt_hash: SHA-256 of the prompt (for dedup/caching).
        created_at: Row creation timestamp.
        updated_at: Last modification timestamp.
    """

    __tablename__ = "draft_petitions"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
    )
    case_id: uuid.UUID = Field(foreign_key="cases.id", nullable=False, index=True)
    statement_id: uuid.UUID = Field(
        foreign_key="client_statements.id", nullable=False, index=True
    )
    user_id: uuid.UUID = Field(foreign_key="users.id", nullable=False, index=True)
    version: int = Field(default=1, nullable=False)

    # JSONB for efficient querying of facts (e.g., filter by incident type)
    facts_json: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default="{}"),
        description="ChronologyOfEvents JSON — the structured facts used",
    )
    legal_sections_used: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, server_default="[]"),
        description="DV Act sections injected e.g. ['Section 12', 'Section 18']",
    )
    template_ids_used: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False, server_default="[]"),
        description="LawyerTemplate UUIDs used as few-shot examples",
    )
    draft_text: str = Field(
        description="Full petition text — legally sensitive, never log"
    )
    disclaimer: str = Field(
        default=(
            "DISCLAIMER: This document is AI-assisted and has been generated for "
            "informational purposes only. It must be reviewed, verified, and approved "
            "by a qualified advocate before filing. This does not constitute legal advice."
        )
    )
    status: DraftStatus = Field(default=DraftStatus.DRAFT, index=True)
    lawyer_feedback: str | None = Field(
        default=None,
        description="Lawyer's review notes — used for pipeline improvement",
    )
    generation_model: str = Field(default="gpt-4o", max_length=100)
    generation_prompt_hash: str | None = Field(default=None, max_length=64)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    # Relationships
    statement: "ClientStatement" = Relationship(  # type: ignore[name-defined]  # noqa: F821
        back_populates="draft_petitions"
    )