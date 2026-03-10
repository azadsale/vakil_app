"""Case model — core entity representing a legal case.

CNR (Case Number Record) is the unique identifier from the e-Courts system.
"""

import uuid
from datetime import date, datetime
from enum import Enum

from sqlmodel import Field, Relationship, SQLModel


class CaseType(str, Enum):
    """Type of legal case handled by the lawyer."""

    PROPERTY = "property"
    CIVIL = "civil"
    CRIMINAL = "criminal"
    FAMILY = "family"
    REVENUE = "revenue"          # Maharashtra Land Revenue Code cases
    MUTATION = "mutation"        # Ferfar / property mutation cases
    OTHER = "other"


class CaseStatus(str, Enum):
    """Current lifecycle status of a case."""

    ACTIVE = "active"
    PENDING = "pending"
    HEARING_SCHEDULED = "hearing_scheduled"
    JUDGMENT_RESERVED = "judgment_reserved"
    DISPOSED = "disposed"
    APPEALED = "appealed"
    CLOSED = "closed"


class Case(SQLModel, table=True):
    """Legal case record.

    Attributes:
        id: UUID primary key.
        cnr_number: e-Courts Case Number Record (e.g. MHPC010012342024).
        title: Short descriptive title of the case.
        case_type: Category from CaseType enum.
        court_name: Full name of the court (e.g. "Civil Court, Panvel").
        court_district: District (default: Raigad for Panvel jurisdiction).
        status: Current lifecycle status.
        filed_date: Date case was filed in court.
        user_id: FK → users.id (RLS owner).
        description: Internal notes (not shown to opposing party).
        created_at: Row creation timestamp.
        updated_at: Last modification timestamp.
    """

    __tablename__ = "cases"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
    )
    cnr_number: str | None = Field(
        default=None,
        max_length=25,
        index=True,
        unique=True,
        description="e-Courts CNR identifier",
    )
    title: str = Field(max_length=500, index=True)
    case_type: CaseType = Field(default=CaseType.PROPERTY)
    court_name: str = Field(max_length=255)
    court_district: str = Field(default="Raigad", max_length=100)
    status: CaseStatus = Field(default=CaseStatus.ACTIVE, index=True)
    filed_date: date | None = Field(default=None)
    description: str | None = Field(default=None)

    # Foreign key — RLS owner
    user_id: uuid.UUID = Field(foreign_key="users.id", nullable=False, index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    # Relationships
    owner: "User" = Relationship(back_populates="cases")  # type: ignore[name-defined]  # noqa: F821
    parties: list["Party"] = Relationship(back_populates="case")  # type: ignore[name-defined]  # noqa: F821
    hearings: list["Hearing"] = Relationship(back_populates="case")  # type: ignore[name-defined]  # noqa: F821
    documents: list["Document"] = Relationship(back_populates="case")  # type: ignore[name-defined]  # noqa: F821