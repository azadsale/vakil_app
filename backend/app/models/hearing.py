"""Hearing model — court date and outcome tracking for a case."""

import uuid
from datetime import date, datetime, time
from enum import Enum

from sqlmodel import Field, Relationship, SQLModel


class HearingPurpose(str, Enum):
    """Purpose or stage of the hearing."""

    ADMISSION = "admission"
    ARGUMENTS = "arguments"
    EVIDENCE = "evidence"
    CROSS_EXAMINATION = "cross_examination"
    JUDGMENT = "judgment"
    INTERIM_ORDER = "interim_order"
    MENTION = "mention"
    OTHER = "other"


class Hearing(SQLModel, table=True):
    """A scheduled or completed court hearing.

    Attributes:
        id: UUID primary key.
        case_id: FK → cases.id.
        hearing_date: Scheduled date of the hearing.
        hearing_time: Scheduled time (if known).
        judge_name: Presiding judge's name.
        court_room: Court room/hall number.
        purpose: Stage/purpose of the hearing.
        outcome_notes: Internal notes on what happened (NOT logged).
        next_date: Next hearing date set by the court.
        is_completed: Whether the hearing has occurred.
        created_at: Row creation timestamp.
        updated_at: Last modification timestamp.
    """

    __tablename__ = "hearings"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
    )
    case_id: uuid.UUID = Field(foreign_key="cases.id", nullable=False, index=True)
    hearing_date: date = Field(index=True)
    hearing_time: time | None = Field(default=None)
    judge_name: str | None = Field(default=None, max_length=255)
    court_room: str | None = Field(default=None, max_length=50)
    purpose: HearingPurpose = Field(default=HearingPurpose.OTHER)
    outcome_notes: str | None = Field(
        default=None,
        description="Internal notes — not PII but legally sensitive",
    )
    next_date: date | None = Field(default=None, index=True)
    is_completed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    # Relationships
    case: "Case" = Relationship(back_populates="hearings")  # type: ignore[name-defined]  # noqa: F821