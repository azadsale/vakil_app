"""Party model — petitioner, respondent, or client linked to a case."""

import uuid
from datetime import datetime
from enum import Enum

from sqlmodel import Field, Relationship, SQLModel


class PartyRole(str, Enum):
    """Role of the party in the case."""

    PETITIONER = "petitioner"
    RESPONDENT = "respondent"
    CLIENT = "client"              # The lawyer's client (may overlap with petitioner/respondent)
    WITNESS = "witness"
    INTERVENER = "intervener"
    THIRD_PARTY = "third_party"


class Party(SQLModel, table=True):
    """A party (person or organization) involved in a legal case.

    Note:
        Contact information (phone/email) is PII and must not appear in logs.
        The ``contact_encrypted`` field stores AES-256 encrypted contact blob.

    Attributes:
        id: UUID primary key.
        case_id: FK → cases.id.
        name: Full legal name of the party.
        role: PartyRole enum.
        address: Registered address (used in legal notices).
        contact_encrypted: AES-256 encrypted JSON of phone/email.
        aadhar_ref: Last 4 digits only — for identity reference (NO full Aadhar stored).
        is_organization: True for companies/trusts/societies.
        created_at: Row creation timestamp.
        updated_at: Last modification timestamp.
    """

    __tablename__ = "parties"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
    )
    case_id: uuid.UUID = Field(foreign_key="cases.id", nullable=False, index=True)
    name: str = Field(max_length=500, index=True)
    role: PartyRole = Field(default=PartyRole.CLIENT)
    address: str | None = Field(default=None)
    contact_encrypted: bytes | None = Field(
        default=None,
        description="AES-256-GCM encrypted JSON blob {phone, email}",
    )
    aadhar_ref: str | None = Field(
        default=None,
        max_length=4,
        description="Last 4 digits of Aadhar only",
    )
    is_organization: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    # Relationships
    case: "Case" = Relationship(back_populates="parties")  # type: ignore[name-defined]  # noqa: F821