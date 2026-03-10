"""User model — represents a lawyer account.

RLS owner: all other tables reference user_id → users.id.
"""

import uuid
from datetime import datetime

from sqlmodel import Field, Relationship, SQLModel


class User(SQLModel, table=True):
    """Lawyer user account.

    Attributes:
        id: UUID primary key.
        email: Unique login email (NOT logged — PII).
        hashed_password: bcrypt hash.
        full_name: Display name.
        bar_registration_number: Maharashtra Bar Council number.
        is_active: Soft-disable without deleting data.
        is_superuser: Admin flag.
        created_at: Row creation timestamp.
        updated_at: Last modification timestamp.
    """

    __tablename__ = "users"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
    )
    email: str = Field(unique=True, index=True, max_length=255)
    hashed_password: str = Field(max_length=255)
    full_name: str = Field(max_length=255)
    bar_registration_number: str | None = Field(default=None, max_length=50)
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    # Relationships
    cases: list["Case"] = Relationship(back_populates="owner")  # type: ignore[name-defined]  # noqa: F821