"""SQLModel ORM models — re-exported for Alembic autogenerate discovery.

Import order matters: models with FK dependencies must be imported
after the models they reference.
"""

from app.models.case import Case, CaseStatus, CaseType
from app.models.client_statement import ClientStatement, StatementLanguage, StatementStatus
from app.models.document import Document, DocumentLanguage, OCRStatus
from app.models.draft_petition import DraftPetition, DraftStatus
from app.models.embedding import DocumentChunk
from app.models.hearing import Hearing
from app.models.lawyer_template import LawyerTemplate, TemplateType
from app.models.legal_document import IndexingStatus, LegalDocument, LegalDocumentType
from app.models.party import Party, PartyRole
from app.models.user import User

__all__ = [
    # Core
    "User",
    "Case",
    "CaseStatus",
    "CaseType",
    "Party",
    "PartyRole",
    "Hearing",
    "Document",
    "DocumentLanguage",
    "OCRStatus",
    "DocumentChunk",
    # Drafting V1
    "ClientStatement",
    "StatementLanguage",
    "StatementStatus",
    "DraftPetition",
    "DraftStatus",
    "LawyerTemplate",
    "TemplateType",
    "LegalDocument",
    "LegalDocumentType",
    "IndexingStatus",
]