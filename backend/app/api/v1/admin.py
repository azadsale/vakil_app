"""Admin API endpoints — one-time setup operations.

Endpoints:
    POST /admin/upload-statute    — Upload + index legal PDF (DV Act, etc.)
    POST /admin/upload-template   — Upload lawyer's past petition as style template
    GET  /admin/statutes          — List indexed statutes
    GET  /admin/templates         — List lawyer templates
    DELETE /admin/templates/{id}  — Soft-delete a template
"""

import io
import os
import tempfile
import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.database import get_db
from app.models.lawyer_template import LawyerTemplate, TemplateType
from app.models.legal_document import IndexingStatus, LegalDocument, LegalDocumentType
from app.services.rag_service import ingest_legal_pdf
from app.services.template_service import TemplateServiceError, add_template
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


async def get_current_user_id() -> uuid.UUID:
    """Placeholder auth — replace with JWT in V2."""
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


CurrentUser = Annotated[uuid.UUID, Depends(get_current_user_id)]
DB = Annotated[AsyncSession, Depends(get_db)]


# ---------------------------------------------------------------------------
# POST /admin/upload-statute
# ---------------------------------------------------------------------------
@router.post("/upload-statute", status_code=status.HTTP_201_CREATED)
async def upload_statute(
    db: DB,
    current_user: CurrentUser,
    pdf_file: UploadFile = File(..., description="Legal statute PDF"),
    title: str = Form(..., description="Full title e.g. 'Protection of Women from DV Act 2005'"),
    short_name: str = Form(..., description="Short name e.g. 'DV Act 2005'"),
    doc_type: LegalDocumentType = Form(default=LegalDocumentType.DV_ACT),
    jurisdiction: str = Form(default="India"),
    effective_date: str | None = Form(default=None),
) -> dict[str, Any]:
    """Upload and index a legal statute PDF for RAG grounding.

    This is a one-time setup operation per statute.
    The PDF is:
        1. Saved to a temp file
        2. Indexed into pgvector via LlamaIndex
        3. Metadata stored in legal_documents table

    Args:
        db: Database session.
        current_user: Authenticated lawyer UUID.
        pdf_file: The statute PDF file.
        title: Full title of the statute.
        short_name: Short reference name.
        doc_type: Category from LegalDocumentType.
        jurisdiction: Applicable jurisdiction.
        effective_date: ISO date when statute came into force.

    Returns:
        Legal document metadata including chunk count.
    """
    if pdf_file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=422,
            detail="Only PDF files are supported for statute upload",
        )

    pdf_bytes = await pdf_file.read()
    file_size_mb = len(pdf_bytes) / (1024 * 1024)

    logger.info(
        "statute_upload_start",
        short_name=short_name,
        doc_type=doc_type,
        size_mb=round(file_size_mb, 2),
        uploaded_by=str(current_user),
    )

    # Create DB record first (get the ID for chunk metadata)
    legal_doc = LegalDocument(
        title=title,
        short_name=short_name,
        doc_type=doc_type,
        storage_path=f"statutes/{short_name.replace(' ', '_')}.pdf",  # MinIO path TBD
        encryption_iv="placeholder",  # MinIO encryption TBD
        jurisdiction=jurisdiction,
        effective_date=effective_date,
        indexing_status=IndexingStatus.INDEXING,
        uploaded_by=current_user,
    )
    db.add(legal_doc)
    await db.flush()
    await db.refresh(legal_doc)

    # Save to temp file for LlamaIndex processing
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        chunk_count = await ingest_legal_pdf(
            pdf_path=tmp_path,
            document_id=str(legal_doc.id),
            short_name=short_name,
            doc_type=doc_type.value,
            db=db,
        )
    except Exception as exc:
        legal_doc.indexing_status = IndexingStatus.FAILED
        db.add(legal_doc)
        logger.error(
            "statute_indexing_failed",
            document_id=str(legal_doc.id),
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {exc}",
        ) from exc
    finally:
        os.unlink(tmp_path)

    # Update with results
    legal_doc.indexing_status = IndexingStatus.INDEXED
    legal_doc.total_chunks = chunk_count
    legal_doc.updated_at = datetime.utcnow()
    db.add(legal_doc)

    logger.info(
        "statute_upload_complete",
        document_id=str(legal_doc.id),
        short_name=short_name,
        chunk_count=chunk_count,
    )

    return {
        "document_id": str(legal_doc.id),
        "title": legal_doc.title,
        "short_name": legal_doc.short_name,
        "doc_type": legal_doc.doc_type.value,
        "indexing_status": legal_doc.indexing_status.value,
        "total_chunks": chunk_count,
        "created_at": legal_doc.created_at.isoformat(),
        "message": f"Successfully indexed {chunk_count} chunks from {short_name}",
    }


# ---------------------------------------------------------------------------
# POST /admin/upload-template
# ---------------------------------------------------------------------------
@router.post("/upload-template", status_code=status.HTTP_201_CREATED)
async def upload_template(
    db: DB,
    current_user: CurrentUser,
    title: str = Form(...),
    template_type: TemplateType = Form(default=TemplateType.DV_PETITION),
    content: str | None = Form(
        default=None,
        description="Paste petition text directly",
    ),
    pdf_file: UploadFile | None = File(
        default=None,
        description="Or upload a PDF of the past petition",
    ),
) -> dict[str, Any]:
    """Upload a lawyer's past petition as a style template.

    Accepts either raw text or a PDF file.
    The content is embedded and stored in lawyer_templates.

    Args:
        db: Database session.
        current_user: Authenticated lawyer UUID.
        title: Descriptive title for the template.
        template_type: Type of petition.
        content: Raw text of the petition.
        pdf_file: PDF file of the petition (alternative to content).

    Returns:
        Template metadata.
    """
    if not content and not pdf_file:
        raise HTTPException(
            status_code=422,
            detail="Either 'content' (text) or 'pdf_file' must be provided",
        )

    template_text = content

    # Extract text from PDF if provided
    if pdf_file and not content:
        pdf_bytes = await pdf_file.read()
        try:
            import pypdf  # type: ignore[import]
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            template_text = "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to extract text from PDF: {exc}",
            ) from exc

    if not template_text or not template_text.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from the provided source",
        )

    try:
        template = await add_template(
            db=db,
            user_id=current_user,
            title=title,
            content=template_text,
            template_type=template_type,
        )
    except TemplateServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "template_id": str(template.id),
        "title": template.title,
        "template_type": template.template_type.value,
        "content_length": len(template_text),
        "embed_model": template.embed_model,
        "created_at": template.created_at.isoformat(),
        "message": "Template added successfully. It will be used in future DV petition drafts.",
    }


# ---------------------------------------------------------------------------
# GET /admin/statutes
# ---------------------------------------------------------------------------
@router.get("/statutes")
async def list_statutes(
    db: DB,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """List all indexed legal statutes.

    Returns:
        List of statute metadata records.
    """
    result = await db.execute(
        select(LegalDocument).order_by(LegalDocument.created_at.desc())
    )
    docs = result.scalars().all()

    return {
        "total": len(docs),
        "statutes": [
            {
                "document_id": str(d.id),
                "title": d.title,
                "short_name": d.short_name,
                "doc_type": d.doc_type.value,
                "indexing_status": d.indexing_status.value,
                "total_chunks": d.total_chunks,
                "language": d.language,
                "jurisdiction": d.jurisdiction,
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ],
    }


# ---------------------------------------------------------------------------
# GET /admin/templates
# ---------------------------------------------------------------------------
@router.get("/templates")
async def list_templates(
    db: DB,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """List all lawyer templates for the current user.

    Returns:
        List of template summaries (no full content for performance).
    """
    result = await db.execute(
        select(LawyerTemplate)
        .where(LawyerTemplate.user_id == current_user)
        .where(LawyerTemplate.is_active == True)  # noqa: E712
        .order_by(LawyerTemplate.usage_count.desc())
    )
    templates = result.scalars().all()

    return {
        "total": len(templates),
        "templates": [
            {
                "template_id": str(t.id),
                "title": t.title,
                "template_type": t.template_type.value,
                "usage_count": t.usage_count,
                "embed_model": t.embed_model,
                "has_embedding": t.embedding is not None,
                "created_at": t.created_at.isoformat(),
            }
            for t in templates
        ],
    }


# ---------------------------------------------------------------------------
# DELETE /admin/templates/{template_id}
# ---------------------------------------------------------------------------
@router.delete("/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(
    template_id: uuid.UUID,
    db: DB,
    current_user: CurrentUser,
) -> None:
    """Soft-delete a lawyer template.

    Args:
        template_id: UUID of the LawyerTemplate.
        db: Database session.
        current_user: Authenticated lawyer UUID.
    """
    template = await db.get(LawyerTemplate, template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    if template.user_id != current_user:
        raise HTTPException(status_code=403, detail="Access denied")

    template.is_active = False
    template.updated_at = datetime.utcnow()
    db.add(template)

    logger.info(
        "template_soft_deleted",
        template_id=str(template_id),
        user_id=str(current_user),
    )