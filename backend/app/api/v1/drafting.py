"""Drafting API endpoints — V1 DV petition pipeline.

Endpoints:
    POST /drafting/transcribe        — Upload audio → Sarvam → transcript
    POST /drafting/extract-facts     — Transcript text → Chronology JSON
    POST /drafting/generate          — Full pipeline (transcribe + extract + draft)
    GET  /drafting/{draft_id}        — Get draft by ID
    GET  /drafting/                  — List drafts (with filters)
    PATCH /drafting/{draft_id}/feedback — Lawyer approves/rejects with notes
    POST /drafting/{draft_id}/promote   — Promote approved draft to template
"""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.database import get_db
from app.models.case import Case, CaseType, CaseStatus
from app.models.client_statement import ClientStatement, StatementLanguage, StatementStatus
from app.models.draft_petition import DraftPetition, DraftStatus
from app.services.draft_service import DraftGenerationError, generate_dv_petition_draft
from app.services.fact_extraction_service import (
    FactExtractionError,
    extract_facts_from_transcript,
    get_missing_fields,
)
from app.services.sarvam_service import SarvamTranscriptionError, transcribe_upload_file
from app.services.template_service import TemplateServiceError, promote_draft_to_template
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency: get current user (placeholder — replace with real JWT auth)
# For V1 MVP, returns a hardcoded user UUID until auth is implemented
# ---------------------------------------------------------------------------
async def get_current_user_id() -> uuid.UUID:
    """Placeholder auth dependency — replace with JWT verification in V2."""
    # TODO: Implement JWT auth
    return uuid.UUID("00000000-0000-0000-0000-000000000001")


CurrentUser = Annotated[uuid.UUID, Depends(get_current_user_id)]
DB = Annotated[AsyncSession, Depends(get_db)]


# ---------------------------------------------------------------------------
# POST /drafting/transcribe
# ---------------------------------------------------------------------------
@router.post("/transcribe", status_code=status.HTTP_201_CREATED)
async def transcribe_audio(
    db: DB,
    current_user: CurrentUser,
    audio_file: UploadFile = File(..., description="Audio file (webm/wav/mp3)"),
    case_id: uuid.UUID | None = Form(default=None),
    language_code: str = Form(default="mr-IN"),
) -> dict[str, Any]:
    """Upload an audio recording and transcribe it using Sarvam Saaras v3.

    The audio is transcribed but NOT stored in MinIO in this endpoint
    (MinIO storage is wired in the full generate pipeline).
    Use this for a quick "preview transcript" before committing.

    Args:
        db: Database session.
        current_user: Authenticated lawyer UUID.
        audio_file: Uploaded audio file.
        case_id: Optional case UUID to associate.
        language_code: BCP-47 language code (default: mr-IN for Marathi).

    Returns:
        JSON with statement_id, transcript preview (first 500 chars), language, duration.
    """
    try:
        result = await transcribe_upload_file(
            upload_file=audio_file,
            language_code=language_code,
        )
    except SarvamTranscriptionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # Persist statement record
    statement = ClientStatement(
        case_id=case_id,
        user_id=current_user,
        language=StatementLanguage(result.language_code)
        if result.language_code in [lang.value for lang in StatementLanguage]
        else StatementLanguage.MARATHI,
        transcript_raw=result.transcript,
        transcript_clean=result.transcript,  # same at this stage — cleaning TBD
        sarvam_request_id=result.request_id,
        audio_duration_seconds=result.duration_seconds,
        audio_mime_type=audio_file.content_type or "audio/webm",
        status=StatementStatus.TRANSCRIBED,
    )
    db.add(statement)
    await db.flush()
    await db.refresh(statement)

    logger.info(
        "transcription_saved",
        statement_id=str(statement.id),
        case_id=str(case_id) if case_id else None,
        sarvam_request_id=result.request_id,
    )

    return {
        "statement_id": str(statement.id),
        "status": statement.status.value,
        "language_detected": result.language_code,
        "duration_seconds": result.duration_seconds,
        "sarvam_request_id": result.request_id,
        # Return first 500 chars as preview — full text available via separate endpoint
        "transcript_preview": (result.transcript[:500] + "...")
        if len(result.transcript) > 500
        else result.transcript,
        "transcript_length": len(result.transcript),
    }


# ---------------------------------------------------------------------------
# POST /drafting/extract-facts
# ---------------------------------------------------------------------------
@router.post("/extract-facts")
async def extract_facts(
    db: DB,
    current_user: CurrentUser,
    statement_id: uuid.UUID = Form(...),
) -> dict[str, Any]:
    """Extract structured facts (Chronology of Events) from a transcribed statement.

    Args:
        db: Database session.
        current_user: Authenticated lawyer UUID.
        statement_id: UUID of the ClientStatement to process.

    Returns:
        Chronology JSON with missing field list.
    """
    statement = await db.get(ClientStatement, statement_id)
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")
    if statement.user_id != current_user:
        raise HTTPException(status_code=403, detail="Access denied")
    if not statement.transcript_clean and not statement.transcript_raw:
        raise HTTPException(status_code=422, detail="Statement has no transcript")

    transcript = statement.transcript_clean or statement.transcript_raw or ""

    try:
        facts = await extract_facts_from_transcript(
            transcript=transcript,
            case_id=str(statement.case_id) if statement.case_id else None,
        )
    except FactExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    missing_fields = get_missing_fields(facts)

    # Update statement status
    statement.status = StatementStatus.FACTS_EXTRACTED
    db.add(statement)

    return {
        "statement_id": str(statement_id),
        "facts": facts,
        "missing_fields": missing_fields,
        "missing_count": len(missing_fields),
        "ready_to_draft": len(missing_fields) == 0,
    }


# ---------------------------------------------------------------------------
# POST /drafting/generate
# ---------------------------------------------------------------------------
@router.post("/generate", status_code=status.HTTP_201_CREATED)
async def generate_draft(
    db: DB,
    current_user: CurrentUser,
    statement_id: uuid.UUID = Form(...),
    case_id: uuid.UUID = Form(...),
    pre_extracted_facts: str | None = Form(
        default=None,
        description="Optional: JSON string of already-extracted facts",
    ),
) -> dict[str, Any]:
    """Run the full Three-Point Blend to generate a DV petition draft.

    Prerequisites:
        1. Statement must have a transcript (use /transcribe first).
        2. DV Act must be indexed (use /admin/upload-statute first — one-time setup).
        3. Optionally: lawyer templates uploaded via /admin/upload-template.

    Args:
        db: Database session.
        current_user: Authenticated lawyer UUID.
        statement_id: UUID of the ClientStatement.
        case_id: UUID of the Case.
        pre_extracted_facts: Optional JSON string of pre-extracted facts
                             (from /extract-facts endpoint).

    Returns:
        Draft petition metadata. Use GET /drafting/{draft_id} to get full text.
    """
    statement = await db.get(ClientStatement, statement_id)
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")
    if statement.user_id != current_user:
        raise HTTPException(status_code=403, detail="Access denied")

    # Auto-create the case if it doesn't exist yet.
    # The facts page generates a fresh UUID for every session; we honour it
    # by creating a minimal DV case record on first use.
    case = await db.get(Case, case_id)
    if not case:
        from datetime import date as _date
        case = Case(
            id=case_id,
            title=f"DV Petition — {_date.today().strftime('%d %b %Y')}",
            case_type=CaseType.FAMILY,
            court_name="Family Court",
            court_district="Raigad",
            status=CaseStatus.ACTIVE,
            user_id=current_user,
        )
        db.add(case)
        await db.flush()
        logger.info("case_auto_created", case_id=str(case_id), user_id=str(current_user))
    elif case.user_id != current_user:
        raise HTTPException(status_code=403, detail="Access denied to case")

    # Parse pre-extracted facts if provided
    import json as _json
    facts_dict = None
    if pre_extracted_facts:
        try:
            facts_dict = _json.loads(pre_extracted_facts)
        except _json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid facts JSON: {exc}",
            ) from exc

    try:
        draft = await generate_dv_petition_draft(
            db=db,
            statement=statement,
            user_id=current_user,
            case_id=case_id,
            pre_extracted_facts=facts_dict,
        )
    except (DraftGenerationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "draft_id": str(draft.id),
        "case_id": str(draft.case_id),
        "statement_id": str(draft.statement_id),
        "status": draft.status.value,
        "version": draft.version,
        "legal_sections_used": draft.legal_sections_used,
        "templates_used_count": len(draft.template_ids_used),
        "missing_fields": get_missing_fields(draft.facts_json),
        "created_at": draft.created_at.isoformat(),
        # Full text available via GET /drafting/{draft_id}
        "draft_preview": (draft.draft_text[:800] + "...")
        if len(draft.draft_text) > 800
        else draft.draft_text,
    }


# ---------------------------------------------------------------------------
# GET /drafting/{draft_id}
# ---------------------------------------------------------------------------
@router.get("/{draft_id}")
async def get_draft(
    draft_id: uuid.UUID,
    db: DB,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Retrieve a complete draft petition by ID.

    Args:
        draft_id: UUID of the DraftPetition.
        db: Database session.
        current_user: Authenticated lawyer UUID.

    Returns:
        Full draft text + metadata.
    """
    draft = await db.get(DraftPetition, draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    if draft.user_id != current_user:
        raise HTTPException(status_code=403, detail="Access denied")

    return {
        "draft_id": str(draft.id),
        "case_id": str(draft.case_id),
        "statement_id": str(draft.statement_id),
        "status": draft.status.value,
        "version": draft.version,
        "draft_text": draft.draft_text,
        "disclaimer": draft.disclaimer,
        "facts_json": draft.facts_json,
        "legal_sections_used": draft.legal_sections_used,
        "missing_fields": get_missing_fields(draft.facts_json),
        "lawyer_feedback": draft.lawyer_feedback,
        "generation_model": draft.generation_model,
        "created_at": draft.created_at.isoformat(),
        "updated_at": draft.updated_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# GET /drafting/
# ---------------------------------------------------------------------------
@router.get("/")
async def list_drafts(
    db: DB,
    current_user: CurrentUser,
    case_id: uuid.UUID | None = None,
    status_filter: DraftStatus | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """List draft petitions for the current user.

    Args:
        db: Database session.
        current_user: Authenticated lawyer UUID.
        case_id: Optional filter by case.
        status_filter: Optional filter by draft status.
        limit: Page size.
        offset: Pagination offset.

    Returns:
        List of draft summaries (no full text for performance).
    """
    query = select(DraftPetition).where(DraftPetition.user_id == current_user)

    if case_id:
        query = query.where(DraftPetition.case_id == case_id)
    if status_filter:
        query = query.where(DraftPetition.status == status_filter)

    query = query.order_by(DraftPetition.created_at.desc()).limit(limit).offset(offset)

    result = await db.execute(query)
    drafts = result.scalars().all()

    return {
        "total": len(drafts),
        "drafts": [
            {
                "draft_id": str(d.id),
                "case_id": str(d.case_id),
                "status": d.status.value,
                "version": d.version,
                "legal_sections_used": d.legal_sections_used,
                "missing_fields_count": len(get_missing_fields(d.facts_json)),
                "created_at": d.created_at.isoformat(),
            }
            for d in drafts
        ],
    }


# ---------------------------------------------------------------------------
# PATCH /drafting/{draft_id}/feedback
# ---------------------------------------------------------------------------
@router.patch("/{draft_id}/feedback")
async def submit_feedback(
    draft_id: uuid.UUID,
    db: DB,
    current_user: CurrentUser,
    new_status: DraftStatus = Form(...),
    feedback_notes: str | None = Form(default=None),
    updated_draft_text: str | None = Form(
        default=None,
        description="If lawyer edits the draft text directly",
    ),
) -> dict[str, Any]:
    """Submit lawyer's review feedback on a draft.

    Transitions:
        DRAFT → UNDER_REVIEW → APPROVED (eligible for template promotion)
        DRAFT → REJECTED

    Args:
        draft_id: UUID of the DraftPetition.
        db: Database session.
        current_user: Authenticated lawyer UUID.
        new_status: Target status.
        feedback_notes: Lawyer's comments for pipeline improvement.
        updated_draft_text: If the lawyer edited the text in the UI.

    Returns:
        Updated draft metadata.
    """
    draft = await db.get(DraftPetition, draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    if draft.user_id != current_user:
        raise HTTPException(status_code=403, detail="Access denied")

    draft.status = new_status
    if feedback_notes:
        draft.lawyer_feedback = feedback_notes
    if updated_draft_text:
        draft.draft_text = updated_draft_text

    from datetime import datetime
    draft.updated_at = datetime.utcnow()
    db.add(draft)

    logger.info(
        "draft_feedback_submitted",
        draft_id=str(draft_id),
        user_id=str(current_user),
        new_status=new_status.value,
    )

    return {
        "draft_id": str(draft.id),
        "status": draft.status.value,
        "message": f"Draft status updated to {new_status.value}",
    }


# ---------------------------------------------------------------------------
# POST /drafting/{draft_id}/promote
# ---------------------------------------------------------------------------
@router.post("/{draft_id}/promote", status_code=status.HTTP_201_CREATED)
async def promote_to_template(
    draft_id: uuid.UUID,
    db: DB,
    current_user: CurrentUser,
    template_title: str = Form(...),
) -> dict[str, Any]:
    """Promote an APPROVED draft to a reusable lawyer template.

    Only APPROVED drafts can be promoted. Once promoted, this draft's
    style will be used as a few-shot example for future cases.

    Args:
        draft_id: UUID of the approved DraftPetition.
        db: Database session.
        current_user: Authenticated lawyer UUID.
        template_title: Descriptive title for the template.

    Returns:
        New template metadata.
    """
    try:
        template = await promote_draft_to_template(
            db=db,
            draft_id=draft_id,
            user_id=current_user,
            title=template_title,
        )
    except TemplateServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "template_id": str(template.id),
        "title": template.title,
        "template_type": template.template_type.value,
        "created_at": template.created_at.isoformat(),
        "message": "Draft successfully promoted to template. It will be used in future drafts.",
    }