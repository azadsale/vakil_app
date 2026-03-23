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

import asyncio
import json
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
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
from app.services.document_extraction_service import (
    DocumentExtractionError,
    SUPPORTED_DOC_MIME_TYPES,
    extract_text_from_document,
)
from app.services.model_router import get_all_quotas
from app.services.sarvam_service import SarvamTranscriptionError, transcribe_upload_file
from app.services.template_service import TemplateServiceError, promote_draft_to_template
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _collect_leaf_values(obj: Any, results: list | None = None) -> list:
    """Recursively collect all leaf (non-dict, non-list) values from a nested structure."""
    if results is None:
        results = []
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_leaf_values(v, results)
    elif isinstance(obj, list):
        for item in obj:
            _collect_leaf_values(item, results)
    else:
        results.append(obj)
    return results


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
# POST /drafting/upload-document
# ---------------------------------------------------------------------------
@router.post("/upload-document", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    db: DB,
    current_user: CurrentUser,
    document_file: UploadFile = File(..., description="PDF, DOCX, JPG, or PNG of client's written statement"),
    case_id: uuid.UUID | None = Form(default=None),
    language_code: str = Form(default="mr-IN"),
) -> dict[str, Any]:
    """Upload a client's written statement — returns a job_id immediately.

    For typed PDFs/DOCX: completes quickly (seconds).
    For scanned/CamScanner PDFs and images: OCR runs in the background.
    Poll GET /drafting/ocr-status/{job_id} for live progress.

    Returns:
        {job_id, mode} — poll /ocr-status/{job_id} until status == "done".
    """
    mime_type = document_file.content_type or "application/octet-stream"
    mime_base = mime_type.split(";")[0].strip().lower()

    if mime_base not in {m.split(";")[0].strip() for m in SUPPORTED_DOC_MIME_TYPES}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: {mime_type}. Supported: PDF, DOCX, JPG, PNG, TIFF",
        )

    file_bytes = await document_file.read()
    filename = document_file.filename or "statement.pdf"
    job_id = str(uuid.uuid4())

    # Kick off background processing — returns immediately
    background_tasks.add_task(
        _run_document_ocr_job,
        job_id=job_id,
        file_bytes=file_bytes,
        mime_type=mime_type,
        filename=filename,
        language_code=language_code,
        case_id=str(case_id) if case_id else None,
        user_id=str(current_user),
    )

    return {"job_id": job_id, "status": "processing"}


# ---------------------------------------------------------------------------
# Background OCR worker
# ---------------------------------------------------------------------------
async def _run_document_ocr_job(
    job_id: str,
    file_bytes: bytes,
    mime_type: str,
    filename: str,
    language_code: str,
    case_id: str | None,
    user_id: str,
) -> None:
    """Background task: run extraction, update Redis with progress, save statement."""
    import redis.asyncio as aioredis  # type: ignore[import]
    from app.config import get_settings
    from app.database import AsyncSessionLocal as async_session_factory
    from app.models.client_statement import StatementLanguage

    settings = get_settings()
    r = aioredis.from_url(settings.redis_url, decode_responses=True)

    async def _set(key: str, val: Any) -> None:
        await r.setex(f"ocr:{job_id}:{key}", 3600, json.dumps(val))  # 1hr TTL

    await _set("status", "processing")
    await _set("progress", {"current": 0, "total": 0, "filename": filename})

    # Shared list so the thread can push updates; the main loop drains them
    _progress_updates: list[dict] = []

    def on_page_done(current: int, total: int) -> None:
        _progress_updates.append({"current": current, "total": total, "filename": filename})

    async def _flush_progress() -> None:
        """Drain progress updates written by the OCR thread into Redis."""
        while True:
            await asyncio.sleep(1)
            while _progress_updates:
                update = _progress_updates.pop(0)
                await _set("progress", update)

    flush_task = asyncio.create_task(_flush_progress())

    try:
        # extract_text_from_document is now async (uses Gemini Vision for OCR)
        extracted_text, method, char_count = await extract_text_from_document(
            file_bytes=file_bytes,
            mime_type=mime_type,
            filename=filename,
            on_page_done=on_page_done,
        )

        # Persist statement in a new DB session
        async with async_session_factory() as db:
            statement = ClientStatement(
                case_id=uuid.UUID(case_id) if case_id else None,
                user_id=uuid.UUID(user_id),
                language=StatementLanguage(language_code)
                if language_code in [l.value for l in StatementLanguage]
                else StatementLanguage.MARATHI,
                transcript_raw=extracted_text,
                transcript_clean=extracted_text,
                audio_mime_type=mime_type,
                status=StatementStatus.TRANSCRIBED,
            )
            db.add(statement)
            await db.commit()
            await db.refresh(statement)
            statement_id = str(statement.id)

        preview = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
        await _set("status", "done")
        await _set("result", {
            "statement_id": statement_id,
            "extraction_method": method,
            "char_count": char_count,
            "transcript_preview": preview,
            "transcript_length": char_count,
            "language_detected": language_code,
        })
        logger.info("ocr_job_complete", job_id=job_id, method=method, chars=char_count)

    except Exception as exc:
        await _set("status", "error")
        await _set("error", str(exc))
        logger.error("ocr_job_failed", job_id=job_id, error=str(exc))
    finally:
        flush_task.cancel()
        await r.aclose()


# ---------------------------------------------------------------------------
# GET /drafting/ocr-status/{job_id}
# ---------------------------------------------------------------------------
@router.get("/ocr-status/{job_id}")
async def get_ocr_status(job_id: str) -> dict[str, Any]:
    """Poll the status of a background OCR document extraction job.

    Returns:
        processing → {status, current_page, total_pages, filename}
        done       → {status, statement_id, extraction_method, char_count, preview}
        error      → {status, error}
    """
    import redis.asyncio as aioredis  # type: ignore[import]
    from app.config import get_settings

    settings = get_settings()
    r = aioredis.from_url(settings.redis_url, decode_responses=True)

    try:
        raw_status = await r.get(f"ocr:{job_id}:status")
        if raw_status is None:
            raise HTTPException(status_code=404, detail="OCR job not found or expired")

        job_status = json.loads(raw_status)

        if job_status == "processing":
            raw_progress = await r.get(f"ocr:{job_id}:progress")
            progress = json.loads(raw_progress) if raw_progress else {}
            return {
                "status": "processing",
                "current_page": progress.get("current", 0),
                "total_pages": progress.get("total", 0),
                "filename": progress.get("filename", ""),
            }

        if job_status == "done":
            raw_result = await r.get(f"ocr:{job_id}:result")
            result = json.loads(raw_result) if raw_result else {}
            return {"status": "done", **result}

        # error
        raw_error = await r.get(f"ocr:{job_id}:error")
        error_msg = json.loads(raw_error) if raw_error else "Unknown error"
        return {"status": "error", "error": error_msg}

    finally:
        await r.aclose()


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
    language: str = Form(default="english", description="Draft language: 'english' or 'marathi'"),
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
            court_district="Pune",
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
            # Log to verify edited facts are actually reaching the backend
            pet_name = facts_dict.get("petitioner", {}).get("name", "?")
            n_missing = sum(
                1 for v in _collect_leaf_values(facts_dict)
                if isinstance(v, str) and v.startswith("[MISSING:")
            )
            logger.info(
                "pre_extracted_facts_received",
                statement_id=str(statement_id),
                petitioner_name=pet_name,
                missing_count=n_missing,
                total_keys=len(str(facts_dict)),
            )
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
            language=language,
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


# -------------------------------------------------------------------------
# Recent statements cache — MUST be before /{draft_id} catch-all
# -------------------------------------------------------------------------

@router.get("/recent-statements")
async def list_recent_statements(
    db: DB,
    current_user: CurrentUser,
    limit: int = 10,
) -> dict[str, Any]:
    """List recent successful OCR/transcription results for the current user.

    Returns cached Step 1 outputs so the user can reuse them without
    re-uploading and re-OCR'ing the same document (saves Gemini quota).

    Args:
        db: Database session.
        current_user: Authenticated lawyer UUID.
        limit: Max number of results (default 10).

    Returns:
        List of recent statements with id, preview, char_count, created_at.
    """
    from sqlalchemy import desc

    query = (
        select(ClientStatement)
        .where(
            ClientStatement.user_id == current_user,
            ClientStatement.status.in_([
                StatementStatus.TRANSCRIBED,
                StatementStatus.FACTS_EXTRACTED,
                StatementStatus.DRAFT_GENERATED,
            ]),
            ClientStatement.transcript_clean.isnot(None),  # type: ignore[union-attr]
        )
        .order_by(desc(ClientStatement.created_at))  # type: ignore[arg-type]
        .limit(limit)
    )
    result = await db.execute(query)
    statements = result.scalars().all()

    items = []
    for stmt in statements:
        text = stmt.transcript_clean or stmt.transcript_raw or ""
        char_count = len(text)

        # Skip obviously failed extractions (all pages = "OCR failed")
        if char_count < 100:
            continue
        if "OCR failed" in text and text.count("OCR failed") > text.count("\n") * 0.5:
            continue

        preview = (text[:300] + "...") if len(text) > 300 else text

        # Determine source type from audio_mime_type
        mime = stmt.audio_mime_type or ""
        if "pdf" in mime or "octet-stream" in mime:
            source = "document"
        elif "image" in mime:
            source = "image"
        elif "docx" in mime or "word" in mime:
            source = "document"
        else:
            source = "voice"

        # Quality indicator based on char count and content
        quality = "good" if char_count > 5000 else "partial"

        items.append({
            "statement_id": str(stmt.id),
            "preview": preview,
            "char_count": char_count,
            "quality": quality,
            "language": stmt.language.value if stmt.language else "mr-IN",
            "source": source,
            "status": stmt.status.value,
            "created_at": stmt.created_at.isoformat() if stmt.created_at else None,
        })

    return {
        "statements": items,
        "count": len(items),
    }


# -------------------------------------------------------------------------
# Quota / health — MUST be before /{draft_id} catch-all
# -------------------------------------------------------------------------

@router.get("/quota")
async def get_quota_status():
    """Return current daily usage and remaining quota for all Gemini model tiers.

    The model router automatically selects the best model that has enough
    remaining quota for a given task. This endpoint shows the current state.
    """
    quotas = await get_all_quotas()
    from app.config import get_settings
    n_keys = len(get_settings().gemini_api_keys) or 1
    return {
        "quotas": quotas,
        "num_api_keys": n_keys,
        "note": f"Using {n_keys} API key(s). The router tries the best model+key first. "
                "If quota is exhausted, it rotates to the next key or falls back to a lower tier.",
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