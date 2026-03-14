"""Template service — manage and retrieve lawyer's style templates.

Handles:
- Embedding new lawyer templates (past petition drafts)
- Cosine similarity search to find top-K relevant templates for a new case
- Template promotion: approved DraftPetition → LawyerTemplate
- PII-safe handling (template content never logged)
"""

import uuid
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.draft_petition import DraftPetition, DraftStatus
from app.models.lawyer_template import LawyerTemplate, TemplateType
from app.services.rag_service import embed_single
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class TemplateServiceError(Exception):
    """Raised on template service errors."""


async def generate_embedding(text_content: str) -> list[float]:
    """Generate a 384-dim embedding using fastembed (delegated to rag_service).

    Args:
        text_content: Text to embed.

    Returns:
        List of 384 floats.

    Raises:
        TemplateServiceError: If embedding fails.
    """
    try:
        return embed_single(text_content)
    except Exception as exc:
        logger.error("embedding_generation_failed", error=str(exc))
        raise TemplateServiceError(f"Embedding failed: {exc}") from exc


async def add_template(
    db: AsyncSession,
    user_id: uuid.UUID,
    title: str,
    content: str,
    template_type: TemplateType = TemplateType.DV_PETITION,
    source_draft_id: uuid.UUID | None = None,
) -> LawyerTemplate:
    """Add a new lawyer template with embedding.

    Args:
        db: Async DB session.
        user_id: Lawyer's user UUID.
        title: Template title for admin reference.
        content: Full petition text (PII — not logged).
        template_type: Category of the template.
        source_draft_id: If promoted from DraftPetition, the draft's UUID.

    Returns:
        Persisted LawyerTemplate instance.
    """
    logger.info(
        "template_add_start",
        user_id=str(user_id),
        template_type=template_type,
        title_length=len(title),  # not logging title itself — may contain client name
    )

    embedding = await generate_embedding(content)

    template = LawyerTemplate(
        user_id=user_id,
        title=title,
        template_type=template_type,
        content=content,
        embedding=embedding,
        embed_model=settings.llama_index_embed_model,
        source_draft_id=source_draft_id,
    )

    db.add(template)
    await db.flush()
    await db.refresh(template)

    logger.info(
        "template_add_complete",
        template_id=str(template.id),
        user_id=str(user_id),
    )

    return template


async def get_top_templates(
    db: AsyncSession,
    user_id: uuid.UUID,
    query_text: str,
    top_k: int = 3,
    template_type: TemplateType = TemplateType.DV_PETITION,
) -> list[dict[str, Any]]:
    """Retrieve top-K lawyer templates most similar to the query.

    Uses pgvector cosine similarity (1 - cosine_distance).
    Only returns templates belonging to the requesting user (RLS + explicit filter).

    Args:
        db: Async DB session.
        user_id: Lawyer's UUID — enforces data isolation.
        query_text: The case facts summary or transcript excerpt to match against.
        top_k: Number of templates to return.
        template_type: Filter by petition type.

    Returns:
        List of dicts with keys: ``id``, ``title``, ``content``, ``similarity``.
        Ordered by similarity descending.
    """
    if not query_text.strip():
        return []

    query_embedding = await generate_embedding(query_text)

    # pgvector cosine similarity via raw SQL
    # <=> is the cosine distance operator; 1 - distance = similarity
    stmt = text("""
        SELECT
            id,
            title,
            content,
            template_type,
            usage_count,
            1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM lawyer_templates
        WHERE
            user_id = :user_id
            AND template_type = :template_type
            AND is_active = TRUE
            AND embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    result = await db.execute(
        stmt,
        {
            "embedding": str(query_embedding),
            "user_id": str(user_id),
            "template_type": template_type.value,
            "top_k": top_k,
        },
    )
    rows = result.fetchall()

    templates = []
    for row in rows:
        templates.append({
            "id": str(row.id),
            "title": row.title,
            "content": row.content,  # PII — caller must not log this
            "template_type": row.template_type,
            "usage_count": row.usage_count,
            "similarity": float(row.similarity),
        })

    logger.info(
        "template_retrieval_complete",
        user_id=str(user_id),
        templates_found=len(templates),
        top_similarity=templates[0]["similarity"] if templates else 0.0,
    )

    return templates


async def promote_draft_to_template(
    db: AsyncSession,
    draft_id: uuid.UUID,
    user_id: uuid.UUID,
    title: str,
) -> LawyerTemplate:
    """Promote an approved DraftPetition to a LawyerTemplate.

    Called when the lawyer approves a generated draft for use as a future style reference.

    Args:
        db: Async DB session.
        draft_id: UUID of the approved DraftPetition.
        user_id: Lawyer's UUID (ownership verification).
        title: Title for the new template.

    Returns:
        New LawyerTemplate instance.

    Raises:
        TemplateServiceError: If draft not found, not approved, or wrong owner.
    """
    draft = await db.get(DraftPetition, draft_id)
    if not draft:
        raise TemplateServiceError(f"DraftPetition {draft_id} not found")
    if draft.user_id != user_id:
        raise TemplateServiceError("Unauthorized: draft does not belong to this user")
    if draft.status != DraftStatus.APPROVED:
        raise TemplateServiceError(
            f"Only APPROVED drafts can be promoted to templates. "
            f"Current status: {draft.status}"
        )

    # Check not already promoted
    existing = await db.execute(
        select(LawyerTemplate).where(
            LawyerTemplate.source_draft_id == draft_id
        )
    )
    if existing.scalar_one_or_none():
        raise TemplateServiceError(
            f"Draft {draft_id} has already been promoted to a template"
        )

    template = await add_template(
        db=db,
        user_id=user_id,
        title=title,
        content=draft.draft_text,
        template_type=TemplateType.DV_PETITION,
        source_draft_id=draft_id,
    )

    logger.info(
        "draft_promoted_to_template",
        draft_id=str(draft_id),
        template_id=str(template.id),
        user_id=str(user_id),
    )

    return template


async def increment_template_usage(
    db: AsyncSession,
    template_ids: list[str],
) -> None:
    """Increment usage_count for templates after they are used in generation.

    Args:
        db: Async DB session.
        template_ids: List of template UUID strings.
    """
    if not template_ids:
        return

    await db.execute(
        text("""
            UPDATE lawyer_templates
            SET usage_count = usage_count + 1,
                updated_at = NOW()
            WHERE id = ANY(CAST(:ids AS uuid[]))
        """),
        {"ids": template_ids},
    )