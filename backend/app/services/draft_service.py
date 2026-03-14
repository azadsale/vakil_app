"""Draft orchestrator service — the Three-Point Blend.

Combines:
    1. Client facts (ChronologyOfEvents JSON from fact_extraction_service)
    2. Legal grounding (DV Act sections from rag_service)
    3. Lawyer style (top-K templates from template_service)

→ Generates a complete, citation-backed DV petition draft.

Anti-hallucination guarantees:
    - LLM receives ONLY the retrieved law text, not its training knowledge of law.
    - System prompt explicitly forbids citing sections not in context.
    - [MISSING: FIELD] placeholders are preserved — never filled by the LLM.
    - Mandatory disclaimer appended to every draft.
"""

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.client_statement import ClientStatement, StatementStatus
from app.models.draft_petition import DraftPetition, DraftStatus
from app.services.fact_extraction_service import (
    CHRONOLOGY_SCHEMA,
    extract_facts_from_transcript,
    get_missing_fields,
)
from app.services.rag_service import build_legal_context_string, query_dv_act
from app.services.template_service import (
    get_top_templates,
    increment_template_usage,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Draft Generation Prompt
# ---------------------------------------------------------------------------
DRAFT_GENERATION_SYSTEM_PROMPT = """You are an expert legal drafter specializing in Protection of Women from Domestic Violence Act, 2005 petitions for Indian courts.

Your task is to draft a formal petition (application under Section 12 of the DV Act) based on:
1. The client's facts (provided as structured JSON)
2. Relevant legal provisions (retrieved from the DV Act — use ONLY these, do not cite from memory)
3. A style reference (a past petition by the advocate — match this tone, structure, and language exactly)

STRICT RULES:
1. NEVER cite any legal section or case law not explicitly provided in the "LEGAL GROUNDING" section below.
2. If the facts JSON contains "[MISSING: FIELD]", include it verbatim as a placeholder in the draft — do NOT invent values.
3. Use formal legal language consistent with Maharashtra district court standards.
4. Structure: Title → Court Header → Petitioner Details → Respondent Details → Background → Incidents (chronological) → Legal Basis → Reliefs Sought → Prayer Clause → Verification.
5. Address the court as "Hon'ble Court" or "Your Honour".
6. Use "the Respondent" not the person's name in the body (refer by name only in the header).
7. Reliefs claimed must map ONLY to the sections present in the legal grounding.
8. Do NOT add any facts not present in the JSON.
9. End with the standard verification clause.

OUTPUT: Return the complete petition text only. No commentary, no JSON wrapper."""

DRAFT_GENERATION_USER_TEMPLATE = """CASE FACTS (Chronology of Events):
{facts_json}

LEGAL GROUNDING (DV Act 2005 — retrieved sections):
{legal_context}

STYLE REFERENCE (past petition by the advocate — match this tone and structure):
{style_reference}

---
Draft a complete Section 12 petition based on the above. Preserve all [MISSING: FIELD] placeholders exactly as-is."""

MANDATORY_DISCLAIMER = """
---
LEGAL DISCLAIMER: This petition has been AI-assisted using Legal-CoPilot. It has been generated based on information provided by the client and must be thoroughly reviewed, verified for accuracy, and approved by the advocate before filing. This draft does not constitute legal advice and should not be filed without advocate supervision. Facts marked as [MISSING: FIELD] must be verified and completed before filing.
Generated: {timestamp} | Model: {model}
---"""


class DraftGenerationError(Exception):
    """Raised when draft generation fails."""


async def generate_dv_petition_draft(
    db: AsyncSession,
    statement: ClientStatement,
    user_id: uuid.UUID,
    case_id: uuid.UUID,
    pre_extracted_facts: dict[str, Any] | None = None,
) -> DraftPetition:
    """Orchestrate the Three-Point Blend to generate a DV petition draft.

    This is the core V1 pipeline:
        Step 1: Extract facts (or use pre-extracted)
        Step 2: Query DV Act RAG for relevant sections
        Step 3: Retrieve lawyer's top-3 style templates
        Step 4: Generate draft via GPT-4o
        Step 5: Persist to draft_petitions table
        Step 6: Update statement status

    Args:
        db: Async DB session.
        statement: ClientStatement instance with transcript.
        user_id: Lawyer's UUID.
        case_id: Case UUID.
        pre_extracted_facts: If already extracted (from /extract-facts endpoint),
                              skip re-extraction.

    Returns:
        Persisted DraftPetition instance.

    Raises:
        DraftGenerationError: On any pipeline failure.
        ValueError: If statement has no transcript.
    """
    if not statement.transcript_clean and not statement.transcript_raw:
        raise ValueError("Statement has no transcript — transcribe audio first")

    transcript = statement.transcript_clean or statement.transcript_raw or ""

    logger.info(
        "draft_pipeline_start",
        statement_id=str(statement.id),
        case_id=str(case_id),
        user_id=str(user_id),
        has_pre_extracted_facts=pre_extracted_facts is not None,
    )

    # ------------------------------------------------------------------
    # Step 1: Fact Extraction
    # ------------------------------------------------------------------
    if pre_extracted_facts:
        facts = pre_extracted_facts
        logger.info("draft_using_pre_extracted_facts", case_id=str(case_id))
    else:
        facts = await extract_facts_from_transcript(
            transcript=transcript,
            case_id=str(case_id),
        )

    missing_fields = get_missing_fields(facts)
    if missing_fields:
        logger.info(
            "draft_missing_fields_detected",
            case_id=str(case_id),
            missing_count=len(missing_fields),
            # Log field NAMES (not values) — safe
            missing_fields=missing_fields,
        )

    # ------------------------------------------------------------------
    # Step 2: Legal Grounding (RAG over DV Act)
    # ------------------------------------------------------------------
    # Build a query from the facts for better retrieval
    relief_types = facts.get("reliefs_sought", [])
    incident_types = [
        inc.get("incident_type", "")
        for inc in facts.get("incidents", [])
    ]
    rag_query = _build_rag_query(relief_types, incident_types)

    retrieved_sections = await query_dv_act(query=rag_query, top_k=5)
    legal_context = build_legal_context_string(retrieved_sections)
    legal_sections_used = [s["source_citation"] for s in retrieved_sections]

    # ------------------------------------------------------------------
    # Step 3: Style Templates (Few-Shot)
    # ------------------------------------------------------------------
    # Use a summary of facts as the query for semantic template matching
    facts_summary = _summarize_facts_for_template_query(facts)
    templates = await get_top_templates(
        db=db,
        user_id=user_id,
        query_text=facts_summary,
        top_k=3,
    )

    # Combine top templates as style reference
    if templates:
        style_reference = _format_templates_as_shots(templates)
        template_ids_used = [t["id"] for t in templates]
    else:
        style_reference = (
            "No previous templates available. "
            "Draft in standard Maharashtra district court petition format."
        )
        template_ids_used = []

    # ------------------------------------------------------------------
    # Step 4: Draft Generation
    # ------------------------------------------------------------------
    api_key = settings.openai_api_key.get_secret_value()
    if not api_key:
        raise DraftGenerationError("OPENAI_API_KEY not configured")

    client = AsyncOpenAI(api_key=api_key)

    user_prompt = DRAFT_GENERATION_USER_TEMPLATE.format(
        facts_json=json.dumps(facts, indent=2, ensure_ascii=False),
        legal_context=legal_context,
        style_reference=style_reference,
    )

    # Hash prompt for dedup
    prompt_hash = hashlib.sha256(user_prompt.encode()).hexdigest()

    logger.info(
        "draft_llm_call_start",
        case_id=str(case_id),
        prompt_hash=prompt_hash[:16],
        model=settings.llama_index_llm_model,
        legal_sections_count=len(legal_sections_used),
        templates_used=len(template_ids_used),
    )

    try:
        response = await client.chat.completions.create(
            model=settings.llama_index_llm_model,
            messages=[
                {"role": "system", "content": DRAFT_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # slight creativity for natural language, not for facts
            max_tokens=8192,
        )
    except Exception as exc:
        logger.error(
            "draft_llm_call_failed",
            case_id=str(case_id),
            error=str(exc),
        )
        raise DraftGenerationError(f"LLM generation failed: {exc}") from exc

    draft_text = response.choices[0].message.content or ""
    if not draft_text:
        raise DraftGenerationError("LLM returned empty draft")

    # Append mandatory disclaimer
    disclaimer = MANDATORY_DISCLAIMER.format(
        timestamp=datetime.utcnow().isoformat(),
        model=settings.llama_index_llm_model,
    )
    draft_text_with_disclaimer = draft_text + disclaimer

    # ------------------------------------------------------------------
    # Step 5: Persist DraftPetition
    # ------------------------------------------------------------------
    draft = DraftPetition(
        case_id=case_id,
        statement_id=statement.id,
        user_id=user_id,
        facts_json=facts,
        legal_sections_used=legal_sections_used,
        template_ids_used=template_ids_used,
        draft_text=draft_text_with_disclaimer,
        generation_model=settings.llama_index_llm_model,
        generation_prompt_hash=prompt_hash,
    )
    db.add(draft)
    await db.flush()
    await db.refresh(draft)

    # ------------------------------------------------------------------
    # Step 6: Update template usage counts + statement status
    # ------------------------------------------------------------------
    await increment_template_usage(db=db, template_ids=template_ids_used)

    statement.status = StatementStatus.DRAFT_GENERATED
    db.add(statement)

    logger.info(
        "draft_pipeline_complete",
        draft_id=str(draft.id),
        case_id=str(case_id),
        statement_id=str(statement.id),
        draft_char_count=len(draft_text),
        missing_fields_count=len(missing_fields),
    )

    return draft


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_rag_query(
    relief_types: list[str],
    incident_types: list[str],
) -> str:
    """Build a semantic query for DV Act RAG retrieval.

    Args:
        relief_types: List of relief codes (e.g. ["protection_order", "monetary_relief"]).
        incident_types: List of incident types (e.g. ["physical", "economic"]).

    Returns:
        Natural language query string.
    """
    relief_map = {
        "protection_order": "protection order Section 18",
        "residence_order": "residence order Section 19",
        "monetary_relief": "monetary relief maintenance Section 20",
        "custody_order": "custody of children Section 21",
        "compensation_order": "compensation damages Section 22",
    }
    incident_map = {
        "physical": "physical violence bodily harm",
        "sexual": "sexual violence",
        "emotional": "emotional abuse mental cruelty",
        "verbal": "verbal abuse threats intimidation",
        "economic": "economic abuse financial control",
    }

    relief_terms = " ".join(relief_map.get(r, r) for r in relief_types)
    incident_terms = " ".join(incident_map.get(i, i) for i in incident_types)

    return (
        f"Domestic violence petition application Section 12 "
        f"{incident_terms} {relief_terms} "
        "aggrieved person shared household respondent"
    ).strip()


def _summarize_facts_for_template_query(facts: dict[str, Any]) -> str:
    """Create a short text summary of case facts for template similarity search.

    Args:
        facts: ChronologyOfEvents dictionary.

    Returns:
        Short text summary (100-200 words) suitable for embedding.
    """
    parts = []

    incidents = facts.get("incidents", [])
    if incidents:
        types = list({inc.get("incident_type", "") for inc in incidents})
        parts.append(f"Incidents: {', '.join(types)}")

    reliefs = facts.get("reliefs_sought", [])
    if reliefs:
        parts.append(f"Reliefs: {', '.join(reliefs)}")

    relationship = facts.get("relationship_details", {})
    if relationship.get("children"):
        parts.append(f"Children: {len(relationship['children'])}")

    return " | ".join(parts) if parts else "DV petition domestic violence"


def _format_templates_as_shots(
    templates: list[dict[str, Any]],
) -> str:
    """Format retrieved templates as few-shot examples for the generation prompt.

    Includes only the first 2000 chars of each template to stay within context limits.

    Args:
        templates: List of template dicts from template_service.

    Returns:
        Formatted string of style examples.
    """
    shots = []
    for i, template in enumerate(templates, 1):
        excerpt = template["content"][:2000]
        if len(template["content"]) > 2000:
            excerpt += "\n... [excerpt — full template available]"
        shots.append(f"[STYLE EXAMPLE {i} — similarity: {template['similarity']:.2f}]\n{excerpt}")

    return "\n\n---\n\n".join(shots)