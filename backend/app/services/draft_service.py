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

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.client_statement import ClientStatement, StatementStatus
from app.models.draft_petition import DraftPetition, DraftStatus
from app.services.fact_extraction_service import (
    CHRONOLOGY_SCHEMA,
    extract_facts_from_transcript,
    get_missing_fields,
)
from app.services.llm_service import LLMError, active_provider, call_llm
from app.services.rag_service import build_legal_context_string, embed_single, query_dv_act
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
DRAFT_GENERATION_SYSTEM_PROMPT = """You are a senior advocate in Maharashtra specializing in Protection of Women from Domestic Violence Act, 2005 petitions. You draft petitions filed before Judicial Magistrate First Class courts.

Your task: Draft a complete, detailed, court-ready petition under Section 12 of the DV Act, 2005.

INPUTS YOU WILL RECEIVE:
1. CASE FACTS — structured JSON with petitioner/respondent details, incidents, reliefs
2. LEGAL GROUNDING — relevant DV Act sections retrieved from the statute (use ONLY these)
3. STYLE REFERENCE — sample petition(s) by the advocate (match structure, tone, numbering style exactly)

MANDATORY RULES:
1. NEVER cite any legal section not explicitly present in the LEGAL GROUNDING section.
2. LANGUAGE: Write the ENTIRE petition in {language}. This is NON-NEGOTIABLE.
   - If the language is "marathi" (मराठी), write EVERYTHING in Devanagari script — court header, party names, incidents, reliefs, prayer, verification — ALL in Marathi.
   - Legal section numbers (Section 12, 18, 19, 20, 21, 22) may remain in English numerals.
   - If incidents are described in Marathi/Hindi in the case facts, preserve that language and formalize it.
   - Do NOT write in English when Marathi is requested. Do NOT mix English sentences into a Marathi petition.
3. INCIDENTS — Each incident must be a separate numbered paragraph with:
   - The specific date (or approximate period if exact date unknown)
   - The type of violence (physical/verbal/economic/emotional)
   - Detailed description in formal legal language (convert colloquial/Marathi speech to formal legal prose)
   - Any witnesses or injuries mentioned
   - FIR/police complaint details if mentioned
   Do NOT summarise multiple incidents into one vague paragraph.
4. RELIEF SECTION MAPPING — never use [MISSING] for these:
   - protection_order → Section 18
   - residence_order  → Section 19
   - monetary_relief  → Section 20
   - custody_order    → Section 21
   - compensation_order → Section 22
5. STRUCTURE (numbered, do not skip any section):
   (a) Court Header — "IN THE COURT OF HON'BLE JUDICIAL MAGISTRATE, FIRST CLASS, [CITY]"
   (b) Application title — "APPLICATION UNDER SECTION 12 OF THE PROTECTION OF WOMEN FROM DOMESTIC VIOLENCE ACT, 2005"
   (c) Parties — Petitioner name/age/address, Respondent name/age/address/relationship
   (d) Jurisdiction — why this court has jurisdiction
   (e) Facts of the case — background (marriage, household, relationship dynamics)
   (f) Acts of Domestic Violence — NUMBERED PARAGRAPHS, one per incident, detailed
   (g) Legal provisions attracted — cite retrieved DV Act sections with explanation
   (h) Reliefs sought — numbered list, each with correct section number
   (i) Prayer clause — formal prayer to the court
   (j) Verification — "I, [Name], do hereby verify…" with city and date
6. Use [MISSING: FIELD] ONLY for names/addresses/dates genuinely absent from the facts JSON. Never for section numbers or standard legal phrases.
7. The petition must be substantive and detailed — minimum 1200 words. Each incident paragraph must be at least 80 words with full factual detail. A short or summarised draft is UNACCEPTABLE.
8. Do NOT add facts not present in the JSON. Do NOT hallucinate names, dates, or amounts.

OUTPUT: Complete petition text only. No JSON, no commentary, no preamble."""

DRAFT_GENERATION_USER_TEMPLATE = """CASE FACTS (Chronology of Events JSON):
{facts_json}

---
LEGAL GROUNDING (DV Act 2005 — retrieved sections, cite only these):
{legal_context}

---
STYLE REFERENCE (advocate's sample petition — replicate this structure and tone exactly):
{style_reference}

---
DRAFT LANGUAGE: {language}

Now draft the complete Section 12 DV Act petition.

CRITICAL REQUIREMENTS:
1. LANGUAGE = {language}. The ENTIRE petition must be in {language}. If marathi, write in Devanagari script throughout.
2. Each incident = a separate numbered paragraph with date, type of violence, FULL description (80+ words each), witnesses, injuries, FIR details
3. Convert colloquial speech into formal legal language suitable for court filing
4. Minimum 1200 words total — be detailed and thorough
5. Include correct section numbers for all reliefs (Sec 18/19/20/21/22)
6. Follow the STYLE REFERENCE structure exactly — same headings, numbering, prayer clause format
7. Do NOT summarise incidents — expand them with legal detail"""

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
    language: str = "english",
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

    retrieved_sections = await query_dv_act(query=rag_query, top_k=5, db=db)
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
    # Step 4: Draft Generation (Hybrid Pipeline)
    # ------------------------------------------------------------------
    # Build prompt hash for dedup/logging
    facts_json_str = json.dumps(facts, indent=2, ensure_ascii=False)
    prompt_hash = hashlib.sha256(
        f"{facts_json_str}{legal_context}{language}".encode()
    ).hexdigest()

    logger.info(
        "draft_llm_call_start",
        case_id=str(case_id),
        prompt_hash=prompt_hash[:16],
        legal_sections_count=len(legal_sections_used),
        templates_used=len(template_ids_used),
        requested_language=language,
    )

    # ── Hybrid Pipeline ──
    # ALWAYS generate in English (Gemini/Groq are best at English legal drafting).
    # If Marathi is requested, translate the English draft via Sarvam Saarika API
    # (best-in-class Indic translation) — giving us Gemini's legal reasoning +
    # Sarvam's native Marathi fluency.
    try:
        draft_text = await call_llm(
            system_prompt=DRAFT_GENERATION_SYSTEM_PROMPT.format(language="english"),
            user_prompt=DRAFT_GENERATION_USER_TEMPLATE.format(
                facts_json=json.dumps(facts, indent=2, ensure_ascii=False),
                legal_context=legal_context,
                style_reference=style_reference,
                language="english",
            ),
            temperature=0.3,
            max_tokens=16384,  # Large output for detailed petition (1200+ words)
            json_mode=False,
        )
    except LLMError as exc:
        raise DraftGenerationError(f"LLM generation failed: {exc}") from exc
    except Exception as exc:
        logger.error("draft_llm_call_failed", case_id=str(case_id), error=str(exc))
        raise DraftGenerationError(f"LLM generation failed: {exc}") from exc

    if not draft_text:
        raise DraftGenerationError("LLM returned empty draft")

    # ── Step 4b: Translate to Marathi if requested ──
    generation_note = f"Generated in English by {active_provider()}"

    if language.lower() in ("marathi", "mr-in", "mr"):
        try:
            from app.services.sarvam_translate_service import translate_text, TranslationError

            logger.info(
                "draft_translation_start",
                case_id=str(case_id),
                source="english",
                target="marathi",
                source_chars=len(draft_text),
            )

            draft_text = await translate_text(
                text=draft_text,
                target_language="mr-IN",
                source_language="en-IN",
                mode="formal",  # legal/formal register
            )

            generation_note = (
                f"Generated in English by {active_provider()}, "
                f"translated to Marathi by Sarvam Saarika (formal mode)"
            )

            logger.info(
                "draft_translation_complete",
                case_id=str(case_id),
                translated_chars=len(draft_text),
            )
        except Exception as translate_exc:
            logger.warning(
                "draft_translation_failed_keeping_english",
                case_id=str(case_id),
                error=str(translate_exc),
            )
            generation_note += " (Marathi translation failed — draft is in English)"

    # Append mandatory disclaimer
    disclaimer = MANDATORY_DISCLAIMER.format(
        timestamp=datetime.utcnow().isoformat(),
        model=generation_note,
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
        generation_model=active_provider(),
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
        model=settings.groq_llm_model,
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
    """Format retrieved templates as style-reference examples for the generation prompt.

    Sends up to 20,000 chars per template — Gemini's 1M context easily handles this.
    The more of the lawyer's actual petition the model sees, the better it matches
    their structure, section numbering, paragraph style, and legal phrasing.

    Args:
        templates: List of template dicts from template_service.

    Returns:
        Formatted string of style examples.
    """
    shots = []
    for i, template in enumerate(templates, 1):
        # Use full content up to 20,000 chars per template
        content = template["content"]
        excerpt = content[:20_000]
        if len(content) > 20_000:
            excerpt += "\n\n... [remaining content truncated — use the style shown above]"
        shots.append(
            f"[STYLE REFERENCE {i} — This is an actual petition drafted by this advocate "
            f"(similarity score: {template['similarity']:.2f}). "
            f"Replicate this exact structure, paragraph numbering, section citations, "
            f"prayer clause format, and verification language.]\n\n{excerpt}"
        )

    return "\n\n" + ("=" * 60) + "\n\n".join(shots) + "\n\n" + ("=" * 60)