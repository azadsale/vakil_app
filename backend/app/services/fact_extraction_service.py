"""Fact extraction service — converts raw transcript to structured Chronology of Events.

Uses LLM (GPT-4o) with structured output to extract case facts from client statement.
Missing information is replaced with [MISSING: FIELD_NAME] placeholders — never guessed.

Anti-hallucination rules (enforced in prompt):
    1. Only extract facts explicitly stated in the transcript.
    2. Insert [MISSING: <field>] for any required field not mentioned.
    3. Do NOT infer dates, names, or addresses from context.
    4. Do NOT add legal conclusions (that's the draft generator's job).
"""

import hashlib
import json
from datetime import datetime
from typing import Any

from app.config import get_settings
from app.services.llm_service import LLMError, call_llm
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Chronology of Events — JSON Schema
# This is the canonical structure for DV petition facts
# ---------------------------------------------------------------------------
CHRONOLOGY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "petitioner": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "string"},
                "address": {"type": "string"},
                "occupation": {"type": "string"},
            },
            "required": ["name", "age", "address"],
        },
        "respondent": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "string"},
                "address": {"type": "string"},
                "relationship_to_petitioner": {"type": "string"},
                "occupation": {"type": "string"},
            },
            "required": ["name", "relationship_to_petitioner"],
        },
        "shared_household": {
            "type": "object",
            "properties": {
                "address": {"type": "string"},
                "ownership": {"type": "string", "enum": ["owned_by_respondent", "rented", "joint", "unknown"]},
                "duration_of_residence": {"type": "string"},
            },
        },
        "relationship_details": {
            "type": "object",
            "properties": {
                "date_of_marriage": {"type": "string"},
                "place_of_marriage": {"type": "string"},
                "marriage_type": {"type": "string", "enum": ["registered", "religious", "live_in", "unknown"]},
                "children": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "string"},
                            "currently_with": {"type": "string"},
                        },
                    },
                },
            },
        },
        "incidents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "incident_date": {"type": "string"},
                    "incident_type": {
                        "type": "string",
                        "enum": ["physical", "sexual", "emotional", "verbal", "economic", "multiple"],
                    },
                    "description": {"type": "string"},
                    "witnesses": {"type": "array", "items": {"type": "string"}},
                    "injuries_reported": {"type": "boolean"},
                    "police_complaint_filed": {"type": "boolean"},
                    "fir_number": {"type": "string"},
                },
                "required": ["incident_type", "description"],
            },
        },
        "reliefs_sought": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "protection_order",     # Sec 18
                    "residence_order",      # Sec 19
                    "monetary_relief",      # Sec 20
                    "custody_order",        # Sec 21
                    "compensation_order",   # Sec 22
                ],
            },
        },
        "maintenance_details": {
            "type": "object",
            "properties": {
                "monthly_amount_requested": {"type": "string"},
                "respondent_income": {"type": "string"},
                "petitioner_income": {"type": "string"},
            },
        },
        "previous_legal_proceedings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "court": {"type": "string"},
                    "case_number": {"type": "string"},
                    "nature": {"type": "string"},
                    "status": {"type": "string"},
                },
            },
        },
        "additional_facts": {"type": "string"},
    },
    "required": ["petitioner", "respondent", "incidents", "reliefs_sought"],
}

# ---------------------------------------------------------------------------
# System Prompt — Fact Extraction
# ---------------------------------------------------------------------------
FACT_EXTRACTION_SYSTEM_PROMPT = """You are a legal fact extraction assistant for an Indian advocate specializing in Domestic Violence cases under the Protection of Women from Domestic Violence Act, 2005.

Your ONLY job is to extract factual information from the client's statement transcript and organize it into a structured JSON object.

STRICT RULES (non-negotiable):
1. NEVER infer, guess, or hallucinate any fact not explicitly stated in the transcript.
2. For any required field where information is missing, use the exact string "[MISSING: <FIELD_NAME>]" as the value. Example: "[MISSING: DATE_OF_MARRIAGE]"
3. Do NOT add legal conclusions, interpretations, or advice.
4. Do NOT rephrase or summarize incident descriptions — extract verbatim or near-verbatim.
5. Dates must be in ISO format (YYYY-MM-DD) if the exact date is stated. Use "[MISSING: DATE]" if only approximate (e.g., "last year").
6. Names and addresses are PII — extract exactly as stated.
7. If the client mentions police complaints, extract FIR numbers exactly as stated.
8. For reliefs_sought: infer ONLY from explicit requests by the client (e.g., "I want to stay in the house" → residence_order, "I need money for children" → monetary_relief).

OUTPUT: Return ONLY valid JSON. No commentary, no markdown, no explanation."""

FACT_EXTRACTION_USER_PROMPT_TEMPLATE = """Extract facts from the following client statement transcript and return structured JSON.

Client Statement Transcript:
---
{transcript}
---

Return a JSON object following the schema. Use "[MISSING: FIELD_NAME]" for any required information not present in the transcript."""


class FactExtractionError(Exception):
    """Raised when fact extraction fails."""


async def extract_facts_from_transcript(
    transcript: str,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Extract structured Chronology of Events from a client statement transcript.

    Args:
        transcript: Raw or cleaned transcript from Sarvam Saaras v3.
        case_id: Optional case UUID for logging correlation (NOT the transcript content).

    Returns:
        Dictionary conforming to CHRONOLOGY_SCHEMA with [MISSING] placeholders
        for any fields not explicitly mentioned in the transcript.

    Raises:
        FactExtractionError: If LLM returns invalid JSON or extraction fails.
        ValueError: If transcript is empty.
    """
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty for fact extraction")

    # Hash transcript for dedup/caching (not the content itself)
    transcript_hash = hashlib.sha256(transcript.encode()).hexdigest()[:16]

    logger.info(
        "fact_extraction_start",
        case_id=case_id,
        transcript_hash=transcript_hash,
        transcript_char_count=len(transcript),
    )

    user_prompt = FACT_EXTRACTION_USER_PROMPT_TEMPLATE.format(transcript=transcript)

    try:
        raw_output = await call_llm(
            system_prompt=FACT_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=4096,
            json_mode=True,
        )
    except LLMError as exc:
        raise FactExtractionError(f"LLM call failed: {exc}") from exc
    except Exception as exc:
        logger.error("fact_extraction_llm_error", case_id=case_id, error=str(exc))
        raise FactExtractionError(f"LLM call failed: {exc}") from exc

    # Empty response usually means Gemini's safety filter silently blocked the output.
    # Retry once with json_mode=False and extract JSON from the text response.
    if not raw_output or not raw_output.strip():
        logger.warning(
            "fact_extraction_empty_response_retrying",
            case_id=case_id,
            note="Gemini returned empty — retrying with json_mode=False",
        )
        try:
            raw_output = await call_llm(
                system_prompt=FACT_EXTRACTION_SYSTEM_PROMPT
                    + "\n\nIMPORTANT: Wrap your JSON in ```json ... ``` markers.",
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=4096,
                json_mode=False,
            )
        except Exception as exc2:
            raise FactExtractionError(f"LLM retry also failed: {exc2}") from exc2

    # Strip markdown fences if present (```json ... ```)
    raw_output = raw_output.strip()
    if raw_output.startswith("```"):
        # Remove opening fence
        raw_output = raw_output.split("\n", 1)[-1] if "\n" in raw_output else raw_output[3:]
        # Remove closing fence
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]
        raw_output = raw_output.strip()

    try:
        facts: dict[str, Any] = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        logger.error(
            "fact_extraction_json_parse_error",
            case_id=case_id,
            error=str(exc),
            raw_preview=raw_output[:200],
        )
        raise FactExtractionError(
            f"LLM returned invalid JSON: {exc}"
        ) from exc

    # Count missing fields for quality metrics (log count only, not values)
    missing_count = _count_missing_placeholders(facts)
    logger.info(
        "fact_extraction_complete",
        case_id=case_id,
        transcript_hash=transcript_hash,
        missing_fields_count=missing_count,
        incidents_count=len(facts.get("incidents", [])),
        reliefs_count=len(facts.get("reliefs_sought", [])),
    )

    # Inject metadata
    facts["_metadata"] = {
        "extracted_at": datetime.utcnow().isoformat(),
        "model": settings.groq_llm_model,
        "transcript_hash": transcript_hash,
        "missing_fields_count": missing_count,
    }

    return facts


def _count_missing_placeholders(obj: Any, count: int = 0) -> int:
    """Recursively count [MISSING: ...] placeholders in the facts dict.

    Args:
        obj: Object to scan (dict, list, or string).
        count: Running count.

    Returns:
        Total count of missing placeholders.
    """
    if isinstance(obj, str) and obj.startswith("[MISSING:"):
        return count + 1
    if isinstance(obj, dict):
        for v in obj.values():
            count = _count_missing_placeholders(v, count)
    elif isinstance(obj, list):
        for item in obj:
            count = _count_missing_placeholders(item, count)
    return count


def get_missing_fields(facts: dict[str, Any]) -> list[str]:
    """Extract list of field names that have [MISSING] placeholders.

    Used to generate a checklist for the lawyer to fill before drafting.

    Args:
        facts: ChronologyOfEvents dictionary.

    Returns:
        List of field path strings like ["petitioner.age", "incidents[0].incident_date"].
    """
    missing: list[str] = []
    _collect_missing(facts, "", missing)
    return missing


def _collect_missing(obj: Any, path: str, missing: list[str]) -> None:
    """Recursively collect paths to [MISSING] values."""
    if isinstance(obj, str) and obj.startswith("[MISSING:"):
        missing.append(path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            _collect_missing(value, new_path, missing)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _collect_missing(item, f"{path}[{i}]", missing)