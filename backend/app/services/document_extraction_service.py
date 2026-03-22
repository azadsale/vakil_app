"""Document text extraction service — converts uploaded files to plain text.

Supports:
    - PDF (typed/digital)        → pypdf direct text extraction
    - PDF (scanned/handwritten)  → Gemini Vision OCR (primary) or Tesseract (fallback)
    - DOCX (Word document)       → python-docx paragraph extraction
    - Images (JPG/PNG/TIFF)      → Gemini Vision OCR (primary) or Tesseract (fallback)

OCR Priority:
    1. Google Gemini Flash-Lite Vision — excellent Marathi/Hindi handwriting, FREE
       Model: gemini-2.0-flash-lite (1,500 req/day free — much better than flash's 50/day)
    2. pytesseract (Tesseract)   — open source fallback if Gemini key not set

Used for the "Upload Written Statement" flow where a client submits a
handwritten or typed document instead of giving a voice recording.
"""

import asyncio
import io
import tempfile
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from app.utils.logging import get_logger

logger = get_logger(__name__)

# Supported MIME types → extraction method
SUPPORTED_DOC_MIME_TYPES = {
    "application/pdf",
    "application/octet-stream",           # generic fallback for PDFs
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/msword",                 # .doc (older Word)
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "image/bmp",
}

ExtractionMethod = Literal["pdf", "docx", "image"]

# Prompt sent to Gemini Vision for each scanned page / image
_GEMINI_OCR_PROMPT = """This is a scanned page from a handwritten statement written by a client \
in a Domestic Violence court case in Maharashtra, India.

Your task: Extract ALL text from this image exactly as written.

Rules:
- The text may be in Marathi (Devanagari script), Hindi, or English — or mixed.
- Preserve all names, dates, addresses, amounts of money, and incident descriptions exactly.
- If a word or character is unclear or ambiguous, make your best guess based on context.
- Do NOT add any commentary, translation, explanation, or summary.
- Output ONLY the extracted text, preserving paragraph and line breaks.
- Do NOT skip any part of the page even if it seems repetitive or difficult to read."""


class DocumentExtractionError(Exception):
    """Raised when text extraction from a document fails."""


def _detect_method(mime_type: str, filename: str) -> ExtractionMethod:
    """Determine extraction method from MIME type and filename extension."""
    mime = mime_type.lower().split(";")[0].strip()
    ext = Path(filename).suffix.lower()

    if mime == "application/pdf" or ext == ".pdf":
        return "pdf"
    if mime in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ) or ext in (".docx", ".doc"):
        return "docx"
    if mime.startswith("image/") or ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
        return "image"

    # Last resort: try PDF
    return "pdf"


def _is_watermark_noise(text: str) -> bool:
    """Detect if extracted PDF text is just watermark noise (e.g. CamScanner repeated).

    Returns True if the text is dominated by a single repeated word — meaning
    the PDF has no real text layer and needs image OCR instead.
    """
    words = text.split()
    if not words:
        return True
    from collections import Counter
    top_word, top_count = Counter(words).most_common(1)[0]
    # If >60% of words are the same token it's watermark noise
    return (top_count / len(words)) > 0.6


# ---------------------------------------------------------------------------
# PDF text layer extraction (sync, instant for typed PDFs)
# ---------------------------------------------------------------------------

def _extract_pdf_text_layer(file_bytes: bytes) -> str:
    """Extract text layer from PDF using pypdf. Returns empty string if none."""
    try:
        import pypdf  # type: ignore[import]
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text.strip())
        return "\n\n".join(pages_text)
    except Exception as exc:
        raise DocumentExtractionError(f"PDF read failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Gemini Vision OCR (async) — PRIMARY for scanned / handwritten content
# ---------------------------------------------------------------------------

# Model is selected dynamically by model_router based on remaining daily quota.
# Fallback if router fails — use the model with the largest free quota.
_GEMINI_OCR_MODEL_FALLBACK = "gemini-2.0-flash"
_GEMINI_OCR_MAX_RETRIES = 3
_GEMINI_OCR_RETRY_DELAY = 40  # seconds to wait after a 429 rate-limit


async def _gemini_ocr_image_bytes(image_png_bytes: bytes, page_label: str = "") -> str:
    """Send a single image (PNG bytes) to Gemini Flash-Lite Vision and extract text.

    Uses gemini-2.0-flash-lite (1,500 req/day free) instead of flash (50/day).
    Retries up to 3× on 429 rate-limit errors with a 40-second wait.

    Args:
        image_png_bytes: Raw PNG bytes of the page / image.
        page_label: Human-readable label for logging (e.g. "Page 3").

    Returns:
        Extracted text string.
    """
    from google import genai  # type: ignore[import]
    from google.genai import types  # type: ignore[import]
    from google.genai import errors as genai_errors  # type: ignore[import]
    from app.config import get_settings
    settings = get_settings()

    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    contents = [
        types.Part.from_bytes(data=image_png_bytes, mime_type="image/png"),
        _GEMINI_OCR_PROMPT,
    ]

    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]

    config = types.GenerateContentConfig(
        safety_settings=safety_settings,
        temperature=0.1,
        max_output_tokens=4096,
    )

    last_exc: Exception | None = None
    for attempt in range(1, _GEMINI_OCR_MAX_RETRIES + 1):
        try:
            logger.info("gemini_vision_ocr_page", page=page_label, attempt=attempt, model=_GEMINI_OCR_MODEL_FALLBACK)
            response = await client.aio.models.generate_content(
                model=_GEMINI_OCR_MODEL_FALLBACK,
                contents=contents,
                config=config,
            )
            return response.text.strip() if response.text else ""

        except genai_errors.ClientError as exc:
            if exc.code == 429:
                logger.warning(
                    "gemini_ocr_rate_limited",
                    page=page_label,
                    attempt=attempt,
                    retry_in_seconds=_GEMINI_OCR_RETRY_DELAY,
                )
                if attempt < _GEMINI_OCR_MAX_RETRIES:
                    await asyncio.sleep(_GEMINI_OCR_RETRY_DELAY)
                    last_exc = exc
                    continue
            last_exc = exc
            break
        except Exception as exc:
            last_exc = exc
            break

    logger.warning("gemini_vision_ocr_failed", page=page_label, error=str(last_exc))
    raise DocumentExtractionError(f"Gemini Vision OCR failed for {page_label}: {last_exc}") from last_exc


_GEMINI_OCR_BATCH_SIZE = 5   # pages per API call — reduces 41 calls to ~9 calls


async def _gemini_ocr_batch(
    page_pngs: "list[bytes]",
    page_start: int,
    batch_label: str,
    model: str | None = None,
) -> "list[str]":
    """Send a batch of page images to Gemini in ONE API call.

    Batching reduces API calls from N pages to ceil(N/5), staying well within
    rate limits (1,500 RPD for flash-lite) while processing large PDFs quickly.

    Args:
        page_pngs: List of PNG bytes for consecutive pages.
        page_start: 1-based index of the first page in the batch.
        batch_label: Human-readable label for logging.

    Returns:
        List of extracted text strings, one per page.
    """
    from google import genai  # type: ignore[import]
    from google.genai import types  # type: ignore[import]
    from google.genai import errors as genai_errors  # type: ignore[import]
    from app.config import get_settings
    settings = get_settings()

    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    # Build multi-image prompt
    page_labels = [f"Page {page_start + i}" for i in range(len(page_pngs))]

    # Build contents: alternate page label + image, then final instruction
    contents: list = []
    for png_bytes, lbl in zip(page_pngs, page_labels):
        contents.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))

    # Single clear instruction at the end
    contents.append(
        f"These are {len(page_pngs)} scanned pages from a handwritten statement "
        "written by a client in a Domestic Violence case in Maharashtra, India.\n\n"
        "Extract ALL text from every page. The handwriting may be in Marathi "
        "(Devanagari script), Hindi, or English — or a mix.\n\n"
        "IMPORTANT RULES:\n"
        "1. Copy every word exactly as written — names, dates, addresses, incident "
        "descriptions, amounts.\n"
        "2. If a character is unclear, write your best guess.\n"
        "3. Do NOT translate, summarise, explain, or add any commentary.\n"
        "4. Separate each page's text with a line: === " + page_labels[0] + " === "
        "(use the same format for every page).\n"
        "5. If a page is blank or completely unreadable, write: === " + page_labels[0] + " === [blank]\n\n"
        "Begin extraction now, starting with === " + page_labels[0] + " ==="
    )

    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
    ]

    config = types.GenerateContentConfig(
        safety_settings=safety_settings,
        temperature=0.1,
        max_output_tokens=8192,
    )

    use_model = model or _GEMINI_OCR_MODEL_FALLBACK

    last_exc: Exception | None = None
    for attempt in range(1, _GEMINI_OCR_MAX_RETRIES + 1):
        try:
            logger.info(
                "gemini_vision_ocr_batch",
                batch=batch_label,
                pages=len(page_pngs),
                attempt=attempt,
                model=use_model,
            )
            response = await client.aio.models.generate_content(
                model=use_model,
                contents=contents,
                config=config,
            )
            raw = response.text or ""
            logger.info(
                "gemini_vision_ocr_batch_raw_preview",
                batch=batch_label,
                preview=raw[:200],
                total_chars=len(raw),
            )

            # Try to parse === PAGE X === sections
            result_texts: list[str] = []
            found_any_marker = False

            for lbl in page_labels:
                # Try several marker formats Gemini might use
                candidates = [
                    f"=== {lbl} ===",
                    f"**{lbl}**",
                    f"## {lbl}",
                    f"--- {lbl} ---",
                    lbl + ":",
                ]
                start = -1
                matched_marker = ""
                for candidate in candidates:
                    pos = raw.find(candidate)
                    if pos != -1:
                        start = pos + len(candidate)
                        matched_marker = candidate
                        found_any_marker = True
                        break

                if start == -1:
                    result_texts.append("")
                    continue

                # Find the next marker of any known format
                next_positions = []
                for next_lbl in page_labels:
                    if next_lbl == lbl:
                        continue
                    for candidate in [
                        f"=== {next_lbl} ===",
                        f"**{next_lbl}**",
                        f"## {next_lbl}",
                        f"--- {next_lbl} ---",
                        next_lbl + ":",
                    ]:
                        pos = raw.find(candidate, start)
                        if pos != -1:
                            next_positions.append(pos)
                            break

                end = min(next_positions) if next_positions else len(raw)
                chunk = raw[start:end].strip()
                if chunk.lower() in ("[blank]", "blank", ""):
                    chunk = ""
                result_texts.append(chunk)

            # Fallback: if NO markers were found at all, treat the entire response
            # as the text for all pages in this batch (better than empty)
            if not found_any_marker and raw.strip():
                logger.warning(
                    "gemini_ocr_no_markers_using_full_response",
                    batch=batch_label,
                    chars=len(raw),
                )
                # Divide text evenly across pages as best-effort
                chunk_size = max(1, len(raw) // len(page_labels))
                result_texts = []
                for i in range(len(page_labels)):
                    s = i * chunk_size
                    e = s + chunk_size if i < len(page_labels) - 1 else len(raw)
                    result_texts.append(raw[s:e].strip())

            return result_texts

        except genai_errors.ClientError as exc:
            if exc.code == 429:
                # ── Mark this model as exhausted and try next tier ──
                try:
                    from app.services.model_router import select_model, QuotaExhaustedError
                    import redis.asyncio as aioredis
                    from app.config import get_settings as _gs
                    _s = _gs()
                    _r = aioredis.from_url(_s.redis_url, decode_responses=True)
                    import datetime
                    _key = f"gemini_quota:{use_model}:{datetime.date.today().isoformat()}"
                    # Set usage to daily_limit so router skips this model
                    _limits = {"gemini-2.5-flash": 20, "gemini-2.0-flash": 1500, "gemini-2.0-flash-lite": 1500}
                    await _r.setex(_key, 90_000, str(_limits.get(use_model, 9999)))
                    await _r.aclose()
                    logger.warning(
                        "gemini_model_exhausted_switching",
                        exhausted_model=use_model,
                        batch=batch_label,
                    )
                    # Pick the next available model
                    try:
                        new_model = await select_model(
                            task="vision_ocr_fallback",
                            required_requests=1,
                            require_vision=True,
                        )
                        if new_model != use_model:
                            logger.info(
                                "gemini_model_switched",
                                from_model=use_model,
                                to_model=new_model,
                                batch=batch_label,
                            )
                            use_model = new_model
                            # Recreate client is fine — same API key, different model
                            await asyncio.sleep(2)  # brief pause
                            last_exc = exc
                            continue  # retry this batch with new model
                    except QuotaExhaustedError:
                        logger.error("all_models_exhausted_in_batch", batch=batch_label)
                except Exception as router_err:
                    logger.warning("model_switch_failed", error=str(router_err))

                # Fallback: wait and retry same model
                logger.warning(
                    "gemini_ocr_rate_limited_batch",
                    batch=batch_label,
                    attempt=attempt,
                    retry_in_seconds=_GEMINI_OCR_RETRY_DELAY,
                )
                if attempt < _GEMINI_OCR_MAX_RETRIES:
                    await asyncio.sleep(_GEMINI_OCR_RETRY_DELAY)
                    last_exc = exc
                    continue
            last_exc = exc
            break
        except Exception as exc:
            last_exc = exc
            break

    logger.warning("gemini_vision_ocr_batch_failed", batch=batch_label, error=str(last_exc))
    # Return empty strings for all pages so the job continues
    return [""] * len(page_pngs)


async def _pdf_ocr_gemini(
    file_bytes: bytes,
    on_page_done: "Callable[[int, int], None] | None" = None,
) -> str:
    """Render PDF pages in batches and extract text via Gemini Vision.

    Sends _GEMINI_OCR_BATCH_SIZE pages per API call to minimise API calls.
    For a 41-page PDF: 41 individual calls → 9 batched calls.

    Args:
        file_bytes: Raw PDF bytes.
        on_page_done: Optional callback(current_page, total_pages).

    Returns:
        Concatenated text from all pages.
    """
    try:
        import fitz  # PyMuPDF  # type: ignore[import]

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        file_mb = len(file_bytes) / (1024 * 1024)
        n_batches = (total_pages + _GEMINI_OCR_BATCH_SIZE - 1) // _GEMINI_OCR_BATCH_SIZE

        # ── Quota-aware model selection ──
        # Pre-scan: we know the page count, calculate how many API requests
        # we'll need, then pick the best model that has enough daily quota.
        from app.services.model_router import select_model, track_usage, QuotaExhaustedError

        try:
            selected_model = await select_model(
                task="vision_ocr",
                required_requests=n_batches,
                require_vision=True,
            )
        except QuotaExhaustedError:
            logger.warning("all_gemini_models_exhausted_using_fallback")
            selected_model = _GEMINI_OCR_MODEL_FALLBACK

        logger.info(
            "pdf_gemini_vision_ocr_start",
            total_pages=total_pages,
            batch_size=_GEMINI_OCR_BATCH_SIZE,
            n_batches=n_batches,
            file_mb=round(file_mb, 1),
            model=selected_model,
        )

        # Render at 2× (144 DPI) — good quality for Gemini, reasonable file size
        mat = fitz.Matrix(2.0, 2.0)
        pages_text: list[str] = []

        for batch_idx in range(n_batches):
            batch_start = batch_idx * _GEMINI_OCR_BATCH_SIZE          # 0-based
            batch_end   = min(batch_start + _GEMINI_OCR_BATCH_SIZE, total_pages)
            batch_pngs: list[bytes] = []

            for page_num in range(batch_start, batch_end):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                batch_pngs.append(pix.tobytes("png"))

            batch_label = f"Batch {batch_idx + 1}/{n_batches} (pages {batch_start + 1}–{batch_end})"
            batch_texts = await _gemini_ocr_batch(
                page_pngs=batch_pngs,
                page_start=batch_start + 1,   # 1-based
                batch_label=batch_label,
                model=selected_model,
            )

            # Track this batch call against the model's daily quota
            await track_usage(selected_model, count=1)

            for i, text in enumerate(batch_texts):
                page_num = batch_start + i
                if text.strip():
                    pages_text.append(f"[Page {page_num + 1}]\n{text.strip()}")
                if on_page_done:
                    on_page_done(page_num + 1, total_pages)

            # Short pause between batches to respect RPM limits
            if batch_idx < n_batches - 1:
                await asyncio.sleep(3)

        doc.close()
        return "\n\n".join(pages_text)

    except DocumentExtractionError:
        raise
    except Exception as exc:
        raise DocumentExtractionError(f"PDF Gemini Vision OCR failed: {exc}") from exc


async def _image_ocr_gemini(file_bytes: bytes, filename: str) -> str:
    """Extract text from a single image file using Gemini Vision.

    Args:
        file_bytes: Raw image bytes (JPG/PNG/TIFF etc.)
        filename: Original filename (for extension detection).

    Returns:
        Extracted text string.
    """
    try:
        from PIL import Image  # type: ignore[import]
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        png_buf = io.BytesIO()
        img.save(png_buf, format="PNG")
        png_bytes = png_buf.getvalue()
    except Exception as exc:
        raise DocumentExtractionError(f"Image conversion failed: {exc}") from exc

    # Reuse batch function with a single image
    results = await _gemini_ocr_batch([png_bytes], page_start=1, batch_label=filename)
    return results[0] if results else ""


# ---------------------------------------------------------------------------
# Tesseract OCR (sync) — FALLBACK when Gemini key is not set
# ---------------------------------------------------------------------------

# PyMuPDF renders at 72 DPI base. Tesseract needs ≥200 DPI for reliable results.
_OCR_RENDER_DPI = 3.5  # 3.5× ≈ 252 DPI


def _pdf_ocr_tesseract(
    file_bytes: bytes,
    on_page_done: "Callable[[int, int], None] | None" = None,
) -> str:
    """Render each PDF page as image and run pytesseract OCR (fallback).

    Significantly lower accuracy than Gemini Vision for Marathi handwriting,
    but works without any API key.
    """
    try:
        import fitz  # PyMuPDF  # type: ignore[import]
        import pytesseract  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        file_mb = len(file_bytes) / (1024 * 1024)

        logger.info(
            "pdf_tesseract_ocr_start",
            total_pages=total_pages,
            file_mb=round(file_mb, 1),
        )

        pages_text = []
        mat = fitz.Matrix(_OCR_RENDER_DPI, _OCR_RENDER_DPI)

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            try:
                text = pytesseract.image_to_string(
                    img, lang="mar+hin+eng", config="--psm 6",
                )
            except pytesseract.pytesseract.TesseractError:
                text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")

            if text.strip():
                pages_text.append(f"[Page {page_num + 1}]\n{text.strip()}")

            if on_page_done:
                on_page_done(page_num + 1, total_pages)

        doc.close()
        return "\n\n".join(pages_text)

    except Exception as exc:
        raise DocumentExtractionError(f"PDF Tesseract OCR failed: {exc}") from exc


def _image_ocr_tesseract(file_bytes: bytes, filename: str) -> str:
    """Extract text from an image using pytesseract (fallback)."""
    try:
        import pytesseract  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        ext = Path(filename).suffix.lower() or ".png"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            image = Image.open(tmp_path)
            try:
                text = pytesseract.image_to_string(image, lang="mar+hin+eng", config="--psm 6")
            except pytesseract.pytesseract.TesseractError:
                logger.warning("tesseract_lang_fallback", filename=filename)
                text = pytesseract.image_to_string(image, lang="eng", config="--psm 6")
            return text.strip()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as exc:
        raise DocumentExtractionError(f"Image Tesseract OCR failed: {exc}") from exc


# ---------------------------------------------------------------------------
# DOCX extraction (sync)
# ---------------------------------------------------------------------------

def _extract_docx(file_bytes: bytes) -> str:
    """Extract text from a Word .docx document using python-docx."""
    try:
        import docx  # type: ignore[import]
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as exc:
        raise DocumentExtractionError(f"DOCX extraction failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API — async, auto-selects best OCR engine
# ---------------------------------------------------------------------------

def _gemini_key_available() -> bool:
    """Return True if a Gemini API key is configured."""
    try:
        from app.config import get_settings
        return bool(get_settings().gemini_api_key.get_secret_value())
    except Exception:
        return False


async def extract_text_from_document(
    file_bytes: bytes,
    mime_type: str,
    filename: str,
    on_page_done: "Callable[[int, int], None] | None" = None,
) -> "tuple[str, ExtractionMethod, int]":
    """Extract plain text from an uploaded document file.

    OCR engine selection:
    - If GEMINI_API_KEY is set → Gemini Flash Vision (excellent Marathi handwriting)
    - Otherwise → Tesseract OCR (open source, lower accuracy on handwriting)

    Args:
        file_bytes: Raw bytes of the uploaded file.
        mime_type: MIME type reported by the browser.
        filename: Original filename (used for extension detection).
        on_page_done: Optional callback(current_page, total_pages) for progress.

    Returns:
        Tuple of (extracted_text, method_used, char_count).

    Raises:
        DocumentExtractionError: If extraction fails or text is empty.
        ValueError: If file is empty.
    """
    if not file_bytes:
        raise ValueError("Uploaded file is empty")

    method = _detect_method(mime_type, filename)
    file_size_kb = len(file_bytes) / 1024
    use_gemini = _gemini_key_available()

    logger.info(
        "document_extraction_start",
        filename=filename,
        mime_type=mime_type,
        method=method,
        size_kb=round(file_size_kb, 1),
        ocr_engine="gemini_vision" if use_gemini else "tesseract",
    )

    # ── PDF ──────────────────────────────────────────────────────────────────
    if method == "pdf":
        # Try text layer first (instant, perfect for typed PDFs)
        text_layer = await asyncio.to_thread(_extract_pdf_text_layer, file_bytes)

        if text_layer.strip() and not _is_watermark_noise(text_layer):
            # Good typed PDF — no OCR needed
            logger.info("pdf_text_layer_ok", char_count=len(text_layer))
            text = text_layer
        else:
            # Scanned / handwritten PDF → OCR
            logger.info(
                "pdf_scanned_needs_ocr",
                engine="gemini_vision" if use_gemini else "tesseract",
            )
            if use_gemini:
                text = await _pdf_ocr_gemini(file_bytes, on_page_done)
            else:
                # Tesseract is sync — run in thread so we don't block event loop
                text = await asyncio.to_thread(_pdf_ocr_tesseract, file_bytes, on_page_done)

    # ── DOCX ─────────────────────────────────────────────────────────────────
    elif method == "docx":
        text = await asyncio.to_thread(_extract_docx, file_bytes)

    # ── Image ─────────────────────────────────────────────────────────────────
    else:
        if use_gemini:
            text = await _image_ocr_gemini(file_bytes, filename)
        else:
            text = await asyncio.to_thread(_image_ocr_tesseract, file_bytes, filename)

    if not text or not text.strip():
        if method == "pdf":
            raise DocumentExtractionError(
                "No text found in PDF. "
                + ("Please check the image quality — ensure the scan is clear."
                   if use_gemini else
                   "Please set GEMINI_API_KEY for better handwriting recognition, "
                   "or upload as an image file.")
            )
        raise DocumentExtractionError(
            f"No text could be extracted from the {method.upper()} file. "
            "Please check the file is not corrupted or too blurry."
        )

    text = text.strip()
    char_count = len(text)

    logger.info(
        "document_extraction_complete",
        filename=filename,
        method=method,
        char_count=char_count,
        ocr_engine="gemini_vision" if use_gemini else "tesseract",
    )

    return text, method, char_count
