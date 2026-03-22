"""Document text extraction service — converts uploaded files to plain text.

Supports:
    - PDF (typed/digital)   → pypdf direct text extraction
    - DOCX (Word document)  → python-docx paragraph extraction
    - Images (JPG/PNG/TIFF) → pytesseract OCR (Marathi + Hindi + English)

Used for the "Upload Written Statement" flow where a client submits a
handwritten or typed document instead of giving a voice recording.
"""

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
    # Count frequency of the most common word
    from collections import Counter
    top_word, top_count = Counter(words).most_common(1)[0]
    # If >60% of words are the same token it's watermark noise
    return (top_count / len(words)) > 0.6


_OCR_RENDER_DPI = 1.5        # render scale (1.5× ≈ 115 DPI — fast + readable)


def _pdf_ocr_fallback(
    file_bytes: bytes,
    on_page_done: "Callable[[int, int], None] | None" = None,
) -> str:
    """Render each PDF page as an image and run pytesseract OCR.

    No page or size limits — all pages are processed. Progress is reported
    via the optional `on_page_done(current, total)` callback so callers can
    update a Redis job record in real time.

    Args:
        file_bytes: Raw PDF bytes.
        on_page_done: Optional callback(current_page, total_pages) called
                      after each page finishes OCR.
    """
    try:
        import fitz  # PyMuPDF  # type: ignore[import]
        import pytesseract  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        file_mb = len(file_bytes) / (1024 * 1024)

        logger.info(
            "pdf_ocr_start",
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
        raise DocumentExtractionError(f"PDF OCR failed: {exc}") from exc


def _extract_pdf(
    file_bytes: bytes,
    on_page_done: "Callable[[int, int], None] | None" = None,
) -> str:
    """Extract text from a PDF — tries text layer first, falls back to OCR.

    Strategy:
    1. pypdf text layer extraction (instant, perfect for typed PDFs)
    2. If result is watermark noise (CamScanner etc.) → PyMuPDF render + pytesseract OCR
    """
    try:
        import pypdf  # type: ignore[import]
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text.strip())
        text_layer = "\n\n".join(pages_text)
    except Exception as exc:
        raise DocumentExtractionError(f"PDF read failed: {exc}") from exc

    # If text layer is empty or watermark noise → use image OCR
    if not text_layer.strip() or _is_watermark_noise(text_layer):
        logger.info("pdf_text_layer_noise_detected_falling_back_to_ocr")
        return _pdf_ocr_fallback(file_bytes, on_page_done=on_page_done)

    return text_layer


def _extract_docx(file_bytes: bytes) -> str:
    """Extract text from a Word .docx document using python-docx."""
    try:
        import docx  # type: ignore[import]
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as exc:
        raise DocumentExtractionError(f"DOCX extraction failed: {exc}") from exc


def _extract_image(file_bytes: bytes, filename: str) -> str:
    """Extract text from an image (handwritten/printed) using pytesseract OCR.

    Uses Marathi + Hindi + English language packs for best results on
    Maharashtra court documents.
    """
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
            # Try Marathi + Hindi + English; fall back to English only if langs missing
            try:
                text = pytesseract.image_to_string(
                    image,
                    lang="mar+hin+eng",
                    config="--psm 6",  # Assume uniform block of text
                )
            except pytesseract.pytesseract.TesseractError:
                # Language packs not available — fall back to English only
                logger.warning("tesseract_lang_fallback", filename=filename)
                text = pytesseract.image_to_string(image, lang="eng", config="--psm 6")

            return text.strip()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as exc:
        raise DocumentExtractionError(f"Image OCR failed: {exc}") from exc


def extract_text_from_document(
    file_bytes: bytes,
    mime_type: str,
    filename: str,
    on_page_done: "Callable[[int, int], None] | None" = None,
) -> tuple[str, ExtractionMethod, int]:
    """Extract plain text from an uploaded document file.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        mime_type: MIME type reported by the browser.
        filename: Original filename (used for extension detection).

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

    logger.info(
        "document_extraction_start",
        filename=filename,
        mime_type=mime_type,
        method=method,
        size_kb=round(file_size_kb, 1),
    )

    if method == "pdf":
        text = _extract_pdf(file_bytes, on_page_done=on_page_done)
    elif method == "docx":
        text = _extract_docx(file_bytes)
    else:
        text = _extract_image(file_bytes, filename)

    if not text or not text.strip():
        if method == "pdf":
            raise DocumentExtractionError(
                "No text found in PDF. This may be a scanned/image PDF. "
                "Please save as DOCX or upload as an image file instead."
            )
        raise DocumentExtractionError(
            f"No text could be extracted from the {method.upper()} file. "
            "Please check the file is not corrupted."
        )

    text = text.strip()
    char_count = len(text)

    logger.info(
        "document_extraction_complete",
        filename=filename,
        method=method,
        char_count=char_count,
    )

    return text, method, char_count
