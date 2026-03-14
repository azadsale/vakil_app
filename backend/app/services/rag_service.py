"""RAG service — PDF ingestion and semantic search using fastembed + pgvector.

Architecture (no LlamaIndex — pure fastembed + SQLAlchemy):
    1. PDF ingestion: pypdf text extraction → chunk → embed → INSERT to postgres
    2. Semantic search: embed query → pgvector cosine similarity → return sections

Embeddings: BAAI/bge-small-en-v1.5 via fastembed (ONNX, ~100MB, no API key, 384-dim)
LLM: Groq llama-3.3-70b-versatile — called directly in draft_service.py

Anti-hallucination guarantee:
    Every chunk stored with source_citation. Returned as part of context window.
    LLM is strictly forbidden from citing anything outside the retrieved context.
"""

import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from pypdf import PdfReader
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Fastembed model (lazy-loaded, cached in-process)
# BAAI/bge-small-en-v1.5: 384-dim, ~100MB ONNX, no API key needed
# ---------------------------------------------------------------------------
_embed_model: TextEmbedding | None = None


def get_embed_model() -> TextEmbedding:
    """Lazy-load fastembed model (downloaded once to ~/.cache/fastembed)."""
    global _embed_model
    if _embed_model is None:
        logger.info("fastembed_model_loading", model="BAAI/bge-small-en-v1.5")
        _embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        logger.info("fastembed_model_loaded")
    return _embed_model


def configure_llama_index() -> None:
    """Pre-warm the embedding model at startup.

    Called from app lifespan. Downloads BAAI/bge-small-en-v1.5 if not cached.
    No LLM config here — Groq called directly in draft_service.py.
    """
    get_embed_model()  # pre-warm
    logger.info(
        "embedding_model_ready",
        model="BAAI/bge-small-en-v1.5",
        dim=settings.embedding_dim,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed texts using fastembed.

    Args:
        texts: List of strings to embed.

    Returns:
        List of 384-dim float vectors.
    """
    model = get_embed_model()
    embeddings = list(model.embed(texts))
    return [emb.tolist() for emb in embeddings]


def embed_single(text: str) -> list[float]:
    """Embed a single text string.

    Args:
        text: String to embed.

    Returns:
        384-dim float vector.
    """
    return embed_texts([text])[0]


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks by sentence boundaries.

    Args:
        text: Full document text.
        chunk_size: Target characters per chunk.
        overlap: Overlap characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) > chunk_size and current:
            chunks.append(current.strip())
            # Start new chunk with overlap from end of previous
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + sentence
        else:
            current = (current + " " + sentence).strip()

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 50]  # skip tiny fragments


async def ingest_legal_pdf(
    pdf_path: str | Path,
    document_id: str,
    short_name: str,
    doc_type: str = "dv_act_2005",
    db: AsyncSession | None = None,
) -> int:
    """Ingest a legal statute PDF into pgvector.

    Reads PDF, extracts text page by page, chunks it, embeds with fastembed,
    and INSERTs into the ``legal_statute_chunks`` table.

    Args:
        pdf_path: Local path to the PDF.
        document_id: UUID of the LegalDocument DB record.
        short_name: Short statute name (e.g. "DV Act 2005") — used in citations.
        doc_type: Document type string.
        db: Async DB session (if None, only returns chunk count — use for testing).

    Returns:
        Number of chunks indexed.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("rag_ingest_start", document_id=document_id, short_name=short_name)

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    all_chunks: list[dict[str, Any]] = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue

        chunks = _chunk_text(page_text)
        for chunk_idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "document_id": document_id,
                "page_number": page_num,
                "chunk_index": chunk_idx,
                "content": chunk_text,
                "source_citation": f"{short_name}, Page {page_num}",
            })

    if not all_chunks:
        raise ValueError(f"No text extracted from PDF: {pdf_path}")

    # Batch embed all chunks
    texts = [c["content"] for c in all_chunks]
    logger.info("rag_embedding_start", chunk_count=len(texts))
    embeddings = embed_texts(texts)
    logger.info("rag_embedding_complete", chunk_count=len(embeddings))

    # Insert into pgvector table
    if db is not None:
        # Create table if not exists (LlamaIndex-independent)
        await db.execute(text("""
            CREATE TABLE IF NOT EXISTS legal_statute_chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID NOT NULL,
                page_number INTEGER,
                chunk_index INTEGER,
                content TEXT NOT NULL,
                source_citation VARCHAR(500),
                embedding vector(384),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_legal_statute_chunks_document_id
                ON legal_statute_chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_legal_statute_chunks_embedding
                ON legal_statute_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 50);
        """))

        for chunk, embedding in zip(all_chunks, embeddings):
            await db.execute(
                text("""
                    INSERT INTO legal_statute_chunks
                        (document_id, page_number, chunk_index, content, source_citation, embedding)
                    VALUES
                        (:doc_id, :page, :idx, :content, :citation, CAST(:emb AS vector))
                """),
                {
                    "doc_id": chunk["document_id"],
                    "page": chunk["page_number"],
                    "idx": chunk["chunk_index"],
                    "content": chunk["content"],
                    "citation": chunk["source_citation"],
                    "emb": str(embedding),
                },
            )

        await db.flush()

    logger.info(
        "rag_ingest_complete",
        document_id=document_id,
        chunk_count=len(all_chunks),
        pages_processed=total_pages,
    )

    return len(all_chunks)


async def query_dv_act(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    db: AsyncSession | None = None,
) -> list[dict[str, Any]]:
    """Query indexed DV Act for relevant sections using pgvector cosine similarity.

    Args:
        query: Natural language query (e.g. "protection order for physical violence").
        top_k: Number of relevant sections to retrieve.
        document_id: Filter to specific document UUID (optional).
        db: Async DB session. If None, returns empty list (graceful degradation).

    Returns:
        List of dicts: ``text``, ``source_citation``, ``score``, ``page``.
        Ordered by similarity descending.
    """
    if db is None:
        return []

    query_embedding = embed_single(query)

    filter_clause = "AND document_id = :doc_id" if document_id else ""

    result = await db.execute(
        text(f"""
            SELECT
                content,
                source_citation,
                page_number,
                document_id::text,
                1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM legal_statute_chunks
            WHERE 1=1 {filter_clause}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """),
        {
            "embedding": str(query_embedding),
            "top_k": top_k,
            **({"doc_id": document_id} if document_id else {}),
        },
    )

    rows = result.fetchall()
    results = [
        {
            "text": row.content,
            "source_citation": row.source_citation or "DV Act 2005",
            "score": float(row.similarity),
            "page": row.page_number,
            "document_id": row.document_id,
        }
        for row in rows
    ]

    logger.info("rag_query_complete", results_count=len(results))
    return results


def build_legal_context_string(retrieved_sections: list[dict[str, Any]]) -> str:
    """Format retrieved legal sections into a prompt-ready string.

    Args:
        retrieved_sections: Output from query_dv_act().

    Returns:
        Formatted string with citations ready for LLM context injection.
    """
    if not retrieved_sections:
        return "No specific law found in provided grounding."
    return "\n\n".join(
        f"[SOURCE: {s['source_citation']}]\n{s['text']}"
        for s in retrieved_sections
    )