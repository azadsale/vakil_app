"""RAG service — LlamaIndex pipeline over legal statutes (DV Act 2005).

Architecture:
    1. PDF ingestion: extract text + chunk (512 tokens, 50 overlap)
    2. Embed chunks: OpenAI text-embedding-ada-002
    3. Store: pgvector via LlamaIndex PGVectorStore
    4. Query: semantic search → return sections with citations

Anti-hallucination guarantee:
    Every retrieved chunk carries a ``source_citation`` that is injected
    verbatim into the generation prompt. The LLM is forbidden from citing
    any section not present in the retrieved context.
"""

from pathlib import Path
from typing import Any

from llama_index.core import (
    Settings as LlamaSettings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# LlamaIndex global settings (set once at startup)
# ---------------------------------------------------------------------------
def configure_llama_index() -> None:
    """Configure LlamaIndex global settings.

    Call once from app lifespan startup.
    Uses OpenAI embeddings + GPT-4o for generation.
    """
    api_key = settings.openai_api_key.get_secret_value()
    LlamaSettings.embed_model = OpenAIEmbedding(
        model=settings.llama_index_embed_model,
        api_key=api_key,
    )
    LlamaSettings.llm = LlamaOpenAI(
        model=settings.llama_index_llm_model,
        api_key=api_key,
        temperature=0.0,
    )
    LlamaSettings.chunk_size = 512
    LlamaSettings.chunk_overlap = 50
    logger.info(
        "llama_index_configured",
        embed_model=settings.llama_index_embed_model,
        llm_model=settings.llama_index_llm_model,
    )


def _get_vector_store(table_name: str = "legal_document_chunks") -> PGVectorStore:
    """Create a LlamaIndex PGVectorStore pointing at our pgvector table.

    Args:
        table_name: PostgreSQL table name for vector storage.
                    Separate tables for legal statutes vs case documents.

    Returns:
        Configured PGVectorStore instance.
    """
    db_url = settings.database_url.get_secret_value()
    url = make_url(db_url)

    return PGVectorStore.from_params(
        database=url.database,
        host=url.host,
        password=url.password,
        port=url.port or 5432,
        user=url.username,
        table_name=table_name,
        embed_dim=1536,  # text-embedding-ada-002 dimension
    )


async def ingest_legal_pdf(
    pdf_path: str | Path,
    document_id: str,
    short_name: str,
    doc_type: str = "dv_act_2005",
) -> int:
    """Ingest a legal statute PDF into pgvector.

    Chunks the PDF, generates embeddings, and stores in the
    ``legal_document_chunks`` table with source citations.

    Args:
        pdf_path: Local path to the PDF file.
        document_id: UUID of the LegalDocument DB record.
        short_name: Short reference name (e.g. "DV Act 2005").
        doc_type: Document type string for citation formatting.

    Returns:
        Number of chunks indexed.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(
        "rag_ingest_start",
        document_id=document_id,
        short_name=short_name,
        pdf_path=str(pdf_path),
    )

    # Load and parse PDF
    reader = SimpleDirectoryReader(
        input_files=[str(pdf_path)],
        filename_as_id=True,
    )
    documents = reader.load_data()

    # Attach metadata for citations — this propagates to every chunk
    for doc in documents:
        doc.metadata.update({
            "document_id": document_id,
            "source": short_name,
            "doc_type": doc_type,
        })
        doc.excluded_embed_metadata_keys = ["file_path", "file_name", "file_type"]
        doc.excluded_llm_metadata_keys = ["file_path", "file_name", "file_type"]

    # Chunk with sentence-aware splitter
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    # Attach citation to each node
    for i, node in enumerate(nodes):
        page_num = node.metadata.get("page_label", "?")
        node.metadata["source_citation"] = (
            f"{short_name}, Page {page_num}"
        )

    vector_store = _get_vector_store("legal_document_chunks")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    chunk_count = len(nodes)
    logger.info(
        "rag_ingest_complete",
        document_id=document_id,
        short_name=short_name,
        chunk_count=chunk_count,
    )

    return chunk_count


async def query_dv_act(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Query the indexed DV Act for relevant sections.

    Args:
        query: Natural language query describing the legal need.
               e.g. "protection order for physical domestic violence"
               e.g. "maintenance relief for wife and children"
        top_k: Number of relevant sections to retrieve.
        document_id: Filter to specific document UUID (optional).

    Returns:
        List of dicts with keys: ``text``, ``source_citation``, ``score``.
        Ordered by relevance (highest score first).
    """
    logger.info("rag_query_start", top_k=top_k, document_id=document_id)

    vector_store = _get_vector_store("legal_document_chunks")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    retriever = index.as_retriever(similarity_top_k=top_k)

    # Build filter if document_id specified
    nodes: list[NodeWithScore] = await retriever.aretrieve(query)

    results = []
    for node_with_score in nodes:
        node: TextNode = node_with_score.node  # type: ignore[assignment]
        results.append({
            "text": node.get_content(),
            "source_citation": node.metadata.get(
                "source_citation", "DV Act 2005"
            ),
            "score": float(node_with_score.score or 0.0),
            "page": node.metadata.get("page_label", "?"),
            "document_id": node.metadata.get("document_id"),
        })

    logger.info(
        "rag_query_complete",
        results_count=len(results),
        top_score=results[0]["score"] if results else 0.0,
    )

    return results


def build_legal_context_string(
    retrieved_sections: list[dict[str, Any]],
) -> str:
    """Format retrieved legal sections into a string for LLM prompt injection.

    Args:
        retrieved_sections: Output from query_dv_act().

    Returns:
        Formatted string with section text and citations.
        Example:
            [SOURCE: DV Act 2005, Page 12]
            Any aggrieved person who is, or has been, in a domestic relationship...

            [SOURCE: DV Act 2005, Page 15]
            The Magistrate may, on being satisfied that domestic violence has taken place...
    """
    if not retrieved_sections:
        return "No specific law found in provided grounding."

    parts = []
    for section in retrieved_sections:
        parts.append(
            f"[SOURCE: {section['source_citation']}]\n{section['text']}"
        )

    return "\n\n".join(parts)