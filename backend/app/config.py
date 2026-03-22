"""Application configuration via Pydantic-Settings.

Reads from environment variables (or .env file).
All sensitive values are SecretStr to prevent accidental logging.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Legal-CoPilot backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # App
    # -------------------------------------------------------------------------
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    database_url: SecretStr  # postgresql+asyncpg://...

    # Connection pool tuning
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_echo: bool = False  # Set True only in dev for SQL logging (no PII!)

    # -------------------------------------------------------------------------
    # Auth / JWT
    # -------------------------------------------------------------------------
    secret_key: SecretStr
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # -------------------------------------------------------------------------
    # Document Encryption (AES-256-GCM)
    # base64-encoded 32-byte key: openssl rand -base64 32
    # -------------------------------------------------------------------------
    document_encryption_key: SecretStr

    # -------------------------------------------------------------------------
    # MinIO / S3
    # -------------------------------------------------------------------------
    minio_endpoint: str = "minio:9000"
    minio_access_key: SecretStr = SecretStr("vakilminio")
    minio_secret_key: SecretStr = SecretStr("vakilminio123")
    minio_bucket_documents: str = "legal-documents"
    minio_secure: bool = False

    # -------------------------------------------------------------------------
    # Redis
    # -------------------------------------------------------------------------
    redis_url: str = "redis://redis:6379/0"

    # -------------------------------------------------------------------------
    # OCR — Azure Document Intelligence
    # -------------------------------------------------------------------------
    azure_document_intelligence_endpoint: str = ""
    azure_document_intelligence_key: SecretStr = SecretStr("")

    # -------------------------------------------------------------------------
    # OCR — Google Document AI
    # -------------------------------------------------------------------------
    google_project_id: str = ""
    google_processor_id: str = ""

    # -------------------------------------------------------------------------
    # Google Gemini (Free LLM — 1M tokens/min, get key at aistudio.google.com)
    # Primary LLM — used when GEMINI_API_KEY is set (recommended)
    # -------------------------------------------------------------------------
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-2.0-flash"   # free tier: 1500 req/day, 15 RPM — use 2.0 not 2.5 (2.5 is only 20/day)

    # -------------------------------------------------------------------------
    # Groq (Fallback LLM — 12K tokens/min free, get key at console.groq.com)
    # Used automatically when GEMINI_API_KEY is not set
    # -------------------------------------------------------------------------
    groq_api_key: SecretStr = SecretStr("")
    groq_api_base_url: str = "https://api.groq.com/openai/v1"
    groq_llm_model: str = "llama-3.3-70b-versatile"

    # -------------------------------------------------------------------------
    # LlamaIndex / Embeddings (sentence-transformers — local, no API key)
    # -------------------------------------------------------------------------
    openai_api_key: SecretStr = SecretStr("")  # kept for backward compat
    llama_index_embed_model: str = "BAAI/bge-small-en-v1.5"  # fastembed ONNX model
    llama_index_llm_model: str = "llama-3.3-70b-versatile"
    embedding_dim: int = 384  # bge-small-en-v1.5 dimension

    # -------------------------------------------------------------------------
    # Sarvam AI (Saaras v3 — Audio Transcription)
    # Indian-first: Marathi, Hindi, English, code-switching
    # -------------------------------------------------------------------------
    sarvam_api_key: SecretStr = SecretStr("")
    sarvam_api_base_url: str = "https://api.sarvam.ai"
    sarvam_transcribe_model: str = "saaras:v3"
    sarvam_max_audio_size_mb: int = 25           # Sarvam API hard limit
    sarvam_max_duration_seconds: int = 600        # 10 minutes max per call

    # -------------------------------------------------------------------------
    # e-Courts
    # -------------------------------------------------------------------------
    ecourts_base_url: str = "https://services.ecourts.gov.in"

    @field_validator("database_url", mode="before")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Ensure asyncpg driver is specified."""
        if not str(v).startswith("postgresql+asyncpg"):
            raise ValueError("DATABASE_URL must use postgresql+asyncpg:// driver")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance.

    Use FastAPI dependency injection: Depends(get_settings).
    """
    return Settings()