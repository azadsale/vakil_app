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
    # LlamaIndex / LLM
    # -------------------------------------------------------------------------
    openai_api_key: SecretStr = SecretStr("")
    llama_index_embed_model: str = "text-embedding-ada-002"
    llama_index_llm_model: str = "gpt-4o"

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