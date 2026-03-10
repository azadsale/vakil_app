"""Structured JSON logging with PII redaction.

Uses structlog for structured output. All legal text and PII fields
are explicitly excluded from log context.

PII_SENSITIVE_KEYS: any key whose value must never appear in logs.
"""

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

# Keys that must NEVER appear in log output
PII_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "email",
        "password",
        "hashed_password",
        "phone",
        "contact",
        "aadhar",
        "aadhar_ref",
        "address",
        "full_name",
        "name",          # party names are PII in legal context
        "content",       # legal document text
        "outcome_notes", # hearing notes are legally sensitive
        "ocr_text",
        "embedding",     # embeddings can reconstruct source text
        "token",
        "access_token",
        "refresh_token",
        "secret_key",
        "encryption_key",
        "encryption_iv",
    }
)


def _redact_pii(
    logger: Any,
    method: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Structlog processor: redact PII_SENSITIVE_KEYS from log events.

    Args:
        logger: structlog logger instance.
        method: Log method name (info, warning, etc.).
        event_dict: The log event dictionary to process.

    Returns:
        Sanitized event dictionary with PII values replaced by [REDACTED].
    """
    for key in list(event_dict.keys()):
        if key.lower() in PII_SENSITIVE_KEYS:
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for the application.

    Call once at application startup (in main.py lifespan).

    Args:
        log_level: One of DEBUG | INFO | WARNING | ERROR.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        _redact_pii,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structlog logger.

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        Configured structlog BoundLogger.

    Example:
        logger = get_logger(__name__)
        logger.info("case_created", case_id=str(case.id), case_type=case.case_type)
    """
    return structlog.get_logger(name)