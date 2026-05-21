"""Structured logging configuration for evalyn.

Provides JSON-formatted log output for machine parsing, configurable
verbosity, and file output. Respects EVALYN_LOG_LEVEL and EVALYN_LOG_FILE
environment variables.

Usage:
    from evalyn_sdk.logging_config import configure_logging

    configure_logging(level="info", json_format=True, log_file="evalyn.log")
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Format log records as JSON objects (one per line)."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        if hasattr(record, "evalyn_data"):
            log_entry["data"] = record.evalyn_data
        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable format with timestamp and level."""

    def __init__(self):
        super().__init__(fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s")


def configure_logging(
    level: str | None = None,
    json_format: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure evalyn logging.

    Priority: function args > env vars > defaults.

    Args:
        level: Log level (debug/info/warning/error). Defaults to "warning".
        json_format: If True, output JSON-formatted logs.
        log_file: Optional file path for log output.
    """
    # Resolve level
    effective_level = (level or os.environ.get("EVALYN_LOG_LEVEL") or "warning").upper()

    log_level = getattr(logging, effective_level, logging.WARNING)

    # Resolve file
    effective_file = log_file or os.environ.get("EVALYN_LOG_FILE")

    # Resolve format
    use_json = json_format or os.environ.get("EVALYN_LOG_FORMAT", "").lower() == "json"

    # Create formatter
    formatter = JSONFormatter() if use_json else TextFormatter()

    # Configure the evalyn root logger
    evalyn_logger = logging.getLogger("evalyn_sdk")
    evalyn_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    evalyn_logger.handlers.clear()

    # Add stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(log_level)
    evalyn_logger.addHandler(stderr_handler)

    # Add file handler if requested
    if effective_file:
        file_handler = logging.FileHandler(effective_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        evalyn_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the evalyn_sdk namespace.

    Args:
        name: Logger name (will be prefixed with evalyn_sdk.)

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"evalyn_sdk.{name}")
