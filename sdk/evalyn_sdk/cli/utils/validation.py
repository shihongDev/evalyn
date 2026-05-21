"""Validation helpers for CLI commands."""

from __future__ import annotations

import os

from .config import get_config_default, load_config


def check_llm_api_keys(quiet: bool = False) -> tuple[str | None, str | None]:
    """Check for LLM API keys and warn if missing.

    Checks config file first, then environment variables.

    Returns:
        Tuple of (gemini_key, openai_key) - values may be empty strings.
    """
    config = load_config()
    gemini_key = get_config_default(config, "api_keys", "gemini") or os.environ.get(
        "GEMINI_API_KEY", ""
    )
    openai_key = get_config_default(config, "api_keys", "openai") or os.environ.get(
        "OPENAI_API_KEY", ""
    )

    if not quiet:
        if not gemini_key and not openai_key:
            print()
            print("Warning: No API key found for LLM judges.")
            print("   Set GEMINI_API_KEY or OPENAI_API_KEY to enable subjective metrics.")
            print("   Continuing anyway, but LLM judge scores will fail.")
        elif gemini_key and len(gemini_key) < 10:
            print()
            print("Warning: GEMINI_API_KEY appears to be invalid (too short).")

    return gemini_key, openai_key


def extract_project_id(metadata: dict) -> str | None:
    """Extract project ID from metadata dict.

    Handles multiple naming conventions: project_id, project, project_name.
    """
    if not isinstance(metadata, dict):
        return None
    return metadata.get("project_id") or metadata.get("project") or metadata.get("project_name")
