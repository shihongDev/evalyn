"""Validation helpers for CLI commands."""

from __future__ import annotations

import os
from typing import Optional, Tuple

from .errors import fatal_error


def check_llm_api_keys(quiet: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """Check for LLM API keys and warn if missing.

    Returns:
        Tuple of (gemini_key, openai_key) - values may be empty strings.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if not quiet:
        if not gemini_key and not openai_key:
            print()
            print("Warning: No API key found for LLM judges.")
            print(
                "   Set GEMINI_API_KEY or OPENAI_API_KEY to enable subjective metrics."
            )
            print("   Continuing anyway, but LLM judge scores will fail.")
        elif gemini_key and len(gemini_key) < 10:
            print()
            print("Warning: GEMINI_API_KEY appears to be invalid (too short).")

    return gemini_key, openai_key


def require_llm_api_key(quiet: bool = False) -> str:
    """Require at least one LLM API key, exit if missing.

    Returns:
        The available API key (prefers GEMINI_API_KEY).
    """
    gemini_key, openai_key = check_llm_api_keys(quiet=True)

    if not gemini_key and not openai_key:
        fatal_error(
            "No API key found for LLM operations",
            "Set GEMINI_API_KEY or OPENAI_API_KEY environment variable",
        )

    return gemini_key or openai_key


def extract_project_id(metadata: dict) -> Optional[str]:
    """Extract project ID from metadata dict.

    Handles multiple naming conventions: project_id, project, project_name.
    """
    if not isinstance(metadata, dict):
        return None
    return (
        metadata.get("project_id")
        or metadata.get("project")
        or metadata.get("project_name")
    )
