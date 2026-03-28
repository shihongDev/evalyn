"""Environment variable validation for CLI commands.

Checks required environment variables and config before starting
long-running operations. Provides clear error messages with fix hints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EnvCheckResult:
    """Result of environment validation."""

    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    found_keys: Dict[str, str] = field(default_factory=dict)  # key -> source

    def format_text(self) -> str:
        lines = []
        if self.errors:
            for e in self.errors:
                lines.append(f"  ERROR: {e}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  WARN:  {w}")
        if self.found_keys:
            for key, source in sorted(self.found_keys.items()):
                lines.append(f"  OK:    {key} ({source})")
        if not lines:
            lines.append("  All checks passed.")
        return "\n".join(lines)


def check_eval_environment(
    provider: str = "gemini",
    needs_subjective: bool = True,
    config: Optional[dict] = None,
) -> EnvCheckResult:
    """Validate environment for evaluation operations.

    Checks that required API keys are available and valid-looking.

    Args:
        provider: LLM provider to check ("gemini", "openai", "ollama").
        needs_subjective: Whether subjective (LLM judge) metrics are used.
        config: Optional evalyn config dict for key lookup.

    Returns:
        EnvCheckResult with pass/fail status and messages.
    """
    errors = []
    warnings = []
    found: Dict[str, str] = {}

    if not needs_subjective:
        # Objective-only evaluation needs no API keys
        return EnvCheckResult(passed=True, found_keys={"mode": "objective-only"})

    if provider == "ollama":
        # Ollama is local, no API key needed
        # But check if ollama is accessible
        import shutil
        if shutil.which("ollama"):
            found["ollama"] = "binary found in PATH"
        else:
            warnings.append(
                "ollama binary not found in PATH. "
                "Install from https://ollama.ai or set provider to gemini/openai."
            )
        return EnvCheckResult(passed=True, warnings=warnings, found_keys=found)

    # Check for API keys
    key_map = {
        "gemini": [("GEMINI_API_KEY", "env"), ("GOOGLE_API_KEY", "env")],
        "openai": [("OPENAI_API_KEY", "env")],
    }

    env_keys = key_map.get(provider, key_map["gemini"])
    key_found = False

    for key_name, source in env_keys:
        value = os.environ.get(key_name, "")

        # Also check config
        if not value and config:
            config_keys = config.get("api_keys", {})
            value = config_keys.get(provider, "") or config_keys.get(key_name.lower(), "")
            source = "evalyn.yaml"

        if value:
            # Basic format validation
            if len(value) < 10:
                warnings.append(f"{key_name} appears too short ({len(value)} chars) - may be invalid")
            else:
                found[key_name] = source
                key_found = True
            break

    if not key_found:
        expected_keys = [k for k, _ in env_keys]
        errors.append(
            f"No API key found for provider '{provider}'. "
            f"Set {' or '.join(expected_keys)} environment variable, "
            f"or add to evalyn.yaml under api_keys."
        )

    return EnvCheckResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        found_keys=found,
    )


def check_storage_environment() -> EnvCheckResult:
    """Validate storage environment.

    Checks that the database directory is writable.
    """
    errors = []
    warnings = []
    found: Dict[str, str] = {}

    db_path = os.environ.get("EVALYN_DB")
    if db_path:
        found["EVALYN_DB"] = db_path
        # Check if parent directory exists and is writable
        from pathlib import Path
        parent = Path(db_path).parent
        if not parent.exists():
            warnings.append(f"Database parent directory does not exist: {parent}")
        elif not os.access(str(parent), os.W_OK):
            errors.append(f"Database directory is not writable: {parent}")
    else:
        found["storage"] = "default (data/prod/traces.sqlite)"

    return EnvCheckResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        found_keys=found,
    )
