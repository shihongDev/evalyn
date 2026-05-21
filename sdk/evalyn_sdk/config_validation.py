"""Config validation: check evalyn.yaml for errors, type mismatches, and deprecations."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ConfigError:
    """A single configuration error or warning."""

    field: str
    error_type: str  # "unknown_key", "missing_required", "type_mismatch", "deprecated"
    message: str
    suggestion: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "error_type": self.error_type,
            "message": self.message,
            "suggestion": self.suggestion,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigError:
        return cls(
            field=data["field"],
            error_type=data["error_type"],
            message=data["message"],
            suggestion=data.get("suggestion", ""),
        )


@dataclass
class ConfigSchema:
    """Schema definition for config validation."""

    required_fields: list[str] = field(default_factory=list)
    optional_fields: list[str] = field(default_factory=list)
    field_types: dict[str, str] = field(default_factory=dict)
    deprecated_fields: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "required_fields": list(self.required_fields),
            "optional_fields": list(self.optional_fields),
            "field_types": dict(self.field_types),
            "deprecated_fields": dict(self.deprecated_fields),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigSchema:
        return cls(
            required_fields=data.get("required_fields", []),
            optional_fields=data.get("optional_fields", []),
            field_types=data.get("field_types", {}),
            deprecated_fields=data.get("deprecated_fields", {}),
        )

    @property
    def all_known_fields(self) -> set[str]:
        """All recognized field names (required + optional + deprecated keys)."""
        return (
            set(self.required_fields)
            | set(self.optional_fields)
            | set(self.deprecated_fields.keys())
        )


@dataclass
class ValidationResult:
    """Result of validating a config against a schema."""

    errors: list[ConfigError] = field(default_factory=list)
    warnings: list[ConfigError] = field(default_factory=list)
    valid: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "errors": [e.as_dict() for e in self.errors],
            "warnings": [w.as_dict() for w in self.warnings],
            "valid": self.valid,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult:
        return cls(
            errors=[ConfigError.from_dict(e) for e in data.get("errors", [])],
            warnings=[ConfigError.from_dict(w) for w in data.get("warnings", [])],
            valid=data.get("valid", True),
        )


# ---------------------------------------------------------------------------
# Default schema
# ---------------------------------------------------------------------------

DEFAULT_SCHEMA = ConfigSchema(
    required_fields=["provider", "model"],
    optional_fields=[
        "dataset_path",
        "metrics_path",
        "output_dir",
        "log_level",
        "parallel_workers",
        "timeout",
        "retry_limit",
        "judge_model",
        "confidence_threshold",
        "cache_enabled",
        "results_format",
    ],
    field_types={
        "provider": "str",
        "model": "str",
        "parallel_workers": "int",
        "timeout": "int",
        "cache_enabled": "bool",
    },
    deprecated_fields={
        "llm_provider": "provider",
        "max_retries": "retry_limit",
    },
)


# ---------------------------------------------------------------------------
# Type checking helpers
# ---------------------------------------------------------------------------

_TYPE_CHECKERS: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


def _check_type(value: Any, expected_type_name: str) -> bool:
    """Check whether a value matches the expected type name.

    For parsed YAML-like values (all strings from simple parsing), we check
    whether the string can be interpreted as the target type.
    """
    if expected_type_name == "str":
        return isinstance(value, str)
    if expected_type_name == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return True
        if isinstance(value, str):
            try:
                int(value)
                return True
            except ValueError:
                return False
        return False
    if expected_type_name == "float":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                return False
        return False
    if expected_type_name == "bool":
        if isinstance(value, bool):
            return True
        if isinstance(value, str):
            return value.lower() in ("true", "false")
        return False
    return True


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def validate_config(config: dict, schema: ConfigSchema | None = None) -> ValidationResult:
    """Validate a config dict against a schema.

    Checks:
    - unknown keys (error)
    - missing required fields (error)
    - type mismatches (error)
    - deprecated fields (warning with suggestion)
    """
    if schema is None:
        schema = DEFAULT_SCHEMA

    errors: list[ConfigError] = []
    warnings: list[ConfigError] = []
    known = schema.all_known_fields

    # Check unknown keys
    for key in config:
        if key not in known:
            errors.append(
                ConfigError(
                    field=key,
                    error_type="unknown_key",
                    message=f"Unknown configuration key: {key}",
                    suggestion=suggest_fix(
                        ConfigError(field=key, error_type="unknown_key", message="")
                    ),
                )
            )

    # Check missing required fields
    for req in schema.required_fields:
        if req not in config:
            errors.append(
                ConfigError(
                    field=req,
                    error_type="missing_required",
                    message=f"Required field missing: {req}",
                    suggestion=f"Add '{req}' to your config file.",
                )
            )

    # Check type mismatches
    for fld, expected in schema.field_types.items():
        if fld in config:
            if not _check_type(config[fld], expected):
                errors.append(
                    ConfigError(
                        field=fld,
                        error_type="type_mismatch",
                        message=f"Type mismatch for '{fld}': expected {expected}, got {type(config[fld]).__name__}",
                        suggestion=f"Change '{fld}' to a {expected} value.",
                    )
                )

    # Check deprecated fields
    for old_name, new_name in schema.deprecated_fields.items():
        if old_name in config:
            warnings.append(
                ConfigError(
                    field=old_name,
                    error_type="deprecated",
                    message=f"Deprecated field: '{old_name}' has been renamed to '{new_name}'",
                    suggestion=f"Rename '{old_name}' to '{new_name}'.",
                )
            )

    valid = len(errors) == 0
    return ValidationResult(errors=errors, warnings=warnings, valid=valid)


def suggest_fix(error: ConfigError) -> str:
    """Return a suggestion string for a given config error."""
    if error.error_type == "unknown_key":
        return f"Remove or check spelling of '{error.field}'."
    if error.error_type == "missing_required":
        return f"Add '{error.field}' to your config file."
    if error.error_type == "type_mismatch":
        return f"Check the type of '{error.field}'."
    if error.error_type == "deprecated":
        return f"Update '{error.field}' to its replacement."
    return ""


def validate_config_file(path: str, schema: ConfigSchema | None = None) -> ValidationResult:
    """Load a YAML-like config file (simple key: value parsing) and validate.

    Supports simple single-line "key: value" format. Lines starting with #
    are comments. Blank lines are skipped.
    """
    if not os.path.isfile(path):
        return ValidationResult(
            errors=[
                ConfigError(
                    field="",
                    error_type="missing_required",
                    message=f"Config file not found: {path}",
                    suggestion="Check the file path.",
                )
            ],
            warnings=[],
            valid=False,
        )

    config: dict[str, Any] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, raw_value = line.partition(":")
            key = key.strip()
            raw_value = raw_value.strip()

            # Attempt to parse typed values
            config[key] = _parse_value(raw_value)

    return validate_config(config, schema)


def _parse_value(raw: str) -> Any:
    """Parse a raw string value into a Python type."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def format_validation_result(result: ValidationResult) -> str:
    """Format a ValidationResult as a human-readable string."""
    lines: list[str] = []

    if result.valid and not result.warnings:
        lines.append("Config validation: PASSED")
        lines.append("No errors or warnings found.")
        return "\n".join(lines)

    if result.valid:
        lines.append("Config validation: PASSED (with warnings)")
    else:
        lines.append("Config validation: FAILED")

    if result.errors:
        lines.append("")
        lines.append(f"Errors ({len(result.errors)}):")
        for err in result.errors:
            lines.append(f"  [{err.error_type}] {err.message}")
            if err.suggestion:
                lines.append(f"    -> {err.suggestion}")

    if result.warnings:
        lines.append("")
        lines.append(f"Warnings ({len(result.warnings)}):")
        for warn in result.warnings:
            lines.append(f"  [{warn.error_type}] {warn.message}")
            if warn.suggestion:
                lines.append(f"    -> {warn.suggestion}")

    return "\n".join(lines)
