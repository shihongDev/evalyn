"""Structured output enforcement for judge LLM calls.

Force JSON mode on judge responses for reliable parsing. Provides
schema validation, regex fallback extraction, and a full enforcement
pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OutputSchema:
    """Schema describing expected JSON output fields and types."""

    fields: dict[str, str] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fields": dict(self.fields),
            "required_fields": list(self.required_fields),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputSchema:
        return cls(
            fields=data.get("fields", {}),
            required_fields=data.get("required_fields", []),
        )


@dataclass
class ParseResult:
    """Result of attempting to parse structured output."""

    success: bool
    parsed_data: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "parsed_data": dict(self.parsed_data),
            "raw_text": self.raw_text,
            "error": self.error,
        }


@dataclass
class StructuredConfig:
    """Configuration for structured output enforcement."""

    enforce_json: bool = True
    schema: OutputSchema | None = None
    fallback_regex: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "enforce_json": self.enforce_json,
            "schema": self.schema.as_dict() if self.schema else None,
            "fallback_regex": self.fallback_regex,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredConfig:
        schema_data = data.get("schema")
        return cls(
            enforce_json=data.get("enforce_json", True),
            schema=OutputSchema.from_dict(schema_data) if schema_data else None,
            fallback_regex=data.get("fallback_regex", True),
        )


# ---------------------------------------------------------------------------
# Type checking helpers
# ---------------------------------------------------------------------------

_TYPE_CHECKERS = {
    "float": lambda v: isinstance(v, (int, float)),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "str": lambda v: isinstance(v, str),
    "list": lambda v: isinstance(v, list),
    "dict": lambda v: isinstance(v, dict),
}


def _check_type(value: Any, expected: str) -> bool:
    checker = _TYPE_CHECKERS.get(expected)
    if checker is None:
        return True  # unknown type - accept
    return checker(value)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def build_json_instruction(schema: OutputSchema) -> str:
    """Generate an instruction telling the LLM to respond with valid JSON.

    Example output:
        "Respond with valid JSON containing: score (float), passed (bool),
         reason (str)."
    """
    parts = []
    for name, type_str in schema.fields.items():
        parts.append(f"{name} ({type_str})")
    field_list = ", ".join(parts)
    return f"Respond with valid JSON containing: {field_list}."


def parse_json_response(text: str) -> ParseResult:
    """Try to parse JSON from text.

    Handles:
    - Raw JSON strings
    - JSON in markdown code blocks (```json ... ```)
    - JSON objects embedded in surrounding text
    """
    if not text or not text.strip():
        return ParseResult(success=False, raw_text=text, error="empty input")

    stripped = text.strip()

    # 1. Try direct parse
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return ParseResult(success=True, parsed_data=data, raw_text=text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Try extracting from markdown code block
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if code_block:
        try:
            data = json.loads(code_block.group(1).strip())
            if isinstance(data, dict):
                return ParseResult(success=True, parsed_data=data, raw_text=text)
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Try to find a JSON object in the text (regex fallback)
    brace_match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(0))
            if isinstance(data, dict):
                return ParseResult(success=True, parsed_data=data, raw_text=text)
        except (json.JSONDecodeError, ValueError):
            pass

    return ParseResult(success=False, raw_text=text, error="no valid JSON found")


def validate_against_schema(data: dict[str, Any], schema: OutputSchema) -> tuple[bool, list[str]]:
    """Check required fields are present and types match.

    Returns (valid, error_messages).
    """
    errors: list[str] = []

    for field_name in schema.required_fields:
        if field_name not in data:
            errors.append(f"missing required field: {field_name}")

    for field_name, expected_type in schema.fields.items():
        if field_name in data:
            if not _check_type(data[field_name], expected_type):
                errors.append(
                    f"field '{field_name}' expected {expected_type}, "
                    f"got {type(data[field_name]).__name__}"
                )

    return (len(errors) == 0, errors)


def extract_score_from_text(text: str) -> float | None:
    """Regex fallback: extract the first float-like number from text.

    E.g. "Score: 0.85" -> 0.85, "rating is 7" -> 7.0
    """
    if not text or not text.strip():
        return None

    match = re.search(r"(\d+\.\d+|\d+)", text)
    if match:
        return float(match.group(1))
    return None


def enforce_structured_output(text: str, schema: OutputSchema | None = None) -> ParseResult:
    """Full pipeline: try JSON parse, validate schema, fallback to regex.

    1. Attempt to parse JSON from text.
    2. If schema provided, validate against it.
    3. If JSON parse fails and fallback is possible, attempt regex score
       extraction and return a minimal result.
    """
    result = parse_json_response(text)

    if result.success and schema is not None:
        valid, errors = validate_against_schema(result.parsed_data, schema)
        if not valid:
            result = ParseResult(
                success=False,
                parsed_data=result.parsed_data,
                raw_text=text,
                error="; ".join(errors),
            )

    if not result.success:
        score = extract_score_from_text(text)
        if score is not None:
            return ParseResult(
                success=True,
                parsed_data={"score": score},
                raw_text=text,
            )

    return result
