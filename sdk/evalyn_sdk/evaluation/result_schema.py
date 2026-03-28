"""Evaluation result schema standard.

Defines a JSON schema for evaluation results with validation,
JSON Schema generation, and backwards compatibility checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type mapping for validation and JSON Schema generation
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "str": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
    "dict": dict,
    "list": list,
}

_JSON_SCHEMA_TYPE_MAP = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
}

_EXAMPLE_VALUES: Dict[str, Any] = {
    "str": "example",
    "int": 0,
    "float": 0.5,
    "bool": True,
    "dict": {},
    "list": [],
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SchemaField:
    """A single field in a result schema."""

    name: str
    field_type: str
    required: bool = True
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "field_type": self.field_type,
            "required": self.required,
            "description": self.description,
        }


@dataclass
class ResultSchema:
    """Schema definition for evaluation results."""

    version: str = "1.0"
    fields: List[SchemaField] = field(default_factory=list)
    name: str = "evalyn_result"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "fields": [f.as_dict() for f in self.fields],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResultSchema":
        return cls(
            version=data.get("version", "1.0"),
            name=data.get("name", "evalyn_result"),
            fields=[
                SchemaField(
                    name=f["name"],
                    field_type=f["field_type"],
                    required=f.get("required", True),
                    description=f.get("description", ""),
                )
                for f in data.get("fields", [])
            ],
        )


@dataclass
class ValidationResult:
    """Result of validating one or more results against a schema."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    def format_text(self) -> str:
        lines: List[str] = []
        status = "VALID" if self.valid else "INVALID"
        lines.append(f"Validation: {status}")
        for err in self.errors:
            lines.append(f"  ERROR: {err}")
        for warn in self.warnings:
            lines.append(f"  WARNING: {warn}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default Schema
# ---------------------------------------------------------------------------

EVALYN_RESULT_SCHEMA = ResultSchema(
    version="1.0",
    name="evalyn_result",
    fields=[
        SchemaField("run_id", "str", True, "Unique evaluation run identifier"),
        SchemaField("metric_id", "str", True, "Metric that was evaluated"),
        SchemaField("item_id", "str", True, "Dataset item identifier"),
        SchemaField("score", "float", True, "Numeric score 0-1"),
        SchemaField("passed", "bool", True, "Whether the item passed"),
        SchemaField("confidence", "float", False, "Confidence in the score"),
        SchemaField("details", "dict", False, "Additional evaluation details"),
        SchemaField("timestamp", "str", False, "ISO 8601 timestamp"),
    ],
)

# Registry for versioned schemas
_SCHEMA_REGISTRY: Dict[str, ResultSchema] = {
    "1.0": EVALYN_RESULT_SCHEMA,
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_schema(version: str = "1.0") -> ResultSchema:
    """Get schema by version.

    Args:
        version: Schema version string.

    Returns:
        ResultSchema for the requested version.

    Raises:
        KeyError: If version not found.
    """
    if version not in _SCHEMA_REGISTRY:
        raise KeyError(f"Unknown schema version: {version}")
    return _SCHEMA_REGISTRY[version]


def validate_result(
    result: Dict[str, Any],
    schema: Optional[ResultSchema] = None,
) -> ValidationResult:
    """Validate a single result dict against a schema.

    Args:
        result: Result dict to validate.
        schema: Schema to validate against (defaults to EVALYN_RESULT_SCHEMA).

    Returns:
        ValidationResult with errors and warnings.
    """
    schema = schema or EVALYN_RESULT_SCHEMA
    errors: List[str] = []
    warnings: List[str] = []

    schema_field_names = {f.name for f in schema.fields}

    for sf in schema.fields:
        if sf.name not in result:
            if sf.required:
                errors.append(f"Missing required field: {sf.name}")
            continue

        value = result[sf.name]
        expected_type = _TYPE_MAP.get(sf.field_type)
        if expected_type is not None and value is not None:
            if not isinstance(value, expected_type):
                errors.append(
                    f"Wrong type for {sf.name}: expected {sf.field_type}, got {type(value).__name__}"
                )

    # Warn about extra fields
    for key in result:
        if key not in schema_field_names:
            warnings.append(f"Extra field: {key}")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_results_batch(
    results: List[Dict[str, Any]],
    schema: Optional[ResultSchema] = None,
) -> ValidationResult:
    """Validate a batch of results against a schema.

    Args:
        results: List of result dicts.
        schema: Schema to validate against (defaults to EVALYN_RESULT_SCHEMA).

    Returns:
        Combined ValidationResult across all results.
    """
    schema = schema or EVALYN_RESULT_SCHEMA
    all_errors: List[str] = []
    all_warnings: List[str] = []

    for i, result in enumerate(results):
        vr = validate_result(result, schema)
        for err in vr.errors:
            all_errors.append(f"[{i}] {err}")
        for warn in vr.warnings:
            all_warnings.append(f"[{i}] {warn}")

    return ValidationResult(
        valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings,
    )


def generate_json_schema(schema: ResultSchema) -> Dict[str, Any]:
    """Generate a JSON Schema (draft-07 compatible) from a ResultSchema.

    Args:
        schema: The result schema to convert.

    Returns:
        Dict representing a JSON Schema document.
    """
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for sf in schema.fields:
        prop: Dict[str, Any] = {}
        json_type = _JSON_SCHEMA_TYPE_MAP.get(sf.field_type, "string")
        prop["type"] = json_type
        if sf.description:
            prop["description"] = sf.description
        properties[sf.name] = prop
        if sf.required:
            required.append(sf.name)

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": schema.name,
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": True,
    }


def generate_example_result(schema: ResultSchema) -> Dict[str, Any]:
    """Generate an example result matching the schema.

    Args:
        schema: Schema to generate an example for.

    Returns:
        Dict with example values for all fields.
    """
    example: Dict[str, Any] = {}
    for sf in schema.fields:
        if sf.field_type == "str" and sf.name == "timestamp":
            example[sf.name] = datetime.now(timezone.utc).isoformat()
        elif sf.field_type == "str":
            example[sf.name] = f"example_{sf.name}"
        elif sf.field_type == "float" and sf.name == "score":
            example[sf.name] = 0.85
        elif sf.field_type == "float":
            example[sf.name] = _EXAMPLE_VALUES.get(sf.field_type, 0.5)
        else:
            example[sf.name] = _EXAMPLE_VALUES.get(sf.field_type, None)
    return example


def check_backwards_compatibility(
    old_schema: ResultSchema,
    new_schema: ResultSchema,
) -> Tuple[bool, List[str]]:
    """Check if new schema is backwards-compatible with old schema.

    Backwards compatibility is broken if:
    - A new required field is added (not present in old schema)
    - An existing field is removed

    Args:
        old_schema: The previous schema version.
        new_schema: The proposed new schema version.

    Returns:
        Tuple of (is_compatible, list of issues).
    """
    issues: List[str] = []

    old_fields = {f.name: f for f in old_schema.fields}
    new_fields = {f.name: f for f in new_schema.fields}

    # Check for removed fields
    for name in old_fields:
        if name not in new_fields:
            issues.append(f"Removed field: {name}")

    # Check for new required fields
    for name, nf in new_fields.items():
        if name not in old_fields and nf.required:
            issues.append(f"New required field: {name}")

    return (len(issues) == 0, issues)
