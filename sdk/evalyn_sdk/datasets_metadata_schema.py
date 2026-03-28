"""Dataset metadata schema enforcement.

Validate item metadata against a defined schema, infer schemas
from existing data, and apply defaults for missing fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Type map for validation
# ---------------------------------------------------------------------------

_TYPE_MAP: Dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
}

_PYTHON_TYPE_TO_FIELD: Dict[type, str] = {v: k for k, v in _TYPE_MAP.items()}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class FieldSchema:
    """Schema definition for a single metadata field."""

    name: str
    field_type: str = "str"
    required: bool = False
    allowed_values: List[Any] = field(default_factory=list)
    default: Any = None
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "field_type": self.field_type,
            "required": self.required,
            "allowed_values": list(self.allowed_values),
            "default": self.default,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FieldSchema:
        return cls(
            name=data["name"],
            field_type=data.get("field_type", "str"),
            required=data.get("required", False),
            allowed_values=data.get("allowed_values", []),
            default=data.get("default"),
            description=data.get("description", ""),
        )


@dataclass
class MetadataSchema:
    """Schema describing expected metadata fields for a dataset."""

    name: str = ""
    fields: List[FieldSchema] = field(default_factory=list)
    version: int = 1

    @property
    def required_fields(self) -> List[str]:
        return [f.name for f in self.fields if f.required]

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "fields": [f.as_dict() for f in self.fields],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetadataSchema:
        return cls(
            name=data.get("name", ""),
            fields=[FieldSchema.from_dict(f) for f in data.get("fields", [])],
            version=data.get("version", 1),
        )


@dataclass
class ValidationError:
    """A single validation error or warning for one item/field."""

    item_id: str
    field: str
    error: str
    severity: str = "error"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "field": self.field,
            "error": self.error,
            "severity": self.severity,
        }


@dataclass
class SchemaValidationResult:
    """Aggregated result of validating a dataset against a schema."""

    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    items_checked: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [e.as_dict() for e in self.errors],
            "warnings": [w.as_dict() for w in self.warnings],
            "items_checked": self.items_checked,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SchemaValidationResult:
        errors = [
            ValidationError(
                item_id=e["item_id"],
                field=e["field"],
                error=e["error"],
                severity=e.get("severity", "error"),
            )
            for e in data.get("errors", [])
        ]
        warnings = [
            ValidationError(
                item_id=w["item_id"],
                field=w["field"],
                error=w["error"],
                severity=w.get("severity", "warning"),
            )
            for w in data.get("warnings", [])
        ]
        return cls(
            valid=data.get("valid", True),
            errors=errors,
            warnings=warnings,
            items_checked=data.get("items_checked", 0),
        )

    def format_text(self) -> str:
        """Format the validation result as human-readable text."""
        lines: List[str] = []
        status = "VALID" if self.valid else "INVALID"
        lines.append(f"Schema validation: {status} ({self.items_checked} items checked)")
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"    [{e.item_id}] {e.field}: {e.error}")
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"    [{w.item_id}] {w.field}: {w.error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_schema(name: str, fields: List[Dict[str, Any]]) -> MetadataSchema:
    """Build a MetadataSchema from a name and a list of field dicts."""
    return MetadataSchema(
        name=name,
        fields=[FieldSchema.from_dict(f) for f in fields],
    )


def validate_item(item: Dict[str, Any], schema: MetadataSchema) -> List[ValidationError]:
    """Validate one item's metadata against a schema.

    The item dict is expected to have an "id" key and a "metadata" key
    containing the fields to validate. If "metadata" is absent the item
    dict itself is treated as the metadata (minus "id").
    """
    item_id = str(item.get("id", "unknown"))
    metadata = item.get("metadata")
    if metadata is None:
        metadata = {k: v for k, v in item.items() if k != "id"}

    errors: List[ValidationError] = []

    for fs in schema.fields:
        value = metadata.get(fs.name)

        # Required check
        if fs.required and value is None:
            errors.append(ValidationError(
                item_id=item_id,
                field=fs.name,
                error="required field missing",
            ))
            continue

        if value is None:
            continue

        # Type check
        expected_type = _TYPE_MAP.get(fs.field_type)
        if expected_type is not None and not isinstance(value, expected_type):
            # Allow int where float is expected
            if fs.field_type == "float" and isinstance(value, int):
                pass
            else:
                errors.append(ValidationError(
                    item_id=item_id,
                    field=fs.name,
                    error=f"expected type {fs.field_type}, got {type(value).__name__}",
                ))
                continue

        # Allowed values check
        if fs.allowed_values and value not in fs.allowed_values:
            errors.append(ValidationError(
                item_id=item_id,
                field=fs.name,
                error=f"value {value!r} not in allowed values {fs.allowed_values}",
            ))

    return errors


def validate_dataset(
    items: List[Dict[str, Any]], schema: MetadataSchema
) -> SchemaValidationResult:
    """Validate all items in a dataset against a schema."""
    all_errors: List[ValidationError] = []
    all_warnings: List[ValidationError] = []

    for item in items:
        item_errors = validate_item(item, schema)
        for ve in item_errors:
            if ve.severity == "warning":
                all_warnings.append(ve)
            else:
                all_errors.append(ve)

    return SchemaValidationResult(
        valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings,
        items_checked=len(items),
    )


def infer_schema(
    items: List[Dict[str, Any]], metadata_field: str = "metadata"
) -> MetadataSchema:
    """Infer a MetadataSchema from items by inspecting metadata fields and types.

    Examines every item's metadata to discover field names and their
    most common Python types. Fields present in all items are marked
    as required.
    """
    if not items:
        return MetadataSchema(name="inferred")

    field_types: Dict[str, Dict[str, int]] = {}
    field_counts: Dict[str, int] = {}

    for item in items:
        metadata = item.get(metadata_field)
        if metadata is None:
            metadata = {k: v for k, v in item.items() if k != "id"}
        if not isinstance(metadata, dict):
            continue

        for key, value in metadata.items():
            field_counts[key] = field_counts.get(key, 0) + 1
            py_type = type(value)
            ft = _PYTHON_TYPE_TO_FIELD.get(py_type, "str")
            if key not in field_types:
                field_types[key] = {}
            field_types[key][ft] = field_types[key].get(ft, 0) + 1

    total_items = len(items)
    fields: List[FieldSchema] = []

    for name in sorted(field_types.keys()):
        # Pick the most common type
        type_counts = field_types[name]
        best_type = max(type_counts, key=lambda t: type_counts[t])
        required = field_counts.get(name, 0) == total_items

        fields.append(FieldSchema(
            name=name,
            field_type=best_type,
            required=required,
        ))

    return MetadataSchema(name="inferred", fields=fields)


def apply_defaults(
    item: Dict[str, Any], schema: MetadataSchema
) -> Dict[str, Any]:
    """Apply default values for missing metadata fields.

    Returns a new dict (shallow copy) with defaults filled in.
    """
    result = dict(item)
    metadata = result.get("metadata")
    if metadata is None:
        metadata = {}
        result["metadata"] = metadata
    else:
        metadata = dict(metadata)
        result["metadata"] = metadata

    for fs in schema.fields:
        if fs.name not in metadata and fs.default is not None:
            metadata[fs.name] = fs.default

    return result
