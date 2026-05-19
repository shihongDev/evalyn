"""
Structured input simulation: generate dict/JSON inputs from inferred schemas.

Infers field schemas from seed data, then generates conforming inputs,
single-field mutations, and validation checks. Pure Python - no external deps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class FieldSchema:
    """Schema for a single field in a structured input."""

    name: str
    field_type: str  # "string" / "int" / "float" / "bool" / "list" / "dict"
    example_values: list[Any] = field(default_factory=list)
    required: bool = True
    min_val: float | None = None
    max_val: float | None = None
    choices: list[Any] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "field_type": self.field_type,
            "example_values": list(self.example_values),
            "required": self.required,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "choices": list(self.choices),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldSchema:
        return cls(
            name=data["name"],
            field_type=data["field_type"],
            example_values=data.get("example_values", []),
            required=data.get("required", True),
            min_val=data.get("min_val"),
            max_val=data.get("max_val"),
            choices=data.get("choices", []),
        )


@dataclass
class InputSchema:
    """Schema inferred from seed items."""

    fields: list[FieldSchema] = field(default_factory=list)
    inferred_from: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "fields": [f.as_dict() for f in self.fields],
            "inferred_from": self.inferred_from,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputSchema:
        return cls(
            fields=[FieldSchema.from_dict(f) for f in data.get("fields", [])],
            inferred_from=data.get("inferred_from", 0),
        )


@dataclass
class StructuredInput:
    """A generated structured input with provenance."""

    data: dict[str, Any] = field(default_factory=dict)
    schema_id: str = ""
    mutated_field: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "data": dict(self.data),
            "schema_id": self.schema_id,
            "mutated_field": self.mutated_field,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredInput:
        return cls(
            data=data.get("data", {}),
            schema_id=data.get("schema_id", ""),
            mutated_field=data.get("mutated_field", ""),
        )


# ---------------------------------------------------------------------------
# Type Detection
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    str: "string",
    int: "int",
    float: "float",
    bool: "bool",
    list: "list",
    dict: "dict",
}


def _detect_type(value: Any) -> str:
    """Map a Python value to a field_type string."""
    # bool must be checked before int (bool is subclass of int)
    if isinstance(value, bool):
        return "bool"
    for py_type, type_str in _TYPE_MAP.items():
        if isinstance(value, py_type):
            return type_str
    return "string"


def _majority_type(values: list[Any]) -> str:
    """Return the most common detected type among values."""
    if not values:
        return "string"
    counts: dict[str, int] = {}
    for v in values:
        t = _detect_type(v)
        counts[t] = counts.get(t, 0) + 1
    return max(counts, key=lambda k: counts[k])


# ---------------------------------------------------------------------------
# Schema Inference
# ---------------------------------------------------------------------------


def infer_schema(seed_items: list[dict[str, Any]]) -> InputSchema:
    """Infer an InputSchema from a list of seed dicts.

    Detects field names, types (by inspecting values), collects example values,
    detects min/max for numeric fields, and detects choices for fields with
    few unique values (<= 10).
    """
    if not seed_items:
        return InputSchema(fields=[], inferred_from=0)

    # Gather values per field across all seed items
    field_values: dict[str, list[Any]] = {}
    field_presence: dict[str, int] = {}

    for item in seed_items:
        for key, val in item.items():
            field_values.setdefault(key, []).append(val)
            field_presence[key] = field_presence.get(key, 0) + 1

    n = len(seed_items)
    fields: list[FieldSchema] = []

    for name, values in field_values.items():
        ft = _majority_type(values)
        required = field_presence[name] == n

        min_val: float | None = None
        max_val: float | None = None
        if ft in ("int", "float"):
            numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if numeric:
                min_val = float(min(numeric))
                max_val = float(max(numeric))

        unique = list(dict.fromkeys(repr(v) for v in values))
        choices: list[Any] = []
        if len(unique) <= 10:
            seen: set = set()
            for v in values:
                key = repr(v)
                if key not in seen:
                    seen.add(key)
                    choices.append(v)

        fields.append(
            FieldSchema(
                name=name,
                field_type=ft,
                example_values=values,
                required=required,
                min_val=min_val,
                max_val=max_val,
                choices=choices,
            )
        )

    return InputSchema(fields=fields, inferred_from=n)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _generate_field_value(
    fs: FieldSchema, rng: random.Random, counter: int = 0
) -> Any:
    """Generate a single field value conforming to the schema."""
    # If choices exist, pick from them
    if fs.choices:
        return rng.choice(fs.choices)

    if fs.field_type == "string":
        if fs.example_values:
            return rng.choice(fs.example_values)
        return f"value_{counter}"

    if fs.field_type == "int":
        lo = int(fs.min_val) if fs.min_val is not None else 0
        hi = int(fs.max_val) if fs.max_val is not None else 100
        if lo > hi:
            lo, hi = hi, lo
        return rng.randint(lo, hi)

    if fs.field_type == "float":
        lo = fs.min_val if fs.min_val is not None else 0.0
        hi = fs.max_val if fs.max_val is not None else 1.0
        if lo > hi:
            lo, hi = hi, lo
        return round(rng.uniform(lo, hi), 4)

    if fs.field_type == "bool":
        return rng.choice([True, False])

    if fs.field_type == "list":
        if fs.example_values:
            examples = [v for v in fs.example_values if isinstance(v, list)]
            if examples:
                return rng.choice(examples)
        return []

    if fs.field_type == "dict":
        if fs.example_values:
            examples = [v for v in fs.example_values if isinstance(v, dict)]
            if examples:
                return rng.choice(examples)
        return {}

    # Fallback
    return f"value_{counter}"


def generate_structured_input(
    schema: InputSchema, rng: random.Random | None = None
) -> StructuredInput:
    """Generate a single valid input conforming to the schema."""
    if rng is None:
        rng = random.Random()

    data: dict[str, Any] = {}
    for i, fs in enumerate(schema.fields):
        data[fs.name] = _generate_field_value(fs, rng, counter=i)

    return StructuredInput(data=data)


def generate_batch_structured(
    schema: InputSchema,
    count: int,
    rng: random.Random | None = None,
) -> list[StructuredInput]:
    """Generate multiple structured inputs."""
    if rng is None:
        rng = random.Random()
    return [generate_structured_input(schema, rng) for _ in range(count)]


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


def mutate_single_field(
    base: dict[str, Any],
    schema: InputSchema,
    field_name: str,
    rng: random.Random | None = None,
) -> StructuredInput:
    """Create a mutation by changing only one field and keeping the rest."""
    if rng is None:
        rng = random.Random()

    new_data = dict(base)

    # Find the field schema
    fs = None
    for f in schema.fields:
        if f.name == field_name:
            fs = f
            break

    if fs is None:
        raise ValueError(f"Field '{field_name}' not found in schema")

    # Generate a new value that differs from the current one if possible
    attempts = 0
    while attempts < 20:
        new_val = _generate_field_value(fs, rng)
        if new_val != base.get(field_name) or attempts >= 19:
            break
        attempts += 1

    new_data[field_name] = new_val
    return StructuredInput(data=new_data, mutated_field=field_name)


def generate_mutation_suite(
    base: dict[str, Any], schema: InputSchema
) -> list[StructuredInput]:
    """Generate one mutation per field in the schema."""
    rng = random.Random(42)
    results: list[StructuredInput] = []
    for fs in schema.fields:
        results.append(mutate_single_field(base, schema, fs.name, rng))
    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


_VALID_TYPES = {
    "string": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
    "list": list,
    "dict": dict,
}


def validate_against_schema(
    item: dict[str, Any], schema: InputSchema
) -> list[str]:
    """Return a list of validation errors for item against schema."""
    errors: list[str] = []

    for fs in schema.fields:
        if fs.name not in item:
            if fs.required:
                errors.append(f"Missing required field: {fs.name}")
            continue

        val = item[fs.name]

        # Type check - bool must be checked specially since bool is subclass of int
        if fs.field_type == "bool":
            if not isinstance(val, bool):
                errors.append(
                    f"Field '{fs.name}': expected bool, got {type(val).__name__}"
                )
        elif fs.field_type == "int":
            if isinstance(val, bool) or not isinstance(val, int):
                errors.append(
                    f"Field '{fs.name}': expected int, got {type(val).__name__}"
                )
        elif fs.field_type in _VALID_TYPES:
            expected = _VALID_TYPES[fs.field_type]
            if not isinstance(val, expected):
                errors.append(
                    f"Field '{fs.name}': expected {fs.field_type}, got {type(val).__name__}"
                )

        # Range check for numeric fields
        if fs.field_type in ("int", "float") and isinstance(val, (int, float)) and not isinstance(val, bool):
            if fs.min_val is not None and val < fs.min_val:
                errors.append(
                    f"Field '{fs.name}': value {val} below min {fs.min_val}"
                )
            if fs.max_val is not None and val > fs.max_val:
                errors.append(
                    f"Field '{fs.name}': value {val} above max {fs.max_val}"
                )

        # Choices check
        if fs.choices and val not in fs.choices:
            errors.append(
                f"Field '{fs.name}': value {val!r} not in choices {fs.choices}"
            )

    return errors


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_schema_summary(schema: InputSchema) -> str:
    """Return a human-readable schema description."""
    lines = [
        f"Schema (inferred from {schema.inferred_from} items)",
        "-" * 40,
    ]
    for fs in schema.fields:
        req = "required" if fs.required else "optional"
        parts = [f"  {fs.name}: {fs.field_type} ({req})"]
        if fs.min_val is not None or fs.max_val is not None:
            parts.append(f"    range: [{fs.min_val}, {fs.max_val}]")
        if fs.choices:
            parts.append(f"    choices: {fs.choices}")
        lines.extend(parts)
    return "\n".join(lines)
