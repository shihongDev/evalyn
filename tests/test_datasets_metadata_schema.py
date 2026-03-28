"""Tests for datasets_metadata_schema module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.datasets_metadata_schema import (
    FieldSchema,
    MetadataSchema,
    ValidationError,
    SchemaValidationResult,
    create_schema,
    validate_item,
    validate_dataset,
    infer_schema,
    apply_defaults,
)


# ---------------------------------------------------------------------------
# create_schema
# ---------------------------------------------------------------------------


class TestCreateSchema:
    def test_create_from_dicts(self):
        schema = create_schema("test", [
            {"name": "source", "field_type": "str", "required": True},
            {"name": "score", "field_type": "float"},
        ])
        assert schema.name == "test"
        assert len(schema.fields) == 2
        assert schema.fields[0].name == "source"
        assert schema.fields[0].required is True
        assert schema.fields[1].field_type == "float"

    def test_create_empty_fields(self):
        schema = create_schema("empty", [])
        assert schema.name == "empty"
        assert schema.fields == []


# ---------------------------------------------------------------------------
# validate_item
# ---------------------------------------------------------------------------


class TestValidateItem:
    def test_valid_item(self):
        schema = create_schema("s", [
            {"name": "source", "field_type": "str", "required": True},
        ])
        item = {"id": "i1", "metadata": {"source": "web"}}
        errors = validate_item(item, schema)
        assert errors == []

    def test_missing_required(self):
        schema = create_schema("s", [
            {"name": "source", "field_type": "str", "required": True},
        ])
        item = {"id": "i1", "metadata": {}}
        errors = validate_item(item, schema)
        assert len(errors) == 1
        assert "required" in errors[0].error

    def test_wrong_type(self):
        schema = create_schema("s", [
            {"name": "count", "field_type": "int"},
        ])
        item = {"id": "i1", "metadata": {"count": "not_int"}}
        errors = validate_item(item, schema)
        assert len(errors) == 1
        assert "expected type int" in errors[0].error

    def test_allowed_values_violation(self):
        schema = create_schema("s", [
            {"name": "level", "field_type": "str", "allowed_values": ["low", "mid", "high"]},
        ])
        item = {"id": "i1", "metadata": {"level": "extreme"}}
        errors = validate_item(item, schema)
        assert len(errors) == 1
        assert "not in allowed values" in errors[0].error

    def test_allowed_values_ok(self):
        schema = create_schema("s", [
            {"name": "level", "field_type": "str", "allowed_values": ["low", "mid", "high"]},
        ])
        item = {"id": "i1", "metadata": {"level": "mid"}}
        errors = validate_item(item, schema)
        assert errors == []

    def test_optional_field_missing_ok(self):
        schema = create_schema("s", [
            {"name": "tag", "field_type": "str", "required": False},
        ])
        item = {"id": "i1", "metadata": {}}
        errors = validate_item(item, schema)
        assert errors == []

    def test_int_accepted_for_float(self):
        schema = create_schema("s", [
            {"name": "score", "field_type": "float"},
        ])
        item = {"id": "i1", "metadata": {"score": 5}}
        errors = validate_item(item, schema)
        assert errors == []

    def test_item_without_metadata_key(self):
        """When no 'metadata' key, fields from item dict itself are validated."""
        schema = create_schema("s", [
            {"name": "source", "field_type": "str", "required": True},
        ])
        item = {"id": "i1", "source": "web"}
        errors = validate_item(item, schema)
        assert errors == []

    def test_multiple_errors(self):
        schema = create_schema("s", [
            {"name": "a", "field_type": "str", "required": True},
            {"name": "b", "field_type": "int", "required": True},
        ])
        item = {"id": "i1", "metadata": {}}
        errors = validate_item(item, schema)
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------


class TestValidateDataset:
    def test_all_valid(self):
        schema = create_schema("s", [
            {"name": "source", "field_type": "str", "required": True},
        ])
        items = [
            {"id": "i1", "metadata": {"source": "web"}},
            {"id": "i2", "metadata": {"source": "api"}},
        ]
        result = validate_dataset(items, schema)
        assert result.valid is True
        assert result.items_checked == 2
        assert result.errors == []

    def test_mixed_errors(self):
        schema = create_schema("s", [
            {"name": "source", "field_type": "str", "required": True},
        ])
        items = [
            {"id": "i1", "metadata": {"source": "web"}},
            {"id": "i2", "metadata": {}},
        ]
        result = validate_dataset(items, schema)
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.items_checked == 2

    def test_empty_items(self):
        schema = create_schema("s", [
            {"name": "x", "field_type": "str", "required": True},
        ])
        result = validate_dataset([], schema)
        assert result.valid is True
        assert result.items_checked == 0


# ---------------------------------------------------------------------------
# infer_schema
# ---------------------------------------------------------------------------


class TestInferSchema:
    def test_infer_from_items(self):
        items = [
            {"id": "i1", "metadata": {"source": "web", "count": 5}},
            {"id": "i2", "metadata": {"source": "api", "count": 3}},
        ]
        schema = infer_schema(items)
        assert schema.name == "inferred"
        names = schema.field_names
        assert "source" in names
        assert "count" in names
        # Both present in all items, so required
        assert "source" in schema.required_fields
        assert "count" in schema.required_fields

    def test_infer_partial_fields(self):
        items = [
            {"id": "i1", "metadata": {"source": "web"}},
            {"id": "i2", "metadata": {"source": "api", "tag": "test"}},
        ]
        schema = infer_schema(items)
        # source present in both, tag only in one
        assert "source" in schema.required_fields
        assert "tag" not in schema.required_fields

    def test_infer_types(self):
        items = [
            {"id": "i1", "metadata": {"flag": True, "score": 0.5}},
            {"id": "i2", "metadata": {"flag": False, "score": 0.8}},
        ]
        schema = infer_schema(items)
        field_map = {f.name: f for f in schema.fields}
        assert field_map["flag"].field_type == "bool"
        assert field_map["score"].field_type == "float"

    def test_infer_empty_items(self):
        schema = infer_schema([])
        assert schema.fields == []

    def test_infer_no_metadata_field(self):
        """Items without 'metadata' key: falls back to item keys minus 'id'."""
        items = [
            {"id": "i1", "color": "red"},
            {"id": "i2", "color": "blue"},
        ]
        schema = infer_schema(items)
        assert "color" in schema.field_names


# ---------------------------------------------------------------------------
# apply_defaults
# ---------------------------------------------------------------------------


class TestApplyDefaults:
    def test_fills_missing(self):
        schema = create_schema("s", [
            {"name": "source", "field_type": "str", "default": "unknown"},
            {"name": "priority", "field_type": "int", "default": 0},
        ])
        item = {"id": "i1", "metadata": {}}
        result = apply_defaults(item, schema)
        assert result["metadata"]["source"] == "unknown"
        assert result["metadata"]["priority"] == 0

    def test_does_not_overwrite_existing(self):
        schema = create_schema("s", [
            {"name": "source", "field_type": "str", "default": "unknown"},
        ])
        item = {"id": "i1", "metadata": {"source": "web"}}
        result = apply_defaults(item, schema)
        assert result["metadata"]["source"] == "web"

    def test_no_default_no_change(self):
        schema = create_schema("s", [
            {"name": "tag", "field_type": "str"},
        ])
        item = {"id": "i1", "metadata": {}}
        result = apply_defaults(item, schema)
        assert "tag" not in result["metadata"]

    def test_creates_metadata_key_if_missing(self):
        schema = create_schema("s", [
            {"name": "x", "field_type": "str", "default": "y"},
        ])
        item = {"id": "i1"}
        result = apply_defaults(item, schema)
        assert result["metadata"]["x"] == "y"


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestSchemaProperties:
    def test_required_fields(self):
        schema = create_schema("s", [
            {"name": "a", "required": True},
            {"name": "b", "required": False},
            {"name": "c", "required": True},
        ])
        assert schema.required_fields == ["a", "c"]

    def test_field_names(self):
        schema = create_schema("s", [
            {"name": "x"},
            {"name": "y"},
        ])
        assert schema.field_names == ["x", "y"]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_field_schema_roundtrip(self):
        fs = FieldSchema(name="src", field_type="str", required=True, description="source")
        d = fs.as_dict()
        fs2 = FieldSchema.from_dict(d)
        assert fs2.name == "src"
        assert fs2.required is True

    def test_metadata_schema_roundtrip(self):
        schema = create_schema("test", [
            {"name": "a", "field_type": "int", "required": True},
        ])
        d = schema.as_dict()
        schema2 = MetadataSchema.from_dict(d)
        assert schema2.name == "test"
        assert len(schema2.fields) == 1
        assert schema2.fields[0].name == "a"
        assert schema2.version == 1

    def test_validation_error_as_dict(self):
        ve = ValidationError(item_id="i1", field="x", error="bad", severity="error")
        d = ve.as_dict()
        assert d["item_id"] == "i1"
        assert d["severity"] == "error"

    def test_schema_validation_result_roundtrip(self):
        r = SchemaValidationResult(
            valid=False,
            errors=[ValidationError("i1", "x", "bad")],
            warnings=[ValidationError("i2", "y", "warn", severity="warning")],
            items_checked=5,
        )
        d = r.as_dict()
        r2 = SchemaValidationResult.from_dict(d)
        assert r2.valid is False
        assert len(r2.errors) == 1
        assert len(r2.warnings) == 1
        assert r2.items_checked == 5

    def test_format_text_valid(self):
        r = SchemaValidationResult(valid=True, items_checked=3)
        text = r.format_text()
        assert "VALID" in text
        assert "3 items" in text

    def test_format_text_invalid(self):
        r = SchemaValidationResult(
            valid=False,
            errors=[ValidationError("i1", "x", "bad")],
            items_checked=2,
        )
        text = r.format_text()
        assert "INVALID" in text
        assert "Errors: 1" in text
        assert "[i1]" in text
