"""Tests for the evaluation result schema standard."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.evaluation.result_schema import (
    EVALYN_RESULT_SCHEMA,
    ResultSchema,
    SchemaField,
    ValidationResult,
    check_backwards_compatibility,
    generate_example_result,
    generate_json_schema,
    get_schema,
    validate_result,
    validate_results_batch,
)


# ---------------------------------------------------------------------------
# get_schema
# ---------------------------------------------------------------------------


class TestGetSchema:
    def test_default_version(self):
        schema = get_schema()
        assert schema.version == "1.0"
        assert schema.name == "evalyn_result"

    def test_explicit_version(self):
        schema = get_schema("1.0")
        assert schema is EVALYN_RESULT_SCHEMA

    def test_unknown_version_raises(self):
        try:
            get_schema("99.0")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass


# ---------------------------------------------------------------------------
# validate_result
# ---------------------------------------------------------------------------


class TestValidateResult:
    def test_valid_result(self):
        result = {
            "run_id": "run-1",
            "metric_id": "helpfulness",
            "item_id": "item-1",
            "score": 0.9,
            "passed": True,
        }
        vr = validate_result(result)
        assert vr.valid is True
        assert vr.errors == []

    def test_valid_result_all_fields(self):
        result = {
            "run_id": "run-1",
            "metric_id": "helpfulness",
            "item_id": "item-1",
            "score": 0.9,
            "passed": True,
            "confidence": 0.95,
            "details": {"reason": "Good"},
            "timestamp": "2026-01-01T00:00:00Z",
        }
        vr = validate_result(result)
        assert vr.valid is True
        assert vr.errors == []
        assert vr.warnings == []

    def test_missing_required_field(self):
        result = {
            "metric_id": "helpfulness",
            "item_id": "item-1",
            "score": 0.9,
            "passed": True,
        }
        vr = validate_result(result)
        assert vr.valid is False
        assert any("run_id" in e for e in vr.errors)

    def test_missing_multiple_required_fields(self):
        result = {"score": 0.5}
        vr = validate_result(result)
        assert vr.valid is False
        assert len(vr.errors) >= 3  # run_id, metric_id, item_id, passed

    def test_wrong_type_string(self):
        result = {
            "run_id": 123,  # should be str
            "metric_id": "helpfulness",
            "item_id": "item-1",
            "score": 0.9,
            "passed": True,
        }
        vr = validate_result(result)
        assert vr.valid is False
        assert any("run_id" in e and "type" in e.lower() for e in vr.errors)

    def test_wrong_type_score(self):
        result = {
            "run_id": "run-1",
            "metric_id": "helpfulness",
            "item_id": "item-1",
            "score": "not_a_number",
            "passed": True,
        }
        vr = validate_result(result)
        assert vr.valid is False
        assert any("score" in e for e in vr.errors)

    def test_wrong_type_passed(self):
        result = {
            "run_id": "run-1",
            "metric_id": "helpfulness",
            "item_id": "item-1",
            "score": 0.9,
            "passed": "yes",
        }
        vr = validate_result(result)
        assert vr.valid is False
        assert any("passed" in e for e in vr.errors)

    def test_extra_fields_warning(self):
        result = {
            "run_id": "run-1",
            "metric_id": "helpfulness",
            "item_id": "item-1",
            "score": 0.9,
            "passed": True,
            "custom_field": "extra",
        }
        vr = validate_result(result)
        assert vr.valid is True
        assert any("custom_field" in w for w in vr.warnings)

    def test_custom_schema(self):
        schema = ResultSchema(
            version="custom",
            name="custom_result",
            fields=[SchemaField("name", "str", True, "Name field")],
        )
        vr = validate_result({"name": "test"}, schema)
        assert vr.valid is True

    def test_integer_for_float_field_is_valid(self):
        """int is acceptable where float is expected."""
        result = {
            "run_id": "run-1",
            "metric_id": "helpfulness",
            "item_id": "item-1",
            "score": 1,  # int, not float
            "passed": True,
        }
        vr = validate_result(result)
        assert vr.valid is True

    def test_empty_result(self):
        vr = validate_result({})
        assert vr.valid is False
        assert len(vr.errors) >= 4  # all required fields missing


# ---------------------------------------------------------------------------
# validate_results_batch
# ---------------------------------------------------------------------------


class TestValidateResultsBatch:
    def test_all_valid(self):
        results = [
            {
                "run_id": "run-1",
                "metric_id": "m1",
                "item_id": "i1",
                "score": 0.8,
                "passed": True,
            },
            {
                "run_id": "run-1",
                "metric_id": "m2",
                "item_id": "i2",
                "score": 0.6,
                "passed": False,
            },
        ]
        vr = validate_results_batch(results)
        assert vr.valid is True
        assert vr.errors == []

    def test_some_invalid(self):
        results = [
            {
                "run_id": "run-1",
                "metric_id": "m1",
                "item_id": "i1",
                "score": 0.8,
                "passed": True,
            },
            {"score": 0.5},  # missing fields
        ]
        vr = validate_results_batch(results)
        assert vr.valid is False
        assert any("[1]" in e for e in vr.errors)

    def test_empty_batch(self):
        vr = validate_results_batch([])
        assert vr.valid is True
        assert vr.errors == []


# ---------------------------------------------------------------------------
# generate_json_schema
# ---------------------------------------------------------------------------


class TestGenerateJsonSchema:
    def test_has_properties(self):
        js = generate_json_schema(EVALYN_RESULT_SCHEMA)
        assert "properties" in js
        assert "run_id" in js["properties"]
        assert "score" in js["properties"]

    def test_has_required(self):
        js = generate_json_schema(EVALYN_RESULT_SCHEMA)
        assert "required" in js
        assert "run_id" in js["required"]
        assert "confidence" not in js["required"]

    def test_draft_07(self):
        js = generate_json_schema(EVALYN_RESULT_SCHEMA)
        assert js["$schema"] == "http://json-schema.org/draft-07/schema#"

    def test_type_mapping(self):
        js = generate_json_schema(EVALYN_RESULT_SCHEMA)
        assert js["properties"]["run_id"]["type"] == "string"
        assert js["properties"]["score"]["type"] == "number"
        assert js["properties"]["passed"]["type"] == "boolean"
        assert js["properties"]["details"]["type"] == "object"

    def test_descriptions_included(self):
        js = generate_json_schema(EVALYN_RESULT_SCHEMA)
        assert "description" in js["properties"]["run_id"]

    def test_custom_schema(self):
        schema = ResultSchema(
            fields=[SchemaField("x", "int", True, "An integer")]
        )
        js = generate_json_schema(schema)
        assert js["properties"]["x"]["type"] == "integer"
        assert "x" in js["required"]


# ---------------------------------------------------------------------------
# generate_example_result
# ---------------------------------------------------------------------------


class TestGenerateExampleResult:
    def test_has_all_fields(self):
        example = generate_example_result(EVALYN_RESULT_SCHEMA)
        for sf in EVALYN_RESULT_SCHEMA.fields:
            assert sf.name in example

    def test_score_is_numeric(self):
        example = generate_example_result(EVALYN_RESULT_SCHEMA)
        assert isinstance(example["score"], (int, float))

    def test_passed_is_bool(self):
        example = generate_example_result(EVALYN_RESULT_SCHEMA)
        assert isinstance(example["passed"], bool)

    def test_example_validates(self):
        example = generate_example_result(EVALYN_RESULT_SCHEMA)
        vr = validate_result(example)
        assert vr.valid is True


# ---------------------------------------------------------------------------
# check_backwards_compatibility
# ---------------------------------------------------------------------------


class TestBackwardsCompatibility:
    def test_identical_schemas_compatible(self):
        ok, issues = check_backwards_compatibility(
            EVALYN_RESULT_SCHEMA, EVALYN_RESULT_SCHEMA
        )
        assert ok is True
        assert issues == []

    def test_add_optional_field_compatible(self):
        new_schema = ResultSchema(
            version="1.1",
            fields=list(EVALYN_RESULT_SCHEMA.fields)
            + [SchemaField("notes", "str", False, "Optional notes")],
        )
        ok, issues = check_backwards_compatibility(EVALYN_RESULT_SCHEMA, new_schema)
        assert ok is True

    def test_add_required_field_breaks(self):
        new_schema = ResultSchema(
            version="2.0",
            fields=list(EVALYN_RESULT_SCHEMA.fields)
            + [SchemaField("new_required", "str", True, "Breaking")],
        )
        ok, issues = check_backwards_compatibility(EVALYN_RESULT_SCHEMA, new_schema)
        assert ok is False
        assert any("new_required" in i for i in issues)

    def test_remove_field_breaks(self):
        new_schema = ResultSchema(
            version="2.0",
            fields=[f for f in EVALYN_RESULT_SCHEMA.fields if f.name != "score"],
        )
        ok, issues = check_backwards_compatibility(EVALYN_RESULT_SCHEMA, new_schema)
        assert ok is False
        assert any("score" in i for i in issues)

    def test_remove_and_add_required_both_reported(self):
        new_schema = ResultSchema(
            version="2.0",
            fields=[
                f for f in EVALYN_RESULT_SCHEMA.fields if f.name != "confidence"
            ]
            + [SchemaField("mandatory_new", "str", True)],
        )
        ok, issues = check_backwards_compatibility(EVALYN_RESULT_SCHEMA, new_schema)
        assert ok is False
        assert len(issues) >= 2


# ---------------------------------------------------------------------------
# EVALYN_RESULT_SCHEMA constant
# ---------------------------------------------------------------------------


class TestEvalynResultSchema:
    def test_has_fields(self):
        assert len(EVALYN_RESULT_SCHEMA.fields) > 0

    def test_required_fields_present(self):
        required = [f.name for f in EVALYN_RESULT_SCHEMA.fields if f.required]
        assert "run_id" in required
        assert "metric_id" in required
        assert "item_id" in required
        assert "score" in required
        assert "passed" in required

    def test_optional_fields_present(self):
        optional = [f.name for f in EVALYN_RESULT_SCHEMA.fields if not f.required]
        assert "confidence" in optional
        assert "details" in optional
        assert "timestamp" in optional

    def test_version(self):
        assert EVALYN_RESULT_SCHEMA.version == "1.0"

    def test_name(self):
        assert EVALYN_RESULT_SCHEMA.name == "evalyn_result"


# ---------------------------------------------------------------------------
# ResultSchema from_dict / as_dict roundtrip
# ---------------------------------------------------------------------------


class TestResultSchemaRoundtrip:
    def test_from_dict_basic(self):
        data = {
            "version": "2.0",
            "name": "custom",
            "fields": [
                {"name": "x", "field_type": "str", "required": True, "description": "X"},
            ],
        }
        schema = ResultSchema.from_dict(data)
        assert schema.version == "2.0"
        assert schema.name == "custom"
        assert len(schema.fields) == 1
        assert schema.fields[0].name == "x"

    def test_roundtrip(self):
        data = EVALYN_RESULT_SCHEMA.as_dict()
        restored = ResultSchema.from_dict(data)
        assert restored.version == EVALYN_RESULT_SCHEMA.version
        assert restored.name == EVALYN_RESULT_SCHEMA.name
        assert len(restored.fields) == len(EVALYN_RESULT_SCHEMA.fields)
        for orig, rest in zip(EVALYN_RESULT_SCHEMA.fields, restored.fields):
            assert orig.name == rest.name
            assert orig.field_type == rest.field_type
            assert orig.required == rest.required

    def test_from_dict_defaults(self):
        schema = ResultSchema.from_dict({})
        assert schema.version == "1.0"
        assert schema.name == "evalyn_result"
        assert schema.fields == []


# ---------------------------------------------------------------------------
# ValidationResult format_text
# ---------------------------------------------------------------------------


class TestValidationResultFormatText:
    def test_valid_format(self):
        vr = ValidationResult(valid=True)
        text = vr.format_text()
        assert "VALID" in text
        assert "ERROR" not in text

    def test_invalid_format(self):
        vr = ValidationResult(valid=False, errors=["Missing field: x"])
        text = vr.format_text()
        assert "INVALID" in text
        assert "Missing field: x" in text

    def test_warnings_shown(self):
        vr = ValidationResult(valid=True, warnings=["Extra field: foo"])
        text = vr.format_text()
        assert "WARNING" in text
        assert "foo" in text


# ---------------------------------------------------------------------------
# SchemaField as_dict
# ---------------------------------------------------------------------------


class TestSchemaField:
    def test_as_dict(self):
        sf = SchemaField("score", "float", True, "A score")
        d = sf.as_dict()
        assert d["name"] == "score"
        assert d["field_type"] == "float"
        assert d["required"] is True
        assert d["description"] == "A score"

    def test_defaults(self):
        sf = SchemaField("x", "str")
        assert sf.required is True
        assert sf.description == ""
