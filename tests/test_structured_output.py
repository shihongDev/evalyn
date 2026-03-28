"""Tests for the structured output enforcement module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.evaluation.structured_output import (
    OutputSchema,
    ParseResult,
    StructuredConfig,
    build_json_instruction,
    parse_json_response,
    validate_against_schema,
    extract_score_from_text,
    enforce_structured_output,
)


# ---------------------------------------------------------------------------
# build_json_instruction
# ---------------------------------------------------------------------------


class TestBuildJsonInstruction:
    def test_includes_all_fields(self):
        schema = OutputSchema(
            fields={"score": "float", "passed": "bool", "reason": "str"}
        )
        result = build_json_instruction(schema)
        assert "score (float)" in result
        assert "passed (bool)" in result
        assert "reason (str)" in result

    def test_starts_with_respond(self):
        schema = OutputSchema(fields={"score": "float"})
        result = build_json_instruction(schema)
        assert result.startswith("Respond with valid JSON")

    def test_single_field(self):
        schema = OutputSchema(fields={"value": "int"})
        result = build_json_instruction(schema)
        assert "value (int)" in result

    def test_empty_fields(self):
        schema = OutputSchema(fields={})
        result = build_json_instruction(schema)
        assert "Respond with valid JSON containing: ." == result


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_valid_json(self):
        text = '{"score": 0.85, "passed": true}'
        result = parse_json_response(text)
        assert result.success is True
        assert result.parsed_data["score"] == 0.85
        assert result.parsed_data["passed"] is True

    def test_json_in_code_block(self):
        text = '```json\n{"score": 0.9}\n```'
        result = parse_json_response(text)
        assert result.success is True
        assert result.parsed_data["score"] == 0.9

    def test_json_in_code_block_no_lang(self):
        text = '```\n{"score": 0.7}\n```'
        result = parse_json_response(text)
        assert result.success is True
        assert result.parsed_data["score"] == 0.7

    def test_json_in_surrounding_text(self):
        text = 'Here is my evaluation: {"score": 0.5, "reason": "ok"} done.'
        result = parse_json_response(text)
        assert result.success is True
        assert result.parsed_data["score"] == 0.5

    def test_invalid_json(self):
        text = "this is not json at all"
        result = parse_json_response(text)
        assert result.success is False
        assert result.error != ""

    def test_empty_string(self):
        result = parse_json_response("")
        assert result.success is False
        assert result.error == "empty input"

    def test_whitespace_only(self):
        result = parse_json_response("   \n  ")
        assert result.success is False
        assert result.error == "empty input"

    def test_raw_text_preserved(self):
        text = '{"x": 1}'
        result = parse_json_response(text)
        assert result.raw_text == text

    def test_array_not_treated_as_dict(self):
        text = "[1, 2, 3]"
        result = parse_json_response(text)
        assert result.success is False


# ---------------------------------------------------------------------------
# validate_against_schema
# ---------------------------------------------------------------------------


class TestValidateAgainstSchema:
    def test_valid_data(self):
        schema = OutputSchema(
            fields={"score": "float", "passed": "bool"},
            required_fields=["score", "passed"],
        )
        valid, errors = validate_against_schema(
            {"score": 0.9, "passed": True}, schema
        )
        assert valid is True
        assert errors == []

    def test_missing_required_field(self):
        schema = OutputSchema(
            fields={"score": "float", "reason": "str"},
            required_fields=["score", "reason"],
        )
        valid, errors = validate_against_schema({"score": 0.5}, schema)
        assert valid is False
        assert any("reason" in e for e in errors)

    def test_wrong_type(self):
        schema = OutputSchema(
            fields={"score": "float"},
            required_fields=[],
        )
        valid, errors = validate_against_schema({"score": "not a float"}, schema)
        assert valid is False
        assert any("expected float" in e for e in errors)

    def test_int_accepted_as_float(self):
        schema = OutputSchema(fields={"score": "float"})
        valid, errors = validate_against_schema({"score": 1}, schema)
        assert valid is True

    def test_extra_fields_ok(self):
        schema = OutputSchema(fields={"score": "float"}, required_fields=["score"])
        valid, errors = validate_against_schema(
            {"score": 0.5, "extra": "data"}, schema
        )
        assert valid is True

    def test_unknown_type_accepted(self):
        schema = OutputSchema(fields={"data": "custom_type"})
        valid, errors = validate_against_schema({"data": [1, 2]}, schema)
        assert valid is True


# ---------------------------------------------------------------------------
# extract_score_from_text
# ---------------------------------------------------------------------------


class TestExtractScoreFromText:
    def test_score_colon_float(self):
        assert extract_score_from_text("Score: 0.85") == 0.85

    def test_integer(self):
        assert extract_score_from_text("rating is 7") == 7.0

    def test_float_in_sentence(self):
        assert extract_score_from_text("I give this a 3.5 out of 5") == 3.5

    def test_no_number(self):
        assert extract_score_from_text("no numbers here") is None

    def test_empty_string(self):
        assert extract_score_from_text("") is None

    def test_whitespace_only(self):
        assert extract_score_from_text("   ") is None


# ---------------------------------------------------------------------------
# enforce_structured_output
# ---------------------------------------------------------------------------


class TestEnforceStructuredOutput:
    def test_valid_json_no_schema(self):
        result = enforce_structured_output('{"score": 0.8}')
        assert result.success is True
        assert result.parsed_data["score"] == 0.8

    def test_valid_json_with_schema(self):
        schema = OutputSchema(
            fields={"score": "float"}, required_fields=["score"]
        )
        result = enforce_structured_output('{"score": 0.8}', schema)
        assert result.success is True

    def test_schema_validation_failure_falls_back(self):
        schema = OutputSchema(
            fields={"score": "float", "reason": "str"},
            required_fields=["reason"],
        )
        result = enforce_structured_output('{"score": 0.8}', schema)
        # JSON parsed but schema failed; regex fallback finds 0.8
        assert result.success is True
        assert result.parsed_data == {"score": 0.8}

    def test_no_json_regex_fallback(self):
        result = enforce_structured_output("The score is 0.75 overall")
        assert result.success is True
        assert result.parsed_data["score"] == 0.75

    def test_no_json_no_number(self):
        result = enforce_structured_output("no useful data here")
        assert result.success is False

    def test_empty_text(self):
        result = enforce_structured_output("")
        assert result.success is False


# ---------------------------------------------------------------------------
# OutputSchema as_dict / from_dict
# ---------------------------------------------------------------------------


class TestOutputSchemaRoundtrip:
    def test_roundtrip(self):
        original = OutputSchema(
            fields={"score": "float", "reason": "str"},
            required_fields=["score"],
        )
        data = original.as_dict()
        restored = OutputSchema.from_dict(data)
        assert restored.fields == original.fields
        assert restored.required_fields == original.required_fields

    def test_from_dict_defaults(self):
        restored = OutputSchema.from_dict({})
        assert restored.fields == {}
        assert restored.required_fields == []


# ---------------------------------------------------------------------------
# StructuredConfig as_dict / from_dict
# ---------------------------------------------------------------------------


class TestStructuredConfigRoundtrip:
    def test_roundtrip_with_schema(self):
        schema = OutputSchema(fields={"x": "int"}, required_fields=["x"])
        original = StructuredConfig(
            enforce_json=False, schema=schema, fallback_regex=False
        )
        data = original.as_dict()
        restored = StructuredConfig.from_dict(data)
        assert restored.enforce_json is False
        assert restored.fallback_regex is False
        assert restored.schema is not None
        assert restored.schema.fields == {"x": "int"}

    def test_roundtrip_no_schema(self):
        original = StructuredConfig()
        data = original.as_dict()
        restored = StructuredConfig.from_dict(data)
        assert restored.schema is None
        assert restored.enforce_json is True
        assert restored.fallback_regex is True

    def test_from_dict_defaults(self):
        restored = StructuredConfig.from_dict({})
        assert restored.enforce_json is True
        assert restored.schema is None
        assert restored.fallback_regex is True
