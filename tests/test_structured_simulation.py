"""Tests for structured_simulation module."""

from __future__ import annotations

import random

import pytest

from evalyn_sdk.structured_simulation import (
    FieldSchema,
    InputSchema,
    StructuredInput,
    format_schema_summary,
    generate_batch_structured,
    generate_mutation_suite,
    generate_structured_input,
    infer_schema,
    mutate_single_field,
    validate_against_schema,
)


# ---------------------------------------------------------------------------
# Seed data fixtures
# ---------------------------------------------------------------------------

SEED_ITEMS = [
    {"name": "alice", "age": 30, "score": 0.95, "active": True, "tags": ["a", "b"]},
    {"name": "bob", "age": 25, "score": 0.80, "active": False, "tags": ["c"]},
    {"name": "carol", "age": 35, "score": 0.70, "active": True, "tags": ["a"]},
]


# ---------------------------------------------------------------------------
# FieldSchema
# ---------------------------------------------------------------------------


class TestFieldSchema:
    def test_defaults(self):
        f = FieldSchema(name="x", field_type="string")
        assert f.required is True
        assert f.min_val is None
        assert f.choices == []

    def test_as_dict(self):
        f = FieldSchema(name="age", field_type="int", min_val=0.0, max_val=100.0)
        d = f.as_dict()
        assert d["name"] == "age"
        assert d["field_type"] == "int"
        assert d["min_val"] == 0.0

    def test_from_dict(self):
        d = {"name": "x", "field_type": "float", "min_val": 1.0, "max_val": 5.0, "choices": [1.0, 2.0]}
        f = FieldSchema.from_dict(d)
        assert f.name == "x"
        assert f.field_type == "float"
        assert f.choices == [1.0, 2.0]

    def test_from_dict_defaults(self):
        f = FieldSchema.from_dict({"name": "y", "field_type": "bool"})
        assert f.required is True
        assert f.example_values == []

    def test_roundtrip(self):
        f = FieldSchema(name="t", field_type="string", example_values=["a", "b"], required=False)
        f2 = FieldSchema.from_dict(f.as_dict())
        assert f2.name == f.name
        assert f2.example_values == f.example_values
        assert f2.required == f.required


# ---------------------------------------------------------------------------
# InputSchema
# ---------------------------------------------------------------------------


class TestInputSchema:
    def test_empty(self):
        s = InputSchema()
        assert s.fields == []
        assert s.inferred_from == 0

    def test_as_dict(self):
        fs = FieldSchema(name="a", field_type="string")
        s = InputSchema(fields=[fs], inferred_from=5)
        d = s.as_dict()
        assert d["inferred_from"] == 5
        assert len(d["fields"]) == 1

    def test_from_dict(self):
        d = {
            "fields": [{"name": "x", "field_type": "int"}],
            "inferred_from": 10,
        }
        s = InputSchema.from_dict(d)
        assert len(s.fields) == 1
        assert s.inferred_from == 10

    def test_roundtrip(self):
        fs = FieldSchema(name="q", field_type="bool", choices=[True, False])
        s = InputSchema(fields=[fs], inferred_from=3)
        s2 = InputSchema.from_dict(s.as_dict())
        assert s2.inferred_from == s.inferred_from
        assert s2.fields[0].name == "q"


# ---------------------------------------------------------------------------
# StructuredInput
# ---------------------------------------------------------------------------


class TestStructuredInput:
    def test_defaults(self):
        si = StructuredInput()
        assert si.data == {}
        assert si.schema_id == ""
        assert si.mutated_field == ""

    def test_as_dict(self):
        si = StructuredInput(data={"k": 1}, schema_id="s1", mutated_field="k")
        d = si.as_dict()
        assert d["data"] == {"k": 1}
        assert d["mutated_field"] == "k"

    def test_from_dict(self):
        d = {"data": {"a": 2}, "schema_id": "abc", "mutated_field": "a"}
        si = StructuredInput.from_dict(d)
        assert si.data == {"a": 2}
        assert si.schema_id == "abc"

    def test_roundtrip(self):
        si = StructuredInput(data={"x": "y"}, schema_id="id1")
        si2 = StructuredInput.from_dict(si.as_dict())
        assert si2.data == si.data
        assert si2.schema_id == si.schema_id


# ---------------------------------------------------------------------------
# infer_schema
# ---------------------------------------------------------------------------


class TestInferSchema:
    def test_empty_input(self):
        s = infer_schema([])
        assert s.fields == []
        assert s.inferred_from == 0

    def test_basic_inference(self):
        s = infer_schema(SEED_ITEMS)
        assert s.inferred_from == 3
        names = {f.name for f in s.fields}
        assert "name" in names
        assert "age" in names
        assert "score" in names
        assert "active" in names
        assert "tags" in names

    def test_type_detection_string(self):
        s = infer_schema(SEED_ITEMS)
        name_field = next(f for f in s.fields if f.name == "name")
        assert name_field.field_type == "string"

    def test_type_detection_int(self):
        s = infer_schema(SEED_ITEMS)
        age_field = next(f for f in s.fields if f.name == "age")
        assert age_field.field_type == "int"

    def test_type_detection_float(self):
        s = infer_schema(SEED_ITEMS)
        score_field = next(f for f in s.fields if f.name == "score")
        assert score_field.field_type == "float"

    def test_type_detection_bool(self):
        s = infer_schema(SEED_ITEMS)
        active_field = next(f for f in s.fields if f.name == "active")
        assert active_field.field_type == "bool"

    def test_type_detection_list(self):
        s = infer_schema(SEED_ITEMS)
        tags_field = next(f for f in s.fields if f.name == "tags")
        assert tags_field.field_type == "list"

    def test_type_detection_dict(self):
        s = infer_schema([{"meta": {"k": "v"}}, {"meta": {"k": "w"}}])
        meta_field = next(f for f in s.fields if f.name == "meta")
        assert meta_field.field_type == "dict"

    def test_min_max_int(self):
        s = infer_schema(SEED_ITEMS)
        age_field = next(f for f in s.fields if f.name == "age")
        assert age_field.min_val == 25.0
        assert age_field.max_val == 35.0

    def test_min_max_float(self):
        s = infer_schema(SEED_ITEMS)
        score_field = next(f for f in s.fields if f.name == "score")
        assert score_field.min_val == pytest.approx(0.70, abs=0.01)
        assert score_field.max_val == pytest.approx(0.95, abs=0.01)

    def test_choices_detected(self):
        items = [{"color": "red"}, {"color": "blue"}, {"color": "red"}]
        s = infer_schema(items)
        color_field = next(f for f in s.fields if f.name == "color")
        assert set(color_field.choices) == {"red", "blue"}

    def test_no_choices_for_many_unique(self):
        items = [{"id": i} for i in range(20)]
        s = infer_schema(items)
        id_field = next(f for f in s.fields if f.name == "id")
        assert id_field.choices == []

    def test_required_field(self):
        s = infer_schema(SEED_ITEMS)
        for f in s.fields:
            assert f.required is True

    def test_optional_field(self):
        items = [{"a": 1, "b": 2}, {"a": 3}]
        s = infer_schema(items)
        b_field = next(f for f in s.fields if f.name == "b")
        assert b_field.required is False

    def test_example_values_collected(self):
        s = infer_schema(SEED_ITEMS)
        name_field = next(f for f in s.fields if f.name == "name")
        assert len(name_field.example_values) == 3


# ---------------------------------------------------------------------------
# generate_structured_input
# ---------------------------------------------------------------------------


class TestGenerateStructuredInput:
    def test_basic(self):
        schema = infer_schema(SEED_ITEMS)
        si = generate_structured_input(schema, rng=random.Random(42))
        assert "name" in si.data
        assert "age" in si.data
        assert "score" in si.data

    def test_deterministic_with_seed(self):
        schema = infer_schema(SEED_ITEMS)
        si1 = generate_structured_input(schema, rng=random.Random(99))
        si2 = generate_structured_input(schema, rng=random.Random(99))
        assert si1.data == si2.data

    def test_types_match_schema(self):
        schema = infer_schema(SEED_ITEMS)
        si = generate_structured_input(schema, rng=random.Random(42))
        assert isinstance(si.data["name"], str)
        assert isinstance(si.data["age"], int)
        assert isinstance(si.data["score"], float)
        assert isinstance(si.data["active"], bool)
        assert isinstance(si.data["tags"], list)

    def test_numeric_in_range(self):
        schema = infer_schema(SEED_ITEMS)
        rng = random.Random(42)
        for _ in range(50):
            si = generate_structured_input(schema, rng=rng)
            age_field = next(f for f in schema.fields if f.name == "age")
            assert age_field.min_val <= si.data["age"] <= age_field.max_val

    def test_no_rng(self):
        schema = infer_schema(SEED_ITEMS)
        si = generate_structured_input(schema)
        assert "name" in si.data

    def test_choices_respected(self):
        items = [{"status": "on"}, {"status": "off"}]
        schema = infer_schema(items)
        rng = random.Random(42)
        for _ in range(20):
            si = generate_structured_input(schema, rng=rng)
            assert si.data["status"] in ["on", "off"]


# ---------------------------------------------------------------------------
# generate_batch_structured
# ---------------------------------------------------------------------------


class TestGenerateBatchStructured:
    def test_count(self):
        schema = infer_schema(SEED_ITEMS)
        batch = generate_batch_structured(schema, 10, rng=random.Random(42))
        assert len(batch) == 10

    def test_all_valid_types(self):
        schema = infer_schema(SEED_ITEMS)
        batch = generate_batch_structured(schema, 5, rng=random.Random(42))
        for si in batch:
            assert isinstance(si.data["name"], str)
            assert isinstance(si.data["age"], int)

    def test_zero_count(self):
        schema = infer_schema(SEED_ITEMS)
        batch = generate_batch_structured(schema, 0)
        assert batch == []

    def test_no_rng(self):
        schema = infer_schema(SEED_ITEMS)
        batch = generate_batch_structured(schema, 3)
        assert len(batch) == 3


# ---------------------------------------------------------------------------
# mutate_single_field
# ---------------------------------------------------------------------------


class TestMutateSingleField:
    def test_basic(self):
        schema = infer_schema(SEED_ITEMS)
        base = SEED_ITEMS[0]
        result = mutate_single_field(base, schema, "name", rng=random.Random(42))
        assert result.mutated_field == "name"
        # Other fields should be unchanged
        assert result.data["age"] == base["age"]
        assert result.data["score"] == base["score"]

    def test_unknown_field_raises(self):
        schema = infer_schema(SEED_ITEMS)
        with pytest.raises(ValueError, match="not found"):
            mutate_single_field(SEED_ITEMS[0], schema, "nonexistent")

    def test_preserves_other_fields(self):
        schema = infer_schema(SEED_ITEMS)
        base = SEED_ITEMS[0]
        result = mutate_single_field(base, schema, "age", rng=random.Random(42))
        assert result.data["name"] == base["name"]
        assert result.data["active"] == base["active"]
        assert result.data["tags"] == base["tags"]

    def test_mutated_field_recorded(self):
        schema = infer_schema(SEED_ITEMS)
        result = mutate_single_field(SEED_ITEMS[0], schema, "score", rng=random.Random(42))
        assert result.mutated_field == "score"


# ---------------------------------------------------------------------------
# generate_mutation_suite
# ---------------------------------------------------------------------------


class TestGenerateMutationSuite:
    def test_one_per_field(self):
        schema = infer_schema(SEED_ITEMS)
        base = SEED_ITEMS[0]
        suite = generate_mutation_suite(base, schema)
        assert len(suite) == len(schema.fields)

    def test_each_mutates_different_field(self):
        schema = infer_schema(SEED_ITEMS)
        base = SEED_ITEMS[0]
        suite = generate_mutation_suite(base, schema)
        mutated = [si.mutated_field for si in suite]
        field_names = [f.name for f in schema.fields]
        assert mutated == field_names

    def test_deterministic(self):
        schema = infer_schema(SEED_ITEMS)
        base = SEED_ITEMS[0]
        s1 = generate_mutation_suite(base, schema)
        s2 = generate_mutation_suite(base, schema)
        for a, b in zip(s1, s2):
            assert a.data == b.data


# ---------------------------------------------------------------------------
# validate_against_schema
# ---------------------------------------------------------------------------


class TestValidateAgainstSchema:
    def test_valid_item(self):
        schema = infer_schema(SEED_ITEMS)
        errors = validate_against_schema(SEED_ITEMS[0], schema)
        assert errors == []

    def test_missing_required_field(self):
        schema = infer_schema(SEED_ITEMS)
        item = {"name": "test"}  # missing age, score, active, tags
        errors = validate_against_schema(item, schema)
        assert len(errors) >= 4  # at least 4 missing fields

    def test_wrong_type(self):
        schema = infer_schema(SEED_ITEMS)
        item = {"name": 123, "age": 30, "score": 0.5, "active": True, "tags": []}
        errors = validate_against_schema(item, schema)
        assert any("expected string" in e for e in errors)

    def test_bool_not_treated_as_int(self):
        schema = infer_schema(SEED_ITEMS)
        item = {"name": "x", "age": True, "score": 0.5, "active": True, "tags": []}
        errors = validate_against_schema(item, schema)
        assert any("age" in e for e in errors)

    def test_int_not_treated_as_bool(self):
        schema = infer_schema(SEED_ITEMS)
        item = {"name": "x", "age": 30, "score": 0.5, "active": 1, "tags": []}
        errors = validate_against_schema(item, schema)
        assert any("active" in e for e in errors)

    def test_out_of_range(self):
        schema = infer_schema(SEED_ITEMS)
        item = {"name": "x", "age": 999, "score": 0.5, "active": True, "tags": []}
        errors = validate_against_schema(item, schema)
        assert any("above max" in e for e in errors)

    def test_below_range(self):
        schema = infer_schema(SEED_ITEMS)
        item = {"name": "x", "age": -10, "score": 0.5, "active": True, "tags": []}
        errors = validate_against_schema(item, schema)
        assert any("below min" in e for e in errors)

    def test_optional_missing_ok(self):
        items = [{"a": 1, "b": 2}, {"a": 3}]
        schema = infer_schema(items)
        errors = validate_against_schema({"a": 5}, schema)
        assert not any("b" in e and "Missing" in e for e in errors)

    def test_choices_violation(self):
        items = [{"color": "red"}, {"color": "blue"}]
        schema = infer_schema(items)
        errors = validate_against_schema({"color": "green"}, schema)
        assert any("not in choices" in e for e in errors)

    def test_choices_valid(self):
        items = [{"color": "red"}, {"color": "blue"}]
        schema = infer_schema(items)
        errors = validate_against_schema({"color": "red"}, schema)
        assert errors == []

    def test_generated_items_validate(self):
        schema = infer_schema(SEED_ITEMS)
        rng = random.Random(42)
        for _ in range(20):
            si = generate_structured_input(schema, rng=rng)
            errors = validate_against_schema(si.data, schema)
            assert errors == [], f"Generated item failed validation: {errors}"


# ---------------------------------------------------------------------------
# format_schema_summary
# ---------------------------------------------------------------------------


class TestFormatSchemaSummary:
    def test_basic(self):
        schema = infer_schema(SEED_ITEMS)
        summary = format_schema_summary(schema)
        assert "Schema" in summary
        assert "inferred from 3 items" in summary
        assert "name" in summary
        assert "age" in summary

    def test_shows_range(self):
        schema = infer_schema(SEED_ITEMS)
        summary = format_schema_summary(schema)
        assert "range:" in summary

    def test_shows_choices(self):
        items = [{"color": "red"}, {"color": "blue"}]
        schema = infer_schema(items)
        summary = format_schema_summary(schema)
        assert "choices:" in summary

    def test_empty_schema(self):
        schema = InputSchema()
        summary = format_schema_summary(schema)
        assert "inferred from 0 items" in summary


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_infer_generate_validate_cycle(self):
        schema = infer_schema(SEED_ITEMS)
        batch = generate_batch_structured(schema, 20, rng=random.Random(42))
        for si in batch:
            errors = validate_against_schema(si.data, schema)
            assert errors == []

    def test_mutation_suite_validates(self):
        schema = infer_schema(SEED_ITEMS)
        base = SEED_ITEMS[0]
        suite = generate_mutation_suite(base, schema)
        for si in suite:
            errors = validate_against_schema(si.data, schema)
            assert errors == [], f"Mutation {si.mutated_field} failed: {errors}"

    def test_roundtrip_schema(self):
        schema = infer_schema(SEED_ITEMS)
        d = schema.as_dict()
        schema2 = InputSchema.from_dict(d)
        # Generate from restored schema
        si = generate_structured_input(schema2, rng=random.Random(42))
        errors = validate_against_schema(si.data, schema2)
        assert errors == []
