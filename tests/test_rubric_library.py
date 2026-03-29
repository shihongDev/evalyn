"""Tests for the community rubric library module."""

from __future__ import annotations

import json

from evalyn_sdk.rubric_library import (
    PortableRubric,
    RubricMetadata,
    export_rubric,
    export_to_json_string,
    export_to_yaml_string,
    format_rubric_detail,
    format_rubric_summary,
    import_from_json_string,
    import_from_yaml_string,
    merge_rubrics,
    validate_rubric,
)


# -- Helpers ------------------------------------------------------------------


def _make_metadata(**overrides) -> RubricMetadata:
    defaults = {
        "author": "test-author",
        "version": "1.0.0",
        "tested_on": "my-dataset",
        "accuracy": 0.85,
        "created_at": "2026-01-01T00:00:00Z",
        "tags": ["quality", "llm"],
    }
    defaults.update(overrides)
    return RubricMetadata(**defaults)


def _make_rubric(**overrides) -> PortableRubric:
    defaults = {
        "metric_id": "helpfulness",
        "metric_type": "llm_judge",
        "description": "Measures helpfulness",
        "category": "quality",
        "scope": "item",
        "prompt": "Rate the helpfulness of the response.",
        "rubric": {"1": "Not helpful", "3": "Somewhat helpful", "5": "Very helpful"},
        "metadata": _make_metadata(),
    }
    defaults.update(overrides)
    return PortableRubric(**defaults)


# -- RubricMetadata tests -----------------------------------------------------


class TestRubricMetadata:
    def test_as_dict(self):
        meta = _make_metadata()
        d = meta.as_dict()
        assert d["author"] == "test-author"
        assert d["tags"] == ["quality", "llm"]
        assert d["accuracy"] == 0.85

    def test_from_dict(self):
        meta = RubricMetadata.from_dict({"author": "alice", "version": "2.0"})
        assert meta.author == "alice"
        assert meta.version == "2.0"
        assert meta.tags == []

    def test_from_dict_defaults(self):
        meta = RubricMetadata.from_dict({})
        assert meta.author == ""
        assert meta.version == "0.0.0"
        assert meta.accuracy == 0.0

    def test_roundtrip(self):
        meta = _make_metadata()
        restored = RubricMetadata.from_dict(meta.as_dict())
        assert restored.author == meta.author
        assert restored.tags == meta.tags
        assert restored.accuracy == meta.accuracy


# -- PortableRubric tests -----------------------------------------------------


class TestPortableRubric:
    def test_as_dict(self):
        rubric = _make_rubric()
        d = rubric.as_dict()
        assert d["metric_id"] == "helpfulness"
        assert "metadata" in d
        assert d["metadata"]["author"] == "test-author"

    def test_from_dict(self):
        rubric = PortableRubric.from_dict({"metric_id": "foo", "prompt": "bar"})
        assert rubric.metric_id == "foo"
        assert rubric.prompt == "bar"
        assert rubric.rubric == {}

    def test_from_dict_defaults(self):
        rubric = PortableRubric.from_dict({})
        assert rubric.metric_id == ""
        assert rubric.metric_type == ""

    def test_roundtrip(self):
        rubric = _make_rubric()
        restored = PortableRubric.from_dict(rubric.as_dict())
        assert restored.metric_id == rubric.metric_id
        assert restored.rubric == rubric.rubric
        assert restored.metadata.author == rubric.metadata.author


# -- export_rubric tests ------------------------------------------------------


class TestExportRubric:
    def test_basic_export(self):
        rd = {
            "metric_id": "accuracy",
            "metric_type": "exact_match",
            "description": "Exact match",
            "category": "correctness",
            "scope": "item",
            "prompt": "Is the answer correct?",
            "rubric": {"0": "Wrong", "1": "Correct"},
        }
        meta = _make_metadata()
        result = export_rubric(rd, meta)
        assert result.metric_id == "accuracy"
        assert result.rubric == {"0": "Wrong", "1": "Correct"}
        assert result.metadata is meta

    def test_export_with_id_fallback(self):
        rd = {"id": "fallback_id", "type": "custom"}
        result = export_rubric(rd, _make_metadata())
        assert result.metric_id == "fallback_id"
        assert result.metric_type == "custom"

    def test_export_empty_dict(self):
        result = export_rubric({}, _make_metadata())
        assert result.metric_id == ""
        assert result.rubric == {}


# -- JSON serialization tests --------------------------------------------------


class TestJsonSerialization:
    def test_export_json(self):
        rubric = _make_rubric()
        result = export_to_json_string(rubric)
        parsed = json.loads(result)
        assert parsed["metric_id"] == "helpfulness"

    def test_import_json(self):
        rubric = _make_rubric()
        json_str = export_to_json_string(rubric)
        restored = import_from_json_string(json_str)
        assert restored.metric_id == "helpfulness"
        assert restored.rubric == rubric.rubric

    def test_json_roundtrip(self):
        rubric = _make_rubric()
        restored = import_from_json_string(export_to_json_string(rubric))
        assert restored.as_dict() == rubric.as_dict()

    def test_json_preserves_metadata(self):
        rubric = _make_rubric()
        restored = import_from_json_string(export_to_json_string(rubric))
        assert restored.metadata.tags == ["quality", "llm"]
        assert restored.metadata.accuracy == 0.85


# -- YAML-like serialization tests --------------------------------------------


class TestYamlSerialization:
    def test_export_yaml(self):
        rubric = _make_rubric()
        result = export_to_yaml_string(rubric)
        assert "metric_id: helpfulness" in result
        assert "---" in result

    def test_import_yaml(self):
        rubric = _make_rubric()
        yaml_str = export_to_yaml_string(rubric)
        restored = import_from_yaml_string(yaml_str)
        assert restored.metric_id == "helpfulness"

    def test_yaml_roundtrip(self):
        rubric = _make_rubric()
        yaml_str = export_to_yaml_string(rubric)
        restored = import_from_yaml_string(yaml_str)
        assert restored.metric_id == rubric.metric_id
        assert restored.metric_type == rubric.metric_type
        assert restored.description == rubric.description
        assert restored.category == rubric.category
        assert restored.scope == rubric.scope

    def test_yaml_preserves_rubric_levels(self):
        rubric = _make_rubric()
        restored = import_from_yaml_string(export_to_yaml_string(rubric))
        assert restored.rubric == rubric.rubric

    def test_yaml_preserves_metadata(self):
        rubric = _make_rubric()
        restored = import_from_yaml_string(export_to_yaml_string(rubric))
        assert restored.metadata.author == "test-author"
        assert restored.metadata.version == "1.0.0"
        assert restored.metadata.accuracy == 0.85

    def test_yaml_preserves_tags(self):
        rubric = _make_rubric()
        restored = import_from_yaml_string(export_to_yaml_string(rubric))
        assert restored.metadata.tags == ["quality", "llm"]

    def test_yaml_multiline_prompt(self):
        rubric = _make_rubric(prompt="Line 1\nLine 2\nLine 3")
        restored = import_from_yaml_string(export_to_yaml_string(rubric))
        assert restored.prompt == "Line 1\nLine 2\nLine 3"

    def test_yaml_contains_separator(self):
        result = export_to_yaml_string(_make_rubric())
        assert result.count("---") >= 4


# -- validate_rubric tests ----------------------------------------------------


class TestValidateRubric:
    def test_valid_rubric(self):
        errors = validate_rubric(_make_rubric())
        assert errors == []

    def test_empty_metric_id(self):
        rubric = _make_rubric(metric_id="")
        errors = validate_rubric(rubric)
        assert "metric_id is empty" in errors

    def test_missing_prompt(self):
        rubric = _make_rubric(prompt="")
        errors = validate_rubric(rubric)
        assert "prompt is missing" in errors

    def test_no_rubric_levels(self):
        rubric = _make_rubric(rubric={})
        errors = validate_rubric(rubric)
        assert "no rubric levels defined" in errors

    def test_empty_metric_type(self):
        rubric = _make_rubric(metric_type="")
        errors = validate_rubric(rubric)
        assert "metric_type is empty" in errors

    def test_accuracy_too_high(self):
        rubric = _make_rubric(metadata=_make_metadata(accuracy=1.5))
        errors = validate_rubric(rubric)
        assert "accuracy must be between 0.0 and 1.0" in errors

    def test_accuracy_negative(self):
        rubric = _make_rubric(metadata=_make_metadata(accuracy=-0.1))
        errors = validate_rubric(rubric)
        assert "accuracy must be between 0.0 and 1.0" in errors

    def test_missing_author(self):
        rubric = _make_rubric(metadata=_make_metadata(author=""))
        errors = validate_rubric(rubric)
        assert "metadata author is missing" in errors

    def test_multiple_errors(self):
        rubric = _make_rubric(
            metric_id="",
            prompt="",
            rubric={},
        )
        errors = validate_rubric(rubric)
        assert len(errors) >= 3


# -- merge_rubrics tests ------------------------------------------------------


class TestMergeRubrics:
    def test_no_conflict(self):
        existing = [_make_rubric(metric_id="a")]
        incoming = [_make_rubric(metric_id="b")]
        merged, actions = merge_rubrics(existing, incoming)
        assert len(merged) == 2
        assert "added b" in actions

    def test_skip_strategy(self):
        existing = [_make_rubric(metric_id="a")]
        incoming = [_make_rubric(metric_id="a")]
        merged, actions = merge_rubrics(existing, incoming, strategy="skip")
        assert len(merged) == 1
        assert any("skipped" in a for a in actions)

    def test_overwrite_strategy(self):
        existing = [_make_rubric(metric_id="a", description="old")]
        incoming = [_make_rubric(metric_id="a", description="new")]
        merged, actions = merge_rubrics(existing, incoming, strategy="overwrite")
        assert len(merged) == 1
        assert merged[0].description == "new"
        assert any("overwrote" in a for a in actions)

    def test_rename_strategy(self):
        existing = [_make_rubric(metric_id="a")]
        incoming = [_make_rubric(metric_id="a")]
        merged, actions = merge_rubrics(existing, incoming, strategy="rename")
        assert len(merged) == 2
        ids = {r.metric_id for r in merged}
        assert "a" in ids
        assert "a_v1" in ids
        assert any("renamed" in a for a in actions)

    def test_rename_increments_suffix(self):
        existing = [
            _make_rubric(metric_id="a"),
            _make_rubric(metric_id="a_v1"),
        ]
        incoming = [_make_rubric(metric_id="a")]
        merged, actions = merge_rubrics(existing, incoming, strategy="rename")
        ids = {r.metric_id for r in merged}
        assert "a_v2" in ids

    def test_merge_empty_incoming(self):
        existing = [_make_rubric(metric_id="a")]
        merged, actions = merge_rubrics(existing, [])
        assert len(merged) == 1
        assert actions == []

    def test_merge_empty_existing(self):
        incoming = [_make_rubric(metric_id="a")]
        merged, actions = merge_rubrics([], incoming)
        assert len(merged) == 1

    def test_merge_multiple(self):
        existing = [_make_rubric(metric_id="a")]
        incoming = [
            _make_rubric(metric_id="b"),
            _make_rubric(metric_id="c"),
        ]
        merged, actions = merge_rubrics(existing, incoming)
        assert len(merged) == 3


# -- format tests --------------------------------------------------------------


class TestFormatRubric:
    def test_summary_contains_id(self):
        result = format_rubric_summary(_make_rubric())
        assert "helpfulness" in result

    def test_summary_contains_type(self):
        result = format_rubric_summary(_make_rubric())
        assert "llm_judge" in result

    def test_summary_contains_accuracy(self):
        result = format_rubric_summary(_make_rubric())
        assert "85%" in result

    def test_summary_contains_level_count(self):
        result = format_rubric_summary(_make_rubric())
        assert "3 levels" in result

    def test_detail_contains_all_fields(self):
        result = format_rubric_detail(_make_rubric())
        assert "helpfulness" in result
        assert "llm_judge" in result
        assert "Measures helpfulness" in result
        assert "quality" in result
        assert "item" in result

    def test_detail_contains_rubric_levels(self):
        result = format_rubric_detail(_make_rubric())
        assert "Not helpful" in result
        assert "Very helpful" in result

    def test_detail_contains_metadata(self):
        result = format_rubric_detail(_make_rubric())
        assert "test-author" in result
        assert "1.0.0" in result
        assert "my-dataset" in result
