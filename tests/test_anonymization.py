from __future__ import annotations

import copy
from datetime import datetime, timezone

import pytest

from evalyn_sdk.models import Span
from evalyn_sdk.trace.anonymization import (
    AnonymizationConfig,
    AnonymizationResult,
    anonymize_span,
    anonymize_spans,
    anonymize_text,
    export_anonymized,
)


def _span(sid="s1", parent_id=None, **attrs):
    return Span(
        id=sid,
        name="test",
        span_type="llm_call",
        parent_id=parent_id,
        start_time=datetime.now(timezone.utc),
        attributes=attrs,
    )


# --- anonymize_text ---


class TestAnonymizeText:
    def test_produces_different_output(self):
        result = anonymize_text("hello world")
        assert result != "hello world"

    def test_deterministic_same_salt(self):
        a = anonymize_text("hello", salt="x")
        b = anonymize_text("hello", salt="x")
        assert a == b

    def test_different_salt_different_output(self):
        a = anonymize_text("hello", salt="x")
        b = anonymize_text("hello", salt="y")
        assert a != b

    def test_empty_text_returns_empty(self):
        assert anonymize_text("") == ""

    def test_format_contains_anon_prefix(self):
        result = anonymize_text("data")
        assert result.startswith("[anon-")
        assert result.endswith("chars]")

    def test_format_contains_length(self):
        text = "abcdef"
        result = anonymize_text(text)
        assert f"{len(text)}chars" in result

    def test_different_texts_different_output(self):
        a = anonymize_text("alpha")
        b = anonymize_text("beta")
        assert a != b


# --- AnonymizationConfig ---


class TestAnonymizationConfig:
    def test_defaults(self):
        cfg = AnonymizationConfig()
        assert cfg.anonymize_inputs is True
        assert cfg.anonymize_outputs is True
        assert cfg.preserve_structure is True
        assert cfg.salt == ""

    def test_as_dict_from_dict_roundtrip(self):
        cfg = AnonymizationConfig(
            anonymize_inputs=False,
            anonymize_outputs=True,
            preserve_structure=False,
            salt="secret",
        )
        d = cfg.as_dict()
        restored = AnonymizationConfig.from_dict(d)
        assert restored.anonymize_inputs == cfg.anonymize_inputs
        assert restored.anonymize_outputs == cfg.anonymize_outputs
        assert restored.preserve_structure == cfg.preserve_structure
        assert restored.salt == cfg.salt

    def test_from_dict_defaults(self):
        cfg = AnonymizationConfig.from_dict({})
        assert cfg.anonymize_inputs is True
        assert cfg.salt == ""


# --- anonymize_span ---


class TestAnonymizeSpan:
    def test_creates_new_span(self):
        original = _span("s1", input="secret")
        result = anonymize_span(original, AnonymizationConfig())
        assert result is not original

    def test_original_unchanged(self):
        original = _span("s1", input="secret")
        original_copy = copy.deepcopy(original)
        anonymize_span(original, AnonymizationConfig())
        assert original.id == original_copy.id
        assert original.attributes == original_copy.attributes

    def test_span_id_changed(self):
        original = _span("s1", input="data")
        result = anonymize_span(original, AnonymizationConfig())
        assert result.id != "s1"

    def test_input_anonymized(self):
        original = _span("s1", input="secret data")
        result = anonymize_span(original, AnonymizationConfig())
        assert result.attributes["input"] != "secret data"
        assert "[anon-" in result.attributes["input"]

    def test_output_anonymized(self):
        original = _span("s1", output="response text")
        result = anonymize_span(original, AnonymizationConfig())
        assert result.attributes["output"] != "response text"

    def test_inputs_key_anonymized(self):
        original = _span("s1", inputs="multi input")
        result = anonymize_span(original, AnonymizationConfig())
        assert result.attributes["inputs"] != "multi input"

    def test_outputs_key_anonymized(self):
        original = _span("s1", outputs="multi output")
        result = anonymize_span(original, AnonymizationConfig())
        assert result.attributes["outputs"] != "multi output"

    def test_non_input_output_preserved(self):
        original = _span("s1", model="gpt-4", temperature=0.7)
        result = anonymize_span(original, AnonymizationConfig())
        assert result.attributes["model"] == "gpt-4"
        assert result.attributes["temperature"] == 0.7

    def test_preserve_structure_dict_keys_kept(self):
        original = _span("s1", input={"query": "secret", "mode": "fast"})
        result = anonymize_span(original, AnonymizationConfig(preserve_structure=True))
        anon_input = result.attributes["input"]
        assert isinstance(anon_input, dict)
        assert "query" in anon_input
        assert "mode" in anon_input
        assert anon_input["query"] != "secret"
        assert anon_input["mode"] != "fast"

    def test_skip_input_anonymization(self):
        original = _span("s1", input="keep me")
        cfg = AnonymizationConfig(anonymize_inputs=False)
        result = anonymize_span(original, cfg)
        assert result.attributes["input"] == "keep me"

    def test_skip_output_anonymization(self):
        original = _span("s1", output="keep me")
        cfg = AnonymizationConfig(anonymize_outputs=False)
        result = anonymize_span(original, cfg)
        assert result.attributes["output"] == "keep me"

    def test_non_string_attribute_values_preserved(self):
        original = _span("s1", input=42, output=True)
        result = anonymize_span(original, AnonymizationConfig())
        # int/bool are not strings, so not anonymized
        assert result.attributes["input"] == 42
        assert result.attributes["output"] is True

    def test_parent_id_remapped(self):
        original = _span("child", parent_id="parent")
        result = anonymize_span(original, AnonymizationConfig())
        assert result.parent_id is not None
        assert result.parent_id != "parent"

    def test_none_parent_stays_none(self):
        original = _span("root", parent_id=None)
        result = anonymize_span(original, AnonymizationConfig())
        assert result.parent_id is None


# --- anonymize_spans ---


class TestAnonymizeSpans:
    def test_remaps_parent_id(self):
        parent = _span("p1")
        child = _span("c1", parent_id="p1", input="data")
        anonymized, result = anonymize_spans([parent, child], AnonymizationConfig())
        # Child's parent_id should equal parent's new id
        assert anonymized[1].parent_id == anonymized[0].id

    def test_maintains_hierarchy(self):
        root = _span("root")
        mid = _span("mid", parent_id="root")
        leaf = _span("leaf", parent_id="mid")
        anonymized, _ = anonymize_spans([root, mid, leaf], AnonymizationConfig())
        assert anonymized[1].parent_id == anonymized[0].id
        assert anonymized[2].parent_id == anonymized[1].id

    def test_result_counts(self):
        spans = [
            _span("s1", input="a", output="b"),
            _span("s2", input="c"),
        ]
        _, result = anonymize_spans(spans, AnonymizationConfig())
        assert result.spans_processed == 2
        assert result.fields_anonymized == 3  # input+output for s1, input for s2
        assert len(result.original_span_ids) == 2
        assert len(result.anonymized_span_ids) == 2

    def test_empty_list(self):
        anonymized, result = anonymize_spans([], AnonymizationConfig())
        assert anonymized == []
        assert result.spans_processed == 0
        assert result.fields_anonymized == 0


# --- AnonymizationResult ---


class TestAnonymizationResult:
    def test_as_dict(self):
        r = AnonymizationResult(
            spans_processed=3,
            fields_anonymized=5,
            original_span_ids=["a", "b", "c"],
            anonymized_span_ids=["x", "y", "z"],
        )
        d = r.as_dict()
        assert d["spans_processed"] == 3
        assert d["fields_anonymized"] == 5
        assert d["original_span_ids"] == ["a", "b", "c"]
        assert d["anonymized_span_ids"] == ["x", "y", "z"]


# --- export_anonymized ---


class TestExportAnonymized:
    def test_returns_dicts(self):
        spans = [_span("s1", input="hello")]
        result = export_anonymized(spans)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)

    def test_uses_default_config(self):
        spans = [_span("s1", input="secret")]
        result = export_anonymized(spans)
        assert result[0]["attributes"]["input"] != "secret"

    def test_custom_config(self):
        spans = [_span("s1", input="keep", output="hide")]
        cfg = AnonymizationConfig(anonymize_inputs=False, anonymize_outputs=True)
        result = export_anonymized(spans, cfg)
        assert result[0]["attributes"]["input"] == "keep"
        assert result[0]["attributes"]["output"] != "hide"

    def test_list_values_in_input(self):
        spans = [_span("s1", input=["a", "b"])]
        result = export_anonymized(spans)
        anon = result[0]["attributes"]["input"]
        assert isinstance(anon, list)
        assert anon[0] != "a"
        assert anon[1] != "b"

    def test_nested_dict_preserve_structure(self):
        spans = [_span("s1", input={"outer": {"inner": "val"}})]
        result = export_anonymized(spans, AnonymizationConfig(preserve_structure=True))
        anon = result[0]["attributes"]["input"]
        assert "outer" in anon
        assert "inner" in anon["outer"]
        assert anon["outer"]["inner"] != "val"
