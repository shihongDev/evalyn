"""Tests for openai_evals_export module."""

from __future__ import annotations

import json

from evalyn_sdk.openai_evals_export import (
    OpenAIEvalSample,
    OpenAIEvalSet,
    convert_evalyn_to_openai,
    export_as_json,
    export_as_jsonl,
    format_export_summary,
    import_from_openai_jsonl,
)


class TestOpenAIEvalSample:
    def test_as_dict(self):
        s = OpenAIEvalSample(
            input_messages=[{"role": "user", "content": "hello"}],
            ideal="hi",
            output="hey",
            metrics={"accuracy": 0.9},
        )
        d = s.as_dict()
        assert d["input"] == [{"role": "user", "content": "hello"}]
        assert d["ideal"] == "hi"
        assert d["metrics"]["accuracy"] == 0.9

    def test_from_dict_roundtrip(self):
        s = OpenAIEvalSample(
            input_messages=[{"role": "user", "content": "test"}],
            ideal="expected",
            output="actual",
        )
        restored = OpenAIEvalSample.from_dict(s.as_dict())
        assert restored.ideal == "expected"
        assert restored.output == "actual"

    def test_from_dict_defaults(self):
        s = OpenAIEvalSample.from_dict({})
        assert s.input_messages == []
        assert s.ideal == ""
        assert s.metrics == {}

    def test_empty_metrics(self):
        s = OpenAIEvalSample(input_messages=[], ideal="x")
        assert s.metrics == {}


class TestOpenAIEvalSet:
    def test_as_dict(self):
        sample = OpenAIEvalSample(input_messages=[{"role": "user", "content": "q"}])
        es = OpenAIEvalSet(eval_name="test", samples=[sample], created_at="2026-01-01")
        d = es.as_dict()
        assert d["eval_name"] == "test"
        assert len(d["samples"]) == 1

    def test_from_dict_roundtrip(self):
        sample = OpenAIEvalSample(input_messages=[{"role": "user", "content": "q"}])
        es = OpenAIEvalSet(eval_name="test", samples=[sample], created_at="now")
        restored = OpenAIEvalSet.from_dict(es.as_dict())
        assert restored.eval_name == "test"
        assert len(restored.samples) == 1

    def test_empty_samples(self):
        es = OpenAIEvalSet(eval_name="empty", samples=[])
        assert len(es.samples) == 0


class TestConvertEvalynToOpenai:
    def test_basic(self):
        items = [{"input": "What is 2+2?", "output": "4", "expected_output": "4"}]
        result = convert_evalyn_to_openai(items)
        assert len(result.samples) == 1
        assert result.samples[0].input_messages[0]["content"] == "What is 2+2?"
        assert result.samples[0].output == "4"
        assert result.samples[0].ideal == "4"

    def test_with_results(self):
        items = [{"input": "q", "id": "i1"}]
        results = [{"item_id": "i1", "metric_id": "accuracy", "score": 0.95}]
        es = convert_evalyn_to_openai(items, results)
        assert es.samples[0].metrics["accuracy"] == 0.95

    def test_custom_fields(self):
        items = [{"text": "hello", "response": "hi"}]
        es = convert_evalyn_to_openai(items, input_field="text", output_field="response")
        assert es.samples[0].input_messages[0]["content"] == "hello"
        assert es.samples[0].output == "hi"

    def test_empty_items(self):
        es = convert_evalyn_to_openai([])
        assert len(es.samples) == 0

    def test_multiple_items(self):
        items = [{"input": f"q{i}"} for i in range(5)]
        es = convert_evalyn_to_openai(items)
        assert len(es.samples) == 5

    def test_missing_fields(self):
        items = [{"input": "q"}]
        es = convert_evalyn_to_openai(items)
        assert es.samples[0].output == ""
        assert es.samples[0].ideal == ""

    def test_created_at_populated(self):
        es = convert_evalyn_to_openai([{"input": "q"}])
        assert es.created_at != ""

    def test_eval_name(self):
        es = convert_evalyn_to_openai([{"input": "q"}])
        assert es.eval_name == "evalyn_export"

    def test_multiple_results_per_item(self):
        items = [{"input": "q", "id": "i1"}]
        results = [
            {"item_id": "i1", "metric_id": "a", "score": 0.8},
            {"item_id": "i1", "metric_id": "b", "score": 0.6},
        ]
        es = convert_evalyn_to_openai(items, results)
        assert es.samples[0].metrics["a"] == 0.8
        assert es.samples[0].metrics["b"] == 0.6


class TestExportAsJsonl:
    def test_basic(self):
        sample = OpenAIEvalSample(input_messages=[{"role": "user", "content": "hi"}])
        es = OpenAIEvalSet(eval_name="t", samples=[sample])
        jsonl = export_as_jsonl(es)
        parsed = json.loads(jsonl.strip())
        assert "input" in parsed

    def test_multiple_lines(self):
        samples = [
            OpenAIEvalSample(input_messages=[{"role": "user", "content": f"q{i}"}])
            for i in range(3)
        ]
        es = OpenAIEvalSet(eval_name="t", samples=samples)
        lines = export_as_jsonl(es).strip().split("\n")
        assert len(lines) == 3

    def test_empty(self):
        es = OpenAIEvalSet(eval_name="t", samples=[])
        assert export_as_jsonl(es) == ""

    def test_valid_json_per_line(self):
        samples = [
            OpenAIEvalSample(input_messages=[{"role": "user", "content": "q"}])
        ]
        es = OpenAIEvalSet(eval_name="t", samples=samples)
        for line in export_as_jsonl(es).strip().split("\n"):
            json.loads(line)


class TestExportAsJson:
    def test_valid_json(self):
        es = OpenAIEvalSet(eval_name="t", samples=[], created_at="now")
        parsed = json.loads(export_as_json(es))
        assert parsed["eval_name"] == "t"

    def test_pretty_printed(self):
        es = OpenAIEvalSet(eval_name="t", samples=[])
        output = export_as_json(es)
        assert "\n" in output


class TestImportFromOpenaiJsonl:
    def test_roundtrip(self):
        samples = [
            OpenAIEvalSample(
                input_messages=[{"role": "user", "content": "q"}],
                ideal="a",
                output="b",
                metrics={"x": 0.5},
            )
        ]
        es = OpenAIEvalSet(eval_name="t", samples=samples)
        jsonl = export_as_jsonl(es)
        imported = import_from_openai_jsonl(jsonl)
        assert len(imported.samples) == 1
        assert imported.samples[0].ideal == "a"
        assert imported.samples[0].metrics["x"] == 0.5

    def test_empty(self):
        imported = import_from_openai_jsonl("")
        assert len(imported.samples) == 0

    def test_blank_lines_skipped(self):
        jsonl = '{"input":[]}\n\n{"input":[]}\n'
        imported = import_from_openai_jsonl(jsonl)
        assert len(imported.samples) == 2


class TestFormatExportSummary:
    def test_basic(self):
        samples = [
            OpenAIEvalSample(
                input_messages=[{"role": "user", "content": "q"}],
                ideal="a",
                output="b",
                metrics={"x": 1.0},
            )
        ]
        es = OpenAIEvalSet(eval_name="test_export", samples=samples, created_at="now")
        summary = format_export_summary(es)
        assert "test_export" in summary
        assert "Samples: 1" in summary
        assert "With ideal: 1" in summary
        assert "With output: 1" in summary
        assert "With metrics: 1" in summary

    def test_empty_set(self):
        es = OpenAIEvalSet(eval_name="empty", samples=[], created_at="now")
        summary = format_export_summary(es)
        assert "Samples: 0" in summary

    def test_partial_data(self):
        samples = [
            OpenAIEvalSample(input_messages=[]),
            OpenAIEvalSample(input_messages=[], ideal="x"),
        ]
        es = OpenAIEvalSet(eval_name="t", samples=samples, created_at="now")
        summary = format_export_summary(es)
        assert "Samples: 2" in summary
        assert "With ideal: 1" in summary
