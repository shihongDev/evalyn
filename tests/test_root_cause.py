"""Tests for analysis.root_cause module."""

from __future__ import annotations

import pytest

from evalyn_sdk.analysis.root_cause import (
    FailurePattern,
    RootCauseReport,
    analyze_root_causes,
    detect_category_pattern,
    detect_input_complexity_pattern,
    detect_metric_concentration,
    detect_output_length_pattern,
    suggest_fixes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _failure(
    item_id: str = "item-1",
    output: str = "some output",
    score: float = 0.0,
    metric_id: str = "accuracy",
    input: str = "question",
    category: str = "",
) -> dict:
    d = {
        "item_id": item_id,
        "output": output,
        "score": score,
        "metric_id": metric_id,
        "input": input,
    }
    if category:
        d["category"] = category
    return d


# ---------------------------------------------------------------------------
# detect_output_length_pattern
# ---------------------------------------------------------------------------


class TestDetectOutputLengthPattern:
    def test_short_outputs_detected(self):
        failures = [
            _failure(item_id=f"i{i}", output="ok") for i in range(8)
        ] + [
            _failure(item_id="i8", output="x" * 100),
            _failure(item_id="i9", output="y" * 100),
        ]
        result = detect_output_length_pattern(failures)
        assert result is not None
        assert result.pattern_name == "short_output"
        assert result.confidence >= 0.6
        assert len(result.affected_items) == 8

    def test_long_outputs_detected(self):
        failures = [
            _failure(item_id=f"i{i}", output="x" * 600) for i in range(8)
        ] + [
            _failure(item_id="i8", output="short"),
            _failure(item_id="i9", output="also short"),
        ]
        result = detect_output_length_pattern(failures)
        assert result is not None
        assert result.pattern_name == "long_output"

    def test_no_pattern(self):
        failures = [
            _failure(item_id=f"i{i}", output="x" * 50) for i in range(10)
        ]
        result = detect_output_length_pattern(failures)
        assert result is None

    def test_empty_failures(self):
        result = detect_output_length_pattern([])
        assert result is None

    def test_single_failure_short(self):
        failures = [_failure(output="hi")]
        result = detect_output_length_pattern(failures)
        assert result is not None
        assert result.pattern_name == "short_output"

    def test_none_output_treated_as_empty(self):
        failures = [_failure(output=None) for _ in range(5)]
        result = detect_output_length_pattern(failures)
        assert result is not None
        assert result.pattern_name == "short_output"


# ---------------------------------------------------------------------------
# detect_input_complexity_pattern
# ---------------------------------------------------------------------------


class TestDetectInputComplexityPattern:
    def test_complex_inputs_detected(self):
        long_input = "This is a complex question. It has multiple parts. Please answer all of them."
        failures = [
            _failure(item_id=f"i{i}", input=long_input) for i in range(8)
        ] + [
            _failure(item_id="i8", input="short"),
            _failure(item_id="i9", input="brief"),
        ]
        result = detect_input_complexity_pattern(failures)
        assert result is not None
        assert result.pattern_name == "complex_input"
        assert result.confidence >= 0.6

    def test_simple_inputs_no_pattern(self):
        failures = [
            _failure(item_id=f"i{i}", input="hi") for i in range(10)
        ]
        result = detect_input_complexity_pattern(failures)
        assert result is None

    def test_empty_failures(self):
        result = detect_input_complexity_pattern([])
        assert result is None

    def test_long_char_count_triggers(self):
        long_text = "a" * 250
        failures = [_failure(input=long_text) for _ in range(5)]
        result = detect_input_complexity_pattern(failures)
        assert result is not None


# ---------------------------------------------------------------------------
# detect_metric_concentration
# ---------------------------------------------------------------------------


class TestDetectMetricConcentration:
    def test_one_metric_dominates(self):
        failures = [
            _failure(item_id=f"i{i}", metric_id="safety") for i in range(7)
        ] + [
            _failure(item_id="i7", metric_id="accuracy"),
            _failure(item_id="i8", metric_id="fluency"),
            _failure(item_id="i9", metric_id="accuracy"),
        ]
        result = detect_metric_concentration(failures)
        assert result is not None
        assert result.pattern_name == "metric_concentration"
        assert "safety" in result.description

    def test_even_distribution_no_pattern(self):
        failures = [
            _failure(item_id=f"i{i}", metric_id=f"metric-{i % 5}")
            for i in range(10)
        ]
        result = detect_metric_concentration(failures)
        assert result is None

    def test_empty_failures(self):
        result = detect_metric_concentration([])
        assert result is None


# ---------------------------------------------------------------------------
# detect_category_pattern
# ---------------------------------------------------------------------------


class TestDetectCategoryPattern:
    def test_one_category_dominates(self):
        failures = [
            _failure(item_id=f"i{i}", category="math") for i in range(7)
        ] + [
            _failure(item_id="i7", category="science"),
            _failure(item_id="i8", category="history"),
            _failure(item_id="i9", category="science"),
        ]
        result = detect_category_pattern(failures)
        assert result is not None
        assert result.pattern_name == "category_concentration"
        assert "math" in result.description

    def test_no_categories(self):
        failures = [_failure(item_id=f"i{i}") for i in range(5)]
        result = detect_category_pattern(failures)
        assert result is None

    def test_even_categories_no_pattern(self):
        cats = ["math", "science", "history", "english", "art"]
        failures = [
            _failure(item_id=f"i{i}", category=cats[i % 5])
            for i in range(10)
        ]
        result = detect_category_pattern(failures)
        assert result is None

    def test_empty_failures(self):
        result = detect_category_pattern([])
        assert result is None


# ---------------------------------------------------------------------------
# analyze_root_causes
# ---------------------------------------------------------------------------


class TestAnalyzeRootCauses:
    def test_full_pipeline(self):
        failures = [
            _failure(item_id=f"i{i}", output="hi", metric_id="safety")
            for i in range(10)
        ]
        report = analyze_root_causes(failures)
        assert report.total_failures == 10
        assert len(report.patterns) > 0
        assert report.primary_cause != ""

    def test_primary_cause_is_highest_confidence(self):
        # Short outputs (all match) + metric concentration (all match)
        failures = [
            _failure(item_id=f"i{i}", output="hi", metric_id="safety")
            for i in range(10)
        ]
        report = analyze_root_causes(failures)
        confidences = [p.confidence for p in report.patterns]
        assert confidences == sorted(confidences, reverse=True)
        assert report.primary_cause == report.patterns[0].pattern_name

    def test_empty_failures(self):
        report = analyze_root_causes([])
        assert report.total_failures == 0
        assert report.patterns == []
        assert report.primary_cause == ""

    def test_single_failure(self):
        report = analyze_root_causes([_failure(output="x")])
        assert report.total_failures == 1

    def test_secondary_causes_populated(self):
        failures = [
            _failure(item_id=f"i{i}", output="hi", metric_id="safety")
            for i in range(10)
        ]
        report = analyze_root_causes(failures)
        if len(report.patterns) > 1:
            assert len(report.secondary_causes) == len(report.patterns) - 1


# ---------------------------------------------------------------------------
# suggest_fixes
# ---------------------------------------------------------------------------


class TestSuggestFixes:
    def test_generates_suggestions(self):
        failures = [
            _failure(item_id=f"i{i}", output="hi") for i in range(10)
        ]
        report = analyze_root_causes(failures)
        fixes = suggest_fixes(report)
        assert len(fixes) > 0
        assert all(isinstance(f, str) for f in fixes)

    def test_no_patterns_still_gives_advice(self):
        report = RootCauseReport(total_failures=5, patterns=[])
        fixes = suggest_fixes(report)
        assert len(fixes) == 1
        assert "No clear pattern" in fixes[0]

    def test_empty_report_no_fixes(self):
        report = RootCauseReport(total_failures=0, patterns=[])
        fixes = suggest_fixes(report)
        assert fixes == []


# ---------------------------------------------------------------------------
# RootCauseReport
# ---------------------------------------------------------------------------


class TestRootCauseReport:
    def test_format_text(self):
        pattern = FailurePattern(
            pattern_name="short_output",
            description="8/10 failures have short outputs",
            affected_items=["i1", "i2"],
            confidence=0.8,
            suggested_fix="Increase max_tokens",
        )
        report = RootCauseReport(
            patterns=[pattern],
            total_failures=10,
            primary_cause="short_output",
        )
        text = report.format_text()
        assert "Root Cause Report" in text
        assert "10 failures" in text
        assert "short_output" in text
        assert "Increase max_tokens" in text

    def test_as_dict(self):
        report = RootCauseReport(
            patterns=[
                FailurePattern(
                    pattern_name="test", description="desc", confidence=0.5
                )
            ],
            total_failures=5,
            primary_cause="test",
            secondary_causes=["other"],
        )
        d = report.as_dict()
        assert d["total_failures"] == 5
        assert d["primary_cause"] == "test"
        assert len(d["patterns"]) == 1
        assert d["secondary_causes"] == ["other"]

    def test_from_dict_roundtrip(self):
        original = RootCauseReport(
            patterns=[
                FailurePattern(
                    pattern_name="p1",
                    description="d1",
                    affected_items=["a", "b"],
                    confidence=0.7,
                    suggested_fix="fix it",
                )
            ],
            total_failures=10,
            primary_cause="p1",
            secondary_causes=["p2"],
        )
        restored = RootCauseReport.from_dict(original.as_dict())
        assert restored.total_failures == original.total_failures
        assert restored.primary_cause == original.primary_cause
        assert len(restored.patterns) == 1
        assert restored.patterns[0].pattern_name == "p1"
        assert restored.patterns[0].confidence == 0.7
        assert restored.secondary_causes == ["p2"]

    def test_from_dict_empty(self):
        report = RootCauseReport.from_dict({})
        assert report.total_failures == 0
        assert report.patterns == []


# ---------------------------------------------------------------------------
# FailurePattern
# ---------------------------------------------------------------------------


class TestFailurePattern:
    def test_as_dict(self):
        fp = FailurePattern(
            pattern_name="test",
            description="desc",
            affected_items=["a"],
            confidence=0.9,
            suggested_fix="do something",
        )
        d = fp.as_dict()
        assert d["pattern_name"] == "test"
        assert d["confidence"] == 0.9
        assert d["affected_items"] == ["a"]

    def test_defaults(self):
        fp = FailurePattern(pattern_name="x", description="y")
        assert fp.affected_items == []
        assert fp.confidence == 0.0
        assert fp.suggested_fix == ""
