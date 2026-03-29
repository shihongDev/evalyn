"""Tests for regression_simulation module."""

from __future__ import annotations

import random

import pytest

from evalyn_sdk.regression_simulation import (
    FailurePattern,
    RegressionInput,
    RegressionReport,
    RegressionResult,
    build_regression_report,
    compute_regression_results,
    extract_failure_patterns,
    format_regression_report,
    generate_regression_suite,
    generate_regression_variants,
    identify_recurring_failures,
)


# ---------------------------------------------------------------------------
# FailurePattern
# ---------------------------------------------------------------------------


class TestFailurePattern:
    def test_as_dict(self):
        fp = FailurePattern(
            pattern_id="p1",
            description="timeout error",
            example_inputs=["input a", "input b"],
            category="performance",
            frequency=2,
        )
        d = fp.as_dict()
        assert d["pattern_id"] == "p1"
        assert d["description"] == "timeout error"
        assert d["example_inputs"] == ["input a", "input b"]
        assert d["category"] == "performance"
        assert d["frequency"] == 2

    def test_from_dict(self):
        d = {
            "pattern_id": "p2",
            "description": "parse error",
            "example_inputs": ["x"],
            "category": "formatting",
            "frequency": 5,
        }
        fp = FailurePattern.from_dict(d)
        assert fp.pattern_id == "p2"
        assert fp.frequency == 5

    def test_roundtrip(self):
        fp = FailurePattern("id1", "desc", ["a"], "cat", 3)
        assert FailurePattern.from_dict(fp.as_dict()).as_dict() == fp.as_dict()

    def test_from_dict_defaults(self):
        d = {"pattern_id": "x", "description": "d"}
        fp = FailurePattern.from_dict(d)
        assert fp.example_inputs == []
        assert fp.category == ""
        assert fp.frequency == 1

    def test_default_frequency(self):
        fp = FailurePattern("id", "desc", [], "cat")
        assert fp.frequency == 1


# ---------------------------------------------------------------------------
# RegressionInput
# ---------------------------------------------------------------------------


class TestRegressionInput:
    def test_as_dict(self):
        ri = RegressionInput("hello", "p1", 0, ["tag1"])
        d = ri.as_dict()
        assert d["input_text"] == "hello"
        assert d["source_pattern_id"] == "p1"
        assert d["variant_index"] == 0
        assert d["tags"] == ["tag1"]

    def test_from_dict(self):
        d = {
            "input_text": "world",
            "source_pattern_id": "p2",
            "variant_index": 3,
            "tags": ["a", "b"],
        }
        ri = RegressionInput.from_dict(d)
        assert ri.input_text == "world"
        assert ri.variant_index == 3

    def test_roundtrip(self):
        ri = RegressionInput("text", "pat", 1, ["t"])
        assert RegressionInput.from_dict(ri.as_dict()).as_dict() == ri.as_dict()

    def test_default_tags(self):
        ri = RegressionInput("a", "b", 0)
        assert ri.tags == []

    def test_from_dict_defaults(self):
        d = {"input_text": "x", "source_pattern_id": "y"}
        ri = RegressionInput.from_dict(d)
        assert ri.variant_index == 0
        assert ri.tags == []


# ---------------------------------------------------------------------------
# RegressionResult
# ---------------------------------------------------------------------------


class TestRegressionResult:
    def test_as_dict(self):
        rr = RegressionResult("p1", 10, 8, 2, 0.8)
        d = rr.as_dict()
        assert d["pattern_id"] == "p1"
        assert d["fix_rate"] == 0.8

    def test_from_dict(self):
        d = {
            "pattern_id": "p1",
            "total_variants": 5,
            "passed_count": 3,
            "failed_count": 2,
            "fix_rate": 0.6,
        }
        rr = RegressionResult.from_dict(d)
        assert rr.total_variants == 5
        assert rr.fix_rate == 0.6

    def test_roundtrip(self):
        rr = RegressionResult("id", 4, 2, 2, 0.5)
        assert RegressionResult.from_dict(rr.as_dict()).as_dict() == rr.as_dict()

    def test_from_dict_defaults(self):
        d = {"pattern_id": "x"}
        rr = RegressionResult.from_dict(d)
        assert rr.total_variants == 0
        assert rr.fix_rate == 0.0


# ---------------------------------------------------------------------------
# RegressionReport
# ---------------------------------------------------------------------------


class TestRegressionReport:
    def test_as_dict(self):
        results = [RegressionResult("p1", 3, 3, 0, 1.0)]
        report = RegressionReport(results, 1.0, 1, 0, 1)
        d = report.as_dict()
        assert d["overall_fix_rate"] == 1.0
        assert len(d["results"]) == 1

    def test_from_dict(self):
        d = {
            "results": [
                {
                    "pattern_id": "p1",
                    "total_variants": 2,
                    "passed_count": 1,
                    "failed_count": 1,
                    "fix_rate": 0.5,
                }
            ],
            "overall_fix_rate": 0.5,
            "fully_fixed_patterns": 0,
            "still_failing_patterns": 1,
            "total_patterns": 1,
        }
        report = RegressionReport.from_dict(d)
        assert report.total_patterns == 1
        assert report.results[0].pattern_id == "p1"

    def test_roundtrip(self):
        results = [RegressionResult("a", 2, 1, 1, 0.5)]
        report = RegressionReport(results, 0.5, 0, 1, 1)
        d = report.as_dict()
        restored = RegressionReport.from_dict(d)
        assert restored.as_dict() == d


# ---------------------------------------------------------------------------
# extract_failure_patterns
# ---------------------------------------------------------------------------


class TestExtractFailurePatterns:
    def test_basic_grouping(self):
        failures = [
            {"input": "a", "failure_reason": "timeout"},
            {"input": "b", "failure_reason": "timeout"},
            {"input": "c", "failure_reason": "parse error"},
        ]
        patterns = extract_failure_patterns(failures)
        assert len(patterns) == 2
        ids = {p.pattern_id for p in patterns}
        assert "timeout" in ids
        assert "parse_error" in ids

    def test_frequency_count(self):
        failures = [
            {"input": "a", "failure_reason": "timeout"},
            {"input": "b", "failure_reason": "timeout"},
            {"input": "c", "failure_reason": "timeout"},
        ]
        patterns = extract_failure_patterns(failures)
        assert len(patterns) == 1
        assert patterns[0].frequency == 3

    def test_example_inputs_collected(self):
        failures = [
            {"input": "hello", "failure_reason": "err"},
            {"input": "world", "failure_reason": "err"},
        ]
        patterns = extract_failure_patterns(failures)
        assert patterns[0].example_inputs == ["hello", "world"]

    def test_custom_fields(self):
        failures = [
            {"text": "abc", "reason": "bad format"},
        ]
        patterns = extract_failure_patterns(
            failures, text_field="text", reason_field="reason"
        )
        assert len(patterns) == 1
        assert patterns[0].example_inputs == ["abc"]

    def test_missing_reason_defaults_unknown(self):
        failures = [{"input": "x"}]
        patterns = extract_failure_patterns(failures)
        assert patterns[0].pattern_id == "unknown"

    def test_empty_failures(self):
        patterns = extract_failure_patterns([])
        assert patterns == []

    def test_category_performance(self):
        failures = [{"input": "x", "failure_reason": "timeout exceeded"}]
        patterns = extract_failure_patterns(failures)
        assert patterns[0].category == "performance"

    def test_category_formatting(self):
        failures = [{"input": "x", "failure_reason": "parse failure"}]
        patterns = extract_failure_patterns(failures)
        assert patterns[0].category == "formatting"

    def test_category_missing_data(self):
        failures = [{"input": "x", "failure_reason": "null response"}]
        patterns = extract_failure_patterns(failures)
        assert patterns[0].category == "missing_data"

    def test_category_validation(self):
        failures = [{"input": "x", "failure_reason": "invalid input"}]
        patterns = extract_failure_patterns(failures)
        assert patterns[0].category == "validation"

    def test_category_other(self):
        failures = [{"input": "x", "failure_reason": "something weird"}]
        patterns = extract_failure_patterns(failures)
        assert patterns[0].category == "other"


# ---------------------------------------------------------------------------
# generate_regression_variants
# ---------------------------------------------------------------------------


class TestGenerateRegressionVariants:
    def test_generates_correct_count(self):
        fp = FailurePattern("p1", "desc", ["hello world"], "cat", 1)
        variants = generate_regression_variants(fp, n_variants=3)
        assert len(variants) == 3

    def test_multiple_examples(self):
        fp = FailurePattern("p1", "desc", ["a b", "c d"], "cat", 2)
        variants = generate_regression_variants(fp, n_variants=2)
        assert len(variants) == 4  # 2 examples * 2 variants

    def test_deterministic_with_seed(self):
        fp = FailurePattern("p1", "desc", ["hello world"], "cat", 1)
        rng1 = random.Random(99)
        rng2 = random.Random(99)
        v1 = generate_regression_variants(fp, 3, rng1)
        v2 = generate_regression_variants(fp, 3, rng2)
        assert [v.input_text for v in v1] == [v.input_text for v in v2]

    def test_variant_index_sequential(self):
        fp = FailurePattern("p1", "desc", ["a b c"], "cat", 1)
        variants = generate_regression_variants(fp, 5)
        indices = [v.variant_index for v in variants]
        assert indices == list(range(5))

    def test_tags_include_category(self):
        fp = FailurePattern("p1", "desc", ["hello"], "performance", 1)
        variants = generate_regression_variants(fp, 1)
        assert "performance" in variants[0].tags
        assert "regression" in variants[0].tags

    def test_empty_examples(self):
        fp = FailurePattern("p1", "desc", [], "cat", 0)
        variants = generate_regression_variants(fp, 3)
        assert variants == []

    def test_empty_string_example_skipped(self):
        fp = FailurePattern("p1", "desc", ["", "hello"], "cat", 1)
        variants = generate_regression_variants(fp, 2)
        # Only "hello" generates variants
        assert len(variants) == 2

    def test_source_pattern_id_set(self):
        fp = FailurePattern("my_pattern", "desc", ["text"], "cat", 1)
        variants = generate_regression_variants(fp, 1)
        assert variants[0].source_pattern_id == "my_pattern"

    def test_default_rng(self):
        fp = FailurePattern("p1", "desc", ["x y z"], "cat", 1)
        # Should not raise
        variants = generate_regression_variants(fp, 2)
        assert len(variants) == 2


# ---------------------------------------------------------------------------
# generate_regression_suite
# ---------------------------------------------------------------------------


class TestGenerateRegressionSuite:
    def test_combines_patterns(self):
        p1 = FailurePattern("a", "d1", ["x y"], "cat", 1)
        p2 = FailurePattern("b", "d2", ["a b"], "cat", 1)
        suite = generate_regression_suite([p1, p2], n_variants=2)
        assert len(suite) == 4

    def test_deterministic(self):
        p1 = FailurePattern("a", "d", ["hello world"], "cat", 1)
        s1 = generate_regression_suite([p1], 3)
        s2 = generate_regression_suite([p1], 3)
        assert [v.input_text for v in s1] == [v.input_text for v in s2]

    def test_empty_patterns(self):
        suite = generate_regression_suite([], 5)
        assert suite == []


# ---------------------------------------------------------------------------
# compute_regression_results
# ---------------------------------------------------------------------------


class TestComputeRegressionResults:
    def test_all_passed(self):
        inputs = [
            RegressionInput("a", "p1", 0),
            RegressionInput("b", "p1", 1),
        ]
        scores = {"a": 0.9, "b": 0.7}
        results = compute_regression_results(inputs, scores)
        assert len(results) == 1
        assert results[0].passed_count == 2
        assert results[0].fix_rate == 1.0

    def test_all_failed(self):
        inputs = [
            RegressionInput("a", "p1", 0),
            RegressionInput("b", "p1", 1),
        ]
        scores = {"a": 0.1, "b": 0.3}
        results = compute_regression_results(inputs, scores)
        assert results[0].failed_count == 2
        assert results[0].fix_rate == 0.0

    def test_mixed_results(self):
        inputs = [
            RegressionInput("a", "p1", 0),
            RegressionInput("b", "p1", 1),
        ]
        scores = {"a": 0.8, "b": 0.2}
        results = compute_regression_results(inputs, scores)
        assert results[0].passed_count == 1
        assert results[0].failed_count == 1
        assert results[0].fix_rate == 0.5

    def test_multiple_patterns(self):
        inputs = [
            RegressionInput("a", "p1", 0),
            RegressionInput("b", "p2", 0),
        ]
        scores = {"a": 1.0, "b": 0.0}
        results = compute_regression_results(inputs, scores)
        assert len(results) == 2
        by_id = {r.pattern_id: r for r in results}
        assert by_id["p1"].fix_rate == 1.0
        assert by_id["p2"].fix_rate == 0.0

    def test_missing_score_treated_as_zero(self):
        inputs = [RegressionInput("missing", "p1", 0)]
        scores = {}
        results = compute_regression_results(inputs, scores)
        assert results[0].failed_count == 1

    def test_boundary_score_0_5_is_pass(self):
        inputs = [RegressionInput("x", "p1", 0)]
        scores = {"x": 0.5}
        results = compute_regression_results(inputs, scores)
        assert results[0].passed_count == 1

    def test_boundary_score_below_0_5_is_fail(self):
        inputs = [RegressionInput("x", "p1", 0)]
        scores = {"x": 0.49}
        results = compute_regression_results(inputs, scores)
        assert results[0].failed_count == 1

    def test_empty_inputs(self):
        results = compute_regression_results([], {})
        assert results == []


# ---------------------------------------------------------------------------
# build_regression_report
# ---------------------------------------------------------------------------


class TestBuildRegressionReport:
    def test_all_fixed(self):
        results = [
            RegressionResult("p1", 3, 3, 0, 1.0),
            RegressionResult("p2", 2, 2, 0, 1.0),
        ]
        report = build_regression_report(results)
        assert report.fully_fixed_patterns == 2
        assert report.still_failing_patterns == 0
        assert report.overall_fix_rate == 1.0

    def test_none_fixed(self):
        results = [RegressionResult("p1", 4, 0, 4, 0.0)]
        report = build_regression_report(results)
        assert report.fully_fixed_patterns == 0
        assert report.still_failing_patterns == 1
        assert report.overall_fix_rate == 0.0

    def test_mixed(self):
        results = [
            RegressionResult("p1", 4, 4, 0, 1.0),
            RegressionResult("p2", 4, 2, 2, 0.5),
        ]
        report = build_regression_report(results)
        assert report.fully_fixed_patterns == 1
        assert report.still_failing_patterns == 1
        assert report.overall_fix_rate == 0.75

    def test_empty_results(self):
        report = build_regression_report([])
        assert report.total_patterns == 0
        assert report.overall_fix_rate == 0.0

    def test_total_patterns(self):
        results = [
            RegressionResult("a", 1, 1, 0, 1.0),
            RegressionResult("b", 1, 0, 1, 0.0),
            RegressionResult("c", 1, 1, 0, 1.0),
        ]
        report = build_regression_report(results)
        assert report.total_patterns == 3


# ---------------------------------------------------------------------------
# format_regression_report
# ---------------------------------------------------------------------------


class TestFormatRegressionReport:
    def test_contains_header(self):
        report = RegressionReport([], 0.0, 0, 0, 0)
        text = format_regression_report(report)
        assert "REGRESSION REPORT" in text

    def test_contains_fix_rate(self):
        results = [RegressionResult("p1", 2, 1, 1, 0.5)]
        report = build_regression_report(results)
        text = format_regression_report(report)
        assert "50.0%" in text

    def test_contains_pattern_status(self):
        results = [
            RegressionResult("p1", 2, 2, 0, 1.0),
            RegressionResult("p2", 2, 0, 2, 0.0),
        ]
        report = build_regression_report(results)
        text = format_regression_report(report)
        assert "FIXED" in text
        assert "FAILING" in text

    def test_contains_pattern_ids(self):
        results = [RegressionResult("my_pattern", 1, 1, 0, 1.0)]
        report = build_regression_report(results)
        text = format_regression_report(report)
        assert "my_pattern" in text


# ---------------------------------------------------------------------------
# identify_recurring_failures
# ---------------------------------------------------------------------------


class TestIdentifyRecurringFailures:
    def test_below_threshold(self):
        results = [
            RegressionResult("p1", 4, 1, 3, 0.25),
            RegressionResult("p2", 4, 3, 1, 0.75),
        ]
        report = build_regression_report(results)
        recurring = identify_recurring_failures(report, threshold=0.5)
        assert recurring == ["p1"]

    def test_all_above_threshold(self):
        results = [RegressionResult("p1", 2, 2, 0, 1.0)]
        report = build_regression_report(results)
        recurring = identify_recurring_failures(report, threshold=0.5)
        assert recurring == []

    def test_all_below_threshold(self):
        results = [
            RegressionResult("p1", 2, 0, 2, 0.0),
            RegressionResult("p2", 2, 0, 2, 0.0),
        ]
        report = build_regression_report(results)
        recurring = identify_recurring_failures(report, threshold=0.5)
        assert set(recurring) == {"p1", "p2"}

    def test_at_threshold_not_recurring(self):
        results = [RegressionResult("p1", 4, 2, 2, 0.5)]
        report = build_regression_report(results)
        recurring = identify_recurring_failures(report, threshold=0.5)
        assert recurring == []

    def test_custom_threshold(self):
        results = [RegressionResult("p1", 4, 3, 1, 0.75)]
        report = build_regression_report(results)
        recurring = identify_recurring_failures(report, threshold=0.8)
        assert recurring == ["p1"]

    def test_empty_report(self):
        report = RegressionReport([], 0.0, 0, 0, 0)
        recurring = identify_recurring_failures(report)
        assert recurring == []


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline(self):
        failures = [
            {"input": "find the error in code", "failure_reason": "invalid output"},
            {"input": "show the bad results", "failure_reason": "invalid output"},
            {"input": "help with timeout", "failure_reason": "timeout"},
        ]
        patterns = extract_failure_patterns(failures)
        assert len(patterns) == 2

        suite = generate_regression_suite(patterns, n_variants=2)
        assert len(suite) > 0

        # Simulate scores: all pass
        scores = {inp.input_text: 0.9 for inp in suite}
        results = compute_regression_results(suite, scores)
        report = build_regression_report(results)
        assert report.overall_fix_rate == 1.0
        assert report.fully_fixed_patterns == 2

        text = format_regression_report(report)
        assert "FIXED" in text

        recurring = identify_recurring_failures(report)
        assert recurring == []

    def test_full_pipeline_partial_fix(self):
        failures = [
            {"input": "use good data", "failure_reason": "bad format"},
        ]
        patterns = extract_failure_patterns(failures)
        suite = generate_regression_suite(patterns, n_variants=4)

        # Fail half the variants
        scores = {}
        for i, inp in enumerate(suite):
            scores[inp.input_text] = 0.9 if i % 2 == 0 else 0.1

        results = compute_regression_results(suite, scores)
        report = build_regression_report(results)
        assert 0 < report.overall_fix_rate < 1.0
