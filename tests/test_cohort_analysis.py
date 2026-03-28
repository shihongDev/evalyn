"""Tests for cohort analysis."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.cohort_analysis import (
    CohortComparison,
    CohortReport,
    CohortStats,
    build_cohort_report,
    compare_cohorts,
    compute_cohort_stats,
    define_cohort_by_length,
    define_cohort_by_metadata,
    render_cohort_chart,
)


# ---------------------------------------------------------------------------
# define_cohort_by_metadata
# ---------------------------------------------------------------------------


class TestDefineCohortByMetadata:
    def test_groups_by_key(self):
        """Should group items by metadata field value."""
        items = [
            {"id": "1", "metadata": {"category": "A"}, "score": 0.9},
            {"id": "2", "metadata": {"category": "B"}, "score": 0.8},
            {"id": "3", "metadata": {"category": "A"}, "score": 0.7},
        ]
        cohorts = define_cohort_by_metadata(items, "category")
        assert "A" in cohorts
        assert "B" in cohorts
        assert len(cohorts["A"]) == 2
        assert len(cohorts["B"]) == 1

    def test_missing_key_goes_to_unknown(self):
        """Items missing the key should be in 'unknown' cohort."""
        items = [
            {"id": "1", "metadata": {"category": "A"}, "score": 0.9},
            {"id": "2", "metadata": {}, "score": 0.5},
            {"id": "3", "score": 0.4},
        ]
        cohorts = define_cohort_by_metadata(items, "category")
        assert "A" in cohorts
        assert "unknown" in cohorts
        assert len(cohorts["unknown"]) == 2

    def test_empty_items(self):
        """Empty items list should return empty dict."""
        cohorts = define_cohort_by_metadata([], "category")
        assert cohorts == {}

    def test_non_dict_metadata(self):
        """Non-dict metadata should put item in 'unknown'."""
        items = [{"id": "1", "metadata": "not_a_dict", "score": 0.5}]
        cohorts = define_cohort_by_metadata(items, "category")
        assert "unknown" in cohorts

    def test_numeric_metadata_values(self):
        """Numeric metadata values should be converted to string keys."""
        items = [
            {"id": "1", "metadata": {"level": 1}, "score": 0.9},
            {"id": "2", "metadata": {"level": 2}, "score": 0.8},
            {"id": "3", "metadata": {"level": 1}, "score": 0.7},
        ]
        cohorts = define_cohort_by_metadata(items, "level")
        assert "1" in cohorts
        assert "2" in cohorts


# ---------------------------------------------------------------------------
# define_cohort_by_length
# ---------------------------------------------------------------------------


class TestDefineCohortByLength:
    def test_three_buckets(self):
        """Should split into short/medium/long."""
        items = [
            {"input": "hi"},            # len 2
            {"input": "hello world"},    # len 11
            {"input": "a" * 30},         # len 30
        ]
        cohorts = define_cohort_by_length(items, "input", num_buckets=3)
        assert "short" in cohorts
        assert "medium" in cohorts
        assert "long" in cohorts
        # Total items should be preserved
        total = sum(len(v) for v in cohorts.values())
        assert total == 3

    def test_custom_num_buckets(self):
        """Non-3 buckets should use bucket_N names."""
        items = [{"input": "a" * i} for i in range(1, 11)]
        cohorts = define_cohort_by_length(items, "input", num_buckets=5)
        assert "bucket_0" in cohorts
        assert "bucket_4" in cohorts

    def test_empty_items(self):
        """Empty items should return empty dict."""
        cohorts = define_cohort_by_length([], "input")
        assert cohorts == {}

    def test_same_length_items(self):
        """All items with same length should go to same bucket."""
        items = [{"input": "abc"}, {"input": "def"}, {"input": "ghi"}]
        cohorts = define_cohort_by_length(items, "input", num_buckets=3)
        # All same length, so all in first bucket
        total_nonempty = sum(1 for v in cohorts.values() if v)
        assert total_nonempty >= 1

    def test_dict_input_field(self):
        """Dict input values should use JSON length."""
        items = [
            {"input": {"a": 1}},
            {"input": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}},
            {"input": {"x": "y" * 100}},
        ]
        cohorts = define_cohort_by_length(items, "input", num_buckets=3)
        total = sum(len(v) for v in cohorts.values())
        assert total == 3


# ---------------------------------------------------------------------------
# compute_cohort_stats
# ---------------------------------------------------------------------------


class TestComputeCohortStats:
    def test_basic_stats(self):
        """Should compute avg score and pass rate."""
        items = [
            {"score": 0.9, "passed": True},
            {"score": 0.8, "passed": True},
            {"score": 0.3, "passed": False},
        ]
        stats = compute_cohort_stats("test", items)
        assert stats.cohort_name == "test"
        assert stats.item_count == 3
        assert abs(stats.avg_score - 0.6667) < 0.01
        assert abs(stats.pass_rate - 0.6667) < 0.01

    def test_empty_cohort(self):
        """Empty cohort should have zero stats."""
        stats = compute_cohort_stats("empty", [])
        assert stats.item_count == 0
        assert stats.avg_score == 0.0
        assert stats.pass_rate == 0.0
        assert stats.std_dev == 0.0

    def test_all_pass(self):
        """All passing items should have pass rate 1.0."""
        items = [
            {"score": 1.0, "passed": True},
            {"score": 0.9, "passed": True},
        ]
        stats = compute_cohort_stats("all_pass", items)
        assert stats.pass_rate == 1.0

    def test_no_pass_field(self):
        """Items without 'passed' field should have pass_rate 0."""
        items = [{"score": 0.5}, {"score": 0.7}]
        stats = compute_cohort_stats("no_pass", items)
        assert stats.pass_rate == 0.0

    def test_std_dev_computed(self):
        """Standard deviation should be computed correctly."""
        items = [{"score": 0.0}, {"score": 1.0}]
        stats = compute_cohort_stats("std_test", items)
        assert stats.std_dev == 0.5

    def test_custom_score_field(self):
        """Should use custom score field."""
        items = [{"quality": 0.8}, {"quality": 0.6}]
        stats = compute_cohort_stats("custom", items, score_field="quality")
        assert abs(stats.avg_score - 0.7) < 0.01


# ---------------------------------------------------------------------------
# compare_cohorts
# ---------------------------------------------------------------------------


class TestCompareCohorts:
    def test_significant_difference(self):
        """Score delta > 0.1 should be significant."""
        a = CohortStats("high", 10, avg_score=0.9, pass_rate=0.9, std_dev=0.1)
        b = CohortStats("low", 10, avg_score=0.3, pass_rate=0.3, std_dev=0.1)
        comp = compare_cohorts(a, b)
        assert comp.significant is True
        assert comp.score_delta > 0
        assert comp.cohort_a == "high"
        assert comp.cohort_b == "low"

    def test_not_significant(self):
        """Score delta <= 0.1 should not be significant."""
        a = CohortStats("a", 10, avg_score=0.5, pass_rate=0.5, std_dev=0.1)
        b = CohortStats("b", 10, avg_score=0.45, pass_rate=0.45, std_dev=0.1)
        comp = compare_cohorts(a, b)
        assert comp.significant is False

    def test_exactly_at_boundary(self):
        """Score delta of exactly 0.1 should not be significant."""
        a = CohortStats("a", 10, avg_score=0.6, pass_rate=0.6, std_dev=0.1)
        b = CohortStats("b", 10, avg_score=0.5, pass_rate=0.5, std_dev=0.1)
        comp = compare_cohorts(a, b)
        assert comp.significant is False

    def test_negative_delta(self):
        """Lower a than b should produce negative delta."""
        a = CohortStats("a", 10, avg_score=0.3, pass_rate=0.3, std_dev=0.1)
        b = CohortStats("b", 10, avg_score=0.9, pass_rate=0.9, std_dev=0.1)
        comp = compare_cohorts(a, b)
        assert comp.score_delta < 0
        assert comp.significant is True


# ---------------------------------------------------------------------------
# build_cohort_report
# ---------------------------------------------------------------------------


class TestBuildCohortReport:
    def test_best_worst(self):
        """Should identify best and worst cohorts."""
        cohorts = {
            "good": [{"score": 0.9, "passed": True}, {"score": 0.8, "passed": True}],
            "bad": [{"score": 0.2, "passed": False}, {"score": 0.3, "passed": False}],
            "mid": [{"score": 0.5, "passed": True}, {"score": 0.6, "passed": True}],
        }
        report = build_cohort_report(cohorts)
        assert report.best_cohort == "good"
        assert report.worst_cohort == "bad"

    def test_pairwise_comparisons(self):
        """Should have C(n,2) comparisons for n cohorts."""
        cohorts = {
            "a": [{"score": 0.9}],
            "b": [{"score": 0.5}],
            "c": [{"score": 0.1}],
        }
        report = build_cohort_report(cohorts)
        # 3 choose 2 = 3 comparisons
        assert len(report.comparisons) == 3

    def test_single_cohort(self):
        """Single cohort should produce no comparisons."""
        cohorts = {"only": [{"score": 0.7}]}
        report = build_cohort_report(cohorts)
        assert len(report.cohorts) == 1
        assert len(report.comparisons) == 0
        assert report.best_cohort == "only"
        assert report.worst_cohort == "only"

    def test_empty_cohorts(self):
        """Empty cohorts dict should produce empty report."""
        report = build_cohort_report({})
        assert len(report.cohorts) == 0
        assert report.best_cohort == ""
        assert report.worst_cohort == ""

    def test_custom_score_field(self):
        """Should respect custom score field."""
        cohorts = {
            "a": [{"quality": 0.9}],
            "b": [{"quality": 0.1}],
        }
        report = build_cohort_report(cohorts, score_field="quality")
        assert report.best_cohort == "a"


# ---------------------------------------------------------------------------
# render_cohort_chart
# ---------------------------------------------------------------------------


class TestRenderCohortChart:
    def test_output_has_lines(self):
        """Should produce one line per cohort."""
        report = CohortReport(
            cohorts=[
                CohortStats("a", 10, avg_score=0.8),
                CohortStats("b", 10, avg_score=0.4),
            ]
        )
        chart = render_cohort_chart(report, width=40)
        lines = chart.strip().split("\n")
        assert len(lines) == 2

    def test_bars_proportional(self):
        """Higher score should have longer bar."""
        report = CohortReport(
            cohorts=[
                CohortStats("high", 10, avg_score=0.9),
                CohortStats("low", 10, avg_score=0.1),
            ]
        )
        chart = render_cohort_chart(report, width=60)
        lines = chart.strip().split("\n")
        # Count '#' in each line
        high_hashes = lines[0].count("#")
        low_hashes = lines[1].count("#")
        assert high_hashes > low_hashes

    def test_empty_report(self):
        """Empty report should return empty string."""
        report = CohortReport()
        assert render_cohort_chart(report) == ""

    def test_score_shown(self):
        """Score value should appear in chart."""
        report = CohortReport(
            cohorts=[CohortStats("test", 5, avg_score=0.750)]
        )
        chart = render_cohort_chart(report)
        assert "0.750" in chart


# ---------------------------------------------------------------------------
# CohortComparison format_text
# ---------------------------------------------------------------------------


class TestCohortComparisonFormatText:
    def test_significant_shown(self):
        """Significant comparison should show marker."""
        comp = CohortComparison("a", "b", score_delta=0.5, pass_rate_delta=0.3, significant=True)
        text = comp.format_text()
        assert "SIGNIFICANT" in text
        assert "a vs b" in text

    def test_not_significant(self):
        """Non-significant comparison should not show marker."""
        comp = CohortComparison("a", "b", score_delta=0.05, pass_rate_delta=0.01, significant=False)
        text = comp.format_text()
        assert "SIGNIFICANT" not in text


# ---------------------------------------------------------------------------
# CohortReport serialization
# ---------------------------------------------------------------------------


class TestCohortReportSerialization:
    def test_as_dict(self):
        """as_dict should include all fields."""
        report = CohortReport(
            cohorts=[CohortStats("a", 5, avg_score=0.8, pass_rate=0.9, std_dev=0.1)],
            comparisons=[],
            best_cohort="a",
            worst_cohort="a",
        )
        d = report.as_dict()
        assert d["best_cohort"] == "a"
        assert len(d["cohorts"]) == 1
        assert d["cohorts"][0]["cohort_name"] == "a"

    def test_from_dict(self):
        """from_dict should reconstruct report."""
        d = {
            "cohorts": [
                {"cohort_name": "x", "item_count": 3, "avg_score": 0.7,
                 "pass_rate": 0.8, "std_dev": 0.1}
            ],
            "comparisons": [
                {"cohort_a": "x", "cohort_b": "y", "score_delta": 0.2,
                 "pass_rate_delta": 0.1, "significant": True}
            ],
            "best_cohort": "x",
            "worst_cohort": "y",
        }
        report = CohortReport.from_dict(d)
        assert report.best_cohort == "x"
        assert len(report.cohorts) == 1
        assert report.cohorts[0].avg_score == 0.7
        assert len(report.comparisons) == 1
        assert report.comparisons[0].significant is True

    def test_roundtrip(self):
        """as_dict -> from_dict should preserve data."""
        original = CohortReport(
            cohorts=[
                CohortStats("a", 10, 0.9, 0.95, 0.05),
                CohortStats("b", 8, 0.4, 0.5, 0.2),
            ],
            comparisons=[
                CohortComparison("a", "b", 0.5, 0.45, True)
            ],
            best_cohort="a",
            worst_cohort="b",
        )
        restored = CohortReport.from_dict(original.as_dict())
        assert restored.best_cohort == original.best_cohort
        assert restored.worst_cohort == original.worst_cohort
        assert len(restored.cohorts) == 2
        assert len(restored.comparisons) == 1

    def test_format_text(self):
        """format_text should produce readable output."""
        report = CohortReport(
            cohorts=[
                CohortStats("good", 10, avg_score=0.9, pass_rate=0.95, std_dev=0.05),
                CohortStats("bad", 10, avg_score=0.3, pass_rate=0.2, std_dev=0.15),
            ],
            comparisons=[
                CohortComparison("good", "bad", 0.6, 0.75, True)
            ],
            best_cohort="good",
            worst_cohort="bad",
        )
        text = report.format_text()
        assert "Cohort Report" in text
        assert "Best cohort: good" in text
        assert "Worst cohort: bad" in text
        assert "good" in text
        assert "bad" in text

    def test_format_text_empty(self):
        """format_text with empty report should still work."""
        report = CohortReport()
        text = report.format_text()
        assert "Cohort Report" in text


# ---------------------------------------------------------------------------
# CohortStats serialization
# ---------------------------------------------------------------------------


class TestCohortStatsSerialization:
    def test_as_dict(self):
        """as_dict should produce correct dict."""
        stats = CohortStats("test", 5, 0.8, 0.9, 0.1)
        d = stats.as_dict()
        assert d["cohort_name"] == "test"
        assert d["item_count"] == 5
        assert d["avg_score"] == 0.8
        assert d["pass_rate"] == 0.9
        assert d["std_dev"] == 0.1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_cohort_with_none_scores(self):
        """Items with None scores should be handled."""
        items = [{"score": None}, {"score": 0.5}]
        stats = compute_cohort_stats("mixed", items)
        assert stats.item_count == 2
        assert stats.avg_score == 0.5

    def test_empty_items_in_cohort(self):
        """Cohort with items that have no fields should not crash."""
        items = [{}, {}]
        stats = compute_cohort_stats("empty_items", items)
        assert stats.item_count == 2
        assert stats.avg_score == 0.0

    def test_large_number_of_cohorts(self):
        """Many cohorts should be handled correctly."""
        cohorts = {
            f"cohort_{i}": [{"score": float(i) / 100}]
            for i in range(20)
        }
        report = build_cohort_report(cohorts)
        assert len(report.cohorts) == 20
        # C(20, 2) = 190 comparisons
        assert len(report.comparisons) == 190
