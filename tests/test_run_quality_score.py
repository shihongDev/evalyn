"""Tests for run quality score module."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.run_quality_score import (
    QualityDimension,
    RunQualityResult,
    score_pass_rate,
    score_coverage,
    score_confidence,
    score_cost_efficiency,
    compute_run_quality,
    compare_run_quality,
)


# ---------------------------------------------------------------------------
# score_pass_rate
# ---------------------------------------------------------------------------


class TestScorePassRate:
    def test_perfect(self):
        dim = score_pass_rate(1.0)
        assert dim.score == 1.0
        assert dim.name == "pass_rate"

    def test_zero(self):
        dim = score_pass_rate(0.0)
        assert dim.score == 0.0

    def test_half(self):
        dim = score_pass_rate(0.5)
        assert dim.score == 0.5

    def test_clamped_above(self):
        """Values above 1.0 should be clamped."""
        dim = score_pass_rate(1.5)
        assert dim.score == 1.0

    def test_clamped_below(self):
        """Negative values should be clamped to 0."""
        dim = score_pass_rate(-0.1)
        assert dim.score == 0.0


# ---------------------------------------------------------------------------
# score_coverage
# ---------------------------------------------------------------------------


class TestScoreCoverage:
    def test_full_coverage(self):
        dim = score_coverage(5, 5)
        assert dim.score == 1.0
        assert dim.name == "coverage"

    def test_partial_coverage(self):
        dim = score_coverage(3, 6)
        assert dim.score == 0.5

    def test_zero_total(self):
        dim = score_coverage(0, 0)
        assert dim.score == 0.0

    def test_overcoverage_clamped(self):
        """More evaluated than total should clamp to 1.0."""
        dim = score_coverage(10, 5)
        assert dim.score == 1.0


# ---------------------------------------------------------------------------
# score_confidence
# ---------------------------------------------------------------------------


class TestScoreConfidence:
    def test_high_confidence(self):
        dim = score_confidence(0.95)
        assert dim.score == 0.95

    def test_low_confidence(self):
        dim = score_confidence(0.1)
        assert dim.score == 0.1

    def test_clamped(self):
        dim = score_confidence(1.5)
        assert dim.score == 1.0


# ---------------------------------------------------------------------------
# score_cost_efficiency
# ---------------------------------------------------------------------------


class TestScoreCostEfficiency:
    def test_zero_cost(self):
        """Free runs should score 1.0."""
        dim = score_cost_efficiency(0.0, 10)
        assert dim.score == 1.0

    def test_expensive(self):
        """$0.10/item should score 0.0."""
        dim = score_cost_efficiency(1.0, 10)  # $0.10/item
        assert dim.score == 0.0

    def test_moderate_cost(self):
        """$0.05/item should score 0.5."""
        dim = score_cost_efficiency(0.5, 10)  # $0.05/item
        assert dim.score == 0.5

    def test_zero_items(self):
        """Zero items should score 1.0 (no cost incurred per item)."""
        dim = score_cost_efficiency(5.0, 0)
        assert dim.score == 1.0

    def test_very_cheap(self):
        """$0.001/item should score near 1.0."""
        dim = score_cost_efficiency(0.01, 10)  # $0.001/item
        assert dim.score > 0.9


# ---------------------------------------------------------------------------
# compute_run_quality grades
# ---------------------------------------------------------------------------


class TestComputeRunQuality:
    def test_grade_a(self):
        """Perfect run should get grade A."""
        result = compute_run_quality("run1", pass_rate=1.0)
        assert result.grade == "A"
        assert result.overall_score >= 90

    def test_grade_b(self):
        """Run averaging ~85 should get grade B."""
        # All four dims average: (0.7+1.0+1.0+1.0)/4 = 0.925 -> too high
        # Need coverage < 1 and confidence < 1 to pull it down.
        result = compute_run_quality(
            "run2", pass_rate=0.7, avg_confidence=0.7,
            metrics_evaluated=1, total_metrics=1, cost_usd=0.02, item_count=1,
        )
        assert result.grade == "B"

    def test_grade_c(self):
        """Run averaging ~70-79 should get grade C."""
        result = compute_run_quality(
            "run3", pass_rate=0.5, avg_confidence=0.5,
            metrics_evaluated=1, total_metrics=1, cost_usd=0.02, item_count=1,
        )
        assert result.grade == "C"

    def test_grade_d(self):
        """Run averaging ~60-69 should get grade D."""
        result = compute_run_quality(
            "run4", pass_rate=0.4, avg_confidence=0.4,
            metrics_evaluated=1, total_metrics=1, cost_usd=0.02, item_count=1,
        )
        assert result.grade == "D"

    def test_grade_f(self):
        """Very low scores should get F."""
        result = compute_run_quality(
            "run5", pass_rate=0.1, avg_confidence=0.1,
            metrics_evaluated=0, total_metrics=1, cost_usd=0.10, item_count=1,
        )
        assert result.grade == "F"

    def test_dimensions_present(self):
        """Result should contain all four dimensions."""
        result = compute_run_quality("r", pass_rate=0.9)
        names = [d.name for d in result.dimensions]
        assert "pass_rate" in names
        assert "coverage" in names
        assert "confidence" in names
        assert "cost_efficiency" in names

    def test_recommendations_low_pass_rate(self):
        """Low pass rate should generate recommendation."""
        result = compute_run_quality("r", pass_rate=0.3)
        assert any("pass rate" in r.lower() for r in result.recommendations)

    def test_recommendations_low_confidence(self):
        """Low confidence should generate recommendation."""
        result = compute_run_quality("r", pass_rate=1.0, avg_confidence=0.3)
        assert any("confidence" in r.lower() for r in result.recommendations)

    def test_overall_score_range(self):
        """Overall score should be 0-100."""
        result = compute_run_quality("r", pass_rate=0.5)
        assert 0 <= result.overall_score <= 100


# ---------------------------------------------------------------------------
# compare_run_quality
# ---------------------------------------------------------------------------


class TestCompareRunQuality:
    def test_better_run(self):
        a = compute_run_quality("a", pass_rate=0.5)
        b = compute_run_quality("b", pass_rate=0.9)
        cmp = compare_run_quality(a, b)
        assert cmp["better"] == "b"
        assert cmp["overall_delta"] > 0

    def test_worse_run(self):
        a = compute_run_quality("a", pass_rate=0.9)
        b = compute_run_quality("b", pass_rate=0.5)
        cmp = compare_run_quality(a, b)
        assert cmp["better"] == "a"
        assert cmp["overall_delta"] < 0

    def test_tie(self):
        a = compute_run_quality("a", pass_rate=0.7)
        b = compute_run_quality("b", pass_rate=0.7)
        cmp = compare_run_quality(a, b)
        assert cmp["better"] == "tie"
        assert cmp["overall_delta"] == 0.0

    def test_dimension_deltas(self):
        a = compute_run_quality("a", pass_rate=0.5)
        b = compute_run_quality("b", pass_rate=0.9)
        cmp = compare_run_quality(a, b)
        assert "pass_rate" in cmp["dimension_deltas"]
        assert cmp["dimension_deltas"]["pass_rate"] > 0

    def test_grades_included(self):
        a = compute_run_quality("a", pass_rate=1.0)
        b = compute_run_quality("b", pass_rate=0.2)
        cmp = compare_run_quality(a, b)
        assert "grade_a" in cmp
        assert "grade_b" in cmp


# ---------------------------------------------------------------------------
# format_text
# ---------------------------------------------------------------------------


class TestRunQualityResultFormatText:
    def test_contains_grade(self):
        result = compute_run_quality("r", pass_rate=0.9)
        text = result.format_text()
        assert "Grade:" in text

    def test_contains_dimensions(self):
        result = compute_run_quality("r", pass_rate=0.9)
        text = result.format_text()
        assert "pass_rate" in text
        assert "coverage" in text

    def test_contains_recommendations(self):
        result = compute_run_quality("r", pass_rate=0.3)
        text = result.format_text()
        assert "Recommendations:" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundTrips:
    def test_quality_dimension_as_dict(self):
        dim = QualityDimension(name="test", score=0.8, weight=2.0, details="ok")
        d = dim.as_dict()
        assert d["name"] == "test"
        assert d["score"] == 0.8
        assert d["weight"] == 2.0

    def test_run_quality_result_roundtrip(self):
        result = compute_run_quality("r1", pass_rate=0.85)
        d = result.as_dict()
        result2 = RunQualityResult.from_dict(d)
        assert result2.run_id == result.run_id
        assert result2.overall_score == result.overall_score
        assert result2.grade == result.grade
        assert len(result2.dimensions) == len(result.dimensions)

    def test_from_dict_defaults(self):
        """from_dict with empty dict should use defaults."""
        result = RunQualityResult.from_dict({})
        assert result.run_id == ""
        assert result.overall_score == 0.0
        assert result.grade == ""

    def test_roundtrip_preserves_recommendations(self):
        result = compute_run_quality("r", pass_rate=0.3)
        d = result.as_dict()
        result2 = RunQualityResult.from_dict(d)
        assert result2.recommendations == result.recommendations


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_items(self):
        """Zero item_count should not crash."""
        result = compute_run_quality("r", pass_rate=0.0, item_count=0)
        assert result.overall_score >= 0

    def test_zero_cost(self):
        """Zero cost should give max cost efficiency."""
        result = compute_run_quality("r", pass_rate=0.5, cost_usd=0.0)
        cost_dim = [d for d in result.dimensions if d.name == "cost_efficiency"][0]
        assert cost_dim.score == 1.0

    def test_zero_total_metrics(self):
        """Zero total metrics should not crash."""
        result = compute_run_quality("r", pass_rate=0.5, total_metrics=0)
        cov = [d for d in result.dimensions if d.name == "coverage"][0]
        assert cov.score == 0.0
