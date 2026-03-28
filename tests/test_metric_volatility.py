"""Tests for the metric volatility index module."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SDK to path
SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.metric_volatility import (
    VolatilityMetric,
    VolatilityReport,
    compute_volatility_index,
    detect_volatility_trend,
    compute_max_swing,
    analyze_metric_volatility,
    analyze_all_volatility,
    render_volatility_chart,
)


# ---------------------------------------------------------------------------
# compute_volatility_index
# ---------------------------------------------------------------------------


class TestComputeVolatilityIndex:
    def test_constant_scores_zero(self):
        """All identical scores should yield 0.0 volatility."""
        assert compute_volatility_index([0.8, 0.8, 0.8, 0.8]) == 0.0

    def test_oscillating_scores_high(self):
        """Alternating high/low should yield high volatility."""
        result = compute_volatility_index([0.0, 1.0, 0.0, 1.0, 0.0])
        assert result > 0.8

    def test_single_score(self):
        """Single score should return 0.0."""
        assert compute_volatility_index([0.5]) == 0.0

    def test_two_scores(self):
        """Two different scores should return a valid index."""
        result = compute_volatility_index([0.0, 1.0])
        assert result == 1.0

    def test_two_identical_scores(self):
        """Two identical scores should return 0.0."""
        assert compute_volatility_index([0.5, 0.5]) == 0.0

    def test_empty_scores(self):
        """Empty list should return 0.0."""
        assert compute_volatility_index([]) == 0.0

    def test_gradually_increasing(self):
        """Linearly increasing scores should have moderate volatility."""
        scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        result = compute_volatility_index(scores)
        # Mean diff = 0.1, range = 0.5, so ratio = 0.2
        assert 0.1 < result < 0.3

    def test_capped_at_one(self):
        """Result should never exceed 1.0."""
        result = compute_volatility_index([0.0, 1.0])
        assert result <= 1.0

    def test_small_variations(self):
        """Small noise should yield low volatility."""
        scores = [0.50, 0.51, 0.49, 0.50, 0.52, 0.48]
        result = compute_volatility_index(scores)
        assert result < 0.6


# ---------------------------------------------------------------------------
# detect_volatility_trend
# ---------------------------------------------------------------------------


class TestDetectVolatilityTrend:
    def test_stable(self):
        """Uniform changes should be stable."""
        scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        assert detect_volatility_trend(scores) == "stable"

    def test_increasing(self):
        """Calm start, wild finish should be increasing."""
        scores = [0.5, 0.51, 0.5, 0.51, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0]
        result = detect_volatility_trend(scores)
        assert result == "increasing"

    def test_decreasing(self):
        """Wild start, calm finish should be decreasing."""
        scores = [0.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]
        result = detect_volatility_trend(scores)
        assert result == "decreasing"

    def test_too_few_scores(self):
        """Fewer than 4 scores should return stable."""
        assert detect_volatility_trend([0.1, 0.9, 0.1]) == "stable"

    def test_exactly_four_scores_stable(self):
        """Four scores with uniform diffs should be stable."""
        assert detect_volatility_trend([0.0, 0.1, 0.2, 0.3]) == "stable"


# ---------------------------------------------------------------------------
# compute_max_swing
# ---------------------------------------------------------------------------


class TestComputeMaxSwing:
    def test_basic(self):
        """Max swing should be the largest consecutive difference."""
        assert compute_max_swing([0.1, 0.5, 0.2, 0.9]) == 0.7

    def test_constant(self):
        """Constant scores should have 0.0 max swing."""
        assert compute_max_swing([0.5, 0.5, 0.5]) == 0.0

    def test_single_score(self):
        """Single score should return 0.0."""
        assert compute_max_swing([0.5]) == 0.0

    def test_empty(self):
        """Empty list should return 0.0."""
        assert compute_max_swing([]) == 0.0

    def test_two_scores(self):
        """Two scores: swing is the absolute difference."""
        assert abs(compute_max_swing([0.2, 0.8]) - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# analyze_metric_volatility
# ---------------------------------------------------------------------------


class TestAnalyzeMetricVolatility:
    def test_returns_volatility_metric(self):
        """Should return a complete VolatilityMetric."""
        result = analyze_metric_volatility("accuracy", [0.7, 0.8, 0.75, 0.9])
        assert result.metric_id == "accuracy"
        assert result.run_count == 4
        assert result.volatility_index >= 0.0
        assert result.max_swing > 0.0
        assert result.trend in ("stable", "increasing", "decreasing")

    def test_single_score_metric(self):
        """Single score should yield zero volatility."""
        result = analyze_metric_volatility("latency", [250.0])
        assert result.volatility_index == 0.0
        assert result.max_swing == 0.0
        assert result.run_count == 1


# ---------------------------------------------------------------------------
# analyze_all_volatility
# ---------------------------------------------------------------------------


class TestAnalyzeAllVolatility:
    def test_basic_report(self):
        """Should produce report with correct most/least volatile."""
        metric_scores = {
            "stable_metric": [0.8, 0.8, 0.8, 0.8],
            "volatile_metric": [0.0, 1.0, 0.0, 1.0],
        }
        report = analyze_all_volatility(metric_scores)
        assert report.most_volatile == "volatile_metric"
        assert report.least_volatile == "stable_metric"
        assert len(report.metrics) == 2
        assert report.avg_volatility > 0.0

    def test_empty_input(self):
        """Empty dict should produce empty report."""
        report = analyze_all_volatility({})
        assert report.metrics == []
        assert report.most_volatile == ""
        assert report.avg_volatility == 0.0

    def test_single_metric(self):
        """Single metric: most and least volatile should be same."""
        report = analyze_all_volatility({"m1": [0.5, 0.6, 0.7]})
        assert report.most_volatile == "m1"
        assert report.least_volatile == "m1"


# ---------------------------------------------------------------------------
# render_volatility_chart
# ---------------------------------------------------------------------------


class TestRenderVolatilityChart:
    def test_contains_metric_names(self):
        """Chart output should contain metric names."""
        report = analyze_all_volatility({
            "accuracy": [0.8, 0.8, 0.8],
            "latency": [0.0, 1.0, 0.0],
        })
        chart = render_volatility_chart(report)
        assert "accuracy" in chart
        assert "latency" in chart

    def test_contains_header(self):
        """Chart should contain a header."""
        report = analyze_all_volatility({"m1": [0.5, 0.6]})
        chart = render_volatility_chart(report)
        assert "Volatility Chart" in chart

    def test_empty_report(self):
        """Empty report should return placeholder text."""
        chart = render_volatility_chart(VolatilityReport())
        assert "No metrics" in chart

    def test_custom_width(self):
        """Custom width should be respected in output."""
        report = analyze_all_volatility({"m1": [0.0, 1.0]})
        chart = render_volatility_chart(report, width=40)
        # Each line should be within the width range
        for line in chart.split("\n"):
            # Header separator is exactly width
            if line.startswith("="):
                assert len(line) == 40


# ---------------------------------------------------------------------------
# VolatilityReport.format_text
# ---------------------------------------------------------------------------


class TestVolatilityReportFormatText:
    def test_format_text_contains_summary(self):
        """Format text should include summary stats."""
        report = analyze_all_volatility({
            "accuracy": [0.8, 0.8, 0.8],
            "safety": [0.0, 1.0, 0.0],
        })
        text = report.format_text()
        assert "METRIC VOLATILITY REPORT" in text
        assert "Average volatility" in text
        assert "Most volatile" in text
        assert "accuracy" in text
        assert "safety" in text

    def test_format_text_empty(self):
        """Empty report format should say no metrics."""
        text = VolatilityReport().format_text()
        assert "No metrics" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestVolatilityMetricRoundtrip:
    def test_roundtrip(self):
        """as_dict -> from_dict should preserve all fields."""
        original = VolatilityMetric(
            metric_id="accuracy",
            volatility_index=0.42,
            trend="increasing",
            run_count=10,
            max_swing=0.35,
        )
        restored = VolatilityMetric.from_dict(original.as_dict())
        assert restored.metric_id == original.metric_id
        assert restored.volatility_index == original.volatility_index
        assert restored.trend == original.trend
        assert restored.run_count == original.run_count
        assert restored.max_swing == original.max_swing


class TestVolatilityReportRoundtrip:
    def test_roundtrip(self):
        """as_dict -> from_dict should preserve all fields including nested metrics."""
        report = analyze_all_volatility({
            "m1": [0.5, 0.6, 0.7],
            "m2": [0.0, 1.0, 0.0],
        })
        restored = VolatilityReport.from_dict(report.as_dict())
        assert restored.most_volatile == report.most_volatile
        assert restored.least_volatile == report.least_volatile
        assert abs(restored.avg_volatility - report.avg_volatility) < 1e-9
        assert len(restored.metrics) == len(report.metrics)
        for orig, rest in zip(report.metrics, restored.metrics):
            assert orig.metric_id == rest.metric_id
            assert abs(orig.volatility_index - rest.volatility_index) < 1e-9
