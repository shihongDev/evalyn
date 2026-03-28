"""Tests for cross-run stability analysis."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.cross_run_stability import (
    StabilityMetric,
    StabilityReport,
    compute_stability,
    analyze_stability,
    detect_flaky_metrics,
    compute_reproducibility_score,
    render_stability_chart,
    suggest_stability_improvements,
)


# ---------------------------------------------------------------------------
# compute_stability
# ---------------------------------------------------------------------------


class TestComputeStability:
    def test_stable_low_cv(self):
        scores = [0.90, 0.91, 0.89, 0.90, 0.90]
        result = compute_stability("m1", scores)
        assert result.stability_label == "stable"
        assert result.coefficient_of_variation < 0.05

    def test_unstable_high_cv(self):
        scores = [0.1, 0.9, 0.2, 0.8, 0.15]
        result = compute_stability("m2", scores)
        assert result.stability_label == "unstable"
        assert result.coefficient_of_variation >= 0.15

    def test_moderate_cv(self):
        scores = [0.70, 0.75, 0.80, 0.65, 0.78]
        result = compute_stability("m3", scores)
        assert result.stability_label in ("stable", "moderate")
        assert result.coefficient_of_variation < 0.15

    def test_mean_calculation(self):
        scores = [0.4, 0.6]
        result = compute_stability("m", scores)
        assert abs(result.mean_score - 0.5) < 1e-6

    def test_std_dev_calculation(self):
        scores = [0.4, 0.6]
        result = compute_stability("m", scores)
        # std = sqrt(((0.4-0.5)^2 + (0.6-0.5)^2)/2) = sqrt(0.01) = 0.1
        assert abs(result.std_dev - 0.1) < 1e-6

    def test_min_max(self):
        scores = [0.3, 0.5, 0.7]
        result = compute_stability("m", scores)
        assert result.min_score == 0.3
        assert result.max_score == 0.7

    def test_num_runs(self):
        scores = [0.5, 0.5, 0.5]
        result = compute_stability("m", scores)
        assert result.num_runs == 3

    def test_empty_scores(self):
        result = compute_stability("m", [])
        assert result.num_runs == 0
        assert result.mean_score == 0.0

    def test_constant_scores_stable(self):
        scores = [0.8, 0.8, 0.8, 0.8]
        result = compute_stability("m", scores)
        assert result.stability_label == "stable"
        assert result.std_dev == 0.0
        assert result.coefficient_of_variation == 0.0

    def test_single_run(self):
        result = compute_stability("m", [0.75])
        assert result.num_runs == 1
        assert result.mean_score == 0.75
        assert result.std_dev == 0.0


# ---------------------------------------------------------------------------
# analyze_stability
# ---------------------------------------------------------------------------


class TestAnalyzeStability:
    def test_all_metrics_analyzed(self):
        data = {
            "a": [0.9, 0.9, 0.9],
            "b": [0.5, 0.5, 0.5],
        }
        report = analyze_stability(data)
        assert len(report.metrics) == 2

    def test_overall_stability_label(self):
        data = {"a": [0.9, 0.9, 0.9]}
        report = analyze_stability(data)
        assert report.overall_stability == "stable"

    def test_avg_cv(self):
        data = {"a": [0.5, 0.5, 0.5]}  # CV=0
        report = analyze_stability(data)
        assert report.avg_cv == 0.0

    def test_empty_input(self):
        report = analyze_stability({})
        assert len(report.metrics) == 0
        assert report.most_stable == ""

    def test_most_and_least_stable(self):
        data = {
            "steady": [0.9, 0.9, 0.9, 0.9],
            "wobbly": [0.1, 0.9, 0.2, 0.8],
        }
        report = analyze_stability(data)
        assert report.most_stable == "steady"
        assert report.least_stable == "wobbly"


# ---------------------------------------------------------------------------
# detect_flaky_metrics
# ---------------------------------------------------------------------------


class TestDetectFlakyMetrics:
    def test_finds_unstable(self):
        data = {
            "ok": [0.9, 0.9, 0.9],
            "flaky": [0.1, 0.9, 0.2, 0.8],
        }
        report = analyze_stability(data)
        flaky = detect_flaky_metrics(report)
        assert "flaky" in flaky
        assert "ok" not in flaky

    def test_no_flaky(self):
        data = {"a": [0.5, 0.5, 0.5]}
        report = analyze_stability(data)
        assert detect_flaky_metrics(report) == []


# ---------------------------------------------------------------------------
# compute_reproducibility_score
# ---------------------------------------------------------------------------


class TestComputeReproducibilityScore:
    def test_perfect_reproducibility(self):
        data = {"a": [0.8, 0.8, 0.8]}
        report = analyze_stability(data)
        score = compute_reproducibility_score(report)
        assert abs(score - 1.0) < 1e-9

    def test_lower_reproducibility(self):
        data = {"a": [0.1, 0.9, 0.2, 0.8]}
        report = analyze_stability(data)
        score = compute_reproducibility_score(report)
        assert 0.0 <= score <= 1.0
        assert score < 1.0

    def test_clamped_to_zero(self):
        # Construct a report with very high avg_cv manually
        report = StabilityReport(avg_cv=2.0)
        score = compute_reproducibility_score(report)
        assert score == 0.0


# ---------------------------------------------------------------------------
# render_stability_chart
# ---------------------------------------------------------------------------


class TestRenderStabilityChart:
    def test_contains_metric_names(self):
        data = {"alpha": [0.9, 0.9], "beta": [0.5, 0.6]}
        report = analyze_stability(data)
        chart = render_stability_chart(report)
        assert "alpha" in chart
        assert "beta" in chart

    def test_contains_header(self):
        data = {"a": [0.5, 0.5]}
        report = analyze_stability(data)
        chart = render_stability_chart(report)
        assert "STABILITY CHART" in chart

    def test_empty_report(self):
        report = StabilityReport()
        chart = render_stability_chart(report)
        assert "No metrics" in chart


# ---------------------------------------------------------------------------
# suggest_stability_improvements
# ---------------------------------------------------------------------------


class TestSuggestStabilityImprovements:
    def test_suggestions_for_unstable(self):
        data = {"flaky": [0.1, 0.9, 0.2, 0.8]}
        report = analyze_stability(data)
        suggestions = suggest_stability_improvements(report)
        assert len(suggestions) >= 1
        assert "flaky" in suggestions[0]

    def test_no_suggestions_for_stable(self):
        data = {"solid": [0.9, 0.9, 0.9]}
        report = analyze_stability(data)
        suggestions = suggest_stability_improvements(report)
        assert len(suggestions) == 0


# ---------------------------------------------------------------------------
# StabilityReport format_text
# ---------------------------------------------------------------------------


class TestStabilityReportFormatText:
    def test_format_text_contains_header(self):
        data = {"a": [0.5, 0.6], "b": [0.9, 0.9]}
        report = analyze_stability(data)
        text = report.format_text()
        assert "CROSS-RUN STABILITY REPORT" in text

    def test_format_text_contains_metric_ids(self):
        data = {"alpha": [0.5, 0.6], "beta": [0.9, 0.9]}
        report = analyze_stability(data)
        text = report.format_text()
        assert "alpha" in text
        assert "beta" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_stability_metric_roundtrip(self):
        original = StabilityMetric(
            metric_id="x",
            mean_score=0.75,
            std_dev=0.05,
            coefficient_of_variation=0.067,
            min_score=0.7,
            max_score=0.8,
            num_runs=4,
            stability_label="moderate",
        )
        d = original.as_dict()
        restored = StabilityMetric.from_dict(d)
        assert restored.metric_id == original.metric_id
        assert restored.mean_score == original.mean_score
        assert restored.stability_label == original.stability_label

    def test_stability_report_roundtrip(self):
        data = {"a": [0.9, 0.9, 0.9], "b": [0.5, 0.6, 0.7]}
        report = analyze_stability(data)
        d = report.as_dict()
        restored = StabilityReport.from_dict(d)
        assert restored.overall_stability == report.overall_stability
        assert restored.most_stable == report.most_stable
        assert restored.least_stable == report.least_stable
        assert len(restored.metrics) == len(report.metrics)

    def test_empty_report_roundtrip(self):
        report = StabilityReport()
        d = report.as_dict()
        restored = StabilityReport.from_dict(d)
        assert restored.metrics == []
        assert restored.most_stable == ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_run_all_metrics(self):
        data = {"a": [0.5], "b": [0.9]}
        report = analyze_stability(data)
        assert len(report.metrics) == 2
        for m in report.metrics:
            assert m.num_runs == 1

    def test_constant_scores_all_metrics(self):
        data = {"a": [0.8, 0.8, 0.8], "b": [0.3, 0.3, 0.3]}
        report = analyze_stability(data)
        for m in report.metrics:
            assert m.stability_label == "stable"
            assert m.std_dev < 1e-9
        assert report.avg_cv < 1e-9

    def test_zero_mean_scores(self):
        """Zero mean scores should not cause division by zero."""
        result = compute_stability("z", [0.0, 0.0, 0.0])
        assert result.mean_score == 0.0
        assert result.coefficient_of_variation == 0.0
        assert result.stability_label == "stable"
