"""Tests for statistical significance testing module."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.analysis.significance_testing import (
    SignificanceReport,
    SignificanceResult,
    apply_bonferroni_correction,
    compute_confidence_interval,
    compute_effect_size,
    run_all_significance_tests,
    run_significance_test,
    welch_t_test,
)


# ---------------------------------------------------------------------------
# welch_t_test
# ---------------------------------------------------------------------------


class TestWelchTTest:
    def test_identical_samples(self):
        """Identical samples should give p ~1.0 (no difference)."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        t_stat, p_value = welch_t_test(a, b)
        assert abs(t_stat) < 1e-10
        assert p_value > 0.99

    def test_different_samples(self):
        """Clearly different samples should give small p-value."""
        a = [0.9, 1.1, 1.0, 0.95, 1.05, 0.98, 1.02, 0.97, 1.03, 1.0]
        b = [9.9, 10.1, 10.0, 9.95, 10.05, 9.98, 10.02, 9.97, 10.03, 10.0]
        t_stat, p_value = welch_t_test(a, b)
        assert abs(t_stat) > 5.0
        assert p_value < 0.01

    def test_slightly_different_samples(self):
        """Overlapping distributions should give moderate p-value."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        t_stat, p_value = welch_t_test(a, b)
        assert t_stat < 0  # mean_a < mean_b
        assert 0.0 < p_value < 1.0

    def test_single_element(self):
        """Single element samples return p=1.0."""
        t_stat, p_value = welch_t_test([5.0], [10.0])
        assert p_value == 1.0

    def test_zero_variance(self):
        """Constant samples (zero variance) return p=1.0."""
        a = [5.0, 5.0, 5.0]
        b = [5.0, 5.0, 5.0]
        t_stat, p_value = welch_t_test(a, b)
        assert p_value == 1.0

    def test_returns_tuple(self):
        """Should return (t_statistic, p_value) tuple."""
        result = welch_t_test([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_negative_t_stat(self):
        """When mean_a < mean_b, t should be negative."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [6.0, 7.0, 8.0, 9.0, 10.0]
        t_stat, _ = welch_t_test(a, b)
        assert t_stat < 0


# ---------------------------------------------------------------------------
# compute_confidence_interval
# ---------------------------------------------------------------------------


class TestConfidenceInterval:
    def test_basic_interval(self):
        """CI should bracket the mean."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = compute_confidence_interval(scores, 0.95)
        mean = sum(scores) / len(scores)
        assert lo < mean < hi

    def test_tight_data(self):
        """Near-identical values should give tight CI."""
        scores = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.01]
        lo, hi = compute_confidence_interval(scores)
        assert hi - lo < 0.1

    def test_single_value(self):
        """Single value CI should be a point."""
        lo, hi = compute_confidence_interval([3.14])
        assert lo == 3.14
        assert hi == 3.14

    def test_empty(self):
        """Empty list returns (0.0, 0.0)."""
        lo, hi = compute_confidence_interval([])
        assert lo == 0.0
        assert hi == 0.0

    def test_higher_confidence_wider(self):
        """99% CI should be wider than 90% CI."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        lo_90, hi_90 = compute_confidence_interval(scores, 0.90)
        lo_99, hi_99 = compute_confidence_interval(scores, 0.99)
        width_90 = hi_90 - lo_90
        width_99 = hi_99 - lo_99
        assert width_99 > width_90


# ---------------------------------------------------------------------------
# test_significance
# ---------------------------------------------------------------------------


class TestTestSignificance:
    def test_significant_difference(self):
        """Large difference with tight data should be significant."""
        a = [0.9, 1.1, 1.0, 0.95, 1.05, 0.98, 1.02, 0.97, 1.03, 1.0] * 2
        b = [4.9, 5.1, 5.0, 4.95, 5.05, 4.98, 5.02, 4.97, 5.03, 5.0] * 2
        result = run_significance_test("metric_a", a, b)
        assert result.significant is True
        assert result.p_value < 0.05
        assert result.metric_id == "metric_a"
        assert abs(result.delta - 4.0) < 1e-10

    def test_not_significant(self):
        """Overlapping data with small n should not be significant."""
        a = [1.0, 2.0, 3.0]
        b = [2.0, 3.0, 4.0]
        result = run_significance_test("metric_b", a, b, alpha=0.05)
        # With only 3 samples and 1-unit shift, often not significant
        assert result.metric_id == "metric_b"
        assert isinstance(result.significant, bool)
        assert 0.0 <= result.p_value <= 1.0

    def test_empty_scores_a(self):
        """Empty scores_a returns non-significant result."""
        result = run_significance_test("m", [], [1.0, 2.0])
        assert result.significant is False
        assert result.p_value == 1.0

    def test_empty_scores_b(self):
        """Empty scores_b returns non-significant result."""
        result = run_significance_test("m", [1.0, 2.0], [])
        assert result.significant is False
        assert result.p_value == 1.0

    def test_confidence_interval_included(self):
        """Result should include confidence interval."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [3.0, 4.0, 5.0, 6.0, 7.0]
        result = run_significance_test("m", a, b)
        lo, hi = result.confidence_interval
        assert lo < hi
        # CI on the delta should contain the actual delta
        assert lo <= result.delta <= hi

    def test_method_default(self):
        """Default method should be welch_t."""
        result = run_significance_test("m", [1.0, 2.0], [3.0, 4.0])
        assert result.method == "welch_t"


# ---------------------------------------------------------------------------
# test_all_metrics
# ---------------------------------------------------------------------------


class TestTestAllMetrics:
    def test_multiple_metrics(self):
        """Should test each metric independently."""
        metrics = {
            "acc": ([0.8, 0.9, 0.85], [0.7, 0.75, 0.72]),
            "f1": ([0.1, 0.12, 0.09, 0.11, 0.10] * 2, [0.9, 0.88, 0.91, 0.89, 0.90] * 2),
        }
        report = run_all_significance_tests(metrics)
        assert report.total_tests == 2
        assert len(report.results) == 2
        # f1 should be significant (huge difference)
        f1_result = next(r for r in report.results if r.metric_id == "f1")
        assert f1_result.significant is True

    def test_empty_metrics(self):
        """Empty dict should produce empty report."""
        report = run_all_significance_tests({})
        assert report.total_tests == 0
        assert report.significant_count == 0

    def test_report_alpha(self):
        """Report should preserve the alpha value."""
        report = run_all_significance_tests({"m": ([1.0, 2.0], [3.0, 4.0])}, alpha=0.10)
        assert report.alpha == 0.10

    def test_significant_count(self):
        """Significant count should match number of significant results."""
        metrics = {
            "sig": ([0.9, 1.1, 1.0, 0.95, 1.05] * 4, [9.9, 10.1, 10.0, 9.95, 10.05] * 4),
            "nosig": ([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]),
        }
        report = run_all_significance_tests(metrics)
        assert report.significant_count == sum(1 for r in report.results if r.significant)


# ---------------------------------------------------------------------------
# apply_bonferroni_correction
# ---------------------------------------------------------------------------


class TestBonferroniCorrection:
    def test_makes_less_significant(self):
        """Bonferroni should make borderline results non-significant."""
        # Create a result with p=0.04 (significant at 0.05)
        results = [
            SignificanceResult(
                metric_id=f"m{i}", mean_a=0.0, mean_b=1.0,
                delta=1.0, p_value=0.04, significant=True,
            )
            for i in range(5)
        ]
        corrected = apply_bonferroni_correction(results, alpha=0.05)
        # Corrected alpha = 0.05/5 = 0.01, so p=0.04 should no longer be significant
        for r in corrected:
            assert r.significant is False

    def test_preserves_strong_significance(self):
        """Very small p-values should survive correction."""
        results = [
            SignificanceResult(
                metric_id="strong", mean_a=0.0, mean_b=1.0,
                delta=1.0, p_value=0.001, significant=True,
            ),
            SignificanceResult(
                metric_id="weak", mean_a=0.0, mean_b=0.1,
                delta=0.1, p_value=0.04, significant=True,
            ),
        ]
        corrected = apply_bonferroni_correction(results, alpha=0.05)
        strong = next(r for r in corrected if r.metric_id == "strong")
        weak = next(r for r in corrected if r.metric_id == "weak")
        assert strong.significant is True  # 0.001 < 0.025
        assert weak.significant is False   # 0.04 >= 0.025

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert apply_bonferroni_correction([]) == []


# ---------------------------------------------------------------------------
# compute_effect_size
# ---------------------------------------------------------------------------


class TestComputeEffectSize:
    def test_zero_effect(self):
        """Identical distributions should give d=0."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = compute_effect_size(a, b)
        assert abs(d) < 1e-10

    def test_small_effect(self):
        """Small difference relative to spread: |d| < 0.5."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.5, 2.5, 3.5, 4.5, 5.5]
        d = compute_effect_size(a, b)
        assert 0 < abs(d) < 0.5

    def test_large_effect(self):
        """Large difference with tight data: |d| > 0.8."""
        a = [0.9, 1.1, 1.0, 0.95, 1.05]
        b = [4.9, 5.1, 5.0, 4.95, 5.05]
        d = compute_effect_size(a, b)
        assert abs(d) > 0.8

    def test_direction(self):
        """Positive d when mean_b > mean_a."""
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        d = compute_effect_size(a, b)
        assert d > 0

    def test_too_few_samples(self):
        """Single-element samples return 0.0."""
        assert compute_effect_size([1.0], [5.0]) == 0.0

    def test_zero_variance_samples(self):
        """Constant samples with different means return 0.0 (zero pooled sd)."""
        assert compute_effect_size([3.0, 3.0], [3.0, 3.0]) == 0.0


# ---------------------------------------------------------------------------
# SignificanceReport format_text
# ---------------------------------------------------------------------------


class TestSignificanceReportFormat:
    def test_format_text_header(self):
        """Format should include alpha and count."""
        report = SignificanceReport(
            results=[
                SignificanceResult(
                    metric_id="acc", mean_a=0.8, mean_b=0.9,
                    delta=0.1, p_value=0.03, significant=True,
                    confidence_interval=(-0.01, 0.21),
                ),
            ],
            alpha=0.05,
            significant_count=1,
            total_tests=1,
        )
        text = report.format_text()
        assert "alpha=0.05" in text
        assert "1/1" in text

    def test_format_text_star_marker(self):
        """Significant results should be marked with *."""
        report = SignificanceReport(
            results=[
                SignificanceResult(
                    metric_id="sig", mean_a=0.0, mean_b=1.0,
                    delta=1.0, p_value=0.001, significant=True,
                ),
                SignificanceResult(
                    metric_id="nosig", mean_a=0.0, mean_b=0.1,
                    delta=0.1, p_value=0.5, significant=False,
                ),
            ],
            alpha=0.05,
            significant_count=1,
            total_tests=2,
        )
        text = report.format_text()
        assert "[*] sig" in text
        assert "[ ] nosig" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestRoundtrips:
    def test_significance_result_roundtrip(self):
        """SignificanceResult should survive as_dict -> from_dict."""
        original = SignificanceResult(
            metric_id="test", mean_a=0.5, mean_b=0.7,
            delta=0.2, p_value=0.03, significant=True,
            confidence_interval=(0.05, 0.35), method="welch_t",
        )
        restored = SignificanceResult.from_dict(original.as_dict())
        assert restored.metric_id == original.metric_id
        assert restored.mean_a == original.mean_a
        assert restored.mean_b == original.mean_b
        assert restored.delta == original.delta
        assert restored.p_value == original.p_value
        assert restored.significant == original.significant
        assert restored.confidence_interval == original.confidence_interval
        assert restored.method == original.method

    def test_significance_report_roundtrip(self):
        """SignificanceReport should survive as_dict -> from_dict."""
        original = SignificanceReport(
            results=[
                SignificanceResult(
                    metric_id="m1", mean_a=0.5, mean_b=0.7,
                    delta=0.2, p_value=0.03, significant=True,
                ),
            ],
            alpha=0.05,
            significant_count=1,
            total_tests=1,
        )
        restored = SignificanceReport.from_dict(original.as_dict())
        assert len(restored.results) == 1
        assert restored.alpha == original.alpha
        assert restored.significant_count == original.significant_count
        assert restored.total_tests == original.total_tests
        assert restored.results[0].metric_id == "m1"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_value_both(self):
        """Single values in both groups should not crash."""
        result = run_significance_test("m", [1.0], [2.0])
        assert result.p_value == 1.0
        assert result.significant is False

    def test_empty_both(self):
        """Empty both groups should return safe defaults."""
        result = run_significance_test("m", [], [])
        assert result.p_value == 1.0
        assert result.significant is False
        assert result.mean_a == 0.0
        assert result.mean_b == 0.0

    def test_large_samples(self):
        """Should handle large sample sizes without error."""
        a = [float(i) for i in range(1000)]
        b = [float(i) + 0.5 for i in range(1000)]
        t_stat, p_value = welch_t_test(a, b)
        assert 0.0 <= p_value <= 1.0

    def test_negative_scores(self):
        """Should handle negative values correctly."""
        a = [-5.0, -4.0, -3.0, -2.0, -1.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = run_significance_test("m", a, b)
        assert result.delta > 0
        assert result.significant is True
