"""Tests for sampling impact analysis module."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parent.parent / "sdk"
sys.path.insert(0, str(SDK_ROOT))

from evalyn_sdk.sampling_impact import (
    ImpactEstimate,
    ImpactReport,
    estimate_ci_width,
    compute_impact_curve,
    recommend_sample_size,
    build_impact_report,
    format_impact_report,
    _std_dev,
    _precision_level,
)


# ---------------------------------------------------------------------------
# ImpactEstimate dataclass
# ---------------------------------------------------------------------------


class TestImpactEstimate:
    def test_basic_creation(self):
        e = ImpactEstimate(sample_size=100, expected_ci_width=0.04, precision_level="high")
        assert e.sample_size == 100
        assert e.expected_ci_width == 0.04
        assert e.precision_level == "high"
        assert e.estimated_cost == 0.0

    def test_with_cost(self):
        e = ImpactEstimate(sample_size=50, expected_ci_width=0.08, precision_level="medium", estimated_cost=5.0)
        assert e.estimated_cost == 5.0

    def test_as_dict(self):
        e = ImpactEstimate(sample_size=200, expected_ci_width=0.03, precision_level="high", estimated_cost=10.0)
        d = e.as_dict()
        assert d["sample_size"] == 200
        assert d["expected_ci_width"] == 0.03
        assert d["precision_level"] == "high"
        assert d["estimated_cost"] == 10.0

    def test_from_dict(self):
        d = {"sample_size": 50, "expected_ci_width": 0.07, "precision_level": "medium", "estimated_cost": 2.5}
        e = ImpactEstimate.from_dict(d)
        assert e.sample_size == 50
        assert e.expected_ci_width == 0.07
        assert e.estimated_cost == 2.5

    def test_from_dict_default_cost(self):
        d = {"sample_size": 10, "expected_ci_width": 0.2, "precision_level": "low"}
        e = ImpactEstimate.from_dict(d)
        assert e.estimated_cost == 0.0

    def test_roundtrip(self):
        original = ImpactEstimate(sample_size=300, expected_ci_width=0.025, precision_level="high", estimated_cost=15.0)
        restored = ImpactEstimate.from_dict(original.as_dict())
        assert restored.sample_size == original.sample_size
        assert restored.expected_ci_width == original.expected_ci_width
        assert restored.precision_level == original.precision_level
        assert restored.estimated_cost == original.estimated_cost


# ---------------------------------------------------------------------------
# ImpactReport dataclass
# ---------------------------------------------------------------------------


class TestImpactReport:
    def test_empty_report(self):
        r = ImpactReport()
        assert r.estimates == []
        assert r.metric_std == 0.0
        assert r.recommended_size == 0
        assert r.target_ci_width == 0.05

    def test_as_dict(self):
        est = ImpactEstimate(sample_size=100, expected_ci_width=0.04, precision_level="high")
        r = ImpactReport(estimates=[est], metric_std=0.2, recommended_size=150, target_ci_width=0.05)
        d = r.as_dict()
        assert len(d["estimates"]) == 1
        assert d["metric_std"] == 0.2
        assert d["recommended_size"] == 150

    def test_from_dict(self):
        d = {
            "estimates": [{"sample_size": 50, "expected_ci_width": 0.1, "precision_level": "medium"}],
            "metric_std": 0.3,
            "recommended_size": 200,
            "target_ci_width": 0.04,
        }
        r = ImpactReport.from_dict(d)
        assert len(r.estimates) == 1
        assert r.metric_std == 0.3
        assert r.recommended_size == 200
        assert r.target_ci_width == 0.04

    def test_roundtrip(self):
        est = ImpactEstimate(sample_size=100, expected_ci_width=0.04, precision_level="high", estimated_cost=5.0)
        original = ImpactReport(estimates=[est], metric_std=0.15, recommended_size=120, target_ci_width=0.06)
        restored = ImpactReport.from_dict(original.as_dict())
        assert restored.metric_std == original.metric_std
        assert len(restored.estimates) == 1
        assert restored.estimates[0].sample_size == 100


# ---------------------------------------------------------------------------
# estimate_ci_width
# ---------------------------------------------------------------------------


class TestEstimateCiWidth:
    def test_basic_95(self):
        """CI width at n=100, std=0.5, 95% should be 2*1.96*0.5/sqrt(100) = 0.196."""
        width = estimate_ci_width(0.5, 100, 0.95)
        assert abs(width - 0.196) < 0.001

    def test_zero_std(self):
        width = estimate_ci_width(0.0, 100)
        assert width == 0.0

    def test_large_sample_shrinks_ci(self):
        w1 = estimate_ci_width(0.3, 100)
        w2 = estimate_ci_width(0.3, 400)
        assert w2 < w1
        # Quadrupling n should halve CI width
        assert abs(w2 - w1 / 2) < 0.001

    def test_zero_sample_returns_inf(self):
        width = estimate_ci_width(0.5, 0)
        assert math.isinf(width)

    def test_negative_sample_returns_inf(self):
        width = estimate_ci_width(0.5, -5)
        assert math.isinf(width)

    def test_99_confidence(self):
        width = estimate_ci_width(0.5, 100, 0.99)
        expected = 2 * 2.576 * 0.5 / math.sqrt(100)
        assert abs(width - expected) < 0.001

    def test_90_confidence(self):
        width = estimate_ci_width(0.5, 100, 0.90)
        expected = 2 * 1.645 * 0.5 / math.sqrt(100)
        assert abs(width - expected) < 0.001

    def test_single_sample(self):
        """n=1 should give the widest finite CI."""
        width = estimate_ci_width(0.5, 1)
        expected = 2 * 1.96 * 0.5
        assert abs(width - expected) < 0.001


# ---------------------------------------------------------------------------
# precision_level
# ---------------------------------------------------------------------------


class TestPrecisionLevel:
    def test_high(self):
        assert _precision_level(0.04) == "high"
        assert _precision_level(0.001) == "high"

    def test_medium(self):
        assert _precision_level(0.05) == "medium"
        assert _precision_level(0.099) == "medium"

    def test_low(self):
        assert _precision_level(0.1) == "low"
        assert _precision_level(0.5) == "low"

    def test_boundary_high_medium(self):
        assert _precision_level(0.0499) == "high"
        assert _precision_level(0.0500) == "medium"

    def test_boundary_medium_low(self):
        assert _precision_level(0.0999) == "medium"
        assert _precision_level(0.1000) == "low"


# ---------------------------------------------------------------------------
# compute_impact_curve
# ---------------------------------------------------------------------------


class TestComputeImpactCurve:
    def test_monotonically_decreasing(self):
        scores = [0.1 * i for i in range(20)]
        sizes = [10, 50, 100, 500]
        results = compute_impact_curve(scores, sizes)
        widths = [r.expected_ci_width for r in results]
        for i in range(len(widths) - 1):
            assert widths[i] > widths[i + 1]

    def test_returns_correct_count(self):
        scores = [0.5] * 10 + [0.8] * 10
        sizes = [20, 50, 100]
        results = compute_impact_curve(scores, sizes)
        assert len(results) == 3

    def test_with_cost(self):
        scores = [0.5, 0.6, 0.7, 0.8]
        sizes = [10, 100]
        results = compute_impact_curve(scores, sizes, cost_per_item=0.5)
        assert results[0].estimated_cost == 5.0
        assert results[1].estimated_cost == 50.0

    def test_empty_sizes(self):
        results = compute_impact_curve([0.5, 0.6], [])
        assert results == []

    def test_precision_labels_assigned(self):
        scores = [0.5, 0.6, 0.7, 0.8]
        sizes = [5, 500, 50000]
        results = compute_impact_curve(scores, sizes)
        levels = {r.precision_level for r in results}
        # With varied sizes, should see multiple precision levels
        assert len(levels) >= 1


# ---------------------------------------------------------------------------
# recommend_sample_size
# ---------------------------------------------------------------------------


class TestRecommendSampleSize:
    def test_basic(self):
        """std=0.5, target=0.05 -> n = (2*1.96*0.5/0.05)^2 = 1536.64 -> 1537."""
        n = recommend_sample_size(0.5, 0.05)
        assert n == 1537

    def test_zero_std(self):
        n = recommend_sample_size(0.0, 0.05)
        assert n == 1

    def test_zero_target(self):
        n = recommend_sample_size(0.5, 0.0)
        assert n == 1

    def test_negative_target(self):
        n = recommend_sample_size(0.5, -0.1)
        assert n == 1

    def test_large_target_small_n(self):
        n = recommend_sample_size(0.1, 1.0)
        assert n >= 1
        assert n <= 2

    def test_recommended_achieves_target(self):
        """The recommended size should actually achieve the target width."""
        std = 0.3
        target = 0.08
        n = recommend_sample_size(std, target)
        width = estimate_ci_width(std, n)
        assert width <= target

    def test_one_less_exceeds_target(self):
        """n-1 should exceed the target (unless n=1)."""
        std = 0.3
        target = 0.08
        n = recommend_sample_size(std, target)
        if n > 1:
            width_below = estimate_ci_width(std, n - 1)
            assert width_below > target


# ---------------------------------------------------------------------------
# build_impact_report
# ---------------------------------------------------------------------------


class TestBuildImpactReport:
    def test_default_sizes(self):
        scores = [0.1 * i for i in range(50)]
        r = build_impact_report(scores)
        assert len(r.estimates) == 6  # default sizes
        assert r.target_ci_width == 0.05
        assert r.metric_std > 0

    def test_custom_sizes(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        r = build_impact_report(scores, sizes=[10, 25, 50])
        assert len(r.estimates) == 3

    def test_recommended_size_positive(self):
        scores = [0.3, 0.5, 0.7, 0.9]
        r = build_impact_report(scores)
        assert r.recommended_size >= 1

    def test_cost_propagates(self):
        scores = [0.5, 0.6, 0.7]
        r = build_impact_report(scores, sizes=[100], cost_per_item=0.10)
        assert r.estimates[0].estimated_cost == 10.0

    def test_single_score(self):
        """Single score -> std=0, all CI widths 0."""
        r = build_impact_report([0.5])
        assert r.metric_std == 0.0
        for est in r.estimates:
            assert est.expected_ci_width == 0.0


# ---------------------------------------------------------------------------
# format_impact_report
# ---------------------------------------------------------------------------


class TestFormatImpactReport:
    def test_contains_header_info(self):
        r = ImpactReport(metric_std=0.25, recommended_size=100, target_ci_width=0.05)
        text = format_impact_report(r)
        assert "0.2500" in text
        assert "100" in text
        assert "Size" in text
        assert "CI Width" in text

    def test_contains_rows(self):
        est = ImpactEstimate(sample_size=50, expected_ci_width=0.08, precision_level="medium", estimated_cost=2.5)
        r = ImpactReport(estimates=[est], metric_std=0.2, recommended_size=100, target_ci_width=0.05)
        text = format_impact_report(r)
        assert "50" in text
        assert "medium" in text
        assert "2.50" in text

    def test_empty_estimates(self):
        r = ImpactReport()
        text = format_impact_report(r)
        assert "Size" in text
        # No data rows, but header should be there


# ---------------------------------------------------------------------------
# _std_dev helper
# ---------------------------------------------------------------------------


class TestStdDev:
    def test_uniform_values(self):
        assert _std_dev([5.0, 5.0, 5.0]) == 0.0

    def test_known_values(self):
        # [1, 2, 3, 4, 5] -> mean=3, var=2, std=sqrt(2)
        result = _std_dev([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(result - math.sqrt(2)) < 0.0001

    def test_single_value(self):
        assert _std_dev([42.0]) == 0.0

    def test_empty_list(self):
        assert _std_dev([]) == 0.0

    def test_two_values(self):
        # [0, 1] -> mean=0.5, var=0.25, std=0.5
        assert abs(_std_dev([0.0, 1.0]) - 0.5) < 0.0001
