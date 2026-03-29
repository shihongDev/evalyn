"""Tests for bootstrap resampling module."""

from __future__ import annotations

import math
import random

import pytest

from evalyn_sdk.bootstrap_resampling import (
    BootstrapCI,
    BootstrapConfig,
    BootstrapReport,
    bootstrap_all_metrics,
    bootstrap_metric,
    compute_bootstrap_distribution,
    compute_confidence_interval,
    format_bootstrap_report,
    percentile,
    resample,
)


# ---- BootstrapConfig ----


class TestBootstrapConfig:
    def test_defaults(self):
        c = BootstrapConfig()
        assert c.n_iterations == 1000
        assert c.confidence_level == 0.95
        assert c.seed == 42

    def test_custom_values(self):
        c = BootstrapConfig(n_iterations=500, confidence_level=0.99, seed=7)
        assert c.n_iterations == 500
        assert c.confidence_level == 0.99
        assert c.seed == 7

    def test_as_dict(self):
        c = BootstrapConfig(n_iterations=200, confidence_level=0.90, seed=1)
        d = c.as_dict()
        assert d == {
            "n_iterations": 200,
            "confidence_level": 0.90,
            "seed": 1,
        }

    def test_from_dict_full(self):
        d = {"n_iterations": 300, "confidence_level": 0.80, "seed": 99}
        c = BootstrapConfig.from_dict(d)
        assert c.n_iterations == 300
        assert c.confidence_level == 0.80
        assert c.seed == 99

    def test_from_dict_defaults(self):
        c = BootstrapConfig.from_dict({})
        assert c.n_iterations == 1000
        assert c.confidence_level == 0.95
        assert c.seed == 42

    def test_roundtrip(self):
        c = BootstrapConfig(n_iterations=50, confidence_level=0.99, seed=13)
        assert BootstrapConfig.from_dict(c.as_dict()) == c


# ---- BootstrapCI ----


class TestBootstrapCI:
    def test_creation(self):
        ci = BootstrapCI(
            metric_id="accuracy",
            point_estimate=0.85,
            ci_lower=0.80,
            ci_upper=0.90,
            std_error=0.025,
            n_iterations=1000,
        )
        assert ci.metric_id == "accuracy"
        assert ci.point_estimate == 0.85
        assert ci.ci_lower == 0.80
        assert ci.ci_upper == 0.90
        assert ci.std_error == 0.025
        assert ci.n_iterations == 1000

    def test_as_dict(self):
        ci = BootstrapCI(
            metric_id="f1",
            point_estimate=0.75,
            ci_lower=0.70,
            ci_upper=0.80,
            std_error=0.03,
            n_iterations=500,
        )
        d = ci.as_dict()
        assert d["metric_id"] == "f1"
        assert d["point_estimate"] == 0.75
        assert d["n_iterations"] == 500

    def test_from_dict(self):
        d = {
            "metric_id": "recall",
            "point_estimate": 0.9,
            "ci_lower": 0.85,
            "ci_upper": 0.95,
            "std_error": 0.02,
            "n_iterations": 200,
        }
        ci = BootstrapCI.from_dict(d)
        assert ci.metric_id == "recall"
        assert ci.ci_lower == 0.85

    def test_roundtrip(self):
        ci = BootstrapCI(
            metric_id="prec",
            point_estimate=0.6,
            ci_lower=0.5,
            ci_upper=0.7,
            std_error=0.05,
            n_iterations=100,
        )
        assert BootstrapCI.from_dict(ci.as_dict()) == ci


# ---- BootstrapReport ----


class TestBootstrapReport:
    def test_defaults(self):
        r = BootstrapReport()
        assert r.intervals == []
        assert r.confidence_level == 0.95
        assert r.n_samples == 0

    def test_as_dict(self):
        ci = BootstrapCI(
            metric_id="m1",
            point_estimate=0.5,
            ci_lower=0.4,
            ci_upper=0.6,
            std_error=0.05,
            n_iterations=100,
        )
        r = BootstrapReport(intervals=[ci], confidence_level=0.90, n_samples=50)
        d = r.as_dict()
        assert len(d["intervals"]) == 1
        assert d["confidence_level"] == 0.90
        assert d["n_samples"] == 50

    def test_from_dict(self):
        d = {
            "intervals": [
                {
                    "metric_id": "x",
                    "point_estimate": 0.5,
                    "ci_lower": 0.4,
                    "ci_upper": 0.6,
                    "std_error": 0.05,
                    "n_iterations": 10,
                }
            ],
            "confidence_level": 0.99,
            "n_samples": 30,
        }
        r = BootstrapReport.from_dict(d)
        assert len(r.intervals) == 1
        assert r.intervals[0].metric_id == "x"
        assert r.confidence_level == 0.99

    def test_from_dict_defaults(self):
        r = BootstrapReport.from_dict({})
        assert r.intervals == []
        assert r.confidence_level == 0.95
        assert r.n_samples == 0

    def test_roundtrip(self):
        ci = BootstrapCI(
            metric_id="a",
            point_estimate=1.0,
            ci_lower=0.9,
            ci_upper=1.0,
            std_error=0.01,
            n_iterations=50,
        )
        r = BootstrapReport(intervals=[ci], confidence_level=0.90, n_samples=10)
        assert BootstrapReport.from_dict(r.as_dict()).intervals[0].metric_id == "a"


# ---- percentile ----


class TestPercentile:
    def test_single_value(self):
        assert percentile([5.0], 0.5) == 5.0

    def test_empty_list(self):
        assert percentile([], 0.5) == 0.0

    def test_median_odd(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(vals, 0.5) == 3.0

    def test_median_even(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        assert percentile(vals, 0.5) == 2.5

    def test_lower_bound(self):
        vals = [10.0, 20.0, 30.0]
        assert percentile(vals, 0.0) == 10.0

    def test_upper_bound(self):
        vals = [10.0, 20.0, 30.0]
        assert percentile(vals, 1.0) == 30.0

    def test_interpolation_quarter(self):
        vals = [0.0, 10.0, 20.0, 30.0, 40.0]
        result = percentile(vals, 0.25)
        assert result == 10.0

    def test_interpolation_three_quarter(self):
        vals = [0.0, 10.0, 20.0, 30.0, 40.0]
        result = percentile(vals, 0.75)
        assert result == 30.0

    def test_clamp_negative(self):
        vals = [1.0, 2.0, 3.0]
        assert percentile(vals, -0.5) == 1.0

    def test_clamp_above_one(self):
        vals = [1.0, 2.0, 3.0]
        assert percentile(vals, 1.5) == 3.0


# ---- resample ----


class TestResample:
    def test_length_preserved(self):
        rng = random.Random(42)
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = resample(scores, rng)
        assert len(result) == len(scores)

    def test_values_from_original(self):
        rng = random.Random(42)
        scores = [10.0, 20.0, 30.0]
        result = resample(scores, rng)
        for v in result:
            assert v in scores

    def test_empty_input(self):
        rng = random.Random(42)
        assert resample([], rng) == []

    def test_single_value(self):
        rng = random.Random(42)
        result = resample([7.0], rng)
        assert result == [7.0]

    def test_deterministic(self):
        scores = [1.0, 2.0, 3.0, 4.0]
        r1 = resample(scores, random.Random(99))
        r2 = resample(scores, random.Random(99))
        assert r1 == r2


# ---- compute_bootstrap_distribution ----


class TestComputeBootstrapDistribution:
    def test_length_matches_iterations(self):
        config = BootstrapConfig(n_iterations=50, seed=1)
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        dist = compute_bootstrap_distribution(
            scores, lambda x: sum(x) / len(x), config
        )
        assert len(dist) == 50

    def test_deterministic(self):
        config = BootstrapConfig(n_iterations=100, seed=42)
        scores = [1.0, 2.0, 3.0]
        d1 = compute_bootstrap_distribution(
            scores, lambda x: sum(x) / len(x), config
        )
        d2 = compute_bootstrap_distribution(
            scores, lambda x: sum(x) / len(x), config
        )
        assert d1 == d2

    def test_custom_statistic(self):
        config = BootstrapConfig(n_iterations=20, seed=1)
        scores = [10.0, 20.0, 30.0]
        dist = compute_bootstrap_distribution(scores, max, config)
        for v in dist:
            assert v in [10.0, 20.0, 30.0]

    def test_distribution_values_reasonable(self):
        config = BootstrapConfig(n_iterations=200, seed=42)
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        dist = compute_bootstrap_distribution(
            scores, lambda x: sum(x) / len(x), config
        )
        # Mean of [1..5] is 3.0; distribution should be near that
        avg = sum(dist) / len(dist)
        assert 2.0 < avg < 4.0


# ---- compute_confidence_interval ----


class TestComputeConfidenceInterval:
    def test_empty_distribution(self):
        assert compute_confidence_interval([]) == (0.0, 0.0)

    def test_single_value(self):
        lo, hi = compute_confidence_interval([5.0])
        assert lo == 5.0
        assert hi == 5.0

    def test_symmetric_distribution(self):
        dist = list(range(1, 101))
        dist_float = [float(x) for x in dist]
        lo, hi = compute_confidence_interval(dist_float, 0.90)
        assert lo < hi
        # 5th percentile around 5-6, 95th around 95-96
        assert lo < 10.0
        assert hi > 90.0

    def test_95_interval(self):
        dist = [float(i) for i in range(1000)]
        lo, hi = compute_confidence_interval(dist, 0.95)
        assert lo < 50.0
        assert hi > 950.0

    def test_narrow_with_tight_data(self):
        dist = [5.0] * 100
        lo, hi = compute_confidence_interval(dist, 0.95)
        assert lo == 5.0
        assert hi == 5.0


# ---- bootstrap_metric ----


class TestBootstrapMetric:
    def test_basic(self):
        config = BootstrapConfig(n_iterations=100, seed=42)
        scores = [0.8, 0.9, 0.7, 0.85, 0.95]
        ci = bootstrap_metric("accuracy", scores, config)
        assert ci.metric_id == "accuracy"
        assert ci.n_iterations == 100
        assert ci.ci_lower <= ci.point_estimate <= ci.ci_upper

    def test_point_estimate_is_mean(self):
        config = BootstrapConfig(n_iterations=10, seed=1)
        scores = [2.0, 4.0, 6.0]
        ci = bootstrap_metric("m", scores, config)
        assert ci.point_estimate == pytest.approx(4.0)

    def test_ci_lower_le_upper(self):
        config = BootstrapConfig(n_iterations=200, seed=42)
        scores = [0.1, 0.5, 0.9, 0.3, 0.7]
        ci = bootstrap_metric("x", scores, config)
        assert ci.ci_lower <= ci.ci_upper

    def test_std_error_positive(self):
        config = BootstrapConfig(n_iterations=100, seed=42)
        scores = [0.1, 0.9, 0.5, 0.3, 0.7]
        ci = bootstrap_metric("y", scores, config)
        assert ci.std_error > 0

    def test_constant_scores_zero_error(self):
        config = BootstrapConfig(n_iterations=50, seed=42)
        scores = [0.5, 0.5, 0.5, 0.5]
        ci = bootstrap_metric("const", scores, config)
        assert ci.std_error == 0.0
        assert ci.ci_lower == ci.ci_upper

    def test_empty_scores(self):
        config = BootstrapConfig(n_iterations=10, seed=1)
        ci = bootstrap_metric("empty", [], config)
        assert ci.point_estimate == 0.0


# ---- bootstrap_all_metrics ----


class TestBootstrapAllMetrics:
    def test_multiple_metrics(self):
        metrics = {
            "accuracy": [0.8, 0.9, 0.85],
            "f1": [0.7, 0.75, 0.72],
        }
        report = bootstrap_all_metrics(metrics)
        assert len(report.intervals) == 2
        assert report.confidence_level == 0.95
        ids = [ci.metric_id for ci in report.intervals]
        assert "accuracy" in ids
        assert "f1" in ids

    def test_sorted_metric_order(self):
        metrics = {"z_metric": [1.0], "a_metric": [2.0], "m_metric": [3.0]}
        report = bootstrap_all_metrics(metrics)
        ids = [ci.metric_id for ci in report.intervals]
        assert ids == ["a_metric", "m_metric", "z_metric"]

    def test_custom_config(self):
        config = BootstrapConfig(n_iterations=50, confidence_level=0.90, seed=7)
        metrics = {"m": [1.0, 2.0, 3.0]}
        report = bootstrap_all_metrics(metrics, config)
        assert report.confidence_level == 0.90
        assert report.intervals[0].n_iterations == 50

    def test_default_config(self):
        metrics = {"m": [1.0, 2.0]}
        report = bootstrap_all_metrics(metrics)
        assert report.intervals[0].n_iterations == 1000

    def test_n_samples_is_max_length(self):
        metrics = {"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0, 4.0]}
        report = bootstrap_all_metrics(metrics)
        assert report.n_samples == 4

    def test_empty_metrics_dict(self):
        report = bootstrap_all_metrics({})
        assert report.intervals == []
        assert report.n_samples == 0


# ---- format_bootstrap_report ----


class TestFormatBootstrapReport:
    def test_contains_header(self):
        report = BootstrapReport(confidence_level=0.95, n_samples=10)
        text = format_bootstrap_report(report)
        assert "Bootstrap Report" in text
        assert "95%" in text

    def test_contains_metric_names(self):
        ci = BootstrapCI(
            metric_id="accuracy",
            point_estimate=0.85,
            ci_lower=0.80,
            ci_upper=0.90,
            std_error=0.025,
            n_iterations=100,
        )
        report = BootstrapReport(intervals=[ci], confidence_level=0.95, n_samples=50)
        text = format_bootstrap_report(report)
        assert "accuracy" in text
        assert "0.8500" in text

    def test_multiple_metrics(self):
        intervals = [
            BootstrapCI("a", 0.5, 0.4, 0.6, 0.05, 100),
            BootstrapCI("b", 0.9, 0.85, 0.95, 0.02, 100),
        ]
        report = BootstrapReport(intervals=intervals, n_samples=20)
        text = format_bootstrap_report(report)
        assert "a" in text
        assert "b" in text

    def test_separator_lines(self):
        report = BootstrapReport(n_samples=5)
        text = format_bootstrap_report(report)
        assert "---" in text
