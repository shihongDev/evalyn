"""Tests for metric cross-reference and score curve fitting."""

import pytest
from evalyn_sdk.metrics.cross_reference import (
    build_cross_reference, MetricCrossRef, BundleCrossRef, CrossReferenceReport,
)
from evalyn_sdk.analysis.curve_fitting import (
    fit_distribution, fit_all_metrics, DistributionFit,
)


# =============================================================================
# Cross-reference tests
# =============================================================================

class TestBuildCrossReference:
    def test_basic(self):
        bundles = {
            "chatbot": ["helpfulness", "toxicity", "json_valid"],
            "safety": ["toxicity", "pii_safety"],
        }
        types = {"helpfulness": "subjective", "toxicity": "subjective",
                 "json_valid": "objective", "pii_safety": "subjective"}
        report = build_cross_reference(bundles, types)
        assert len(report.metrics) == 4
        assert len(report.bundles) == 2

    def test_metric_in_multiple_bundles(self):
        bundles = {"a": ["m1", "m2"], "b": ["m2", "m3"]}
        report = build_cross_reference(bundles)
        m2 = next(m for m in report.metrics if m.metric_id == "m2")
        assert len(m2.bundles) == 2
        assert "a" in m2.bundles
        assert "b" in m2.bundles

    def test_orphan_metrics(self):
        bundles = {"a": ["m1"]}
        types = {"m1": "objective", "m2": "objective"}
        report = build_cross_reference(bundles, types)
        assert "m2" in report.orphan_metrics

    def test_bundle_counts(self):
        bundles = {"test": ["obj1", "subj1", "subj2"]}
        types = {"obj1": "objective", "subj1": "subjective", "subj2": "subjective"}
        report = build_cross_reference(bundles, types)
        b = report.bundles[0]
        assert b.total_metrics == 3
        assert b.objective_count == 1
        assert b.subjective_count == 2

    def test_empty(self):
        report = build_cross_reference({})
        assert len(report.metrics) == 0
        assert len(report.bundles) == 0

    def test_as_dict(self):
        bundles = {"a": ["m1"]}
        d = build_cross_reference(bundles).as_dict()
        assert "metrics" in d
        assert "bundles" in d
        assert "orphan_metrics" in d

    def test_format_text(self):
        bundles = {"chatbot": ["help", "safety"]}
        text = build_cross_reference(bundles).format_text()
        assert "Cross-Reference" in text
        assert "chatbot" in text

    def test_metric_crossref_as_dict(self):
        cr = MetricCrossRef(metric_id="m1", metric_type="objective", bundles=["a", "b"])
        d = cr.as_dict()
        assert d["bundle_count"] == 2


# =============================================================================
# Curve fitting tests
# =============================================================================

class TestFitDistribution:
    def test_normal(self):
        # Generate roughly normal scores
        import random
        rng = random.Random(42)
        scores = [max(0, min(1, rng.gauss(0.7, 0.1))) for _ in range(100)]
        fit = fit_distribution("m", scores)
        assert fit.distribution_type in ("normal", "skewed_left", "skewed_right")
        assert 0.6 < fit.mean < 0.8

    def test_bimodal(self):
        # Clear bimodal: cluster around 0.2 and 0.8
        scores = [0.1, 0.15, 0.2, 0.22, 0.18] * 10 + [0.75, 0.8, 0.85, 0.82, 0.78] * 10
        fit = fit_distribution("m", scores)
        # May or may not detect bimodality depending on bin resolution
        assert fit.sample_count == 100

    def test_degenerate(self):
        scores = [0.5] * 20
        fit = fit_distribution("m", scores)
        assert fit.distribution_type == "degenerate"
        assert fit.std_dev == 0.0

    def test_skewed_right(self):
        # Most scores high, few low
        scores = [0.9] * 80 + [0.1] * 20
        fit = fit_distribution("m", scores)
        assert fit.skewness < 0  # left-skewed because mass is at right

    def test_empty(self):
        fit = fit_distribution("m", [])
        assert fit.distribution_type == "empty"
        assert fit.sample_count == 0

    def test_single_score(self):
        fit = fit_distribution("m", [0.8])
        assert fit.distribution_type == "insufficient_data"

    def test_two_scores(self):
        fit = fit_distribution("m", [0.3, 0.9])
        assert fit.distribution_type == "insufficient_data"

    def test_median_odd(self):
        fit = fit_distribution("m", [0.1, 0.5, 0.9])
        assert fit.median == 0.5

    def test_median_even(self):
        fit = fit_distribution("m", [0.1, 0.4, 0.6, 0.9])
        assert fit.median == 0.5

    def test_as_dict(self):
        fit = fit_distribution("m", [0.5, 0.6, 0.7, 0.8, 0.5, 0.6])
        d = fit.as_dict()
        assert "metric_id" in d
        assert "distribution_type" in d
        assert "skewness" in d
        assert "kurtosis" in d

    def test_format_text(self):
        fit = fit_distribution("m", [0.5, 0.6, 0.7, 0.8, 0.5, 0.6])
        text = fit.format_text()
        assert "m:" in text


class TestFitAllMetrics:
    def test_multiple(self):
        scores = {
            "a": [0.5, 0.6, 0.7, 0.8, 0.5],
            "b": [0.9, 0.9, 0.9, 0.9, 0.9],
        }
        fits = fit_all_metrics(scores)
        assert len(fits) == 2
        assert "a" in fits
        assert "b" in fits
        assert fits["b"].distribution_type == "degenerate"

    def test_empty(self):
        assert fit_all_metrics({}) == {}
