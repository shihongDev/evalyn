"""Tests for normality testing module."""

import math
import random

import pytest
from evalyn_sdk.analysis.normality_testing import (
    NormalityReport,
    NormalityResult,
    check_all_normality,
    check_normality,
    compute_kurtosis,
    compute_skewness,
    jarque_bera_test,
    suggest_transformations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normal_samples(n: int, mean: float = 0.0, std: float = 1.0, seed: int = 42) -> list:
    """Generate approximately normal samples using Box-Muller."""
    rng = random.Random(seed)
    samples = []
    for _ in range(n // 2 + 1):
        u1 = rng.random()
        u2 = rng.random()
        while u1 == 0.0:
            u1 = rng.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        samples.append(mean + std * z0)
        samples.append(mean + std * z1)
    return samples[:n]


def _right_skewed_samples(n: int, seed: int = 42) -> list:
    """Generate right-skewed (exponential-like) samples."""
    rng = random.Random(seed)
    return [-math.log(1 - rng.random()) for _ in range(n)]


# ---------------------------------------------------------------------------
# compute_skewness
# ---------------------------------------------------------------------------


class TestComputeSkewness:
    def test_symmetric_near_zero(self):
        data = _normal_samples(500, seed=10)
        s = compute_skewness(data)
        assert abs(s) < 0.3

    def test_right_skewed_positive(self):
        data = _right_skewed_samples(500)
        s = compute_skewness(data)
        assert s > 0.5

    def test_left_skewed_negative(self):
        data = [-x for x in _right_skewed_samples(500)]
        s = compute_skewness(data)
        assert s < -0.5

    def test_two_values_returns_zero(self):
        assert compute_skewness([1.0, 2.0]) == 0.0

    def test_empty_returns_zero(self):
        assert compute_skewness([]) == 0.0

    def test_constant_returns_zero(self):
        assert compute_skewness([5.0] * 50) == 0.0

    def test_three_values(self):
        # Minimal case that computes
        result = compute_skewness([1.0, 2.0, 3.0])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_kurtosis
# ---------------------------------------------------------------------------


class TestComputeKurtosis:
    def test_normal_near_zero(self):
        data = _normal_samples(1000, seed=7)
        k = compute_kurtosis(data)
        assert abs(k) < 1.0

    def test_uniform_negative(self):
        rng = random.Random(99)
        data = [rng.random() for _ in range(1000)]
        k = compute_kurtosis(data)
        # Uniform has negative excess kurtosis (~-1.2)
        assert k < 0

    def test_three_values_returns_zero(self):
        assert compute_kurtosis([1.0, 2.0, 3.0]) == 0.0

    def test_empty_returns_zero(self):
        assert compute_kurtosis([]) == 0.0

    def test_constant_returns_zero(self):
        assert compute_kurtosis([3.0] * 50) == 0.0

    def test_four_values(self):
        result = compute_kurtosis([1.0, 2.0, 3.0, 4.0])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# jarque_bera_test
# ---------------------------------------------------------------------------


class TestJarqueBeraTest:
    def test_normal_data_high_p(self):
        data = _normal_samples(500, seed=20)
        jb, p = jarque_bera_test(data)
        # Normal data should have high p-value (fail to reject normality)
        assert p > 0.05

    def test_non_normal_low_p(self):
        data = _right_skewed_samples(500)
        jb, p = jarque_bera_test(data)
        assert jb > 5.0
        assert p < 0.05

    def test_few_values_returns_defaults(self):
        jb, p = jarque_bera_test([1.0, 2.0])
        assert jb == 0.0
        assert p == 1.0

    def test_empty_returns_defaults(self):
        jb, p = jarque_bera_test([])
        assert jb == 0.0
        assert p == 1.0

    def test_returns_tuple(self):
        result = jarque_bera_test(_normal_samples(50))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_p_value_between_0_and_1(self):
        _, p = jarque_bera_test(_normal_samples(200))
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# check_normality
# ---------------------------------------------------------------------------


class TestCheckNormality:
    def test_normal_data_is_normal(self):
        data = _normal_samples(500, seed=30)
        r = check_normality("accuracy", data)
        assert r.is_normal is True
        assert r.metric_id == "accuracy"
        assert r.sample_size == 500

    def test_skewed_data_not_normal(self):
        data = _right_skewed_samples(500)
        r = check_normality("latency", data)
        assert r.is_normal is False
        assert r.skewness > 0

    def test_few_values_defaults_normal(self):
        r = check_normality("m", [1.0, 2.0])
        assert r.is_normal is True
        assert r.sample_size == 2

    def test_custom_alpha(self):
        data = _normal_samples(100, seed=55)
        r = check_normality("m", data, alpha=0.001)
        # Even with very strict alpha, truly normal data should pass
        assert r.is_normal is True


# ---------------------------------------------------------------------------
# check_all_normality
# ---------------------------------------------------------------------------


class TestCheckAllNormality:
    def test_multiple_metrics(self):
        metric_scores = {
            "accuracy": _normal_samples(200, seed=1),
            "latency": _right_skewed_samples(200),
        }
        report = check_all_normality(metric_scores)
        assert report.total_metrics == 2
        assert report.normal_count + report.non_normal_count == 2
        # accuracy should be normal
        acc = [r for r in report.results if r.metric_id == "accuracy"][0]
        assert acc.is_normal is True
        # latency should be non-normal
        lat = [r for r in report.results if r.metric_id == "latency"][0]
        assert lat.is_normal is False

    def test_empty_dict(self):
        report = check_all_normality({})
        assert report.total_metrics == 0
        assert report.normal_count == 0

    def test_sorted_by_metric_id(self):
        scores = {"z_metric": [1.0, 2.0, 3.0], "a_metric": [4.0, 5.0, 6.0]}
        report = check_all_normality(scores)
        assert report.results[0].metric_id == "a_metric"
        assert report.results[1].metric_id == "z_metric"


# ---------------------------------------------------------------------------
# suggest_transformations
# ---------------------------------------------------------------------------


class TestSuggestTransformations:
    def test_normal_no_suggestions(self):
        r = NormalityResult(metric_id="m", is_normal=True)
        assert suggest_transformations(r) == []

    def test_right_skewed(self):
        r = NormalityResult(metric_id="m", is_normal=False, skewness=1.5)
        suggestions = suggest_transformations(r)
        assert any("log" in s for s in suggestions)

    def test_moderate_right_skew(self):
        r = NormalityResult(metric_id="m", is_normal=False, skewness=0.7)
        suggestions = suggest_transformations(r)
        assert any("square root" in s for s in suggestions)

    def test_left_skewed(self):
        r = NormalityResult(metric_id="m", is_normal=False, skewness=-1.5)
        suggestions = suggest_transformations(r)
        assert any("reflect" in s and "log" in s for s in suggestions)

    def test_moderate_left_skew(self):
        r = NormalityResult(metric_id="m", is_normal=False, skewness=-0.7)
        suggestions = suggest_transformations(r)
        assert any("reflect" in s and "square root" in s for s in suggestions)

    def test_high_kurtosis_low_skew(self):
        r = NormalityResult(metric_id="m", is_normal=False, skewness=0.1, kurtosis=3.0)
        suggestions = suggest_transformations(r)
        assert any("Box-Cox" in s for s in suggestions)

    def test_fallback_nonparametric(self):
        r = NormalityResult(metric_id="m", is_normal=False, skewness=0.3, kurtosis=0.5)
        suggestions = suggest_transformations(r)
        assert any("non-parametric" in s for s in suggestions)


# ---------------------------------------------------------------------------
# NormalityReport.format_text
# ---------------------------------------------------------------------------


class TestNormalityReportFormatText:
    def test_contains_header(self):
        report = NormalityReport(total_metrics=0)
        text = report.format_text()
        assert "Normality Testing Report" in text

    def test_shows_counts(self):
        report = NormalityReport(
            results=[
                NormalityResult(metric_id="a", is_normal=True, sample_size=100),
            ],
            normal_count=1,
            non_normal_count=0,
            total_metrics=1,
        )
        text = report.format_text()
        assert "Normal: 1" in text
        assert "Non-normal: 0" in text

    def test_shows_non_normal_suggestions(self):
        report = NormalityReport(
            results=[
                NormalityResult(
                    metric_id="lat",
                    is_normal=False,
                    skewness=2.0,
                    sample_size=50,
                ),
            ],
            normal_count=0,
            non_normal_count=1,
            total_metrics=1,
        )
        text = report.format_text()
        assert "NON-NORMAL" in text
        assert "log" in text


# ---------------------------------------------------------------------------
# as_dict / from_dict roundtrips
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_normality_result_roundtrip(self):
        r = NormalityResult(
            metric_id="acc",
            is_normal=False,
            skewness=0.5,
            kurtosis=1.2,
            jarque_bera_stat=8.3,
            p_value_approx=0.016,
            sample_size=100,
        )
        d = r.as_dict()
        r2 = NormalityResult.from_dict(d)
        assert r2.metric_id == r.metric_id
        assert r2.is_normal == r.is_normal
        assert r2.skewness == r.skewness
        assert r2.kurtosis == r.kurtosis
        assert r2.jarque_bera_stat == r.jarque_bera_stat
        assert r2.p_value_approx == r.p_value_approx
        assert r2.sample_size == r.sample_size

    def test_normality_report_roundtrip(self):
        report = NormalityReport(
            results=[
                NormalityResult(metric_id="a", is_normal=True, sample_size=50),
                NormalityResult(metric_id="b", is_normal=False, sample_size=50),
            ],
            normal_count=1,
            non_normal_count=1,
            total_metrics=2,
        )
        d = report.as_dict()
        report2 = NormalityReport.from_dict(d)
        assert report2.total_metrics == 2
        assert report2.normal_count == 1
        assert len(report2.results) == 2
        assert report2.results[0].metric_id == "a"

    def test_normality_result_as_dict_keys(self):
        r = NormalityResult(metric_id="x", is_normal=True)
        d = r.as_dict()
        expected_keys = {
            "metric_id", "is_normal", "skewness", "kurtosis",
            "jarque_bera_stat", "p_value_approx", "sample_size",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_value(self):
        r = check_normality("m", [5.0])
        assert r.is_normal is True
        assert r.sample_size == 1

    def test_constant_values(self):
        r = check_normality("m", [3.0] * 100)
        assert r.is_normal is True
        assert r.skewness == 0.0
        assert r.kurtosis == 0.0

    def test_two_distinct_values(self):
        r = check_normality("m", [0.0, 1.0, 0.0, 1.0, 0.0])
        assert isinstance(r.is_normal, bool)
        assert r.sample_size == 5

    def test_large_sample_normal(self):
        data = _normal_samples(2000, seed=77)
        r = check_normality("m", data)
        assert r.is_normal is True

    def test_empty_list(self):
        r = check_normality("m", [])
        assert r.is_normal is True
        assert r.sample_size == 0
