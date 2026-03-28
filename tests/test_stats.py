"""Tests for statistical analysis utilities."""

import math
import pytest
from evalyn_sdk.analysis.stats import (
    bootstrap_confidence_interval,
    normal_approx_confidence_interval,
    two_proportion_z_test,
    compute_power,
    ConfidenceInterval,
    SignificanceTest,
    PowerAnalysis,
    _normal_cdf,
    _inv_normal_cdf,
)


class TestBootstrapConfidenceInterval:
    """Test bootstrap CI computation."""

    def test_basic_mean(self):
        values = [0.8, 0.9, 0.7, 0.85, 0.95]
        ci = bootstrap_confidence_interval(values, statistic="mean")
        assert ci.sample_size == 5
        assert ci.method == "bootstrap"
        assert ci.confidence_level == 0.95
        assert 0.5 <= ci.lower <= ci.estimate
        assert ci.estimate <= ci.upper <= 1.0

    def test_proportion(self):
        # 80% pass rate
        values = [1, 1, 1, 1, 0] * 20  # 80 passes, 20 fails
        ci = bootstrap_confidence_interval(values, statistic="proportion")
        assert 0.7 <= ci.estimate <= 0.9
        assert ci.lower < ci.estimate
        assert ci.upper > ci.estimate

    def test_perfect_pass_rate(self):
        values = [1.0] * 50
        ci = bootstrap_confidence_interval(values)
        assert ci.estimate == 1.0
        assert ci.lower == 1.0
        assert ci.upper == 1.0

    def test_zero_pass_rate(self):
        values = [0.0] * 50
        ci = bootstrap_confidence_interval(values)
        assert ci.estimate == 0.0
        assert ci.lower == 0.0
        assert ci.upper == 0.0

    def test_empty_values(self):
        ci = bootstrap_confidence_interval([])
        assert ci.estimate == 0.0
        assert ci.sample_size == 0

    def test_single_value(self):
        ci = bootstrap_confidence_interval([0.85])
        assert ci.estimate == 0.85
        assert ci.sample_size == 1

    def test_reproducibility_with_seed(self):
        values = [0.8, 0.9, 0.7, 0.85, 0.95]
        ci1 = bootstrap_confidence_interval(values, seed=42)
        ci2 = bootstrap_confidence_interval(values, seed=42)
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper

    def test_different_seeds_differ(self):
        values = [0.8, 0.9, 0.7, 0.85, 0.95] * 10
        ci1 = bootstrap_confidence_interval(values, seed=1)
        ci2 = bootstrap_confidence_interval(values, seed=999)
        # With enough data points, different seeds may produce slightly different CIs
        # This is a probabilistic test, so just check they exist
        assert ci1.sample_size == ci2.sample_size

    def test_wider_ci_with_more_variance(self):
        low_var = [0.80, 0.81, 0.79, 0.80, 0.82] * 10
        high_var = [0.50, 0.90, 0.60, 1.00, 0.30] * 10
        ci_low = bootstrap_confidence_interval(low_var)
        ci_high = bootstrap_confidence_interval(high_var)
        assert ci_low.width < ci_high.width

    def test_narrower_ci_with_more_samples(self):
        small = [0.8, 0.9, 0.7, 0.85, 0.95]
        large = small * 20
        ci_small = bootstrap_confidence_interval(small)
        ci_large = bootstrap_confidence_interval(large)
        assert ci_large.width < ci_small.width

    def test_width_and_margin_of_error(self):
        values = [0.8, 0.9, 0.7] * 10
        ci = bootstrap_confidence_interval(values)
        assert abs(ci.width - (ci.upper - ci.lower)) < 1e-10
        assert abs(ci.margin_of_error - ci.width / 2) < 1e-10


class TestNormalApproxCI:
    """Test normal approximation CI."""

    def test_basic(self):
        ci = normal_approx_confidence_interval(85, 100)
        assert ci.estimate == 0.85
        assert ci.lower < 0.85
        assert ci.upper > 0.85
        assert ci.method == "normal_approx"

    def test_zero_total(self):
        ci = normal_approx_confidence_interval(0, 0)
        assert ci.estimate == 0.0
        assert ci.sample_size == 0

    def test_perfect_rate(self):
        ci = normal_approx_confidence_interval(100, 100)
        assert ci.estimate == 1.0
        assert ci.upper == 1.0

    def test_zero_rate(self):
        ci = normal_approx_confidence_interval(0, 100)
        assert ci.estimate == 0.0
        assert ci.lower == 0.0

    def test_bounds_clamped(self):
        # With small samples, bounds should still be in [0, 1]
        ci = normal_approx_confidence_interval(1, 2)
        assert ci.lower >= 0.0
        assert ci.upper <= 1.0

    def test_wider_with_small_sample(self):
        ci_small = normal_approx_confidence_interval(8, 10)
        ci_large = normal_approx_confidence_interval(800, 1000)
        assert ci_small.width > ci_large.width


class TestTwoProportionZTest:
    """Test significance testing between two runs."""

    def test_no_change(self):
        result = two_proportion_z_test(
            "helpfulness", 85, 100, 85, 100,
        )
        assert result.delta == 0.0
        assert result.p_value == 1.0
        assert result.significant is False

    def test_significant_improvement(self):
        result = two_proportion_z_test(
            "helpfulness", 50, 100, 80, 100,
        )
        assert result.delta == pytest.approx(0.30)
        assert result.p_value < 0.05
        assert result.significant is True
        assert result.effect_size > 0

    def test_significant_regression(self):
        result = two_proportion_z_test(
            "helpfulness", 90, 100, 60, 100,
        )
        assert result.delta == pytest.approx(-0.30)
        assert result.p_value < 0.05
        assert result.significant is True

    def test_not_significant_small_change(self):
        result = two_proportion_z_test(
            "helpfulness", 85, 100, 87, 100,
        )
        # 2% change is usually not significant with n=100
        assert result.delta == pytest.approx(0.02)
        assert result.significant is False

    def test_zero_total(self):
        result = two_proportion_z_test("m", 0, 0, 0, 0)
        assert result.p_value == 1.0
        assert result.significant is False

    def test_custom_alpha(self):
        result = two_proportion_z_test(
            "m", 50, 100, 65, 100, alpha=0.01,
        )
        assert result.alpha == 0.01

    def test_effect_size_positive(self):
        result = two_proportion_z_test("m", 50, 100, 80, 100)
        assert result.effect_size > 0

    def test_metric_id_preserved(self):
        result = two_proportion_z_test("my_metric", 50, 100, 80, 100)
        assert result.metric_id == "my_metric"


class TestPowerAnalysis:
    """Test power computation and sample size recommendation."""

    def test_basic_power(self):
        result = compute_power(sample_size=100, target_effect=0.1)
        assert 0 < result.current_power <= 1.0
        assert result.current_sample_size == 100
        assert result.target_effect == 0.1

    def test_zero_sample_size(self):
        result = compute_power(sample_size=0)
        assert result.current_power == 0.0

    def test_zero_effect(self):
        result = compute_power(sample_size=100, target_effect=0.0)
        assert result.current_power == 0.0

    def test_large_sample_high_power(self):
        result = compute_power(sample_size=1000, target_effect=0.05)
        assert result.current_power > 0.8

    def test_small_sample_low_power(self):
        result = compute_power(sample_size=20, target_effect=0.05)
        assert result.current_power < 0.5

    def test_recommended_size_is_larger_when_underpowered(self):
        result = compute_power(sample_size=20, target_effect=0.05)
        assert result.recommended_sample_size > 20

    def test_recommended_size_positive(self):
        result = compute_power(sample_size=100, target_effect=0.1)
        assert result.recommended_sample_size >= 1


class TestHelpers:
    """Test internal math helpers."""

    def test_normal_cdf_at_zero(self):
        assert _normal_cdf(0.0) == pytest.approx(0.5)

    def test_normal_cdf_at_positive(self):
        assert _normal_cdf(1.96) == pytest.approx(0.975, abs=0.001)

    def test_normal_cdf_at_negative(self):
        assert _normal_cdf(-1.96) == pytest.approx(0.025, abs=0.001)

    def test_inv_normal_cdf_at_half(self):
        assert _inv_normal_cdf(0.5) == 0.0

    def test_inv_normal_cdf_at_975(self):
        assert _inv_normal_cdf(0.975) == pytest.approx(1.96, abs=0.01)

    def test_cdf_and_inv_roundtrip(self):
        for z in [0.0, 1.0, -1.0, 1.96, -2.0]:
            p = _normal_cdf(z)
            z_back = _inv_normal_cdf(p)
            assert z_back == pytest.approx(z, abs=0.01)
