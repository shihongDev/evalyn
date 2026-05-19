"""Statistical analysis utilities for evaluation results.

Provides confidence intervals, power analysis, and significance testing
for metric pass rates and scores. No external dependencies beyond stdlib.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List


@dataclass
class ConfidenceInterval:
    """A confidence interval for a metric statistic."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float  # e.g. 0.95
    method: str  # "bootstrap" or "normal_approx"
    sample_size: int

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def margin_of_error(self) -> float:
        return self.width / 2


@dataclass
class SignificanceTest:
    """Result of a statistical significance test between two proportions."""

    metric_id: str
    baseline_rate: float
    current_rate: float
    delta: float
    z_statistic: float
    p_value: float
    significant: bool  # at the given alpha
    alpha: float
    effect_size: float  # Cohen's h


@dataclass
class PowerAnalysis:
    """Result of a statistical power analysis."""

    current_sample_size: int
    current_power: float  # power to detect the target effect
    target_effect: float  # the minimum detectable effect
    recommended_sample_size: int  # for 80% power
    alpha: float


def bootstrap_confidence_interval(
    values: list[float],
    statistic: str = "mean",
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> ConfidenceInterval:
    """Compute a bootstrap confidence interval for a statistic.

    Args:
        values: List of observed values (scores or binary pass/fail).
        statistic: "mean" or "proportion" (for pass rates, values should be 0/1).
        confidence_level: Confidence level (default 0.95 for 95% CI).
        n_resamples: Number of bootstrap resamples (default 1000).
        seed: Random seed for reproducibility.

    Returns:
        ConfidenceInterval with estimate, lower, upper bounds.
    """
    if not values:
        return ConfidenceInterval(
            estimate=0.0, lower=0.0, upper=0.0,
            confidence_level=confidence_level, method="bootstrap",
            sample_size=0,
        )

    n = len(values)
    rng = random.Random(seed)

    if statistic == "proportion":
        point_estimate = sum(values) / n
    else:
        point_estimate = sum(values) / n

    # Generate bootstrap distribution
    bootstrap_stats = []
    for _ in range(n_resamples):
        resample = [rng.choice(values) for _ in range(n)]
        bootstrap_stats.append(sum(resample) / n)

    bootstrap_stats.sort()

    alpha = 1 - confidence_level
    lower_idx = int(math.floor(alpha / 2 * n_resamples))
    upper_idx = int(math.ceil((1 - alpha / 2) * n_resamples)) - 1
    lower_idx = max(0, min(lower_idx, n_resamples - 1))
    upper_idx = max(0, min(upper_idx, n_resamples - 1))

    return ConfidenceInterval(
        estimate=point_estimate,
        lower=bootstrap_stats[lower_idx],
        upper=bootstrap_stats[upper_idx],
        confidence_level=confidence_level,
        method="bootstrap",
        sample_size=n,
    )


def normal_approx_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Compute a normal approximation confidence interval for a proportion.

    Uses the Wald interval: p +/- z * sqrt(p(1-p)/n).

    Args:
        successes: Number of successes (passes).
        total: Total number of observations.
        confidence_level: Confidence level.

    Returns:
        ConfidenceInterval for the pass rate.
    """
    if total == 0:
        return ConfidenceInterval(
            estimate=0.0, lower=0.0, upper=0.0,
            confidence_level=confidence_level, method="normal_approx",
            sample_size=0,
        )

    p = successes / total
    z = _z_score_for_confidence(confidence_level)
    se = math.sqrt(p * (1 - p) / total)
    margin = z * se

    return ConfidenceInterval(
        estimate=p,
        lower=max(0.0, p - margin),
        upper=min(1.0, p + margin),
        confidence_level=confidence_level,
        method="normal_approx",
        sample_size=total,
    )


def two_proportion_z_test(
    metric_id: str,
    baseline_successes: int,
    baseline_total: int,
    current_successes: int,
    current_total: int,
    alpha: float = 0.05,
) -> SignificanceTest:
    """Two-proportion z-test for comparing pass rates between runs.

    Tests whether the pass rate changed significantly between baseline
    and current run.

    Args:
        metric_id: Metric being compared.
        baseline_successes: Passes in baseline run.
        baseline_total: Total items in baseline run.
        current_successes: Passes in current run.
        current_total: Total items in current run.
        alpha: Significance level (default 0.05).

    Returns:
        SignificanceTest with z-statistic, p-value, and significance flag.
    """
    if baseline_total == 0 or current_total == 0:
        return SignificanceTest(
            metric_id=metric_id,
            baseline_rate=0.0, current_rate=0.0, delta=0.0,
            z_statistic=0.0, p_value=1.0, significant=False,
            alpha=alpha, effect_size=0.0,
        )

    p1 = baseline_successes / baseline_total
    p2 = current_successes / current_total
    delta = p2 - p1

    # Pooled proportion
    p_pool = (baseline_successes + current_successes) / (baseline_total + current_total)

    # Standard error of difference
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / baseline_total + 1 / current_total))

    if se == 0:
        z_stat = 0.0
        p_value = 1.0
    else:
        z_stat = delta / se
        # Two-tailed p-value using normal approximation
        p_value = 2 * (1 - _normal_cdf(abs(z_stat)))

    # Cohen's h (effect size for proportions)
    h = 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))

    return SignificanceTest(
        metric_id=metric_id,
        baseline_rate=p1,
        current_rate=p2,
        delta=delta,
        z_statistic=z_stat,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=abs(h),
    )


def compute_power(
    sample_size: int,
    target_effect: float = 0.05,
    alpha: float = 0.05,
    baseline_rate: float = 0.85,
) -> PowerAnalysis:
    """Compute statistical power and recommend sample size.

    Estimates the power to detect a given effect size (change in pass rate)
    with the current sample size.

    Args:
        sample_size: Current number of items.
        target_effect: Minimum detectable effect (absolute change in pass rate).
        alpha: Significance level.
        baseline_rate: Assumed baseline pass rate.

    Returns:
        PowerAnalysis with current power and recommended sample size.
    """
    if sample_size == 0 or target_effect == 0:
        return PowerAnalysis(
            current_sample_size=sample_size,
            current_power=0.0,
            target_effect=target_effect,
            recommended_sample_size=0,
            alpha=alpha,
        )

    # Standard error for the difference
    p1 = baseline_rate
    p2 = baseline_rate + target_effect
    se = math.sqrt(p1 * (1 - p1) / sample_size + p2 * (1 - p2) / sample_size)

    if se == 0:
        return PowerAnalysis(
            current_sample_size=sample_size,
            current_power=1.0,
            target_effect=target_effect,
            recommended_sample_size=sample_size,
            alpha=alpha,
        )

    z_alpha = _z_score_for_confidence(1 - alpha)
    z_stat = abs(target_effect) / se
    power = _normal_cdf(z_stat - z_alpha)

    # Recommended sample size for 80% power
    z_beta = 0.842  # z for 80% power
    p_avg = (p1 + p2) / 2
    recommended_n = math.ceil(
        2 * p_avg * (1 - p_avg) * ((z_alpha + z_beta) / target_effect) ** 2
    )

    return PowerAnalysis(
        current_sample_size=sample_size,
        current_power=min(1.0, power),
        target_effect=target_effect,
        recommended_sample_size=max(recommended_n, 1),
        alpha=alpha,
    )


# =============================================================================
# Internal helpers (stdlib only, no scipy)
# =============================================================================


def _z_score_for_confidence(confidence_level: float) -> float:
    """Approximate z-score for a given confidence level."""
    # Common values
    z_table = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }
    if confidence_level in z_table:
        return z_table[confidence_level]
    # Approximation using Abramowitz and Stegun formula
    p = (1 - confidence_level) / 2
    return _inv_normal_cdf(1 - p)


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using Abramowitz and Stegun."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _inv_normal_cdf(p: float) -> float:
    """Approximate inverse normal CDF (quantile function).

    Uses the rational approximation from Abramowitz and Stegun (26.2.23).
    """
    if p <= 0:
        return float("-inf")
    if p >= 1:
        return float("inf")
    if p == 0.5:
        return 0.0

    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))

    # Rational approximation coefficients
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)

    if p < 0.5:
        return -z
    return z
