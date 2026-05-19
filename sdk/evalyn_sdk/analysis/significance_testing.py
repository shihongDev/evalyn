"""
Statistical significance testing for run-to-run comparisons.

Provides Welch's t-test, confidence intervals, effect sizes,
and Bonferroni correction for multiple comparisons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class SignificanceResult:
    """Result of a significance test for a single metric."""

    metric_id: str
    mean_a: float
    mean_b: float
    delta: float
    p_value: float
    significant: bool
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    method: str = "welch_t"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "delta": self.delta,
            "p_value": self.p_value,
            "significant": self.significant,
            "confidence_interval": list(self.confidence_interval),
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SignificanceResult:
        ci = data.get("confidence_interval", [0.0, 0.0])
        return cls(
            metric_id=data["metric_id"],
            mean_a=data["mean_a"],
            mean_b=data["mean_b"],
            delta=data["delta"],
            p_value=data["p_value"],
            significant=data["significant"],
            confidence_interval=(ci[0], ci[1]),
            method=data.get("method", "welch_t"),
        )


@dataclass
class SignificanceReport:
    """Report summarizing significance tests across metrics."""

    results: List[SignificanceResult] = field(default_factory=list)
    alpha: float = 0.05
    significant_count: int = 0
    total_tests: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "alpha": self.alpha,
            "significant_count": self.significant_count,
            "total_tests": self.total_tests,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SignificanceReport:
        results = [SignificanceResult.from_dict(r) for r in data.get("results", [])]
        return cls(
            results=results,
            alpha=data.get("alpha", 0.05),
            significant_count=data.get("significant_count", 0),
            total_tests=data.get("total_tests", 0),
        )

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append(f"Significance Report (alpha={self.alpha})")
        lines.append(f"  {self.significant_count}/{self.total_tests} metrics significant")
        lines.append("")
        for r in self.results:
            marker = "*" if r.significant else " "
            lines.append(
                f"  [{marker}] {r.metric_id}: delta={r.delta:+.4f}, "
                f"p={r.p_value:.4f}, CI=({r.confidence_interval[0]:.4f}, {r.confidence_interval[1]:.4f})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical Functions
# ---------------------------------------------------------------------------


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _variance(values: List[float]) -> float:
    """Sample variance (Bessel's correction)."""
    n = len(values)
    if n < 2:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (n - 1)


def welch_t_test(scores_a: List[float], scores_b: List[float]) -> Tuple[float, float]:
    """Welch's t-test for unequal variances.

    Returns (t_statistic, p_value). Computed manually without scipy.
    Uses two-tailed test with normal approximation for p-value.
    """
    n_a = len(scores_a)
    n_b = len(scores_b)
    if n_a < 2 or n_b < 2:
        return (0.0, 1.0)

    mean_a = _mean(scores_a)
    mean_b = _mean(scores_b)
    var_a = _variance(scores_a)
    var_b = _variance(scores_b)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0.0:
        return (0.0, 1.0)

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom_a = (var_a / n_a) ** 2 / (n_a - 1) if var_a > 0 else 0.0
    denom_b = (var_b / n_b) ** 2 / (n_b - 1) if var_b > 0 else 0.0
    denom = denom_a + denom_b
    if denom == 0.0:
        return (0.0, 1.0)
    df = num / denom

    # Approximate p-value using normal distribution for large df
    # For smaller df, use a rough t-distribution approximation
    p_value = _approx_two_tailed_p(t_stat, df)
    return (t_stat, p_value)


def _approx_two_tailed_p(t: float, df: float) -> float:
    """Approximate two-tailed p-value from t-statistic and degrees of freedom.

    Uses normal approximation for df >= 30, otherwise uses a conservative
    adjustment factor.
    """
    abs_t = abs(t)

    if df >= 30:
        # Normal approximation
        return 2.0 * _normal_sf(abs_t)

    # For smaller df, apply a correction factor
    # This makes the p-value slightly larger (more conservative)
    correction = 1.0 + (1.0 / (4.0 * max(df, 1.0)))
    adjusted_t = abs_t / correction
    return min(1.0, 2.0 * _normal_sf(adjusted_t))


def _normal_sf(x: float) -> float:
    """Survival function (1 - CDF) for standard normal distribution.

    Uses the complementary error function approximation.
    """
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def compute_confidence_interval(
    scores: List[float], confidence: float = 0.95
) -> Tuple[float, float]:
    """Confidence interval using z-score approximation.

    Uses z=1.96 for 95% confidence, scaled proportionally for other levels.
    """
    n = len(scores)
    if n < 2:
        if n == 1:
            return (scores[0], scores[0])
        return (0.0, 0.0)

    m = _mean(scores)
    se = math.sqrt(_variance(scores) / n)

    # z-score lookup for common confidence levels
    if confidence == 0.95:
        z = 1.96
    elif confidence == 0.99:
        z = 2.576
    elif confidence == 0.90:
        z = 1.645
    else:
        # Generic approximation: use inverse normal via probit approximation
        alpha = 1.0 - confidence
        z = _approx_z(1.0 - alpha / 2.0)

    margin = z * se
    return (m - margin, m + margin)


def _approx_z(p: float) -> float:
    """Approximate inverse normal CDF (probit function).

    Rational approximation for 0.5 < p < 1.
    """
    if p <= 0.5:
        return -_approx_z(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    # Abramowitz and Stegun approximation 26.2.23
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def run_significance_test(
    metric_id: str,
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> SignificanceResult:
    """Full significance test for a single metric between two runs."""
    if not scores_a or not scores_b:
        return SignificanceResult(
            metric_id=metric_id,
            mean_a=_mean(scores_a) if scores_a else 0.0,
            mean_b=_mean(scores_b) if scores_b else 0.0,
            delta=0.0,
            p_value=1.0,
            significant=False,
        )

    mean_a = _mean(scores_a)
    mean_b = _mean(scores_b)
    delta = mean_b - mean_a

    _, p_value = welch_t_test(scores_a, scores_b)
    significant = p_value < alpha

    # CI on the difference of means
    if len(scores_a) >= 2 and len(scores_b) >= 2:
        diff_se = math.sqrt(_variance(scores_a) / len(scores_a) + _variance(scores_b) / len(scores_b))
        ci = (delta - 1.96 * diff_se, delta + 1.96 * diff_se)
    else:
        ci = (delta, delta)

    return SignificanceResult(
        metric_id=metric_id,
        mean_a=mean_a,
        mean_b=mean_b,
        delta=delta,
        p_value=p_value,
        significant=significant,
        confidence_interval=ci,
    )


def run_all_significance_tests(
    metrics: Dict[str, Tuple[List[float], List[float]]],
    alpha: float = 0.05,
) -> SignificanceReport:
    """Test significance for all metrics.

    Args:
        metrics: {metric_id: (scores_a, scores_b)}
        alpha: significance threshold
    """
    results: List[SignificanceResult] = []
    for metric_id, (scores_a, scores_b) in sorted(metrics.items()):
        result = run_significance_test(metric_id, scores_a, scores_b, alpha)
        results.append(result)

    sig_count = sum(1 for r in results if r.significant)
    return SignificanceReport(
        results=results,
        alpha=alpha,
        significant_count=sig_count,
        total_tests=len(results),
    )


def apply_bonferroni_correction(
    results: List[SignificanceResult], alpha: float = 0.05
) -> List[SignificanceResult]:
    """Adjust significance for multiple comparisons using Bonferroni correction.

    New threshold = alpha / num_tests. Re-evaluates significance for each result.
    """
    n = len(results)
    if n == 0:
        return []

    corrected_alpha = alpha / n
    corrected: List[SignificanceResult] = []
    for r in results:
        corrected.append(
            SignificanceResult(
                metric_id=r.metric_id,
                mean_a=r.mean_a,
                mean_b=r.mean_b,
                delta=r.delta,
                p_value=r.p_value,
                significant=r.p_value < corrected_alpha,
                confidence_interval=r.confidence_interval,
                method=r.method,
            )
        )
    return corrected


def compute_effect_size(scores_a: List[float], scores_b: List[float]) -> float:
    """Cohen's d effect size.

    Uses pooled standard deviation. Returns 0.0 for degenerate cases.
    """
    n_a = len(scores_a)
    n_b = len(scores_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a = _mean(scores_a)
    mean_b = _mean(scores_b)
    var_a = _variance(scores_a)
    var_b = _variance(scores_b)

    # Pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_sd = math.sqrt(pooled_var)

    if pooled_sd == 0.0:
        return 0.0

    return (mean_b - mean_a) / pooled_sd
