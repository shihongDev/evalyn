"""Score distribution normality testing.

Verify if metric scores follow expected distributions using the
Jarque-Bera test. Provides skewness, kurtosis, and transformation
suggestions for non-normal distributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class NormalityResult:
    """Normality test result for a single metric."""

    metric_id: str
    is_normal: bool
    skewness: float = 0.0
    kurtosis: float = 0.0
    jarque_bera_stat: float = 0.0
    p_value_approx: float = 0.0
    sample_size: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "is_normal": self.is_normal,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "jarque_bera_stat": self.jarque_bera_stat,
            "p_value_approx": self.p_value_approx,
            "sample_size": self.sample_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalityResult:
        return cls(
            metric_id=data["metric_id"],
            is_normal=data["is_normal"],
            skewness=data.get("skewness", 0.0),
            kurtosis=data.get("kurtosis", 0.0),
            jarque_bera_stat=data.get("jarque_bera_stat", 0.0),
            p_value_approx=data.get("p_value_approx", 0.0),
            sample_size=data.get("sample_size", 0),
        )


@dataclass
class NormalityReport:
    """Normality test report across all metrics."""

    results: list[NormalityResult] = field(default_factory=list)
    normal_count: int = 0
    non_normal_count: int = 0
    total_metrics: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "normal_count": self.normal_count,
            "non_normal_count": self.non_normal_count,
            "total_metrics": self.total_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalityReport:
        results = [NormalityResult.from_dict(r) for r in data.get("results", [])]
        return cls(
            results=results,
            normal_count=data.get("normal_count", 0),
            non_normal_count=data.get("non_normal_count", 0),
            total_metrics=data.get("total_metrics", 0),
        )

    def format_text(self) -> str:
        lines = ["Normality Testing Report", "=" * 40]
        lines.append(
            f"  Total metrics: {self.total_metrics}  "
            f"Normal: {self.normal_count}  Non-normal: {self.non_normal_count}"
        )
        lines.append("")
        for r in self.results:
            status = "NORMAL" if r.is_normal else "NON-NORMAL"
            lines.append(
                f"  {r.metric_id}: {status} "
                f"(skew={r.skewness:.3f}, kurt={r.kurtosis:.3f}, "
                f"JB={r.jarque_bera_stat:.3f}, p={r.p_value_approx:.4f}, "
                f"n={r.sample_size})"
            )
            if not r.is_normal:
                suggestions = suggest_transformations(r)
                if suggestions:
                    for s in suggestions:
                        lines.append(f"    - {s}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chi-squared CDF approximation (2 degrees of freedom)
# ---------------------------------------------------------------------------


def _chi2_cdf_2(x: float) -> float:
    """CDF of chi-squared distribution with 2 degrees of freedom.

    For df=2: CDF(x) = 1 - exp(-x/2).
    """
    if x <= 0.0:
        return 0.0
    return 1.0 - math.exp(-x / 2.0)


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def compute_skewness(values: list[float]) -> float:
    """Compute sample skewness.

    Uses the adjusted Fisher-Pearson formula:
    G1 = (n / ((n-1)(n-2))) * sum((x - mean)^3) / std^3
    """
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    m2 = sum((x - mean) ** 2 for x in values) / n
    if m2 == 0.0:
        return 0.0
    std = math.sqrt(m2)
    m3 = sum((x - mean) ** 3 for x in values) / n
    # Adjusted for sample: G1 = (n / ((n-1)(n-2))) * n * m3 / std^3
    # Simplification: G1 = (n^2 / ((n-1)(n-2))) * (m3 / std^3)
    adjustment = n / ((n - 1) * (n - 2))
    g1 = adjustment * n * m3 / (std**3)
    return g1


def compute_kurtosis(values: list[float]) -> float:
    """Compute excess kurtosis (normal distribution = 0).

    Uses sample excess kurtosis formula.
    """
    n = len(values)
    if n < 4:
        return 0.0
    mean = sum(values) / n
    m2 = sum((x - mean) ** 2 for x in values) / n
    if m2 == 0.0:
        return 0.0
    m4 = sum((x - mean) ** 4 for x in values) / n
    # Population excess kurtosis
    kurt_pop = (m4 / (m2**2)) - 3.0
    # Sample correction
    # G2 = ((n-1) / ((n-2)(n-3))) * ((n+1)*kurt_pop + 6)
    g2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * kurt_pop + 6.0)
    return g2


def jarque_bera_test(values: list[float]) -> tuple[float, float]:
    """Compute Jarque-Bera statistic and approximate p-value.

    JB = (n/6) * (S^2 + K^2/4)

    where S = skewness, K = excess kurtosis.
    P-value approximated using chi-squared distribution with 2 df.

    Returns:
        Tuple of (JB statistic, approximate p-value).
    """
    n = len(values)
    if n < 3:
        return (0.0, 1.0)
    s = compute_skewness(values)
    k = compute_kurtosis(values)
    jb = (n / 6.0) * (s**2 + (k**2) / 4.0)
    p_value = 1.0 - _chi2_cdf_2(jb)
    return (jb, p_value)


def check_normality(
    metric_id: str,
    scores: list[float],
    alpha: float = 0.05,
) -> NormalityResult:
    """Run full normality test on a metric's score distribution."""
    n = len(scores)
    if n < 3:
        return NormalityResult(
            metric_id=metric_id,
            is_normal=True,
            sample_size=n,
        )
    s = compute_skewness(scores)
    k = compute_kurtosis(scores)
    jb, p = jarque_bera_test(scores)
    return NormalityResult(
        metric_id=metric_id,
        is_normal=p >= alpha,
        skewness=s,
        kurtosis=k,
        jarque_bera_stat=jb,
        p_value_approx=p,
        sample_size=n,
    )


def check_all_normality(
    metric_scores: dict[str, list[float]],
    alpha: float = 0.05,
) -> NormalityReport:
    """Run normality tests on all metrics."""
    results = []
    for metric_id in sorted(metric_scores):
        results.append(check_normality(metric_id, metric_scores[metric_id], alpha))
    normal_count = sum(1 for r in results if r.is_normal)
    non_normal_count = len(results) - normal_count
    return NormalityReport(
        results=results,
        normal_count=normal_count,
        non_normal_count=non_normal_count,
        total_metrics=len(results),
    )


def suggest_transformations(result: NormalityResult) -> list[str]:
    """Suggest normalizing transforms based on skewness and kurtosis."""
    if result.is_normal:
        return []
    suggestions: list[str] = []
    skew = result.skewness
    kurt = result.kurtosis
    abs_skew = abs(skew)
    if skew > 1.0:
        suggestions.append("log transform for right-skewed data")
    elif skew > 0.5:
        suggestions.append("square root transform for moderate right skew")
    elif skew < -1.0:
        suggestions.append("reflect and log transform for left-skewed data")
    elif skew < -0.5:
        suggestions.append("reflect and square root transform for moderate left skew")
    if abs_skew <= 0.5 and abs(kurt) > 2.0:
        suggestions.append("Box-Cox transform for high kurtosis with low skew")
    if not suggestions:
        suggestions.append("consider non-parametric methods")
    return suggestions
