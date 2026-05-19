"""Metric cold start detection.

Detects when a metric's first N items score differently than the rest,
indicating LLM judge "warming up" or prompt caching effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColdStartResult:
    """Cold start detection result for a single metric."""

    metric_id: str
    cold_start_detected: bool
    cold_mean: float       # mean of first K items
    warm_mean: float       # mean of remaining items
    delta: float           # warm_mean - cold_mean
    p_value_approx: float  # approximate significance
    cold_count: int
    warm_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "cold_start_detected": self.cold_start_detected,
            "cold_mean": round(self.cold_mean, 4),
            "warm_mean": round(self.warm_mean, 4),
            "delta": round(self.delta, 4),
            "p_value_approx": round(self.p_value_approx, 4),
            "cold_count": self.cold_count,
            "warm_count": self.warm_count,
        }

    def format_text(self) -> str:
        status = "DETECTED" if self.cold_start_detected else "not detected"
        return (
            f"{self.metric_id}: cold start {status} "
            f"(cold={self.cold_mean:.3f}, warm={self.warm_mean:.3f}, "
            f"delta={self.delta:+.3f})"
        )


@dataclass
class ColdStartReport:
    """Cold start analysis across all metrics."""

    results: list[ColdStartResult] = field(default_factory=list)

    @property
    def affected_metrics(self) -> list[str]:
        return [r.metric_id for r in self.results if r.cold_start_detected]

    @property
    def has_cold_starts(self) -> bool:
        return len(self.affected_metrics) > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "affected_metrics": self.affected_metrics,
            "has_cold_starts": self.has_cold_starts,
            "results": [r.as_dict() for r in self.results],
        }

    def format_text(self) -> str:
        lines = ["Cold Start Detection"]
        if not self.has_cold_starts:
            lines.append("  No cold start effects detected.")
        else:
            lines.append(f"  Affected metrics: {', '.join(self.affected_metrics)}")
            lines.append("  Recommendation: use --warmup to discard first K results")
        for r in self.results:
            lines.append(f"  {r.format_text()}")
        return "\n".join(lines)


def detect_cold_start(
    scores_by_metric: dict[str, list[float]],
    cold_window: int = 5,
    significance_threshold: float = 0.1,
) -> ColdStartReport:
    """Detect cold start effects across metrics.

    Compares mean scores of the first `cold_window` items against the
    remaining items using a two-sample t-approximation.

    Args:
        scores_by_metric: Dict of metric_id -> list of scores IN EVALUATION ORDER.
        cold_window: Number of initial items to consider "cold" (default 5).
        significance_threshold: Maximum p-value to flag as cold start.

    Returns:
        ColdStartReport with per-metric detection results.
    """
    results = []
    for metric_id, scores in sorted(scores_by_metric.items()):
        if len(scores) < cold_window + 3:
            # Not enough data for meaningful comparison
            results.append(ColdStartResult(
                metric_id=metric_id,
                cold_start_detected=False,
                cold_mean=_mean(scores[:cold_window]) if scores else 0.0,
                warm_mean=_mean(scores[cold_window:]) if len(scores) > cold_window else 0.0,
                delta=0.0,
                p_value_approx=1.0,
                cold_count=min(cold_window, len(scores)),
                warm_count=max(0, len(scores) - cold_window),
            ))
            continue

        cold = scores[:cold_window]
        warm = scores[cold_window:]

        cold_mean = _mean(cold)
        warm_mean = _mean(warm)
        delta = warm_mean - cold_mean

        # Approximate two-sample t-test (Welch's)
        p_value = _welch_t_test_p(cold, warm)

        detected = p_value < significance_threshold and abs(delta) > 0.05

        results.append(ColdStartResult(
            metric_id=metric_id,
            cold_start_detected=detected,
            cold_mean=cold_mean,
            warm_mean=warm_mean,
            delta=delta,
            p_value_approx=p_value,
            cold_count=len(cold),
            warm_count=len(warm),
        ))

    return ColdStartReport(results=results)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _welch_t_test_p(a: list[float], b: list[float]) -> float:
    """Approximate p-value for Welch's t-test (two-tailed)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 1.0

    ma, mb = _mean(a), _mean(b)
    sa, sb = _std(a), _std(b)

    se = math.sqrt(sa ** 2 / na + sb ** 2 / nb) if (sa > 0 or sb > 0) else 0
    if se == 0:
        return 1.0 if ma == mb else 0.0

    t = abs(ma - mb) / se

    # Approximate p-value using normal distribution (good for df > 30)
    # For small df, this is conservative (actual p would be larger)
    from .stats import _normal_cdf
    p = 2 * (1 - _normal_cdf(t))
    return max(0.0, min(1.0, p))
