"""Run quality score: composite metric summarizing overall run health.

Combines pass rate, cost efficiency, coverage, and judge confidence
into a single 0-100 score for quick run quality assessment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QualityScoreBreakdown:
    """Breakdown of the quality score components."""

    pass_rate_score: float = 0.0      # 0-100, based on overall pass rate
    coverage_score: float = 0.0       # 0-100, based on metric/item coverage
    cost_efficiency_score: float = 0.0  # 0-100, based on cost per item
    error_rate_score: float = 0.0     # 0-100, inverse of error rate

    # Weights for each component
    pass_rate_weight: float = 0.5
    coverage_weight: float = 0.2
    cost_efficiency_weight: float = 0.15
    error_rate_weight: float = 0.15

    @property
    def overall(self) -> float:
        """Weighted composite score (0-100)."""
        return (
            self.pass_rate_score * self.pass_rate_weight
            + self.coverage_score * self.coverage_weight
            + self.cost_efficiency_score * self.cost_efficiency_weight
            + self.error_rate_score * self.error_rate_weight
        )

    @property
    def grade(self) -> str:
        """Letter grade from the overall score."""
        s = self.overall
        if s >= 90:
            return "A"
        if s >= 80:
            return "B"
        if s >= 70:
            return "C"
        if s >= 60:
            return "D"
        return "F"

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 1),
            "grade": self.grade,
            "pass_rate_score": round(self.pass_rate_score, 1),
            "coverage_score": round(self.coverage_score, 1),
            "cost_efficiency_score": round(self.cost_efficiency_score, 1),
            "error_rate_score": round(self.error_rate_score, 1),
        }

    def format_text(self) -> str:
        lines = [
            f"Run Quality: {self.overall:.0f}/100 (Grade: {self.grade})",
            f"  Pass rate:       {self.pass_rate_score:.0f}/100 (weight: {self.pass_rate_weight:.0%})",
            f"  Coverage:        {self.coverage_score:.0f}/100 (weight: {self.coverage_weight:.0%})",
            f"  Cost efficiency: {self.cost_efficiency_score:.0f}/100 (weight: {self.cost_efficiency_weight:.0%})",
            f"  Error rate:      {self.error_rate_score:.0f}/100 (weight: {self.error_rate_weight:.0%})",
        ]
        return "\n".join(lines)


def compute_quality_score(
    overall_pass_rate: float,
    total_items: int,
    total_metrics: int,
    total_cost_usd: float = 0.0,
    error_count: int = 0,
    total_evaluations: int = 0,
    *,
    weights: dict[str, float] | None = None,
) -> QualityScoreBreakdown:
    """Compute a composite quality score for an evaluation run.

    Args:
        overall_pass_rate: Fraction of items passing all metrics (0.0-1.0).
        total_items: Number of items evaluated.
        total_metrics: Number of metrics applied.
        total_cost_usd: Total LLM cost in USD.
        error_count: Number of evaluation errors (failed to score).
        total_evaluations: Total item-metric evaluations attempted.
        weights: Optional custom weights for components.

    Returns:
        QualityScoreBreakdown with component scores and overall.
    """
    breakdown = QualityScoreBreakdown()

    if weights:
        breakdown.pass_rate_weight = weights.get("pass_rate", 0.5)
        breakdown.coverage_weight = weights.get("coverage", 0.2)
        breakdown.cost_efficiency_weight = weights.get("cost_efficiency", 0.15)
        breakdown.error_rate_weight = weights.get("error_rate", 0.15)

    # Pass rate score: direct mapping 0-1 -> 0-100
    breakdown.pass_rate_score = overall_pass_rate * 100

    # Coverage score: based on items * metrics evaluated
    if total_items > 0 and total_metrics > 0:
        expected = total_items * total_metrics
        actual = total_evaluations if total_evaluations > 0 else expected
        coverage = min(actual / expected, 1.0)
        breakdown.coverage_score = coverage * 100
    else:
        breakdown.coverage_score = 0.0

    # Cost efficiency: lower cost per item = better score
    # Score 100 if free, 50 at $0.01/item, 0 at $0.10/item
    if total_items > 0 and total_cost_usd > 0:
        cost_per_item = total_cost_usd / total_items
        # Exponential decay: score = 100 * exp(-cost_per_item / 0.02)
        import math
        breakdown.cost_efficiency_score = min(100.0, 100.0 * math.exp(-cost_per_item / 0.02))
    else:
        breakdown.cost_efficiency_score = 100.0  # free = perfect

    # Error rate score: inverse of error rate
    if total_evaluations > 0:
        error_rate = error_count / total_evaluations
        breakdown.error_rate_score = max(0.0, (1.0 - error_rate * 10) * 100)  # 10% errors = 0 score
    else:
        breakdown.error_rate_score = 100.0  # no evaluations = no errors

    return breakdown
