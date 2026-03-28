"""Run quality score: composite score summarizing overall run health.

Different from quality_score.py - this module provides a dimension-based
scoring system with explicit weights and comparison support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class QualityDimension:
    """A single scored dimension contributing to overall quality."""

    name: str = ""
    score: float = 0.0  # 0-1
    weight: float = 1.0
    details: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "weight": self.weight,
            "details": self.details,
        }


@dataclass
class RunQualityResult:
    """Composite quality result for a single run."""

    run_id: str = ""
    overall_score: float = 0.0  # 0-100
    grade: str = ""  # A/B/C/D/F
    dimensions: List[QualityDimension] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "overall_score": self.overall_score,
            "grade": self.grade,
            "dimensions": [d.as_dict() for d in self.dimensions],
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunQualityResult:
        dims = []
        for d in data.get("dimensions", []):
            dims.append(QualityDimension(
                name=d.get("name", ""),
                score=d.get("score", 0.0),
                weight=d.get("weight", 1.0),
                details=d.get("details", ""),
            ))
        return cls(
            run_id=data.get("run_id", ""),
            overall_score=data.get("overall_score", 0.0),
            grade=data.get("grade", ""),
            dimensions=dims,
            recommendations=data.get("recommendations", []),
        )

    def format_text(self) -> str:
        lines = [
            f"Run Quality: {self.overall_score:.1f}/100 (Grade: {self.grade})",
        ]
        for dim in self.dimensions:
            lines.append(f"  {dim.name}: {dim.score:.2f} (weight: {dim.weight:.1f})")
            if dim.details:
                lines.append(f"    {dim.details}")
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dimension Scorers
# ---------------------------------------------------------------------------


def score_pass_rate(pass_rate: float) -> QualityDimension:
    """Score based on pass rate. 1.0 = perfect.

    Args:
        pass_rate: Fraction of items passing (0.0-1.0).
    """
    clamped = max(0.0, min(1.0, pass_rate))
    return QualityDimension(
        name="pass_rate",
        score=clamped,
        weight=1.0,
        details=f"{clamped:.0%} of items passing",
    )


def score_coverage(metrics_evaluated: int, total_metrics: int) -> QualityDimension:
    """Score based on fraction of metrics actually run.

    Args:
        metrics_evaluated: Number of metrics that produced results.
        total_metrics: Total metrics configured.
    """
    if total_metrics <= 0:
        ratio = 0.0
    else:
        ratio = min(metrics_evaluated / total_metrics, 1.0)
    return QualityDimension(
        name="coverage",
        score=ratio,
        weight=1.0,
        details=f"{metrics_evaluated}/{total_metrics} metrics evaluated",
    )


def score_confidence(avg_confidence: float) -> QualityDimension:
    """Score based on average judge confidence. Higher = better.

    Args:
        avg_confidence: Average confidence (0.0-1.0).
    """
    clamped = max(0.0, min(1.0, avg_confidence))
    return QualityDimension(
        name="confidence",
        score=clamped,
        weight=1.0,
        details=f"avg confidence {clamped:.2f}",
    )


def score_cost_efficiency(cost_usd: float, item_count: int) -> QualityDimension:
    """Score based on cost per item. Lower cost = better.

    Score = 1 - min(cost_per_item / 0.10, 1.0).
    Zero cost or zero items yields score 1.0.

    Args:
        cost_usd: Total cost in USD.
        item_count: Number of items evaluated.
    """
    if item_count <= 0 or cost_usd <= 0.0:
        return QualityDimension(
            name="cost_efficiency",
            score=1.0,
            weight=1.0,
            details="no cost incurred",
        )
    cost_per_item = cost_usd / item_count
    score = 1.0 - min(cost_per_item / 0.10, 1.0)
    return QualityDimension(
        name="cost_efficiency",
        score=max(0.0, score),
        weight=1.0,
        details=f"${cost_per_item:.4f}/item",
    )


# ---------------------------------------------------------------------------
# Composite Scoring
# ---------------------------------------------------------------------------


def _assign_grade(score: float) -> str:
    """Assign letter grade from 0-100 score."""
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def compute_run_quality(
    run_id: str,
    pass_rate: float,
    metrics_evaluated: int = 1,
    total_metrics: int = 1,
    avg_confidence: float = 1.0,
    cost_usd: float = 0.0,
    item_count: int = 1,
) -> RunQualityResult:
    """Compute overall run quality from individual dimensions.

    Args:
        run_id: Identifier for the run.
        pass_rate: Fraction of items passing (0.0-1.0).
        metrics_evaluated: Number of metrics that produced results.
        total_metrics: Total metrics configured.
        avg_confidence: Average confidence score (0.0-1.0).
        cost_usd: Total cost in USD.
        item_count: Number of items evaluated.

    Returns:
        RunQualityResult with overall score, grade, and dimension breakdown.
    """
    dims = [
        score_pass_rate(pass_rate),
        score_coverage(metrics_evaluated, total_metrics),
        score_confidence(avg_confidence),
        score_cost_efficiency(cost_usd, item_count),
    ]

    total_weight = sum(d.weight for d in dims)
    if total_weight > 0:
        weighted_sum = sum(d.score * d.weight for d in dims)
        overall = (weighted_sum / total_weight) * 100
    else:
        overall = 0.0

    grade = _assign_grade(overall)

    recommendations: List[str] = []
    for dim in dims:
        if dim.name == "pass_rate" and dim.score < 0.7:
            recommendations.append("Pass rate below 70% - review failing items")
        if dim.name == "coverage" and dim.score < 1.0:
            recommendations.append("Not all metrics evaluated - check metric configuration")
        if dim.name == "confidence" and dim.score < 0.5:
            recommendations.append("Low judge confidence - consider stronger models or clearer rubrics")
        if dim.name == "cost_efficiency" and dim.score < 0.5:
            recommendations.append("High cost per item - consider caching or cheaper models")

    return RunQualityResult(
        run_id=run_id,
        overall_score=overall,
        grade=grade,
        dimensions=dims,
        recommendations=recommendations,
    )


def compare_run_quality(
    result_a: RunQualityResult,
    result_b: RunQualityResult,
) -> Dict[str, Any]:
    """Compare quality between two runs.

    Args:
        result_a: First run quality result.
        result_b: Second run quality result.

    Returns:
        Dict with overall delta, per-dimension deltas, and which run is better.
    """
    overall_delta = result_b.overall_score - result_a.overall_score

    dim_deltas: Dict[str, float] = {}
    dims_a = {d.name: d.score for d in result_a.dimensions}
    dims_b = {d.name: d.score for d in result_b.dimensions}
    all_names = set(dims_a.keys()) | set(dims_b.keys())
    for name in sorted(all_names):
        score_a = dims_a.get(name, 0.0)
        score_b = dims_b.get(name, 0.0)
        dim_deltas[name] = score_b - score_a

    if overall_delta > 0:
        better = result_b.run_id
    elif overall_delta < 0:
        better = result_a.run_id
    else:
        better = "tie"

    return {
        "run_a": result_a.run_id,
        "run_b": result_b.run_id,
        "overall_delta": overall_delta,
        "dimension_deltas": dim_deltas,
        "better": better,
        "grade_a": result_a.grade,
        "grade_b": result_b.grade,
    }
