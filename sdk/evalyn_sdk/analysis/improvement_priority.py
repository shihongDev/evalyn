"""
Improvement priority ranking: rank metrics by expected ROI of improvement.

Helps teams focus on the metrics where improvement effort will yield
the greatest overall evaluation gains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class MetricROI:
    """Expected ROI of improving a single metric."""

    metric_id: str
    current_pass_rate: float
    weight: float = 1.0
    headroom: float = 0.0
    projected_overall_lift: float = 0.0
    priority_score: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "current_pass_rate": self.current_pass_rate,
            "weight": self.weight,
            "headroom": self.headroom,
            "projected_overall_lift": self.projected_overall_lift,
            "priority_score": self.priority_score,
        }


@dataclass
class PriorityReport:
    """Full improvement priority ranking report."""

    rankings: List[MetricROI] = field(default_factory=list)
    top_priority: str = ""
    total_metrics: int = 0
    max_projected_lift: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rankings": [r.as_dict() for r in self.rankings],
            "top_priority": self.top_priority,
            "total_metrics": self.total_metrics,
            "max_projected_lift": self.max_projected_lift,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PriorityReport:
        rankings = []
        for r in data.get("rankings", []):
            rankings.append(
                MetricROI(
                    metric_id=r["metric_id"],
                    current_pass_rate=r["current_pass_rate"],
                    weight=r.get("weight", 1.0),
                    headroom=r.get("headroom", 0.0),
                    projected_overall_lift=r.get("projected_overall_lift", 0.0),
                    priority_score=r.get("priority_score", 0.0),
                )
            )
        return cls(
            rankings=rankings,
            top_priority=data.get("top_priority", ""),
            total_metrics=data.get("total_metrics", 0),
            max_projected_lift=data.get("max_projected_lift", 0.0),
        )

    def format_text(self) -> str:
        """Render a human-readable text summary."""
        lines = [
            "IMPROVEMENT PRIORITY RANKING",
            "=" * 50,
            f"Total metrics: {self.total_metrics}",
            f"Top priority: {self.top_priority}",
            f"Max projected lift: {self.max_projected_lift:.4f}",
            "",
            f"  {'Metric':<25} {'Pass%':>7} {'Headroom':>9} {'Score':>8}",
            f"  {'-' * 25} {'-' * 7} {'-' * 9} {'-' * 8}",
        ]
        for r in self.rankings:
            lines.append(
                f"  {r.metric_id:<25} {r.current_pass_rate * 100:>6.1f}% "
                f"{r.headroom:>8.3f} {r.priority_score:>8.3f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_headroom(pass_rate: float) -> float:
    """1.0 - pass_rate. Higher means more room to improve."""
    return 1.0 - pass_rate


def compute_priority_score(headroom: float, weight: float, item_count: int) -> float:
    """Prioritize high-weight metrics with lots of room and many items.

    score = headroom * weight * log2(item_count + 1)
    """
    return headroom * weight * math.log2(item_count + 1)


def project_overall_lift(
    metric_id: str,
    current_rates: Dict[str, float],
    weights: Dict[str, float] | None = None,
    improvement: float = 0.5,
) -> float:
    """Project how much overall weighted pass rate would increase if metric_id improved.

    If the target metric improved by `improvement` fraction of its headroom,
    compute the difference in weighted-average pass rate.

    Args:
        metric_id: The metric to hypothetically improve.
        current_rates: {metric_id: pass_rate} for all metrics.
        weights: Optional {metric_id: weight}. Defaults to equal weight.
        improvement: Fraction of headroom to close (0.0 to 1.0).

    Returns:
        Projected increase in overall weighted pass rate.
    """
    if weights is None:
        weights = {}
    if metric_id not in current_rates:
        return 0.0

    total_weight = 0.0
    current_weighted_sum = 0.0
    new_weighted_sum = 0.0

    for mid, rate in current_rates.items():
        w = weights.get(mid, 1.0)
        total_weight += w
        current_weighted_sum += rate * w
        if mid == metric_id:
            headroom = compute_headroom(rate)
            new_rate = rate + headroom * improvement
            new_weighted_sum += new_rate * w
        else:
            new_weighted_sum += rate * w

    if total_weight == 0:
        return 0.0

    current_overall = current_weighted_sum / total_weight
    new_overall = new_weighted_sum / total_weight
    return new_overall - current_overall


def rank_improvements(
    metric_rates: Dict[str, float],
    weights: Dict[str, float] | None = None,
    item_counts: Dict[str, int] | None = None,
) -> PriorityReport:
    """Produce a full improvement priority ranking.

    Args:
        metric_rates: {metric_id: pass_rate} for each metric.
        weights: Optional {metric_id: weight}. Defaults to 1.0.
        item_counts: Optional {metric_id: num_items}. Defaults to 1.

    Returns:
        PriorityReport with rankings sorted by priority_score descending.
    """
    if weights is None:
        weights = {}
    if item_counts is None:
        item_counts = {}

    rankings: List[MetricROI] = []
    for mid, rate in metric_rates.items():
        w = weights.get(mid, 1.0)
        count = item_counts.get(mid, 1)
        headroom = compute_headroom(rate)
        score = compute_priority_score(headroom, w, count)
        lift = project_overall_lift(mid, metric_rates, weights)
        rankings.append(
            MetricROI(
                metric_id=mid,
                current_pass_rate=rate,
                weight=w,
                headroom=headroom,
                projected_overall_lift=lift,
                priority_score=score,
            )
        )

    rankings.sort(key=lambda r: r.priority_score, reverse=True)

    top = rankings[0].metric_id if rankings else ""
    max_lift = max((r.projected_overall_lift for r in rankings), default=0.0)

    return PriorityReport(
        rankings=rankings,
        top_priority=top,
        total_metrics=len(rankings),
        max_projected_lift=max_lift,
    )


def render_priority_chart(report: PriorityReport, width: int = 60) -> str:
    """ASCII bar chart of priority scores."""
    if not report.rankings:
        return "No metrics to chart."

    max_score = max(r.priority_score for r in report.rankings)
    if max_score == 0:
        max_score = 1.0

    lines: List[str] = ["PRIORITY CHART", "=" * width]
    for r in report.rankings:
        bar_len = int((r.priority_score / max_score) * (width - 30))
        bar_len = max(bar_len, 0)
        bar = "#" * bar_len
        lines.append(f"  {r.metric_id:<20} |{bar} {r.priority_score:.3f}")
    return "\n".join(lines)


def suggest_improvements(
    report: PriorityReport, max_suggestions: int = 3
) -> List[str]:
    """Return top improvement suggestions as human-readable strings."""
    suggestions: List[str] = []
    for r in report.rankings[:max_suggestions]:
        pct = r.current_pass_rate * 100
        suggestions.append(
            f"Improve {r.metric_id}: current pass rate {pct:.0f}%, "
            f"headroom {r.headroom:.2f}, priority score {r.priority_score:.3f}"
        )
    return suggestions
