"""Metric contribution analysis: SHAP-style attribution.

Measures each metric's contribution to overall pass/fail outcomes
by computing marginal contributions - how often removing a metric
would change the overall result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class MetricContribution:
    """Contribution of a single metric to overall pass/fail."""

    metric_id: str
    contribution_score: float
    direction: str = "positive"  # "positive", "negative", "neutral"
    items_influenced: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "contribution_score": self.contribution_score,
            "direction": self.direction,
            "items_influenced": self.items_influenced,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetricContribution:
        return cls(
            metric_id=data["metric_id"],
            contribution_score=data["contribution_score"],
            direction=data.get("direction", "positive"),
            items_influenced=data.get("items_influenced", 0),
        )


@dataclass
class ContributionReport:
    """Report of metric contributions across all metrics."""

    contributions: List[MetricContribution] = field(default_factory=list)
    most_positive: str = ""
    most_negative: str = ""
    total_items: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contributions": [c.as_dict() for c in self.contributions],
            "most_positive": self.most_positive,
            "most_negative": self.most_negative,
            "total_items": self.total_items,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContributionReport:
        contributions = [
            MetricContribution.from_dict(c)
            for c in data.get("contributions", [])
        ]
        return cls(
            contributions=contributions,
            most_positive=data.get("most_positive", ""),
            most_negative=data.get("most_negative", ""),
            total_items=data.get("total_items", 0),
        )

    def format_text(self) -> str:
        lines = ["Metric Contribution Analysis", "=" * 40]
        lines.append(f"  Total items: {self.total_items}")
        if self.most_positive:
            lines.append(f"  Most positive contributor: {self.most_positive}")
        if self.most_negative:
            lines.append(f"  Most negative contributor: {self.most_negative}")
        lines.append("")
        for c in sorted(
            self.contributions,
            key=lambda x: abs(x.contribution_score),
            reverse=True,
        ):
            sign = "+" if c.direction == "positive" else (
                "-" if c.direction == "negative" else " "
            )
            lines.append(
                f"  {sign} {c.metric_id}: {c.contribution_score:.3f} "
                f"({c.items_influenced} items influenced)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def _overall_without_metric(
    item: Dict[str, Any],
    exclude_metric: str,
) -> bool:
    """Recompute overall pass/fail excluding one metric.

    Uses majority rule: pass if more than half of remaining metrics pass.
    If no remaining metrics, default to pass.
    """
    metrics = item.get("metrics", {})
    remaining = {
        mid: mdata for mid, mdata in metrics.items() if mid != exclude_metric
    }
    if not remaining:
        return True
    passed_count = sum(1 for mdata in remaining.values() if mdata.get("passed", False))
    return passed_count > len(remaining) / 2


def compute_marginal_contribution(
    metric_id: str,
    item_results: List[Dict[str, Any]],
) -> MetricContribution:
    """Compute marginal contribution of a metric.

    For each item, check if removing this metric's result would change
    the overall pass/fail outcome. Contribution is the fraction of items
    where this metric is the deciding factor.

    Positive contribution: metric helps items pass (removing it would
    flip pass -> fail).
    Negative contribution: metric causes items to fail (removing it would
    flip fail -> pass).
    """
    if not item_results:
        return MetricContribution(
            metric_id=metric_id,
            contribution_score=0.0,
            direction="neutral",
            items_influenced=0,
        )
    positive_influence = 0
    negative_influence = 0
    for item in item_results:
        overall = item.get("overall_passed", True)
        without = _overall_without_metric(item, metric_id)
        if overall and not without:
            # Removing this metric flips pass -> fail: metric helps pass
            positive_influence += 1
        elif not overall and without:
            # Removing this metric flips fail -> pass: metric causes fail
            negative_influence += 1

    n = len(item_results)
    net = positive_influence - negative_influence
    score = net / n if n > 0 else 0.0
    items_influenced = positive_influence + negative_influence

    if net > 0:
        direction = "positive"
    elif net < 0:
        direction = "negative"
    else:
        direction = "neutral"

    return MetricContribution(
        metric_id=metric_id,
        contribution_score=score,
        direction=direction,
        items_influenced=items_influenced,
    )


def compute_all_contributions(
    item_results: List[Dict[str, Any]],
    metric_ids: List[str],
) -> ContributionReport:
    """Compute contributions for all metrics.

    Each item_result has:
        "item_id": str
        "metrics": {metric_id: {"passed": bool}}
        "overall_passed": bool
    """
    contributions = []
    for mid in sorted(metric_ids):
        contributions.append(compute_marginal_contribution(mid, item_results))

    most_positive = ""
    most_negative = ""
    best_score = 0.0
    worst_score = 0.0
    for c in contributions:
        if c.contribution_score > best_score:
            best_score = c.contribution_score
            most_positive = c.metric_id
        if c.contribution_score < worst_score:
            worst_score = c.contribution_score
            most_negative = c.metric_id

    return ContributionReport(
        contributions=contributions,
        most_positive=most_positive,
        most_negative=most_negative,
        total_items=len(item_results),
    )


def rank_by_contribution(
    report: ContributionReport,
) -> List[MetricContribution]:
    """Sort contributions by absolute contribution score descending."""
    return sorted(
        report.contributions,
        key=lambda c: abs(c.contribution_score),
        reverse=True,
    )


def identify_bottleneck_metrics(
    report: ContributionReport,
    threshold: float = 0.3,
) -> List[str]:
    """Identify metrics with high negative contribution (causing most failures).

    Returns metric IDs where contribution_score <= -threshold.
    """
    return [
        c.metric_id
        for c in report.contributions
        if c.contribution_score <= -threshold
    ]


def render_contribution_chart(
    report: ContributionReport,
    width: int = 60,
) -> str:
    """Render an ASCII bar chart of metric contributions.

    Positive contributions go right, negative go left, centered.
    """
    if not report.contributions:
        return "No contributions to display."

    ranked = rank_by_contribution(report)
    max_abs = max(abs(c.contribution_score) for c in ranked)
    if max_abs == 0:
        max_abs = 1.0

    half = width // 2
    lines = ["Metric Contributions", "-" * (width + 20)]

    # Find longest metric name for padding
    max_name = max(len(c.metric_id) for c in ranked) if ranked else 0

    for c in ranked:
        name = c.metric_id.ljust(max_name)
        bar_len = int(abs(c.contribution_score) / max_abs * half)
        bar_len = max(bar_len, 1) if c.contribution_score != 0.0 else 0
        if c.contribution_score >= 0:
            left = " " * half
            right = "#" * bar_len
            line = f"  {name} |{left}|{right}"
        else:
            padding = half - bar_len
            left = " " * padding + "#" * bar_len
            right = ""
            line = f"  {name} |{left}|{right}"
        score_str = f" {c.contribution_score:+.3f}"
        lines.append(line + score_str)

    return "\n".join(lines)
