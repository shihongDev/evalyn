"""
Metric budget analysis: estimate cost savings from dropping low-signal metrics.

Computes signal strength, detects redundant metric pairs, and generates
budget recommendations with estimated savings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class MetricCostProfile:
    """Cost and signal profile for a single metric."""

    metric_id: str
    avg_cost_per_item: float = 0.0
    avg_tokens_per_item: int = 0
    signal_strength: float = 0.0
    is_redundant: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "avg_cost_per_item": self.avg_cost_per_item,
            "avg_tokens_per_item": self.avg_tokens_per_item,
            "signal_strength": self.signal_strength,
            "is_redundant": self.is_redundant,
        }


@dataclass
class BudgetRecommendation:
    """Recommendation for a metric: keep, drop, or replace."""

    metric_id: str
    action: str = "keep"
    reason: str = ""
    estimated_savings_pct: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "action": self.action,
            "reason": self.reason,
            "estimated_savings_pct": self.estimated_savings_pct,
        }


@dataclass
class MetricBudgetReport:
    """Full budget analysis report."""

    profiles: list[MetricCostProfile] = field(default_factory=list)
    recommendations: list[BudgetRecommendation] = field(default_factory=list)
    total_cost_usd: float = 0.0
    potential_savings_usd: float = 0.0
    potential_savings_pct: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "profiles": [p.as_dict() for p in self.profiles],
            "recommendations": [r.as_dict() for r in self.recommendations],
            "total_cost_usd": self.total_cost_usd,
            "potential_savings_usd": self.potential_savings_usd,
            "potential_savings_pct": self.potential_savings_pct,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricBudgetReport:
        profiles = [
            MetricCostProfile(
                metric_id=p["metric_id"],
                avg_cost_per_item=p.get("avg_cost_per_item", 0.0),
                avg_tokens_per_item=p.get("avg_tokens_per_item", 0),
                signal_strength=p.get("signal_strength", 0.0),
                is_redundant=p.get("is_redundant", False),
            )
            for p in data.get("profiles", [])
        ]
        recommendations = [
            BudgetRecommendation(
                metric_id=r["metric_id"],
                action=r.get("action", "keep"),
                reason=r.get("reason", ""),
                estimated_savings_pct=r.get("estimated_savings_pct", 0.0),
            )
            for r in data.get("recommendations", [])
        ]
        return cls(
            profiles=profiles,
            recommendations=recommendations,
            total_cost_usd=data.get("total_cost_usd", 0.0),
            potential_savings_usd=data.get("potential_savings_usd", 0.0),
            potential_savings_pct=data.get("potential_savings_pct", 0.0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("METRIC BUDGET REPORT")
        lines.append(f"Total cost: ${self.total_cost_usd:.4f}")
        lines.append(
            f"Potential savings: ${self.potential_savings_usd:.4f}"
            f" ({self.potential_savings_pct:.1f}%)"
        )
        lines.append("")
        lines.append("Profiles:")
        for p in self.profiles:
            tag = " [redundant]" if p.is_redundant else ""
            lines.append(
                f"  {p.metric_id}: cost=${p.avg_cost_per_item:.4f}"
                f"  tokens={p.avg_tokens_per_item}"
                f"  signal={p.signal_strength:.3f}{tag}"
            )
        lines.append("")
        lines.append("Recommendations:")
        for r in self.recommendations:
            lines.append(f"  {r.metric_id}: {r.action} - {r.reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

LOW_SIGNAL_THRESHOLD = 0.3


def compute_signal_strength(
    metric_scores: list[float],
    overall_scores: list[float],
) -> float:
    """Pearson correlation between metric scores and overall pass rate.

    Returns 0.0 if correlation is undefined (constant data, too few items).
    """
    n = len(metric_scores)
    if n != len(overall_scores) or n < 2:
        return 0.0

    mean_m = sum(metric_scores) / n
    mean_o = sum(overall_scores) / n

    cov = sum((m - mean_m) * (o - mean_o) for m, o in zip(metric_scores, overall_scores)) / n

    std_m = math.sqrt(sum((m - mean_m) ** 2 for m in metric_scores) / n)
    std_o = math.sqrt(sum((o - mean_o) ** 2 for o in overall_scores) / n)

    if std_m == 0 or std_o == 0:
        return 0.0

    return cov / (std_m * std_o)


def detect_redundant_metrics(
    metric_scores: dict[str, list[float]],
    threshold: float = 0.9,
) -> list[tuple[str, str]]:
    """Find pairs of metrics with correlation above threshold."""
    ids = sorted(metric_scores.keys())
    pairs: list[tuple[str, str]] = []

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            scores_a = metric_scores[a]
            scores_b = metric_scores[b]
            if len(scores_a) != len(scores_b) or len(scores_a) < 2:
                continue
            r = compute_signal_strength(scores_a, scores_b)
            if r >= threshold:
                pairs.append((a, b))

    return pairs


def build_cost_profile(
    metric_id: str,
    costs: list[float],
    tokens: list[int],
    signal: float = 0.0,
) -> MetricCostProfile:
    """Build a MetricCostProfile from per-item cost and token data."""
    avg_cost = sum(costs) / len(costs) if costs else 0.0
    avg_tokens = int(sum(tokens) / len(tokens)) if tokens else 0
    return MetricCostProfile(
        metric_id=metric_id,
        avg_cost_per_item=avg_cost,
        avg_tokens_per_item=avg_tokens,
        signal_strength=signal,
    )


def generate_budget_recommendations(
    profiles: list[MetricCostProfile],
    redundant_pairs: list[tuple[str, str]] | None = None,
) -> list[BudgetRecommendation]:
    """Recommend drop for low-signal or redundant metrics.

    Rules:
    - signal_strength < LOW_SIGNAL_THRESHOLD and not redundant: drop (low signal)
    - redundant pair: replace the one with lower signal (or higher cost if tied)
    - otherwise: keep
    """
    if redundant_pairs is None:
        redundant_pairs = []

    profile_map = {p.metric_id: p for p in profiles}

    # Mark redundant metrics
    redundant_ids: set[str] = set()
    replace_ids: set[str] = set()
    for a, b in redundant_pairs:
        redundant_ids.add(a)
        redundant_ids.add(b)
        # Replace the weaker one (lower signal, or higher cost if tied)
        pa = profile_map.get(a)
        pb = profile_map.get(b)
        if pa and pb:
            if pa.signal_strength < pb.signal_strength:
                replace_ids.add(a)
            elif pb.signal_strength < pa.signal_strength:
                replace_ids.add(b)
            else:
                # Tie: drop the more expensive one
                if pa.avg_cost_per_item >= pb.avg_cost_per_item:
                    replace_ids.add(a)
                else:
                    replace_ids.add(b)

    recs: list[BudgetRecommendation] = []
    total_cost = sum(p.avg_cost_per_item for p in profiles)

    for p in profiles:
        savings_pct = (p.avg_cost_per_item / total_cost * 100) if total_cost > 0 else 0.0

        if p.metric_id in replace_ids:
            recs.append(
                BudgetRecommendation(
                    metric_id=p.metric_id,
                    action="replace",
                    reason="redundant with higher-signal metric",
                    estimated_savings_pct=savings_pct,
                )
            )
        elif p.signal_strength < LOW_SIGNAL_THRESHOLD and p.metric_id not in redundant_ids:
            recs.append(
                BudgetRecommendation(
                    metric_id=p.metric_id,
                    action="drop",
                    reason="low signal strength",
                    estimated_savings_pct=savings_pct,
                )
            )
        else:
            recs.append(
                BudgetRecommendation(
                    metric_id=p.metric_id,
                    action="keep",
                    reason="adequate signal",
                    estimated_savings_pct=0.0,
                )
            )

    return recs


def build_budget_report(
    profiles: list[MetricCostProfile],
    redundant_pairs: list[tuple[str, str]] | None = None,
) -> MetricBudgetReport:
    """Build a full budget analysis report."""
    if redundant_pairs is None:
        redundant_pairs = []

    # Mark redundant flags on profiles
    redundant_ids: set[str] = set()
    for a, b in redundant_pairs:
        redundant_ids.add(a)
        redundant_ids.add(b)
    for p in profiles:
        p.is_redundant = p.metric_id in redundant_ids

    recs = generate_budget_recommendations(profiles, redundant_pairs)
    total_cost = sum(p.avg_cost_per_item for p in profiles)
    drop_cost = sum(
        p.avg_cost_per_item for p, r in zip(profiles, recs) if r.action in ("drop", "replace")
    )
    # Match recommendations to profiles by metric_id for accurate calculation
    rec_map = {r.metric_id: r for r in recs}
    drop_cost = sum(
        p.avg_cost_per_item
        for p in profiles
        if rec_map.get(p.metric_id, BudgetRecommendation(metric_id="")).action
        in ("drop", "replace")
    )
    savings_pct = (drop_cost / total_cost * 100) if total_cost > 0 else 0.0

    return MetricBudgetReport(
        profiles=profiles,
        recommendations=recs,
        total_cost_usd=total_cost,
        potential_savings_usd=drop_cost,
        potential_savings_pct=savings_pct,
    )


def estimate_savings(report: MetricBudgetReport) -> dict[str, float]:
    """Per-metric savings (in USD) if dropped."""
    result: dict[str, float] = {}
    rec_map = {r.metric_id: r for r in report.recommendations}
    for p in report.profiles:
        rec = rec_map.get(p.metric_id)
        if rec and rec.action in ("drop", "replace"):
            result[p.metric_id] = p.avg_cost_per_item
        else:
            result[p.metric_id] = 0.0
    return result
