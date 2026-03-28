"""Calibration A/B testing.

Compare calibrated vs uncalibrated prompts side by side by running
paired score comparisons and summarizing wins, losses, ties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ABComparison:
    """Single item-level comparison between two prompt variants."""

    item_id: str
    metric_id: str
    score_a: float
    score_b: float
    delta: float  # score_a - score_b

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "metric_id": self.metric_id,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "delta": self.delta,
        }


@dataclass
class ABTestResult:
    """Aggregated result of an A/B test for a single metric."""

    metric_id: str
    comparisons: List[ABComparison]
    mean_score_a: float
    mean_score_b: float
    mean_delta: float
    a_wins: int
    b_wins: int
    ties: int
    is_significant: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "comparisons": [c.as_dict() for c in self.comparisons],
            "mean_score_a": self.mean_score_a,
            "mean_score_b": self.mean_score_b,
            "mean_delta": self.mean_delta,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "ties": self.ties,
            "is_significant": self.is_significant,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ABTestResult:
        comparisons = []
        for c in data.get("comparisons", []):
            comparisons.append(
                ABComparison(
                    item_id=c["item_id"],
                    metric_id=c["metric_id"],
                    score_a=c["score_a"],
                    score_b=c["score_b"],
                    delta=c["delta"],
                )
            )
        return cls(
            metric_id=data["metric_id"],
            comparisons=comparisons,
            mean_score_a=data["mean_score_a"],
            mean_score_b=data["mean_score_b"],
            mean_delta=data["mean_delta"],
            a_wins=data["a_wins"],
            b_wins=data["b_wins"],
            ties=data["ties"],
            is_significant=data.get("is_significant", False),
        )

    def format_text(self) -> str:
        lines = [f"A/B Test: {self.metric_id}"]
        lines.append(f"  Mean A: {self.mean_score_a:.4f}  Mean B: {self.mean_score_b:.4f}  Delta: {self.mean_delta:.4f}")
        lines.append(f"  A wins: {self.a_wins}  B wins: {self.b_wins}  Ties: {self.ties}")
        sig = "yes" if self.is_significant else "no"
        lines.append(f"  Significant: {sig}")
        return "\n".join(lines)


@dataclass
class ABTestConfig:
    """Configuration for an A/B calibration test."""

    metric_id: str
    prompt_a: str = ""
    prompt_b: str = ""
    label_a: str = "calibrated"
    label_b: str = "uncalibrated"
    significance_threshold: float = 0.05

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "prompt_a": self.prompt_a,
            "prompt_b": self.prompt_b,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "significance_threshold": self.significance_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ABTestConfig:
        return cls(
            metric_id=data["metric_id"],
            prompt_a=data.get("prompt_a", ""),
            prompt_b=data.get("prompt_b", ""),
            label_a=data.get("label_a", "calibrated"),
            label_b=data.get("label_b", "uncalibrated"),
            significance_threshold=data.get("significance_threshold", 0.05),
        )


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compare_scores(
    item_id: str,
    metric_id: str,
    score_a: float,
    score_b: float,
    tie_margin: float = 0.01,
) -> ABComparison:
    """Create a comparison between two scores for one item.

    delta = score_a - score_b.
    """
    delta = score_a - score_b
    return ABComparison(
        item_id=item_id,
        metric_id=metric_id,
        score_a=score_a,
        score_b=score_b,
        delta=delta,
    )


def run_ab_test(
    metric_id: str,
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
    tie_margin: float = 0.01,
    significance_threshold: float = 0.05,
) -> ABTestResult:
    """Compare all items for a metric and compute aggregate statistics.

    Only items present in both scores_a and scores_b are compared.
    A wins when delta > tie_margin, B wins when delta < -tie_margin,
    otherwise it is a tie.
    is_significant is True when abs(mean_delta) > significance_threshold.
    """
    common_ids = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    comparisons: List[ABComparison] = []
    a_wins = 0
    b_wins = 0
    ties = 0

    for item_id in common_ids:
        comp = compare_scores(item_id, metric_id, scores_a[item_id], scores_b[item_id], tie_margin)
        comparisons.append(comp)
        if comp.delta > tie_margin:
            a_wins += 1
        elif comp.delta < -tie_margin:
            b_wins += 1
        else:
            ties += 1

    if comparisons:
        mean_a = sum(c.score_a for c in comparisons) / len(comparisons)
        mean_b = sum(c.score_b for c in comparisons) / len(comparisons)
        mean_delta = sum(c.delta for c in comparisons) / len(comparisons)
    else:
        mean_a = 0.0
        mean_b = 0.0
        mean_delta = 0.0

    is_significant = abs(mean_delta) > significance_threshold

    return ABTestResult(
        metric_id=metric_id,
        comparisons=comparisons,
        mean_score_a=mean_a,
        mean_score_b=mean_b,
        mean_delta=mean_delta,
        a_wins=a_wins,
        b_wins=b_wins,
        ties=ties,
        is_significant=is_significant,
    )


def summarize_ab_tests(results: List[ABTestResult]) -> Dict[str, Any]:
    """Summarize multiple A/B test results.

    Returns a dict with total metrics tested, improved, degraded, and unchanged.
    """
    improved = 0
    degraded = 0
    unchanged = 0

    for r in results:
        if r.mean_delta > 0:
            improved += 1
        elif r.mean_delta < 0:
            degraded += 1
        else:
            unchanged += 1

    return {
        "total_metrics": len(results),
        "improved": improved,
        "degraded": degraded,
        "unchanged": unchanged,
        "results": [r.as_dict() for r in results],
    }
