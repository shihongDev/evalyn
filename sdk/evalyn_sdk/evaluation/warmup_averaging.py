"""Metric warmup averaging: run each metric N times and average scores.

Reduces variance from LLM non-determinism by running each subjective
metric evaluation multiple times and averaging the results.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AveragedResult:
    """Result from averaging multiple metric evaluations."""

    metric_id: str
    item_id: str
    samples: int
    scores: list[float] = field(default_factory=list)
    passed_votes: int = 0
    failed_votes: int = 0

    @property
    def mean_score(self) -> float:
        return statistics.mean(self.scores) if self.scores else 0.0

    @property
    def score_std(self) -> float:
        return statistics.stdev(self.scores) if len(self.scores) >= 2 else 0.0

    @property
    def majority_passed(self) -> bool:
        return self.passed_votes > self.failed_votes

    @property
    def agreement_rate(self) -> float:
        total = self.passed_votes + self.failed_votes
        if total == 0:
            return 0.0
        majority = max(self.passed_votes, self.failed_votes)
        return majority / total

    @property
    def high_variance(self) -> bool:
        """Flag if score variance is high (suggests unreliable metric)."""
        return self.score_std > 0.15 and len(self.scores) >= 3

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "item_id": self.item_id,
            "samples": self.samples,
            "mean_score": round(self.mean_score, 4),
            "score_std": round(self.score_std, 4),
            "majority_passed": self.majority_passed,
            "agreement_rate": round(self.agreement_rate, 4),
            "high_variance": self.high_variance,
            "passed_votes": self.passed_votes,
            "failed_votes": self.failed_votes,
        }


@dataclass
class WarmupAveragingReport:
    """Report from warmup averaging across all items and metrics."""

    results: list[AveragedResult] = field(default_factory=list)

    @property
    def high_variance_count(self) -> int:
        return sum(1 for r in self.results if r.high_variance)

    @property
    def avg_agreement(self) -> float:
        if not self.results:
            return 0.0
        return statistics.mean(r.agreement_rate for r in self.results)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_evaluations": len(self.results),
            "high_variance_count": self.high_variance_count,
            "avg_agreement": round(self.avg_agreement, 4),
            "results": [r.as_dict() for r in self.results],
        }

    def format_text(self) -> str:
        lines = [
            "Warmup Averaging Report:",
            f"  Evaluations:      {len(self.results)}",
            f"  High variance:    {self.high_variance_count}",
            f"  Avg agreement:    {self.avg_agreement:.1%}",
        ]
        hv = [r for r in self.results if r.high_variance]
        if hv:
            lines.append("\n  High-variance items:")
            for r in hv[:10]:
                lines.append(
                    f"    {r.metric_id}/{r.item_id}: std={r.score_std:.3f}, agreement={r.agreement_rate:.0%}"
                )
        return "\n".join(lines)


def average_results(
    sample_results: list[list],
    metric_id: str,
    item_id: str,
) -> AveragedResult:
    """Average multiple evaluation results for one metric-item pair.

    Args:
        sample_results: List of MetricResult lists (one per sample run).
        metric_id: Metric being averaged.
        item_id: Item being evaluated.

    Returns:
        AveragedResult with mean score, majority vote, and variance.
    """
    scores = []
    passed = 0
    failed = 0

    for results in sample_results:
        for r in results:
            if r.metric_id == metric_id and r.item_id == item_id:
                if r.score is not None:
                    scores.append(r.score)
                if r.passed is True:
                    passed += 1
                elif r.passed is False:
                    failed += 1

    return AveragedResult(
        metric_id=metric_id,
        item_id=item_id,
        samples=len(sample_results),
        scores=scores,
        passed_votes=passed,
        failed_votes=failed,
    )
