"""Evaluation warm-up: discard first K results to reduce cold-start variance.

Run the first K items, discard their results, then re-evaluate them
alongside the remaining items for consistent scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WarmupConfig:
    """Configuration for evaluation warm-up."""

    warmup_count: int = 5       # number of initial items to warm up on
    discard_warmup: bool = True  # if True, re-evaluate warmup items after priming
    enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "warmup_count": self.warmup_count,
            "discard_warmup": self.discard_warmup,
            "enabled": self.enabled,
        }


@dataclass
class WarmupResult:
    """Result of warm-up phase."""

    warmup_items_count: int
    warmup_results_discarded: int
    warmup_pass_rate: float  # pass rate of discarded warmup results
    post_warmup_pass_rate: float  # pass rate after warm-up
    variance_reduction: float  # estimated variance reduction

    def as_dict(self) -> dict[str, Any]:
        return {
            "warmup_items_count": self.warmup_items_count,
            "warmup_results_discarded": self.warmup_results_discarded,
            "warmup_pass_rate": round(self.warmup_pass_rate, 4),
            "post_warmup_pass_rate": round(self.post_warmup_pass_rate, 4),
            "variance_reduction": round(self.variance_reduction, 4),
        }

    def format_text(self) -> str:
        lines = [
            "Warm-Up Summary:",
            f"  Items warmed up:     {self.warmup_items_count}",
            f"  Results discarded:   {self.warmup_results_discarded}",
            f"  Warmup pass rate:    {self.warmup_pass_rate:.1%}",
            f"  Post-warmup rate:    {self.post_warmup_pass_rate:.1%}",
        ]
        if self.variance_reduction > 0:
            lines.append(f"  Variance reduction:  {self.variance_reduction:.1%}")
        return "\n".join(lines)


def split_warmup_items(
    items: list,
    warmup_count: int = 5,
) -> tuple:
    """Split items into warmup set and evaluation set.

    Args:
        items: All dataset items.
        warmup_count: Number of items for warmup.

    Returns:
        Tuple of (warmup_items, eval_items). When discard_warmup=False,
        eval_items includes the warmup items.
    """
    if warmup_count >= len(items):
        return list(items), list(items)

    warmup = items[:warmup_count]
    remaining = items[warmup_count:]
    return warmup, remaining


def analyze_warmup_effect(
    warmup_results: list,
    post_warmup_results: list,
) -> WarmupResult:
    """Analyze the effect of warm-up on score quality.

    Compares pass rates and variance between warmup (discarded) results
    and post-warmup (kept) results.

    Args:
        warmup_results: MetricResult list from warmup phase (discarded).
        post_warmup_results: MetricResult list from main evaluation.

    Returns:
        WarmupResult with comparison data.
    """
    warmup_pr = _pass_rate(warmup_results)
    post_pr = _pass_rate(post_warmup_results)

    # Estimate variance reduction
    warmup_var = _score_variance(warmup_results)
    post_var = _score_variance(post_warmup_results)
    if warmup_var > 0:
        var_reduction = max(0, 1 - post_var / warmup_var)
    else:
        var_reduction = 0.0

    return WarmupResult(
        warmup_items_count=len(set(r.item_id for r in warmup_results)) if warmup_results else 0,
        warmup_results_discarded=len(warmup_results),
        warmup_pass_rate=warmup_pr,
        post_warmup_pass_rate=post_pr,
        variance_reduction=var_reduction,
    )


def _pass_rate(results: list) -> float:
    if not results:
        return 0.0
    passed = sum(1 for r in results if r.passed is True)
    total = sum(1 for r in results if r.passed is not None)
    return passed / total if total > 0 else 0.0


def _score_variance(results: list) -> float:
    scores = [r.score for r in results if r.score is not None]
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    return sum((s - mean) ** 2 for s in scores) / len(scores)
