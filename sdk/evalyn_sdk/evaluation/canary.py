"""Evaluation canary: run a small subset first to check viability.

Before committing to a full evaluation run, evaluate a small random
sample. If the pass rate is below a threshold, abort to save tokens.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CanaryResult:
    """Result from a canary evaluation check."""

    sample_size: int
    pass_rate: float
    passed_count: int
    failed_count: int
    abort_threshold: float
    should_abort: bool
    estimated_cost_saved: float = 0.0

    @property
    def status(self) -> str:
        return "ABORT" if self.should_abort else "PROCEED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sample_size": self.sample_size,
            "pass_rate": round(self.pass_rate, 4),
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "abort_threshold": self.abort_threshold,
            "should_abort": self.should_abort,
            "estimated_cost_saved": round(self.estimated_cost_saved, 4),
        }

    def format_text(self) -> str:
        lines = [
            f"Canary Check: {self.status}",
            f"  Sample: {self.sample_size} items",
            f"  Pass rate: {self.pass_rate:.1%} ({self.passed_count}/{self.sample_size})",
            f"  Threshold: {self.abort_threshold:.1%}",
        ]
        if self.should_abort:
            lines.append("  Aborting: pass rate below threshold")
            if self.estimated_cost_saved > 0:
                lines.append(f"  Estimated savings: ${self.estimated_cost_saved:.4f}")
        else:
            lines.append("  Proceeding with full evaluation")
        return "\n".join(lines)


def select_canary_sample(
    items: list,
    sample_size: int = 10,
    seed: int = 42,
) -> list:
    """Select a random sample for canary evaluation.

    Args:
        items: Full dataset items.
        sample_size: Number of items to sample (default 10).
        seed: Random seed for reproducibility.

    Returns:
        Sampled items (or all items if dataset is smaller than sample_size).
    """
    if len(items) <= sample_size:
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, sample_size)


def evaluate_canary(
    results: list,
    abort_threshold: float = 0.2,
    total_items: int = 0,
    estimated_cost_per_item: float = 0.0,
) -> CanaryResult:
    """Evaluate canary results and decide whether to proceed.

    Args:
        results: MetricResult list from canary evaluation.
        abort_threshold: Abort if pass rate is below this (default 0.2 = 20%).
        total_items: Total items in full dataset (for cost savings estimate).
        estimated_cost_per_item: Estimated cost per item (for savings estimate).

    Returns:
        CanaryResult with proceed/abort recommendation.
    """
    if not results:
        return CanaryResult(
            sample_size=0, pass_rate=0.0, passed_count=0, failed_count=0,
            abort_threshold=abort_threshold, should_abort=True,
        )

    # Count unique items and their pass/fail status
    item_results: dict[str, bool] = {}
    for r in results:
        if r.item_id not in item_results:
            item_results[r.item_id] = True  # assume pass until proven otherwise
        if r.passed is False:
            item_results[r.item_id] = False

    passed = sum(1 for v in item_results.values() if v)
    failed = len(item_results) - passed
    total = len(item_results)
    pass_rate = passed / total if total > 0 else 0.0

    should_abort = pass_rate < abort_threshold

    # Estimate savings if aborting
    remaining_items = max(0, total_items - total)
    cost_saved = remaining_items * estimated_cost_per_item if should_abort else 0.0

    return CanaryResult(
        sample_size=total,
        pass_rate=pass_rate,
        passed_count=passed,
        failed_count=failed,
        abort_threshold=abort_threshold,
        should_abort=should_abort,
        estimated_cost_saved=cost_saved,
    )
