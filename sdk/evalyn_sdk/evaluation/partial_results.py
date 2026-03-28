"""Evaluation partial result access.

Query in-progress evaluation results from checkpoint data without
waiting for the full run to complete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PartialResults:
    """Partial evaluation results from an in-progress run."""

    run_id: str
    completed_items: int
    total_results: int
    metric_results: List = field(default_factory=list)
    is_complete: bool = False

    @property
    def pass_rate_estimate(self) -> float:
        """Estimate overall pass rate from completed items."""
        if not self.metric_results:
            return 0.0
        # Group by item, count items where all metrics pass
        item_pass: Dict[str, bool] = {}
        for r in self.metric_results:
            iid = r.get("item_id", r.get("metric_id", ""))
            if iid not in item_pass:
                item_pass[iid] = True
            if r.get("passed") is False:
                item_pass[iid] = False
        if not item_pass:
            return 0.0
        return sum(1 for v in item_pass.values() if v) / len(item_pass)

    @property
    def metric_pass_rates(self) -> Dict[str, float]:
        """Per-metric pass rates from partial data."""
        counts: Dict[str, list] = {}
        for r in self.metric_results:
            mid = r.get("metric_id", "unknown")
            if mid not in counts:
                counts[mid] = [0, 0]
            counts[mid][1] += 1
            if r.get("passed") is True:
                counts[mid][0] += 1
        return {
            mid: p / t if t > 0 else 0.0
            for mid, (p, t) in counts.items()
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "completed_items": self.completed_items,
            "total_results": self.total_results,
            "is_complete": self.is_complete,
            "pass_rate_estimate": round(self.pass_rate_estimate, 4),
            "metric_pass_rates": {
                k: round(v, 4) for k, v in self.metric_pass_rates.items()
            },
        }

    def format_text(self) -> str:
        lines = [
            f"Partial Results (run: {self.run_id[:8]})",
            f"  Items completed: {self.completed_items}",
            f"  Results so far:  {self.total_results}",
            f"  Est. pass rate:  {self.pass_rate_estimate:.1%}",
        ]
        if self.metric_pass_rates:
            lines.append("  Per-metric:")
            for mid, rate in sorted(self.metric_pass_rates.items()):
                lines.append(f"    {mid}: {rate:.1%}")
        return "\n".join(lines)


def load_partial_results(checkpoint_path: Path) -> Optional[PartialResults]:
    """Load partial results from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint.json file.

    Returns:
        PartialResults if checkpoint exists and is valid, None otherwise.
    """
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    run_id = data.get("run_id", "unknown")
    completed_items = data.get("completed_items", [])
    results = data.get("results", [])

    return PartialResults(
        run_id=run_id,
        completed_items=len(completed_items) if isinstance(completed_items, list) else int(completed_items),
        total_results=len(results),
        metric_results=results,
        is_complete=False,
    )


def estimate_remaining_time(
    partial: PartialResults,
    total_items: int,
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Estimate remaining evaluation time from partial progress.

    Args:
        partial: Current partial results.
        total_items: Total items in the dataset.
        elapsed_seconds: Time spent so far.

    Returns:
        Dict with ETA and rate estimates.
    """
    if partial.completed_items == 0 or elapsed_seconds <= 0:
        return {
            "items_per_second": 0.0,
            "remaining_items": total_items,
            "eta_seconds": 0.0,
        }

    rate = partial.completed_items / elapsed_seconds
    remaining = max(0, total_items - partial.completed_items)
    eta = remaining / rate if rate > 0 else 0.0

    return {
        "items_per_second": round(rate, 2),
        "remaining_items": remaining,
        "eta_seconds": round(eta, 1),
        "progress_pct": round(partial.completed_items / total_items * 100, 1) if total_items > 0 else 0.0,
    }
