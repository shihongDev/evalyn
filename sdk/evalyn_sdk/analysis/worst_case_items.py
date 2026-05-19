"""
Worst-case item identification.

Surfaces items that fail across the most metrics simultaneously,
computes failure correlations, and renders ASCII tables.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class WorstCaseItem:
    """An item with its cross-metric failure profile."""

    item_id: str
    failed_metrics: list[str] = field(default_factory=list)
    total_metrics: int = 0
    failure_rate: float = 0.0
    avg_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "failed_metrics": list(self.failed_metrics),
            "total_metrics": self.total_metrics,
            "failure_rate": self.failure_rate,
            "avg_score": self.avg_score,
        }


@dataclass
class WorstCaseReport:
    """Aggregate report of worst-case items."""

    items: list[WorstCaseItem] = field(default_factory=list)
    total_items: int = 0
    worst_item_id: str = ""
    max_failures: int = 0
    avg_failure_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": [i.as_dict() for i in self.items],
            "total_items": self.total_items,
            "worst_item_id": self.worst_item_id,
            "max_failures": self.max_failures,
            "avg_failure_rate": self.avg_failure_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorstCaseReport:
        items = []
        for item_data in data.get("items", []):
            items.append(WorstCaseItem(
                item_id=item_data["item_id"],
                failed_metrics=item_data.get("failed_metrics", []),
                total_metrics=item_data.get("total_metrics", 0),
                failure_rate=item_data.get("failure_rate", 0.0),
                avg_score=item_data.get("avg_score", 0.0),
            ))
        return cls(
            items=items,
            total_items=data.get("total_items", 0),
            worst_item_id=data.get("worst_item_id", ""),
            max_failures=data.get("max_failures", 0),
            avg_failure_rate=data.get("avg_failure_rate", 0.0),
        )

    def format_text(self) -> str:
        """Human-readable multi-line summary."""
        lines: list[str] = []
        lines.append(f"Worst-Case Items Report ({self.total_items} items)")
        lines.append(f"  Worst item: {self.worst_item_id} ({self.max_failures} failures)")
        lines.append(f"  Avg failure rate: {self.avg_failure_rate:.1%}")
        if self.items:
            lines.append("")
            lines.append("  Top items:")
            for item in self.items[:5]:
                lines.append(
                    f"    {item.item_id}: {len(item.failed_metrics)}/{item.total_metrics}"
                    f" failed ({item.failure_rate:.1%}), avg score {item.avg_score:.2f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def analyze_item_failures(item_results: list[dict[str, Any]]) -> list[WorstCaseItem]:
    """For each item, count how many metrics failed.

    Args:
        item_results: List of dicts with keys "item_id", "metric_id", "passed", "score".

    Returns:
        List of WorstCaseItem, one per unique item_id.
    """
    # Collect per-item data
    item_metrics: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in item_results:
        item_id = r["item_id"]
        metric_id = r["metric_id"]
        item_metrics[item_id][metric_id] = {
            "passed": r.get("passed", True),
            "score": r.get("score", 0.0),
        }

    items: list[WorstCaseItem] = []
    for item_id, metrics in item_metrics.items():
        failed = [mid for mid, m in metrics.items() if not m["passed"]]
        scores = [m["score"] for m in metrics.values() if m["score"] is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        total = len(metrics)
        failure_rate = len(failed) / total if total > 0 else 0.0

        items.append(WorstCaseItem(
            item_id=item_id,
            failed_metrics=failed,
            total_metrics=total,
            failure_rate=failure_rate,
            avg_score=avg_score,
        ))

    return items


def build_worst_case_report(items: list[WorstCaseItem]) -> WorstCaseReport:
    """Aggregate WorstCaseItems into a report."""
    if not items:
        return WorstCaseReport()

    worst = max(items, key=lambda i: len(i.failed_metrics))
    avg_rate = sum(i.failure_rate for i in items) / len(items)

    return WorstCaseReport(
        items=items,
        total_items=len(items),
        worst_item_id=worst.item_id,
        max_failures=len(worst.failed_metrics),
        avg_failure_rate=avg_rate,
    )


def rank_worst_items(
    items: list[WorstCaseItem], top_n: int = 10
) -> list[WorstCaseItem]:
    """Return the top N items sorted by failure_rate descending."""
    ranked = sorted(items, key=lambda i: i.failure_rate, reverse=True)
    return ranked[:top_n]


def find_universally_failing(items: list[WorstCaseItem]) -> list[str]:
    """Return item IDs that fail ALL their metrics."""
    return [
        i.item_id
        for i in items
        if i.total_metrics > 0 and len(i.failed_metrics) == i.total_metrics
    ]


def compute_failure_correlation(
    items: list[WorstCaseItem],
) -> dict[str, int]:
    """Count how often each metric pair co-fails across items.

    Returns:
        Dict mapping "metric_a,metric_b" (sorted) to co-failure count.
    """
    pair_counts: dict[str, int] = defaultdict(int)
    for item in items:
        if len(item.failed_metrics) < 2:
            continue
        for a, b in combinations(sorted(item.failed_metrics), 2):
            key = f"{a},{b}"
            pair_counts[key] += 1
    return dict(pair_counts)


def render_worst_case_table(
    items: list[WorstCaseItem], max_items: int = 10
) -> str:
    """Render an ASCII table of worst-case items.

    Columns: Item ID, Failed, Total, Failure Rate, Avg Score
    """
    ranked = rank_worst_items(items, top_n=max_items)
    if not ranked:
        return "No items to display."

    # Determine column widths
    id_width = max(len("Item ID"), max(len(i.item_id) for i in ranked))
    header = (
        f"{'Item ID':<{id_width}}  {'Failed':>6}  {'Total':>5}"
        f"  {'Fail %':>6}  {'Avg Score':>9}"
    )
    sep = "-" * len(header)

    lines = [header, sep]
    for item in ranked:
        lines.append(
            f"{item.item_id:<{id_width}}  {len(item.failed_metrics):>6}"
            f"  {item.total_metrics:>5}  {item.failure_rate:>6.1%}"
            f"  {item.avg_score:>9.2f}"
        )

    return "\n".join(lines)
