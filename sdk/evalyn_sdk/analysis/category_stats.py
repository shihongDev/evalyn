"""Metric category pass rates: aggregate by category (safety, correctness, etc).

Groups metrics by their category from the metric registry and computes
per-category pass rates for quick identification of weak areas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CategoryStats:
    """Statistics for a metric category."""

    category: str
    metric_count: int = 0
    total_evaluations: int = 0
    passed: int = 0
    failed: int = 0

    @property
    def pass_rate(self) -> float:
        total = self.passed + self.failed
        return self.passed / total if total > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "metric_count": self.metric_count,
            "total_evaluations": self.total_evaluations,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 4),
        }


@dataclass
class CategoryReport:
    """Category-level pass rate report."""

    categories: list[CategoryStats] = field(default_factory=list)

    @property
    def weakest_category(self) -> str | None:
        rated = [c for c in self.categories if c.total_evaluations > 0]
        if not rated:
            return None
        return min(rated, key=lambda c: c.pass_rate).category

    @property
    def strongest_category(self) -> str | None:
        rated = [c for c in self.categories if c.total_evaluations > 0]
        if not rated:
            return None
        return max(rated, key=lambda c: c.pass_rate).category

    def as_dict(self) -> dict[str, Any]:
        return {
            "categories": [c.as_dict() for c in self.categories],
            "weakest": self.weakest_category,
            "strongest": self.strongest_category,
        }

    def format_text(self) -> str:
        lines = ["Category Pass Rates:"]
        for c in sorted(self.categories, key=lambda x: x.pass_rate):
            bar = "#" * int(c.pass_rate * 20)
            lines.append(
                f"  {c.category:<20} {bar:<20} {c.pass_rate:.0%} ({c.metric_count} metrics)"
            )
        if self.weakest_category:
            lines.append(f"\n  Weakest: {self.weakest_category}")
        return "\n".join(lines)


def compute_category_pass_rates(
    metric_results: list,
    category_map: dict[str, str] | None = None,
) -> CategoryReport:
    """Compute per-category pass rates from evaluation results.

    Args:
        metric_results: List of MetricResult objects.
        category_map: Dict of metric_id -> category name. If None, uses
            built-in category assignments from the metric registry.

    Returns:
        CategoryReport with per-category statistics.
    """
    if category_map is None:
        category_map = _load_default_categories()

    # Aggregate by category
    cat_stats: dict[str, CategoryStats] = {}
    seen_metrics: dict[str, set] = {}

    for mr in metric_results:
        mid = mr.metric_id
        category = category_map.get(mid, "uncategorized")

        if category not in cat_stats:
            cat_stats[category] = CategoryStats(category=category)
            seen_metrics[category] = set()

        stats = cat_stats[category]
        seen_metrics[category].add(mid)
        stats.total_evaluations += 1

        if mr.passed is True:
            stats.passed += 1
        elif mr.passed is False:
            stats.failed += 1

    # Set metric counts
    for category, metrics in seen_metrics.items():
        cat_stats[category].metric_count = len(metrics)

    return CategoryReport(
        categories=sorted(cat_stats.values(), key=lambda c: c.pass_rate),
    )


def _load_default_categories() -> dict[str, str]:
    """Load default category assignments from metric registries."""
    categories = {}

    try:
        from ..metrics.objective import OBJECTIVE_REGISTRY

        for m in OBJECTIVE_REGISTRY:
            categories[m["id"]] = m.get("category", "uncategorized")
    except ImportError:
        pass

    try:
        from ..metrics.subjective import SUBJECTIVE_REGISTRY

        for m in SUBJECTIVE_REGISTRY:
            categories[m["id"]] = m.get("category", "uncategorized")
    except ImportError:
        pass

    return categories
