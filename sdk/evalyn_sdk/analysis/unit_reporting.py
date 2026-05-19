"""
Unit-level reporting: per-unit-type metric breakdowns in analysis.

Groups evaluation items by unit type (outcome, single_turn, tool_use, etc.)
and computes per-type statistics including pass rates and metric breakdowns.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UnitTypeStats:
    """Statistics for a single unit type across items."""

    unit_type: str
    item_count: int
    pass_rate: float
    avg_score: float
    metric_breakdown: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_type": self.unit_type,
            "item_count": self.item_count,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "metric_breakdown": dict(self.metric_breakdown),
        }


@dataclass
class UnitReport:
    """Complete unit-level report across all unit types."""

    unit_stats: list[UnitTypeStats] = field(default_factory=list)
    total_items: int = 0
    unit_type_distribution: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_stats": [s.as_dict() for s in self.unit_stats],
            "total_items": self.total_items,
            "unit_type_distribution": dict(self.unit_type_distribution),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnitReport:
        stats = [
            UnitTypeStats(
                unit_type=s["unit_type"],
                item_count=s["item_count"],
                pass_rate=s["pass_rate"],
                avg_score=s["avg_score"],
                metric_breakdown=s.get("metric_breakdown", {}),
            )
            for s in data.get("unit_stats", [])
        ]
        return cls(
            unit_stats=stats,
            total_items=data.get("total_items", 0),
            unit_type_distribution=data.get("unit_type_distribution", {}),
        )

    def format_text(self) -> str:
        """Format the report as human-readable text."""
        lines: list[str] = []
        lines.append(f"Unit Report - {self.total_items} total items")
        lines.append("")

        if not self.unit_stats:
            lines.append("No unit data available.")
            return "\n".join(lines)

        lines.append("Distribution:")
        for ut, count in sorted(self.unit_type_distribution.items()):
            lines.append(f"  {ut}: {count} items")
        lines.append("")

        for stats in self.unit_stats:
            lines.append(f"[{stats.unit_type}] {stats.item_count} items")
            lines.append(f"  pass rate: {stats.pass_rate:.1%}")
            lines.append(f"  avg score: {stats.avg_score:.3f}")
            if stats.metric_breakdown:
                for metric_id, rate in sorted(stats.metric_breakdown.items()):
                    lines.append(f"  {metric_id}: {rate:.1%}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def classify_item_unit_type(item_metrics: dict[str, Any]) -> str:
    """Infer unit type from an item's metric results.

    Looks for a "unit_type" key in the metrics dict. Falls back to "outcome".
    """
    if "unit_type" in item_metrics:
        return str(item_metrics["unit_type"])
    return "outcome"


def compute_unit_breakdown(items: list[dict[str, Any]]) -> UnitReport:
    """Group items by unit type and compute per-type stats.

    Each item dict has:
      - "item_id": str
      - "unit_type": str
      - "metrics": Dict[str, {"score": float, "passed": bool}]
    """
    if not items:
        return UnitReport(unit_stats=[], total_items=0, unit_type_distribution={})

    # Group items by unit type
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        ut = item.get("unit_type", "outcome")
        groups[ut].append(item)

    distribution = {ut: len(group) for ut, group in groups.items()}
    stats_list: list[UnitTypeStats] = []

    for ut, group in sorted(groups.items()):
        total = len(group)
        passed_count = 0
        score_sum = 0.0
        score_count = 0
        metric_passed: dict[str, int] = defaultdict(int)
        metric_total: dict[str, int] = defaultdict(int)

        for item in group:
            metrics = item.get("metrics", {})
            item_passed = True
            for metric_id, result in metrics.items():
                if isinstance(result, dict):
                    metric_total[metric_id] += 1
                    if result.get("passed", False):
                        metric_passed[metric_id] += 1
                    else:
                        item_passed = False
                    if "score" in result:
                        score_sum += result["score"]
                        score_count += 1

            if item_passed and metrics:
                passed_count += 1

        pass_rate = passed_count / total if total > 0 else 0.0
        avg_score = score_sum / score_count if score_count > 0 else 0.0
        metric_breakdown = {
            mid: metric_passed[mid] / metric_total[mid]
            for mid in sorted(metric_total)
            if metric_total[mid] > 0
        }

        stats_list.append(
            UnitTypeStats(
                unit_type=ut,
                item_count=total,
                pass_rate=pass_rate,
                avg_score=avg_score,
                metric_breakdown=metric_breakdown,
            )
        )

    return UnitReport(
        unit_stats=stats_list,
        total_items=len(items),
        unit_type_distribution=distribution,
    )


def filter_by_unit_type(
    items: list[dict[str, Any]], unit_type: str
) -> list[dict[str, Any]]:
    """Filter items to a single unit type."""
    return [item for item in items if item.get("unit_type", "outcome") == unit_type]
