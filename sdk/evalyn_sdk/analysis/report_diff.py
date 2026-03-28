"""
Analysis report diff.

Diffs two RunAnalysis-style report dicts, showing what changed
between runs: improved, degraded, added, or removed metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class MetricChange:
    """Change in a single metric between two reports."""

    metric_id: str
    old_pass_rate: Optional[float] = None
    new_pass_rate: Optional[float] = None
    delta: float = 0.0
    status: str = ""  # improved / degraded / unchanged / added / removed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "old_pass_rate": self.old_pass_rate,
            "new_pass_rate": self.new_pass_rate,
            "delta": self.delta,
            "status": self.status,
        }


@dataclass
class ReportDiff:
    """Diff between two analysis reports."""

    run_a_id: str
    run_b_id: str
    metric_changes: List[MetricChange] = field(default_factory=list)
    pass_rate_delta: float = 0.0
    item_count_delta: int = 0
    improved_count: int = 0
    degraded_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_a_id": self.run_a_id,
            "run_b_id": self.run_b_id,
            "metric_changes": [c.as_dict() for c in self.metric_changes],
            "pass_rate_delta": self.pass_rate_delta,
            "item_count_delta": self.item_count_delta,
            "improved_count": self.improved_count,
            "degraded_count": self.degraded_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportDiff":
        changes = []
        for c in data.get("metric_changes", []):
            changes.append(MetricChange(
                metric_id=c["metric_id"],
                old_pass_rate=c.get("old_pass_rate"),
                new_pass_rate=c.get("new_pass_rate"),
                delta=c.get("delta", 0.0),
                status=c.get("status", ""),
            ))
        return cls(
            run_a_id=data.get("run_a_id", ""),
            run_b_id=data.get("run_b_id", ""),
            metric_changes=changes,
            pass_rate_delta=data.get("pass_rate_delta", 0.0),
            item_count_delta=data.get("item_count_delta", 0),
            improved_count=data.get("improved_count", 0),
            degraded_count=data.get("degraded_count", 0),
        )

    def format_text(self) -> str:
        """Human-readable multi-line summary."""
        lines: List[str] = []
        lines.append(f"Report Diff: {self.run_a_id} -> {self.run_b_id}")
        lines.append(f"  Pass rate delta: {self.pass_rate_delta:+.1%}")
        lines.append(f"  Item count delta: {self.item_count_delta:+d}")
        lines.append(
            f"  Improved: {self.improved_count}, Degraded: {self.degraded_count}"
        )
        if self.metric_changes:
            lines.append("")
            lines.append("  Metric changes:")
            for c in self.metric_changes:
                old_str = f"{c.old_pass_rate:.1%}" if c.old_pass_rate is not None else "n/a"
                new_str = f"{c.new_pass_rate:.1%}" if c.new_pass_rate is not None else "n/a"
                lines.append(
                    f"    {c.metric_id}: {old_str} -> {new_str}"
                    f" ({c.delta:+.1%}) [{c.status}]"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_metric_change(
    metric_id: str,
    old_rate: Optional[float],
    new_rate: Optional[float],
    threshold: float = 0.01,
) -> MetricChange:
    """Classify a single metric's change between two reports.

    Args:
        metric_id: Metric identifier.
        old_rate: Pass rate in report A (None if metric absent).
        new_rate: Pass rate in report B (None if metric absent).
        threshold: Minimum absolute delta to count as improved/degraded.

    Returns:
        MetricChange with status: improved, degraded, unchanged, added, or removed.
    """
    if old_rate is None and new_rate is None:
        return MetricChange(metric_id=metric_id, status="unchanged")

    if old_rate is None:
        return MetricChange(
            metric_id=metric_id,
            new_pass_rate=new_rate,
            delta=0.0,
            status="added",
        )

    if new_rate is None:
        return MetricChange(
            metric_id=metric_id,
            old_pass_rate=old_rate,
            delta=0.0,
            status="removed",
        )

    delta = new_rate - old_rate
    if delta > threshold:
        status = "improved"
    elif delta < -threshold:
        status = "degraded"
    else:
        status = "unchanged"

    return MetricChange(
        metric_id=metric_id,
        old_pass_rate=old_rate,
        new_pass_rate=new_rate,
        delta=delta,
        status=status,
    )


def diff_reports(
    report_a: Dict[str, Any],
    report_b: Dict[str, Any],
) -> ReportDiff:
    """Compare two analysis report dicts.

    Each report has: "run_id", "pass_rate", "total_items",
    "metric_stats": {metric_id: {"pass_rate": float}}.

    Returns:
        ReportDiff summarizing all changes.
    """
    run_a_id = report_a.get("run_id", "")
    run_b_id = report_b.get("run_id", "")

    stats_a = report_a.get("metric_stats", {})
    stats_b = report_b.get("metric_stats", {})
    all_metrics = sorted(set(stats_a.keys()) | set(stats_b.keys()))

    changes: List[MetricChange] = []
    improved = 0
    degraded = 0

    for mid in all_metrics:
        old_rate = stats_a[mid]["pass_rate"] if mid in stats_a else None
        new_rate = stats_b[mid]["pass_rate"] if mid in stats_b else None
        change = compute_metric_change(mid, old_rate, new_rate)
        changes.append(change)
        if change.status == "improved":
            improved += 1
        elif change.status == "degraded":
            degraded += 1

    pass_a = report_a.get("pass_rate", 0.0)
    pass_b = report_b.get("pass_rate", 0.0)
    items_a = report_a.get("total_items", 0)
    items_b = report_b.get("total_items", 0)

    return ReportDiff(
        run_a_id=run_a_id,
        run_b_id=run_b_id,
        metric_changes=changes,
        pass_rate_delta=pass_b - pass_a,
        item_count_delta=items_b - items_a,
        improved_count=improved,
        degraded_count=degraded,
    )


def filter_changes(diff: ReportDiff, status: str) -> List[MetricChange]:
    """Filter metric changes by status."""
    return [c for c in diff.metric_changes if c.status == status]


def render_diff_table(diff: ReportDiff) -> str:
    """Render an ASCII table of metric changes with +/- symbols."""
    if not diff.metric_changes:
        return "No metric changes."

    id_width = max(
        len("Metric"),
        max(len(c.metric_id) for c in diff.metric_changes),
    )
    header = (
        f"{'Metric':<{id_width}}  {'Old':>6}  {'New':>6}"
        f"  {'Delta':>8}  {'Status':>10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for c in diff.metric_changes:
        old_str = f"{c.old_pass_rate:.1%}" if c.old_pass_rate is not None else "  n/a"
        new_str = f"{c.new_pass_rate:.1%}" if c.new_pass_rate is not None else "  n/a"
        if c.delta > 0:
            delta_str = f"+{c.delta:.1%}"
        elif c.delta < 0:
            delta_str = f"{c.delta:.1%}"
        else:
            delta_str = " 0.0%"
        lines.append(
            f"{c.metric_id:<{id_width}}  {old_str:>6}  {new_str:>6}"
            f"  {delta_str:>8}  {c.status:>10}"
        )

    return "\n".join(lines)


def summarize_diff(diff: ReportDiff) -> str:
    """One-line summary of a report diff."""
    unchanged = sum(1 for c in diff.metric_changes if c.status == "unchanged")
    added = sum(1 for c in diff.metric_changes if c.status == "added")
    removed = sum(1 for c in diff.metric_changes if c.status == "removed")

    parts: List[str] = []
    if diff.improved_count:
        parts.append(f"{diff.improved_count} improved")
    if diff.degraded_count:
        parts.append(f"{diff.degraded_count} degraded")
    if unchanged:
        parts.append(f"{unchanged} unchanged")
    if added:
        parts.append(f"{added} added")
    if removed:
        parts.append(f"{removed} removed")

    return ", ".join(parts) if parts else "no changes"


def is_regression(diff: ReportDiff, threshold: float = 0.05) -> bool:
    """True if the overall pass rate dropped more than threshold."""
    return diff.pass_rate_delta < -threshold
