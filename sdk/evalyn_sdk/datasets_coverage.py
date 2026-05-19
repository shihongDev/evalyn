"""
Dataset annotation coverage map: visualize which items have annotations
and which need them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CoverageItem:
    """Coverage status for a single dataset item."""

    item_id: str
    has_annotation: bool = False
    annotator_count: int = 0
    metrics_evaluated: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "has_annotation": self.has_annotation,
            "annotator_count": self.annotator_count,
            "metrics_evaluated": list(self.metrics_evaluated),
        }


@dataclass
class CoverageStats:
    """Aggregate statistics about annotation coverage."""

    total_items: int = 0
    annotated_items: int = 0
    unannotated_items: int = 0
    coverage_rate: float = 0.0
    avg_annotators: float = 0.0
    metrics_coverage: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "annotated_items": self.annotated_items,
            "unannotated_items": self.unannotated_items,
            "coverage_rate": self.coverage_rate,
            "avg_annotators": self.avg_annotators,
            "metrics_coverage": dict(self.metrics_coverage),
        }

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("Coverage Stats")
        lines.append("-" * 40)
        lines.append(f"  Total items: {self.total_items}")
        lines.append(f"  Annotated: {self.annotated_items}")
        lines.append(f"  Unannotated: {self.unannotated_items}")
        lines.append(f"  Coverage rate: {self.coverage_rate:.1%}")
        lines.append(f"  Avg annotators per item: {self.avg_annotators:.2f}")
        if self.metrics_coverage:
            lines.append("  Metrics coverage:")
            for metric, rate in sorted(self.metrics_coverage.items()):
                lines.append(f"    {metric}: {rate:.1%}")
        return "\n".join(lines)


@dataclass
class CoverageReport:
    """Full coverage report with per-item detail, stats, and gaps."""

    items: List[CoverageItem] = field(default_factory=list)
    stats: Optional[CoverageStats] = None
    gaps: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.as_dict() for item in self.items],
            "stats": self.stats.as_dict() if self.stats else None,
            "gaps": list(self.gaps),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CoverageReport:
        items = [CoverageItem(**i) for i in data.get("items", [])]
        stats_data = data.get("stats")
        stats = CoverageStats(**stats_data) if stats_data else None
        return cls(
            items=items,
            stats=stats,
            gaps=data.get("gaps", []),
        )

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("Coverage Report")
        lines.append("=" * 40)
        if self.stats:
            lines.append(self.stats.format_text())
        if self.gaps:
            lines.append(f"\nGaps ({len(self.gaps)} items without annotations):")
            for gap_id in self.gaps[:20]:
                lines.append(f"  - {gap_id}")
            if len(self.gaps) > 20:
                lines.append(f"  ... and {len(self.gaps) - 20} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_coverage(
    items: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
) -> CoverageReport:
    """
    Compute coverage report.

    Items have "id". Annotations have "item_id", "annotator_id",
    optionally "metric_id".
    """
    # Index annotations by item_id
    annot_by_item: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for annot in annotations:
        item_id = annot.get("item_id", "")
        annot_by_item[item_id].append(annot)

    coverage_items: List[CoverageItem] = []
    for item in items:
        item_id = str(item.get("id", ""))
        item_annots = annot_by_item.get(item_id, [])
        annotators = {a.get("annotator_id") for a in item_annots}
        metrics = sorted({a["metric_id"] for a in item_annots if "metric_id" in a})
        coverage_items.append(
            CoverageItem(
                item_id=item_id,
                has_annotation=len(item_annots) > 0,
                annotator_count=len(annotators),
                metrics_evaluated=metrics,
            )
        )

    # Compute stats
    total = len(coverage_items)
    annotated = sum(1 for ci in coverage_items if ci.has_annotation)
    unannotated = total - annotated
    coverage_rate = annotated / total if total > 0 else 0.0
    total_annotators = sum(ci.annotator_count for ci in coverage_items)
    avg_annotators = total_annotators / total if total > 0 else 0.0

    # Metrics coverage
    all_metrics: set[str] = set()
    for ci in coverage_items:
        all_metrics.update(ci.metrics_evaluated)
    metrics_cov: Dict[str, float] = {}
    for metric in sorted(all_metrics):
        count = sum(1 for ci in coverage_items if metric in ci.metrics_evaluated)
        metrics_cov[metric] = count / total if total > 0 else 0.0

    stats = CoverageStats(
        total_items=total,
        annotated_items=annotated,
        unannotated_items=unannotated,
        coverage_rate=coverage_rate,
        avg_annotators=avg_annotators,
        metrics_coverage=metrics_cov,
    )

    gaps = find_coverage_gaps_from_items(coverage_items)

    return CoverageReport(items=coverage_items, stats=stats, gaps=gaps)


def find_coverage_gaps_from_items(coverage_items: List[CoverageItem]) -> List[str]:
    """Return item IDs with no annotations (internal helper)."""
    return [ci.item_id for ci in coverage_items if not ci.has_annotation]


def find_coverage_gaps(report: CoverageReport) -> List[str]:
    """Return item IDs with no annotations."""
    return [ci.item_id for ci in report.items if not ci.has_annotation]


def compute_metric_coverage(
    annotations: List[Dict[str, Any]],
    all_metrics: List[str],
) -> Dict[str, float]:
    """Per-metric coverage rate: fraction of unique item_ids covered by each metric."""
    if not all_metrics:
        return {}
    all_item_ids: set[str] = {a.get("item_id", "") for a in annotations}
    total_items = len(all_item_ids) if all_item_ids else 1
    result: Dict[str, float] = {}
    for metric in all_metrics:
        covered = {
            a.get("item_id", "")
            for a in annotations
            if a.get("metric_id") == metric
        }
        result[metric] = len(covered) / total_items
    return result


def prioritize_for_annotation(
    report: CoverageReport,
    max_items: int = 10,
) -> List[str]:
    """Return item IDs most needing annotation (0 annotators first, then lowest count)."""
    sorted_items = sorted(report.items, key=lambda ci: ci.annotator_count)
    return [ci.item_id for ci in sorted_items[:max_items]]


def render_coverage_ascii(report: CoverageReport, width: int = 60) -> str:
    """ASCII visualization of coverage."""
    if not report.stats or report.stats.total_items == 0:
        return "[no items]"
    rate = report.stats.coverage_rate
    filled = round(rate * width)
    empty = width - filled
    bar = "#" * filled + "." * empty
    pct = f"{rate:.1%}"
    lines: List[str] = []
    lines.append(f"Coverage: [{bar}] {pct}")
    lines.append(
        f"  {report.stats.annotated_items}/{report.stats.total_items} items annotated"
    )
    if report.stats.metrics_coverage:
        lines.append("  Per-metric:")
        for metric, mrate in sorted(report.stats.metrics_coverage.items()):
            mfilled = round(mrate * 20)
            mempty = 20 - mfilled
            mbar = "#" * mfilled + "." * mempty
            lines.append(f"    {metric}: [{mbar}] {mrate:.1%}")
    return "\n".join(lines)


def suggest_annotation_targets(report: CoverageReport) -> List[str]:
    """Suggestions like '15 items need first annotation'."""
    suggestions: List[str] = []
    if not report.stats:
        return suggestions

    zero_annot = sum(1 for ci in report.items if ci.annotator_count == 0)
    if zero_annot > 0:
        suggestions.append(f"{zero_annot} items need first annotation")

    single_annot = sum(1 for ci in report.items if ci.annotator_count == 1)
    if single_annot > 0:
        suggestions.append(
            f"{single_annot} items have only one annotator - consider adding a second"
        )

    if report.stats.coverage_rate < 1.0 and report.stats.coverage_rate > 0:
        remaining = report.stats.unannotated_items
        suggestions.append(
            f"Annotate {remaining} more items to reach full coverage"
        )

    if report.stats.metrics_coverage:
        for metric, rate in sorted(report.stats.metrics_coverage.items()):
            if rate < 0.5:
                suggestions.append(
                    f"Metric '{metric}' covers only {rate:.0%} of items"
                )

    return suggestions
