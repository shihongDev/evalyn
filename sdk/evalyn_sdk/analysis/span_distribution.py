"""Span type distribution statistics.

Computes how span types are distributed across a trace, tracks trends
over time, and detects shifts between two snapshots.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class TypeCount:
    """Count and percentage for a single span type."""

    span_type: str
    count: int
    percentage: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_type": self.span_type,
            "count": self.count,
            "percentage": round(self.percentage, 4),
        }


@dataclass
class DistributionReport:
    """Full distribution of span types."""

    counts: list[TypeCount] = field(default_factory=list)
    total_spans: int = 0
    unique_types: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": [c.as_dict() for c in self.counts],
            "total_spans": self.total_spans,
            "unique_types": self.unique_types,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DistributionReport:
        counts = [
            TypeCount(
                span_type=c["span_type"],
                count=c["count"],
                percentage=c["percentage"],
            )
            for c in data.get("counts", [])
        ]
        return cls(
            counts=counts,
            total_spans=data.get("total_spans", 0),
            unique_types=data.get("unique_types", 0),
        )

    def format_text(self) -> str:
        lines = [f"Span Distribution: {self.total_spans} spans, {self.unique_types} types"]
        for tc in self.counts:
            pct = tc.percentage * 100
            lines.append(f"  {tc.span_type}: {tc.count} ({pct:.1f}%)")
        return "\n".join(lines)


def compute_distribution(spans: list[Span]) -> DistributionReport:
    """Count each span_type, compute percentages, sort by count descending."""
    if not spans:
        return DistributionReport()

    counter = Counter(s.span_type for s in spans)
    total = len(spans)
    counts = [
        TypeCount(
            span_type=stype,
            count=cnt,
            percentage=cnt / total,
        )
        for stype, cnt in counter.most_common()
    ]
    return DistributionReport(
        counts=counts,
        total_spans=total,
        unique_types=len(counter),
    )


def compute_distribution_trend(
    span_groups: list[tuple[str, list[Span]]],
) -> dict[str, list[tuple[str, float]]]:
    """Compute per-type percentage over time.

    Args:
        span_groups: List of (period_label, spans) tuples.

    Returns:
        Dict mapping span_type to list of (period_label, percentage) tuples.
    """
    result: dict[str, list[tuple[str, float]]] = {}

    for label, spans in span_groups:
        report = compute_distribution(spans)
        seen_types = set()
        for tc in report.counts:
            result.setdefault(tc.span_type, []).append((label, tc.percentage))
            seen_types.add(tc.span_type)
        # Types seen in other periods but not this one get 0.0
        for stype in result:
            if stype not in seen_types:
                result[stype].append((label, 0.0))

    return result


def detect_distribution_shift(
    before: list[Span],
    after: list[Span],
    threshold: float = 0.1,
) -> list[dict[str, Any]]:
    """Compare two distributions and return shifts exceeding threshold.

    Each shift: {"span_type", "before_pct", "after_pct", "delta"}.
    """
    before_report = compute_distribution(before)
    after_report = compute_distribution(after)

    before_pcts: dict[str, float] = {tc.span_type: tc.percentage for tc in before_report.counts}
    after_pcts: dict[str, float] = {tc.span_type: tc.percentage for tc in after_report.counts}

    all_types = set(before_pcts) | set(after_pcts)
    shifts = []
    for stype in sorted(all_types):
        b = before_pcts.get(stype, 0.0)
        a = after_pcts.get(stype, 0.0)
        delta = a - b
        if abs(delta) > threshold:
            shifts.append({
                "span_type": stype,
                "before_pct": round(b, 4),
                "after_pct": round(a, 4),
                "delta": round(delta, 4),
            })

    return shifts
