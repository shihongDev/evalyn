"""
Cohort analysis: compare metrics across user-defined item groups.

Provides functions to group items into cohorts by metadata or input length,
compute per-cohort statistics, and compare cohorts pairwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CohortStats:
    """Statistics for a single cohort."""

    cohort_name: str
    item_count: int = 0
    avg_score: float = 0.0
    pass_rate: float = 0.0
    std_dev: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "cohort_name": self.cohort_name,
            "item_count": self.item_count,
            "avg_score": self.avg_score,
            "pass_rate": self.pass_rate,
            "std_dev": self.std_dev,
        }


@dataclass
class CohortComparison:
    """Pairwise comparison between two cohorts."""

    cohort_a: str
    cohort_b: str
    score_delta: float = 0.0
    pass_rate_delta: float = 0.0
    significant: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "cohort_a": self.cohort_a,
            "cohort_b": self.cohort_b,
            "score_delta": self.score_delta,
            "pass_rate_delta": self.pass_rate_delta,
            "significant": self.significant,
        }

    def format_text(self) -> str:
        """Format as human-readable text."""
        sig = " [SIGNIFICANT]" if self.significant else ""
        return (
            f"{self.cohort_a} vs {self.cohort_b}:"
            f" score delta={self.score_delta:+.3f},"
            f" pass rate delta={self.pass_rate_delta:+.3f}{sig}"
        )


@dataclass
class CohortReport:
    """Full cohort analysis report."""

    cohorts: list[CohortStats] = field(default_factory=list)
    comparisons: list[CohortComparison] = field(default_factory=list)
    best_cohort: str = ""
    worst_cohort: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "cohorts": [c.as_dict() for c in self.cohorts],
            "comparisons": [c.as_dict() for c in self.comparisons],
            "best_cohort": self.best_cohort,
            "worst_cohort": self.worst_cohort,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CohortReport:
        cohorts = [
            CohortStats(**c) for c in data.get("cohorts", [])
        ]
        comparisons_raw = data.get("comparisons", [])
        comparisons = [
            CohortComparison(**c) for c in comparisons_raw
        ]
        return cls(
            cohorts=cohorts,
            comparisons=comparisons,
            best_cohort=data.get("best_cohort", ""),
            worst_cohort=data.get("worst_cohort", ""),
        )

    def format_text(self) -> str:
        """Format as human-readable text."""
        lines: list[str] = []
        lines.append("Cohort Report")
        if self.best_cohort:
            lines.append(f"  Best cohort: {self.best_cohort}")
        if self.worst_cohort:
            lines.append(f"  Worst cohort: {self.worst_cohort}")
        lines.append("  Cohorts:")
        for c in self.cohorts:
            lines.append(
                f"    {c.cohort_name}: n={c.item_count},"
                f" avg={c.avg_score:.3f},"
                f" pass_rate={c.pass_rate:.1%},"
                f" std={c.std_dev:.3f}"
            )
        if self.comparisons:
            lines.append("  Comparisons:")
            for comp in self.comparisons:
                lines.append(f"    {comp.format_text()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def define_cohort_by_metadata(
    items: list[dict[str, Any]], metadata_key: str
) -> dict[str, list[dict[str, Any]]]:
    """Group items by metadata field value.

    Items missing the metadata_key are placed in an "unknown" cohort.
    """
    cohorts: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        metadata = item.get("metadata", {})
        if isinstance(metadata, dict):
            val = metadata.get(metadata_key, "unknown")
        else:
            val = "unknown"
        key = str(val)
        if key not in cohorts:
            cohorts[key] = []
        cohorts[key].append(item)
    return cohorts


def define_cohort_by_length(
    items: list[dict[str, Any]],
    input_field: str = "input",
    num_buckets: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Group by input length buckets (short/medium/long).

    Divides the range of input lengths evenly into num_buckets. Default
    bucket names are 'short', 'medium', 'long' for 3 buckets; otherwise
    'bucket_0', 'bucket_1', etc.
    """
    if not items:
        return {}

    # Compute lengths
    lengths: list[int] = []
    for item in items:
        val = item.get(input_field, "")
        if isinstance(val, dict):
            import json as _json
            lengths.append(len(_json.dumps(val)))
        elif isinstance(val, str):
            lengths.append(len(val))
        else:
            lengths.append(len(str(val)))

    if not lengths:
        return {}

    min_len = min(lengths)
    max_len = max(lengths)
    bucket_range = (max_len - min_len + 1) / num_buckets if max_len > min_len else 1

    # Bucket names
    if num_buckets == 3:
        names = ["short", "medium", "long"]
    else:
        names = [f"bucket_{i}" for i in range(num_buckets)]

    cohorts: dict[str, list[dict[str, Any]]] = {name: [] for name in names}

    for item, length in zip(items, lengths):
        if max_len == min_len:
            bucket_idx = 0
        else:
            bucket_idx = min(int((length - min_len) / bucket_range), num_buckets - 1)
        cohorts[names[bucket_idx]].append(item)

    return cohorts


def compute_cohort_stats(
    cohort_name: str,
    items: list[dict[str, Any]],
    score_field: str = "score",
) -> CohortStats:
    """Compute stats for one cohort.

    Each item should have a score_field (float) and optionally a 'passed' field (bool).
    """
    if not items:
        return CohortStats(cohort_name=cohort_name)

    scores: list[float] = []
    passed_count = 0
    total_with_pass = 0

    for item in items:
        score = item.get(score_field)
        if score is not None:
            scores.append(float(score))
        p = item.get("passed")
        if p is not None:
            total_with_pass += 1
            if p:
                passed_count += 1

    n = len(items)
    avg = sum(scores) / len(scores) if scores else 0.0
    pass_rate = passed_count / total_with_pass if total_with_pass > 0 else 0.0

    if len(scores) >= 2:
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        std = math.sqrt(variance)
    else:
        std = 0.0

    return CohortStats(
        cohort_name=cohort_name,
        item_count=n,
        avg_score=round(avg, 4),
        pass_rate=round(pass_rate, 4),
        std_dev=round(std, 4),
    )


def compare_cohorts(
    stats_a: CohortStats, stats_b: CohortStats
) -> CohortComparison:
    """Pairwise comparison. Significant if |score_delta| > 0.1."""
    score_delta = stats_a.avg_score - stats_b.avg_score
    pass_rate_delta = stats_a.pass_rate - stats_b.pass_rate
    significant = abs(score_delta) > 0.1

    return CohortComparison(
        cohort_a=stats_a.cohort_name,
        cohort_b=stats_b.cohort_name,
        score_delta=round(score_delta, 4),
        pass_rate_delta=round(pass_rate_delta, 4),
        significant=significant,
    )


def build_cohort_report(
    cohorts: dict[str, list[dict[str, Any]]],
    score_field: str = "score",
) -> CohortReport:
    """Full analysis with all pairwise comparisons."""
    stats_list: list[CohortStats] = []
    for name in sorted(cohorts.keys()):
        stats_list.append(compute_cohort_stats(name, cohorts[name], score_field))

    comparisons: list[CohortComparison] = []
    for i, sa in enumerate(stats_list):
        for sb in stats_list[i + 1 :]:
            comparisons.append(compare_cohorts(sa, sb))

    best = ""
    worst = ""
    if stats_list:
        best = max(stats_list, key=lambda s: s.avg_score).cohort_name
        worst = min(stats_list, key=lambda s: s.avg_score).cohort_name

    return CohortReport(
        cohorts=stats_list,
        comparisons=comparisons,
        best_cohort=best,
        worst_cohort=worst,
    )


def render_cohort_chart(report: CohortReport, width: int = 60) -> str:
    """ASCII bar chart of cohort scores.

    Each cohort gets a horizontal bar proportional to its avg_score.
    Assumes scores are in 0-1 range.
    """
    if not report.cohorts:
        return ""

    max_name_len = max(len(c.cohort_name) for c in report.cohorts)
    lines: list[str] = []

    # Determine scale: use max score or 1.0, whichever is larger
    max_score = max(c.avg_score for c in report.cohorts)
    scale = max(max_score, 1.0)

    for c in report.cohorts:
        bar_len = int(c.avg_score / scale * width) if scale > 0 else 0
        bar = "#" * bar_len
        name = c.cohort_name.ljust(max_name_len)
        lines.append(f"{name} |{bar} {c.avg_score:.3f}")

    return "\n".join(lines)
