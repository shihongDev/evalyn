"""Dataset sampling preview: show sample items and summary stats before building full dataset."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PreviewStats:
    """Summary statistics for a dataset."""

    total_items: int = 0
    avg_input_length: float = 0.0
    avg_output_length: float = 0.0
    min_input_length: int = 0
    max_input_length: int = 0
    categories: dict[str, int] = field(default_factory=dict)
    has_outputs: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_items": self.total_items,
            "avg_input_length": self.avg_input_length,
            "avg_output_length": self.avg_output_length,
            "min_input_length": self.min_input_length,
            "max_input_length": self.max_input_length,
            "categories": dict(self.categories),
            "has_outputs": self.has_outputs,
        }

    def format_text(self) -> str:
        lines = [
            f"Total items: {self.total_items}",
            f"Avg input length: {self.avg_input_length:.1f}",
            f"Avg output length: {self.avg_output_length:.1f}",
            f"Input length range: {self.min_input_length} - {self.max_input_length}",
            f"Has outputs: {self.has_outputs}",
        ]
        if self.categories:
            lines.append("Categories:")
            for cat, count in sorted(self.categories.items()):
                lines.append(f"  {cat}: {count}")
        return "\n".join(lines)


@dataclass
class DatasetPreview:
    """Preview of a dataset with sample items and stats."""

    sample_items: list[dict[str, Any]] = field(default_factory=list)
    stats: PreviewStats | None = None
    total_count: int = 0
    sample_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_items": list(self.sample_items),
            "stats": self.stats.as_dict() if self.stats else None,
            "total_count": self.total_count,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetPreview:
        stats_raw = data.get("stats")
        stats = PreviewStats(**stats_raw) if stats_raw else None
        return cls(
            sample_items=data.get("sample_items", []),
            stats=stats,
            total_count=data.get("total_count", 0),
            sample_count=data.get("sample_count", 0),
        )

    def format_text(self) -> str:
        lines = [
            f"Dataset Preview ({self.sample_count} of {self.total_count} items)",
        ]
        if self.stats:
            lines.append("")
            lines.append(self.stats.format_text())
        if self.sample_items:
            lines.append("")
            lines.append("Sample items:")
            for i, item in enumerate(self.sample_items, 1):
                inp = item.get("input", "")
                lines.append(f"  {i}. {inp[:60]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_preview_stats(items: list[dict[str, Any]]) -> PreviewStats:
    """Compute summary statistics from a list of dataset items."""
    if not items:
        return PreviewStats()

    input_lengths: list[int] = []
    output_lengths: list[int] = []
    categories: dict[str, int] = {}
    has_outputs = False

    for item in items:
        inp = item.get("input", "")
        input_lengths.append(len(inp))

        out = item.get("output", "")
        if out:
            has_outputs = True
        output_lengths.append(len(out))

        cat = item.get("category", "")
        if cat:
            categories[cat] = categories.get(cat, 0) + 1

    total = len(items)
    avg_in = sum(input_lengths) / total
    avg_out = sum(output_lengths) / total

    return PreviewStats(
        total_items=total,
        avg_input_length=avg_in,
        avg_output_length=avg_out,
        min_input_length=min(input_lengths),
        max_input_length=max(input_lengths),
        categories=categories,
        has_outputs=has_outputs,
    )


def sample_items(
    items: list[dict[str, Any]], n: int = 5, seed: int | None = None
) -> list[dict[str, Any]]:
    """Return a random sample of n items. If fewer than n items, return all."""
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)


def build_preview(
    items: list[dict[str, Any]],
    sample_size: int = 5,
    seed: int | None = None,
) -> DatasetPreview:
    """Build a full preview with stats and a random sample."""
    stats = compute_preview_stats(items)
    sampled = sample_items(items, n=sample_size, seed=seed)
    return DatasetPreview(
        sample_items=sampled,
        stats=stats,
        total_count=len(items),
        sample_count=len(sampled),
    )


def render_preview_table(preview: DatasetPreview, max_text_length: int = 80) -> str:
    """Render an ASCII table of the sample items with truncated text."""
    if not preview.sample_items:
        return "(no sample items)"

    def _trunc(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    col_limit = max(10, max_text_length // 2)

    header = f"{'#':<4} {'Input':<{col_limit}} {'Output':<{col_limit}}"
    sep = "-" * len(header)
    lines = [header, sep]

    for i, item in enumerate(preview.sample_items, 1):
        inp = _trunc(item.get("input", ""), col_limit)
        out = _trunc(item.get("output", ""), col_limit)
        lines.append(f"{i:<4} {inp:<{col_limit}} {out:<{col_limit}}")

    return "\n".join(lines)


def detect_data_quality_issues(stats: PreviewStats) -> list[str]:
    """Flag potential data quality issues based on preview stats."""
    issues: list[str] = []

    if not stats.has_outputs:
        issues.append("No outputs found")

    if stats.total_items > 0 and stats.avg_input_length < 10:
        issues.append("Very short inputs (avg < 10 chars)")

    if stats.total_items > 0 and stats.max_input_length == 0:
        issues.append("All inputs are empty")

    if stats.total_items > 0 and stats.avg_output_length < 1 and stats.has_outputs:
        issues.append("Very short outputs (avg < 1 char)")

    return issues
