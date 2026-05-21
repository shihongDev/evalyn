"""Metric versioning utilities for detecting changes between runs.

Compares metric version hashes across evaluation runs to detect when
metric definitions have changed, which could affect comparison validity.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricVersionChange:
    """A detected change in a metric's version between two runs."""

    metric_id: str
    baseline_hash: str | None
    current_hash: str | None
    change_type: str  # "modified", "added", "removed"


@dataclass
class VersionComparisonReport:
    """Report of metric version differences between two runs."""

    changes: list[MetricVersionChange] = field(default_factory=list)
    unchanged_metrics: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return len(self.changes) > 0

    @property
    def modified_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == "modified")

    @property
    def added_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == "added")

    @property
    def removed_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == "removed")

    def format_warnings(self) -> list[str]:
        """Generate warning messages for detected changes."""
        warnings = []
        for change in self.changes:
            if change.change_type == "modified":
                warnings.append(
                    f"Metric '{change.metric_id}' definition changed between runs "
                    f"(hash: {change.baseline_hash[:8]}.. -> {change.current_hash[:8]}..)"
                )
            elif change.change_type == "added":
                warnings.append(f"Metric '{change.metric_id}' is new (not in baseline run)")
            elif change.change_type == "removed":
                warnings.append(f"Metric '{change.metric_id}' was removed (was in baseline run)")
        return warnings


def compare_metric_versions(
    baseline_metrics: list,
    current_metrics: list,
) -> VersionComparisonReport:
    """Compare metric versions between two runs.

    Args:
        baseline_metrics: MetricSpec list from baseline run.
        current_metrics: MetricSpec list from current run.

    Returns:
        VersionComparisonReport with detected changes.
    """
    baseline_hashes: dict[str, str] = {}
    for m in baseline_metrics:
        h = getattr(m, "version_hash", None)
        if h:
            baseline_hashes[m.id] = h

    current_hashes: dict[str, str] = {}
    for m in current_metrics:
        h = getattr(m, "version_hash", None)
        if h:
            current_hashes[m.id] = h

    all_ids: set[str] = set(baseline_hashes.keys()) | set(current_hashes.keys())

    changes = []
    unchanged = []

    for metric_id in sorted(all_ids):
        bh = baseline_hashes.get(metric_id)
        ch = current_hashes.get(metric_id)

        if bh and ch:
            if bh != ch:
                changes.append(
                    MetricVersionChange(
                        metric_id=metric_id,
                        baseline_hash=bh,
                        current_hash=ch,
                        change_type="modified",
                    )
                )
            else:
                unchanged.append(metric_id)
        elif bh and not ch:
            changes.append(
                MetricVersionChange(
                    metric_id=metric_id,
                    baseline_hash=bh,
                    current_hash=None,
                    change_type="removed",
                )
            )
        elif ch and not bh:
            changes.append(
                MetricVersionChange(
                    metric_id=metric_id,
                    baseline_hash=None,
                    current_hash=ch,
                    change_type="added",
                )
            )

    return VersionComparisonReport(changes=changes, unchanged_metrics=unchanged)
