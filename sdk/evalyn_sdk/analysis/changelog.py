"""Evaluation result changelog: auto-generated diff summary between runs.

Produces a human-readable summary of what changed between two evaluation
runs, highlighting regressions, improvements, and new/removed metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ChangelogEntry:
    """A single change between two runs."""

    metric_id: str
    change_type: str  # "improved", "regressed", "added", "removed", "unchanged"
    baseline_rate: float | None = None
    current_rate: float | None = None
    delta: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "change_type": self.change_type,
            "baseline_rate": round(self.baseline_rate, 4) if self.baseline_rate is not None else None,
            "current_rate": round(self.current_rate, 4) if self.current_rate is not None else None,
            "delta": round(self.delta, 4),
        }


@dataclass
class Changelog:
    """Auto-generated changelog between two evaluation runs."""

    baseline_run_id: str
    current_run_id: str
    entries: list[ChangelogEntry] = field(default_factory=list)

    @property
    def regressions(self) -> list[ChangelogEntry]:
        return [e for e in self.entries if e.change_type == "regressed"]

    @property
    def improvements(self) -> list[ChangelogEntry]:
        return [e for e in self.entries if e.change_type == "improved"]

    @property
    def added(self) -> list[ChangelogEntry]:
        return [e for e in self.entries if e.change_type == "added"]

    @property
    def removed(self) -> list[ChangelogEntry]:
        return [e for e in self.entries if e.change_type == "removed"]

    @property
    def summary_line(self) -> str:
        """One-line summary of changes."""
        parts = []
        if self.regressions:
            parts.append(f"{len(self.regressions)} regressed")
        if self.improvements:
            parts.append(f"{len(self.improvements)} improved")
        if self.added:
            parts.append(f"{len(self.added)} added")
        if self.removed:
            parts.append(f"{len(self.removed)} removed")
        unchanged = sum(1 for e in self.entries if e.change_type == "unchanged")
        if unchanged:
            parts.append(f"{unchanged} unchanged")
        return ", ".join(parts) if parts else "no changes"

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "current_run_id": self.current_run_id,
            "summary": self.summary_line,
            "regressions": [e.as_dict() for e in self.regressions],
            "improvements": [e.as_dict() for e in self.improvements],
            "added": [e.as_dict() for e in self.added],
            "removed": [e.as_dict() for e in self.removed],
        }

    def format_text(self) -> str:
        lines = [
            f"Changelog: {self.baseline_run_id[:8]} -> {self.current_run_id[:8]}",
            f"  Summary: {self.summary_line}",
        ]
        if self.regressions:
            lines.append("")
            lines.append("  Regressions:")
            for e in sorted(self.regressions, key=lambda x: x.delta):
                lines.append(f"    {e.metric_id}: {e.baseline_rate*100:.0f}% -> {e.current_rate*100:.0f}% ({e.delta*100:+.1f}%)")
        if self.improvements:
            lines.append("")
            lines.append("  Improvements:")
            for e in sorted(self.improvements, key=lambda x: -x.delta):
                lines.append(f"    {e.metric_id}: {e.baseline_rate*100:.0f}% -> {e.current_rate*100:.0f}% ({e.delta*100:+.1f}%)")
        if self.added:
            lines.append("")
            lines.append(f"  New metrics: {', '.join(e.metric_id for e in self.added)}")
        if self.removed:
            lines.append("")
            lines.append(f"  Removed metrics: {', '.join(e.metric_id for e in self.removed)}")
        return "\n".join(lines)


def generate_changelog(
    baseline_run,
    current_run,
    regression_threshold: float = 0.05,
) -> Changelog:
    """Generate a changelog between two evaluation runs.

    Args:
        baseline_run: Previous EvalRun.
        current_run: Current EvalRun.
        regression_threshold: Minimum absolute change to flag (default 5%).

    Returns:
        Changelog with categorized changes.
    """
    baseline_rates = _compute_pass_rates(baseline_run)
    current_rates = _compute_pass_rates(current_run)

    all_metrics = sorted(set(baseline_rates.keys()) | set(current_rates.keys()))

    entries = []
    for mid in all_metrics:
        br = baseline_rates.get(mid)
        cr = current_rates.get(mid)

        if br is not None and cr is not None:
            delta = cr - br
            if delta < -regression_threshold:
                change_type = "regressed"
            elif delta > regression_threshold:
                change_type = "improved"
            else:
                change_type = "unchanged"
            entries.append(ChangelogEntry(mid, change_type, br, cr, delta))
        elif br is None and cr is not None:
            entries.append(ChangelogEntry(mid, "added", None, cr, 0.0))
        elif br is not None and cr is None:
            entries.append(ChangelogEntry(mid, "removed", br, None, 0.0))

    return Changelog(
        baseline_run_id=baseline_run.id,
        current_run_id=current_run.id,
        entries=entries,
    )


def _compute_pass_rates(run) -> dict[str, float]:
    """Compute per-metric pass rates."""
    counts: dict[str, tuple[int, int]] = {}
    for mr in run.metric_results:
        mid = mr.metric_id
        if mid not in counts:
            counts[mid] = (0, 0)
        p, t = counts[mid]
        if mr.passed is True:
            counts[mid] = (p + 1, t + 1)
        elif mr.passed is False:
            counts[mid] = (p, t + 1)
    return {mid: p / t if t > 0 else 0.0 for mid, (p, t) in counts.items()}
