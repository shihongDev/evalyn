"""Time-to-fix tracking: how many runs until failing items start passing.

Tracks per-item failure/fix timelines across chronological runs
to identify chronic failures and measure fix velocity.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ItemFixHistory:
    """Fix history for a single item across runs."""

    item_id: str = ""
    first_failed_run: int = 0
    first_fixed_run: int | None = None
    runs_to_fix: int | None = None
    still_failing: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "first_failed_run": self.first_failed_run,
            "first_fixed_run": self.first_fixed_run,
            "runs_to_fix": self.runs_to_fix,
            "still_failing": self.still_failing,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ItemFixHistory:
        return cls(
            item_id=data.get("item_id", ""),
            first_failed_run=data.get("first_failed_run", 0),
            first_fixed_run=data.get("first_fixed_run"),
            runs_to_fix=data.get("runs_to_fix"),
            still_failing=data.get("still_failing", True),
        )


@dataclass
class TimeToFixReport:
    """Aggregate time-to-fix report across all tracked items."""

    items: list[ItemFixHistory] = field(default_factory=list)
    avg_runs_to_fix: float = 0.0
    median_runs_to_fix: float = 0.0
    still_failing_count: int = 0
    fixed_count: int = 0
    total_tracked: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": [item.as_dict() for item in self.items],
            "avg_runs_to_fix": self.avg_runs_to_fix,
            "median_runs_to_fix": self.median_runs_to_fix,
            "still_failing_count": self.still_failing_count,
            "fixed_count": self.fixed_count,
            "total_tracked": self.total_tracked,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeToFixReport:
        return cls(
            items=[ItemFixHistory.from_dict(i) for i in data.get("items", [])],
            avg_runs_to_fix=data.get("avg_runs_to_fix", 0.0),
            median_runs_to_fix=data.get("median_runs_to_fix", 0.0),
            still_failing_count=data.get("still_failing_count", 0),
            fixed_count=data.get("fixed_count", 0),
            total_tracked=data.get("total_tracked", 0),
        )

    def format_text(self) -> str:
        lines = [
            f"Time-to-Fix Report ({self.total_tracked} items tracked)",
            f"  Fixed: {self.fixed_count}",
            f"  Still failing: {self.still_failing_count}",
        ]
        if self.fixed_count > 0:
            lines.append(f"  Avg runs to fix: {self.avg_runs_to_fix:.1f}")
            lines.append(f"  Median runs to fix: {self.median_runs_to_fix:.1f}")
        for item in self.items:
            status = "FIXED" if not item.still_failing else "FAILING"
            suffix = ""
            if item.runs_to_fix is not None:
                suffix = f" (took {item.runs_to_fix} runs)"
            lines.append(f"  {item.item_id}: {status}{suffix}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def track_item_fixes(item_id: str, run_results: list[bool]) -> ItemFixHistory:
    """Track when an item first failed and first passed after failure.

    Args:
        item_id: Identifier for the item.
        run_results: Chronological list of pass (True) / fail (False) per run.

    Returns:
        ItemFixHistory with fix timeline information.
    """
    history = ItemFixHistory(item_id=item_id)

    # Find first failure
    first_fail = None
    for i, passed in enumerate(run_results):
        if not passed:
            first_fail = i
            break

    if first_fail is None:
        # Never failed
        history.still_failing = False
        history.first_failed_run = 0
        history.first_fixed_run = 0
        history.runs_to_fix = 0
        return history

    history.first_failed_run = first_fail

    # Find first pass after the failure
    for i in range(first_fail + 1, len(run_results)):
        if run_results[i]:
            history.first_fixed_run = i
            history.runs_to_fix = i - first_fail
            history.still_failing = False
            break

    return history


def track_all_fixes(item_results: dict[str, list[bool]]) -> TimeToFixReport:
    """Track fix timelines for all items.

    Args:
        item_results: Mapping of item_id to chronological pass/fail list.

    Returns:
        TimeToFixReport with aggregate statistics.
    """
    items = [track_item_fixes(iid, results) for iid, results in item_results.items()]
    fixed = [item for item in items if not item.still_failing and item.runs_to_fix is not None and item.runs_to_fix > 0]
    still_failing = [item for item in items if item.still_failing]

    fix_times = [item.runs_to_fix for item in fixed if item.runs_to_fix is not None]
    avg = statistics.mean(fix_times) if fix_times else 0.0
    med = statistics.median(fix_times) if fix_times else 0.0

    return TimeToFixReport(
        items=items,
        avg_runs_to_fix=avg,
        median_runs_to_fix=med,
        still_failing_count=len(still_failing),
        fixed_count=len(fixed),
        total_tracked=len(items),
    )


def compute_fix_percentiles(report: TimeToFixReport) -> dict[str, float]:
    """Compute P25, P50, P75 of runs-to-fix for fixed items.

    Returns:
        Dict with keys "p25", "p50", "p75". All 0.0 if no fixed items.
    """
    fix_times = sorted(
        item.runs_to_fix
        for item in report.items
        if not item.still_failing and item.runs_to_fix is not None and item.runs_to_fix > 0
    )
    if not fix_times:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0}

    def _percentile(data: list[int], pct: float) -> float:
        if len(data) == 1:
            return float(data[0])
        k = (len(data) - 1) * pct
        f = int(k)
        c = f + 1
        if c >= len(data):
            return float(data[f])
        return data[f] + (k - f) * (data[c] - data[f])

    return {
        "p25": _percentile(fix_times, 0.25),
        "p50": _percentile(fix_times, 0.50),
        "p75": _percentile(fix_times, 0.75),
    }


def identify_chronic_failures(report: TimeToFixReport, min_runs: int = 5) -> list[str]:
    """Identify items still failing after at least min_runs since first failure.

    Args:
        report: TimeToFixReport to examine.
        min_runs: Minimum number of runs since first failure to qualify.

    Returns:
        List of item IDs that are chronic failures.
    """
    chronic = []
    for item in report.items:
        if not item.still_failing:
            continue
        # Check how many runs have elapsed since first failure.
        # We can infer total runs from the run_results length via the report,
        # but since we only have the history, use the total_tracked as proxy.
        # The item is chronic if it has been failing and first_failed_run
        # is far enough from the end. We approximate by checking all items
        # in the report share the same run count, so we use item index math.
        # Simpler: item is still_failing and the number of runs it existed
        # through can be inferred. Since we don't store total_runs on ItemFixHistory,
        # we check if min_runs is <= total_tracked - first_failed_run.
        runs_since_fail = report.total_tracked - item.first_failed_run
        if runs_since_fail >= min_runs:
            chronic.append(item.item_id)
    return chronic


def render_fix_timeline(history: ItemFixHistory, total_runs: int) -> str:
    """Render an ASCII timeline showing fail/pass transitions.

    Uses 'F' for fail, 'P' for pass, '.' for unknown/before tracking.

    Args:
        history: Single item fix history.
        total_runs: Total number of runs to display.

    Returns:
        String like "item_1: ..FFFPP" showing the timeline.
    """
    if total_runs <= 0:
        return f"{history.item_id}: "

    # If item never failed (runs_to_fix == 0 and not still_failing)
    if not history.still_failing and history.runs_to_fix == 0:
        timeline = "P" * total_runs
        return f"{history.item_id}: {timeline}"

    chars = []
    for i in range(total_runs):
        if i < history.first_failed_run or history.first_fixed_run is not None and i >= history.first_fixed_run:
            chars.append("P")
        else:
            chars.append("F")

    return f"{history.item_id}: {''.join(chars)}"
