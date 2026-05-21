"""Run management utilities: cleanup, filtering, and organization.

Provides functions for managing evaluation runs: finding runs by criteria,
bulk cleanup of old or low-quality runs, and tag-based filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class CleanupResult:
    """Result of a run cleanup operation."""

    deleted_count: int
    kept_count: int
    deleted_ids: list[str]
    skipped_pinned: int

    @property
    def total_scanned(self) -> int:
        return self.deleted_count + self.kept_count


def filter_runs_by_tag(
    runs: list,
    tag: str,
) -> list:
    """Filter runs that have a specific tag.

    Args:
        runs: List of EvalRun objects.
        tag: Tag to filter by.

    Returns:
        List of runs that have the tag.
    """
    return [r for r in runs if hasattr(r, "tags") and tag in (r.tags or [])]


def filter_runs_by_age(
    runs: list,
    max_age_days: int,
) -> list:
    """Find runs older than max_age_days.

    Args:
        runs: List of EvalRun objects.
        max_age_days: Maximum age in days.

    Returns:
        List of runs older than the threshold.
    """
    now = datetime.now(timezone.utc)
    old_runs = []
    for run in runs:
        if run.created_at:
            age = now - run.created_at
            if age.days > max_age_days:
                old_runs.append(run)
    return old_runs


def filter_runs_below_pass_rate(
    runs: list,
    min_pass_rate: float,
) -> list:
    """Find runs with overall pass rate below threshold.

    Uses the summary dict's per-metric pass rates to compute overall pass rate.

    Args:
        runs: List of EvalRun objects.
        min_pass_rate: Minimum acceptable pass rate (0.0 to 1.0).

    Returns:
        List of runs below the threshold.
    """
    below = []
    for run in runs:
        metrics = (run.summary or {}).get("metrics", {})
        if not metrics:
            continue
        pass_rates = [
            m.get("pass_rate", 0) for m in metrics.values() if m.get("pass_rate") is not None
        ]
        if pass_rates:
            overall = sum(pass_rates) / len(pass_rates)
            if overall < min_pass_rate:
                below.append(run)
    return below


def select_runs_for_cleanup(
    runs: list,
    *,
    older_than_days: int | None = None,
    below_pass_rate: float | None = None,
    keep_pinned: bool = True,
) -> CleanupResult:
    """Select runs for cleanup based on criteria.

    This is a dry-run function - it identifies runs to delete but does not
    delete them. The caller decides whether to proceed.

    Args:
        runs: All runs to consider.
        older_than_days: Delete runs older than this many days.
        below_pass_rate: Delete runs with pass rate below this threshold.
        keep_pinned: If True, never delete pinned runs.

    Returns:
        CleanupResult with lists of runs to delete/keep.
    """
    candidates = set()

    if older_than_days is not None:
        for r in filter_runs_by_age(runs, older_than_days):
            candidates.add(r.id)

    if below_pass_rate is not None:
        for r in filter_runs_below_pass_rate(runs, below_pass_rate):
            candidates.add(r.id)

    # Protect pinned runs
    skipped_pinned = 0
    if keep_pinned:
        for run in runs:
            if run.id in candidates and getattr(run, "pinned", False):
                candidates.discard(run.id)
                skipped_pinned += 1

    deleted_ids = sorted(candidates)
    kept_count = len(runs) - len(deleted_ids)

    return CleanupResult(
        deleted_count=len(deleted_ids),
        kept_count=kept_count,
        deleted_ids=deleted_ids,
        skipped_pinned=skipped_pinned,
    )
