"""
Core analysis classes and functions.

Provides dataclasses for analysis results and functions to load/analyze eval runs.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MetricStats:
    """Statistics for a single metric across items."""

    metric_id: str
    metric_type: str  # objective or subjective
    count: int = 0
    passed: int = 0
    failed: int = 0
    scores: List[float] = field(default_factory=list)
    has_pass_fail: bool = False  # True if metric has pass/fail semantics

    @property
    def pass_rate(self) -> Optional[float]:
        """Return pass rate or None if metric doesn't have pass/fail semantics."""
        if not self.has_pass_fail:
            return None
        return self.passed / self.count if self.count > 0 else 0.0

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def min_score(self) -> float:
        return min(self.scores) if self.scores else 0.0

    @property
    def max_score(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def std_dev(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        avg = self.avg_score
        variance = sum((s - avg) ** 2 for s in self.scores) / len(self.scores)
        return math.sqrt(variance)


@dataclass
class ItemStats:
    """Statistics for a single dataset item across metrics."""

    item_id: str
    metrics_passed: int = 0
    metrics_failed: int = 0
    metric_results: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # metric_id -> {passed, score, reason, details}

    @property
    def all_passed(self) -> bool:
        return self.metrics_failed == 0 and self.metrics_passed > 0


@dataclass
class RunAnalysis:
    """Complete analysis of an eval run."""

    run_id: str
    dataset_name: str
    created_at: str
    total_items: int
    total_metrics: int
    metric_stats: Dict[str, MetricStats]
    item_stats: Dict[str, ItemStats]
    failed_items: List[str]

    @property
    def overall_pass_rate(self) -> float:
        """Percentage of items that passed ALL metrics."""
        if self.total_items == 0:
            return 0.0
        all_passed = sum(1 for item in self.item_stats.values() if item.all_passed)
        return all_passed / self.total_items


def load_eval_run(path: Path) -> Dict[str, Any]:
    """Load a single eval run JSON file.

    Args:
        path: Path to results.json file, folder containing results.json, or legacy .json file
    """
    path = Path(path)
    # Handle folder path
    if path.is_dir():
        path = path / "results.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_eval_runs(dataset_dir: Path) -> List[Path]:
    """Find all eval run folders/files in a dataset directory.

    Returns paths to results.json files (new structure) or .json files (legacy).
    """
    eval_runs_dir = Path(dataset_dir) / "eval_runs"
    if not eval_runs_dir.exists():
        return []

    runs = []

    # New folder structure: look for folders with results.json
    for item in eval_runs_dir.iterdir():
        if item.is_dir():
            results_file = item / "results.json"
            if results_file.exists():
                runs.append(results_file)

    # Legacy flat JSON files
    for json_file in eval_runs_dir.glob("*.json"):
        runs.append(json_file)

    # Sort by modification time, newest first
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def analyze_run(run_data: Dict[str, Any]) -> RunAnalysis:
    """Analyze a single eval run and compute statistics."""
    metric_stats: Dict[str, MetricStats] = {}
    item_stats: Dict[str, ItemStats] = defaultdict(lambda: ItemStats(item_id=""))

    # Build metric type lookup from run data
    metric_types = {}
    for m in run_data.get("metrics", []):
        metric_types[m["id"]] = m.get("type", "unknown")

    # Process each result
    for result in run_data.get("metric_results", []):
        metric_id = result["metric_id"]
        item_id = result["item_id"]
        score = result.get("score")
        # Handle None scores (API errors, etc.) - keep as 0.0 for averaging
        numeric_score = score if score is not None else 0.0
        passed = result.get("passed")  # Can be True, False, or None

        # Update metric stats
        if metric_id not in metric_stats:
            metric_stats[metric_id] = MetricStats(
                metric_id=metric_id, metric_type=metric_types.get(metric_id, "unknown")
            )
        ms = metric_stats[metric_id]
        ms.count += 1
        ms.scores.append(numeric_score)

        # Only track pass/fail if the metric has pass/fail semantics (passed is not None)
        if passed is not None:
            ms.has_pass_fail = True
            if passed:
                ms.passed += 1
            else:
                ms.failed += 1

        # Update item stats with full details
        if item_stats[item_id].item_id == "":
            item_stats[item_id].item_id = item_id
        details = result.get("details", {})
        item_stats[item_id].metric_results[metric_id] = {
            "passed": passed,
            "score": numeric_score,
            "reason": details.get("reason"),
            "details": details,
        }
        # Only count pass/fail for metrics that have pass/fail semantics
        if passed is True:
            item_stats[item_id].metrics_passed += 1
        elif passed is False:
            item_stats[item_id].metrics_failed += 1

    # Get failed items
    failed_items = [
        item_id for item_id, stats in item_stats.items() if not stats.all_passed
    ]

    return RunAnalysis(
        run_id=run_data.get("id", "unknown"),
        dataset_name=run_data.get("dataset_name", "unknown"),
        created_at=run_data.get("created_at", ""),
        total_items=len(item_stats),
        total_metrics=len(metric_stats),
        metric_stats=dict(metric_stats),
        item_stats=dict(item_stats),
        failed_items=failed_items,
    )
