"""Experiment tracking: group traces and runs by experiment ID.

Enables A/B comparisons by tagging traces and evaluation runs with
an experiment identifier, then comparing metrics across experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExperimentComparison:
    """Comparison of two experiments."""

    experiment_a: str
    experiment_b: str
    metric_deltas: Dict[str, float] = field(default_factory=dict)
    winner: Optional[str] = None  # experiment with higher overall pass rate

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_a": self.experiment_a,
            "experiment_b": self.experiment_b,
            "metric_deltas": {k: round(v, 4) for k, v in self.metric_deltas.items()},
            "winner": self.winner,
        }

    def format_text(self) -> str:
        lines = [f"Experiment Comparison: {self.experiment_a} vs {self.experiment_b}"]
        if self.winner:
            lines.append(f"  Winner: {self.winner}")
        lines.append("")
        for mid, delta in sorted(self.metric_deltas.items()):
            sign = "+" if delta >= 0 else ""
            direction = "B better" if delta > 0.01 else ("A better" if delta < -0.01 else "same")
            lines.append(f"  {mid:<30} {sign}{delta*100:.1f}% ({direction})")
        return "\n".join(lines)


def group_runs_by_experiment(runs: List) -> Dict[str, List]:
    """Group evaluation runs by their experiment tag.

    Looks for experiment_id in run tags or metadata.

    Args:
        runs: List of EvalRun objects.

    Returns:
        Dict of experiment_id -> list of runs.
    """
    groups: Dict[str, List] = {}
    for run in runs:
        exp_id = _get_experiment_id(run)
        if exp_id:
            groups.setdefault(exp_id, []).append(run)
    return groups


def compare_experiments(
    runs_a: List,
    runs_b: List,
    experiment_a: str,
    experiment_b: str,
) -> ExperimentComparison:
    """Compare metrics between two experiments.

    Uses the most recent run from each experiment for comparison.

    Args:
        runs_a: Runs from experiment A.
        runs_b: Runs from experiment B.
        experiment_a: Name of experiment A.
        experiment_b: Name of experiment B.

    Returns:
        ExperimentComparison with per-metric deltas.
    """
    rates_a = _compute_pass_rates(runs_a)
    rates_b = _compute_pass_rates(runs_b)

    all_metrics = sorted(set(rates_a.keys()) | set(rates_b.keys()))
    deltas = {}
    for mid in all_metrics:
        ra = rates_a.get(mid, 0.0)
        rb = rates_b.get(mid, 0.0)
        deltas[mid] = rb - ra  # positive = B is better

    # Determine winner by average delta
    avg_delta = sum(deltas.values()) / len(deltas) if deltas else 0.0
    winner = None
    if avg_delta > 0.01:
        winner = experiment_b
    elif avg_delta < -0.01:
        winner = experiment_a

    return ExperimentComparison(
        experiment_a=experiment_a,
        experiment_b=experiment_b,
        metric_deltas=deltas,
        winner=winner,
    )


def _get_experiment_id(run) -> Optional[str]:
    """Extract experiment ID from a run."""
    # Check tags for experiment-* pattern
    for tag in getattr(run, "tags", []) or []:
        if tag.startswith("experiment:"):
            return tag.split(":", 1)[1]
        if tag.startswith("exp:"):
            return tag.split(":", 1)[1]
    # Check name
    name = getattr(run, "name", None)
    if name and name.startswith("experiment-"):
        return name
    return None


def _compute_pass_rates(runs: List) -> Dict[str, float]:
    """Compute per-metric pass rates from the most recent run."""
    if not runs:
        return {}
    # Use most recent run (sort by created_at descending)
    sorted_runs = sorted(runs, key=lambda r: r.created_at, reverse=True)
    run = sorted_runs[0]

    rates: Dict[str, float] = {}
    metric_counts: Dict[str, Tuple[int, int]] = {}  # metric -> (passed, total)

    for mr in run.metric_results:
        mid = mr.metric_id
        if mid not in metric_counts:
            metric_counts[mid] = (0, 0)
        p, t = metric_counts[mid]
        if mr.passed is True:
            metric_counts[mid] = (p + 1, t + 1)
        elif mr.passed is False:
            metric_counts[mid] = (p, t + 1)

    for mid, (p, t) in metric_counts.items():
        rates[mid] = p / t if t > 0 else 0.0

    return rates
