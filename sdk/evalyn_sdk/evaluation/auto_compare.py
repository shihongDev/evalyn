"""Evaluation comparison auto-trigger.

After each evaluation run, automatically compare against a pinned
baseline run and report regressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AutoCompareResult:
    """Result from an automatic comparison against baseline."""

    has_baseline: bool
    baseline_run_id: Optional[str] = None
    regressions: List[Dict[str, Any]] = field(default_factory=list)
    improvements: List[Dict[str, Any]] = field(default_factory=list)
    unchanged: int = 0
    overall_delta: float = 0.0

    @property
    def has_regressions(self) -> bool:
        return len(self.regressions) > 0

    @property
    def status(self) -> str:
        if not self.has_baseline:
            return "NO_BASELINE"
        if self.has_regressions:
            return "REGRESSION"
        if self.improvements:
            return "IMPROVED"
        return "STABLE"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "has_baseline": self.has_baseline,
            "baseline_run_id": self.baseline_run_id,
            "regressions": self.regressions,
            "improvements": self.improvements,
            "unchanged": self.unchanged,
            "overall_delta": round(self.overall_delta, 4),
        }

    def format_text(self) -> str:
        if not self.has_baseline:
            return "Auto-compare: no pinned baseline run found"

        lines = [f"Auto-compare vs baseline {self.baseline_run_id[:8]}:"]
        lines.append(f"  Status: {self.status}")
        lines.append(f"  Overall delta: {'+' if self.overall_delta >= 0 else ''}{self.overall_delta*100:.1f}%")

        if self.regressions:
            lines.append(f"  Regressions ({len(self.regressions)}):")
            for r in self.regressions[:5]:
                lines.append(f"    {r['metric_id']}: {r['baseline_rate']*100:.0f}% -> {r['current_rate']*100:.0f}% ({r['delta']*100:+.1f}%)")

        if self.improvements:
            lines.append(f"  Improvements ({len(self.improvements)}):")
            for imp in self.improvements[:5]:
                lines.append(f"    {imp['metric_id']}: {imp['baseline_rate']*100:.0f}% -> {imp['current_rate']*100:.0f}% ({imp['delta']*100:+.1f}%)")

        return "\n".join(lines)


def auto_compare(
    current_run,
    all_runs: List,
    regression_threshold: float = 0.05,
) -> AutoCompareResult:
    """Automatically compare current run against the pinned baseline.

    Finds the pinned run (if any), computes per-metric deltas,
    and identifies regressions and improvements.

    Args:
        current_run: The just-completed EvalRun.
        all_runs: All runs in the project (to find baseline).
        regression_threshold: Minimum absolute drop to flag as regression (default 5%).

    Returns:
        AutoCompareResult with comparison details.
    """
    # Find pinned baseline
    baseline = None
    for run in all_runs:
        if getattr(run, "pinned", False) and run.id != current_run.id:
            baseline = run
            break

    if baseline is None:
        return AutoCompareResult(has_baseline=False)

    # Compute pass rates for both runs
    baseline_rates = _compute_pass_rates(baseline)
    current_rates = _compute_pass_rates(current_run)

    all_metrics = sorted(set(baseline_rates.keys()) | set(current_rates.keys()))

    regressions = []
    improvements = []
    unchanged = 0

    for mid in all_metrics:
        br = baseline_rates.get(mid, 0.0)
        cr = current_rates.get(mid, 0.0)
        delta = cr - br

        entry = {
            "metric_id": mid,
            "baseline_rate": br,
            "current_rate": cr,
            "delta": delta,
        }

        if delta < -regression_threshold:
            regressions.append(entry)
        elif delta > regression_threshold:
            improvements.append(entry)
        else:
            unchanged += 1

    # Overall delta (average)
    deltas = [current_rates.get(m, 0) - baseline_rates.get(m, 0) for m in all_metrics]
    overall_delta = sum(deltas) / len(deltas) if deltas else 0.0

    return AutoCompareResult(
        has_baseline=True,
        baseline_run_id=baseline.id,
        regressions=regressions,
        improvements=improvements,
        unchanged=unchanged,
        overall_delta=overall_delta,
    )


def _compute_pass_rates(run) -> Dict[str, float]:
    """Compute per-metric pass rates from a run."""
    counts: Dict[str, Tuple[int, int]] = {}
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
