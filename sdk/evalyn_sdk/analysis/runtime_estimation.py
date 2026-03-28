"""Metric runtime estimation: predict evaluation time per metric.

Uses historical timing data or defaults to estimate how long each
metric will take, helping with planning and parallelism decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Default execution time estimates (milliseconds per evaluation)
_DEFAULT_TIMES_MS = {
    "objective": 5,         # local computation
    "subjective": 2000,     # LLM API call
}


@dataclass
class MetricTimeEstimate:
    """Runtime estimate for a single metric."""

    metric_id: str
    metric_type: str
    avg_time_ms: float          # average time per item
    total_time_ms: float = 0.0  # total for dataset
    source: str = "default"     # "default" or "historical"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "type": self.metric_type,
            "avg_time_ms": round(self.avg_time_ms, 1),
            "total_time_ms": round(self.total_time_ms, 1),
            "source": self.source,
        }


@dataclass
class RuntimeEstimateReport:
    """Runtime estimation report for an evaluation run."""

    metrics: List[MetricTimeEstimate] = field(default_factory=list)
    total_sequential_ms: float = 0.0
    total_parallel_ms: float = 0.0  # with max_workers
    item_count: int = 0
    max_workers: int = 1

    @property
    def slowest_metric(self) -> Optional[str]:
        if not self.metrics:
            return None
        return max(self.metrics, key=lambda m: m.total_time_ms).metric_id

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_sequential_seconds": round(self.total_sequential_ms / 1000, 1),
            "total_parallel_seconds": round(self.total_parallel_ms / 1000, 1),
            "item_count": self.item_count,
            "max_workers": self.max_workers,
            "slowest_metric": self.slowest_metric,
            "metrics": [m.as_dict() for m in self.metrics],
        }

    def format_text(self) -> str:
        lines = [
            "Runtime Estimation:",
            f"  Sequential: {self.total_sequential_ms/1000:.1f}s",
            f"  Parallel ({self.max_workers} workers): {self.total_parallel_ms/1000:.1f}s",
            f"  Items: {self.item_count}",
        ]
        if self.slowest_metric:
            lines.append(f"  Slowest: {self.slowest_metric}")
        lines.append("")
        for m in sorted(self.metrics, key=lambda x: -x.total_time_ms):
            lines.append(f"  {m.metric_id:<30} {m.avg_time_ms:.0f}ms/item  total: {m.total_time_ms/1000:.1f}s")
        return "\n".join(lines)


def estimate_runtime(
    metric_specs: List,
    item_count: int,
    max_workers: int = 1,
    historical_times: Optional[Dict[str, float]] = None,
) -> RuntimeEstimateReport:
    """Estimate evaluation runtime.

    Args:
        metric_specs: List of MetricSpec objects.
        item_count: Number of items to evaluate.
        max_workers: Parallel workers.
        historical_times: Optional dict of metric_id -> avg ms per item
            from previous runs.

    Returns:
        RuntimeEstimateReport with per-metric and total estimates.
    """
    historical_times = historical_times or {}
    estimates = []
    total_seq = 0.0

    for spec in metric_specs:
        mid = spec.id
        mtype = spec.type

        if mid in historical_times:
            avg_ms = historical_times[mid]
            source = "historical"
        else:
            avg_ms = _DEFAULT_TIMES_MS.get(mtype, _DEFAULT_TIMES_MS["subjective"])
            source = "default"

        total_ms = avg_ms * item_count

        estimates.append(MetricTimeEstimate(
            metric_id=mid,
            metric_type=mtype,
            avg_time_ms=avg_ms,
            total_time_ms=total_ms,
            source=source,
        ))
        total_seq += total_ms

    # Parallel estimate
    if max_workers > 1 and estimates:
        # Subjective metrics benefit from parallelism
        obj_time = sum(e.total_time_ms for e in estimates if e.metric_type == "objective")
        subj_time = sum(e.total_time_ms for e in estimates if e.metric_type != "objective")
        effective_workers = min(max_workers, max(1, item_count))
        total_par = obj_time + subj_time / effective_workers
    else:
        total_par = total_seq

    return RuntimeEstimateReport(
        metrics=estimates,
        total_sequential_ms=total_seq,
        total_parallel_ms=total_par,
        item_count=item_count,
        max_workers=max_workers,
    )
