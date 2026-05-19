"""Metric execution priority queue.

Orders metrics by likelihood of failure (based on historical data)
so that failing metrics surface earlier in the evaluation output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MetricPriority:
    """Priority data for a single metric."""

    metric_id: str
    historical_fail_rate: float  # 0.0-1.0, higher = more likely to fail
    priority_score: float = 0.0  # computed: higher = evaluate first

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "historical_fail_rate": round(self.historical_fail_rate, 4),
            "priority_score": round(self.priority_score, 4),
        }


def compute_priorities(
    metrics: list,
    historical_fail_rates: dict[str, float] | None = None,
    metric_types: dict[str, str] | None = None,
) -> list[MetricPriority]:
    """Compute execution priorities for metrics.

    Strategy: metrics most likely to fail run first for fast feedback.
    Objective metrics run before subjective (cheaper = earlier feedback).

    Args:
        metrics: List of Metric objects.
        historical_fail_rates: Optional dict of metric_id -> fail rate (0-1)
            from previous runs. Higher fail rate = higher priority.
        metric_types: Optional dict of metric_id -> "objective"/"subjective".

    Returns:
        List of MetricPriority sorted by priority (highest first).
    """
    historical_fail_rates = historical_fail_rates or {}
    metric_types = metric_types or {}

    priorities = []
    for metric in metrics:
        mid = metric.spec.id
        fail_rate = historical_fail_rates.get(mid, 0.5)  # default 50%
        mtype = metric_types.get(mid, getattr(metric.spec, "type", "unknown"))

        # Priority = fail_rate * type_bonus
        # Objective metrics get 1.5x bonus (cheaper = run first)
        type_bonus = 1.5 if mtype == "objective" else 1.0
        priority_score = fail_rate * type_bonus

        priorities.append(MetricPriority(
            metric_id=mid,
            historical_fail_rate=fail_rate,
            priority_score=priority_score,
        ))

    # Sort highest priority first
    priorities.sort(key=lambda p: p.priority_score, reverse=True)
    return priorities


def sort_metrics_by_priority(
    metrics: list,
    historical_fail_rates: dict[str, float] | None = None,
) -> list:
    """Sort metrics by execution priority (most likely to fail first).

    Args:
        metrics: List of Metric objects.
        historical_fail_rates: Dict of metric_id -> historical fail rate.

    Returns:
        New list of metrics sorted by priority (highest first).
    """
    priorities = compute_priorities(metrics, historical_fail_rates)
    priority_order = {p.metric_id: i for i, p in enumerate(priorities)}

    return sorted(metrics, key=lambda m: priority_order.get(m.spec.id, len(metrics)))


def extract_fail_rates_from_run(run) -> dict[str, float]:
    """Extract per-metric fail rates from an EvalRun.

    Useful for feeding into the priority queue on subsequent runs.

    Args:
        run: EvalRun object.

    Returns:
        Dict of metric_id -> fail rate (0.0-1.0).
    """
    counts: dict[str, list] = {}
    for mr in run.metric_results:
        mid = mr.metric_id
        if mid not in counts:
            counts[mid] = [0, 0]  # [failed, total]
        counts[mid][1] += 1
        if mr.passed is False:
            counts[mid][0] += 1

    return {
        mid: failed / total if total > 0 else 0.0
        for mid, (failed, total) in counts.items()
    }
