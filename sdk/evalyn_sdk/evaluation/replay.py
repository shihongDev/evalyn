"""Evaluation replay: re-run a past evaluation with different judge prompts or providers.

Enables comparing how changes to judge configuration affect evaluation scores,
without re-running the actual agent/function under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReplayConfig:
    """Configuration for replaying an evaluation run."""

    original_run_id: str
    override_provider: Optional[str] = None
    override_model: Optional[str] = None
    override_metrics: Optional[List[str]] = None
    override_prompts: Optional[Dict[str, str]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original_run_id": self.original_run_id,
            "override_provider": self.override_provider,
            "override_model": self.override_model,
            "override_metrics": list(self.override_metrics) if self.override_metrics else None,
            "override_prompts": dict(self.override_prompts) if self.override_prompts else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReplayConfig:
        return cls(
            original_run_id=data["original_run_id"],
            override_provider=data.get("override_provider"),
            override_model=data.get("override_model"),
            override_metrics=data.get("override_metrics"),
            override_prompts=data.get("override_prompts"),
        )


@dataclass
class ReplayPlan:
    """Plan showing what will be replayed before execution."""

    config: ReplayConfig
    items_to_replay: List[str]
    metrics_to_use: List[str]
    estimated_calls: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "items_to_replay": list(self.items_to_replay),
            "metrics_to_use": list(self.metrics_to_use),
            "estimated_calls": self.estimated_calls,
        }

    def format_text(self) -> str:
        """Format plan as human-readable text."""
        lines = [
            "Replay Plan",
            "=" * 40,
            f"Original run:     {self.config.original_run_id}",
            f"Items to replay:  {len(self.items_to_replay)}",
            f"Metrics:          {len(self.metrics_to_use)}",
            f"Estimated calls:  {self.estimated_calls}",
        ]
        if self.config.override_provider:
            lines.append(f"Override provider: {self.config.override_provider}")
        if self.config.override_model:
            lines.append(f"Override model:    {self.config.override_model}")
        if self.config.override_prompts:
            lines.append(f"Override prompts:  {len(self.config.override_prompts)} metric(s)")
        return "\n".join(lines)


@dataclass
class ReplayComparison:
    """Comparison of original vs replayed score for a single item-metric pair."""

    item_id: str
    metric_id: str
    original_score: float
    replayed_score: float
    delta: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "metric_id": self.metric_id,
            "original_score": self.original_score,
            "replayed_score": self.replayed_score,
            "delta": self.delta,
        }


@dataclass
class ReplayReport:
    """Aggregate report comparing original and replayed evaluation results."""

    comparisons: List[ReplayComparison] = field(default_factory=list)
    improved_count: int = 0
    degraded_count: int = 0
    unchanged_count: int = 0
    mean_delta: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "comparisons": [c.as_dict() for c in self.comparisons],
            "improved_count": self.improved_count,
            "degraded_count": self.degraded_count,
            "unchanged_count": self.unchanged_count,
            "mean_delta": self.mean_delta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReplayReport:
        comparisons = [
            ReplayComparison(**c) for c in data.get("comparisons", [])
        ]
        return cls(
            comparisons=comparisons,
            improved_count=data.get("improved_count", 0),
            degraded_count=data.get("degraded_count", 0),
            unchanged_count=data.get("unchanged_count", 0),
            mean_delta=data.get("mean_delta", 0.0),
        )

    def format_text(self) -> str:
        """Format report as human-readable text."""
        lines = [
            "Replay Report",
            "=" * 40,
            f"Total comparisons: {len(self.comparisons)}",
            f"Improved:          {self.improved_count}",
            f"Degraded:          {self.degraded_count}",
            f"Unchanged:         {self.unchanged_count}",
            f"Mean delta:        {self.mean_delta:+.4f}",
        ]
        return "\n".join(lines)


def build_replay_plan(
    config: ReplayConfig,
    original_items: List[str],
    original_metrics: List[str],
) -> ReplayPlan:
    """Build a plan showing what will be replayed.

    Uses override_metrics from config if provided, otherwise uses original_metrics.
    Estimated calls = len(items) * len(metrics).
    """
    metrics = config.override_metrics if config.override_metrics else original_metrics
    estimated_calls = len(original_items) * len(metrics)
    return ReplayPlan(
        config=config,
        items_to_replay=list(original_items),
        metrics_to_use=list(metrics),
        estimated_calls=estimated_calls,
    )


def compare_replay_results(
    original_results: List[Dict[str, Any]],
    replayed_results: List[Dict[str, Any]],
    threshold: float = 0.01,
) -> ReplayReport:
    """Compare original vs replayed scores.

    Each result dict has "item_id", "metric_id", "score".
    Improved: delta > threshold (replayed score higher).
    Degraded: delta < -threshold (replayed score lower).
    Unchanged: abs(delta) <= threshold.
    """
    # Build lookup for replayed results
    replayed_lookup: Dict[tuple, float] = {}
    for r in replayed_results:
        key = (r["item_id"], r["metric_id"])
        replayed_lookup[key] = r["score"]

    comparisons: List[ReplayComparison] = []
    improved = 0
    degraded = 0
    unchanged = 0

    for orig in original_results:
        key = (orig["item_id"], orig["metric_id"])
        if key not in replayed_lookup:
            continue
        original_score = orig["score"]
        replayed_score = replayed_lookup[key]
        delta = replayed_score - original_score

        comparisons.append(ReplayComparison(
            item_id=orig["item_id"],
            metric_id=orig["metric_id"],
            original_score=original_score,
            replayed_score=replayed_score,
            delta=delta,
        ))

        if delta > threshold:
            improved += 1
        elif delta < -threshold:
            degraded += 1
        else:
            unchanged += 1

    mean_delta = sum(c.delta for c in comparisons) / len(comparisons) if comparisons else 0.0

    return ReplayReport(
        comparisons=comparisons,
        improved_count=improved,
        degraded_count=degraded,
        unchanged_count=unchanged,
        mean_delta=mean_delta,
    )


def summarize_replay(report: ReplayReport) -> Dict[str, Any]:
    """Summary stats from a replay report."""
    return {
        "total_comparisons": len(report.comparisons),
        "improved_count": report.improved_count,
        "degraded_count": report.degraded_count,
        "unchanged_count": report.unchanged_count,
        "mean_delta": report.mean_delta,
        "improved_pct": (
            report.improved_count / len(report.comparisons) * 100
            if report.comparisons else 0.0
        ),
        "degraded_pct": (
            report.degraded_count / len(report.comparisons) * 100
            if report.comparisons else 0.0
        ),
    }
