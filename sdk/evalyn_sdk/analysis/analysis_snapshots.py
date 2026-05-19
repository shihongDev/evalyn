"""
Analysis snapshots: save analysis state at a point in time for later comparison.

Provides snapshot creation, storage, comparison, and trend visualization.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AnalysisSnapshot:
    """A frozen view of analysis state at a point in time."""

    snapshot_id: str
    timestamp: str
    run_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    pass_rate: float = 0.0
    total_items: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "metrics": dict(self.metrics),
            "pass_rate": self.pass_rate,
            "total_items": self.total_items,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisSnapshot:
        return cls(
            snapshot_id=data["snapshot_id"],
            timestamp=data["timestamp"],
            run_id=data.get("run_id", ""),
            metrics=dict(data.get("metrics", {})),
            pass_rate=data.get("pass_rate", 0.0),
            total_items=data.get("total_items", 0),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SnapshotComparison:
    """Result of comparing two snapshots."""

    snapshot_a_id: str
    snapshot_b_id: str
    metric_deltas: dict[str, float] = field(default_factory=dict)
    pass_rate_delta: float = 0.0
    improved_metrics: list[str] = field(default_factory=list)
    degraded_metrics: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_a_id": self.snapshot_a_id,
            "snapshot_b_id": self.snapshot_b_id,
            "metric_deltas": dict(self.metric_deltas),
            "pass_rate_delta": self.pass_rate_delta,
            "improved_metrics": list(self.improved_metrics),
            "degraded_metrics": list(self.degraded_metrics),
        }

    def format_text(self) -> str:
        lines = [
            f"Snapshot Comparison: {self.snapshot_a_id} -> {self.snapshot_b_id}",
            f"Pass rate delta: {self.pass_rate_delta:+.4f}",
        ]
        if self.improved_metrics:
            lines.append(f"Improved: {', '.join(self.improved_metrics)}")
        if self.degraded_metrics:
            lines.append(f"Degraded: {', '.join(self.degraded_metrics)}")
        if self.metric_deltas:
            lines.append("Metric deltas:")
            for metric_id, delta in sorted(self.metric_deltas.items()):
                lines.append(f"  {metric_id}: {delta:+.4f}")
        return "\n".join(lines)


@dataclass
class SnapshotStore:
    """In-memory collection of snapshots."""

    snapshots: list[AnalysisSnapshot] = field(default_factory=list)

    def add(self, snapshot: AnalysisSnapshot) -> None:
        self.snapshots.append(snapshot)

    def get(self, snapshot_id: str) -> AnalysisSnapshot | None:
        for s in self.snapshots:
            if s.snapshot_id == snapshot_id:
                return s
        return None

    def get_latest(self) -> AnalysisSnapshot | None:
        if not self.snapshots:
            return None
        return self.snapshots[-1]

    def list_all(self) -> list[AnalysisSnapshot]:
        return list(self.snapshots)

    @property
    def count(self) -> int:
        return len(self.snapshots)

    def as_dict(self) -> dict[str, Any]:
        return {"snapshots": [s.as_dict() for s in self.snapshots]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SnapshotStore:
        snapshots = [
            AnalysisSnapshot.from_dict(s) for s in data.get("snapshots", [])
        ]
        return cls(snapshots=snapshots)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2)

    @classmethod
    def from_json(cls, data: str) -> SnapshotStore:
        return cls.from_dict(json.loads(data))


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_snapshot(
    run_id: str,
    metrics: dict[str, float],
    pass_rate: float = 0.0,
    total_items: int = 0,
) -> AnalysisSnapshot:
    """Factory: create a snapshot with auto-generated ID and timestamp."""
    return AnalysisSnapshot(
        snapshot_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
        metrics=dict(metrics),
        pass_rate=pass_rate,
        total_items=total_items,
    )


def compare_snapshots(
    a: AnalysisSnapshot,
    b: AnalysisSnapshot,
    threshold: float = 0.01,
) -> SnapshotComparison:
    """Compare two snapshots. Positive deltas mean b is higher than a."""
    metric_deltas: dict[str, float] = {}
    improved: list[str] = []
    degraded: list[str] = []

    all_keys = set(a.metrics) | set(b.metrics)
    for key in sorted(all_keys):
        val_a = a.metrics.get(key, 0.0)
        val_b = b.metrics.get(key, 0.0)
        delta = val_b - val_a
        metric_deltas[key] = delta
        if delta > threshold:
            improved.append(key)
        elif delta < -threshold:
            degraded.append(key)

    return SnapshotComparison(
        snapshot_a_id=a.snapshot_id,
        snapshot_b_id=b.snapshot_id,
        metric_deltas=metric_deltas,
        pass_rate_delta=b.pass_rate - a.pass_rate,
        improved_metrics=improved,
        degraded_metrics=degraded,
    )


def find_trend(
    snapshots: list[AnalysisSnapshot],
    metric_id: str,
) -> list[float]:
    """Extract time series for one metric across snapshots (in order given)."""
    return [s.metrics.get(metric_id, 0.0) for s in snapshots]


def render_snapshot_timeline(
    snapshots: list[AnalysisSnapshot],
    metric_id: str,
    width: int = 40,
) -> str:
    """Render an ASCII timeline chart for a metric across snapshots.

    Each row shows the snapshot index and a bar proportional to the metric value.
    """
    values = find_trend(snapshots, metric_id)
    if not values:
        return f"No data for metric: {metric_id}"

    max_val = max(values) if max(values) > 0 else 1.0

    lines = [f"Timeline: {metric_id}"]
    for i, val in enumerate(values):
        bar_len = int((val / max_val) * width) if max_val > 0 else 0
        bar = "#" * bar_len
        lines.append(f"  [{i:3d}] {bar} {val:.4f}")
    return "\n".join(lines)
