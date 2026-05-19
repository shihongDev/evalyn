"""Metric registry freeze: lock the metric set for a project.

Prevents accidental modifications to the metrics used in evaluation.
Saves a frozen snapshot; warns when metrics change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

FREEZE_FILENAME = ".evalyn-metrics-freeze.json"


@dataclass
class FrozenMetricSet:
    """A frozen snapshot of metric configurations."""

    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # id -> {type, version_hash, ...}
    frozen_at: str = ""
    description: str = ""

    @property
    def metric_ids(self) -> List[str]:
        return sorted(self.metrics.keys())

    @property
    def count(self) -> int:
        return len(self.metrics)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "frozen_at": self.frozen_at,
            "description": self.description,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FrozenMetricSet:
        return cls(
            metrics=data.get("metrics", {}),
            frozen_at=data.get("frozen_at", ""),
            description=data.get("description", ""),
        )


@dataclass
class FreezeCheckResult:
    """Result of checking metrics against a frozen set."""

    is_frozen: bool
    matches: bool
    added_metrics: List[str] = field(default_factory=list)
    removed_metrics: List[str] = field(default_factory=list)
    changed_metrics: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "is_frozen": self.is_frozen,
            "matches": self.matches,
            "added": self.added_metrics,
            "removed": self.removed_metrics,
            "changed": self.changed_metrics,
        }

    def format_text(self) -> str:
        if not self.is_frozen:
            return "Metrics: not frozen (no freeze file found)"
        if self.matches:
            return "Metrics: frozen and matching"
        lines = ["Metrics: MISMATCH with frozen set"]
        if self.added_metrics:
            lines.append(f"  Added:   {', '.join(self.added_metrics)}")
        if self.removed_metrics:
            lines.append(f"  Removed: {', '.join(self.removed_metrics)}")
        if self.changed_metrics:
            lines.append(f"  Changed: {', '.join(self.changed_metrics)}")
        return "\n".join(lines)


def freeze_metrics(
    metric_specs: List,
    project_dir: Path,
    description: str = "",
) -> FrozenMetricSet:
    """Freeze the current metric set for a project.

    Args:
        metric_specs: List of MetricSpec objects to freeze.
        project_dir: Directory to save freeze file.
        description: Optional description.

    Returns:
        FrozenMetricSet that was saved.
    """
    from ..models import now_utc

    metrics = {}
    for spec in metric_specs:
        metrics[spec.id] = {
            "type": spec.type,
            "version_hash": getattr(spec, "version_hash", ""),
            "description": spec.description,
        }

    frozen = FrozenMetricSet(
        metrics=metrics,
        frozen_at=now_utc().isoformat(),
        description=description,
    )

    freeze_path = Path(project_dir) / FREEZE_FILENAME
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(frozen.as_dict(), indent=2), encoding="utf-8")

    return frozen


def load_freeze(project_dir: Path) -> Optional[FrozenMetricSet]:
    """Load an existing freeze file.

    Returns None if no freeze file exists.
    """
    freeze_path = Path(project_dir) / FREEZE_FILENAME
    if not freeze_path.exists():
        return None
    try:
        data = json.loads(freeze_path.read_text(encoding="utf-8"))
        return FrozenMetricSet.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        return None


def check_against_freeze(
    metric_specs: List,
    project_dir: Path,
) -> FreezeCheckResult:
    """Check current metrics against the frozen set.

    Args:
        metric_specs: Current MetricSpec list.
        project_dir: Directory containing freeze file.

    Returns:
        FreezeCheckResult with match status and differences.
    """
    frozen = load_freeze(project_dir)
    if frozen is None:
        return FreezeCheckResult(is_frozen=False, matches=True)

    current_ids = {s.id for s in metric_specs}
    frozen_ids = set(frozen.metrics.keys())

    added = sorted(current_ids - frozen_ids)
    removed = sorted(frozen_ids - current_ids)

    changed = []
    for spec in metric_specs:
        if spec.id in frozen.metrics:
            frozen_hash = frozen.metrics[spec.id].get("version_hash", "")
            current_hash = getattr(spec, "version_hash", "")
            if frozen_hash and current_hash and frozen_hash != current_hash:
                changed.append(spec.id)

    matches = not added and not removed and not changed

    return FreezeCheckResult(
        is_frozen=True,
        matches=matches,
        added_metrics=added,
        removed_metrics=removed,
        changed_metrics=changed,
    )


def unfreeze_metrics(project_dir: Path) -> bool:
    """Remove the freeze file.

    Returns True if file was removed, False if it didn't exist.
    """
    freeze_path = Path(project_dir) / FREEZE_FILENAME
    if freeze_path.exists():
        freeze_path.unlink()
        return True
    return False
