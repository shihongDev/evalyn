"""Metric cross-reference view: show which bundles include each metric.

Maps metrics to their bundle memberships and vice versa, enabling
users to understand metric coverage across evaluation profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricCrossRef:
    """Cross-reference data for a single metric."""

    metric_id: str
    metric_type: str
    bundles: list[str] = field(default_factory=list)
    category: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "type": self.metric_type,
            "bundles": list(self.bundles),
            "category": self.category,
            "bundle_count": len(self.bundles),
        }


@dataclass
class BundleCrossRef:
    """Cross-reference data for a single bundle."""

    bundle_name: str
    metrics: list[str] = field(default_factory=list)
    objective_count: int = 0
    subjective_count: int = 0

    @property
    def total_metrics(self) -> int:
        return len(self.metrics)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_name": self.bundle_name,
            "metrics": list(self.metrics),
            "total_metrics": self.total_metrics,
            "objective_count": self.objective_count,
            "subjective_count": self.subjective_count,
        }


@dataclass
class CrossReferenceReport:
    """Complete cross-reference between metrics and bundles."""

    metrics: list[MetricCrossRef] = field(default_factory=list)
    bundles: list[BundleCrossRef] = field(default_factory=list)

    @property
    def orphan_metrics(self) -> list[str]:
        """Metrics not in any bundle."""
        return [m.metric_id for m in self.metrics if not m.bundles]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": [m.as_dict() for m in self.metrics],
            "bundles": [b.as_dict() for b in self.bundles],
            "orphan_metrics": self.orphan_metrics,
        }

    def format_text(self) -> str:
        lines = ["Metric Cross-Reference"]
        lines.append(f"  Total metrics: {len(self.metrics)}")
        lines.append(f"  Total bundles: {len(self.bundles)}")
        lines.append(f"  Orphan metrics: {len(self.orphan_metrics)}")

        if self.bundles:
            lines.append("\n  Bundles:")
            for b in sorted(self.bundles, key=lambda x: x.bundle_name):
                lines.append(f"    {b.bundle_name}: {b.total_metrics} metrics "
                           f"({b.objective_count} obj, {b.subjective_count} subj)")
        return "\n".join(lines)


def build_cross_reference(
    bundles: dict[str, list[str]] | None = None,
    metric_types: dict[str, str] | None = None,
    metric_categories: dict[str, str] | None = None,
) -> CrossReferenceReport:
    """Build a cross-reference between metrics and bundles.

    Args:
        bundles: Dict of bundle_name -> list of metric_ids. If None,
            loads from the built-in BUNDLES constant.
        metric_types: Dict of metric_id -> "objective"/"subjective".
        metric_categories: Dict of metric_id -> category name.

    Returns:
        CrossReferenceReport with bidirectional mapping.
    """
    if bundles is None:
        bundles = _load_default_bundles()

    metric_types = metric_types or {}
    metric_categories = metric_categories or {}

    # Build metric -> bundles mapping
    metric_bundles: dict[str, set[str]] = {}
    for bundle_name, metric_ids in bundles.items():
        for mid in metric_ids:
            metric_bundles.setdefault(mid, set()).add(bundle_name)

    # Build metric cross-refs
    all_metrics = sorted(set(
        mid for mids in bundles.values() for mid in mids
    ) | set(metric_types.keys()))

    metrics = []
    for mid in all_metrics:
        metrics.append(MetricCrossRef(
            metric_id=mid,
            metric_type=metric_types.get(mid, "unknown"),
            bundles=sorted(metric_bundles.get(mid, set())),
            category=metric_categories.get(mid, ""),
        ))

    # Build bundle cross-refs
    bundle_refs = []
    for bname, mids in sorted(bundles.items()):
        obj_count = sum(1 for m in mids if metric_types.get(m) == "objective")
        subj_count = sum(1 for m in mids if metric_types.get(m) == "subjective")
        bundle_refs.append(BundleCrossRef(
            bundle_name=bname,
            metrics=sorted(mids),
            objective_count=obj_count,
            subjective_count=subj_count,
        ))

    return CrossReferenceReport(metrics=metrics, bundles=bundle_refs)


def _load_default_bundles() -> dict[str, list[str]]:
    """Load built-in metric bundles."""
    try:
        from ..cli.constants import BUNDLES
        return {name: list(metrics) for name, metrics in BUNDLES.items()}
    except (ImportError, AttributeError):
        return {}
