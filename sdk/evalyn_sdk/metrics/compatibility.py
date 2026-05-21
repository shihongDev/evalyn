"""Metric compatibility matrix: show which metrics work with which unit types.

Maps metrics to their supported evaluation unit types (outcome, single_turn,
tool_use, multi_turn, custom) and warns when selected metrics are
incompatible with the trace structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompatibilityEntry:
    """Compatibility info for a single metric."""

    metric_id: str
    metric_type: str
    unit_types: list[str] = field(default_factory=lambda: ["outcome"])

    def supports(self, unit_type: str) -> bool:
        return unit_type in self.unit_types

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "type": self.metric_type,
            "unit_types": list(self.unit_types),
        }


@dataclass
class CompatibilityMatrix:
    """Matrix of metrics vs unit types."""

    entries: list[CompatibilityEntry] = field(default_factory=list)
    all_unit_types: list[str] = field(
        default_factory=lambda: ["outcome", "single_turn", "tool_use", "multi_turn", "custom"]
    )

    def get_compatible(self, unit_type: str) -> list[str]:
        """Get metric IDs compatible with a unit type."""
        return [e.metric_id for e in self.entries if e.supports(unit_type)]

    def get_incompatible(self, unit_type: str) -> list[str]:
        """Get metric IDs NOT compatible with a unit type."""
        return [e.metric_id for e in self.entries if not e.supports(unit_type)]

    def check_selection(self, metric_ids: list[str], unit_type: str) -> dict[str, Any]:
        """Check if selected metrics are compatible with a unit type.

        Returns dict with compatible/incompatible lists and warnings.
        """
        entry_map = {e.metric_id: e for e in self.entries}
        compatible = []
        incompatible = []
        unknown = []

        for mid in metric_ids:
            entry = entry_map.get(mid)
            if entry is None:
                unknown.append(mid)
            elif entry.supports(unit_type):
                compatible.append(mid)
            else:
                incompatible.append(mid)

        warnings = []
        if incompatible:
            warnings.append(
                f"{len(incompatible)} metrics incompatible with unit type '{unit_type}': "
                f"{', '.join(incompatible)}"
            )

        return {
            "compatible": compatible,
            "incompatible": incompatible,
            "unknown": unknown,
            "warnings": warnings,
            "all_compatible": len(incompatible) == 0,
        }

    def as_dict(self) -> dict[str, Any]:
        # Build matrix: metric_id -> {unit_type: bool}
        matrix = {}
        for e in self.entries:
            matrix[e.metric_id] = {ut: e.supports(ut) for ut in self.all_unit_types}
        return {"matrix": matrix, "unit_types": self.all_unit_types}

    def format_text(self) -> str:
        lines = ["Metric Compatibility Matrix:"]
        header = f"  {'Metric':<30}" + "".join(f" {ut:<12}" for ut in self.all_unit_types)
        lines.append(header)
        lines.append("  " + "-" * (30 + 12 * len(self.all_unit_types)))
        for e in sorted(self.entries, key=lambda x: x.metric_id):
            row = f"  {e.metric_id:<30}"
            for ut in self.all_unit_types:
                row += f" {'Y':<12}" if e.supports(ut) else f" {'-':<12}"
            lines.append(row)
        return "\n".join(lines)


def build_compatibility_matrix(metric_specs: list) -> CompatibilityMatrix:
    """Build a compatibility matrix from metric specs.

    Args:
        metric_specs: List of MetricSpec objects.

    Returns:
        CompatibilityMatrix with per-metric unit type support.
    """
    entries = []
    for spec in metric_specs:
        entries.append(
            CompatibilityEntry(
                metric_id=spec.id,
                metric_type=spec.type,
                unit_types=list(getattr(spec, "unit_types", ["outcome"])),
            )
        )
    return CompatibilityMatrix(entries=entries)
