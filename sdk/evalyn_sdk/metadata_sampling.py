"""Metadata-conditional sampling for evaluation datasets.

Variable sample rates by metadata field values. Items are grouped by a
metadata field, and each group is sampled at a configurable rate.
Pure Python, no external dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SamplingRate:
    """Sampling rate for a specific metadata field value."""

    field_value: str
    rate: float  # 0.0 to 1.0
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field_value": self.field_value,
            "rate": self.rate,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SamplingRate:
        return cls(
            field_value=data["field_value"],
            rate=data["rate"],
            description=data.get("description", ""),
        )


@dataclass
class MetadataConfig:
    """Configuration for metadata-conditional sampling."""

    field_name: str
    rates: List[SamplingRate] = field(default_factory=list)
    default_rate: float = 0.5
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "rates": [r.as_dict() for r in self.rates],
            "default_rate": self.default_rate,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetadataConfig:
        return cls(
            field_name=data["field_name"],
            rates=[SamplingRate.from_dict(r) for r in data.get("rates", [])],
            default_rate=data.get("default_rate", 0.5),
            seed=data.get("seed"),
        )


@dataclass
class MetadataResult:
    """Result of a metadata-conditional sampling run."""

    selected_ids: List[str] = field(default_factory=list)
    per_value_counts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_pool: int = 0
    total_selected: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_ids": list(self.selected_ids),
            "per_value_counts": dict(self.per_value_counts),
            "total_pool": self.total_pool,
            "total_selected": self.total_selected,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetadataResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            per_value_counts=data.get("per_value_counts", {}),
            total_pool=data.get("total_pool", 0),
            total_selected=data.get("total_selected", 0),
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def group_by_metadata(
    items: list[dict], field_name: str
) -> dict[str, list[str]]:
    """Group item IDs by metadata field value.

    Each item must have an "id" key. Items missing the field are placed in
    the "_unknown" group.
    """
    groups: dict[str, list[str]] = {}
    for item in items:
        item_id = item.get("id", "")
        value = item.get(field_name, None)
        if value is None:
            key = "_unknown"
        else:
            key = str(value)
        groups.setdefault(key, []).append(item_id)
    return groups


def apply_sampling_rates(
    groups: dict[str, list[str]],
    rates: dict[str, float],
    default_rate: float = 0.5,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """For each group, randomly select rate * count items.

    Returns a dict mapping group name to sampled item IDs.
    """
    rng = random.Random(seed)
    sampled: dict[str, list[str]] = {}
    for group_name, ids in groups.items():
        rate = rates.get(group_name, default_rate)
        rate = max(0.0, min(1.0, rate))
        n = max(0, math.ceil(rate * len(ids)))
        n = min(n, len(ids))
        chosen = rng.sample(ids, n) if n > 0 else []
        sampled[group_name] = chosen
    return sampled


def run_metadata_sampling(
    items: list[dict], config: MetadataConfig
) -> MetadataResult:
    """Full metadata-conditional sampling pipeline.

    Groups items by config.field_name, applies per-value rates, and returns
    a MetadataResult with statistics.
    """
    groups = group_by_metadata(items, config.field_name)

    # Build rates lookup from config
    rates_lookup: dict[str, float] = {}
    for sr in config.rates:
        rates_lookup[sr.field_value] = sr.rate

    sampled = apply_sampling_rates(
        groups, rates_lookup, config.default_rate, config.seed
    )

    # Collect results
    all_selected: list[str] = []
    per_value_counts: dict[str, dict[str, Any]] = {}
    for group_name, ids in groups.items():
        chosen = sampled.get(group_name, [])
        all_selected.extend(chosen)
        rate = rates_lookup.get(group_name, config.default_rate)
        per_value_counts[group_name] = {
            "total": len(ids),
            "sampled": len(chosen),
            "rate": rate,
        }

    return MetadataResult(
        selected_ids=all_selected,
        per_value_counts=per_value_counts,
        total_pool=len(items),
        total_selected=len(all_selected),
    )


def compute_actual_rates(result: MetadataResult) -> dict[str, float]:
    """Compute actual sampling rate achieved per metadata value."""
    actual: dict[str, float] = {}
    for value, counts in result.per_value_counts.items():
        total = counts["total"]
        sampled = counts["sampled"]
        actual[value] = sampled / total if total > 0 else 0.0
    return actual


def format_metadata_report(result: MetadataResult) -> str:
    """Format a metadata sampling result as a human-readable report."""
    lines: list[str] = []
    lines.append(
        f"Metadata Sampling Report "
        f"(pool={result.total_pool}, selected={result.total_selected})"
    )
    lines.append("-" * 64)
    header = (
        f"{'Value':<20} {'Total':>8} {'Sampled':>8} "
        f"{'Requested':>10} {'Actual':>10}"
    )
    lines.append(header)
    lines.append("-" * 64)

    actual_rates = compute_actual_rates(result)
    for value in sorted(result.per_value_counts.keys()):
        counts = result.per_value_counts[value]
        actual = actual_rates.get(value, 0.0)
        row = (
            f"{value:<20} {counts['total']:>8} {counts['sampled']:>8} "
            f"{counts['rate']:>10.2%} {actual:>10.2%}"
        )
        lines.append(row)

    lines.append("-" * 64)
    return "\n".join(lines)
