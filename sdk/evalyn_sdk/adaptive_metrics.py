"""Reference-adaptive metrics: auto-switch metric rubric based on reference presence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Default fallback mappings
# ---------------------------------------------------------------------------

DEFAULT_FALLBACKS: Dict[str, str] = {
    "exact_match": "semantic_similarity",
    "bleu": "coherence",
    "rouge": "completeness",
    "string_distance": "relevance",
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class MetricMode:
    """Resolved metric mode for a single metric on a single item."""

    mode: str  # "reference" or "no_reference"
    metric_id: str
    rubric_variant: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "metric_id": self.metric_id,
            "rubric_variant": self.rubric_variant,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MetricMode:
        return cls(
            mode=data["mode"],
            metric_id=data["metric_id"],
            rubric_variant=data.get("rubric_variant", ""),
        )


@dataclass
class AdaptiveConfig:
    """Configuration for reference-adaptive metric selection."""

    reference_field: str = "expected_output"
    fallback_metrics: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reference_field": self.reference_field,
            "fallback_metrics": dict(self.fallback_metrics),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdaptiveConfig:
        return cls(
            reference_field=data.get("reference_field", "expected_output"),
            fallback_metrics=data.get("fallback_metrics", {}),
        )


@dataclass
class AdaptiveResult:
    """Result of adaptive metric resolution for one metric on one item."""

    item_id: str
    mode_used: str
    metric_id: str
    original_metric: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "mode_used": self.mode_used,
            "metric_id": self.metric_id,
            "original_metric": self.original_metric,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdaptiveResult:
        return cls(
            item_id=data["item_id"],
            mode_used=data["mode_used"],
            metric_id=data["metric_id"],
            original_metric=data["original_metric"],
        )


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------


def has_reference(item: dict, reference_field: str = "expected_output") -> bool:
    """Return True if item has a non-empty reference field."""
    value = item.get(reference_field)
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def select_metric_mode(
    item: dict, metric_id: str, config: AdaptiveConfig
) -> MetricMode:
    """Select reference or no_reference mode for a metric on an item.

    If the item has a reference, use "reference" mode with the original metric.
    Otherwise, use "no_reference" mode and apply fallback if configured.
    """
    if has_reference(item, config.reference_field):
        return MetricMode(mode="reference", metric_id=metric_id)
    fallback = config.fallback_metrics.get(metric_id, metric_id)
    variant = "fallback" if fallback != metric_id else ""
    return MetricMode(mode="no_reference", metric_id=fallback, rubric_variant=variant)


def adapt_metrics_for_item(
    item: dict, metric_ids: list[str], config: AdaptiveConfig
) -> list[AdaptiveResult]:
    """Resolve adaptive mode for each metric on a single item."""
    item_id = str(item.get("id", item.get("item_id", "")))
    results: list[AdaptiveResult] = []
    for mid in metric_ids:
        mode = select_metric_mode(item, mid, config)
        results.append(
            AdaptiveResult(
                item_id=item_id,
                mode_used=mode.mode,
                metric_id=mode.metric_id,
                original_metric=mid,
            )
        )
    return results


def adapt_metrics_for_dataset(
    items: list[dict],
    metric_ids: list[str],
    config: AdaptiveConfig | None = None,
) -> dict[str, list[AdaptiveResult]]:
    """Resolve adaptive metrics for every item in the dataset.

    Returns a mapping of item_id to list of AdaptiveResult.
    If config is None, uses default config with DEFAULT_FALLBACKS.
    """
    if config is None:
        config = AdaptiveConfig(fallback_metrics=dict(DEFAULT_FALLBACKS))
    results: dict[str, list[AdaptiveResult]] = {}
    for item in items:
        item_results = adapt_metrics_for_item(item, metric_ids, config)
        item_id = item_results[0].item_id if item_results else str(
            item.get("id", item.get("item_id", ""))
        )
        results[item_id] = item_results
    return results


def compute_adaptation_stats(results: dict[str, list[AdaptiveResult]]) -> dict:
    """Compute summary statistics from adaptation results.

    Returns dict with: reference_items, no_reference_items, metrics_adapted, fallbacks_used.
    """
    reference_items = 0
    no_reference_items = 0
    metrics_adapted = 0
    fallbacks_used: dict[str, str] = {}

    seen_items_ref: set[str] = set()
    seen_items_noref: set[str] = set()

    for item_id, item_results in results.items():
        for r in item_results:
            if r.mode_used == "reference":
                seen_items_ref.add(r.item_id)
            else:
                seen_items_noref.add(r.item_id)
                metrics_adapted += 1
                if r.metric_id != r.original_metric:
                    fallbacks_used[r.original_metric] = r.metric_id

    reference_items = len(seen_items_ref - seen_items_noref)
    no_reference_items = len(seen_items_noref - seen_items_ref)
    # Items that have both modes (some metrics ref, some not) count based on majority
    mixed = seen_items_ref & seen_items_noref
    reference_items += len(mixed)

    return {
        "reference_items": reference_items,
        "no_reference_items": no_reference_items,
        "metrics_adapted": metrics_adapted,
        "fallbacks_used": fallbacks_used,
    }


def format_adaptation_report(stats: dict) -> str:
    """Format adaptation statistics as a human-readable report."""
    lines = [
        "Adaptive Metrics Report",
        "-" * 40,
        f"Reference items:     {stats.get('reference_items', 0)}",
        f"No-reference items:  {stats.get('no_reference_items', 0)}",
        f"Metrics adapted:     {stats.get('metrics_adapted', 0)}",
    ]
    fallbacks = stats.get("fallbacks_used", {})
    if fallbacks:
        lines.append("")
        lines.append("Fallbacks applied:")
        for original, replacement in sorted(fallbacks.items()):
            lines.append(f"  {original} -> {replacement}")
    return "\n".join(lines)
