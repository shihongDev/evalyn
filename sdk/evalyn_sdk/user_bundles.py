"""
User-defined metric bundles.

Provides a registry for grouping metrics into named bundles, with
search, composition, and recommendation capabilities. Pure Python,
no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class MetricBundle:
    """A named collection of metric IDs with descriptive metadata."""

    bundle_id: str
    name: str
    description: str
    metric_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "name": self.name,
            "description": self.description,
            "metric_ids": list(self.metric_ids),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricBundle:
        return cls(
            bundle_id=data["bundle_id"],
            name=data["name"],
            description=data.get("description", ""),
            metric_ids=list(data.get("metric_ids", [])),
            tags=list(data.get("tags", [])),
        )


# ---------------------------------------------------------------------------
# Bundle Registry
# ---------------------------------------------------------------------------


class BundleRegistry:
    """Registry for managing metric bundles."""

    def __init__(self) -> None:
        self._bundles: dict[str, MetricBundle] = {}

    def register(self, bundle: MetricBundle) -> None:
        """Add or replace a bundle in the registry."""
        self._bundles[bundle.bundle_id] = bundle

    def unregister(self, bundle_id: str) -> bool:
        """Remove a bundle by ID. Returns True if it existed."""
        if bundle_id in self._bundles:
            del self._bundles[bundle_id]
            return True
        return False

    def get(self, bundle_id: str) -> MetricBundle | None:
        """Look up a bundle by ID, or return None."""
        return self._bundles.get(bundle_id)

    def list_bundles(self) -> list[MetricBundle]:
        """Return all registered bundles, sorted by bundle_id."""
        return sorted(self._bundles.values(), key=lambda b: b.bundle_id)

    def search(self, query: str) -> list[MetricBundle]:
        """Search bundles by name, description, and tags (case-insensitive)."""
        q = query.lower()
        results: list[MetricBundle] = []
        for bundle in self._bundles.values():
            searchable = " ".join(
                [bundle.name, bundle.description] + bundle.tags
            ).lower()
            if q in searchable:
                results.append(bundle)
        return sorted(results, key=lambda b: b.bundle_id)

    def get_metrics(self, bundle_id: str) -> list[str]:
        """Return metric IDs for a bundle, or empty list if not found."""
        bundle = self._bundles.get(bundle_id)
        if bundle is None:
            return []
        return list(bundle.metric_ids)

    def compose(self, bundle_ids: list[str]) -> MetricBundle:
        """Combine multiple bundles into a new composite bundle.

        Metric IDs are deduplicated while preserving order. Tags are
        merged and deduplicated similarly.
        """
        metric_ids: list[str] = []
        tags: list[str] = []
        names: list[str] = []
        seen_metrics: set[str] = set()
        seen_tags: set[str] = set()

        for bid in bundle_ids:
            bundle = self._bundles.get(bid)
            if bundle is None:
                continue
            names.append(bundle.name)
            for mid in bundle.metric_ids:
                if mid not in seen_metrics:
                    seen_metrics.add(mid)
                    metric_ids.append(mid)
            for tag in bundle.tags:
                if tag not in seen_tags:
                    seen_tags.add(tag)
                    tags.append(tag)

        composite_id = "+".join(bundle_ids)
        composite_name = " + ".join(names) if names else "Composite"
        return MetricBundle(
            bundle_id=composite_id,
            name=composite_name,
            description=f"Composed from: {', '.join(bundle_ids)}",
            metric_ids=metric_ids,
            tags=tags,
        )

    def recommend(self, trace_patterns: list[str]) -> list[MetricBundle]:
        """Suggest bundles whose tags overlap with the given trace patterns.

        Matching is case-insensitive. Results are sorted by the number
        of matching tags (descending), then by bundle_id.
        """
        patterns_lower = {p.lower() for p in trace_patterns}
        scored: list[tuple[int, MetricBundle]] = []
        for bundle in self._bundles.values():
            matches = sum(1 for t in bundle.tags if t.lower() in patterns_lower)
            if matches > 0:
                scored.append((matches, bundle))
        scored.sort(key=lambda x: (-x[0], x[1].bundle_id))
        return [bundle for _, bundle in scored]


# ---------------------------------------------------------------------------
# Loading Helpers
# ---------------------------------------------------------------------------


def load_bundles_from_config(config: dict[str, Any]) -> BundleRegistry:
    """Load a BundleRegistry from a config dict.

    Expects a ``"bundles"`` key containing a list of bundle dicts.
    """
    registry = BundleRegistry()
    for entry in config.get("bundles", []):
        registry.register(MetricBundle.from_dict(entry))
    return registry


# ---------------------------------------------------------------------------
# Built-in Bundles
# ---------------------------------------------------------------------------


def _create_builtin_bundles() -> BundleRegistry:
    """Create the default set of built-in metric bundles."""
    registry = BundleRegistry()
    registry.register(
        MetricBundle(
            bundle_id="safety",
            name="Safety",
            description="Metrics for content safety evaluation",
            metric_ids=["toxicity", "injection", "pii"],
            tags=["safety", "moderation", "content"],
        )
    )
    registry.register(
        MetricBundle(
            bundle_id="quality",
            name="Quality",
            description="Metrics for output quality evaluation",
            metric_ids=["helpfulness", "coherence", "completeness"],
            tags=["quality", "output", "generation"],
        )
    )
    registry.register(
        MetricBundle(
            bundle_id="accuracy",
            name="Accuracy",
            description="Metrics for factual accuracy evaluation",
            metric_ids=["correctness", "hallucination", "grounding"],
            tags=["accuracy", "factual", "grounding"],
        )
    )
    registry.register(
        MetricBundle(
            bundle_id="agent",
            name="Agent",
            description="Metrics for agent behavior evaluation",
            metric_ids=["tool_call", "goal_completion", "reasoning"],
            tags=["agent", "tool", "reasoning"],
        )
    )
    return registry


BUILTIN_BUNDLES: BundleRegistry = _create_builtin_bundles()


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_bundle_list(bundles: list[MetricBundle]) -> str:
    """Format a list of bundles as a human-readable summary."""
    if not bundles:
        return "No bundles found."
    lines: list[str] = []
    for b in bundles:
        tag_str = ", ".join(b.tags) if b.tags else "none"
        lines.append(f"  [{b.bundle_id}] {b.name} - {b.description}")
        lines.append(f"    Metrics: {', '.join(b.metric_ids)}")
        lines.append(f"    Tags: {tag_str}")
    return "\n".join(lines)


def format_bundle_detail(bundle: MetricBundle) -> str:
    """Format a single bundle as a detailed view."""
    lines: list[str] = [
        f"Bundle: {bundle.name} ({bundle.bundle_id})",
        f"Description: {bundle.description}",
        f"Metrics ({len(bundle.metric_ids)}):",
    ]
    for mid in bundle.metric_ids:
        lines.append(f"  - {mid}")
    lines.append(f"Tags: {', '.join(bundle.tags) if bundle.tags else 'none'}")
    return "\n".join(lines)
