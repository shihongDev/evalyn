"""Auto model selection: choose judge model based on task complexity.

Routes evaluation items to cheaper/faster models when complexity is low,
reserving expensive models for hard cases. Reduces cost without sacrificing
quality on difficult items.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelTier:
    """A model tier with cost and complexity ceiling."""

    name: str
    model: str
    provider: str = "gemini"
    cost_per_1k: float = 0.0
    max_complexity: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "provider": self.provider,
            "cost_per_1k": self.cost_per_1k,
            "max_complexity": self.max_complexity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelTier:
        return cls(
            name=d["name"],
            model=d["model"],
            provider=d.get("provider", "gemini"),
            cost_per_1k=d.get("cost_per_1k", 0.0),
            max_complexity=d.get("max_complexity", 1.0),
        )


@dataclass
class SelectionResult:
    """Result of model selection for a single item."""

    item_id: str
    selected_model: str
    selected_tier: str
    complexity: float
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "selected_model": self.selected_model,
            "selected_tier": self.selected_tier,
            "complexity": self.complexity,
            "reason": self.reason,
        }


@dataclass
class AutoSelectionConfig:
    """Configuration for automatic model selection."""

    tiers: list[ModelTier] = field(default_factory=list)
    complexity_threshold: float = 0.5
    default_tier: str = "fast"

    def as_dict(self) -> dict[str, Any]:
        return {
            "tiers": [t.as_dict() for t in self.tiers],
            "complexity_threshold": self.complexity_threshold,
            "default_tier": self.default_tier,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AutoSelectionConfig:
        return cls(
            tiers=[ModelTier.from_dict(t) for t in d.get("tiers", [])],
            complexity_threshold=d.get("complexity_threshold", 0.5),
            default_tier=d.get("default_tier", "fast"),
        )


@dataclass
class SelectionReport:
    """Aggregated report of model selections for a batch."""

    selections: list[SelectionResult] = field(default_factory=list)
    tier_distribution: dict[str, int] = field(default_factory=dict)
    estimated_cost_savings_pct: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selections": [s.as_dict() for s in self.selections],
            "tier_distribution": dict(self.tier_distribution),
            "estimated_cost_savings_pct": self.estimated_cost_savings_pct,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SelectionReport:
        return cls(
            selections=[
                SelectionResult(**s) for s in d.get("selections", [])
            ],
            tier_distribution=d.get("tier_distribution", {}),
            estimated_cost_savings_pct=d.get("estimated_cost_savings_pct", 0.0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Model Selection Report")
        lines.append("-" * 40)
        total = len(self.selections)
        lines.append(f"Total items: {total}")
        if self.tier_distribution:
            lines.append("Tier distribution:")
            for tier_name, count in sorted(self.tier_distribution.items()):
                pct = (count / total * 100) if total > 0 else 0.0
                lines.append(f"  {tier_name}: {count} ({pct:.1f}%)")
        lines.append(
            f"Estimated cost savings: {self.estimated_cost_savings_pct:.1f}%"
        )
        return "\n".join(lines)


DEFAULT_TIERS = [
    ModelTier("fast", "gemini-2.5-flash", "gemini", 0.00015, 0.5),
    ModelTier("balanced", "gemini-2.5-pro", "gemini", 0.00125, 0.8),
    ModelTier("smart", "gemini-2.5-pro", "gemini", 0.00125, 1.0),
]


def estimate_complexity(
    text: str, avg_score: float | None = None
) -> float:
    """Heuristic complexity estimate from text length and optional score.

    Short simple text yields low complexity. Long text or low average
    scores push complexity higher. Returns a value in [0, 1].
    """
    # Length component: logarithmic scaling, saturates around 2000 chars
    length = len(text)
    if length == 0:
        length_score = 0.0
    else:
        length_score = min(1.0, math.log1p(length) / math.log1p(2000))

    # Score component: low scores indicate harder items
    if avg_score is not None:
        score_component = 1.0 - max(0.0, min(1.0, avg_score))
    else:
        score_component = 0.0

    # Blend: 60% length, 40% score (when score available)
    if avg_score is not None:
        complexity = 0.6 * length_score + 0.4 * score_component
    else:
        complexity = length_score

    return round(min(1.0, max(0.0, complexity)), 4)


def select_model(
    complexity: float, config: AutoSelectionConfig
) -> ModelTier:
    """Pick the cheapest tier whose max_complexity covers the given value."""
    sorted_tiers = sorted(config.tiers, key=lambda t: t.max_complexity)
    for tier in sorted_tiers:
        if complexity <= tier.max_complexity:
            return tier
    # Fall back to the highest tier
    if sorted_tiers:
        return sorted_tiers[-1]
    # No tiers configured - return a minimal default
    return ModelTier(config.default_tier, "unknown", "unknown")


def select_models_batch(
    items: list[dict[str, Any]],
    config: AutoSelectionConfig | None = None,
) -> SelectionReport:
    """Batch model selection. Each item dict has 'id', 'text', optionally 'avg_score'."""
    if config is None:
        config = get_default_config()

    selections: list[SelectionResult] = []
    tier_counts: dict[str, int] = {}

    for item in items:
        item_id = item.get("id", "")
        text = item.get("text", "")
        avg_score = item.get("avg_score")

        complexity = estimate_complexity(text, avg_score)
        tier = select_model(complexity, config)

        reason_parts: list[str] = []
        reason_parts.append(f"complexity={complexity:.4f}")
        if avg_score is not None:
            reason_parts.append(f"avg_score={avg_score}")
        reason = ", ".join(reason_parts)

        selections.append(
            SelectionResult(
                item_id=item_id,
                selected_model=tier.model,
                selected_tier=tier.name,
                complexity=complexity,
                reason=reason,
            )
        )
        tier_counts[tier.name] = tier_counts.get(tier.name, 0) + 1

    report = SelectionReport(
        selections=selections,
        tier_distribution=tier_counts,
    )
    report.estimated_cost_savings_pct = estimate_cost_savings(report)
    return report


def get_default_config() -> AutoSelectionConfig:
    """Return a config with the built-in DEFAULT_TIERS."""
    return AutoSelectionConfig(
        tiers=list(DEFAULT_TIERS),
        complexity_threshold=0.5,
        default_tier="fast",
    )


def estimate_cost_savings(
    report: SelectionReport, uniform_model_cost: float = 0.00125
) -> float:
    """Compare selected tier costs vs uniform smart-model cost.

    Returns savings as a percentage (0-100).
    """
    if not report.selections:
        return 0.0

    config = get_default_config()
    tier_cost_map: dict[str, float] = {t.name: t.cost_per_1k for t in config.tiers}

    uniform_total = len(report.selections) * uniform_model_cost
    selected_total = 0.0
    for sel in report.selections:
        cost = tier_cost_map.get(sel.selected_tier, uniform_model_cost)
        selected_total += cost

    if uniform_total <= 0:
        return 0.0

    savings = (1.0 - selected_total / uniform_total) * 100.0
    return round(max(0.0, savings), 2)
