"""
Cascade model routing: route evaluation items to cheap or expensive model
tiers based on estimated difficulty. Pure Python, no external dependencies.
No actual API calls - this is a routing framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ModelTier:
    """A model tier with associated cost and quality ceiling."""

    tier_name: str  # "fast", "standard", "premium"
    model_id: str
    cost_per_1k_tokens: float
    quality_ceiling: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tier_name": self.tier_name,
            "model_id": self.model_id,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "quality_ceiling": self.quality_ceiling,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelTier:
        return cls(
            tier_name=data["tier_name"],
            model_id=data["model_id"],
            cost_per_1k_tokens=data["cost_per_1k_tokens"],
            quality_ceiling=data.get("quality_ceiling", 1.0),
        )


@dataclass
class RoutingDecision:
    """Routing decision for a single item."""

    item_id: str
    assigned_tier: str
    difficulty_score: float
    confidence: float
    escalated: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "assigned_tier": self.assigned_tier,
            "difficulty_score": self.difficulty_score,
            "confidence": self.confidence,
            "escalated": self.escalated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RoutingDecision:
        return cls(
            item_id=data["item_id"],
            assigned_tier=data["assigned_tier"],
            difficulty_score=data["difficulty_score"],
            confidence=data["confidence"],
            escalated=data.get("escalated", False),
        )


@dataclass
class CascadeConfig:
    """Configuration for cascade routing."""

    tiers: List[ModelTier]
    difficulty_thresholds: List[float]  # boundaries between tiers
    escalation_threshold: float = 0.3  # escalate if quality below this

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tiers": [t.as_dict() for t in self.tiers],
            "difficulty_thresholds": list(self.difficulty_thresholds),
            "escalation_threshold": self.escalation_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CascadeConfig:
        return cls(
            tiers=[ModelTier.from_dict(d) for d in data["tiers"]],
            difficulty_thresholds=list(data["difficulty_thresholds"]),
            escalation_threshold=data.get("escalation_threshold", 0.3),
        )


@dataclass
class CascadeReport:
    """Summary of cascade routing for a batch."""

    decisions: List[RoutingDecision]
    tier_distribution: Dict[str, int]
    estimated_cost_savings: float  # 0-1 range
    total_items: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decisions": [d.as_dict() for d in self.decisions],
            "tier_distribution": dict(self.tier_distribution),
            "estimated_cost_savings": self.estimated_cost_savings,
            "total_items": self.total_items,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CascadeReport:
        return cls(
            decisions=[RoutingDecision.from_dict(d) for d in data["decisions"]],
            tier_distribution=dict(data["tier_distribution"]),
            estimated_cost_savings=data["estimated_cost_savings"],
            total_items=data["total_items"],
        )


# ---------------------------------------------------------------------------
# Default Tiers
# ---------------------------------------------------------------------------

DEFAULT_TIERS: List[ModelTier] = [
    ModelTier(
        tier_name="fast",
        model_id="gpt-4o-mini",
        cost_per_1k_tokens=0.075,
        quality_ceiling=0.7,
    ),
    ModelTier(
        tier_name="standard",
        model_id="gpt-4o",
        cost_per_1k_tokens=0.30,
        quality_ceiling=0.9,
    ),
    ModelTier(
        tier_name="premium",
        model_id="o3",
        cost_per_1k_tokens=3.00,
        quality_ceiling=1.0,
    ),
]


def default_config() -> CascadeConfig:
    """Return a default cascade config with 3 tiers."""
    return CascadeConfig(
        tiers=list(DEFAULT_TIERS),
        difficulty_thresholds=[0.35, 0.70],
        escalation_threshold=0.3,
    )


# ---------------------------------------------------------------------------
# Difficulty Estimation
# ---------------------------------------------------------------------------

# Question words and complexity markers
_COMPLEX_MARKERS = [
    "explain", "compare", "contrast", "analyze", "evaluate",
    "justify", "critique", "synthesize", "design", "prove",
    "derive", "optimize", "implement", "debug", "refactor",
]

_NEGATIONS = ["not", "no", "never", "neither", "nor", "don't", "doesn't", "won't", "can't", "shouldn't"]


def estimate_difficulty(text: str) -> float:
    """
    Heuristic difficulty estimate for a text item. Returns 0-1.

    Components:
    - Word count (longer = harder, sigmoid around 50 words)
    - Average word length (proxy for vocabulary complexity)
    - Question complexity (presence of complex task verbs)
    - Negation count (negations increase difficulty)
    """
    if not text or not text.strip():
        return 0.0

    words = text.lower().split()
    word_count = len(words)

    # Word count signal: sigmoid centered at 50 words
    wc_score = 1.0 / (1.0 + math.exp(-0.06 * (word_count - 50)))

    # Average word length: normalize around 4-8 chars
    if words:
        avg_len = sum(len(w) for w in words) / len(words)
        len_score = min(max((avg_len - 3.0) / 5.0, 0.0), 1.0)
    else:
        len_score = 0.0

    # Complex task markers
    marker_count = sum(1 for w in words if w in _COMPLEX_MARKERS)
    marker_score = min(marker_count / 3.0, 1.0)

    # Negation count
    negation_count = sum(1 for w in words if w in _NEGATIONS)
    neg_score = min(negation_count / 3.0, 1.0)

    # Weighted combination
    raw = 0.35 * wc_score + 0.25 * len_score + 0.25 * marker_score + 0.15 * neg_score
    return min(max(raw, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_item(item_id: str, text: str, config: CascadeConfig) -> RoutingDecision:
    """Assign an item to a tier based on difficulty vs thresholds."""
    difficulty = estimate_difficulty(text)

    # Determine tier index: difficulty below first threshold -> tier 0, etc.
    tier_idx = 0
    for threshold in config.difficulty_thresholds:
        if difficulty >= threshold:
            tier_idx += 1
        else:
            break

    # Clamp to valid tier range
    tier_idx = min(tier_idx, len(config.tiers) - 1)
    tier = config.tiers[tier_idx]

    # Confidence: how far from the nearest threshold boundary (higher = more confident)
    if not config.difficulty_thresholds:
        confidence = 1.0
    else:
        distances = [abs(difficulty - t) for t in config.difficulty_thresholds]
        min_dist = min(distances)
        confidence = min(min_dist / 0.5, 1.0)  # max confidence at 0.5 away from boundary

    return RoutingDecision(
        item_id=item_id,
        assigned_tier=tier.tier_name,
        difficulty_score=difficulty,
        confidence=confidence,
    )


def route_batch(
    items: list[dict],
    config: CascadeConfig,
    text_field: str = "input",
) -> CascadeReport:
    """Route a batch of items and produce a report."""
    decisions: list[RoutingDecision] = []
    tier_dist: Dict[str, int] = {}

    for i, item in enumerate(items):
        item_id = item.get("id", str(i))
        text = item.get(text_field, "")
        decision = route_item(item_id, text, config)
        decisions.append(decision)
        tier_dist[decision.assigned_tier] = tier_dist.get(decision.assigned_tier, 0) + 1

    # Compute cost savings
    if decisions:
        tier_map = {t.tier_name: t for t in config.tiers}
        premium_cost = max(t.cost_per_1k_tokens for t in config.tiers)
        baseline = premium_cost * len(decisions)
        actual = sum(
            tier_map.get(d.assigned_tier, config.tiers[-1]).cost_per_1k_tokens
            for d in decisions
        )
        savings = 1.0 - (actual / baseline) if baseline > 0 else 0.0
    else:
        savings = 0.0

    return CascadeReport(
        decisions=decisions,
        tier_distribution=tier_dist,
        estimated_cost_savings=savings,
        total_items=len(decisions),
    )


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


def should_escalate(
    decision: RoutingDecision,
    quality_score: float,
    config: CascadeConfig,
) -> bool:
    """Return True if quality is below the escalation threshold."""
    return quality_score < config.escalation_threshold


# ---------------------------------------------------------------------------
# Cost Savings
# ---------------------------------------------------------------------------


def compute_cost_savings(report: CascadeReport, baseline_cost: float) -> float:
    """
    Compute savings ratio vs a given baseline cost.

    Returns a float in 0-1 range representing fraction saved.
    """
    if baseline_cost <= 0:
        return 0.0
    # Estimate actual cost from tier distribution
    # This is a simplified version - in production you'd use actual token counts
    return report.estimated_cost_savings


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_cascade_report(report: CascadeReport) -> str:
    """Human-readable cascade routing report."""
    lines = [
        "Cascade Routing Report",
        "=" * 40,
        f"Total items:       {report.total_items}",
        f"Cost savings:      {report.estimated_cost_savings:.1%}",
        "",
        "Tier Distribution:",
    ]

    for tier_name, count in sorted(report.tier_distribution.items()):
        pct = count / max(report.total_items, 1) * 100
        lines.append(f"  {tier_name:12s} {count:4d} ({pct:.1f}%)")

    lines.append("")
    lines.append("Decisions:")
    for d in report.decisions:
        esc = " [ESCALATED]" if d.escalated else ""
        lines.append(
            f"  {d.item_id}: tier={d.assigned_tier} "
            f"difficulty={d.difficulty_score:.3f} "
            f"confidence={d.confidence:.3f}{esc}"
        )

    return "\n".join(lines)
