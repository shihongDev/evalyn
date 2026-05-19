"""Metric weighting profiles for weighted analysis.

Named weight sets that assign importance to different metrics.
Used for computing weighted pass rates and weighted overall scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WeightProfile:
    """A named set of metric weights.

    Example:
        safety_first = WeightProfile(
            name="safety-first",
            weights={"toxicity_safety": 3.0, "helpfulness": 1.0, "json_valid": 1.0},
        )
    """

    name: str
    weights: dict[str, float] = field(default_factory=dict)
    description: str = ""
    default_weight: float = 1.0  # weight for metrics not explicitly listed

    def get_weight(self, metric_id: str) -> float:
        """Get the weight for a metric, falling back to default_weight."""
        return self.weights.get(metric_id, self.default_weight)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": dict(self.weights),
            "description": self.description,
            "default_weight": self.default_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeightProfile:
        return cls(
            name=data.get("name", "custom"),
            weights=data.get("weights", {}),
            description=data.get("description", ""),
            default_weight=data.get("default_weight", 1.0),
        )


def compute_weighted_pass_rate(
    metric_pass_rates: dict[str, float],
    profile: WeightProfile,
) -> float:
    """Compute weighted overall pass rate.

    Args:
        metric_pass_rates: Dict of metric_id -> pass rate (0.0-1.0).
        profile: Weight profile to apply.

    Returns:
        Weighted average pass rate.
    """
    if not metric_pass_rates:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0

    for metric_id, rate in metric_pass_rates.items():
        w = profile.get_weight(metric_id)
        weighted_sum += rate * w
        total_weight += w

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def compute_weighted_score(
    metric_scores: dict[str, float],
    profile: WeightProfile,
) -> float:
    """Compute weighted overall score.

    Args:
        metric_scores: Dict of metric_id -> average score.
        profile: Weight profile to apply.

    Returns:
        Weighted average score.
    """
    if not metric_scores:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0

    for metric_id, score in metric_scores.items():
        w = profile.get_weight(metric_id)
        weighted_sum += score * w
        total_weight += w

    return weighted_sum / total_weight if total_weight > 0 else 0.0


# Built-in profiles
EQUAL_WEIGHTS = WeightProfile(
    name="equal",
    description="All metrics weighted equally (default)",
    default_weight=1.0,
)

SAFETY_FIRST = WeightProfile(
    name="safety-first",
    description="Safety metrics weighted 3x, others 1x",
    weights={
        "toxicity_safety": 3.0,
        "pii_safety": 3.0,
        "manipulation_resistance": 3.0,
        "bias_detection": 3.0,
        "prompt_injection_check": 3.0,
    },
    default_weight=1.0,
)

QUALITY_FOCUSED = WeightProfile(
    name="quality-focused",
    description="Quality and accuracy metrics weighted 2x",
    weights={
        "helpfulness_accuracy": 2.0,
        "factual_accuracy": 2.0,
        "technical_accuracy": 2.0,
        "reasoning_quality": 2.0,
    },
    default_weight=1.0,
)

BUILTIN_WEIGHT_PROFILES: dict[str, WeightProfile] = {
    "equal": EQUAL_WEIGHTS,
    "safety-first": SAFETY_FIRST,
    "quality-focused": QUALITY_FOCUSED,
}


def get_weight_profile(
    name: str,
    custom_profiles: dict[str, dict] | None = None,
) -> WeightProfile:
    """Get a weight profile by name.

    Args:
        name: Profile name.
        custom_profiles: Optional dict of custom profiles from config.

    Returns:
        WeightProfile instance.

    Raises:
        ValueError: If profile not found.
    """
    if custom_profiles and name in custom_profiles:
        data = custom_profiles[name]
        data["name"] = name
        return WeightProfile.from_dict(data)

    if name in BUILTIN_WEIGHT_PROFILES:
        return BUILTIN_WEIGHT_PROFILES[name]

    available = list(BUILTIN_WEIGHT_PROFILES.keys())
    if custom_profiles:
        available.extend(custom_profiles.keys())
    raise ValueError(f"Unknown weight profile '{name}'. Available: {sorted(available)}")
