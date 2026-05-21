"""
Metric interaction effects: detect non-linear interactions between metrics.

Identifies synergy (metrics that pass together more than expected),
conflict (metrics that interfere with each other), and independence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class InteractionEffect:
    """Interaction between two metrics."""

    metric_a: str
    metric_b: str
    interaction_type: str = "synergy"  # "synergy" / "conflict" / "independent"
    strength: float = 0.0
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_a": self.metric_a,
            "metric_b": self.metric_b,
            "interaction_type": self.interaction_type,
            "strength": self.strength,
            "description": self.description,
        }


@dataclass
class InteractionReport:
    """Report of all pairwise interactions."""

    effects: list[InteractionEffect] = field(default_factory=list)
    total_pairs: int = 0
    synergies: int = 0
    conflicts: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "effects": [e.as_dict() for e in self.effects],
            "total_pairs": self.total_pairs,
            "synergies": self.synergies,
            "conflicts": self.conflicts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractionReport:
        effects = []
        for e in data.get("effects", []):
            effects.append(
                InteractionEffect(
                    metric_a=e["metric_a"],
                    metric_b=e["metric_b"],
                    interaction_type=e.get("interaction_type", "synergy"),
                    strength=e.get("strength", 0.0),
                    description=e.get("description", ""),
                )
            )
        return cls(
            effects=effects,
            total_pairs=data.get("total_pairs", 0),
            synergies=data.get("synergies", 0),
            conflicts=data.get("conflicts", 0),
        )

    def format_text(self) -> str:
        lines = [
            f"Interaction Report: {self.total_pairs} pairs analyzed",
            f"  Synergies: {self.synergies}",
            f"  Conflicts: {self.conflicts}",
        ]
        for e in self.effects:
            tag = e.interaction_type.upper()
            lines.append(
                f"  [{tag}] {e.metric_a} x {e.metric_b} "
                f"(strength={e.strength:.4f}) - {e.description}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_joint_pass_rate(scores_a: list[bool], scores_b: list[bool]) -> float:
    """Fraction of items where both A and B pass."""
    if not scores_a or not scores_b:
        return 0.0
    n = min(len(scores_a), len(scores_b))
    both = sum(1 for i in range(n) if scores_a[i] and scores_b[i])
    return both / n


def compute_conditional_pass_rate(scores_a: list[bool], scores_b: list[bool]) -> float:
    """P(B passes | A passes). Returns 0.0 if A never passes."""
    if not scores_a or not scores_b:
        return 0.0
    n = min(len(scores_a), len(scores_b))
    a_pass = sum(1 for i in range(n) if scores_a[i])
    if a_pass == 0:
        return 0.0
    both = sum(1 for i in range(n) if scores_a[i] and scores_b[i])
    return both / a_pass


def _pass_rate(scores: list[bool]) -> float:
    if not scores:
        return 0.0
    return sum(1 for s in scores if s) / len(scores)


def detect_interaction(
    metric_a: str,
    scores_a: list[bool],
    metric_b: str,
    scores_b: list[bool],
) -> InteractionEffect:
    """Classify the interaction between two metrics.

    - Synergy: joint pass rate > product of individual rates
    - Conflict: conditional pass rate (B|A) < individual rate of B
    - Otherwise: independent
    """
    rate_a = _pass_rate(scores_a)
    rate_b = _pass_rate(scores_b)
    joint = compute_joint_pass_rate(scores_a, scores_b)
    expected_independent = rate_a * rate_b
    conditional_b_given_a = compute_conditional_pass_rate(scores_a, scores_b)

    if joint > expected_independent:
        strength = joint - expected_independent
        return InteractionEffect(
            metric_a=metric_a,
            metric_b=metric_b,
            interaction_type="synergy",
            strength=strength,
            description=(
                f"Joint pass rate ({joint:.4f}) exceeds independent expectation "
                f"({expected_independent:.4f})"
            ),
        )

    if conditional_b_given_a < rate_b:
        strength = rate_b - conditional_b_given_a
        return InteractionEffect(
            metric_a=metric_a,
            metric_b=metric_b,
            interaction_type="conflict",
            strength=strength,
            description=(f"P(B|A) ({conditional_b_given_a:.4f}) is lower than P(B) ({rate_b:.4f})"),
        )

    return InteractionEffect(
        metric_a=metric_a,
        metric_b=metric_b,
        interaction_type="independent",
        strength=0.0,
        description="Metrics appear independent",
    )


def detect_all_interactions(
    metric_scores: dict[str, list[bool]],
) -> InteractionReport:
    """Run pairwise interaction detection across all metrics."""
    keys = sorted(metric_scores.keys())
    effects: list[InteractionEffect] = []
    synergies = 0
    conflicts = 0

    for a, b in combinations(keys, 2):
        effect = detect_interaction(a, metric_scores[a], b, metric_scores[b])
        effects.append(effect)
        if effect.interaction_type == "synergy":
            synergies += 1
        elif effect.interaction_type == "conflict":
            conflicts += 1

    return InteractionReport(
        effects=effects,
        total_pairs=len(effects),
        synergies=synergies,
        conflicts=conflicts,
    )


def find_strongest_synergies(report: InteractionReport, top_n: int = 5) -> list[InteractionEffect]:
    """Return the top N synergies by strength."""
    synergy_effects = [e for e in report.effects if e.interaction_type == "synergy"]
    synergy_effects.sort(key=lambda e: e.strength, reverse=True)
    return synergy_effects[:top_n]


def find_strongest_conflicts(report: InteractionReport, top_n: int = 5) -> list[InteractionEffect]:
    """Return the top N conflicts by strength."""
    conflict_effects = [e for e in report.effects if e.interaction_type == "conflict"]
    conflict_effects.sort(key=lambda e: e.strength, reverse=True)
    return conflict_effects[:top_n]
