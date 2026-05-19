"""Rubric optimization: auto-generate and refine evaluation rubrics.

Provides tools to create rubrics from criteria, generate variants via
reordering/simplification/expansion, score clarity, compare variants,
and suggest improvements.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RubricCriterion:
    """A single criterion within a rubric."""

    name: str
    description: str
    weight: float = 1.0
    examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "examples": list(self.examples),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricCriterion:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            weight=data.get("weight", 1.0),
            examples=list(data.get("examples", [])),
        )


@dataclass
class Rubric:
    """An evaluation rubric consisting of weighted criteria."""

    metric_id: str
    criteria: list[RubricCriterion] = field(default_factory=list)
    version: int = 1
    preamble: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "criteria": [c.as_dict() for c in self.criteria],
            "version": self.version,
            "preamble": self.preamble,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rubric:
        criteria = [
            RubricCriterion.from_dict(c) for c in data.get("criteria", [])
        ]
        return cls(
            metric_id=data["metric_id"],
            criteria=criteria,
            version=data.get("version", 1),
            preamble=data.get("preamble", ""),
        )

    def format_text(self) -> str:
        lines = [
            f"Rubric: {self.metric_id} (v{self.version})",
        ]
        if self.preamble:
            lines.append(f"  Preamble: {self.preamble}")
        lines.append(f"  Criteria: {len(self.criteria)}")
        for i, c in enumerate(self.criteria, 1):
            lines.append(f"    {i}. {c.name} (weight={c.weight})")
            lines.append(f"       {c.description}")
            if c.examples:
                for ex in c.examples:
                    lines.append(f"       - {ex}")
        return "\n".join(lines)


@dataclass
class RubricVariant:
    """A rubric variant with an alignment score."""

    variant_id: str
    rubric: Rubric
    alignment_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "rubric": self.rubric.as_dict(),
            "alignment_score": round(self.alignment_score, 4),
        }


@dataclass
class RubricOptimizationResult:
    """Result of comparing multiple rubric variants."""

    original: Rubric
    variants: list[RubricVariant] = field(default_factory=list)
    best_variant_id: str = ""
    improvement: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original.as_dict(),
            "variants": [v.as_dict() for v in self.variants],
            "best_variant_id": self.best_variant_id,
            "improvement": round(self.improvement, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RubricOptimizationResult:
        original = Rubric.from_dict(data["original"])
        variants = []
        for v in data.get("variants", []):
            rubric = Rubric.from_dict(v["rubric"])
            variants.append(
                RubricVariant(
                    variant_id=v["variant_id"],
                    rubric=rubric,
                    alignment_score=v.get("alignment_score", 0.0),
                )
            )
        return cls(
            original=original,
            variants=variants,
            best_variant_id=data.get("best_variant_id", ""),
            improvement=data.get("improvement", 0.0),
        )

    def format_text(self) -> str:
        lines = [
            "Rubric Optimization Result",
            f"  Original:      {self.original.metric_id} (v{self.original.version})",
            f"  Variants:      {len(self.variants)}",
            f"  Best variant:  {self.best_variant_id}",
            f"  Improvement:   {self.improvement:.4f}",
        ]
        for v in self.variants:
            lines.append(f"    {v.variant_id}: alignment={v.alignment_score:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_rubric(
    metric_id: str,
    criteria: list[dict[str, Any]],
    preamble: str = "",
) -> Rubric:
    """Build a Rubric from a list of criteria dicts.

    Each dict should have at least "name" and "description" keys.
    Optional: "weight" (float), "examples" (list of strings).
    """
    parsed = [RubricCriterion.from_dict(c) for c in criteria]
    return Rubric(
        metric_id=metric_id,
        criteria=parsed,
        version=1,
        preamble=preamble,
    )


def generate_rubric_variant(rubric: Rubric, method: str = "reorder") -> Rubric:
    """Generate a rubric variant using the specified method.

    Methods:
        reorder  - shuffle criteria order
        simplify - remove the lowest-weight criterion
        expand   - add a placeholder example to each criterion
    """
    # Deep copy criteria
    new_criteria = [
        RubricCriterion(
            name=c.name,
            description=c.description,
            weight=c.weight,
            examples=list(c.examples),
        )
        for c in rubric.criteria
    ]

    if method == "reorder":
        random.shuffle(new_criteria)
    elif method == "simplify":
        if len(new_criteria) > 1:
            # Remove the criterion with the lowest weight
            min_weight = min(c.weight for c in new_criteria)
            for i, c in enumerate(new_criteria):
                if c.weight == min_weight:
                    new_criteria.pop(i)
                    break
    elif method == "expand":
        for c in new_criteria:
            c.examples.append(f"Example for {c.name}")
    else:
        pass  # unknown method - return unchanged copy

    return Rubric(
        metric_id=rubric.metric_id,
        criteria=new_criteria,
        version=rubric.version + 1,
        preamble=rubric.preamble,
    )


def score_rubric_clarity(rubric: Rubric) -> float:
    """Heuristic clarity score for a rubric (0-1).

    Based on average description length: longer, more detailed descriptions
    are considered clearer. Normalized so that 200+ character descriptions
    score 1.0.
    """
    if not rubric.criteria:
        return 0.0
    avg_len = sum(len(c.description) for c in rubric.criteria) / len(rubric.criteria)
    # Normalize: 200 chars = 1.0
    return min(avg_len / 200.0, 1.0)


def compare_rubric_variants(
    variants: list[RubricVariant],
) -> RubricOptimizationResult:
    """Find the best variant by alignment_score.

    Returns a RubricOptimizationResult. Uses the first variant's rubric
    as the "original" baseline.
    """
    if not variants:
        return RubricOptimizationResult(
            original=Rubric(metric_id="unknown"),
        )
    # Use first variant as the original baseline
    original = variants[0].rubric
    best = max(variants, key=lambda v: v.alignment_score)
    baseline_score = variants[0].alignment_score
    improvement = best.alignment_score - baseline_score
    return RubricOptimizationResult(
        original=original,
        variants=variants,
        best_variant_id=best.variant_id,
        improvement=improvement,
    )


def suggest_rubric_improvements(rubric: Rubric) -> list[str]:
    """Suggest improvements for a rubric.

    Checks for:
    - Criteria without examples
    - Short descriptions (fewer than 20 characters)
    - Empty criteria list
    """
    suggestions: list[str] = []
    if not rubric.criteria:
        suggestions.append("Rubric has no criteria - add at least one")
        return suggestions
    for c in rubric.criteria:
        if not c.examples:
            suggestions.append(f"Add examples to criterion '{c.name}'")
        if len(c.description) < 20:
            suggestions.append(f"Criterion '{c.name}' description is too short")
    return suggestions
