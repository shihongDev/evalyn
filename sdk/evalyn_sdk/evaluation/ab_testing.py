"""Metric A/B variant testing: compare two rubric variants side by side.

Evaluate the same items with two variants of a metric (e.g., different
rubrics or prompts) and compare their agreement and score distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ABTestResult:
    """Result of comparing two metric variants."""

    variant_a_id: str
    variant_b_id: str
    item_count: int
    agreement_rate: float  # fraction of items where both variants agree
    a_pass_rate: float
    b_pass_rate: float
    a_avg_score: float
    b_avg_score: float
    disagreement_items: list[str] = field(default_factory=list)

    @property
    def winner(self) -> str | None:
        """Variant with higher pass rate, or None if tied."""
        if abs(self.a_pass_rate - self.b_pass_rate) < 0.01:
            return None
        return self.variant_a_id if self.a_pass_rate > self.b_pass_rate else self.variant_b_id

    @property
    def score_delta(self) -> float:
        return self.b_avg_score - self.a_avg_score

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_a": self.variant_a_id,
            "variant_b": self.variant_b_id,
            "item_count": self.item_count,
            "agreement_rate": round(self.agreement_rate, 4),
            "a_pass_rate": round(self.a_pass_rate, 4),
            "b_pass_rate": round(self.b_pass_rate, 4),
            "a_avg_score": round(self.a_avg_score, 4),
            "b_avg_score": round(self.b_avg_score, 4),
            "winner": self.winner,
            "disagreement_count": len(self.disagreement_items),
        }

    def format_text(self) -> str:
        lines = [
            f"A/B Test: {self.variant_a_id} vs {self.variant_b_id}",
            f"  Items:           {self.item_count}",
            f"  Agreement:       {self.agreement_rate:.1%}",
            f"  A pass rate:     {self.a_pass_rate:.1%}",
            f"  B pass rate:     {self.b_pass_rate:.1%}",
            f"  A avg score:     {self.a_avg_score:.3f}",
            f"  B avg score:     {self.b_avg_score:.3f}",
            f"  Winner:          {self.winner or 'tie'}",
            f"  Disagreements:   {len(self.disagreement_items)}",
        ]
        return "\n".join(lines)


def compare_variants(
    results_a: list,
    results_b: list,
    variant_a_id: str = "variant_a",
    variant_b_id: str = "variant_b",
) -> ABTestResult:
    """Compare two metric variants evaluated on the same items.

    Args:
        results_a: MetricResult list from variant A.
        results_b: MetricResult list from variant B.
        variant_a_id: Label for variant A.
        variant_b_id: Label for variant B.

    Returns:
        ABTestResult with agreement and score comparison.
    """
    # Index by item_id
    a_by_item = {r.item_id: r for r in results_a}
    b_by_item = {r.item_id: r for r in results_b}

    common_items = sorted(set(a_by_item.keys()) & set(b_by_item.keys()))

    if not common_items:
        return ABTestResult(
            variant_a_id=variant_a_id,
            variant_b_id=variant_b_id,
            item_count=0,
            agreement_rate=0.0,
            a_pass_rate=0.0,
            b_pass_rate=0.0,
            a_avg_score=0.0,
            b_avg_score=0.0,
        )

    agreed = 0
    disagreements = []
    a_scores = []
    b_scores = []
    a_passed = 0
    b_passed = 0
    a_total = 0
    b_total = 0

    for iid in common_items:
        ra = a_by_item[iid]
        rb = b_by_item[iid]

        if ra.passed is not None:
            a_total += 1
            if ra.passed:
                a_passed += 1
        if rb.passed is not None:
            b_total += 1
            if rb.passed:
                b_passed += 1

        if ra.score is not None:
            a_scores.append(ra.score)
        if rb.score is not None:
            b_scores.append(rb.score)

        if ra.passed == rb.passed:
            agreed += 1
        else:
            disagreements.append(iid)

    return ABTestResult(
        variant_a_id=variant_a_id,
        variant_b_id=variant_b_id,
        item_count=len(common_items),
        agreement_rate=agreed / len(common_items),
        a_pass_rate=a_passed / a_total if a_total > 0 else 0.0,
        b_pass_rate=b_passed / b_total if b_total > 0 else 0.0,
        a_avg_score=sum(a_scores) / len(a_scores) if a_scores else 0.0,
        b_avg_score=sum(b_scores) / len(b_scores) if b_scores else 0.0,
        disagreement_items=disagreements,
    )
