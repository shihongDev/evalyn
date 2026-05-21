"""Inter-rater reliability: agreement statistics across multiple judges.

Computes agreement metrics when the same items are scored by multiple
judges (models or prompts). Useful for assessing judge quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgreementStats:
    """Agreement statistics between two or more raters."""

    num_raters: int
    num_items: int
    raw_agreement: float  # fraction of items where all raters agree
    cohens_kappa: float | None = None  # for 2 raters (binary)
    fleiss_kappa: float | None = None  # for N raters (binary)
    items_with_disagreement: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "num_raters": self.num_raters,
            "num_items": self.num_items,
            "raw_agreement": round(self.raw_agreement, 4),
        }
        if self.cohens_kappa is not None:
            d["cohens_kappa"] = round(self.cohens_kappa, 4)
        if self.fleiss_kappa is not None:
            d["fleiss_kappa"] = round(self.fleiss_kappa, 4)
        d["disagreement_count"] = len(self.items_with_disagreement)
        return d

    @property
    def agreement_level(self) -> str:
        """Interpret kappa value using Landis-Koch scale."""
        kappa = self.cohens_kappa if self.cohens_kappa is not None else self.fleiss_kappa
        if kappa is None:
            return "unknown"
        if kappa >= 0.81:
            return "almost perfect"
        if kappa >= 0.61:
            return "substantial"
        if kappa >= 0.41:
            return "moderate"
        if kappa >= 0.21:
            return "fair"
        if kappa >= 0.0:
            return "slight"
        return "poor"

    def format_text(self) -> str:
        lines = [
            f"Inter-Rater Agreement ({self.num_raters} raters, {self.num_items} items)",
            f"  Raw agreement:  {self.raw_agreement:.1%}",
        ]
        if self.cohens_kappa is not None:
            lines.append(f"  Cohen's kappa:  {self.cohens_kappa:.3f} ({self.agreement_level})")
        if self.fleiss_kappa is not None:
            lines.append(f"  Fleiss' kappa:  {self.fleiss_kappa:.3f} ({self.agreement_level})")
        lines.append(f"  Disagreements:  {len(self.items_with_disagreement)} items")
        return "\n".join(lines)


def compute_agreement(
    ratings: dict[str, dict[str, bool]],
) -> AgreementStats:
    """Compute inter-rater agreement from binary ratings.

    Args:
        ratings: Dict of rater_id -> {item_id: passed_bool}.
            All raters must rate the same items.

    Returns:
        AgreementStats with raw agreement and kappa coefficients.
    """
    rater_ids = sorted(ratings.keys())
    num_raters = len(rater_ids)

    if num_raters < 2:
        return AgreementStats(num_raters=num_raters, num_items=0, raw_agreement=0.0)

    # Find common items
    all_items = set()
    for rater_ratings in ratings.values():
        all_items.update(rater_ratings.keys())

    # Only consider items rated by ALL raters
    common_items = sorted(all_items)
    for rater_id in rater_ids:
        common_items = [i for i in common_items if i in ratings[rater_id]]

    num_items = len(common_items)
    if num_items == 0:
        return AgreementStats(num_raters=num_raters, num_items=0, raw_agreement=0.0)

    # Raw agreement: fraction of items where all raters agree
    agreed = 0
    disagreement_items = []
    for item_id in common_items:
        votes = [ratings[r][item_id] for r in rater_ids]
        if len(set(votes)) == 1:
            agreed += 1
        else:
            disagreement_items.append(item_id)

    raw_agreement = agreed / num_items

    stats = AgreementStats(
        num_raters=num_raters,
        num_items=num_items,
        raw_agreement=raw_agreement,
        items_with_disagreement=disagreement_items,
    )

    if num_raters == 2:
        stats.cohens_kappa = _cohens_kappa(
            [ratings[rater_ids[0]][i] for i in common_items],
            [ratings[rater_ids[1]][i] for i in common_items],
        )
    else:
        stats.fleiss_kappa = _fleiss_kappa(
            [[ratings[r][i] for r in rater_ids] for i in common_items]
        )

    return stats


def _cohens_kappa(rater_a: list[bool], rater_b: list[bool]) -> float:
    """Compute Cohen's kappa for two binary raters."""
    n = len(rater_a)
    if n == 0:
        return 0.0

    # Confusion matrix
    a1_b1 = sum(1 for a, b in zip(rater_a, rater_b) if a and b)
    a1_b0 = sum(1 for a, b in zip(rater_a, rater_b) if a and not b)
    a0_b1 = sum(1 for a, b in zip(rater_a, rater_b) if not a and b)
    a0_b0 = sum(1 for a, b in zip(rater_a, rater_b) if not a and not b)

    # Observed agreement
    po = (a1_b1 + a0_b0) / n

    # Expected agreement by chance
    pa = (a1_b1 + a1_b0) / n  # P(A=1)
    pb = (a1_b1 + a0_b1) / n  # P(B=1)
    pe = pa * pb + (1 - pa) * (1 - pb)

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    return (po - pe) / (1 - pe)


def _fleiss_kappa(item_ratings: list[list[bool]]) -> float:
    """Compute Fleiss' kappa for N raters on binary categories."""
    n_items = len(item_ratings)
    if n_items == 0:
        return 0.0

    n_raters = len(item_ratings[0])
    if n_raters < 2:
        return 0.0

    # For binary categories: count pass votes per item
    pass_counts = [sum(1 for v in votes if v) for votes in item_ratings]
    fail_counts = [n_raters - pc for pc in pass_counts]

    # P_i: agreement per item
    sum_pi = 0
    for pc, fc in zip(pass_counts, fail_counts):
        sum_pi += (pc * (pc - 1) + fc * (fc - 1)) / (n_raters * (n_raters - 1))

    p_bar = sum_pi / n_items  # mean observed agreement

    # Category proportions
    total_votes = n_items * n_raters
    p_pass = sum(pass_counts) / total_votes
    p_fail = 1 - p_pass

    pe = p_pass**2 + p_fail**2  # expected agreement

    if pe == 1.0:
        return 1.0 if p_bar == 1.0 else 0.0

    return (p_bar - pe) / (1 - pe)
