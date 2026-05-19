"""Human disagreement sampling: prioritize items where annotators disagree.

Computes per-item disagreement from annotation labels, weights by severity
of disagreement (binary flips vs score differences), and selects the most
contentious items for re-annotation. Pure Python, no external dependencies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DisagreementScore:
    """Disagreement score for a single item."""

    item_id: str
    score: float  # 0 to 1, higher = more disagreement
    n_annotators: int
    label_spread: float  # max - min label

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "score": self.score,
            "n_annotators": self.n_annotators,
            "label_spread": self.label_spread,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DisagreementScore:
        return cls(
            item_id=data["item_id"],
            score=data["score"],
            n_annotators=data["n_annotators"],
            label_spread=data["label_spread"],
        )


@dataclass
class DisagreementConfig:
    """Configuration for disagreement sampling."""

    sample_size: int = 50
    min_annotators: int = 2
    severity_weight: float = 1.0  # boost binary flips vs score differences
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "min_annotators": self.min_annotators,
            "severity_weight": self.severity_weight,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DisagreementConfig:
        return cls(
            sample_size=data.get("sample_size", 50),
            min_annotators=data.get("min_annotators", 2),
            severity_weight=data.get("severity_weight", 1.0),
            seed=data.get("seed"),
        )


@dataclass
class DisagreementResult:
    """Result of a disagreement sampling run."""

    selected_ids: list[str] = field(default_factory=list)
    scores: list[DisagreementScore] = field(default_factory=list)
    mean_disagreement: float = 0.0
    total_pool: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "scores": [s.as_dict() for s in self.scores],
            "mean_disagreement": self.mean_disagreement,
            "total_pool": self.total_pool,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DisagreementResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            scores=[
                DisagreementScore.from_dict(s) for s in data.get("scores", [])
            ],
            mean_disagreement=data.get("mean_disagreement", 0.0),
            total_pool=data.get("total_pool", 0),
        )


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def compute_disagreement(labels: list[float]) -> float:
    """Standard deviation of labels, normalized to 0-1 range.

    Divides by the maximum possible std for the label range. Returns 0.0
    for fewer than 2 labels or when all labels are identical.
    """
    if len(labels) < 2:
        return 0.0

    lo = min(labels)
    hi = max(labels)
    label_range = hi - lo
    if label_range == 0.0:
        return 0.0

    mean = sum(labels) / len(labels)
    variance = sum((x - mean) ** 2 for x in labels) / len(labels)
    std = math.sqrt(variance)

    # Max possible std for n values spanning [lo, hi] is range/2
    # (half at lo, half at hi). Normalize by that.
    max_std = label_range / 2.0
    return min(std / max_std, 1.0)


def compute_severity(labels: list[float], threshold: float = 0.5) -> float:
    """Severity of annotator disagreement.

    Returns 1.0 if any annotator pair has a binary flip (one above, one
    below threshold), 0.5 for moderate spread (all on same side but spread
    exceeds 0.25), 0.0 for agreement.
    """
    if len(labels) < 2:
        return 0.0

    has_above = any(x > threshold for x in labels)
    has_below = any(x < threshold for x in labels)

    if has_above and has_below:
        return 1.0

    spread = max(labels) - min(labels)
    if spread > 0.25:
        return 0.5

    return 0.0


def score_items(
    annotations: dict[str, list[float]],
    config: DisagreementConfig,
) -> list[DisagreementScore]:
    """Compute disagreement score for each item.

    Filters items with fewer than min_annotators annotations. Final score
    is disagreement * (1 + severity * severity_weight) to boost items
    with binary flips.
    """
    scores: list[DisagreementScore] = []
    for item_id in sorted(annotations):
        labels = annotations[item_id]
        if len(labels) < config.min_annotators:
            continue

        disagreement = compute_disagreement(labels)
        severity = compute_severity(labels)
        combined = disagreement * (1.0 + severity * config.severity_weight)
        # Clamp to [0, 1]
        combined = min(combined, 1.0)

        spread = max(labels) - min(labels) if labels else 0.0

        scores.append(
            DisagreementScore(
                item_id=item_id,
                score=combined,
                n_annotators=len(labels),
                label_spread=spread,
            )
        )

    return scores


def sample_by_disagreement(
    scores: list[DisagreementScore],
    sample_size: int,
    seed: int | None = None,
) -> list[str]:
    """Select top-N items by disagreement score, with seeded tiebreaking.

    Items are sorted by score descending, then by a seeded random key for
    deterministic tiebreaking. Returns at most sample_size item IDs.
    """
    if not scores:
        return []

    rng = random.Random(seed)

    # Assign tiebreaker keys
    keyed = []
    for s in scores:
        tiebreaker = rng.random()
        keyed.append((s.score, tiebreaker, s.item_id))

    # Sort by score descending, tiebreaker descending
    keyed.sort(key=lambda x: (x[0], x[1]), reverse=True)

    k = min(sample_size, len(keyed))
    return [item_id for _, _, item_id in keyed[:k]]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_disagreement_sampling(
    annotations: dict[str, list[float]],
    config: DisagreementConfig,
) -> DisagreementResult:
    """Full disagreement sampling pipeline.

    1. Score all items by disagreement and severity
    2. Sample top-N by score
    3. Compute summary statistics
    """
    if not annotations:
        return DisagreementResult(
            selected_ids=[],
            scores=[],
            mean_disagreement=0.0,
            total_pool=0,
        )

    scores = score_items(annotations, config)

    selected = sample_by_disagreement(scores, config.sample_size, config.seed)

    mean_d = (
        sum(s.score for s in scores) / len(scores) if scores else 0.0
    )

    return DisagreementResult(
        selected_ids=selected,
        scores=scores,
        mean_disagreement=mean_d,
        total_pool=len(annotations),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_disagreement_report(result: DisagreementResult) -> str:
    """Format a human-readable disagreement sampling report."""
    lines: list[str] = []
    lines.append("Disagreement Sampling Report")
    lines.append("=" * 40)
    lines.append(f"Total pool size: {result.total_pool}")
    lines.append(f"Items scored: {len(result.scores)}")
    lines.append(f"Selected count: {len(result.selected_ids)}")
    lines.append(f"Mean disagreement: {result.mean_disagreement:.4f}")
    lines.append("")
    lines.append("Top disagreements:")
    top = sorted(result.scores, key=lambda s: s.score, reverse=True)[:10]
    for s in top:
        lines.append(
            f"  {s.item_id}: score={s.score:.3f} "
            f"annotators={s.n_annotators} spread={s.label_spread:.3f}"
        )
    lines.append("")
    lines.append("Selected IDs:")
    for item_id in result.selected_ids:
        lines.append(f"  - {item_id}")
    return "\n".join(lines)
