"""Per-score-level calibration.

Calibrate separately for each score level to reduce systematic bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ScoreLevelStats:
    """Statistics for a single score level."""

    level: str  # "low" / "medium" / "high"
    score_range: tuple[float, float]
    item_count: int
    accuracy: float
    bias: float  # mean(predicted - actual)

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "score_range": list(self.score_range),
            "item_count": self.item_count,
            "accuracy": self.accuracy,
            "bias": self.bias,
        }


@dataclass
class PerLevelCalibration:
    """Per-level calibration results."""

    levels: list[ScoreLevelStats] = field(default_factory=list)
    overall_accuracy: float = 0.0
    overall_bias: float = 0.0
    worst_level: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "levels": [lv.as_dict() for lv in self.levels],
            "overall_accuracy": self.overall_accuracy,
            "overall_bias": self.overall_bias,
            "worst_level": self.worst_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerLevelCalibration:
        levels = []
        for lv in data.get("levels", []):
            levels.append(
                ScoreLevelStats(
                    level=lv["level"],
                    score_range=tuple(lv["score_range"]),
                    item_count=lv["item_count"],
                    accuracy=lv["accuracy"],
                    bias=lv["bias"],
                )
            )
        return cls(
            levels=levels,
            overall_accuracy=data.get("overall_accuracy", 0.0),
            overall_bias=data.get("overall_bias", 0.0),
            worst_level=data.get("worst_level", ""),
        )

    def format_text(self) -> str:
        lines = [
            "Per-Level Calibration",
            f"  Overall accuracy: {self.overall_accuracy:.4f}",
            f"  Overall bias:     {self.overall_bias:.4f}",
            f"  Worst level:      {self.worst_level}",
        ]
        for lv in self.levels:
            lines.append(
                f"  {lv.level}: count={lv.item_count}  "
                f"accuracy={lv.accuracy:.4f}  bias={lv.bias:.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def classify_score_level(score: float) -> str:
    """Classify a score into low/medium/high.

    low: score < 0.4
    medium: 0.4 <= score < 0.7
    high: score >= 0.7
    """
    if score < 0.4:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def compute_per_level_stats(
    predictions: list[tuple[float, bool]],
    level: str,
    score_range: tuple[float, float],
) -> ScoreLevelStats:
    """Compute stats for items in a single level.

    predictions: list of (score, ground_truth) pairs already filtered to this level.
    accuracy: fraction where (score >= 0.5) matches ground_truth.
    bias: mean(score - (1.0 if truth else 0.0)).
    """
    if not predictions:
        return ScoreLevelStats(
            level=level,
            score_range=score_range,
            item_count=0,
            accuracy=0.0,
            bias=0.0,
        )

    correct = 0
    total_bias = 0.0
    for score, truth in predictions:
        predicted = score >= 0.5
        if predicted == truth:
            correct += 1
        actual = 1.0 if truth else 0.0
        total_bias += score - actual

    count = len(predictions)
    return ScoreLevelStats(
        level=level,
        score_range=score_range,
        item_count=count,
        accuracy=correct / count,
        bias=total_bias / count,
    )


def analyze_per_level(
    predictions: list[tuple[float, bool]],
) -> PerLevelCalibration:
    """Split predictions into levels and compute stats per level.

    Identifies worst level (lowest accuracy).
    """
    if not predictions:
        return PerLevelCalibration()

    level_defs = [
        ("low", (0.0, 0.4)),
        ("medium", (0.4, 0.7)),
        ("high", (0.7, 1.0)),
    ]

    buckets: dict[str, list[tuple[float, bool]]] = {name: [] for name, _ in level_defs}
    for score, truth in predictions:
        lv = classify_score_level(score)
        buckets[lv].append((score, truth))

    level_stats: list[ScoreLevelStats] = []
    for name, score_range in level_defs:
        stats = compute_per_level_stats(buckets[name], name, score_range)
        level_stats.append(stats)

    # Overall accuracy and bias
    total_correct = 0
    total_bias = 0.0
    for score, truth in predictions:
        if (score >= 0.5) == truth:
            total_correct += 1
        total_bias += score - (1.0 if truth else 0.0)

    overall_accuracy = total_correct / len(predictions)
    overall_bias = total_bias / len(predictions)

    # Worst level: lowest accuracy among levels with items
    populated = [lv for lv in level_stats if lv.item_count > 0]
    worst = min(populated, key=lambda lv: lv.accuracy) if populated else None
    worst_level = worst.level if worst else ""

    return PerLevelCalibration(
        levels=level_stats,
        overall_accuracy=overall_accuracy,
        overall_bias=overall_bias,
        worst_level=worst_level,
    )


def correct_bias(
    score: float,
    level_stats: dict[str, ScoreLevelStats],
) -> float:
    """Subtract the level's bias from the score. Clamp to [0, 1]."""
    level = classify_score_level(score)
    stats = level_stats.get(level)
    if stats is None:
        return score
    corrected = score - stats.bias
    return max(0.0, min(1.0, corrected))
