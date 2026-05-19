"""Calibration curriculum.

Start optimization on easy examples, progressively add harder ones.
Difficulty is estimated from scores: low-scoring items are harder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class DifficultyScore:
    """Estimated difficulty for a single item."""

    item_id: str
    difficulty: float  # 0 (easy) to 1 (hard)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "difficulty": self.difficulty,
            "reason": self.reason,
        }


@dataclass
class CurriculumStage:
    """A single stage in a calibration curriculum."""

    stage: int
    item_ids: list[str]
    difficulty_range: tuple[float, float]
    num_items: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "item_ids": self.item_ids,
            "difficulty_range": list(self.difficulty_range),
            "num_items": self.num_items,
        }


@dataclass
class CurriculumPlan:
    """Full curriculum plan with multiple stages."""

    stages: list[CurriculumStage]
    total_items: int
    num_stages: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": [s.as_dict() for s in self.stages],
            "total_items": self.total_items,
            "num_stages": self.num_stages,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumPlan:
        stages = [
            CurriculumStage(
                stage=s["stage"],
                item_ids=s["item_ids"],
                difficulty_range=tuple(s["difficulty_range"]),
                num_items=s["num_items"],
            )
            for s in data.get("stages", [])
        ]
        return cls(
            stages=stages,
            total_items=data["total_items"],
            num_stages=data["num_stages"],
        )

    def format_text(self) -> str:
        lines = [
            f"Curriculum Plan ({self.num_stages} stages, {self.total_items} items)",
        ]
        for s in self.stages:
            lo, hi = s.difficulty_range
            lines.append(
                f"  Stage {s.stage}: {s.num_items} items "
                f"(difficulty {lo:.2f}-{hi:.2f})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def estimate_difficulty(
    item_id: str, score: float, confidence: float = 1.0
) -> DifficultyScore:
    """Estimate difficulty from score and confidence.

    Difficulty = 1 - score (low-scoring items are harder).
    If confidence < 0.5, increase difficulty by 0.2 (uncertain items are harder).
    Result is clamped to [0, 1].
    """
    difficulty = 1.0 - score
    reason = f"score={score}"
    if confidence < 0.5:
        difficulty += 0.2
        reason += f", low confidence={confidence}"
    difficulty = max(0.0, min(1.0, difficulty))
    return DifficultyScore(item_id=item_id, difficulty=difficulty, reason=reason)


def sort_by_difficulty(
    items: list[DifficultyScore],
) -> list[DifficultyScore]:
    """Sort items by difficulty ascending (easiest first)."""
    return sorted(items, key=lambda d: d.difficulty)


def create_curriculum(
    items: list[DifficultyScore], num_stages: int = 3
) -> CurriculumPlan:
    """Split sorted items into equal-sized stages.

    Stage 1 = easiest items, stage N = hardest items.
    Items are sorted by difficulty before splitting.
    """
    if not items:
        return CurriculumPlan(stages=[], total_items=0, num_stages=0)

    sorted_items = sort_by_difficulty(items)
    total = len(sorted_items)
    # Clamp num_stages to at most total items
    effective_stages = min(num_stages, total)
    chunk_size = total // effective_stages
    remainder = total % effective_stages

    stages: list[CurriculumStage] = []
    start = 0
    for i in range(effective_stages):
        # Distribute remainder items across early stages
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk = sorted_items[start:end]
        ids = [d.item_id for d in chunk]
        lo = chunk[0].difficulty
        hi = chunk[-1].difficulty
        stages.append(
            CurriculumStage(
                stage=i + 1,
                item_ids=ids,
                difficulty_range=(lo, hi),
                num_items=len(ids),
            )
        )
        start = end

    return CurriculumPlan(
        stages=stages,
        total_items=total,
        num_stages=effective_stages,
    )


def get_stage_items(plan: CurriculumPlan, stage: int) -> list[str]:
    """Get item_ids for a given stage (1-indexed), cumulative.

    Returns items from stage 1 through the requested stage.
    """
    if not plan.stages:
        return []
    # Clamp stage to valid range
    stage = max(1, min(stage, len(plan.stages)))
    ids: list[str] = []
    for s in plan.stages:
        if s.stage <= stage:
            ids.extend(s.item_ids)
    return ids
