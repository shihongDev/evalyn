"""Curriculum sampling: order samples from easy to hard for progressive evaluation.

Pure Python, no external dependencies.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DifficultyEstimate:
    """Estimated difficulty for a single evaluation item."""

    item_id: str
    difficulty: float  # 0.0 (easy) to 1.0 (hard)
    source: str  # "length", "complexity", "score"

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "difficulty": self.difficulty,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DifficultyEstimate:
        return cls(
            item_id=data["item_id"],
            difficulty=data["difficulty"],
            source=data["source"],
        )


@dataclass
class CurriculumConfig:
    """Configuration for curriculum sampling."""

    order: str = "easy_first"  # "easy_first" or "hard_first"
    early_stop_threshold: float = 0.5  # stop if pass rate drops below this
    batch_size: int = 10

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "early_stop_threshold": self.early_stop_threshold,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumConfig:
        return cls(
            order=data.get("order", "easy_first"),
            early_stop_threshold=data.get("early_stop_threshold", 0.5),
            batch_size=data.get("batch_size", 10),
        )


@dataclass
class CurriculumBatch:
    """A single batch in a curriculum evaluation."""

    batch_index: int
    item_ids: list[str] = field(default_factory=list)
    difficulty_range: tuple[float, float] = (0.0, 0.0)
    pass_rate: float = 0.0
    should_stop: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_index": self.batch_index,
            "item_ids": self.item_ids,
            "difficulty_range": list(self.difficulty_range),
            "pass_rate": self.pass_rate,
            "should_stop": self.should_stop,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumBatch:
        dr = data.get("difficulty_range", [0.0, 0.0])
        return cls(
            batch_index=data["batch_index"],
            item_ids=data.get("item_ids", []),
            difficulty_range=(dr[0], dr[1]),
            pass_rate=data.get("pass_rate", 0.0),
            should_stop=data.get("should_stop", False),
        )


@dataclass
class CurriculumResult:
    """Result of a full curriculum evaluation run."""

    batches: list[CurriculumBatch] = field(default_factory=list)
    total_items: int = 0
    items_evaluated: int = 0
    stopped_early: bool = False
    final_batch_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "batches": [b.as_dict() for b in self.batches],
            "total_items": self.total_items,
            "items_evaluated": self.items_evaluated,
            "stopped_early": self.stopped_early,
            "final_batch_index": self.final_batch_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumResult:
        return cls(
            batches=[
                CurriculumBatch.from_dict(b) for b in data.get("batches", [])
            ],
            total_items=data.get("total_items", 0),
            items_evaluated=data.get("items_evaluated", 0),
            stopped_early=data.get("stopped_early", False),
            final_batch_index=data.get("final_batch_index", 0),
        )


# ---------------------------------------------------------------------------
# Difficulty estimation functions
# ---------------------------------------------------------------------------


def estimate_difficulty_by_length(
    items: dict[str, str],
) -> list[DifficultyEstimate]:
    """Estimate difficulty by normalizing text lengths to 0-1 range.

    Longer text is considered harder. Empty input returns empty list.
    """
    if not items:
        return []
    lengths = {k: len(v) for k, v in items.items()}
    min_len = min(lengths.values())
    max_len = max(lengths.values())
    span = max_len - min_len
    return [
        DifficultyEstimate(
            item_id=k,
            difficulty=(lengths[k] - min_len) / span if span > 0 else 0.0,
            source="length",
        )
        for k in items
    ]


def estimate_difficulty_by_complexity(
    items: dict[str, str],
) -> list[DifficultyEstimate]:
    """Estimate difficulty using word count, average word length, and sentence count.

    Each feature is normalized to 0-1 and averaged. Higher values mean harder.
    """
    if not items:
        return []

    def _features(text: str) -> tuple[int, float, int]:
        words = text.split()
        word_count = len(words)
        avg_word_len = (
            sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
        )
        sentence_count = max(len(re.split(r"[.!?]+", text)) - 1, 1)
        return word_count, avg_word_len, sentence_count

    features = {k: _features(v) for k, v in items.items()}

    word_counts = [f[0] for f in features.values()]
    avg_lens = [f[1] for f in features.values()]
    sent_counts = [f[2] for f in features.values()]

    def _normalize(val: float, vals: list[float]) -> float:
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return 0.0
        return (val - lo) / (hi - lo)

    results: list[DifficultyEstimate] = []
    for k in items:
        wc, awl, sc = features[k]
        norm_wc = _normalize(wc, word_counts)
        norm_awl = _normalize(awl, avg_lens)
        norm_sc = _normalize(sc, sent_counts)
        difficulty = (norm_wc + norm_awl + norm_sc) / 3.0
        results.append(
            DifficultyEstimate(item_id=k, difficulty=difficulty, source="complexity")
        )
    return results


def estimate_difficulty_by_scores(
    scores: dict[str, float],
) -> list[DifficultyEstimate]:
    """Estimate difficulty using inverse score: low score means hard.

    Scores are inverted and normalized to 0-1. A score of 0.0 maps to difficulty 1.0.
    """
    if not scores:
        return []
    min_score = min(scores.values())
    max_score = max(scores.values())
    span = max_score - min_score
    results: list[DifficultyEstimate] = []
    for k, s in scores.items():
        if span > 0:
            difficulty = 1.0 - (s - min_score) / span
        else:
            difficulty = 0.0
        results.append(
            DifficultyEstimate(item_id=k, difficulty=difficulty, source="score")
        )
    return results


# ---------------------------------------------------------------------------
# Ordering and batching
# ---------------------------------------------------------------------------


def order_curriculum(
    estimates: list[DifficultyEstimate],
    config: CurriculumConfig,
) -> list[str]:
    """Return item_ids sorted by difficulty according to config.order."""
    reverse = config.order == "hard_first"
    sorted_estimates = sorted(estimates, key=lambda e: e.difficulty, reverse=reverse)
    return [e.item_id for e in sorted_estimates]


def create_batches(
    ordered_ids: list[str],
    estimates: list[DifficultyEstimate],
    batch_size: int = 10,
) -> list[CurriculumBatch]:
    """Split ordered item ids into batches with difficulty ranges."""
    if not ordered_ids:
        return []
    difficulty_map = {e.item_id: e.difficulty for e in estimates}
    batches: list[CurriculumBatch] = []
    for i in range(0, len(ordered_ids), batch_size):
        chunk = ordered_ids[i : i + batch_size]
        diffs = [difficulty_map.get(cid, 0.0) for cid in chunk]
        batches.append(
            CurriculumBatch(
                batch_index=len(batches),
                item_ids=chunk,
                difficulty_range=(min(diffs), max(diffs)),
            )
        )
    return batches


def check_early_stop(
    batch: CurriculumBatch,
    config: CurriculumConfig,
) -> bool:
    """Return True if pass_rate is below the early stop threshold."""
    return batch.pass_rate < config.early_stop_threshold


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_curriculum(
    items: dict[str, str],
    evaluate_fn: Callable[[list[str]], dict[str, float]],
    config: CurriculumConfig | None = None,
) -> CurriculumResult:
    """Full curriculum pipeline: estimate, order, batch, evaluate, check early stop.

    evaluate_fn receives a list of item_ids and returns {item_id: score} where
    score >= 0.5 is considered a pass.
    """
    if config is None:
        config = CurriculumConfig()

    estimates = estimate_difficulty_by_length(items)
    ordered = order_curriculum(estimates, config)
    batches = create_batches(ordered, estimates, config.batch_size)

    evaluated_count = 0
    stopped_early = False
    final_index = 0

    for batch in batches:
        scores = evaluate_fn(batch.item_ids)
        passes = sum(1 for s in scores.values() if s >= 0.5)
        batch.pass_rate = passes / len(batch.item_ids) if batch.item_ids else 0.0
        evaluated_count += len(batch.item_ids)
        final_index = batch.batch_index

        if check_early_stop(batch, config):
            batch.should_stop = True
            stopped_early = True
            break

    return CurriculumResult(
        batches=batches,
        total_items=len(items),
        items_evaluated=evaluated_count,
        stopped_early=stopped_early,
        final_batch_index=final_index,
    )


def format_curriculum_report(result: CurriculumResult) -> str:
    """Format a human-readable curriculum evaluation report."""
    lines: list[str] = []
    lines.append("Curriculum Evaluation Report")
    lines.append("=" * 40)
    lines.append(f"Total items: {result.total_items}")
    lines.append(f"Items evaluated: {result.items_evaluated}")
    lines.append(f"Stopped early: {result.stopped_early}")
    lines.append(f"Final batch index: {result.final_batch_index}")
    lines.append("")

    for batch in result.batches:
        lo, hi = batch.difficulty_range
        lines.append(f"Batch {batch.batch_index}:")
        lines.append(f"  Items: {len(batch.item_ids)}")
        lines.append(f"  Difficulty range: {lo:.2f} - {hi:.2f}")
        lines.append(f"  Pass rate: {batch.pass_rate:.2%}")
        if batch.should_stop:
            lines.append("  ** Early stop triggered **")
        lines.append("")

    return "\n".join(lines)
