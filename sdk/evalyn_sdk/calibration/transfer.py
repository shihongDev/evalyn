"""Transfer calibration across similar metrics.

Apply calibration learned on one metric to similar metrics by measuring
rubric similarity and adapting prompts via text substitution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class MetricSimilarity:
    """Similarity between a source and target metric."""

    source_metric: str
    target_metric: str
    similarity_score: float  # 0-1
    method: str = "word_overlap"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_metric": self.source_metric,
            "target_metric": self.target_metric,
            "similarity_score": self.similarity_score,
            "method": self.method,
        }


@dataclass
class TransferResult:
    """Result of transferring calibration from one metric to another."""

    source_metric: str
    target_metric: str
    transferred_preamble: str
    similarity: float
    validation_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source_metric": self.source_metric,
            "target_metric": self.target_metric,
            "transferred_preamble": self.transferred_preamble,
            "similarity": self.similarity,
            "validation_score": self.validation_score,
        }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransferResult:
        return cls(
            source_metric=data["source_metric"],
            target_metric=data["target_metric"],
            transferred_preamble=data["transferred_preamble"],
            similarity=data["similarity"],
            validation_score=data.get("validation_score"),
        )


@dataclass
class TransferReport:
    """Report of all transfer calibration attempts."""

    transfers: list[TransferResult]
    successful: int
    total: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "transfers": [t.as_dict() for t in self.transfers],
            "successful": self.successful,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransferReport:
        transfers = [TransferResult.from_dict(t) for t in data.get("transfers", [])]
        return cls(
            transfers=transfers,
            successful=data["successful"],
            total=data["total"],
        )

    def format_text(self) -> str:
        lines = [f"Transfer Report: {self.successful}/{self.total} successful"]
        for t in self.transfers:
            status = "ok" if t.validation_score is None or t.validation_score >= 0.5 else "low"
            score_str = f" (validation: {t.validation_score:.2f})" if t.validation_score is not None else ""
            lines.append(
                f"  {t.source_metric} -> {t.target_metric}: "
                f"similarity={t.similarity:.2f}{score_str} [{status}]"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_rubric_similarity(rubric_a: str, rubric_b: str) -> float:
    """Word overlap (Jaccard) between two rubric texts, case-insensitive.

    Returns 0.0 if both rubrics are empty.
    """
    words_a = set(rubric_a.lower().split())
    words_b = set(rubric_b.lower().split())
    if not words_a and not words_b:
        return 0.0
    union = words_a | words_b
    if not union:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / len(union)


def find_similar_metrics(
    source_rubric: str,
    target_rubrics: dict[str, str],
    min_similarity: float = 0.3,
) -> list[MetricSimilarity]:
    """Find target metrics similar to the source rubric.

    Returns a list of MetricSimilarity sorted by similarity descending,
    filtered to those meeting the min_similarity threshold.
    """
    results: list[MetricSimilarity] = []
    for name, rubric in target_rubrics.items():
        score = compute_rubric_similarity(source_rubric, rubric)
        if score >= min_similarity:
            results.append(
                MetricSimilarity(
                    source_metric="source",
                    target_metric=name,
                    similarity_score=score,
                )
            )
    results.sort(key=lambda m: m.similarity_score, reverse=True)
    return results


def transfer_preamble(source_prompt: str, target_metric_name: str) -> str:
    """Adapt a calibrated preamble for a new metric by replacing the metric name.

    Looks for common metric-reference patterns and substitutes with the
    target metric name. If no recognizable pattern is found, appends a
    note about the target metric.
    """
    # Try to find a "metric: <name>" or "Metric: <name>" pattern
    import re

    pattern = re.compile(r"(?i)(metric\s*:\s*)\S+")
    if pattern.search(source_prompt):
        return pattern.sub(rf"\g<1>{target_metric_name}", source_prompt)

    # Fallback: append target context
    return f"{source_prompt}\n[Adapted for metric: {target_metric_name}]"


def validate_transfer(
    transferred_prompt: str,
    test_scores: list[float],
    ground_truth: list[bool],
) -> float:
    """Quick validation of transferred prompt scores against ground truth.

    Uses a 0.5 threshold: score >= 0.5 is treated as True.
    Returns accuracy (fraction of correct predictions).
    Returns 0.0 if inputs are empty or lengths differ.
    """
    if not test_scores or not ground_truth:
        return 0.0
    if len(test_scores) != len(ground_truth):
        return 0.0
    correct = sum(
        1
        for score, truth in zip(test_scores, ground_truth)
        if (score >= 0.5) == truth
    )
    return correct / len(test_scores)


def plan_transfers(
    source_calibrations: dict[str, str],
    target_rubrics: dict[str, str],
    min_similarity: float = 0.3,
) -> TransferReport:
    """Plan all possible transfers from source calibrations to target rubrics.

    source_calibrations maps metric_name to calibrated preamble.
    target_rubrics maps metric_name to rubric text.

    Returns a TransferReport with one TransferResult per viable pair.
    """
    transfers: list[TransferResult] = []

    for source_name, source_preamble in source_calibrations.items():
        # Use the source preamble as the "rubric" for similarity computation
        similar = find_similar_metrics(source_preamble, target_rubrics, min_similarity)
        for match in similar:
            adapted = transfer_preamble(source_preamble, match.target_metric)
            transfers.append(
                TransferResult(
                    source_metric=source_name,
                    target_metric=match.target_metric,
                    transferred_preamble=adapted,
                    similarity=match.similarity_score,
                )
            )

    return TransferReport(
        transfers=transfers,
        successful=len(transfers),
        total=len(transfers),
    )
