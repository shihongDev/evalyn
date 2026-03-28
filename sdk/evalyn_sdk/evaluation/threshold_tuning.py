"""Confidence threshold tuning for evaluation metrics.

Find optimal confidence cutoff per metric that maximizes alignment
with human annotations. Uses grid search over candidate thresholds
to select the one that maximizes F1 score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class ThresholdCandidate:
    """A single candidate threshold with its evaluation metrics."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "threshold": round(self.threshold, 6),
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1_score": round(self.f1_score, 6),
        }


@dataclass
class ThresholdResult:
    """Result of threshold tuning for a single metric."""

    metric_id: str
    optimal_threshold: float
    best_f1: float
    candidates: List[ThresholdCandidate] = field(default_factory=list)
    num_samples: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "optimal_threshold": round(self.optimal_threshold, 6),
            "best_f1": round(self.best_f1, 6),
            "candidates": [c.as_dict() for c in self.candidates],
            "num_samples": self.num_samples,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThresholdResult":
        candidates = [
            ThresholdCandidate(
                threshold=c["threshold"],
                accuracy=c["accuracy"],
                precision=c["precision"],
                recall=c["recall"],
                f1_score=c["f1_score"],
            )
            for c in data.get("candidates", [])
        ]
        return cls(
            metric_id=data["metric_id"],
            optimal_threshold=data["optimal_threshold"],
            best_f1=data["best_f1"],
            candidates=candidates,
            num_samples=data.get("num_samples", 0),
        )

    def format_text(self) -> str:
        lines = [
            f"Threshold Tuning: {self.metric_id}",
            f"  Optimal threshold: {self.optimal_threshold:.4f}",
            f"  Best F1: {self.best_f1:.4f}",
            f"  Samples: {self.num_samples}",
            f"  Candidates evaluated: {len(self.candidates)}",
        ]
        return "\n".join(lines)


def compute_f1(predictions: List[bool], ground_truth: List[bool]) -> float:
    """Standard F1 score from binary predictions and ground truth.

    Handles edge cases: empty lists return 0.0, all-same-class
    predictions return 0.0 when there are no true positives.
    """
    if not predictions or not ground_truth:
        return 0.0
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have same length")

    tp = sum(p and g for p, g in zip(predictions, ground_truth))
    fp = sum(p and not g for p, g in zip(predictions, ground_truth))
    fn = sum(not p and g for p, g in zip(predictions, ground_truth))

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def compute_accuracy(predictions: List[bool], ground_truth: List[bool]) -> float:
    """Fraction of correct predictions."""
    if not predictions or not ground_truth:
        return 0.0
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must have same length")
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def evaluate_threshold(
    scores: List[float],
    ground_truth: List[bool],
    threshold: float,
) -> ThresholdCandidate:
    """Apply a threshold to scores and compute all classification metrics.

    Scores >= threshold are predicted as True, otherwise False.
    """
    if len(scores) != len(ground_truth):
        raise ValueError("scores and ground_truth must have same length")

    predictions = [s >= threshold for s in scores]

    tp = sum(p and g for p, g in zip(predictions, ground_truth))
    fp = sum(p and not g for p, g in zip(predictions, ground_truth))
    fn = sum(not p and g for p, g in zip(predictions, ground_truth))

    accuracy = compute_accuracy(predictions, ground_truth)

    if tp == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    return ThresholdCandidate(
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
    )


def find_optimal_threshold(
    metric_id: str,
    scores: List[float],
    ground_truth: List[bool],
    num_candidates: int = 20,
) -> ThresholdResult:
    """Grid search over thresholds from min(scores) to max(scores).

    Returns the threshold that maximizes F1 score.
    """
    if not scores or not ground_truth:
        return ThresholdResult(
            metric_id=metric_id,
            optimal_threshold=0.5,
            best_f1=0.0,
            candidates=[],
            num_samples=0,
        )

    lo = min(scores)
    hi = max(scores)

    if lo == hi:
        # All scores identical - evaluate at that single point
        candidate = evaluate_threshold(scores, ground_truth, lo)
        return ThresholdResult(
            metric_id=metric_id,
            optimal_threshold=lo,
            best_f1=candidate.f1_score,
            candidates=[candidate],
            num_samples=len(scores),
        )

    step = (hi - lo) / (num_candidates - 1) if num_candidates > 1 else 0
    candidates: List[ThresholdCandidate] = []
    for i in range(num_candidates):
        t = lo + i * step
        candidate = evaluate_threshold(scores, ground_truth, t)
        candidates.append(candidate)

    best = max(candidates, key=lambda c: c.f1_score)

    return ThresholdResult(
        metric_id=metric_id,
        optimal_threshold=best.threshold,
        best_f1=best.f1_score,
        candidates=candidates,
        num_samples=len(scores),
    )


def tune_thresholds(
    metrics: Dict[str, Tuple[List[float], List[bool]]],
) -> Dict[str, ThresholdResult]:
    """Tune thresholds for multiple metrics at once.

    Args:
        metrics: Mapping of metric_id to (scores, ground_truth) tuples.

    Returns:
        Mapping of metric_id to ThresholdResult.
    """
    results: Dict[str, ThresholdResult] = {}
    for metric_id, (scores, ground_truth) in metrics.items():
        results[metric_id] = find_optimal_threshold(metric_id, scores, ground_truth)
    return results
