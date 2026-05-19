"""
Judge confusion matrix for agreement/disagreement analysis.

Visualizes agreement patterns between judge predictions and human labels.
Computes precision, recall, F1, accuracy, and Cohen's kappa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ConfusionCell:
    """A single cell in the confusion matrix."""

    predicted: str
    actual: str
    count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicted": self.predicted,
            "actual": self.actual,
            "count": self.count,
        }


@dataclass
class ConfusionMatrix:
    """Confusion matrix with labels and cell counts."""

    cells: list[ConfusionCell] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    total: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "cells": [c.as_dict() for c in self.cells],
            "labels": list(self.labels),
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfusionMatrix:
        cells = [
            ConfusionCell(
                predicted=c["predicted"],
                actual=c["actual"],
                count=c.get("count", 0),
            )
            for c in data.get("cells", [])
        ]
        return cls(
            cells=cells,
            labels=data.get("labels", []),
            total=data.get("total", 0),
        )

    @property
    def accuracy(self) -> float:
        """Fraction of correct predictions."""
        if self.total == 0:
            return 0.0
        correct = sum(c.count for c in self.cells if c.predicted == c.actual)
        return correct / self.total

    def get_cell(self, predicted: str, actual: str) -> int:
        """Get count for a specific (predicted, actual) cell."""
        for c in self.cells:
            if c.predicted == predicted and c.actual == actual:
                return c.count
        return 0


@dataclass
class ConfusionReport:
    """Full confusion analysis report."""

    matrix: ConfusionMatrix = field(default_factory=ConfusionMatrix)
    accuracy: float = 0.0
    precision: dict[str, float] = field(default_factory=dict)
    recall: dict[str, float] = field(default_factory=dict)
    f1: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix.as_dict(),
            "accuracy": self.accuracy,
            "precision": dict(self.precision),
            "recall": dict(self.recall),
            "f1": dict(self.f1),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfusionReport:
        matrix = ConfusionMatrix.from_dict(data.get("matrix", {}))
        return cls(
            matrix=matrix,
            accuracy=data.get("accuracy", 0.0),
            precision=data.get("precision", {}),
            recall=data.get("recall", {}),
            f1=data.get("f1", {}),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Confusion Report (accuracy={self.accuracy:.4f})")
        lines.append("")
        if self.matrix.labels:
            lines.append("  Per-label metrics:")
            for label in self.matrix.labels:
                p = self.precision.get(label, 0.0)
                r = self.recall.get(label, 0.0)
                f = self.f1.get(label, 0.0)
                lines.append(f"    {label}: P={p:.4f} R={r:.4f} F1={f:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def build_confusion_matrix(
    predictions: list[str], actuals: list[str]
) -> ConfusionMatrix:
    """Build a confusion matrix from paired predictions and actuals."""
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have the same length")

    n = len(predictions)
    if n == 0:
        return ConfusionMatrix(cells=[], labels=[], total=0)

    # Discover labels in order of first appearance
    seen: dict[str, bool] = {}
    labels: list[str] = []
    for p, a in zip(predictions, actuals):
        if p not in seen:
            seen[p] = True
            labels.append(p)
        if a not in seen:
            seen[a] = True
            labels.append(a)

    # Count cells
    counts: dict[tuple[str, str], int] = {}
    for p, a in zip(predictions, actuals):
        key = (p, a)
        counts[key] = counts.get(key, 0) + 1

    # Build cells for all label combinations (including zero counts)
    cells: list[ConfusionCell] = []
    for pred in labels:
        for actual in labels:
            count = counts.get((pred, actual), 0)
            cells.append(ConfusionCell(predicted=pred, actual=actual, count=count))

    return ConfusionMatrix(cells=cells, labels=labels, total=n)


def compute_precision_recall(
    matrix: ConfusionMatrix,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-label precision and recall from a confusion matrix."""
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}

    for label in matrix.labels:
        # Precision: TP / (TP + FP) = correct for this predicted label / total predicted as this label
        tp = matrix.get_cell(label, label)
        total_predicted = sum(matrix.get_cell(label, a) for a in matrix.labels)
        precision[label] = tp / total_predicted if total_predicted > 0 else 0.0

        # Recall: TP / (TP + FN) = correct for this actual label / total actually this label
        total_actual = sum(matrix.get_cell(p, label) for p in matrix.labels)
        recall[label] = tp / total_actual if total_actual > 0 else 0.0

    return precision, recall


def compute_f1_scores(
    precision: dict[str, float], recall: dict[str, float]
) -> dict[str, float]:
    """Harmonic mean of precision and recall per label."""
    f1: dict[str, float] = {}
    for label in precision:
        p = precision[label]
        r = recall.get(label, 0.0)
        if p + r > 0:
            f1[label] = 2.0 * p * r / (p + r)
        else:
            f1[label] = 0.0
    return f1


def build_confusion_report(
    predictions: list[str], actuals: list[str]
) -> ConfusionReport:
    """Build a full confusion report from predictions and actuals."""
    matrix = build_confusion_matrix(predictions, actuals)
    precision, recall = compute_precision_recall(matrix)
    f1 = compute_f1_scores(precision, recall)

    return ConfusionReport(
        matrix=matrix,
        accuracy=matrix.accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def render_confusion_ascii(matrix: ConfusionMatrix) -> str:
    """Render confusion matrix as an ASCII table."""
    if not matrix.labels:
        return "(empty matrix)"

    labels = matrix.labels

    # Determine column widths
    max_label_len = max(len(lbl) for lbl in labels)
    col_width = max(max_label_len, 5)

    lines: list[str] = []

    # Header row
    header = " " * (col_width + 2) + "| " + " | ".join(lbl.rjust(col_width) for lbl in labels) + " |"
    lines.append("Predicted v / Actual >")
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for pred in labels:
        row_values = [str(matrix.get_cell(pred, actual)).rjust(col_width) for actual in labels]
        row = pred.rjust(col_width) + "  | " + " | ".join(row_values) + " |"
        lines.append(row)

    return "\n".join(lines)


def compute_kappa(matrix: ConfusionMatrix) -> float:
    """Cohen's kappa coefficient for inter-rater agreement.

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    Returns 0.0 for empty matrices or when expected agreement is 1.0.
    """
    if matrix.total == 0:
        return 0.0

    n = matrix.total
    labels = matrix.labels

    # Observed agreement (accuracy)
    observed = sum(matrix.get_cell(lbl, lbl) for lbl in labels) / n

    # Expected agreement by chance
    expected = 0.0
    for label in labels:
        # Proportion predicted as this label
        pred_count = sum(matrix.get_cell(label, a) for a in labels)
        # Proportion actually this label
        actual_count = sum(matrix.get_cell(p, label) for p in labels)
        expected += (pred_count / n) * (actual_count / n)

    if expected >= 1.0:
        return 0.0

    return (observed - expected) / (1.0 - expected)
