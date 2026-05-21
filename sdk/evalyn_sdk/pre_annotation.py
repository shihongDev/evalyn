"""
Bulk pre-annotation pipeline via LLM.

Provides the framework for pre-filling annotations for human review:
triage by confidence, prompt building, accuracy computation, and
reporting. No actual LLM calls - pure Python pipeline scaffolding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PreAnnotationConfig:
    """Configuration for the pre-annotation pipeline."""

    confidence_threshold: float = 0.8
    auto_accept_above: float = 0.95
    human_review_below: float = 0.8
    batch_size: int = 10

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence_threshold": self.confidence_threshold,
            "auto_accept_above": self.auto_accept_above,
            "human_review_below": self.human_review_below,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreAnnotationConfig:
        return cls(
            confidence_threshold=data.get("confidence_threshold", 0.8),
            auto_accept_above=data.get("auto_accept_above", 0.95),
            human_review_below=data.get("human_review_below", 0.8),
            batch_size=data.get("batch_size", 10),
        )


@dataclass
class PreAnnotation:
    """A single pre-annotation for an evaluation item."""

    item_id: str
    predicted_label: float
    confidence: float
    status: str = ""  # "auto_accepted", "needs_review", "rejected"
    reasoning: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "status": self.status,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreAnnotation:
        return cls(
            item_id=data.get("item_id", ""),
            predicted_label=data.get("predicted_label", 0.0),
            confidence=data.get("confidence", 0.0),
            status=data.get("status", ""),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class PreAnnotationReport:
    """Summary report for a batch of pre-annotations."""

    annotations: list[PreAnnotation] = field(default_factory=list)
    auto_accepted: int = 0
    needs_review: int = 0
    rejected: int = 0
    total: int = 0
    accuracy: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "annotations": [a.as_dict() for a in self.annotations],
            "auto_accepted": self.auto_accepted,
            "needs_review": self.needs_review,
            "rejected": self.rejected,
            "total": self.total,
            "accuracy": self.accuracy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreAnnotationReport:
        return cls(
            annotations=[PreAnnotation.from_dict(a) for a in data.get("annotations", [])],
            auto_accepted=data.get("auto_accepted", 0),
            needs_review=data.get("needs_review", 0),
            rejected=data.get("rejected", 0),
            total=data.get("total", 0),
            accuracy=data.get("accuracy", 0.0),
        )


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------


def triage_annotation(
    annotation: PreAnnotation,
    config: PreAnnotationConfig,
) -> PreAnnotation:
    """Set annotation status based on confidence vs config thresholds.

    - confidence >= auto_accept_above: "auto_accepted"
    - confidence < human_review_below: "rejected"
    - otherwise: "needs_review"
    """
    if annotation.confidence >= config.auto_accept_above:
        annotation.status = "auto_accepted"
    elif annotation.confidence < config.human_review_below:
        annotation.status = "rejected"
    else:
        annotation.status = "needs_review"
    return annotation


def triage_batch(
    annotations: list[PreAnnotation],
    config: PreAnnotationConfig,
) -> list[PreAnnotation]:
    """Triage all annotations in a batch."""
    return [triage_annotation(a, config) for a in annotations]


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------


def build_pre_annotation_prompt(
    item: dict[str, Any],
    metric_id: str,
    rubric: str = "",
) -> str:
    """Construct an LLM prompt for pre-annotating a single item.

    The prompt asks the LLM to score the item on the given metric
    and provide a confidence level and reasoning.
    """
    lines = [
        f"You are an evaluation annotator. Score the following item on the metric '{metric_id}'.",
        "",
        "Item:",
    ]

    for key, value in sorted(item.items()):
        lines.append(f"  {key}: {value}")

    lines.append("")

    if rubric:
        lines.append("Rubric:")
        lines.append(f"  {rubric}")
        lines.append("")

    lines.extend(
        [
            "Respond with exactly three lines:",
            "  score: <float between 0.0 and 1.0>",
            "  confidence: <float between 0.0 and 1.0>",
            "  reasoning: <brief explanation>",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Accuracy Computation
# ---------------------------------------------------------------------------


def compute_pre_annotation_accuracy(
    pre_annotations: list[PreAnnotation],
    human_labels: dict[str, float],
    threshold: float = 0.5,
) -> float:
    """Compare pre-annotations against human labels.

    For each pre-annotation with a matching human label, checks whether
    both agree on the pass/fail classification (using the threshold).
    Returns the fraction of agreements, or 0.0 if no comparisons are possible.
    """
    if not pre_annotations or not human_labels:
        return 0.0

    matches = 0
    total = 0

    for ann in pre_annotations:
        if ann.item_id not in human_labels:
            continue
        human = human_labels[ann.item_id]
        predicted_pass = ann.predicted_label >= threshold
        human_pass = human >= threshold
        if predicted_pass == human_pass:
            matches += 1
        total += 1

    if total == 0:
        return 0.0
    return matches / total


# ---------------------------------------------------------------------------
# Report Builder
# ---------------------------------------------------------------------------


def build_pre_annotation_report(
    annotations: list[PreAnnotation],
    human_labels: dict[str, float] | None = None,
) -> PreAnnotationReport:
    """Build a summary report from a list of pre-annotations.

    Counts statuses and optionally computes accuracy against human labels.
    """
    auto_accepted = sum(1 for a in annotations if a.status == "auto_accepted")
    needs_review = sum(1 for a in annotations if a.status == "needs_review")
    rejected = sum(1 for a in annotations if a.status == "rejected")

    accuracy = 0.0
    if human_labels:
        accuracy = compute_pre_annotation_accuracy(annotations, human_labels)

    return PreAnnotationReport(
        annotations=list(annotations),
        auto_accepted=auto_accepted,
        needs_review=needs_review,
        rejected=rejected,
        total=len(annotations),
        accuracy=accuracy,
    )


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def format_pre_annotation_report(report: PreAnnotationReport) -> str:
    """Format a PreAnnotationReport as a human-readable string."""
    lines = [
        "Pre-Annotation Report",
        "=====================",
        "",
        f"Total annotations: {report.total}",
        f"Auto-accepted:     {report.auto_accepted}",
        f"Needs review:      {report.needs_review}",
        f"Rejected:          {report.rejected}",
    ]

    if report.accuracy > 0:
        lines.append(f"Accuracy:          {report.accuracy:.1%}")

    lines.append("")

    if report.annotations:
        lines.append("Details:")
        for ann in report.annotations:
            status_tag = f"[{ann.status}]" if ann.status else "[unset]"
            lines.append(
                f"  {ann.item_id}: label={ann.predicted_label:.2f} "
                f"conf={ann.confidence:.2f} {status_tag}"
            )
            if ann.reasoning:
                lines.append(f"    reason: {ann.reasoning}")

    return "\n".join(lines)
