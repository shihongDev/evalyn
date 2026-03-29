"""
Inter-annotator agreement metrics for evaluation consistency.

Computes Cohen's Kappa, percent agreement, Krippendorff's Alpha,
and pairwise agreement matrices. Pure Python, no external dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AnnotatorLabel:
    """A single label assigned by an annotator to an item for a metric."""

    item_id: str
    annotator_id: str
    metric_id: str
    label: float
    confidence: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "annotator_id": self.annotator_id,
            "metric_id": self.metric_id,
            "label": self.label,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnnotatorLabel:
        return cls(
            item_id=data["item_id"],
            annotator_id=data["annotator_id"],
            metric_id=data["metric_id"],
            label=data["label"],
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class AgreementScore:
    """Pairwise agreement score between two annotators for a metric."""

    metric_id: str
    annotator_pair: tuple[str, str]
    cohens_kappa: float
    percent_agreement: float
    n_items: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "annotator_pair": list(self.annotator_pair),
            "cohens_kappa": self.cohens_kappa,
            "percent_agreement": self.percent_agreement,
            "n_items": self.n_items,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgreementScore:
        pair = data["annotator_pair"]
        return cls(
            metric_id=data["metric_id"],
            annotator_pair=(pair[0], pair[1]),
            cohens_kappa=data["cohens_kappa"],
            percent_agreement=data["percent_agreement"],
            n_items=data["n_items"],
        )


@dataclass
class AgreementReport:
    """Full inter-annotator agreement report."""

    scores: list[AgreementScore]
    overall_kappa: float
    overall_agreement: float
    high_disagreement_items: list[str]
    total_items: int
    total_annotators: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scores": [s.as_dict() for s in self.scores],
            "overall_kappa": self.overall_kappa,
            "overall_agreement": self.overall_agreement,
            "high_disagreement_items": self.high_disagreement_items,
            "total_items": self.total_items,
            "total_annotators": self.total_annotators,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgreementReport:
        return cls(
            scores=[AgreementScore.from_dict(s) for s in data["scores"]],
            overall_kappa=data["overall_kappa"],
            overall_agreement=data["overall_agreement"],
            high_disagreement_items=data["high_disagreement_items"],
            total_items=data["total_items"],
            total_annotators=data["total_annotators"],
        )


# ---------------------------------------------------------------------------
# Core Computation Functions
# ---------------------------------------------------------------------------


def compute_cohens_kappa(
    labels_a: list[float],
    labels_b: list[float],
    threshold: float = 0.5,
) -> float:
    """Compute Cohen's Kappa between two annotators.

    Binarizes continuous scores at the threshold (>= threshold -> 1, else 0),
    then computes observed vs expected agreement.
    Returns a value in [-1, 1] where 1 is perfect agreement.
    Returns 0.0 if expected agreement is 1.0 (degenerate case).
    """
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return 0.0

    n = len(labels_a)
    bin_a = [1 if v >= threshold else 0 for v in labels_a]
    bin_b = [1 if v >= threshold else 0 for v in labels_b]

    # Observed agreement
    agree = sum(1 for a, b in zip(bin_a, bin_b) if a == b)
    p_o = agree / n

    # Expected agreement by chance
    a_pos = sum(bin_a) / n
    b_pos = sum(bin_b) / n
    a_neg = 1 - a_pos
    b_neg = 1 - b_pos
    p_e = (a_pos * b_pos) + (a_neg * b_neg)

    if p_e == 1.0:
        return 0.0 if p_o < 1.0 else 1.0

    return (p_o - p_e) / (1 - p_e)


def compute_percent_agreement(
    labels_a: list[float],
    labels_b: list[float],
    threshold: float = 0.5,
) -> float:
    """Simple percent agreement after binarizing at threshold."""
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return 0.0

    n = len(labels_a)
    bin_a = [1 if v >= threshold else 0 for v in labels_a]
    bin_b = [1 if v >= threshold else 0 for v in labels_b]

    agree = sum(1 for a, b in zip(bin_a, bin_b) if a == b)
    return agree / n


def compute_krippendorff_alpha(all_labels: list[AnnotatorLabel]) -> float:
    """Compute Krippendorff's Alpha for multiple annotators (interval scale).

    Handles missing data: not all annotators need to label every item.
    Returns a value in [-1, 1] where 1 is perfect agreement.
    Returns 0.0 for degenerate cases (fewer than 2 coders or items).
    """
    if not all_labels:
        return 0.0

    # Build reliability matrix: item -> {annotator: label}
    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for lbl in all_labels:
        key = f"{lbl.item_id}:{lbl.metric_id}"
        matrix[key][lbl.annotator_id] = lbl.label

    # Collect all pairable values per unit (items with >= 2 annotations)
    # Using the formulation from Krippendorff (2011) for interval data
    total_pairs = 0
    observed_disagreement = 0.0
    all_values: list[float] = []

    for unit_labels in matrix.values():
        values = list(unit_labels.values())
        n_u = len(values)
        if n_u < 2:
            continue
        # Within-unit pairs
        for i in range(n_u):
            for j in range(i + 1, n_u):
                diff_sq = (values[i] - values[j]) ** 2
                observed_disagreement += diff_sq
                total_pairs += 1
        all_values.extend(values)

    if total_pairs == 0:
        return 0.0

    # Expected disagreement: mean squared difference over all value pairs
    n_vals = len(all_values)
    if n_vals < 2:
        return 0.0

    expected_disagreement = 0.0
    expected_pairs = 0
    for i in range(n_vals):
        for j in range(i + 1, n_vals):
            expected_disagreement += (all_values[i] - all_values[j]) ** 2
            expected_pairs += 1

    if expected_pairs == 0 or expected_disagreement == 0.0:
        return 1.0  # All values identical: perfect agreement

    d_o = observed_disagreement / total_pairs
    d_e = expected_disagreement / expected_pairs

    return 1.0 - (d_o / d_e)


# ---------------------------------------------------------------------------
# Pairwise and Aggregation
# ---------------------------------------------------------------------------


def build_pairwise_matrix(
    labels: list[AnnotatorLabel],
) -> dict[tuple[str, str], AgreementScore]:
    """Build pairwise agreement scores for all annotator pairs.

    Groups labels by metric, then for each pair of annotators computes
    Cohen's Kappa and percent agreement on their shared items.
    """
    # Group by metric -> annotator -> {item_id: label}
    by_metric: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for lbl in labels:
        by_metric[lbl.metric_id][lbl.annotator_id][lbl.item_id] = lbl.label

    result: dict[tuple[str, str], AgreementScore] = {}

    for metric_id, annotators in by_metric.items():
        ann_ids = sorted(annotators.keys())
        for a, b in combinations(ann_ids, 2):
            shared_items = sorted(
                set(annotators[a].keys()) & set(annotators[b].keys())
            )
            if not shared_items:
                continue
            labels_a = [annotators[a][item] for item in shared_items]
            labels_b = [annotators[b][item] for item in shared_items]

            kappa = compute_cohens_kappa(labels_a, labels_b)
            pct = compute_percent_agreement(labels_a, labels_b)

            key = (a, b)
            result[key] = AgreementScore(
                metric_id=metric_id,
                annotator_pair=key,
                cohens_kappa=kappa,
                percent_agreement=pct,
                n_items=len(shared_items),
            )

    return result


def find_disagreement_items(
    labels: list[AnnotatorLabel],
    threshold: float = 0.5,
) -> list[str]:
    """Find items where annotators disagree most (highest label variance).

    Returns item IDs sorted by descending variance of labels across annotators.
    Only items with at least 2 annotations are considered.
    Items where the variance exceeds the threshold are returned.
    """
    # Group by (item_id, metric_id) -> list of labels
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for lbl in labels:
        groups[(lbl.item_id, lbl.metric_id)].append(lbl.label)

    item_variances: dict[str, float] = {}
    for (item_id, _metric_id), vals in groups.items():
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        # Take the max variance across metrics for each item
        if item_id not in item_variances or var > item_variances[item_id]:
            item_variances[item_id] = var

    # Filter by threshold and sort descending
    filtered = [
        (item_id, var)
        for item_id, var in item_variances.items()
        if var >= threshold
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in filtered]


def build_agreement_report(
    labels: list[AnnotatorLabel],
    threshold: float = 0.5,
) -> AgreementReport:
    """Build a complete inter-annotator agreement report."""
    if not labels:
        return AgreementReport(
            scores=[],
            overall_kappa=0.0,
            overall_agreement=0.0,
            high_disagreement_items=[],
            total_items=0,
            total_annotators=0,
        )

    pairwise = build_pairwise_matrix(labels)
    scores = list(pairwise.values())

    if scores:
        overall_kappa = sum(s.cohens_kappa for s in scores) / len(scores)
        overall_agreement = sum(s.percent_agreement for s in scores) / len(scores)
    else:
        overall_kappa = 0.0
        overall_agreement = 0.0

    disagreement_items = find_disagreement_items(labels, threshold=threshold)

    all_items = set(lbl.item_id for lbl in labels)
    all_annotators = set(lbl.annotator_id for lbl in labels)

    return AgreementReport(
        scores=scores,
        overall_kappa=overall_kappa,
        overall_agreement=overall_agreement,
        high_disagreement_items=disagreement_items,
        total_items=len(all_items),
        total_annotators=len(all_annotators),
    )


def format_agreement_report(report: AgreementReport) -> str:
    """Format an AgreementReport for human-readable display."""
    lines: list[str] = []
    lines.append("Inter-Annotator Agreement Report")
    lines.append("=" * 40)
    lines.append(f"Total items: {report.total_items}")
    lines.append(f"Total annotators: {report.total_annotators}")
    lines.append(f"Overall Kappa: {report.overall_kappa:.4f}")
    lines.append(f"Overall Agreement: {report.overall_agreement:.2%}")
    lines.append("")

    if report.scores:
        lines.append("Pairwise Scores:")
        lines.append("-" * 40)
        for s in report.scores:
            pair = f"{s.annotator_pair[0]} vs {s.annotator_pair[1]}"
            lines.append(
                f"  {pair} ({s.metric_id}): "
                f"kappa={s.cohens_kappa:.4f}, "
                f"agreement={s.percent_agreement:.2%}, "
                f"n={s.n_items}"
            )
        lines.append("")

    if report.high_disagreement_items:
        lines.append("High Disagreement Items:")
        lines.append("-" * 40)
        for item_id in report.high_disagreement_items:
            lines.append(f"  - {item_id}")
        lines.append("")

    return "\n".join(lines)
