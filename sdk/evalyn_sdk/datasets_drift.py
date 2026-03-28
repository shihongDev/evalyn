"""
Dataset drift detection: statistical tests comparing input distributions
between dataset versions.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class DriftMetric:
    """Single drift measurement for one dimension."""

    name: str
    value: float
    threshold: float = 0.1
    drifted: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "drifted": self.drifted,
        }


@dataclass
class DriftReport:
    """Aggregated drift report across all checked dimensions."""

    metrics: List[DriftMetric] = field(default_factory=list)
    overall_drifted: bool = False
    drift_score: float = 0.0
    summary: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metrics": [m.as_dict() for m in self.metrics],
            "overall_drifted": self.overall_drifted,
            "drift_score": self.drift_score,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DriftReport:
        metrics = [DriftMetric(**m) for m in data.get("metrics", [])]
        return cls(
            metrics=metrics,
            overall_drifted=data.get("overall_drifted", False),
            drift_score=data.get("drift_score", 0.0),
            summary=data.get("summary", ""),
        )

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append("Drift Report")
        lines.append("-" * 40)
        for m in self.metrics:
            status = "DRIFT" if m.drifted else "ok"
            lines.append(f"  {m.name}: {m.value:.4f} (threshold {m.threshold}) [{status}]")
        lines.append(f"  Overall drifted: {self.overall_drifted}")
        lines.append(f"  Drift score: {self.drift_score:.4f}")
        if self.summary:
            lines.append(f"  Summary: {self.summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def compute_length_distribution(texts: List[str]) -> Dict[str, float]:
    """Compute mean, std, min, max of text lengths."""
    if not texts:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    lengths = [len(t) for t in texts]
    n = len(lengths)
    mean = sum(lengths) / n
    variance = sum((l - mean) ** 2 for l in lengths) / n
    std = math.sqrt(variance)
    return {
        "mean": mean,
        "std": std,
        "min": float(min(lengths)),
        "max": float(max(lengths)),
    }


def compute_vocabulary_overlap(texts_a: List[str], texts_b: List[str]) -> float:
    """Jaccard similarity of unique word sets."""
    words_a: set[str] = set()
    for t in texts_a:
        words_a.update(t.split())
    words_b: set[str] = set()
    for t in texts_b:
        words_b.update(t.split())
    if not words_a and not words_b:
        return 1.0
    union = words_a | words_b
    if not union:
        return 1.0
    intersection = words_a & words_b
    return len(intersection) / len(union)


def detect_length_drift(
    texts_a: List[str],
    texts_b: List[str],
    threshold: float = 0.2,
) -> DriftMetric:
    """Compare mean lengths. Drift = abs(mean_a - mean_b) / max(mean_a, mean_b, 1)."""
    dist_a = compute_length_distribution(texts_a)
    dist_b = compute_length_distribution(texts_b)
    mean_a = dist_a["mean"]
    mean_b = dist_b["mean"]
    denominator = max(mean_a, mean_b, 1.0)
    value = abs(mean_a - mean_b) / denominator
    return DriftMetric(
        name="length_drift",
        value=value,
        threshold=threshold,
        drifted=value >= threshold,
    )


def detect_vocabulary_drift(
    texts_a: List[str],
    texts_b: List[str],
    threshold: float = 0.3,
) -> DriftMetric:
    """Vocabulary overlap below threshold = drift."""
    overlap = compute_vocabulary_overlap(texts_a, texts_b)
    # Drift value is 1 - overlap (higher means more drift)
    value = 1.0 - overlap
    return DriftMetric(
        name="vocabulary_drift",
        value=value,
        threshold=threshold,
        drifted=value >= threshold,
    )


def detect_category_drift(
    categories_a: List[str],
    categories_b: List[str],
    threshold: float = 0.15,
) -> DriftMetric:
    """Compare category distributions. Sum of abs(pct_a - pct_b) for each category."""
    if not categories_a and not categories_b:
        return DriftMetric(name="category_drift", value=0.0, threshold=threshold, drifted=False)
    count_a = Counter(categories_a)
    count_b = Counter(categories_b)
    total_a = len(categories_a) or 1
    total_b = len(categories_b) or 1
    all_cats = set(count_a.keys()) | set(count_b.keys())
    value = 0.0
    for cat in all_cats:
        pct_a = count_a.get(cat, 0) / total_a
        pct_b = count_b.get(cat, 0) / total_b
        value += abs(pct_a - pct_b)
    return DriftMetric(
        name="category_drift",
        value=value,
        threshold=threshold,
        drifted=value >= threshold,
    )


def detect_drift(
    items_a: List[Dict[str, Any]],
    items_b: List[Dict[str, Any]],
    input_field: str = "input",
) -> DriftReport:
    """Run all drift checks on two sets of dataset items."""
    texts_a = [str(item.get(input_field, "")) for item in items_a]
    texts_b = [str(item.get(input_field, "")) for item in items_b]

    metrics: List[DriftMetric] = []

    # Length drift
    metrics.append(detect_length_drift(texts_a, texts_b))

    # Vocabulary drift
    metrics.append(detect_vocabulary_drift(texts_a, texts_b))

    # Category drift (if items have a "category" field)
    cats_a = [item["category"] for item in items_a if "category" in item]
    cats_b = [item["category"] for item in items_b if "category" in item]
    if cats_a or cats_b:
        metrics.append(detect_category_drift(cats_a, cats_b))

    drifted_count = sum(1 for m in metrics if m.drifted)
    overall_drifted = drifted_count > 0
    drift_score = sum(m.value for m in metrics) / len(metrics) if metrics else 0.0

    drifted_names = [m.name for m in metrics if m.drifted]
    if drifted_names:
        summary = f"Drift detected in: {', '.join(drifted_names)}"
    else:
        summary = "No significant drift detected"

    return DriftReport(
        metrics=metrics,
        overall_drifted=overall_drifted,
        drift_score=drift_score,
        summary=summary,
    )
