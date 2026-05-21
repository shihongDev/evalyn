"""
Dataset bias auditing: detect systematic biases in input distribution.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Sentiment word lists (simple heuristic)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = frozenset(
    [
        "good",
        "great",
        "excellent",
        "amazing",
        "wonderful",
        "fantastic",
        "best",
        "love",
        "happy",
        "awesome",
        "perfect",
        "brilliant",
        "outstanding",
        "superb",
        "nice",
        "beautiful",
        "positive",
        "joy",
        "pleased",
        "delighted",
    ]
)

_NEGATIVE_WORDS = frozenset(
    [
        "bad",
        "terrible",
        "awful",
        "horrible",
        "worst",
        "hate",
        "ugly",
        "poor",
        "sad",
        "angry",
        "disappointing",
        "negative",
        "dreadful",
        "miserable",
        "failure",
        "broken",
        "useless",
        "annoying",
        "painful",
        "disgusting",
    ]
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class BiasIndicator:
    """Single bias measurement for one dimension."""

    name: str
    value: float
    severity: str = "low"
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "severity": self.severity,
            "description": self.description,
        }


@dataclass
class BiasAuditReport:
    """Aggregated bias audit across all checked dimensions."""

    indicators: list[BiasIndicator] = field(default_factory=list)
    overall_bias_score: float = 0.0
    high_severity_count: int = 0
    recommendations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "indicators": [i.as_dict() for i in self.indicators],
            "overall_bias_score": self.overall_bias_score,
            "high_severity_count": self.high_severity_count,
            "recommendations": list(self.recommendations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BiasAuditReport:
        indicators = [BiasIndicator(**i) for i in data.get("indicators", [])]
        return cls(
            indicators=indicators,
            overall_bias_score=data.get("overall_bias_score", 0.0),
            high_severity_count=data.get("high_severity_count", 0),
            recommendations=data.get("recommendations", []),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append("Bias Audit Report")
        lines.append("-" * 40)
        for ind in self.indicators:
            lines.append(f"  {ind.name}: {ind.value:.4f} [{ind.severity}]")
            if ind.description:
                lines.append(f"    {ind.description}")
        lines.append(f"  Overall bias score: {self.overall_bias_score:.4f}")
        lines.append(f"  High severity count: {self.high_severity_count}")
        if self.recommendations:
            lines.append("  Recommendations:")
            for rec in self.recommendations:
                lines.append(f"    - {rec}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _severity_from_value(value: float, medium_threshold: float, high_threshold: float) -> str:
    """Map a numeric value to low/medium/high severity."""
    if value >= high_threshold:
        return "high"
    if value >= medium_threshold:
        return "medium"
    return "low"


def check_length_bias(texts: list[str]) -> BiasIndicator:
    """Check if input lengths are heavily skewed.

    Uses coefficient of variation (std / mean). CV > 1.0 = high bias.
    """
    if not texts:
        return BiasIndicator(
            name="length_bias",
            value=0.0,
            severity="low",
            description="No texts to analyze",
        )

    lengths = [len(t) for t in texts]
    n = len(lengths)
    mean = sum(lengths) / n
    if mean == 0:
        return BiasIndicator(
            name="length_bias",
            value=0.0,
            severity="low",
            description="All texts are empty",
        )
    variance = sum((length - mean) ** 2 for length in lengths) / n
    std = math.sqrt(variance)
    cv = std / mean

    severity = _severity_from_value(cv, 0.5, 1.0)
    return BiasIndicator(
        name="length_bias",
        value=cv,
        severity=severity,
        description=f"Coefficient of variation: {cv:.4f} (mean length {mean:.1f})",
    )


def check_category_imbalance(categories: list[str]) -> BiasIndicator:
    """Check if categories are highly imbalanced.

    Ratio of max pct to min pct > 5 = high bias.
    """
    if not categories:
        return BiasIndicator(
            name="category_imbalance",
            value=0.0,
            severity="low",
            description="No categories to analyze",
        )

    counts = Counter(categories)
    total = len(categories)
    percentages = [c / total for c in counts.values()]
    max_pct = max(percentages)
    min_pct = min(percentages)

    if min_pct == 0:
        ratio = float("inf")
    else:
        ratio = max_pct / min_pct

    severity = _severity_from_value(ratio, 3.0, 5.0)
    return BiasIndicator(
        name="category_imbalance",
        value=ratio,
        severity=severity,
        description=f"Max/min ratio: {ratio:.2f} ({len(counts)} categories)",
    )


def check_vocabulary_concentration(texts: list[str], top_n: int = 10) -> BiasIndicator:
    """Check if vocabulary is dominated by few words.

    Top-N words covering > 50% of total word count = high bias.
    """
    if not texts:
        return BiasIndicator(
            name="vocabulary_concentration",
            value=0.0,
            severity="low",
            description="No texts to analyze",
        )

    word_counts: Counter[str] = Counter()
    for t in texts:
        word_counts.update(t.lower().split())

    total_words = sum(word_counts.values())
    if total_words == 0:
        return BiasIndicator(
            name="vocabulary_concentration",
            value=0.0,
            severity="low",
            description="No words found",
        )

    top_words = word_counts.most_common(top_n)
    top_count = sum(c for _, c in top_words)
    concentration = top_count / total_words

    severity = _severity_from_value(concentration, 0.3, 0.5)
    return BiasIndicator(
        name="vocabulary_concentration",
        value=concentration,
        severity=severity,
        description=f"Top-{top_n} words cover {concentration:.1%} of total",
    )


def check_sentiment_skew(texts: list[str]) -> BiasIndicator:
    """Simple heuristic: count positive vs negative sentiment words.

    Ratio > 3:1 in either direction = high bias.
    """
    if not texts:
        return BiasIndicator(
            name="sentiment_skew",
            value=0.0,
            severity="low",
            description="No texts to analyze",
        )

    pos_count = 0
    neg_count = 0
    for t in texts:
        words = t.lower().split()
        for w in words:
            if w in _POSITIVE_WORDS:
                pos_count += 1
            elif w in _NEGATIVE_WORDS:
                neg_count += 1

    if pos_count == 0 and neg_count == 0:
        return BiasIndicator(
            name="sentiment_skew",
            value=0.0,
            severity="low",
            description="No sentiment words detected",
        )

    # Ratio of dominant sentiment to the other
    dominant = max(pos_count, neg_count)
    minority = min(pos_count, neg_count)
    if minority == 0:
        ratio = float(dominant)
    else:
        ratio = dominant / minority

    direction = "positive" if pos_count >= neg_count else "negative"
    severity = _severity_from_value(ratio, 2.0, 3.0)
    return BiasIndicator(
        name="sentiment_skew",
        value=ratio,
        severity=severity,
        description=f"Skew toward {direction} ({pos_count} pos, {neg_count} neg, ratio {ratio:.2f})",
    )


def audit_dataset(
    items: list[dict[str, Any]],
    input_field: str = "input",
    category_field: str = "category",
) -> BiasAuditReport:
    """Run all bias checks on a dataset."""
    texts = [str(item.get(input_field, "")) for item in items]
    categories = [item[category_field] for item in items if category_field in item]

    indicators: list[BiasIndicator] = []

    indicators.append(check_length_bias(texts))
    indicators.append(check_vocabulary_concentration(texts))
    indicators.append(check_sentiment_skew(texts))

    if categories:
        indicators.append(check_category_imbalance(categories))

    high_count = sum(1 for ind in indicators if ind.severity == "high")
    severity_scores = {"low": 0.0, "medium": 0.5, "high": 1.0}
    if indicators:
        overall = sum(severity_scores.get(ind.severity, 0.0) for ind in indicators) / len(
            indicators
        )
    else:
        overall = 0.0

    recommendations = suggest_debiasing(
        BiasAuditReport(
            indicators=indicators,
            overall_bias_score=overall,
            high_severity_count=high_count,
        )
    )

    return BiasAuditReport(
        indicators=indicators,
        overall_bias_score=overall,
        high_severity_count=high_count,
        recommendations=recommendations,
    )


def suggest_debiasing(report: BiasAuditReport) -> list[str]:
    """Suggestions to reduce detected biases."""
    suggestions: list[str] = []

    for ind in report.indicators:
        if ind.severity in ("medium", "high"):
            if ind.name == "length_bias":
                suggestions.append(
                    "Consider normalizing input lengths or stratifying by length buckets"
                )
            elif ind.name == "category_imbalance":
                suggestions.append(
                    "Resample underrepresented categories or trim overrepresented ones"
                )
            elif ind.name == "vocabulary_concentration":
                suggestions.append("Diversify input phrasing to reduce vocabulary concentration")
            elif ind.name == "sentiment_skew":
                suggestions.append("Balance positive and negative examples in the dataset")

    if report.high_severity_count >= 2:
        suggestions.append(
            "Multiple high-severity biases detected - consider rebuilding the dataset"
        )

    return suggestions
