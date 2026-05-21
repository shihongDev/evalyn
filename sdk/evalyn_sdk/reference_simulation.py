"""
Simulation with reference answers: manage input-output pairs for golden sets.

Provides the framework for generating, filtering, and exporting reference
answer pairs. Pure Python - no external deps, no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ReferencePair:
    """A single input-output reference pair."""

    input_text: str
    reference_output: str
    confidence: float = 0.0
    quality_passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "reference_output": self.reference_output,
            "confidence": self.confidence,
            "quality_passed": self.quality_passed,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferencePair:
        return cls(
            input_text=data["input_text"],
            reference_output=data["reference_output"],
            confidence=data.get("confidence", 0.0),
            quality_passed=data.get("quality_passed", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class QualityConfig:
    """Configuration for quality filtering of reference pairs."""

    min_confidence: float = 0.8
    min_output_length: int = 10
    max_output_length: int = 5000
    require_non_empty: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_confidence": self.min_confidence,
            "min_output_length": self.min_output_length,
            "max_output_length": self.max_output_length,
            "require_non_empty": self.require_non_empty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityConfig:
        return cls(
            min_confidence=data.get("min_confidence", 0.8),
            min_output_length=data.get("min_output_length", 10),
            max_output_length=data.get("max_output_length", 5000),
            require_non_empty=data.get("require_non_empty", True),
        )


@dataclass
class ReferenceSet:
    """A collection of filtered reference pairs with stats."""

    pairs: list[ReferencePair]
    total_generated: int
    quality_filtered: int
    acceptance_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairs": [p.as_dict() for p in self.pairs],
            "total_generated": self.total_generated,
            "quality_filtered": self.quality_filtered,
            "acceptance_rate": self.acceptance_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceSet:
        return cls(
            pairs=[ReferencePair.from_dict(p) for p in data["pairs"]],
            total_generated=data["total_generated"],
            quality_filtered=data["quality_filtered"],
            acceptance_rate=data["acceptance_rate"],
        )


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def apply_quality_filter(pairs: list[ReferencePair], config: QualityConfig) -> list[ReferencePair]:
    """Filter reference pairs by quality criteria.

    Checks confidence threshold, output length bounds, and non-empty constraint.
    Sets quality_passed=True on pairs that pass.
    """
    result = []
    for pair in pairs:
        # Confidence check
        if pair.confidence < config.min_confidence:
            continue

        output_len = len(pair.reference_output)

        # Non-empty check
        if config.require_non_empty and output_len == 0:
            continue

        # Length bounds
        if output_len < config.min_output_length:
            continue
        if output_len > config.max_output_length:
            continue

        passed_pair = ReferencePair(
            input_text=pair.input_text,
            reference_output=pair.reference_output,
            confidence=pair.confidence,
            quality_passed=True,
            metadata=dict(pair.metadata),
        )
        result.append(passed_pair)
    return result


def build_reference_set(
    pairs: list[ReferencePair], config: QualityConfig | None = None
) -> ReferenceSet:
    """Filter pairs and build a reference set with statistics."""
    if config is None:
        config = QualityConfig()

    total = len(pairs)
    filtered = apply_quality_filter(pairs, config)
    accepted = len(filtered)
    rate = accepted / total if total > 0 else 0.0

    return ReferenceSet(
        pairs=filtered,
        total_generated=total,
        quality_filtered=accepted,
        acceptance_rate=rate,
    )


def score_reference_quality(pair: ReferencePair) -> float:
    """Compute a heuristic quality score for a reference pair.

    Components:
    - Length ratio: output length relative to input (capped contribution)
    - Non-repetitiveness: ratio of unique words to total words
    - Confidence: the pair's confidence value

    Returns a score in [0, 1].
    """
    # Length ratio component: reward outputs that are at least as long as input
    input_len = max(len(pair.input_text), 1)
    output_len = len(pair.reference_output)
    if output_len == 0:
        return 0.0
    length_ratio = min(output_len / input_len, 3.0) / 3.0  # cap at 3x, normalize

    # Non-repetitiveness: unique word ratio
    words = pair.reference_output.split()
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / len(words)

    # Confidence component
    confidence_score = max(0.0, min(pair.confidence, 1.0))

    # Weighted combination
    score = 0.3 * length_ratio + 0.3 * unique_ratio + 0.4 * confidence_score
    return min(max(score, 0.0), 1.0)


def detect_low_quality_pairs(
    pairs: list[ReferencePair], threshold: float = 0.3
) -> list[ReferencePair]:
    """Find pairs below the quality score threshold."""
    return [p for p in pairs if score_reference_quality(p) < threshold]


def export_as_dataset(ref_set: ReferenceSet) -> list[dict[str, Any]]:
    """Convert a reference set to evalyn dataset format.

    Each item has: input, expected_output, metadata.
    """
    items = []
    for pair in ref_set.pairs:
        items.append(
            {
                "input": pair.input_text,
                "expected_output": pair.reference_output,
                "metadata": dict(pair.metadata),
            }
        )
    return items


def format_reference_report(ref_set: ReferenceSet) -> str:
    """Format a reference set as a human-readable report."""
    lines = [
        "Reference Set Report",
        "-" * 40,
        f"Total generated: {ref_set.total_generated}",
        f"Quality filtered: {ref_set.quality_filtered}",
        f"Acceptance rate: {ref_set.acceptance_rate:.1%}",
    ]

    if ref_set.pairs:
        confidences = [p.confidence for p in ref_set.pairs]
        avg_conf = sum(confidences) / len(confidences)
        lengths = [len(p.reference_output) for p in ref_set.pairs]
        avg_len = sum(lengths) / len(lengths)
        scores = [score_reference_quality(p) for p in ref_set.pairs]
        avg_score = sum(scores) / len(scores)

        lines.extend(
            [
                "",
                "Quality distribution:",
                f"  Avg confidence: {avg_conf:.3f}",
                f"  Avg output length: {avg_len:.0f}",
                f"  Avg quality score: {avg_score:.3f}",
            ]
        )

    return "\n".join(lines)


def compute_reference_stats(ref_set: ReferenceSet) -> dict[str, Any]:
    """Compute summary statistics for a reference set."""
    total = ref_set.total_generated
    accepted = ref_set.quality_filtered

    if not ref_set.pairs:
        return {
            "total": total,
            "accepted": accepted,
            "avg_confidence": 0.0,
            "avg_output_length": 0.0,
            "avg_quality_score": 0.0,
        }

    confidences = [p.confidence for p in ref_set.pairs]
    lengths = [len(p.reference_output) for p in ref_set.pairs]
    scores = [score_reference_quality(p) for p in ref_set.pairs]

    return {
        "total": total,
        "accepted": accepted,
        "avg_confidence": sum(confidences) / len(confidences),
        "avg_output_length": sum(lengths) / len(lengths),
        "avg_quality_score": sum(scores) / len(scores),
    }
