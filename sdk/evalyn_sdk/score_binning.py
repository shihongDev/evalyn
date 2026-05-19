"""
Configurable score-to-grade mapping.

Provides dataclasses and pure functions for binning numeric scores into
letter grades (or arbitrary grade labels), computing grade distributions,
and formatting human-readable reports.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class GradeMapping:
    """Maps a score range to a grade label."""

    grade: str
    min_score: float
    max_score: float
    label: str = ""
    color_hint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "grade": self.grade,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "label": self.label,
            "color_hint": self.color_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GradeMapping:
        return cls(
            grade=data["grade"],
            min_score=data["min_score"],
            max_score=data["max_score"],
            label=data.get("label", ""),
            color_hint=data.get("color_hint", ""),
        )


@dataclass
class BinningConfig:
    """Configuration for score-to-grade binning."""

    mappings: list[GradeMapping] = field(default_factory=list)
    default_grade: str = "F"

    def as_dict(self) -> dict[str, Any]:
        return {
            "mappings": [m.as_dict() for m in self.mappings],
            "default_grade": self.default_grade,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BinningConfig:
        return cls(
            mappings=[GradeMapping.from_dict(m) for m in data.get("mappings", [])],
            default_grade=data.get("default_grade", "F"),
        )


@dataclass
class BinnedScore:
    """A single item's score mapped to a grade."""

    item_id: str
    score: float
    grade: str
    label: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "score": self.score,
            "grade": self.grade,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BinnedScore:
        return cls(
            item_id=data["item_id"],
            score=data["score"],
            grade=data["grade"],
            label=data.get("label", ""),
        )


@dataclass
class GradeDistribution:
    """Aggregate grade counts across binned scores."""

    grade_counts: dict[str, int] = field(default_factory=dict)
    total: int = 0
    most_common: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "grade_counts": dict(self.grade_counts),
            "total": self.total,
            "most_common": self.most_common,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GradeDistribution:
        return cls(
            grade_counts=dict(data.get("grade_counts", {})),
            total=data.get("total", 0),
            most_common=data.get("most_common", ""),
        )


# ---------------------------------------------------------------------------
# Built-in Configurations
# ---------------------------------------------------------------------------

DEFAULT_BINNING = BinningConfig(
    mappings=[
        GradeMapping(grade="A", min_score=0.9, max_score=1.0),
        GradeMapping(grade="B", min_score=0.7, max_score=0.9),
        GradeMapping(grade="C", min_score=0.5, max_score=0.7),
        GradeMapping(grade="D", min_score=0.3, max_score=0.5),
        GradeMapping(grade="F", min_score=0.0, max_score=0.3),
    ],
    default_grade="F",
)

LETTER_GRADE_BINNING = BinningConfig(
    mappings=[
        GradeMapping(grade="A", min_score=0.9, max_score=1.0, label="Excellent"),
        GradeMapping(grade="B", min_score=0.7, max_score=0.9, label="Good"),
        GradeMapping(grade="C", min_score=0.5, max_score=0.7, label="Fair"),
        GradeMapping(grade="D", min_score=0.3, max_score=0.5, label="Poor"),
        GradeMapping(grade="F", min_score=0.0, max_score=0.3, label="Failing"),
    ],
    default_grade="F",
)

PASS_FAIL_BINNING = BinningConfig(
    mappings=[
        GradeMapping(grade="Pass", min_score=0.5, max_score=1.0),
        GradeMapping(grade="Fail", min_score=0.0, max_score=0.5),
    ],
    default_grade="Fail",
)


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------


def score_to_grade(score: float, config: BinningConfig) -> str:
    """Map a numeric score to its grade string.

    Mappings are checked in order. A score matches a mapping when
    ``min_score <= score <= max_score`` (for the first mapping) or
    ``min_score <= score < max_score`` when ``max_score`` is not the
    overall maximum. The first matching mapping wins.

    Falls back to ``config.default_grade`` when no mapping matches.
    """
    for mapping in config.mappings:
        if mapping.min_score <= score <= mapping.max_score:
            return mapping.grade
    return config.default_grade


def bin_scores(
    scores: dict[str, float],
    config: BinningConfig | None = None,
) -> list[BinnedScore]:
    """Bin a dict of ``{item_id: score}`` into graded results.

    Uses *DEFAULT_BINNING* when *config* is ``None``.
    """
    if config is None:
        config = DEFAULT_BINNING

    results: list[BinnedScore] = []
    for item_id, score in scores.items():
        grade = score_to_grade(score, config)
        label = ""
        for mapping in config.mappings:
            if mapping.grade == grade:
                label = mapping.label
                break
        results.append(BinnedScore(item_id=item_id, score=score, grade=grade, label=label))
    return results


def compute_grade_distribution(binned: list[BinnedScore]) -> GradeDistribution:
    """Compute aggregate grade counts from a list of binned scores."""
    if not binned:
        return GradeDistribution(grade_counts={}, total=0, most_common="")

    counts: Counter[str] = Counter(b.grade for b in binned)
    most_common_grade = counts.most_common(1)[0][0]
    return GradeDistribution(
        grade_counts=dict(counts),
        total=len(binned),
        most_common=most_common_grade,
    )


def format_grade_report(
    binned: list[BinnedScore],
    distribution: GradeDistribution,
) -> str:
    """Return a human-readable grade report string.

    Includes a distribution summary followed by per-item grades.
    """
    lines: list[str] = []
    lines.append("Grade Distribution")
    lines.append("-" * 40)
    for grade, count in sorted(distribution.grade_counts.items()):
        pct = (count / distribution.total * 100) if distribution.total else 0.0
        lines.append(f"  {grade}: {count} ({pct:.1f}%)")
    lines.append(f"  Total: {distribution.total}")
    lines.append(f"  Most common: {distribution.most_common}")
    lines.append("")
    lines.append("Per-Item Grades")
    lines.append("-" * 40)
    for b in binned:
        label_part = f" ({b.label})" if b.label else ""
        lines.append(f"  {b.item_id}: {b.score:.3f} -> {b.grade}{label_part}")
    return "\n".join(lines)


def create_custom_binning(
    grade_specs: list[tuple[str, float, float]],
) -> BinningConfig:
    """Create a BinningConfig from ``(grade, min_score, max_score)`` tuples."""
    mappings = [
        GradeMapping(grade=grade, min_score=mn, max_score=mx)
        for grade, mn, mx in grade_specs
    ]
    return BinningConfig(mappings=mappings)
