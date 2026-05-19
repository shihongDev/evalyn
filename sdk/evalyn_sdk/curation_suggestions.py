"""
Dataset curation suggestions via gap analysis.

Provides heuristic analysis of dataset items to identify coverage gaps
and constructs LLM prompts for filling them. Pure Python, no external
dependencies, no actual LLM calls.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class CurationGap:
    """A gap identified in the dataset."""

    gap_type: str  # "topic", "difficulty", "length", "category"
    description: str
    severity: str = "medium"
    suggested_count: int = 5

    def as_dict(self) -> dict[str, Any]:
        return {
            "gap_type": self.gap_type,
            "description": self.description,
            "severity": self.severity,
            "suggested_count": self.suggested_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurationGap:
        return cls(
            gap_type=data["gap_type"],
            description=data["description"],
            severity=data.get("severity", "medium"),
            suggested_count=data.get("suggested_count", 5),
        )


@dataclass
class CurationSuggestion:
    """A suggestion for improving the dataset."""

    suggestion_id: str
    gap: CurationGap
    action: str
    example_prompt: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "gap": self.gap.as_dict(),
            "action": self.action,
            "example_prompt": self.example_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurationSuggestion:
        return cls(
            suggestion_id=data["suggestion_id"],
            gap=CurationGap.from_dict(data["gap"]),
            action=data["action"],
            example_prompt=data.get("example_prompt", ""),
        )


@dataclass
class CurationReport:
    """Report summarizing dataset curation suggestions."""

    suggestions: list[CurationSuggestion] = field(default_factory=list)
    total_items: int = 0
    gaps_found: int = 0
    gaps: list[CurationGap] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "suggestions": [s.as_dict() for s in self.suggestions],
            "total_items": self.total_items,
            "gaps_found": self.gaps_found,
            "gaps": [g.as_dict() for g in self.gaps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurationReport:
        return cls(
            suggestions=[
                CurationSuggestion.from_dict(s) for s in data.get("suggestions", [])
            ],
            total_items=data.get("total_items", 0),
            gaps_found=data.get("gaps_found", 0),
            gaps=[CurationGap.from_dict(g) for g in data.get("gaps", [])],
        )


# ---------------------------------------------------------------------------
# Stopwords and helpers
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "it",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "and",
        "or",
        "but",
        "not",
        "with",
        "by",
        "from",
        "as",
        "be",
        "was",
        "are",
        "were",
        "been",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "if",
        "then",
        "so",
        "no",
        "yes",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "about",
    }
)

_WORD_RE = re.compile(r"[a-zA-Z]{2,}")


def _tokenize(text: str) -> list[str]:
    """Extract lowercase words, filtering stopwords."""
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _word_complexity(text: str) -> float:
    """Estimate text complexity as average word length (chars)."""
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def _make_suggestion_id() -> str:
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------


def analyze_topic_coverage(
    items: dict[str, str],
    min_cluster_size: int = 3,
) -> list[CurationGap]:
    """Find underrepresented topics using word frequency analysis.

    Identifies words that appear in fewer than min_cluster_size items
    but occur multiple times overall, suggesting thin topic coverage.
    """
    if not items:
        return []

    # Count how many items each word appears in (document frequency)
    doc_freq: Counter[str] = Counter()
    # Total word occurrences
    word_freq: Counter[str] = Counter()

    for text in items.values():
        tokens = _tokenize(text)
        unique_tokens = set(tokens)
        for t in unique_tokens:
            doc_freq[t] += 1
        word_freq.update(tokens)

    gaps: list[CurationGap] = []

    # Find words that are mentioned overall but in few items
    for word, total_count in word_freq.most_common():
        df = doc_freq[word]
        if total_count >= 2 and df < min_cluster_size and df < len(items):
            severity = "high" if total_count >= 4 else "medium"
            gaps.append(
                CurationGap(
                    gap_type="topic",
                    description=(
                        f"Topic '{word}' appears {total_count} times "
                        f"but only in {df} item(s) - underrepresented"
                    ),
                    severity=severity,
                    suggested_count=min_cluster_size - df,
                )
            )

    # Cap the number of gaps to the most relevant ones
    return gaps[:20]


def analyze_difficulty_distribution(
    items: dict[str, str],
) -> list[CurationGap]:
    """Check if difficulty levels are balanced using word complexity heuristics.

    Classifies items into easy (avg word len < 4.5), medium (4.5-6),
    and hard (> 6), then flags imbalance.
    """
    if not items:
        return []

    buckets: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    for text in items.values():
        complexity = _word_complexity(text)
        if complexity < 4.5:
            buckets["easy"] += 1
        elif complexity <= 6.0:
            buckets["medium"] += 1
        else:
            buckets["hard"] += 1

    total = len(items)
    if total == 0:
        return []

    gaps: list[CurationGap] = []
    ideal_fraction = 1.0 / 3.0

    for level, count in buckets.items():
        fraction = count / total
        if fraction < ideal_fraction * 0.5:
            deficit = max(1, round(ideal_fraction * total - count))
            gaps.append(
                CurationGap(
                    gap_type="difficulty",
                    description=(
                        f"'{level}' difficulty underrepresented: "
                        f"{count}/{total} items ({fraction:.0%})"
                    ),
                    severity="high" if fraction < ideal_fraction * 0.25 else "medium",
                    suggested_count=deficit,
                )
            )
        elif fraction > ideal_fraction * 1.8:
            gaps.append(
                CurationGap(
                    gap_type="difficulty",
                    description=(
                        f"'{level}' difficulty overrepresented: "
                        f"{count}/{total} items ({fraction:.0%})"
                    ),
                    severity="low",
                    suggested_count=0,
                )
            )

    return gaps


def analyze_length_distribution(
    items: dict[str, str],
) -> list[CurationGap]:
    """Check for length imbalance in dataset items.

    Classifies items as short (< 50 chars), medium (50-200), or long (> 200),
    then flags imbalance.
    """
    if not items:
        return []

    buckets: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    for text in items.values():
        length = len(text)
        if length < 50:
            buckets["short"] += 1
        elif length <= 200:
            buckets["medium"] += 1
        else:
            buckets["long"] += 1

    total = len(items)
    if total == 0:
        return []

    gaps: list[CurationGap] = []
    ideal_fraction = 1.0 / 3.0

    for level, count in buckets.items():
        fraction = count / total
        if fraction < ideal_fraction * 0.5:
            deficit = max(1, round(ideal_fraction * total - count))
            gaps.append(
                CurationGap(
                    gap_type="length",
                    description=(
                        f"'{level}' length underrepresented: "
                        f"{count}/{total} items ({fraction:.0%})"
                    ),
                    severity="high" if fraction < ideal_fraction * 0.25 else "medium",
                    suggested_count=deficit,
                )
            )
        elif fraction > ideal_fraction * 1.8:
            gaps.append(
                CurationGap(
                    gap_type="length",
                    description=(
                        f"'{level}' length overrepresented: "
                        f"{count}/{total} items ({fraction:.0%})"
                    ),
                    severity="low",
                    suggested_count=0,
                )
            )

    return gaps


def build_curation_prompt(
    gaps: list[CurationGap],
    existing_items: list[str],
) -> str:
    """Construct an LLM prompt asking to generate items that fill the gaps.

    Returns the prompt string - does not call any LLM.
    """
    lines: list[str] = []
    lines.append("You are a dataset curator. Generate new evaluation items to fill ")
    lines.append("the following gaps in the dataset.\n")
    lines.append("")

    lines.append("## Identified Gaps\n")
    for i, gap in enumerate(gaps, 1):
        lines.append(
            f"{i}. [{gap.gap_type}] {gap.description} "
            f"(severity: {gap.severity}, suggested: {gap.suggested_count} items)"
        )
    lines.append("")

    if existing_items:
        lines.append("## Existing Items (sample)\n")
        sample = existing_items[:10]
        for item in sample:
            preview = item[:100]
            if len(item) > 100:
                preview += "..."
            lines.append(f"- {preview}")
        lines.append("")

    lines.append("## Instructions\n")
    lines.append("Generate new items that:")
    lines.append("1. Address the identified gaps above")
    lines.append("2. Are diverse and distinct from existing items")
    lines.append("3. Match the general format and style of the existing dataset")
    lines.append("")
    lines.append("Return the items as a JSON array of strings.")

    return "\n".join(lines)


def generate_curation_report(
    items: dict[str, str],
) -> CurationReport:
    """Run all analyses and produce a curation report with suggestions."""
    all_gaps: list[CurationGap] = []
    all_gaps.extend(analyze_topic_coverage(items))
    all_gaps.extend(analyze_difficulty_distribution(items))
    all_gaps.extend(analyze_length_distribution(items))

    suggestions: list[CurationSuggestion] = []
    for gap in all_gaps:
        action = _action_for_gap(gap)
        example_prompt = ""
        if gap.suggested_count > 0:
            example_prompt = build_curation_prompt([gap], list(items.values())[:5])
        suggestions.append(
            CurationSuggestion(
                suggestion_id=_make_suggestion_id(),
                gap=gap,
                action=action,
                example_prompt=example_prompt,
            )
        )

    return CurationReport(
        suggestions=suggestions,
        total_items=len(items),
        gaps_found=len(all_gaps),
        gaps=all_gaps,
    )


def _action_for_gap(gap: CurationGap) -> str:
    """Generate a human-readable action for a gap."""
    if gap.gap_type == "topic":
        return f"Add {gap.suggested_count} items covering this topic"
    elif gap.gap_type == "difficulty":
        if gap.suggested_count > 0:
            return f"Add {gap.suggested_count} items at this difficulty level"
        return "Consider rebalancing difficulty distribution"
    elif gap.gap_type == "length":
        if gap.suggested_count > 0:
            return f"Add {gap.suggested_count} items of this length"
        return "Consider rebalancing length distribution"
    elif gap.gap_type == "category":
        return f"Add {gap.suggested_count} items in this category"
    return f"Address gap: {gap.description}"


def format_curation_report(report: CurationReport) -> str:
    """Format a curation report as human-readable text."""
    lines: list[str] = []
    lines.append("Dataset Curation Report")
    lines.append("=" * 40)
    lines.append(f"Total items analyzed: {report.total_items}")
    lines.append(f"Gaps found: {report.gaps_found}")
    lines.append("")

    if not report.suggestions:
        lines.append("No gaps identified - dataset looks well balanced.")
        return "\n".join(lines)

    lines.append("Suggestions:")
    for i, s in enumerate(report.suggestions, 1):
        lines.append(f"  {i}. [{s.gap.gap_type}] {s.gap.description}")
        lines.append(f"     Severity: {s.gap.severity}")
        lines.append(f"     Action: {s.action}")
        if s.example_prompt:
            lines.append("     (LLM prompt available)")
    lines.append("")

    return "\n".join(lines)
