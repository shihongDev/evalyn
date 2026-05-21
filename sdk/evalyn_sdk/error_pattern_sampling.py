"""Error-pattern sampling for preferential selection of failure-matching items.

Uses word-set Jaccard as a similarity proxy to match items against known
failure patterns. Pure Python, no external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ErrorPattern:
    """A known failure pattern with keyword descriptors."""

    pattern_id: str
    description: str
    keywords: list[str] = field(default_factory=list)
    severity: str = "medium"

    def as_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "keywords": self.keywords,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErrorPattern:
        return cls(
            pattern_id=data["pattern_id"],
            description=data["description"],
            keywords=data.get("keywords", []),
            severity=data.get("severity", "medium"),
        )


@dataclass
class PatternMatch:
    """A match between an item and an error pattern."""

    item_id: str
    pattern_id: str
    match_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "pattern_id": self.pattern_id,
            "match_score": self.match_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PatternMatch:
        return cls(
            item_id=data["item_id"],
            pattern_id=data["pattern_id"],
            match_score=data["match_score"],
        )


@dataclass
class ErrorPatternResult:
    """Result of error-pattern sampling."""

    selected_ids: list[str] = field(default_factory=list)
    matches: list[PatternMatch] = field(default_factory=list)
    patterns_covered: int = 0
    total_patterns: int = 0
    total_pool: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_ids": self.selected_ids,
            "matches": [m.as_dict() for m in self.matches],
            "patterns_covered": self.patterns_covered,
            "total_patterns": self.total_patterns,
            "total_pool": self.total_pool,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErrorPatternResult:
        return cls(
            selected_ids=data.get("selected_ids", []),
            matches=[PatternMatch.from_dict(m) for m in data.get("matches", [])],
            patterns_covered=data.get("patterns_covered", 0),
            total_patterns=data.get("total_patterns", 0),
            total_pool=data.get("total_pool", 0),
        )


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase word tokens."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def extract_patterns_from_failures(
    failures: list[dict],
    reason_field: str = "failure_reason",
) -> list[ErrorPattern]:
    """Group failures by reason and extract keyword patterns.

    Each unique reason string becomes one ErrorPattern. Keywords are the
    distinct tokens (lowercased words) extracted from that reason.
    """
    groups: dict[str, list[dict]] = {}
    for f in failures:
        reason = str(f.get(reason_field, ""))
        if not reason:
            continue
        if reason not in groups:
            groups[reason] = []
        groups[reason].append(f)

    patterns: list[ErrorPattern] = []
    for idx, (reason, items) in enumerate(sorted(groups.items())):
        keywords = sorted(_tokenize(reason))
        if not keywords:
            continue
        count = len(items)
        if count >= 5:
            severity = "high"
        elif count >= 2:
            severity = "medium"
        else:
            severity = "low"
        patterns.append(
            ErrorPattern(
                pattern_id=f"p{idx}",
                description=reason,
                keywords=keywords,
                severity=severity,
            )
        )
    return patterns


def compute_pattern_match(text: str, pattern: ErrorPattern) -> float:
    """Fraction of pattern keywords found in text (case-insensitive).

    Returns 0.0 if the pattern has no keywords.
    """
    if not pattern.keywords:
        return 0.0
    text_tokens = _tokenize(text)
    matched = sum(1 for kw in pattern.keywords if kw.lower() in text_tokens)
    return matched / len(pattern.keywords)


def match_items_to_patterns(
    items: dict[str, str],
    patterns: list[ErrorPattern],
    min_match: float = 0.3,
) -> list[PatternMatch]:
    """Find all item-pattern matches above the threshold.

    Args:
        items: mapping of item_id to text content.
        patterns: list of error patterns to match against.
        min_match: minimum match score to include.

    Returns:
        List of PatternMatch objects sorted by match_score descending.
    """
    matches: list[PatternMatch] = []
    for item_id in sorted(items.keys()):
        text = items[item_id]
        for pattern in patterns:
            score = compute_pattern_match(text, pattern)
            if score >= min_match:
                matches.append(
                    PatternMatch(
                        item_id=item_id,
                        pattern_id=pattern.pattern_id,
                        match_score=score,
                    )
                )
    matches.sort(key=lambda m: m.match_score, reverse=True)
    return matches


def sample_by_error_patterns(
    matches: list[PatternMatch],
    sample_size: int = 50,
    seed: int | None = None,
) -> list[str]:
    """Select items ensuring at least 1 per matched pattern, then fill by score.

    Phase 1: for each pattern that has matches, pick the highest-scoring item.
    Phase 2: fill remaining budget from all matches sorted by score, skipping
    items already selected.

    Returns at most sample_size item IDs.
    """
    if not matches:
        return []

    selected: list[str] = []
    selected_set: set[str] = set()

    # Phase 1: one item per pattern (highest score)
    pattern_best: dict[str, PatternMatch] = {}
    for m in matches:
        if (
            m.pattern_id not in pattern_best
            or m.match_score > pattern_best[m.pattern_id].match_score
        ):
            pattern_best[m.pattern_id] = m

    for pid in sorted(pattern_best.keys()):
        if len(selected) >= sample_size:
            break
        best = pattern_best[pid]
        if best.item_id not in selected_set:
            selected.append(best.item_id)
            selected_set.add(best.item_id)

    # Phase 2: fill by descending score
    for m in matches:
        if len(selected) >= sample_size:
            break
        if m.item_id not in selected_set:
            selected.append(m.item_id)
            selected_set.add(m.item_id)

    return selected


def run_error_pattern_sampling(
    items: dict[str, str],
    patterns: list[ErrorPattern],
    sample_size: int = 50,
    seed: int | None = None,
) -> ErrorPatternResult:
    """Full error-pattern sampling pipeline.

    1. Match items to patterns
    2. Sample prioritizing pattern coverage
    3. Return result with coverage stats
    """
    matches = match_items_to_patterns(items, patterns)
    selected = sample_by_error_patterns(matches, sample_size=sample_size, seed=seed)

    # Count how many distinct patterns are covered by selected items
    covered_patterns: set[str] = set()
    selected_set = set(selected)
    selected_matches = [m for m in matches if m.item_id in selected_set]
    for m in selected_matches:
        covered_patterns.add(m.pattern_id)

    return ErrorPatternResult(
        selected_ids=selected,
        matches=selected_matches,
        patterns_covered=len(covered_patterns),
        total_patterns=len(patterns),
        total_pool=len(items),
    )


def format_error_pattern_report(result: ErrorPatternResult) -> str:
    """Format a human-readable error-pattern sampling report."""
    lines: list[str] = []
    lines.append("Error Pattern Sampling Report")
    lines.append("=" * 40)
    lines.append(f"Total pool size: {result.total_pool}")
    lines.append(f"Selected items: {len(result.selected_ids)}")
    lines.append(f"Patterns covered: {result.patterns_covered}/{result.total_patterns}")
    lines.append("")
    lines.append("Matches:")
    for m in result.matches:
        lines.append(f"  item={m.item_id}  pattern={m.pattern_id}  score={m.match_score:.2f}")
    lines.append("")
    lines.append("Selected IDs:")
    for item_id in result.selected_ids:
        lines.append(f"  - {item_id}")
    return "\n".join(lines)
