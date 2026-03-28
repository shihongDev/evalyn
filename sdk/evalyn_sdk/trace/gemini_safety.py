"""Gemini safety rating capture - extract and store safety ratings from Gemini responses."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import Span

SAFETY_CATEGORIES = ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
PROBABILITY_LEVELS = ["NEGLIGIBLE", "LOW", "MEDIUM", "HIGH"]


@dataclass
class SafetyRating:
    """A single safety rating from a Gemini response."""

    category: str  # HARASSMENT, HATE_SPEECH, SEXUALLY_EXPLICIT, DANGEROUS_CONTENT
    probability: str  # NEGLIGIBLE, LOW, MEDIUM, HIGH
    blocked: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "probability": self.probability,
            "blocked": self.blocked,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SafetyRating:
        return cls(
            category=data.get("category", ""),
            probability=data.get("probability", "NEGLIGIBLE"),
            blocked=data.get("blocked", False),
        )


@dataclass
class SafetyReport:
    """Aggregated safety analysis from a set of ratings."""

    ratings: List[SafetyRating] = field(default_factory=list)
    has_block: bool = False
    overall_safe: bool = True
    blocked_categories: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ratings": [r.as_dict() for r in self.ratings],
            "has_block": self.has_block,
            "overall_safe": self.overall_safe,
            "blocked_categories": list(self.blocked_categories),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SafetyReport:
        return cls(
            ratings=[SafetyRating.from_dict(r) for r in data.get("ratings", [])],
            has_block=data.get("has_block", False),
            overall_safe=data.get("overall_safe", True),
            blocked_categories=data.get("blocked_categories", []),
        )

    def format_text(self) -> str:
        lines = ["Safety Report:"]
        if not self.ratings:
            lines.append("  No ratings available")
            return "\n".join(lines)
        lines.append(f"  Overall safe: {self.overall_safe}")
        lines.append(f"  Has block: {self.has_block}")
        if self.blocked_categories:
            lines.append(f"  Blocked categories: {', '.join(self.blocked_categories)}")
        lines.append("  Ratings:")
        for r in self.ratings:
            blocked_tag = " [BLOCKED]" if r.blocked else ""
            lines.append(f"    {r.category}: {r.probability}{blocked_tag}")
        return "\n".join(lines)


def extract_safety_ratings(response_data: Dict[str, Any]) -> List[SafetyRating]:
    """Extract safetyRatings from a Gemini response dict."""
    candidates = response_data.get("candidates", [{}])
    if not candidates:
        return []
    first_candidate = candidates[0]
    raw_ratings = first_candidate.get("safetyRatings", [])
    ratings = []
    for raw in raw_ratings:
        category = raw.get("category", "")
        probability = raw.get("probability", "NEGLIGIBLE")
        blocked = raw.get("blocked", False)
        ratings.append(SafetyRating(category=category, probability=probability, blocked=blocked))
    return ratings


def analyze_safety(ratings: List[SafetyRating]) -> SafetyReport:
    """Build a SafetyReport from a list of ratings."""
    has_block = any(r.blocked for r in ratings)
    blocked_categories = [r.category for r in ratings if r.blocked]
    overall_safe = not has_block and not any(r.probability == "HIGH" for r in ratings)
    return SafetyReport(
        ratings=list(ratings),
        has_block=has_block,
        overall_safe=overall_safe,
        blocked_categories=blocked_categories,
    )


def inject_safety_into_span(span: Span, ratings: List[SafetyRating]) -> Span:
    """Add safety ratings to span attributes under 'gemini.safety_ratings'. Returns a new span."""
    new_span = copy.deepcopy(span)
    new_span.attributes["gemini.safety_ratings"] = [r.as_dict() for r in ratings]
    return new_span


def extract_safety_from_span(span: Span) -> Optional[SafetyReport]:
    """Extract and analyze safety ratings from span attributes."""
    raw = span.attributes.get("gemini.safety_ratings")
    if raw is None:
        return None
    ratings = [SafetyRating.from_dict(r) for r in raw]
    return analyze_safety(ratings)


def is_safe_response(ratings: List[SafetyRating], max_probability: str = "MEDIUM") -> bool:
    """True if no rating exceeds max_probability level and none are blocked."""
    max_index = PROBABILITY_LEVELS.index(max_probability)
    for r in ratings:
        if r.blocked:
            return False
        if r.probability in PROBABILITY_LEVELS:
            level_index = PROBABILITY_LEVELS.index(r.probability)
            if level_index > max_index:
                return False
    return True
