"""
Failure taxonomy: auto-categorize failures into structured taxonomy.

Provides heuristic-based classification of evaluation failures
into categories (prompt, model, data, tool, safety, other) with
aggregation, distribution, and fix suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAILURE_CATEGORIES: dict[str, str] = {
    "prompt": "Prompt-related failures (unclear instructions, missing context)",
    "model": "Model-related failures (hallucination, refusal, wrong format)",
    "data": "Data-related failures (corrupted input, edge case, ambiguous)",
    "tool": "Tool-related failures (tool call error, wrong tool, missing tool)",
    "safety": "Safety-related failures (content blocked, policy violation)",
    "other": "Uncategorized failures",
}

_FIX_SUGGESTIONS: dict[str, list[str]] = {
    "prompt": [
        "Add more context or examples to the prompt",
        "Clarify instructions with explicit formatting requirements",
        "Break complex prompts into simpler steps",
    ],
    "model": [
        "Try a more capable model",
        "Add output format constraints",
        "Increase max tokens if output is truncated",
    ],
    "data": [
        "Review edge cases in the dataset",
        "Add input validation or preprocessing",
        "Remove or fix corrupted items",
    ],
    "tool": [
        "Verify tool definitions are complete and correct",
        "Add fallback behavior for tool errors",
        "Check tool availability and permissions",
    ],
    "safety": [
        "Review content policy alignment",
        "Add pre-filtering for sensitive inputs",
        "Adjust safety thresholds if overly strict",
    ],
    "other": [
        "Manually review uncategorized failures",
        "Add more specific classification rules",
        "Check for infrastructure or timeout issues",
    ],
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class FailureCategory:
    """A category of failures with description and example items."""

    name: str
    description: str = ""
    item_count: int = 0
    examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "item_count": self.item_count,
            "examples": list(self.examples),
        }


@dataclass
class CategorizedFailure:
    """A single failure classified into a category."""

    item_id: str
    category: str
    subcategory: str = ""
    confidence: float = 0.0
    evidence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "subcategory": self.subcategory,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class TaxonomyReport:
    """Aggregated taxonomy report across all failures."""

    categories: list[FailureCategory] = field(default_factory=list)
    categorized: list[CategorizedFailure] = field(default_factory=list)
    uncategorized_count: int = 0
    total_failures: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "categories": [c.as_dict() for c in self.categories],
            "categorized": [c.as_dict() for c in self.categorized],
            "uncategorized_count": self.uncategorized_count,
            "total_failures": self.total_failures,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaxonomyReport:
        return cls(
            categories=[
                FailureCategory(
                    name=c["name"],
                    description=c.get("description", ""),
                    item_count=c.get("item_count", 0),
                    examples=c.get("examples", []),
                )
                for c in data.get("categories", [])
            ],
            categorized=[
                CategorizedFailure(
                    item_id=c["item_id"],
                    category=c["category"],
                    subcategory=c.get("subcategory", ""),
                    confidence=c.get("confidence", 0.0),
                    evidence=c.get("evidence", ""),
                )
                for c in data.get("categorized", [])
            ],
            uncategorized_count=data.get("uncategorized_count", 0),
            total_failures=data.get("total_failures", 0),
        )

    def format_text(self) -> str:
        """Format report as human-readable text."""
        lines: list[str] = []
        lines.append(f"Failure Taxonomy ({self.total_failures} total failures)")
        lines.append("-" * 50)
        for cat in self.categories:
            if cat.item_count > 0:
                lines.append(f"  {cat.name}: {cat.item_count} failures")
                if cat.description:
                    lines.append(f"    {cat.description}")
                if cat.examples:
                    for ex in cat.examples[:3]:
                        lines.append(f"    - {ex}")
        if self.uncategorized_count > 0:
            lines.append(f"  uncategorized: {self.uncategorized_count} failures")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_failure(item: dict[str, Any]) -> CategorizedFailure:
    """Classify a single failure item based on heuristics.

    Item expected keys: item_id, output, error, metric_id.

    Rules (checked in order):
    - Empty/None output -> model (refusal)
    - "tool" in error field -> tool
    - metric_id contains "safety" -> safety
    - Short output (< 20 chars) -> prompt (insufficient context)
    - Otherwise -> other
    """
    item_id = item.get("item_id", "unknown")
    output = item.get("output")
    error = item.get("error") or ""
    metric_id = item.get("metric_id") or ""

    # Empty output - model refusal or failure to generate
    if output is None or (isinstance(output, str) and output.strip() == ""):
        return CategorizedFailure(
            item_id=item_id,
            category="model",
            subcategory="refusal",
            confidence=0.8,
            evidence="Empty or missing output",
        )

    # Tool error in error field
    if "tool" in error.lower():
        return CategorizedFailure(
            item_id=item_id,
            category="tool",
            subcategory="tool_error",
            confidence=0.7,
            evidence=f"Tool error: {error[:100]}",
        )

    # Safety metric
    if "safety" in metric_id.lower():
        return CategorizedFailure(
            item_id=item_id,
            category="safety",
            subcategory="policy",
            confidence=0.6,
            evidence=f"Safety metric failed: {metric_id}",
        )

    # Short output - likely prompt issue
    output_str = str(output)
    if len(output_str) < 20:
        return CategorizedFailure(
            item_id=item_id,
            category="prompt",
            subcategory="insufficient_context",
            confidence=0.5,
            evidence=f"Short output ({len(output_str)} chars)",
        )

    # Default
    return CategorizedFailure(
        item_id=item_id,
        category="other",
        subcategory="",
        confidence=0.3,
        evidence="No specific pattern matched",
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def build_taxonomy(failures: list[dict[str, Any]]) -> TaxonomyReport:
    """Classify all failures and aggregate into a taxonomy report."""
    if not failures:
        return TaxonomyReport(total_failures=0)

    categorized: list[CategorizedFailure] = []
    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}

    for item in failures:
        cf = classify_failure(item)
        categorized.append(cf)
        counts[cf.category] = counts.get(cf.category, 0) + 1
        if cf.category not in examples:
            examples[cf.category] = []
        examples[cf.category].append(cf.item_id)

    categories: list[FailureCategory] = []
    for cat_name, cat_desc in FAILURE_CATEGORIES.items():
        count = counts.get(cat_name, 0)
        if count > 0:
            categories.append(FailureCategory(
                name=cat_name,
                description=cat_desc,
                item_count=count,
                examples=examples.get(cat_name, [])[:5],
            ))

    uncategorized = counts.get("other", 0)

    return TaxonomyReport(
        categories=categories,
        categorized=categorized,
        uncategorized_count=uncategorized,
        total_failures=len(failures),
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_category_distribution(report: TaxonomyReport) -> dict[str, int]:
    """Return count per category from a taxonomy report."""
    dist: dict[str, int] = {}
    for cat in report.categories:
        dist[cat.name] = cat.item_count
    return dist


def suggest_category_fixes(category: str) -> list[str]:
    """Return fix suggestions for a given failure category."""
    return list(_FIX_SUGGESTIONS.get(category, _FIX_SUGGESTIONS["other"]))


def render_taxonomy_tree(report: TaxonomyReport) -> str:
    """Render an ASCII tree of failure categories with counts."""
    lines: list[str] = []
    lines.append(f"Failures ({report.total_failures})")

    cats_with_count = [c for c in report.categories if c.item_count > 0]
    for i, cat in enumerate(cats_with_count):
        is_last = i == len(cats_with_count) - 1
        connector = "`--" if is_last else "|--"
        lines.append(f"  {connector} {cat.name} ({cat.item_count})")
        if cat.examples:
            prefix = "      " if is_last else "  |   "
            for j, ex in enumerate(cat.examples[:3]):
                ex_last = j == min(len(cat.examples), 3) - 1
                ex_conn = "`--" if ex_last else "|--"
                lines.append(f"{prefix}{ex_conn} {ex}")

    return "\n".join(lines)
