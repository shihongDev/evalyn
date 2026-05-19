"""
Regression simulation: re-generate past failure inputs to verify fixes.

Groups historical failures by reason, generates input variants via
deterministic text transforms, then scores per-pattern fix rates.
Pure Python - no external deps, no LLM calls.
"""

from __future__ import annotations

import random
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Built-in synonym map for variant generation
# ---------------------------------------------------------------------------

_SYNONYM_MAP: Dict[str, List[str]] = {
    "good": ["great", "fine", "decent"],
    "bad": ["poor", "terrible", "awful"],
    "help": ["assist", "support", "aid"],
    "error": ["mistake", "fault", "bug"],
    "show": ["display", "present", "reveal"],
    "get": ["obtain", "fetch", "retrieve"],
    "make": ["create", "build", "produce"],
    "use": ["utilize", "employ", "apply"],
    "find": ["locate", "discover", "search"],
    "big": ["large", "huge", "massive"],
    "small": ["tiny", "little", "minor"],
    "fast": ["quick", "rapid", "swift"],
    "slow": ["sluggish", "gradual", "unhurried"],
    "start": ["begin", "initiate", "launch"],
    "stop": ["halt", "cease", "end"],
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class FailurePattern:
    """A cluster of failures sharing the same root reason."""

    pattern_id: str
    description: str
    example_inputs: List[str]
    category: str
    frequency: int = 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "example_inputs": list(self.example_inputs),
            "category": self.category,
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FailurePattern:
        return cls(
            pattern_id=data["pattern_id"],
            description=data["description"],
            example_inputs=data.get("example_inputs", []),
            category=data.get("category", ""),
            frequency=data.get("frequency", 1),
        )


@dataclass
class RegressionInput:
    """A generated variant of a failure pattern input."""

    input_text: str
    source_pattern_id: str
    variant_index: int
    tags: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "input_text": self.input_text,
            "source_pattern_id": self.source_pattern_id,
            "variant_index": self.variant_index,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RegressionInput:
        return cls(
            input_text=data["input_text"],
            source_pattern_id=data["source_pattern_id"],
            variant_index=data.get("variant_index", 0),
            tags=data.get("tags", []),
        )


@dataclass
class RegressionResult:
    """Per-pattern regression test result."""

    pattern_id: str
    total_variants: int
    passed_count: int
    failed_count: int
    fix_rate: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "total_variants": self.total_variants,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "fix_rate": self.fix_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RegressionResult:
        return cls(
            pattern_id=data["pattern_id"],
            total_variants=data.get("total_variants", 0),
            passed_count=data.get("passed_count", 0),
            failed_count=data.get("failed_count", 0),
            fix_rate=data.get("fix_rate", 0.0),
        )


@dataclass
class RegressionReport:
    """Aggregate regression report across all patterns."""

    results: List[RegressionResult]
    overall_fix_rate: float
    fully_fixed_patterns: int
    still_failing_patterns: int
    total_patterns: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "overall_fix_rate": self.overall_fix_rate,
            "fully_fixed_patterns": self.fully_fixed_patterns,
            "still_failing_patterns": self.still_failing_patterns,
            "total_patterns": self.total_patterns,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RegressionReport:
        return cls(
            results=[RegressionResult.from_dict(r) for r in data.get("results", [])],
            overall_fix_rate=data.get("overall_fix_rate", 0.0),
            fully_fixed_patterns=data.get("fully_fixed_patterns", 0),
            still_failing_patterns=data.get("still_failing_patterns", 0),
            total_patterns=data.get("total_patterns", 0),
        )


# ---------------------------------------------------------------------------
# Text Transform Helpers
# ---------------------------------------------------------------------------


def _shuffle_words(text: str, rng: random.Random) -> str:
    """Shuffle word order while preserving punctuation at boundaries."""
    words = text.split()
    if len(words) <= 1:
        return text
    rng.shuffle(words)
    return " ".join(words)


def _change_case(text: str, rng: random.Random) -> str:
    """Randomly change to upper, lower, or title case."""
    choice = rng.randint(0, 2)
    if choice == 0:
        return text.upper()
    elif choice == 1:
        return text.lower()
    else:
        return text.title()


def _add_punctuation(text: str, rng: random.Random) -> str:
    """Add or change trailing punctuation."""
    text = text.rstrip(string.punctuation + " ")
    marks = [".", "!", "?", "...", "!!"]
    return text + rng.choice(marks)


def _remove_punctuation(text: str, _rng: random.Random) -> str:
    """Strip all punctuation from text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def _substitute_synonyms(text: str, rng: random.Random) -> str:
    """Replace words with synonyms from the built-in map."""
    words = text.split()
    result = []
    for word in words:
        lower = word.lower()
        if lower in _SYNONYM_MAP:
            replacement = rng.choice(_SYNONYM_MAP[lower])
            # Preserve original casing style
            if word.isupper():
                replacement = replacement.upper()
            elif word[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement)
        else:
            result.append(word)
    return " ".join(result)


_TRANSFORMS = [
    _shuffle_words,
    _change_case,
    _add_punctuation,
    _remove_punctuation,
    _substitute_synonyms,
]


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def extract_failure_patterns(
    failures: List[Dict[str, Any]],
    text_field: str = "input",
    reason_field: str = "failure_reason",
) -> List[FailurePattern]:
    """Group failures by reason and extract patterns with example inputs.

    Each unique reason becomes a FailurePattern. The pattern_id is derived
    from the reason text (slugified). Example inputs are collected from the
    text_field of matching failure records.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for failure in failures:
        reason = failure.get(reason_field, "unknown")
        text = failure.get(text_field, "")
        groups[reason].append(text)

    patterns: List[FailurePattern] = []
    for reason, inputs in sorted(groups.items()):
        slug = re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_")
        if not slug:
            slug = "unknown"
        pattern = FailurePattern(
            pattern_id=slug,
            description=reason,
            example_inputs=inputs,
            category=_categorize_reason(reason),
            frequency=len(inputs),
        )
        patterns.append(pattern)
    return patterns


def _categorize_reason(reason: str) -> str:
    """Derive a broad category from a failure reason string."""
    lower = reason.lower()
    if any(kw in lower for kw in ["timeout", "slow", "latency"]):
        return "performance"
    if any(kw in lower for kw in ["format", "parse", "syntax", "schema"]):
        return "formatting"
    if any(kw in lower for kw in ["null", "none", "empty", "missing"]):
        return "missing_data"
    if any(kw in lower for kw in ["invalid", "wrong", "incorrect", "error"]):
        return "validation"
    return "other"


def generate_regression_variants(
    pattern: FailurePattern,
    n_variants: int = 3,
    rng: random.Random | None = None,
) -> List[RegressionInput]:
    """Generate n_variants variants for each example input in a pattern.

    Applies deterministic text transforms: word shuffling, case changes,
    punctuation edits, and synonym substitution.
    """
    if rng is None:
        rng = random.Random(42)

    variants: List[RegressionInput] = []
    variant_idx = 0

    for example in pattern.example_inputs:
        if not example:
            continue
        for i in range(n_variants):
            # Pick a transform combination based on variant index
            transform = _TRANSFORMS[i % len(_TRANSFORMS)]
            transformed = transform(example, rng)

            # Apply a second transform for more diversity on later variants
            if i >= len(_TRANSFORMS):
                second = _TRANSFORMS[(i + 1) % len(_TRANSFORMS)]
                transformed = second(transformed, rng)

            tags = ["regression", f"variant_{i}", pattern.category]
            variants.append(
                RegressionInput(
                    input_text=transformed,
                    source_pattern_id=pattern.pattern_id,
                    variant_index=variant_idx,
                    tags=tags,
                )
            )
            variant_idx += 1

    return variants


def generate_regression_suite(
    patterns: List[FailurePattern],
    n_variants: int = 3,
) -> List[RegressionInput]:
    """Generate regression variants for all patterns.

    Uses a shared RNG seeded at 42 for reproducibility.
    """
    rng = random.Random(42)
    suite: List[RegressionInput] = []
    for pattern in patterns:
        suite.extend(generate_regression_variants(pattern, n_variants, rng))
    return suite


def compute_regression_results(
    inputs: List[RegressionInput],
    scores: Dict[str, float],
) -> List[RegressionResult]:
    """Compute per-pattern fix rates from input scores.

    The scores dict maps input_text to a float score. A score >= 0.5
    counts as passed.
    """
    pattern_counts: Dict[str, Tuple[int, int]] = {}  # pattern_id -> (passed, total)

    for inp in inputs:
        pid = inp.source_pattern_id
        score = scores.get(inp.input_text, 0.0)
        passed = 1 if score >= 0.5 else 0
        prev_passed, prev_total = pattern_counts.get(pid, (0, 0))
        pattern_counts[pid] = (prev_passed + passed, prev_total + 1)

    results: List[RegressionResult] = []
    for pid in sorted(pattern_counts):
        passed, total = pattern_counts[pid]
        failed = total - passed
        fix_rate = passed / total if total > 0 else 0.0
        results.append(
            RegressionResult(
                pattern_id=pid,
                total_variants=total,
                passed_count=passed,
                failed_count=failed,
                fix_rate=fix_rate,
            )
        )
    return results


def build_regression_report(
    results: List[RegressionResult],
) -> RegressionReport:
    """Aggregate per-pattern results into a report."""
    total_patterns = len(results)
    fully_fixed = sum(1 for r in results if r.fix_rate >= 1.0)
    still_failing = sum(1 for r in results if r.fix_rate < 1.0)

    total_passed = sum(r.passed_count for r in results)
    total_variants = sum(r.total_variants for r in results)
    overall_fix_rate = total_passed / total_variants if total_variants > 0 else 0.0

    return RegressionReport(
        results=results,
        overall_fix_rate=overall_fix_rate,
        fully_fixed_patterns=fully_fixed,
        still_failing_patterns=still_failing,
        total_patterns=total_patterns,
    )


def format_regression_report(report: RegressionReport) -> str:
    """Format a regression report as human-readable text."""
    lines: List[str] = []
    lines.append("REGRESSION REPORT")
    lines.append("=" * 50)
    lines.append(f"Total patterns: {report.total_patterns}")
    lines.append(f"Overall fix rate: {report.overall_fix_rate:.1%}")
    lines.append(f"Fully fixed: {report.fully_fixed_patterns}")
    lines.append(f"Still failing: {report.still_failing_patterns}")
    lines.append("")
    lines.append("PER-PATTERN BREAKDOWN")
    lines.append("-" * 50)

    for result in report.results:
        status = "FIXED" if result.fix_rate >= 1.0 else "FAILING"
        lines.append(
            f"  [{status}] {result.pattern_id}: "
            f"{result.passed_count}/{result.total_variants} passed "
            f"({result.fix_rate:.1%})"
        )

    return "\n".join(lines)


def identify_recurring_failures(
    report: RegressionReport,
    threshold: float = 0.5,
) -> List[str]:
    """Return pattern_ids still below the given fix rate threshold."""
    return [r.pattern_id for r in report.results if r.fix_rate < threshold]
