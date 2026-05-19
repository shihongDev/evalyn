"""
Root cause analysis for evaluation failures.

Detects patterns in failure data (output length, input complexity,
metric concentration, category concentration) and produces a
prioritized report with suggested fixes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class FailurePattern:
    """A detected pattern explaining a cluster of failures."""

    pattern_name: str
    description: str
    affected_items: List[str] = field(default_factory=list)
    confidence: float = 0.0
    suggested_fix: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pattern_name": self.pattern_name,
            "description": self.description,
            "affected_items": list(self.affected_items),
            "confidence": self.confidence,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class RootCauseReport:
    """Aggregated root cause analysis report."""

    patterns: List[FailurePattern] = field(default_factory=list)
    total_failures: int = 0
    primary_cause: str = ""
    secondary_causes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "patterns": [p.as_dict() for p in self.patterns],
            "total_failures": self.total_failures,
            "primary_cause": self.primary_cause,
            "secondary_causes": list(self.secondary_causes),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RootCauseReport:
        patterns = [
            FailurePattern(
                pattern_name=p["pattern_name"],
                description=p["description"],
                affected_items=p.get("affected_items", []),
                confidence=p.get("confidence", 0.0),
                suggested_fix=p.get("suggested_fix", ""),
            )
            for p in data.get("patterns", [])
        ]
        return cls(
            patterns=patterns,
            total_failures=data.get("total_failures", 0),
            primary_cause=data.get("primary_cause", ""),
            secondary_causes=data.get("secondary_causes", []),
        )

    def format_text(self) -> str:
        """Format the report as human-readable text."""
        lines: List[str] = []
        lines.append(f"Root Cause Report ({self.total_failures} failures)")
        lines.append("-" * 40)

        if self.primary_cause:
            lines.append(f"Primary cause: {self.primary_cause}")

        if self.secondary_causes:
            lines.append(
                f"Secondary causes: {', '.join(self.secondary_causes)}"
            )

        for pattern in self.patterns:
            lines.append("")
            lines.append(
                f"[{pattern.confidence:.0%}] {pattern.pattern_name}"
            )
            lines.append(f"  {pattern.description}")
            if pattern.affected_items:
                lines.append(
                    f"  Affected: {len(pattern.affected_items)} items"
                )
            if pattern.suggested_fix:
                lines.append(f"  Fix: {pattern.suggested_fix}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pattern Detectors
# ---------------------------------------------------------------------------

# Thresholds
_SHORT_OUTPUT_CHARS = 20
_LONG_OUTPUT_CHARS = 500
_COMPLEX_INPUT_SENTENCES = 3
_COMPLEX_INPUT_CHARS = 200
_CONCENTRATION_THRESHOLD = 0.6  # 60% in one bucket


def _get_output_text(failure: Dict[str, Any]) -> str:
    """Extract output text from a failure dict."""
    output = failure.get("output", "")
    if output is None:
        return ""
    if isinstance(output, dict):
        import json
        return json.dumps(output)
    return str(output)


def _get_input_text(failure: Dict[str, Any]) -> str:
    """Extract input text from a failure dict."""
    inp = failure.get("input", "")
    if inp is None:
        return ""
    if isinstance(inp, dict):
        import json
        return json.dumps(inp)
    return str(inp)


def detect_output_length_pattern(
    failures: List[Dict[str, Any]],
) -> Optional[FailurePattern]:
    """Check if failures correlate with short or long outputs.

    Each failure dict has: item_id, output, score, metric_id.
    Returns a pattern if most failures have unusually short or long outputs.
    """
    if not failures:
        return None

    short_items: List[str] = []
    long_items: List[str] = []

    for f in failures:
        text = _get_output_text(f)
        item_id = f.get("item_id", "")
        length = len(text)
        if length <= _SHORT_OUTPUT_CHARS:
            short_items.append(item_id)
        elif length >= _LONG_OUTPUT_CHARS:
            long_items.append(item_id)

    total = len(failures)
    short_ratio = len(short_items) / total
    long_ratio = len(long_items) / total

    if short_ratio >= _CONCENTRATION_THRESHOLD:
        return FailurePattern(
            pattern_name="short_output",
            description=(
                f"{len(short_items)}/{total} failures have very short outputs "
                f"(<= {_SHORT_OUTPUT_CHARS} chars)"
            ),
            affected_items=short_items,
            confidence=short_ratio,
            suggested_fix=(
                "Check if the model is producing truncated or empty responses. "
                "Consider increasing max_tokens or reviewing prompt instructions."
            ),
        )

    if long_ratio >= _CONCENTRATION_THRESHOLD:
        return FailurePattern(
            pattern_name="long_output",
            description=(
                f"{len(long_items)}/{total} failures have very long outputs "
                f"(>= {_LONG_OUTPUT_CHARS} chars)"
            ),
            affected_items=long_items,
            confidence=long_ratio,
            suggested_fix=(
                "Long outputs may indicate verbose or off-topic responses. "
                "Consider adding length constraints to the prompt."
            ),
        )

    return None


def detect_input_complexity_pattern(
    failures: List[Dict[str, Any]],
) -> Optional[FailurePattern]:
    """Check if failures correlate with complex inputs (long, multi-sentence).

    Each failure dict has: item_id, output, score, metric_id, and optionally input.
    """
    if not failures:
        return None

    complex_items: List[str] = []

    for f in failures:
        text = _get_input_text(f)
        item_id = f.get("item_id", "")
        sentence_count = text.count(".") + text.count("?") + text.count("!")
        is_complex = (
            len(text) >= _COMPLEX_INPUT_CHARS
            or sentence_count >= _COMPLEX_INPUT_SENTENCES
        )
        if is_complex:
            complex_items.append(item_id)

    total = len(failures)
    ratio = len(complex_items) / total

    if ratio >= _CONCENTRATION_THRESHOLD:
        return FailurePattern(
            pattern_name="complex_input",
            description=(
                f"{len(complex_items)}/{total} failures involve complex inputs "
                f"(>= {_COMPLEX_INPUT_CHARS} chars or >= {_COMPLEX_INPUT_SENTENCES} sentences)"
            ),
            affected_items=complex_items,
            confidence=ratio,
            suggested_fix=(
                "The model may struggle with complex multi-part queries. "
                "Consider decomposing inputs or adding chain-of-thought prompting."
            ),
        )

    return None


def detect_metric_concentration(
    failures: List[Dict[str, Any]],
) -> Optional[FailurePattern]:
    """Check if failures are concentrated in one metric.

    Each failure dict has: item_id, output, score, metric_id.
    """
    if not failures:
        return None

    metric_counts: Counter = Counter()
    metric_items: Dict[str, List[str]] = {}

    for f in failures:
        metric_id = f.get("metric_id", "unknown")
        item_id = f.get("item_id", "")
        metric_counts[metric_id] += 1
        metric_items.setdefault(metric_id, []).append(item_id)

    total = len(failures)
    if not metric_counts:
        return None

    top_metric, top_count = metric_counts.most_common(1)[0]
    ratio = top_count / total

    if ratio >= _CONCENTRATION_THRESHOLD:
        return FailurePattern(
            pattern_name="metric_concentration",
            description=(
                f"{top_count}/{total} failures are in metric '{top_metric}'"
            ),
            affected_items=metric_items[top_metric],
            confidence=ratio,
            suggested_fix=(
                f"Metric '{top_metric}' accounts for most failures. "
                "Review its rubric or calibrate it."
            ),
        )

    return None


def detect_category_pattern(
    failures: List[Dict[str, Any]],
) -> Optional[FailurePattern]:
    """Check if failures are concentrated in one category.

    Looks for a 'category' field in each failure dict.
    """
    if not failures:
        return None

    category_counts: Counter = Counter()
    category_items: Dict[str, List[str]] = {}

    for f in failures:
        category = f.get("category", "")
        if not category:
            continue
        item_id = f.get("item_id", "")
        category_counts[category] += 1
        category_items.setdefault(category, []).append(item_id)

    if not category_counts:
        return None

    total_with_category = sum(category_counts.values())
    top_category, top_count = category_counts.most_common(1)[0]
    ratio = top_count / total_with_category

    if ratio >= _CONCENTRATION_THRESHOLD:
        return FailurePattern(
            pattern_name="category_concentration",
            description=(
                f"{top_count}/{total_with_category} failures are in "
                f"category '{top_category}'"
            ),
            affected_items=category_items[top_category],
            confidence=ratio,
            suggested_fix=(
                f"Category '{top_category}' has disproportionate failures. "
                "Investigate whether training data covers this category adequately."
            ),
        )

    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def analyze_root_causes(
    failures: List[Dict[str, Any]],
) -> RootCauseReport:
    """Run all pattern detectors and build a root cause report.

    Primary cause is the highest-confidence pattern found.
    """
    if not failures:
        return RootCauseReport(total_failures=0)

    detectors = [
        detect_output_length_pattern,
        detect_input_complexity_pattern,
        detect_metric_concentration,
        detect_category_pattern,
    ]

    patterns: List[FailurePattern] = []
    for detector in detectors:
        result = detector(failures)
        if result is not None:
            patterns.append(result)

    # Sort by confidence descending
    patterns.sort(key=lambda p: p.confidence, reverse=True)

    primary_cause = ""
    secondary_causes: List[str] = []

    if patterns:
        primary_cause = patterns[0].pattern_name
        secondary_causes = [p.pattern_name for p in patterns[1:]]

    return RootCauseReport(
        patterns=patterns,
        total_failures=len(failures),
        primary_cause=primary_cause,
        secondary_causes=secondary_causes,
    )


# ---------------------------------------------------------------------------
# Fix Suggestions
# ---------------------------------------------------------------------------


def suggest_fixes(report: RootCauseReport) -> List[str]:
    """Generate actionable fix suggestions from a root cause report."""
    fixes: List[str] = []

    for pattern in report.patterns:
        if pattern.suggested_fix:
            fixes.append(pattern.suggested_fix)

    if not fixes and report.total_failures > 0:
        fixes.append(
            "No clear pattern detected. "
            "Review individual failures for specific issues."
        )

    return fixes
