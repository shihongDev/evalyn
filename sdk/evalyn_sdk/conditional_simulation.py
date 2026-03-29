"""
Conditional simulation: generate inputs that test edge conditions.

Produces inputs targeting specific edge cases such as empty strings,
Unicode extremes, injection patterns, and whitespace-only content.
Pure Python - no external deps, no LLM calls.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class EdgeCondition:
    """Definition of a single edge condition to test."""

    condition_id: str
    name: str
    description: str
    generator: str  # function name reference
    severity: str = "medium"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "name": self.name,
            "description": self.description,
            "generator": self.generator,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EdgeCondition:
        return cls(
            condition_id=data["condition_id"],
            name=data["name"],
            description=data["description"],
            generator=data["generator"],
            severity=data.get("severity", "medium"),
        )


@dataclass
class ConditionalInput:
    """A generated input with metadata about which conditions were applied."""

    input_text: str
    conditions_applied: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "input_text": self.input_text,
            "conditions_applied": list(self.conditions_applied),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConditionalInput:
        return cls(
            input_text=data["input_text"],
            conditions_applied=data.get("conditions_applied", []),
            tags=data.get("tags", []),
        )


@dataclass
class ConditionalConfig:
    """Configuration for conditional input generation."""

    conditions: List[str] = field(default_factory=list)  # condition IDs to apply
    base_text: str = ""
    combinatorial: bool = False  # if True, generate all pair combinations

    def as_dict(self) -> Dict[str, Any]:
        return {
            "conditions": list(self.conditions),
            "base_text": self.base_text,
            "combinatorial": self.combinatorial,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConditionalConfig:
        return cls(
            conditions=data.get("conditions", []),
            base_text=data.get("base_text", ""),
            combinatorial=data.get("combinatorial", False),
        )


# ---------------------------------------------------------------------------
# Generator Functions
# ---------------------------------------------------------------------------


def generate_empty() -> str:
    """Return an empty string."""
    return ""


def generate_null_like(rng: Optional[random.Random] = None) -> str:
    """Return a None-like string value."""
    choices = ["null", "None", "undefined"]
    if rng is not None:
        return rng.choice(choices)
    return random.choice(choices)


def generate_max_length() -> str:
    """Return a very long string (10000 characters)."""
    return "a" * 10000


def generate_unicode_cjk() -> str:
    """Return a string with CJK characters."""
    return "\u4f60\u597d\u4e16\u754c\u3053\u3093\u306b\u3061\u306f\uc548\ub155\ud558\uc138\uc694"


def generate_unicode_emoji() -> str:
    """Return emoji-like text using bracket notation."""
    return "[emoji] [heart] [fire] [star] [thumbs_up]"


def generate_mixed_language() -> str:
    """Return text mixing English with other scripts."""
    return "Hello \u4e16\u754c and \u043f\u0440\u0438\u0432\u0435\u0442 plus \u0645\u0631\u062d\u0628\u0627"


def generate_special_chars() -> str:
    """Return a string of symbols and punctuation."""
    return "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"


def generate_whitespace_only() -> str:
    """Return a string of only whitespace characters."""
    return "\t  \n  \t\n   "


def generate_sql_injection() -> str:
    """Return a SQL injection pattern."""
    return "'; DROP TABLE users; --"


def generate_html_tags() -> str:
    """Return HTML/script tags."""
    return '<script>alert("xss")</script><b>bold</b>'


# ---------------------------------------------------------------------------
# Condition Registry
# ---------------------------------------------------------------------------

_GENERATORS = {
    "generate_empty": generate_empty,
    "generate_null_like": generate_null_like,
    "generate_max_length": generate_max_length,
    "generate_unicode_cjk": generate_unicode_cjk,
    "generate_unicode_emoji": generate_unicode_emoji,
    "generate_mixed_language": generate_mixed_language,
    "generate_special_chars": generate_special_chars,
    "generate_whitespace_only": generate_whitespace_only,
    "generate_sql_injection": generate_sql_injection,
    "generate_html_tags": generate_html_tags,
}

EDGE_CONDITIONS: Dict[str, EdgeCondition] = {
    "empty": EdgeCondition(
        condition_id="empty",
        name="Empty String",
        description="Empty string input",
        generator="generate_empty",
        severity="high",
    ),
    "null_like": EdgeCondition(
        condition_id="null_like",
        name="Null-like Values",
        description="Strings that represent null/None/undefined",
        generator="generate_null_like",
        severity="high",
    ),
    "max_length": EdgeCondition(
        condition_id="max_length",
        name="Maximum Length",
        description="Very long string (10000 chars)",
        generator="generate_max_length",
        severity="medium",
    ),
    "unicode_cjk": EdgeCondition(
        condition_id="unicode_cjk",
        name="Unicode CJK",
        description="CJK (Chinese, Japanese, Korean) characters",
        generator="generate_unicode_cjk",
        severity="medium",
    ),
    "unicode_emoji": EdgeCondition(
        condition_id="unicode_emoji",
        name="Unicode Emoji",
        description="Emoji-like text",
        generator="generate_unicode_emoji",
        severity="low",
    ),
    "mixed_language": EdgeCondition(
        condition_id="mixed_language",
        name="Mixed Language",
        description="Text mixing English with other scripts",
        generator="generate_mixed_language",
        severity="medium",
    ),
    "special_chars": EdgeCondition(
        condition_id="special_chars",
        name="Special Characters",
        description="Symbols and punctuation",
        generator="generate_special_chars",
        severity="medium",
    ),
    "whitespace_only": EdgeCondition(
        condition_id="whitespace_only",
        name="Whitespace Only",
        description="Only tabs, spaces, and newlines",
        generator="generate_whitespace_only",
        severity="high",
    ),
    "sql_injection": EdgeCondition(
        condition_id="sql_injection",
        name="SQL Injection",
        description="SQL injection patterns",
        generator="generate_sql_injection",
        severity="critical",
    ),
    "html_tags": EdgeCondition(
        condition_id="html_tags",
        name="HTML Tags",
        description="HTML and script tags",
        generator="generate_html_tags",
        severity="critical",
    ),
}


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def _call_generator(
    generator_name: str, rng: Optional[random.Random] = None
) -> str:
    """Call a generator function by name, passing rng if accepted."""
    fn = _GENERATORS[generator_name]
    # Only generate_null_like uses rng
    if generator_name == "generate_null_like":
        return fn(rng)
    return fn()


def generate_conditional_inputs(
    config: ConditionalConfig,
    rng: Optional[random.Random] = None,
) -> List[ConditionalInput]:
    """Generate inputs for specified conditions.

    If combinatorial is True, also generate inputs combining pairs of
    conditions (concatenated with a space separator).
    """
    results: List[ConditionalInput] = []

    # Filter to valid condition IDs
    valid_ids = [c for c in config.conditions if c in EDGE_CONDITIONS]

    # Generate one input per condition
    for cid in valid_ids:
        cond = EDGE_CONDITIONS[cid]
        text = _call_generator(cond.generator, rng)
        if config.base_text:
            text = config.base_text + " " + text
        results.append(
            ConditionalInput(
                input_text=text,
                conditions_applied=[cid],
                tags=[cond.severity],
            )
        )

    # Combinatorial: generate all pair combinations
    if config.combinatorial and len(valid_ids) >= 2:
        for a, b in itertools.combinations(valid_ids, 2):
            cond_a = EDGE_CONDITIONS[a]
            cond_b = EDGE_CONDITIONS[b]
            text_a = _call_generator(cond_a.generator, rng)
            text_b = _call_generator(cond_b.generator, rng)
            combined = text_a + " " + text_b
            if config.base_text:
                combined = config.base_text + " " + combined
            # Use the higher severity
            severity_order = ["low", "medium", "high", "critical"]
            sev_a = severity_order.index(cond_a.severity) if cond_a.severity in severity_order else 1
            sev_b = severity_order.index(cond_b.severity) if cond_b.severity in severity_order else 1
            max_sev = severity_order[max(sev_a, sev_b)]
            results.append(
                ConditionalInput(
                    input_text=combined,
                    conditions_applied=[a, b],
                    tags=[max_sev, "combinatorial"],
                )
            )

    return results


def generate_all_conditions(base_text: str = "") -> List[ConditionalInput]:
    """Generate one input per known edge condition."""
    config = ConditionalConfig(
        conditions=list(EDGE_CONDITIONS.keys()),
        base_text=base_text,
        combinatorial=False,
    )
    return generate_conditional_inputs(config)


def format_conditional_report(inputs: List[ConditionalInput]) -> str:
    """Format a summary report of generated conditional inputs."""
    lines: List[str] = []
    lines.append("Conditional Simulation Report")
    lines.append("=" * 40)
    lines.append(f"Total inputs generated: {len(inputs)}")
    lines.append("")

    # Count by condition
    condition_counts: Dict[str, int] = {}
    for inp in inputs:
        for cid in inp.conditions_applied:
            condition_counts[cid] = condition_counts.get(cid, 0) + 1

    if condition_counts:
        lines.append("Conditions covered:")
        for cid, count in sorted(condition_counts.items()):
            name = EDGE_CONDITIONS[cid].name if cid in EDGE_CONDITIONS else cid
            lines.append(f"  {name}: {count} input(s)")
        lines.append("")

    # Count combinatorial
    combo_count = sum(
        1 for inp in inputs if "combinatorial" in inp.tags
    )
    if combo_count:
        lines.append(f"Combinatorial inputs: {combo_count}")
        lines.append("")

    # Severity breakdown
    severity_counts: Dict[str, int] = {}
    for inp in inputs:
        for tag in inp.tags:
            if tag in ("low", "medium", "high", "critical"):
                severity_counts[tag] = severity_counts.get(tag, 0) + 1
    if severity_counts:
        lines.append("Severity breakdown:")
        for sev in ("critical", "high", "medium", "low"):
            if sev in severity_counts:
                lines.append(f"  {sev}: {severity_counts[sev]}")

    return "\n".join(lines)
