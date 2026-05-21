"""
Inject specific failure patterns into simulation prompts.

Pure Python, no external dependencies. Generates test items targeting
known failure modes via rephrasing, combination, and perturbation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Severity ordering (higher index = more severe)
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class FailureTarget:
    """A known failure pattern to inject into test items."""

    pattern_id: str
    description: str
    example_inputs: list[str]
    severity: str = "medium"
    frequency: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "example_inputs": list(self.example_inputs),
            "severity": self.severity,
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureTarget:
        return cls(
            pattern_id=data["pattern_id"],
            description=data["description"],
            example_inputs=list(data["example_inputs"]),
            severity=data.get("severity", "medium"),
            frequency=data.get("frequency", 1),
        )


@dataclass
class InjectionConfig:
    """Configuration for a failure-injection generation run."""

    targets: list[FailureTarget]
    items_per_target: int = 3
    include_examples: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "targets": [t.as_dict() for t in self.targets],
            "items_per_target": self.items_per_target,
            "include_examples": self.include_examples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InjectionConfig:
        return cls(
            targets=[FailureTarget.from_dict(t) for t in data["targets"]],
            items_per_target=data.get("items_per_target", 3),
            include_examples=data.get("include_examples", True),
        )


@dataclass
class InjectedItem:
    """A single test item generated from a failure target."""

    input_text: str
    target_pattern_id: str
    injection_method: str
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "target_pattern_id": self.target_pattern_id,
            "injection_method": self.injection_method,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InjectedItem:
        return cls(
            input_text=data["input_text"],
            target_pattern_id=data["target_pattern_id"],
            injection_method=data["injection_method"],
            tags=list(data.get("tags", [])),
        )


@dataclass
class InjectionReport:
    """Summary of all injected items for a generation run."""

    items: list[InjectedItem]
    total_items: int
    patterns_covered: int
    total_patterns: int
    coverage_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": [i.as_dict() for i in self.items],
            "total_items": self.total_items,
            "patterns_covered": self.patterns_covered,
            "total_patterns": self.total_patterns,
            "coverage_rate": self.coverage_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InjectionReport:
        return cls(
            items=[InjectedItem.from_dict(i) for i in data["items"]],
            total_items=data["total_items"],
            patterns_covered=data["patterns_covered"],
            total_patterns=data["total_patterns"],
            coverage_rate=data["coverage_rate"],
        )


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _rephrase(text: str, rng: random.Random) -> str:
    """Rephrase by shuffling the words in the text."""
    words = text.split()
    if len(words) <= 1:
        return text
    shuffled = list(words)
    rng.shuffle(shuffled)
    return " ".join(shuffled)


def _combine(a: str, b: str) -> str:
    """Combine two example inputs into one."""
    return f"{a} {b}"


def _perturb(text: str, rng: random.Random) -> str:
    """Apply a random perturbation: case change or punctuation tweak."""
    methods = ["upper", "lower", "strip_punct", "add_punct"]
    method = rng.choice(methods)
    if method == "upper":
        return text.upper()
    if method == "lower":
        return text.lower()
    if method == "strip_punct":
        return text.rstrip(".,!?;:")
    # add_punct
    punct = rng.choice(["!", "?", "...", "!!"])
    return text + punct


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def generate_from_failure_target(
    target: FailureTarget,
    n_items: int = 3,
    rng: random.Random | None = None,
) -> list[InjectedItem]:
    """Generate injected items targeting a specific failure pattern.

    Methods: rephrase (shuffle words), combine (merge two examples),
    perturbation (case/punctuation change).
    """
    if rng is None:
        rng = random.Random()

    items: list[InjectedItem] = []
    examples = target.example_inputs
    tags = [target.pattern_id, target.severity]

    for i in range(n_items):
        method_idx = i % 3
        if method_idx == 0 and examples:
            # Rephrase
            base = rng.choice(examples)
            text = _rephrase(base, rng)
            method = "rephrase"
        elif method_idx == 1 and len(examples) >= 2:
            # Combine two examples
            pair = rng.sample(examples, 2)
            text = _combine(pair[0], pair[1])
            method = "combine"
        elif examples:
            # Perturbation
            base = rng.choice(examples)
            text = _perturb(base, rng)
            method = "perturbation"
        else:
            text = target.description
            method = "description_fallback"

        items.append(
            InjectedItem(
                input_text=text,
                target_pattern_id=target.pattern_id,
                injection_method=method,
                tags=list(tags),
            )
        )

    return items


def generate_injection_suite(
    config: InjectionConfig,
    rng: random.Random | None = None,
) -> InjectionReport:
    """Generate injected items for all targets in the config."""
    if rng is None:
        rng = random.Random()

    all_items: list[InjectedItem] = []
    patterns_covered = 0

    for target in config.targets:
        generated = generate_from_failure_target(target, n_items=config.items_per_target, rng=rng)
        if generated:
            patterns_covered += 1
        all_items.extend(generated)

    total_patterns = len(config.targets)
    coverage = patterns_covered / total_patterns if total_patterns > 0 else 0.0

    return InjectionReport(
        items=all_items,
        total_items=len(all_items),
        patterns_covered=patterns_covered,
        total_patterns=total_patterns,
        coverage_rate=coverage,
    )


# ---------------------------------------------------------------------------
# Coverage and prioritization
# ---------------------------------------------------------------------------


def compute_coverage(report: InjectionReport, all_targets: list[FailureTarget]) -> float:
    """Fraction of all_targets that have at least one generated item."""
    if not all_targets:
        return 0.0
    covered_ids = {item.target_pattern_id for item in report.items}
    target_ids = {t.pattern_id for t in all_targets}
    matched = covered_ids & target_ids
    return len(matched) / len(target_ids)


def prioritize_targets(targets: list[FailureTarget]) -> list[FailureTarget]:
    """Sort targets by severity (critical first) then frequency (highest first)."""
    return sorted(
        targets,
        key=lambda t: (-_SEVERITY_ORDER.get(t.severity, -1), -t.frequency),
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_injection_report(report: InjectionReport) -> str:
    """Human-readable summary of an injection report."""
    lines = [
        f"Total items: {report.total_items}",
        f"Patterns covered: {report.patterns_covered}/{report.total_patterns}",
        f"Coverage rate: {report.coverage_rate:.1%}",
        "",
    ]
    by_pattern: dict[str, list[InjectedItem]] = {}
    for item in report.items:
        by_pattern.setdefault(item.target_pattern_id, []).append(item)

    for pattern_id, items in sorted(by_pattern.items()):
        lines.append(f"[{pattern_id}] ({len(items)} items)")
        for item in items:
            lines.append(f"  - ({item.injection_method}) {item.input_text}")

    return "\n".join(lines)
