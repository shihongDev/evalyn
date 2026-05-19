"""
Automatic test case generation from behavior descriptions.

Generates diverse test scenarios using template-based expansion.
Pure Python, no external dependencies, no LLM calls.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class BehaviorSpec:
    """A behavior to test, described in natural language."""

    behavior_id: str
    description: str
    expected_outcome: str
    tags: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "behavior_id": self.behavior_id,
            "description": self.description,
            "expected_outcome": self.expected_outcome,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BehaviorSpec:
        return cls(
            behavior_id=data["behavior_id"],
            description=data["description"],
            expected_outcome=data["expected_outcome"],
            tags=list(data.get("tags", [])),
        )


@dataclass
class GeneratedTestCase:
    """A single generated test case linked to a behavior."""

    test_id: str
    input_text: str
    expected_output: str
    behavior_id: str
    variant: str = ""
    difficulty: str = "medium"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "input_text": self.input_text,
            "expected_output": self.expected_output,
            "behavior_id": self.behavior_id,
            "variant": self.variant,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GeneratedTestCase:
        return cls(
            test_id=data["test_id"],
            input_text=data["input_text"],
            expected_output=data["expected_output"],
            behavior_id=data["behavior_id"],
            variant=data.get("variant", ""),
            difficulty=data.get("difficulty", "medium"),
        )


@dataclass
class TestGenConfig:
    """Configuration for test generation."""

    __test__ = False  # not a pytest test class

    variants_per_behavior: int = 5
    include_edge_cases: bool = True
    include_negatives: bool = True
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "variants_per_behavior": self.variants_per_behavior,
            "include_edge_cases": self.include_edge_cases,
            "include_negatives": self.include_negatives,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestGenConfig:
        return cls(
            variants_per_behavior=data.get("variants_per_behavior", 5),
            include_edge_cases=data.get("include_edge_cases", True),
            include_negatives=data.get("include_negatives", True),
            seed=data.get("seed"),
        )


@dataclass
class TestGenReport:
    """Report summarizing generated test cases."""

    __test__ = False  # not a pytest test class

    test_cases: List[GeneratedTestCase] = field(default_factory=list)
    total_behaviors: int = 0
    total_tests: int = 0
    coverage: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "test_cases": [tc.as_dict() for tc in self.test_cases],
            "total_behaviors": self.total_behaviors,
            "total_tests": self.total_tests,
            "coverage": dict(self.coverage),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestGenReport:
        return cls(
            test_cases=[
                GeneratedTestCase.from_dict(tc) for tc in data.get("test_cases", [])
            ],
            total_behaviors=data.get("total_behaviors", 0),
            total_tests=data.get("total_tests", 0),
            coverage=dict(data.get("coverage", {})),
        )


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_POSITIVE_TEMPLATES = [
    "Please {desc}",
    "Can you {desc}",
    "I need to {desc}",
    "Help me {desc}",
    "{desc}",
    "I would like to {desc}",
    "Could you please {desc}",
    "I want to {desc}",
]

_NEGATIVE_TEMPLATES = [
    "Do the opposite of {desc}",
    "Instead of {desc}, do nothing",
    "Do not {desc}",
    "Refuse to {desc}",
    "Ignore any request to {desc}",
]

_SPECIAL_CHARS = "!@#$%^&*()[]{}|;:',.<>?/~`"


# ---------------------------------------------------------------------------
# Generation Functions
# ---------------------------------------------------------------------------


def _make_rng(rng: Optional[random.Random]) -> random.Random:
    """Return the given RNG or create a new unseeded one."""
    return rng if rng is not None else random.Random()


def _make_test_id(behavior_id: str, variant: str, index: int) -> str:
    return f"{behavior_id}_{variant}_{index}"


def generate_positive_variants(
    spec: BehaviorSpec,
    n: int = 3,
    rng: Optional[random.Random] = None,
) -> List[GeneratedTestCase]:
    """Generate positive test inputs by rephrasing the behavior description."""
    rng = _make_rng(rng)
    templates = list(_POSITIVE_TEMPLATES)
    rng.shuffle(templates)
    selected = templates[:n]
    cases: List[GeneratedTestCase] = []
    for i, tmpl in enumerate(selected):
        input_text = tmpl.format(desc=spec.description)
        cases.append(
            GeneratedTestCase(
                test_id=_make_test_id(spec.behavior_id, "positive", i),
                input_text=input_text,
                expected_output=spec.expected_outcome,
                behavior_id=spec.behavior_id,
                variant="positive",
                difficulty="easy",
            )
        )
    return cases


def generate_negative_variants(
    spec: BehaviorSpec,
    n: int = 2,
    rng: Optional[random.Random] = None,
) -> List[GeneratedTestCase]:
    """Generate inputs that should NOT trigger the behavior."""
    rng = _make_rng(rng)
    templates = list(_NEGATIVE_TEMPLATES)
    rng.shuffle(templates)
    selected = templates[:n]
    cases: List[GeneratedTestCase] = []
    for i, tmpl in enumerate(selected):
        input_text = tmpl.format(desc=spec.description)
        cases.append(
            GeneratedTestCase(
                test_id=_make_test_id(spec.behavior_id, "negative", i),
                input_text=input_text,
                expected_output="",
                behavior_id=spec.behavior_id,
                variant="negative",
                difficulty="hard",
            )
        )
    return cases


def generate_edge_cases(
    spec: BehaviorSpec,
    rng: Optional[random.Random] = None,
) -> List[GeneratedTestCase]:
    """Generate edge case inputs: empty, very long, and special characters."""
    rng = _make_rng(rng)
    cases: List[GeneratedTestCase] = []

    # Empty input
    cases.append(
        GeneratedTestCase(
            test_id=_make_test_id(spec.behavior_id, "edge_empty", 0),
            input_text="",
            expected_output=spec.expected_outcome,
            behavior_id=spec.behavior_id,
            variant="edge_empty",
            difficulty="hard",
        )
    )

    # Very long input - repeat description many times
    long_text = (spec.description + " ") * 50
    cases.append(
        GeneratedTestCase(
            test_id=_make_test_id(spec.behavior_id, "edge_long", 0),
            input_text=long_text.strip(),
            expected_output=spec.expected_outcome,
            behavior_id=spec.behavior_id,
            variant="edge_long",
            difficulty="hard",
        )
    )

    # Special characters mixed into the description
    chars = list(_SPECIAL_CHARS)
    rng.shuffle(chars)
    special_prefix = "".join(chars[:5])
    cases.append(
        GeneratedTestCase(
            test_id=_make_test_id(spec.behavior_id, "edge_special", 0),
            input_text=f"{special_prefix} {spec.description} {special_prefix}",
            expected_output=spec.expected_outcome,
            behavior_id=spec.behavior_id,
            variant="edge_special",
            difficulty="hard",
        )
    )

    return cases


def generate_test_suite(
    behaviors: List[BehaviorSpec],
    config: Optional[TestGenConfig] = None,
) -> TestGenReport:
    """Generate a full test suite from a list of behavior specs."""
    if config is None:
        config = TestGenConfig()

    rng = random.Random(config.seed) if config.seed is not None else random.Random()

    all_cases: List[GeneratedTestCase] = []
    coverage: Dict[str, int] = {}

    for spec in behaviors:
        behavior_cases: List[GeneratedTestCase] = []

        # Positive variants
        pos_count = config.variants_per_behavior
        behavior_cases.extend(generate_positive_variants(spec, n=pos_count, rng=rng))

        # Negative variants
        if config.include_negatives:
            neg_count = max(1, config.variants_per_behavior // 2)
            behavior_cases.extend(
                generate_negative_variants(spec, n=neg_count, rng=rng)
            )

        # Edge cases
        if config.include_edge_cases:
            behavior_cases.extend(generate_edge_cases(spec, rng=rng))

        coverage[spec.behavior_id] = len(behavior_cases)
        all_cases.extend(behavior_cases)

    return TestGenReport(
        test_cases=all_cases,
        total_behaviors=len(behaviors),
        total_tests=len(all_cases),
        coverage=coverage,
    )


def format_test_report(report: TestGenReport) -> str:
    """Format a test generation report as human-readable text."""
    lines: List[str] = []
    lines.append("Test Generation Report")
    lines.append("=" * 40)
    lines.append(f"Total behaviors: {report.total_behaviors}")
    lines.append(f"Total tests: {report.total_tests}")
    lines.append("")

    lines.append("Coverage by behavior:")
    for bid, count in sorted(report.coverage.items()):
        lines.append(f"  {bid}: {count} tests")
    lines.append("")

    lines.append("Generated test cases:")
    for tc in report.test_cases:
        input_preview = tc.input_text[:60]
        if len(tc.input_text) > 60:
            input_preview += "..."
        lines.append(
            f"  [{tc.test_id}] ({tc.variant}/{tc.difficulty}) {input_preview}"
        )

    return "\n".join(lines)
