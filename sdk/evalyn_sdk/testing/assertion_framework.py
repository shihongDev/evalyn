from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AssertionResult:
    passed: bool
    assertion_type: str
    message: str = ""
    actual: Any = None
    expected: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "assertion_type": self.assertion_type,
            "message": self.message,
            "actual": self.actual,
            "expected": self.expected,
        }


@dataclass
class AssertionConfig:
    assertions: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "assertions": self.assertions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssertionConfig:
        return cls(
            assertions=data.get("assertions", []),
        )


def assert_contains(output: str, substring: str) -> AssertionResult:
    """Output contains substring."""
    passed = substring in output
    return AssertionResult(
        passed=passed,
        assertion_type="contains",
        message="" if passed else f"Expected output to contain '{substring}'",
        actual=output,
        expected=substring,
    )


def assert_not_contains(output: str, substring: str) -> AssertionResult:
    """Output does not contain substring."""
    passed = substring not in output
    return AssertionResult(
        passed=passed,
        assertion_type="not_contains",
        message="" if passed else f"Expected output to not contain '{substring}'",
        actual=output,
        expected=substring,
    )


def assert_matches_regex(output: str, pattern: str) -> AssertionResult:
    """Output matches regex."""
    try:
        passed = bool(re.search(pattern, output))
    except re.error as exc:
        return AssertionResult(
            passed=False,
            assertion_type="matches_regex",
            message=f"Invalid regex pattern: {exc}",
            actual=output,
            expected=pattern,
        )
    return AssertionResult(
        passed=passed,
        assertion_type="matches_regex",
        message="" if passed else f"Expected output to match pattern '{pattern}'",
        actual=output,
        expected=pattern,
    )


def assert_json_valid(output: str) -> AssertionResult:
    """Output is valid JSON."""
    try:
        json.loads(output)
        passed = True
        message = ""
    except (json.JSONDecodeError, ValueError) as exc:
        passed = False
        message = f"Invalid JSON: {exc}"
    return AssertionResult(
        passed=passed,
        assertion_type="json_valid",
        message=message,
        actual=output,
    )


def assert_min_length(output: str, min_len: int) -> AssertionResult:
    """Output length >= min."""
    passed = len(output) >= min_len
    return AssertionResult(
        passed=passed,
        assertion_type="min_length",
        message="" if passed else f"Expected length >= {min_len}, got {len(output)}",
        actual=len(output),
        expected=min_len,
    )


def assert_max_length(output: str, max_len: int) -> AssertionResult:
    """Output length <= max."""
    passed = len(output) <= max_len
    return AssertionResult(
        passed=passed,
        assertion_type="max_length",
        message="" if passed else f"Expected length <= {max_len}, got {len(output)}",
        actual=len(output),
        expected=max_len,
    )


def assert_equals(output: str, expected: str) -> AssertionResult:
    """Exact match."""
    passed = output == expected
    return AssertionResult(
        passed=passed,
        assertion_type="equals",
        message="" if passed else f"Expected '{expected}', got '{output}'",
        actual=output,
        expected=expected,
    )


def assert_score_above(score: float, threshold: float) -> AssertionResult:
    """Score > threshold."""
    passed = score > threshold
    return AssertionResult(
        passed=passed,
        assertion_type="score_above",
        message="" if passed else f"Expected score > {threshold}, got {score}",
        actual=score,
        expected=threshold,
    )


def assert_score_below(score: float, threshold: float) -> AssertionResult:
    """Score < threshold."""
    passed = score < threshold
    return AssertionResult(
        passed=passed,
        assertion_type="score_below",
        message="" if passed else f"Expected score < {threshold}, got {score}",
        actual=score,
        expected=threshold,
    )


def run_assertions(
    output: str, config: AssertionConfig, score: float = 0.0
) -> list[AssertionResult]:
    """Run all configured assertions.

    Config assertions are dicts like {"type": "contains", "value": "hello"}.
    """
    results: list[AssertionResult] = []

    dispatch = {
        "contains": lambda a: assert_contains(output, a["value"]),
        "not_contains": lambda a: assert_not_contains(output, a["value"]),
        "matches_regex": lambda a: assert_matches_regex(output, a["value"]),
        "json_valid": lambda a: assert_json_valid(output),
        "min_length": lambda a: assert_min_length(output, a["value"]),
        "max_length": lambda a: assert_max_length(output, a["value"]),
        "equals": lambda a: assert_equals(output, a["value"]),
        "score_above": lambda a: assert_score_above(score, a["value"]),
        "score_below": lambda a: assert_score_below(score, a["value"]),
    }

    for assertion in config.assertions:
        atype = assertion.get("type", "")
        handler = dispatch.get(atype)
        if handler:
            results.append(handler(assertion))
        else:
            results.append(
                AssertionResult(
                    passed=False,
                    assertion_type=atype,
                    message=f"Unknown assertion type: {atype}",
                )
            )

    return results


def all_passed(results: list[AssertionResult]) -> bool:
    """True if all passed."""
    return all(r.passed for r in results)
