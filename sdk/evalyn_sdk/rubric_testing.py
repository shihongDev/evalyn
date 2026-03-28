"""Rubric testing and validation framework.

Pure Python, no external dependencies. Runs rubric test cases multiple
times to compute consistency, detect edge cases, and compare versions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


def _mean(vals: List[float]) -> float:
    """Arithmetic mean. Returns 0.0 for empty lists."""
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _std_dev(vals: List[float]) -> float:
    """Population standard deviation. Returns 0.0 for fewer than 2 values."""
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    variance = sum((v - m) ** 2 for v in vals) / len(vals)
    return math.sqrt(variance)


@dataclass
class RubricTestCase:
    """A single test case for rubric validation."""

    item_id: str
    input_text: str
    expected_score: float
    expected_pass: bool
    tags: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "input_text": self.input_text,
            "expected_score": self.expected_score,
            "expected_pass": self.expected_pass,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RubricTestCase:
        return cls(
            item_id=data["item_id"],
            input_text=data["input_text"],
            expected_score=data["expected_score"],
            expected_pass=data["expected_pass"],
            tags=data.get("tags", []),
        )


@dataclass
class RubricTestResult:
    """Result of running a single test case multiple times."""

    item_id: str
    scores: List[float]
    mean_score: float
    std_dev: float
    passed_count: int
    total_runs: int
    consistency: float
    is_edge_case: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "scores": list(self.scores),
            "mean_score": self.mean_score,
            "std_dev": self.std_dev,
            "passed_count": self.passed_count,
            "total_runs": self.total_runs,
            "consistency": self.consistency,
            "is_edge_case": self.is_edge_case,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RubricTestResult:
        return cls(
            item_id=data["item_id"],
            scores=data["scores"],
            mean_score=data["mean_score"],
            std_dev=data["std_dev"],
            passed_count=data["passed_count"],
            total_runs=data["total_runs"],
            consistency=data["consistency"],
            is_edge_case=data["is_edge_case"],
        )


@dataclass
class RubricTestSuite:
    """Aggregated results for a full rubric test suite."""

    rubric_id: str
    results: List[RubricTestResult]
    overall_consistency: float
    edge_case_count: int
    total_items: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rubric_id": self.rubric_id,
            "results": [r.as_dict() for r in self.results],
            "overall_consistency": self.overall_consistency,
            "edge_case_count": self.edge_case_count,
            "total_items": self.total_items,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RubricTestSuite:
        return cls(
            rubric_id=data["rubric_id"],
            results=[RubricTestResult.from_dict(r) for r in data["results"]],
            overall_consistency=data["overall_consistency"],
            edge_case_count=data["edge_case_count"],
            total_items=data["total_items"],
        )


def compute_consistency(scores: List[float], threshold: float = 0.5) -> float:
    """Fraction of score pairs within threshold of each other.

    For n scores, checks all n*(n-1)/2 unique pairs. Returns 1.0 if
    fewer than 2 scores are provided (trivially consistent).
    """
    n = len(scores)
    if n < 2:
        return 1.0
    total_pairs = 0
    within = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if abs(scores[i] - scores[j]) <= threshold:
                within += 1
    return within / total_pairs


def detect_edge_cases(
    results: List[RubricTestResult], std_threshold: float = 0.3
) -> List[RubricTestResult]:
    """Return results with standard deviation above the threshold."""
    return [r for r in results if r.std_dev > std_threshold]


def run_rubric_test_suite(
    rubric_id: str,
    test_cases: List[RubricTestCase],
    score_fn: Callable[[str, str], float],
    n_runs: int = 5,
    pass_threshold: float = 0.5,
    consistency_threshold: float = 0.5,
    edge_std_threshold: float = 0.3,
) -> RubricTestSuite:
    """Run each test case n_runs times and compute statistics.

    Args:
        rubric_id: Identifier for the rubric being tested.
        test_cases: List of test cases to evaluate.
        score_fn: Callable(rubric_id, input_text) -> float score.
        n_runs: Number of times to run each test case.
        pass_threshold: Score at or above this value counts as a pass.
        consistency_threshold: Threshold for compute_consistency.
        edge_std_threshold: Std dev above this marks an edge case.

    Returns:
        A RubricTestSuite with per-item results and overall statistics.
    """
    results: List[RubricTestResult] = []

    for tc in test_cases:
        scores: List[float] = []
        passed_count = 0
        for _ in range(n_runs):
            score = score_fn(rubric_id, tc.input_text)
            scores.append(score)
            if score >= pass_threshold:
                passed_count += 1

        mean = _mean(scores)
        sd = _std_dev(scores)
        consistency = compute_consistency(scores, threshold=consistency_threshold)
        is_edge = sd > edge_std_threshold

        results.append(
            RubricTestResult(
                item_id=tc.item_id,
                scores=scores,
                mean_score=mean,
                std_dev=sd,
                passed_count=passed_count,
                total_runs=n_runs,
                consistency=consistency,
                is_edge_case=is_edge,
            )
        )

    consistencies = [r.consistency for r in results]
    overall = _mean(consistencies)
    edge_count = len(detect_edge_cases(results, std_threshold=edge_std_threshold))

    return RubricTestSuite(
        rubric_id=rubric_id,
        results=results,
        overall_consistency=overall,
        edge_case_count=edge_count,
        total_items=len(results),
    )


def compare_rubric_versions(
    suite_a: RubricTestSuite, suite_b: RubricTestSuite
) -> Dict[str, Any]:
    """Compare two rubric test suites showing consistency changes and score drift.

    Returns a dict with:
        - rubric_a / rubric_b: rubric IDs
        - consistency_change: suite_b.overall_consistency - suite_a.overall_consistency
        - items: per-item comparison (score_drift, consistency_change)
        - summary: human-readable summary string
    """
    a_by_id = {r.item_id: r for r in suite_a.results}
    b_by_id = {r.item_id: r for r in suite_b.results}

    all_ids = sorted(set(a_by_id) | set(b_by_id))
    items: List[Dict[str, Any]] = []

    for item_id in all_ids:
        entry: Dict[str, Any] = {"item_id": item_id}
        a_result = a_by_id.get(item_id)
        b_result = b_by_id.get(item_id)

        if a_result and b_result:
            entry["score_drift"] = b_result.mean_score - a_result.mean_score
            entry["consistency_change"] = b_result.consistency - a_result.consistency
            entry["a_mean"] = a_result.mean_score
            entry["b_mean"] = b_result.mean_score
        elif a_result:
            entry["status"] = "removed_in_b"
            entry["a_mean"] = a_result.mean_score
        else:
            entry["status"] = "new_in_b"
            entry["b_mean"] = b_result.mean_score if b_result else 0.0

        items.append(entry)

    consistency_change = suite_b.overall_consistency - suite_a.overall_consistency
    direction = "improved" if consistency_change > 0 else "degraded" if consistency_change < 0 else "unchanged"

    return {
        "rubric_a": suite_a.rubric_id,
        "rubric_b": suite_b.rubric_id,
        "consistency_change": consistency_change,
        "items": items,
        "summary": f"Consistency {direction} by {abs(consistency_change):.3f} ({suite_a.rubric_id} -> {suite_b.rubric_id})",
    }


def format_test_report(suite: RubricTestSuite) -> str:
    """Format a human-readable report for a rubric test suite."""
    lines: List[str] = []
    lines.append(f"Rubric Test Report: {suite.rubric_id}")
    lines.append("=" * 50)
    lines.append(f"Total items: {suite.total_items}")
    lines.append(f"Overall consistency: {suite.overall_consistency:.3f}")
    lines.append(f"Edge cases: {suite.edge_case_count}")
    lines.append("")
    lines.append("Item Results:")
    lines.append("-" * 50)

    for r in suite.results:
        marker = " [EDGE CASE]" if r.is_edge_case else ""
        lines.append(
            f"  {r.item_id}: mean={r.mean_score:.3f} std={r.std_dev:.3f} "
            f"consistency={r.consistency:.3f} "
            f"passed={r.passed_count}/{r.total_runs}{marker}"
        )

    return "\n".join(lines)
