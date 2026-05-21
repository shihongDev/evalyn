"""Agent goal completion metrics.

Evaluate whether agents achieve stated objectives using deterministic
checks: tool call accuracy, topic adherence, keyword coverage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GoalResult:
    """Result of a single goal check."""

    goal_id: str
    achieved: bool
    score: float  # 0-1
    method: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "achieved": self.achieved,
            "score": self.score,
            "method": self.method,
            "details": dict(self.details),
        }


@dataclass
class GoalReport:
    """Aggregated report across multiple goal checks."""

    results: list[GoalResult]
    total_goals: int
    achieved_count: int
    achievement_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "total_goals": self.total_goals,
            "achieved_count": self.achieved_count,
            "achievement_rate": self.achievement_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoalReport:
        results = [
            GoalResult(
                goal_id=r["goal_id"],
                achieved=r["achieved"],
                score=r["score"],
                method=r["method"],
                details=r.get("details", {}),
            )
            for r in data.get("results", [])
        ]
        return cls(
            results=results,
            total_goals=data.get("total_goals", 0),
            achieved_count=data.get("achieved_count", 0),
            achievement_rate=data.get("achievement_rate", 0.0),
        )

    def format_text(self) -> str:
        """Format report as plain text."""
        lines: list[str] = []
        lines.append(
            f"Goal Completion: {self.achieved_count}/{self.total_goals}"
            f" ({self.achievement_rate:.0%})"
        )
        lines.append("")
        for r in self.results:
            status = "PASS" if r.achieved else "FAIL"
            lines.append(f"  [{status}] {r.goal_id} - score={r.score:.2f} ({r.method})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LCS helper
# ---------------------------------------------------------------------------


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length (dynamic programming)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimized: two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


# ---------------------------------------------------------------------------
# Goal check functions
# ---------------------------------------------------------------------------


def check_tool_call_accuracy(
    expected_calls: list[str],
    actual_calls: list[str],
) -> GoalResult:
    """Compare expected vs actual tool call sequences using LCS.

    Score = LCS length / max(len(expected), len(actual)).
    """
    if not expected_calls and not actual_calls:
        return GoalResult(
            goal_id="tool_call_accuracy",
            achieved=True,
            score=1.0,
            method="lcs",
            details={"expected": expected_calls, "actual": actual_calls, "lcs_length": 0},
        )

    max_len = max(len(expected_calls), len(actual_calls))
    lcs_len = _lcs_length(expected_calls, actual_calls)
    score = lcs_len / max_len if max_len > 0 else 0.0

    return GoalResult(
        goal_id="tool_call_accuracy",
        achieved=score >= 1.0,
        score=score,
        method="lcs",
        details={
            "expected": expected_calls,
            "actual": actual_calls,
            "lcs_length": lcs_len,
        },
    )


def check_tool_call_f1(
    expected_calls: list[str],
    actual_calls: list[str],
) -> GoalResult:
    """Unordered comparison of expected vs actual tool calls using F1.

    Precision = |intersection| / |actual|
    Recall = |intersection| / |expected|
    F1 = harmonic mean of precision and recall
    """
    expected_set = set(expected_calls)
    actual_set = set(actual_calls)
    intersection = expected_set & actual_set

    if not expected_set and not actual_set:
        return GoalResult(
            goal_id="tool_call_f1",
            achieved=True,
            score=1.0,
            method="f1",
            details={"precision": 1.0, "recall": 1.0, "f1": 1.0},
        )

    precision = len(intersection) / len(actual_set) if actual_set else 0.0
    recall = len(intersection) / len(expected_set) if expected_set else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return GoalResult(
        goal_id="tool_call_f1",
        achieved=f1 >= 1.0,
        score=f1,
        method="f1",
        details={"precision": precision, "recall": recall, "f1": f1},
    )


def check_topic_adherence(
    output: str,
    allowed_topics: list[str],
    forbidden_topics: list[str] | None = None,
) -> GoalResult:
    """Check if output mentions allowed topics and avoids forbidden ones.

    Uses simple case-insensitive keyword matching.
    Score is based on: (allowed found - forbidden found) / total keywords,
    clamped to [0, 1].
    """
    if forbidden_topics is None:
        forbidden_topics = []

    output_lower = output.lower()

    allowed_found = [t for t in allowed_topics if t.lower() in output_lower]
    forbidden_found = [t for t in forbidden_topics if t.lower() in output_lower]

    total_keywords = len(allowed_topics) + len(forbidden_topics)
    if total_keywords == 0:
        score = 1.0
    else:
        allowed_score = len(allowed_found) / len(allowed_topics) if allowed_topics else 1.0
        forbidden_penalty = (
            len(forbidden_found) / len(forbidden_topics) if forbidden_topics else 0.0
        )
        score = max(0.0, min(1.0, allowed_score - forbidden_penalty))

    achieved = len(forbidden_found) == 0 and len(allowed_found) == len(allowed_topics)

    return GoalResult(
        goal_id="topic_adherence",
        achieved=achieved,
        score=score,
        method="keyword",
        details={
            "allowed_found": allowed_found,
            "forbidden_found": forbidden_found,
            "allowed_total": len(allowed_topics),
            "forbidden_total": len(forbidden_topics),
        },
    )


def check_goal_completion(
    output: str,
    goal_keywords: list[str],
) -> GoalResult:
    """Check if output contains all goal keywords.

    Score = fraction of keywords found (case-insensitive).
    """
    if not goal_keywords:
        return GoalResult(
            goal_id="goal_completion",
            achieved=True,
            score=1.0,
            method="keyword_coverage",
            details={"found": [], "missing": [], "total": 0},
        )

    output_lower = output.lower()
    found = [kw for kw in goal_keywords if kw.lower() in output_lower]
    missing = [kw for kw in goal_keywords if kw.lower() not in output_lower]
    score = len(found) / len(goal_keywords)

    return GoalResult(
        goal_id="goal_completion",
        achieved=len(missing) == 0,
        score=score,
        method="keyword_coverage",
        details={"found": found, "missing": missing, "total": len(goal_keywords)},
    )


def evaluate_goals(
    goals: list[tuple[str, Callable[[], GoalResult]]],
) -> GoalReport:
    """Run all goal checks and build an aggregated report.

    Args:
        goals: List of (goal_id, callable) pairs. Each callable returns a
               GoalResult. The goal_id from the tuple is set on the result.
    """
    results: list[GoalResult] = []

    for goal_id, check_fn in goals:
        result = check_fn()
        result.goal_id = goal_id
        results.append(result)

    total = len(results)
    achieved = sum(1 for r in results if r.achieved)
    rate = achieved / total if total > 0 else 0.0

    return GoalReport(
        results=results,
        total_goals=total,
        achieved_count=achieved,
        achievement_rate=rate,
    )
