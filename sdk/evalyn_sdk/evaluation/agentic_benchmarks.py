"""
Agentic benchmark integration.

Run standard agent benchmarks within evalyn: define tasks, evaluate agent
outputs against expected tool usage and output keywords, and compare runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class BenchmarkTask:
    """A single benchmark task definition."""

    task_id: str
    name: str
    description: str = ""
    category: str = ""
    expected_tools: List[str] = field(default_factory=list)
    expected_output_keywords: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # "easy" / "medium" / "hard"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "expected_tools": list(self.expected_tools),
            "expected_output_keywords": list(self.expected_output_keywords),
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkTask:
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            description=data.get("description", ""),
            category=data.get("category", ""),
            expected_tools=data.get("expected_tools", []),
            expected_output_keywords=data.get("expected_output_keywords", []),
            difficulty=data.get("difficulty", "medium"),
        )


@dataclass
class BenchmarkResult:
    """Result of evaluating a single benchmark task."""

    task_id: str
    passed: bool
    score: float
    tool_calls_correct: bool = False
    output_correct: bool = False
    duration_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "score": self.score,
            "tool_calls_correct": self.tool_calls_correct,
            "output_correct": self.output_correct,
            "duration_ms": self.duration_ms,
        }


@dataclass
class BenchmarkSuite:
    """A collection of benchmark tasks."""

    name: str
    tasks: List[BenchmarkTask] = field(default_factory=list)
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tasks": [t.as_dict() for t in self.tasks],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkSuite:
        tasks = [BenchmarkTask.from_dict(t) for t in data.get("tasks", [])]
        return cls(
            name=data["name"],
            tasks=tasks,
            description=data.get("description", ""),
        )


@dataclass
class BenchmarkReport:
    """Report summarizing a full benchmark suite run."""

    suite_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    pass_rate: float = 0.0
    avg_score: float = 0.0
    by_difficulty: Dict[str, float] = field(default_factory=dict)
    by_category: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "results": [r.as_dict() for r in self.results],
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "by_difficulty": dict(self.by_difficulty),
            "by_category": dict(self.by_category),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkReport:
        results = [
            BenchmarkResult(
                task_id=r["task_id"],
                passed=r["passed"],
                score=r["score"],
                tool_calls_correct=r.get("tool_calls_correct", False),
                output_correct=r.get("output_correct", False),
                duration_ms=r.get("duration_ms", 0.0),
            )
            for r in data.get("results", [])
        ]
        return cls(
            suite_name=data["suite_name"],
            results=results,
            pass_rate=data.get("pass_rate", 0.0),
            avg_score=data.get("avg_score", 0.0),
            by_difficulty=data.get("by_difficulty", {}),
            by_category=data.get("by_category", {}),
        )

    def format_text(self) -> str:
        lines: List[str] = []
        lines.append(f"BENCHMARK REPORT: {self.suite_name}")
        lines.append(f"Pass rate: {self.pass_rate:.1%}")
        lines.append(f"Avg score: {self.avg_score:.2f}")
        lines.append("")

        if self.by_difficulty:
            lines.append("BY DIFFICULTY:")
            for diff, score in sorted(self.by_difficulty.items()):
                lines.append(f"  {diff}: {score:.2f}")
            lines.append("")

        if self.by_category:
            lines.append("BY CATEGORY:")
            for cat, score in sorted(self.by_category.items()):
                lines.append(f"  {cat}: {score:.2f}")
            lines.append("")

        lines.append("RESULTS:")
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(
                f"  {r.task_id}: {status} (score: {r.score:.2f}, "
                f"tools: {'ok' if r.tool_calls_correct else 'wrong'}, "
                f"output: {'ok' if r.output_correct else 'wrong'})"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in benchmark tasks
# ---------------------------------------------------------------------------

BASIC_AGENT_TASKS = [
    BenchmarkTask(
        "search_and_answer",
        "Search and Answer",
        "Agent should search then answer",
        "retrieval",
        ["search"],
        ["answer"],
        "easy",
    ),
    BenchmarkTask(
        "multi_step_tool",
        "Multi-Step Tool Use",
        "Agent uses multiple tools in sequence",
        "tool_use",
        ["search", "calculate"],
        ["result"],
        "medium",
    ),
    BenchmarkTask(
        "error_recovery",
        "Error Recovery",
        "Agent handles tool failure gracefully",
        "robustness",
        ["search"],
        ["sorry", "unable"],
        "hard",
    ),
]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def create_basic_suite() -> BenchmarkSuite:
    """Return a benchmark suite containing the built-in BASIC_AGENT_TASKS."""
    return BenchmarkSuite(
        name="basic_agent",
        tasks=list(BASIC_AGENT_TASKS),
        description="Basic agent capability benchmark",
    )


def evaluate_task(
    task: BenchmarkTask,
    actual_tools: List[str],
    actual_output: str,
) -> BenchmarkResult:
    """Evaluate a single benchmark task against agent output.

    Checks:
    - tool_calls_correct: all expected tools appear in actual_tools
    - output_correct: at least one expected keyword appears in output (case-insensitive)
    - score: average of both boolean checks (0.0, 0.5, or 1.0)
    - passed: score >= 0.5

    Args:
        task: The benchmark task definition.
        actual_tools: List of tool names the agent called.
        actual_output: The agent's output text.

    Returns:
        BenchmarkResult with evaluation details.
    """
    tool_calls_correct = all(t in actual_tools for t in task.expected_tools)

    output_lower = actual_output.lower()
    output_correct = any(kw.lower() in output_lower for kw in task.expected_output_keywords)

    checks = [float(tool_calls_correct), float(output_correct)]
    score = sum(checks) / len(checks) if checks else 0.0
    passed = score >= 0.5

    return BenchmarkResult(
        task_id=task.task_id,
        passed=passed,
        score=score,
        tool_calls_correct=tool_calls_correct,
        output_correct=output_correct,
    )


def run_benchmark_suite(
    suite: BenchmarkSuite,
    agent_outputs: Dict[str, Tuple[List[str], str]],
) -> BenchmarkReport:
    """Run all tasks in a suite against provided agent outputs.

    Args:
        suite: The benchmark suite to run.
        agent_outputs: Mapping of task_id to (tool_calls, output_text).
            Tasks missing from the dict are scored as failures.

    Returns:
        BenchmarkReport with per-task results and aggregate stats.
    """
    results: List[BenchmarkResult] = []
    # Track scores per difficulty and category for breakdown
    difficulty_scores: Dict[str, List[float]] = {}
    category_scores: Dict[str, List[float]] = {}

    for task in suite.tasks:
        if task.task_id in agent_outputs:
            tools, output = agent_outputs[task.task_id]
            result = evaluate_task(task, tools, output)
        else:
            # Missing output - score as failure
            result = BenchmarkResult(
                task_id=task.task_id,
                passed=False,
                score=0.0,
                tool_calls_correct=False,
                output_correct=False,
            )
        results.append(result)

        # Accumulate by difficulty
        difficulty_scores.setdefault(task.difficulty, []).append(result.score)
        # Accumulate by category
        if task.category:
            category_scores.setdefault(task.category, []).append(result.score)

    pass_rate = sum(1 for r in results if r.passed) / len(results) if results else 0.0
    avg_score = sum(r.score for r in results) / len(results) if results else 0.0

    by_difficulty = {
        d: sum(scores) / len(scores) for d, scores in difficulty_scores.items() if scores
    }
    by_category = {
        c: sum(scores) / len(scores) for c, scores in category_scores.items() if scores
    }

    return BenchmarkReport(
        suite_name=suite.name,
        results=results,
        pass_rate=pass_rate,
        avg_score=avg_score,
        by_difficulty=by_difficulty,
        by_category=by_category,
    )


def compare_benchmark_results(
    report_a: BenchmarkReport,
    report_b: BenchmarkReport,
) -> Dict[str, Any]:
    """Compare two benchmark runs and return a diff summary.

    Args:
        report_a: The baseline report.
        report_b: The comparison report.

    Returns:
        Dict with pass_rate_delta, avg_score_delta, per-task deltas, and
        difficulty/category breakdowns.
    """
    scores_a = {r.task_id: r.score for r in report_a.results}
    scores_b = {r.task_id: r.score for r in report_b.results}

    all_task_ids = sorted(set(scores_a.keys()) | set(scores_b.keys()))
    per_task: Dict[str, Dict[str, Any]] = {}
    for tid in all_task_ids:
        sa = scores_a.get(tid, 0.0)
        sb = scores_b.get(tid, 0.0)
        per_task[tid] = {
            "score_a": sa,
            "score_b": sb,
            "delta": sb - sa,
        }

    return {
        "suite_a": report_a.suite_name,
        "suite_b": report_b.suite_name,
        "pass_rate_delta": report_b.pass_rate - report_a.pass_rate,
        "avg_score_delta": report_b.avg_score - report_a.avg_score,
        "per_task": per_task,
        "by_difficulty_a": dict(report_a.by_difficulty),
        "by_difficulty_b": dict(report_b.by_difficulty),
        "by_category_a": dict(report_a.by_category),
        "by_category_b": dict(report_b.by_category),
    }
