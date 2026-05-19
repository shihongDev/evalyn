"""Agent consistency testing: measure reliability across repeated runs.

Run the same input through an agent multiple times and quantify how
consistent the outputs, tool calls, and pass/fail outcomes are.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunOutput:
    """Single run output for one item."""

    run_index: int
    output: str
    tool_calls: list[str] = field(default_factory=list)
    passed: bool = True
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "output": self.output,
            "tool_calls": list(self.tool_calls),
            "passed": self.passed,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ConsistencyResult:
    """Consistency metrics for one item across multiple runs."""

    item_id: str
    num_runs: int
    output_consistency: float  # 0-1: fraction matching most common output
    tool_call_consistency: float  # 0-1: fraction with identical tool call sequences
    pass_consistency: float  # 0-1: fraction that passed
    outputs: list[RunOutput] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "num_runs": self.num_runs,
            "output_consistency": round(self.output_consistency, 4),
            "tool_call_consistency": round(self.tool_call_consistency, 4),
            "pass_consistency": round(self.pass_consistency, 4),
            "outputs": [o.as_dict() for o in self.outputs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsistencyResult:
        outputs = [
            RunOutput(
                run_index=o["run_index"],
                output=o["output"],
                tool_calls=o.get("tool_calls", []),
                passed=o.get("passed", True),
                duration_ms=o.get("duration_ms", 0.0),
            )
            for o in data.get("outputs", [])
        ]
        return cls(
            item_id=data["item_id"],
            num_runs=data["num_runs"],
            output_consistency=data["output_consistency"],
            tool_call_consistency=data["tool_call_consistency"],
            pass_consistency=data["pass_consistency"],
            outputs=outputs,
        )


@dataclass
class ConsistencyReport:
    """Aggregated consistency report across all items."""

    results: list[ConsistencyResult] = field(default_factory=list)
    avg_output_consistency: float = 0.0
    avg_pass_consistency: float = 0.0
    total_items: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "avg_output_consistency": round(self.avg_output_consistency, 4),
            "avg_pass_consistency": round(self.avg_pass_consistency, 4),
            "total_items": self.total_items,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsistencyReport:
        results = [ConsistencyResult.from_dict(r) for r in data.get("results", [])]
        return cls(
            results=results,
            avg_output_consistency=data.get("avg_output_consistency", 0.0),
            avg_pass_consistency=data.get("avg_pass_consistency", 0.0),
            total_items=data.get("total_items", 0),
        )

    def format_text(self) -> str:
        lines = [
            "Consistency Report",
            f"  Total items:            {self.total_items}",
            f"  Avg output consistency: {self.avg_output_consistency:.1%}",
            f"  Avg pass consistency:   {self.avg_pass_consistency:.1%}",
        ]
        for r in self.results:
            lines.append(
                f"  {r.item_id}: output={r.output_consistency:.1%}"
                f"  tools={r.tool_call_consistency:.1%}"
                f"  pass={r.pass_consistency:.1%}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def measure_output_consistency(outputs: list[str]) -> float:
    """Fraction of outputs matching the most common output.

    Example: ["a", "a", "b"] -> 2/3.
    Returns 0.0 for empty input.
    """
    if not outputs:
        return 0.0
    counter = Counter(outputs)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(outputs)


def measure_tool_call_consistency(tool_call_sets: list[list[str]]) -> float:
    """Fraction of runs with identical tool call sequences.

    Compares each run's tool call list to the most common sequence.
    Returns 0.0 for empty input.
    """
    if not tool_call_sets:
        return 0.0
    # Convert each list to a tuple so it is hashable
    tuples = [tuple(tc) for tc in tool_call_sets]
    counter = Counter(tuples)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(tuples)


def analyze_consistency(item_id: str, runs: list[RunOutput]) -> ConsistencyResult:
    """Compute all consistency metrics for one item."""
    if not runs:
        return ConsistencyResult(
            item_id=item_id,
            num_runs=0,
            output_consistency=0.0,
            tool_call_consistency=0.0,
            pass_consistency=0.0,
            outputs=[],
        )
    outputs = [r.output for r in runs]
    tool_call_sets = [r.tool_calls for r in runs]
    passed_count = sum(1 for r in runs if r.passed)
    return ConsistencyResult(
        item_id=item_id,
        num_runs=len(runs),
        output_consistency=measure_output_consistency(outputs),
        tool_call_consistency=measure_tool_call_consistency(tool_call_sets),
        pass_consistency=passed_count / len(runs),
        outputs=list(runs),
    )


def build_consistency_report(results: list[ConsistencyResult]) -> ConsistencyReport:
    """Aggregate individual consistency results into a report."""
    if not results:
        return ConsistencyReport(
            results=[],
            avg_output_consistency=0.0,
            avg_pass_consistency=0.0,
            total_items=0,
        )
    avg_output = sum(r.output_consistency for r in results) / len(results)
    avg_pass = sum(r.pass_consistency for r in results) / len(results)
    return ConsistencyReport(
        results=list(results),
        avg_output_consistency=avg_output,
        avg_pass_consistency=avg_pass,
        total_items=len(results),
    )
