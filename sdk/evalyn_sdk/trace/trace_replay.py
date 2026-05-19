from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any

from ..models import Span


@dataclass
class ReplayInput:
    """Input extracted from a traced LLM span for replay."""

    span_id: str
    original_model: str
    input_messages: list[dict[str, str]] = field(default_factory=list)
    original_output: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "original_model": self.original_model,
            "input_messages": list(self.input_messages),
            "original_output": self.original_output,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayInput:
        return cls(
            span_id=data["span_id"],
            original_model=data["original_model"],
            input_messages=data.get("input_messages", []),
            original_output=data.get("original_output", ""),
        )


@dataclass
class ReplayOutput:
    """Output from replaying a span against a different model."""

    span_id: str
    replayed_model: str
    replayed_output: str = ""
    cost_usd: float = 0.0
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "replayed_model": self.replayed_model,
            "replayed_output": self.replayed_output,
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ReplayComparison:
    """Side-by-side comparison of original and replayed outputs."""

    span_id: str
    original_output: str
    replayed_output: str
    output_diff: str = ""
    similarity: float = 0.0
    cost_delta: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "original_output": self.original_output,
            "replayed_output": self.replayed_output,
            "output_diff": self.output_diff,
            "similarity": self.similarity,
            "cost_delta": self.cost_delta,
        }

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Span: {self.span_id}")
        lines.append(f"  Similarity: {self.similarity:.2%}")
        if self.cost_delta != 0:
            lines.append(f"  Cost delta: {self.cost_delta:+.4f} USD")
        if self.output_diff:
            lines.append("  Diff:")
            for dl in self.output_diff.splitlines():
                lines.append(f"    {dl}")
        return "\n".join(lines)


@dataclass
class TraceReplayPlan:
    """Plan for replaying trace spans against a target model."""

    inputs: list[ReplayInput] = field(default_factory=list)
    target_model: str = ""
    estimated_calls: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "inputs": [i.as_dict() for i in self.inputs],
            "target_model": self.target_model,
            "estimated_calls": self.estimated_calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceReplayPlan:
        return cls(
            inputs=[ReplayInput.from_dict(i) for i in data.get("inputs", [])],
            target_model=data.get("target_model", ""),
            estimated_calls=data.get("estimated_calls", 0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Replay plan: {self.estimated_calls} calls to {self.target_model}")
        for inp in self.inputs:
            msg_count = len(inp.input_messages)
            lines.append(
                f"  {inp.span_id}: {inp.original_model} -> {self.target_model} "
                f"({msg_count} messages)"
            )
        return "\n".join(lines)


@dataclass
class TraceReplayReport:
    """Aggregated report from replaying a trace."""

    comparisons: list[ReplayComparison] = field(default_factory=list)
    avg_similarity: float = 0.0
    total_cost_delta: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "comparisons": [c.as_dict() for c in self.comparisons],
            "avg_similarity": self.avg_similarity,
            "total_cost_delta": self.total_cost_delta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceReplayReport:
        comparisons = [
            ReplayComparison(**c) if isinstance(c, dict) and "span_id" in c else c
            for c in data.get("comparisons", [])
        ]
        return cls(
            comparisons=comparisons,
            avg_similarity=data.get("avg_similarity", 0.0),
            total_cost_delta=data.get("total_cost_delta", 0.0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Replay report: {len(self.comparisons)} comparisons")
        lines.append(f"  Average similarity: {self.avg_similarity:.2%}")
        lines.append(f"  Total cost delta: {self.total_cost_delta:+.4f} USD")
        lines.append("")
        for comp in self.comparisons:
            lines.append(comp.format_text())
        return "\n".join(lines)


def extract_replay_inputs(spans: list[Span]) -> list[ReplayInput]:
    """Extract input messages from LLM spans for replay.

    Looks for 'input', 'messages', or 'prompt' in span attributes.
    Only processes spans with span_type 'llm_call'.
    """
    results: list[ReplayInput] = []
    for span in spans:
        if span.span_type != "llm_call":
            continue

        attrs = span.attributes
        model = str(attrs.get("model", ""))

        # Extract messages from attributes
        messages: list[dict[str, str]] = []
        if "messages" in attrs and isinstance(attrs["messages"], list):
            messages = attrs["messages"]
        elif "input" in attrs:
            content = str(attrs["input"])
            messages = [{"role": "user", "content": content}]
        elif "prompt" in attrs:
            content = str(attrs["prompt"])
            messages = [{"role": "user", "content": content}]

        original_output = str(attrs.get("output", ""))

        results.append(
            ReplayInput(
                span_id=span.id,
                original_model=model,
                input_messages=messages,
                original_output=original_output,
            )
        )
    return results


def build_replay_plan(inputs: list[ReplayInput], target_model: str) -> TraceReplayPlan:
    """Build a replay plan from extracted inputs."""
    return TraceReplayPlan(
        inputs=list(inputs),
        target_model=target_model,
        estimated_calls=len(inputs),
    )


def compute_text_similarity(a: str, b: str) -> float:
    """Compute word-overlap similarity (Jaccard index) between two texts."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def compare_replay(original: ReplayInput, replayed: ReplayOutput) -> ReplayComparison:
    """Build a comparison between original and replayed outputs."""
    orig_lines = original.original_output.splitlines(keepends=True)
    replay_lines = replayed.replayed_output.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(orig_lines, replay_lines, fromfile="original", tofile="replayed")
    )
    output_diff = "".join(diff_lines)

    similarity = compute_text_similarity(original.original_output, replayed.replayed_output)

    return ReplayComparison(
        span_id=original.span_id,
        original_output=original.original_output,
        replayed_output=replayed.replayed_output,
        output_diff=output_diff,
        similarity=similarity,
        cost_delta=replayed.cost_usd,
    )


def build_replay_report(comparisons: list[ReplayComparison]) -> TraceReplayReport:
    """Aggregate comparisons into a replay report."""
    if not comparisons:
        return TraceReplayReport()

    avg_sim = sum(c.similarity for c in comparisons) / len(comparisons)
    total_cost = sum(c.cost_delta for c in comparisons)

    return TraceReplayReport(
        comparisons=list(comparisons),
        avg_similarity=avg_sim,
        total_cost_delta=total_cost,
    )
