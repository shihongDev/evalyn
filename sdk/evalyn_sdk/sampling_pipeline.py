"""Multi-stage sampling pipeline: chain arbitrary strategies in sequence.

Provides dataclasses and pure functions for building, running, and reporting
on multi-stage sampling pipelines. Pure Python, no external dependencies.
"""

from __future__ import annotations

import random as _random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineStage:
    """Execution record for one stage of a sampling pipeline."""

    stage_name: str
    strategy: str
    input_count: int = 0
    output_count: int = 0
    duration_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "strategy": self.strategy,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineStage:
        return cls(
            stage_name=data["stage_name"],
            strategy=data.get("strategy", ""),
            input_count=data.get("input_count", 0),
            output_count=data.get("output_count", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
        )


@dataclass
class PipelineConfig:
    """Declarative pipeline configuration: ordered stage names."""

    stages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"stages": list(self.stages)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        return cls(stages=list(data.get("stages", [])))


@dataclass
class PipelineResult:
    """Outcome of running a full sampling pipeline."""

    final_ids: list[str] = field(default_factory=list)
    stage_results: list[PipelineStage] = field(default_factory=list)
    total_input: int = 0
    total_output: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_ids": list(self.final_ids),
            "stage_results": [s.as_dict() for s in self.stage_results],
            "total_input": self.total_input,
            "total_output": self.total_output,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineResult:
        return cls(
            final_ids=list(data.get("final_ids", [])),
            stage_results=[PipelineStage.from_dict(s) for s in data.get("stage_results", [])],
            total_input=data.get("total_input", 0),
            total_output=data.get("total_output", 0),
        )


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def run_pipeline(
    item_ids: list[str],
    stage_fns: list[tuple[str, Callable[[list[str]], list[str]]]],
) -> PipelineResult:
    """Run each stage function in sequence, feeding output to the next stage.

    Each entry in stage_fns is (stage_name, fn). Per-stage input/output
    counts and wall-clock timing are recorded.
    """
    current = list(item_ids)
    total_input = len(current)
    stage_results: list[PipelineStage] = []

    for name, fn in stage_fns:
        in_count = len(current)
        t0 = time.monotonic()
        current = fn(current)
        elapsed = time.monotonic() - t0
        stage_results.append(
            PipelineStage(
                stage_name=name,
                strategy=name,
                input_count=in_count,
                output_count=len(current),
                duration_seconds=round(elapsed, 6),
            )
        )

    return PipelineResult(
        final_ids=current,
        stage_results=stage_results,
        total_input=total_input,
        total_output=len(current),
    )


# ---------------------------------------------------------------------------
# Built-in stage functions
# ---------------------------------------------------------------------------


def deduplicate_stage(ids: list[str]) -> list[str]:
    """Remove exact duplicates, preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def shuffle_stage(ids: list[str], seed: int = 42) -> list[str]:
    """Deterministic shuffle using the given seed."""
    result = list(ids)
    _random.Random(seed).shuffle(result)
    return result


def limit_stage(ids: list[str], max_count: int = 100) -> list[str]:
    """Truncate to at most max_count items."""
    return ids[:max_count]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_stage_fn(
    name: str, fn: Callable[[list[str]], list[str]]
) -> tuple[str, Callable[[list[str]], list[str]]]:
    """Convenience wrapper: pair a stage name with its function."""
    return (name, fn)


def format_pipeline_report(result: PipelineResult) -> str:
    """Per-stage funnel report showing input -> output counts."""
    lines = [
        "Pipeline Report",
        "=" * 40,
        f"Total input  : {result.total_input}",
        f"Total output : {result.total_output}",
        "-" * 40,
    ]
    for i, stage in enumerate(result.stage_results, 1):
        drop = stage.input_count - stage.output_count
        lines.append(
            f"  {i}. {stage.stage_name}: "
            f"{stage.input_count} -> {stage.output_count} "
            f"(dropped {drop}, {stage.duration_seconds:.4f}s)"
        )
    return "\n".join(lines)
