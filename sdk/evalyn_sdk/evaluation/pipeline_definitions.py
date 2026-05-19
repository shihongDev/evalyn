from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

VALID_STEP_TYPES = ("trace", "evaluate", "analyze", "calibrate", "export")


@dataclass
class PipelineStep:
    """A single step in a pipeline."""

    name: str
    step_type: str
    config: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "step_type": self.step_type,
            "config": self.config,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineStep:
        return cls(
            name=data["name"],
            step_type=data["step_type"],
            config=data.get("config", {}),
            depends_on=data.get("depends_on", []),
        )


@dataclass
class PipelineDefinition:
    """A named sequence of pipeline steps."""

    name: str
    description: str = ""
    steps: list[PipelineStep] = field(default_factory=list)
    version: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": [s.as_dict() for s in self.steps],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineDefinition:
        steps = [PipelineStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            version=data.get("version", 1),
        )

    @property
    def step_names(self) -> list[str]:
        return [s.name for s in self.steps]

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Pipeline: {self.name} (v{self.version})")
        if self.description:
            lines.append(f"  {self.description}")
        lines.append(f"  Steps ({len(self.steps)}):")
        for i, s in enumerate(self.steps, 1):
            deps = ""
            if s.depends_on:
                deps = f" (after: {', '.join(s.depends_on)})"
            lines.append(f"    {i}. {s.name} [{s.step_type}]{deps}")
        return "\n".join(lines)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate the pipeline definition.

        Returns (is_valid, list_of_errors).
        Checks:
        - All depends_on references point to existing step names.
        - No circular dependencies.
        """
        errors: list[str] = []
        names = set(self.step_names)

        # Check missing dependency references
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in names:
                    errors.append(
                        f"Step '{step.name}' depends on unknown step '{dep}'"
                    )

        # Check circular dependencies via topological sort attempt
        in_degree: dict[str, int] = {s.name: 0 for s in self.steps}
        adj: dict[str, list[str]] = {s.name: [] for s in self.steps}
        for step in self.steps:
            for dep in step.depends_on:
                if dep in names:
                    adj[dep].append(step.name)
                    in_degree[step.name] += 1

        queue: deque[str] = deque()
        for name, deg in in_degree.items():
            if deg == 0:
                queue.append(name)

        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited < len(self.steps):
            errors.append("Circular dependency detected among steps")

        return (len(errors) == 0, errors)


@dataclass
class PipelineRunResult:
    """Result of executing a pipeline."""

    pipeline_name: str
    step_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    completed_steps: list[str] = field(default_factory=list)
    failed_step: str = ""
    total_duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "step_results": self.step_results,
            "completed_steps": self.completed_steps,
            "failed_step": self.failed_step,
            "total_duration_ms": self.total_duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineRunResult:
        return cls(
            pipeline_name=data["pipeline_name"],
            step_results=data.get("step_results", {}),
            completed_steps=data.get("completed_steps", []),
            failed_step=data.get("failed_step", ""),
            total_duration_ms=data.get("total_duration_ms", 0.0),
        )

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Pipeline Run: {self.pipeline_name}")
        lines.append(f"  Duration: {self.total_duration_ms:.1f} ms")
        lines.append(f"  Completed: {len(self.completed_steps)} step(s)")
        for name in self.completed_steps:
            lines.append(f"    - {name}")
        if self.failed_step:
            lines.append(f"  Failed at: {self.failed_step}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in pipelines
# ---------------------------------------------------------------------------

QUICK_EVAL_PIPELINE = PipelineDefinition(
    name="quick-eval",
    description="Fast evaluation with objective metrics only",
    steps=[
        PipelineStep("load_data", "trace"),
        PipelineStep("run_eval", "evaluate", {"profile": "smoke-test"}),
        PipelineStep("analyze", "analyze"),
    ],
)

FULL_PIPELINE = PipelineDefinition(
    name="full",
    description="Complete evaluation with calibration",
    steps=[
        PipelineStep("load_data", "trace"),
        PipelineStep("run_eval", "evaluate", {"profile": "standard"}),
        PipelineStep("analyze", "analyze"),
        PipelineStep("calibrate", "calibrate"),
        PipelineStep("export", "export"),
    ],
)

_BUILTIN_PIPELINES: dict[str, PipelineDefinition] = {
    p.name: p for p in [QUICK_EVAL_PIPELINE, FULL_PIPELINE]
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_pipeline(name: str) -> PipelineDefinition:
    """Look up a pipeline by name."""
    if name not in _BUILTIN_PIPELINES:
        raise KeyError(f"Unknown pipeline: {name}")
    return _BUILTIN_PIPELINES[name]


def list_pipelines() -> list[PipelineDefinition]:
    """Return all available pipeline definitions."""
    return list(_BUILTIN_PIPELINES.values())


def create_pipeline(name: str, steps: list[dict[str, Any]]) -> PipelineDefinition:
    """Create a pipeline from a list of step dicts."""
    parsed_steps = [PipelineStep.from_dict(s) for s in steps]
    return PipelineDefinition(name=name, steps=parsed_steps)


def get_execution_order(pipeline: PipelineDefinition) -> list[str]:
    """Return step names in topological order respecting depends_on.

    Steps with no dependencies come first. Among steps at the same
    dependency depth, insertion order from the definition is preserved.
    """
    names = set(pipeline.step_names)
    # Map name -> index for stable ordering
    name_to_idx = {s.name: i for i, s in enumerate(pipeline.steps)}

    in_degree: dict[str, int] = {s.name: 0 for s in pipeline.steps}
    adj: dict[str, list[str]] = {s.name: [] for s in pipeline.steps}
    for step in pipeline.steps:
        for dep in step.depends_on:
            if dep in names:
                adj[dep].append(step.name)
                in_degree[step.name] += 1

    # Use a list sorted by original index for stable BFS
    queue: list[str] = sorted(
        [n for n, d in in_degree.items() if d == 0],
        key=lambda n: name_to_idx[n],
    )
    order: list[str] = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        next_ready: list[str] = []
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                next_ready.append(neighbor)
        # Insert in stable order
        next_ready.sort(key=lambda n: name_to_idx[n])
        for n in next_ready:
            queue.append(n)
        queue.sort(key=lambda n: name_to_idx[n])

    return order


def compare_pipeline_runs(
    result_a: PipelineRunResult,
    result_b: PipelineRunResult,
) -> dict[str, Any]:
    """Compare two pipeline run results."""
    a_steps = set(result_a.completed_steps)
    b_steps = set(result_b.completed_steps)

    duration_diff = result_b.total_duration_ms - result_a.total_duration_ms

    return {
        "pipeline_a": result_a.pipeline_name,
        "pipeline_b": result_b.pipeline_name,
        "completed_a": len(result_a.completed_steps),
        "completed_b": len(result_b.completed_steps),
        "common_steps": sorted(a_steps & b_steps),
        "only_in_a": sorted(a_steps - b_steps),
        "only_in_b": sorted(b_steps - a_steps),
        "duration_diff_ms": duration_diff,
        "a_failed": result_a.failed_step,
        "b_failed": result_b.failed_step,
    }
