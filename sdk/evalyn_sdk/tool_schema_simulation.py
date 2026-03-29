"""
Simulation with tool schemas: generate inputs that exercise specific tool call patterns.

Generates natural-language queries targeting specific tools and tool combinations,
then measures tool coverage across a simulated input suite. Pure Python - no
external deps, no LLM calls.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ToolParameter:
    """A single parameter in a tool schema."""

    name: str
    param_type: str  # "string" / "int" / "float" / "bool" / "array"
    required: bool = True
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "param_type": self.param_type,
            "required": self.required,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolParameter:
        return cls(
            name=data["name"],
            param_type=data["param_type"],
            required=data.get("required", True),
            description=data.get("description", ""),
        )


@dataclass
class ToolSchema:
    """Schema describing a tool and its parameters."""

    tool_name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "description": self.description,
            "parameters": [p.as_dict() for p in self.parameters],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolSchema:
        return cls(
            tool_name=data["tool_name"],
            description=data["description"],
            parameters=[
                ToolParameter.from_dict(p)
                for p in data.get("parameters", [])
            ],
        )


@dataclass
class ToolCoverageResult:
    """Coverage metrics for tool exercise."""

    total_tools: int
    exercised_tools: int
    coverage_rate: float
    uncovered_tools: List[str] = field(default_factory=list)
    per_tool_counts: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_tools": self.total_tools,
            "exercised_tools": self.exercised_tools,
            "coverage_rate": self.coverage_rate,
            "uncovered_tools": self.uncovered_tools,
            "per_tool_counts": self.per_tool_counts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolCoverageResult:
        return cls(
            total_tools=data["total_tools"],
            exercised_tools=data["exercised_tools"],
            coverage_rate=data["coverage_rate"],
            uncovered_tools=data.get("uncovered_tools", []),
            per_tool_counts=data.get("per_tool_counts", {}),
        )


@dataclass
class SimulatedToolInput:
    """A simulated natural-language input targeting a tool or tools."""

    input_text: str
    target_tool: str
    expected_params: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "input_text": self.input_text,
            "target_tool": self.target_tool,
            "expected_params": self.expected_params,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SimulatedToolInput:
        return cls(
            input_text=data["input_text"],
            target_tool=data["target_tool"],
            expected_params=data.get("expected_params", {}),
            tags=data.get("tags", []),
        )


# ---------------------------------------------------------------------------
# Sample value generation helpers
# ---------------------------------------------------------------------------

_SAMPLE_STRINGS = [
    "example", "test input", "hello world", "sample data", "query text",
]
_SAMPLE_INTS = [1, 5, 10, 42, 100]
_SAMPLE_FLOATS = [0.5, 1.0, 3.14, 9.99, 0.01]
_SAMPLE_BOOLS = [True, False]
_SAMPLE_ARRAYS = [["a", "b"], ["x"], [1, 2, 3]]

_QUERY_TEMPLATES = [
    "Please use {tool} to {description}",
    "I need to {description} using {tool}",
    "Can you {description} with {tool}?",
    "Run {tool}: {description}",
    "Use the {tool} tool to {description}",
]

_MULTI_TOOL_TEMPLATES = [
    "First {desc0} using {tool0}, then {desc1} using {tool1}",
    "I need to {desc0} with {tool0} and also {desc1} with {tool1}",
    "Please run {tool0} to {desc0}, followed by {tool1} to {desc1}",
]


def _sample_value(
    param_type: str, rng: random.Random
) -> Any:
    """Generate a sample value for a parameter type."""
    if param_type == "string":
        return rng.choice(_SAMPLE_STRINGS)
    if param_type == "int":
        return rng.choice(_SAMPLE_INTS)
    if param_type == "float":
        return rng.choice(_SAMPLE_FLOATS)
    if param_type == "bool":
        return rng.choice(_SAMPLE_BOOLS)
    if param_type == "array":
        return rng.choice(_SAMPLE_ARRAYS)
    return "unknown"


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def generate_tool_query(
    schema: ToolSchema, rng: Optional[random.Random] = None
) -> SimulatedToolInput:
    """Generate a natural-language query that would require this tool.

    Uses the tool description and parameter names to craft the query.
    """
    if rng is None:
        rng = random.Random()

    template = rng.choice(_QUERY_TEMPLATES)
    base_query = template.format(
        tool=schema.tool_name, description=schema.description
    )

    # Build expected params from required parameters
    expected_params: Dict[str, Any] = {}
    param_fragments: List[str] = []
    for p in schema.parameters:
        if p.required:
            val = _sample_value(p.param_type, rng)
            expected_params[p.name] = val
            param_fragments.append(f"{p.name}={val}")

    if param_fragments:
        base_query += " with " + ", ".join(param_fragments)

    tags = ["single-tool", schema.tool_name]

    return SimulatedToolInput(
        input_text=base_query,
        target_tool=schema.tool_name,
        expected_params=expected_params,
        tags=tags,
    )


def generate_tool_suite(
    schemas: List[ToolSchema], queries_per_tool: int = 3
) -> List[SimulatedToolInput]:
    """Generate queries for all tools in the schema list."""
    inputs: List[SimulatedToolInput] = []
    for i, schema in enumerate(schemas):
        for j in range(queries_per_tool):
            rng = random.Random(i * 1000 + j)
            inputs.append(generate_tool_query(schema, rng))
    return inputs


def compute_tool_coverage(
    inputs: List[SimulatedToolInput], schemas: List[ToolSchema]
) -> ToolCoverageResult:
    """Check which tools are exercised by the input suite."""
    all_tools = {s.tool_name for s in schemas}
    per_tool_counts: Dict[str, int] = {name: 0 for name in all_tools}

    for inp in inputs:
        # A target_tool may refer to a comma-separated list for multi-tool
        targets = [t.strip() for t in inp.target_tool.split(",")]
        for t in targets:
            if t in per_tool_counts:
                per_tool_counts[t] += 1

    exercised = {name for name, count in per_tool_counts.items() if count > 0}
    uncovered = sorted(all_tools - exercised)
    total = len(all_tools)
    exercised_count = len(exercised)
    rate = exercised_count / total if total > 0 else 0.0

    return ToolCoverageResult(
        total_tools=total,
        exercised_tools=exercised_count,
        coverage_rate=rate,
        uncovered_tools=uncovered,
        per_tool_counts=per_tool_counts,
    )


def generate_multi_tool_query(
    schemas: List[ToolSchema],
    n_tools: int = 2,
    rng: Optional[random.Random] = None,
) -> SimulatedToolInput:
    """Generate a query requiring multiple tools."""
    if rng is None:
        rng = random.Random()
    if len(schemas) < n_tools:
        n_tools = len(schemas)

    selected = rng.sample(schemas, n_tools)

    if n_tools >= 2:
        template = rng.choice(_MULTI_TOOL_TEMPLATES)
        query = template.format(
            tool0=selected[0].tool_name,
            desc0=selected[0].description,
            tool1=selected[1].tool_name,
            desc1=selected[1].description,
        )
        # Append remaining tools if n_tools > 2
        for s in selected[2:]:
            query += f", and {s.description} using {s.tool_name}"
    else:
        # Single tool fallback
        query = f"Use {selected[0].tool_name} to {selected[0].description}"

    # Merge expected params from all selected tools
    expected_params: Dict[str, Any] = {}
    for s in selected:
        for p in s.parameters:
            if p.required:
                expected_params[p.name] = _sample_value(p.param_type, rng)

    tool_names = [s.tool_name for s in selected]
    target_tool = ", ".join(tool_names)
    tags = ["multi-tool"] + tool_names

    return SimulatedToolInput(
        input_text=query,
        target_tool=target_tool,
        expected_params=expected_params,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_coverage_report(result: ToolCoverageResult) -> str:
    """Human-readable coverage report."""
    lines: List[str] = []
    lines.append("Tool Coverage Report")
    lines.append("=" * 40)
    lines.append(f"Total tools: {result.total_tools}")
    lines.append(f"Exercised: {result.exercised_tools}")
    lines.append(f"Coverage rate: {result.coverage_rate:.2%}")

    if result.uncovered_tools:
        lines.append("")
        lines.append("Uncovered tools:")
        for t in result.uncovered_tools:
            lines.append(f"  - {t}")

    lines.append("")
    lines.append("Per-tool counts:")
    for name, count in sorted(result.per_tool_counts.items()):
        lines.append(f"  {name}: {count}")

    return "\n".join(lines)


def suggest_missing_queries(
    result: ToolCoverageResult, schemas: List[ToolSchema]
) -> List[SimulatedToolInput]:
    """Generate queries for uncovered tools."""
    schema_map = {s.tool_name: s for s in schemas}
    suggestions: List[SimulatedToolInput] = []
    for tool_name in result.uncovered_tools:
        schema = schema_map.get(tool_name)
        if schema is not None:
            rng = random.Random(hash(tool_name))
            suggestions.append(generate_tool_query(schema, rng))
    return suggestions
