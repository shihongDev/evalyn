"""
Pipeline visualization: ASCII flowchart rendering for CLI pipeline steps.

Renders pipeline steps as horizontal or vertical ASCII flowcharts with
status indicators, suitable for terminal display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Status markers
# ---------------------------------------------------------------------------

_STATUS_MARKERS: Dict[str, str] = {
    "complete": "[OK]",
    "failed": "[!!]",
    "running": "[>>]",
    "skipped": "[--]",
    "pending": "[  ]",
}


def get_status_marker(status: str) -> str:
    """Return an ASCII status marker for the given status string."""
    return _STATUS_MARKERS.get(status, "[  ]")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PipelineStep:
    """A single step in a pipeline."""

    step_id: str
    name: str
    description: str = ""
    status: str = "pending"

    def as_dict(self) -> Dict[str, str]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> PipelineStep:
        return cls(
            step_id=data["step_id"],
            name=data["name"],
            description=data.get("description", ""),
            status=data.get("status", "pending"),
        )


@dataclass
class Pipeline:
    """An ordered sequence of pipeline steps."""

    name: str
    steps: List[PipelineStep] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "steps": [s.as_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Pipeline:
        return cls(
            name=data["name"],
            steps=[PipelineStep.from_dict(s) for s in data.get("steps", [])],
        )


# ---------------------------------------------------------------------------
# Box rendering
# ---------------------------------------------------------------------------


def render_box(text: str, width: int = 20) -> List[str]:
    """Render text inside an ASCII box.

    Returns a list of strings (lines) forming the box.
    The box content is center-aligned within the given width.
    The width specifies the inner content width; the total line
    width will be width + 4 (for "| " and " |" borders).
    """
    inner_width = max(width, len(text))
    border = "+" + "-" * (inner_width + 2) + "+"
    content = "| " + text.center(inner_width) + " |"
    return [border, content, border]


# ---------------------------------------------------------------------------
# Horizontal rendering
# ---------------------------------------------------------------------------


def render_horizontal(pipeline: Pipeline) -> str:
    """Render pipeline steps as a horizontal ASCII flowchart.

    Example output:
        [OK] [Step1] --> [>>] [Step2] --> [  ] [Step3]
    """
    if not pipeline.steps:
        return "(empty pipeline)"

    parts: List[str] = []
    for i, step in enumerate(pipeline.steps):
        marker = get_status_marker(step.status)
        segment = f"{marker} [{step.name}]"
        if i > 0:
            parts.append(" --> ")
        parts.append(segment)

    return "".join(parts)


# ---------------------------------------------------------------------------
# Vertical rendering
# ---------------------------------------------------------------------------


def render_vertical(pipeline: Pipeline) -> str:
    """Render pipeline steps as a vertical ASCII flowchart.

    Each step is shown as a box with a status marker, connected by
    vertical pipe connectors.
    """
    if not pipeline.steps:
        return "(empty pipeline)"

    lines: List[str] = []
    max_name_len = max(len(s.name) for s in pipeline.steps)
    box_width = max(max_name_len, 10)

    for i, step in enumerate(pipeline.steps):
        marker = get_status_marker(step.status)
        box_lines = render_box(step.name, box_width)

        # Add status marker on the content line
        lines.append(box_lines[0])
        lines.append(f"{box_lines[1]}  {marker}")
        lines.append(box_lines[2])

        # Add connector between steps (not after the last one)
        if i < len(pipeline.steps) - 1:
            # Center the pipe under the box
            center = (len(box_lines[0])) // 2
            lines.append(" " * center + "|")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def render_pipeline(pipeline: Pipeline, orientation: str = "vertical") -> str:
    """Render a pipeline flowchart in the given orientation.

    Args:
        pipeline: The pipeline to render.
        orientation: "vertical" or "horizontal".

    Returns:
        A string containing the ASCII flowchart.
    """
    if orientation == "horizontal":
        return render_horizontal(pipeline)
    return render_vertical(pipeline)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_evalyn_pipeline(steps: List[str]) -> Pipeline:
    """Create a standard evalyn pipeline from a list of step names.

    Each step gets a step_id derived from its position (e.g. "step_0").
    """
    pipeline_steps = [
        PipelineStep(step_id=f"step_{i}", name=name)
        for i, name in enumerate(steps)
    ]
    return Pipeline(name="evalyn", steps=pipeline_steps)
