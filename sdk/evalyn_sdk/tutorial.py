"""
Interactive tutorial mode for learning evalyn.

Provides a step-by-step in-terminal tutorial that walks users through the
full evaluation cycle: tracing, dataset building, metric selection, running
evals, analysis, and comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TutorialStep:
    """A single step in an interactive tutorial."""

    step_id: str
    title: str
    explanation: str
    command_hint: str
    sample_output: str
    next_step: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "explanation": self.explanation,
            "command_hint": self.command_hint,
            "sample_output": self.sample_output,
            "next_step": self.next_step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TutorialStep:
        return cls(
            step_id=data.get("step_id", ""),
            title=data.get("title", ""),
            explanation=data.get("explanation", ""),
            command_hint=data.get("command_hint", ""),
            sample_output=data.get("sample_output", ""),
            next_step=data.get("next_step", ""),
        )


@dataclass
class Tutorial:
    """A named tutorial composed of ordered steps."""

    name: str
    steps: list[TutorialStep] = field(default_factory=list)
    current_step_index: int = 0
    completed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": [s.as_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
            "completed": self.completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tutorial:
        return cls(
            name=data.get("name", ""),
            steps=[TutorialStep.from_dict(s) for s in data.get("steps", [])],
            current_step_index=data.get("current_step_index", 0),
            completed=data.get("completed", False),
        )


# ---------------------------------------------------------------------------
# Built-in tutorial
# ---------------------------------------------------------------------------

_STEPS = [
    TutorialStep(
        step_id="welcome",
        title="Welcome to Evalyn",
        explanation=(
            "Evalyn is an evaluation SDK for LLM-powered agents. "
            "It lets you trace agent runs, build datasets from those traces, "
            "and score them with configurable metrics."
        ),
        command_hint="evalyn --help",
        sample_output=(
            "usage: evalyn [-h] {trace,dataset,eval,analyze,compare,...}\n"
            "Evalyn - LLM evaluation toolkit"
        ),
        next_step="tracing",
    ),
    TutorialStep(
        step_id="tracing",
        title="Instrument Your Agent",
        explanation=(
            "Tracing captures every LLM call, tool invocation, and decision "
            "your agent makes. Wrap your agent function with the evalyn "
            "tracing decorator to record structured traces automatically."
        ),
        command_hint="evalyn trace run my_agent.py --input 'hello'",
        sample_output=(
            "Trace saved: .evalyn/traces/trace_001.json\n"
            "Spans captured: 4 (1 agent, 2 llm_call, 1 tool_call)"
        ),
        next_step="build_dataset",
    ),
    TutorialStep(
        step_id="build_dataset",
        title="Build an Evaluation Dataset",
        explanation=(
            "Datasets are collections of input-output pairs that your agent "
            "will be evaluated against. You can build them from recorded "
            "traces, manual JSONL files, or synthetic generation."
        ),
        command_hint="evalyn dataset create --from-traces .evalyn/traces/ --output my_dataset/",
        sample_output=("Dataset created: my_dataset/\nItems: 12\nSource: 12 traces"),
        next_step="select_metrics",
    ),
    TutorialStep(
        step_id="select_metrics",
        title="Choose Evaluation Metrics",
        explanation=(
            "Metrics define how your agent is scored. Evalyn includes "
            "built-in metrics like correctness, helpfulness, and latency. "
            "You can also define custom rubric-based judges."
        ),
        command_hint="evalyn metric list",
        sample_output=(
            "Available metrics:\n"
            "  correctness   - Binary correctness against reference\n"
            "  helpfulness   - 1-5 helpfulness rating\n"
            "  latency       - Response time in milliseconds\n"
            "  cost          - API cost per item"
        ),
        next_step="run_eval",
    ),
    TutorialStep(
        step_id="run_eval",
        title="Run an Evaluation",
        explanation=(
            "Running an eval executes your agent on each dataset item and "
            "scores the outputs with your chosen metrics. Results are saved "
            "as a structured eval run for later analysis."
        ),
        command_hint="evalyn eval run my_dataset/ --metrics correctness helpfulness",
        sample_output=(
            "Eval run started: run_20260328_001\n"
            "Processing: 12/12 items [====] 100%\n"
            "Results saved: my_dataset/.evalyn/runs/run_20260328_001/"
        ),
        next_step="analyze",
    ),
    TutorialStep(
        step_id="analyze",
        title="Analyze Results",
        explanation=(
            "The analyze command produces summary statistics, highlights "
            "worst-performing items, and surfaces key findings from your "
            "eval run. Use it to understand overall quality and find issues."
        ),
        command_hint="evalyn analyze my_dataset/",
        sample_output=(
            "Run: run_20260328_001\n"
            "correctness: mean=0.83, std=0.12\n"
            "helpfulness: mean=4.1, std=0.7\n"
            "KEY FINDINGS: 2 items scored below threshold"
        ),
        next_step="compare",
    ),
    TutorialStep(
        step_id="compare",
        title="Compare Runs",
        explanation=(
            "After making changes to your agent, run another eval and "
            "compare the two runs side by side. Evalyn highlights "
            "regressions and improvements with statistical significance."
        ),
        command_hint="evalyn compare my_dataset/ --runs run_20260328_001 run_20260328_002",
        sample_output=(
            "Comparing: run_20260328_001 vs run_20260328_002\n"
            "correctness: 0.83 -> 0.91 (+0.08) *significant*\n"
            "helpfulness: 4.1 -> 4.3 (+0.2)\n"
            "REGRESSION ALERTS: none"
        ),
        next_step="next_steps",
    ),
    TutorialStep(
        step_id="next_steps",
        title="Next Steps",
        explanation=(
            "You have completed the core evalyn workflow. From here you "
            "can explore advanced features like custom judges, dataset "
            "augmentation, calibration, and CI/CD integration."
        ),
        command_hint="evalyn --help",
        sample_output=(
            "Suggested next steps:\n"
            "  evalyn judge create    - Build a custom rubric judge\n"
            "  evalyn dataset augment - Expand your dataset\n"
            "  evalyn calibrate       - Tune metric thresholds"
        ),
        next_step="",
    ),
]

EVALYN_TUTORIAL = Tutorial(name="evalyn_basics", steps=_STEPS)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def advance(tutorial: Tutorial) -> Tutorial:
    """Move to the next step. Returns an updated Tutorial."""
    if tutorial.current_step_index >= len(tutorial.steps) - 1:
        return Tutorial(
            name=tutorial.name,
            steps=tutorial.steps,
            current_step_index=tutorial.current_step_index,
            completed=True,
        )
    return Tutorial(
        name=tutorial.name,
        steps=tutorial.steps,
        current_step_index=tutorial.current_step_index + 1,
        completed=False,
    )


def go_to_step(tutorial: Tutorial, step_id: str) -> Tutorial:
    """Jump to a specific step by its step_id. Returns an updated Tutorial."""
    for i, step in enumerate(tutorial.steps):
        if step.step_id == step_id:
            return Tutorial(
                name=tutorial.name,
                steps=tutorial.steps,
                current_step_index=i,
                completed=False,
            )
    return tutorial


def get_current_step(tutorial: Tutorial) -> TutorialStep | None:
    """Return the current step, or None if the tutorial has no steps."""
    if not tutorial.steps:
        return None
    if tutorial.current_step_index < 0 or tutorial.current_step_index >= len(tutorial.steps):
        return None
    return tutorial.steps[tutorial.current_step_index]


def format_step(step: TutorialStep, step_num: int, total: int) -> str:
    """Format a single tutorial step for terminal display."""
    lines = [
        f"--- Step {step_num}/{total}: {step.title} ---",
        "",
        step.explanation,
        "",
        f"  Try: {step.command_hint}",
        "",
        "Sample output:",
    ]
    for line in step.sample_output.splitlines():
        lines.append(f"  {line}")
    return "\n".join(lines)


def format_progress(tutorial: Tutorial) -> str:
    """Render a progress bar like 'Step 3/8 - [====>    ]'."""
    total = len(tutorial.steps)
    if total == 0:
        return "Step 0/0 - [          ]"
    current = tutorial.current_step_index + 1
    bar_width = 10
    filled = int(bar_width * current / total)
    if filled < bar_width:
        bar = "=" * filled + ">" + " " * (bar_width - filled - 1)
    else:
        bar = "=" * bar_width
    return f"Step {current}/{total} - [{bar}]"


def list_steps(tutorial: Tutorial) -> list[str]:
    """Return a list of 'step_id: title' strings for all steps."""
    return [f"{s.step_id}: {s.title}" for s in tutorial.steps]


def reset_tutorial(tutorial: Tutorial) -> Tutorial:
    """Reset the tutorial back to the beginning."""
    return Tutorial(
        name=tutorial.name,
        steps=tutorial.steps,
        current_step_index=0,
        completed=False,
    )
