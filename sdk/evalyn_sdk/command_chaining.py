"""CLI command chaining: pipe output of one command as input to another.

Pure Python, no external dependencies. Provides chain parsing, sequential
execution with an executor callback, and result formatting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChainStep:
    """A single step in a command chain.

    Attributes:
        command: The command name.
        args: Arguments for the command.
        input_data: Input received from the previous step.
        output_data: Output produced by this step.
        exit_code: Exit code after execution (0 = success).
    """

    command: str
    args: list[str] = field(default_factory=list)
    input_data: str = ""
    output_data: str = ""
    exit_code: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "args": list(self.args),
            "input_data": self.input_data,
            "output_data": self.output_data,
            "exit_code": self.exit_code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChainStep:
        return cls(
            command=data.get("command", ""),
            args=list(data.get("args", [])),
            input_data=data.get("input_data", ""),
            output_data=data.get("output_data", ""),
            exit_code=data.get("exit_code", 0),
        )


@dataclass
class ChainConfig:
    """Configuration for command chain execution.

    Attributes:
        stop_on_error: Stop the chain if a step returns a non-zero exit code.
        pass_output_as: How to pass output between steps: "stdin", "arg", or "file".
    """

    stop_on_error: bool = True
    pass_output_as: str = "stdin"

    def as_dict(self) -> dict[str, Any]:
        return {
            "stop_on_error": self.stop_on_error,
            "pass_output_as": self.pass_output_as,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChainConfig:
        return cls(
            stop_on_error=data.get("stop_on_error", True),
            pass_output_as=data.get("pass_output_as", "stdin"),
        )


@dataclass
class ChainResult:
    """Result of executing a command chain.

    Attributes:
        steps: All executed steps with their results.
        final_output: Output from the last executed step.
        total_steps: Total number of steps in the chain.
        succeeded: Number of steps that succeeded (exit_code == 0).
        failed: Number of steps that failed (exit_code != 0).
    """

    steps: list[ChainStep] = field(default_factory=list)
    final_output: str = ""
    total_steps: int = 0
    succeeded: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.as_dict() for s in self.steps],
            "final_output": self.final_output,
            "total_steps": self.total_steps,
            "succeeded": self.succeeded,
            "failed": self.failed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChainResult:
        return cls(
            steps=[ChainStep.from_dict(s) for s in data.get("steps", [])],
            final_output=data.get("final_output", ""),
            total_steps=data.get("total_steps", 0),
            succeeded=data.get("succeeded", 0),
            failed=data.get("failed", 0),
        )


def parse_chain(command_string: str, separator: str = "|") -> list[ChainStep]:
    """Parse a piped command string into a list of ChainStep objects.

    Example: "cmd1 arg1 | cmd2 | cmd3 -v" produces three steps.
    """
    parts = command_string.split(separator)
    steps: list[ChainStep] = []
    for part in parts:
        tokens = part.strip().split()
        if not tokens:
            continue
        command = tokens[0]
        args = tokens[1:]
        steps.append(ChainStep(command=command, args=args))
    return steps


def execute_chain(
    steps: list[ChainStep],
    executor: Callable[[str, list[str], str], tuple[int, str]],
    config: ChainConfig | None = None,
) -> ChainResult:
    """Execute a chain of steps sequentially using the provided executor.

    The executor callback receives (command, args, input_data) and returns
    (exit_code, output_data). Output from each step is passed as input
    to the next step.

    Args:
        steps: List of ChainStep to execute.
        executor: Callback that runs a single command.
        config: Chain configuration (defaults to ChainConfig()).

    Returns:
        ChainResult with per-step results and summary counts.
    """
    if config is None:
        config = ChainConfig()

    result = ChainResult(total_steps=len(steps))
    previous_output = ""

    for step in steps:
        step.input_data = previous_output
        exit_code, output = executor(step.command, step.args, step.input_data)
        step.exit_code = exit_code
        step.output_data = output
        result.steps.append(step)

        if exit_code == 0:
            result.succeeded += 1
            previous_output = output
        else:
            result.failed += 1
            previous_output = output
            if config.stop_on_error:
                break

    # Final output is the output of the last executed step
    if result.steps:
        result.final_output = result.steps[-1].output_data

    return result


def format_chain_plan(steps: list[ChainStep]) -> str:
    """Preview what will execute as a readable plan.

    Format: "Step 1: cmd1 -> Step 2: cmd2 -> ..."
    """
    if not steps:
        return ""
    parts: list[str] = []
    for i, step in enumerate(steps, 1):
        label = f"Step {i}: {step.command}"
        if step.args:
            label += " " + " ".join(step.args)
        parts.append(label)
    return " -> ".join(parts)


def format_chain_result(result: ChainResult) -> str:
    """Format a chain result as a per-step report.

    Each step shows its index, command, exit code, and a snippet of output.
    Ends with a summary line.
    """
    if not result.steps:
        return "No steps executed."
    lines: list[str] = []
    for i, step in enumerate(result.steps, 1):
        status = "OK" if step.exit_code == 0 else f"FAILED (exit {step.exit_code})"
        cmd = step.command
        if step.args:
            cmd += " " + " ".join(step.args)
        output_preview = step.output_data.strip()
        if len(output_preview) > 80:
            output_preview = output_preview[:77] + "..."
        lines.append(f"Step {i}: {cmd} - {status}")
        if output_preview:
            lines.append(f"  Output: {output_preview}")
    lines.append("")
    lines.append(
        f"Summary: {result.succeeded}/{result.total_steps} succeeded, "
        f"{result.failed} failed"
    )
    return "\n".join(lines)
