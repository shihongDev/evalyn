"""Batch script execution: run multiple commands from a script file.

Parse script files with one command per line, substitute variables,
execute via a callback, and produce a summary report.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class BatchConfig:
    """Configuration for batch script execution."""

    stop_on_error: bool = True
    dry_run: bool = False
    variables: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stop_on_error": self.stop_on_error,
            "dry_run": self.dry_run,
            "variables": dict(self.variables),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchConfig:
        return cls(
            stop_on_error=data.get("stop_on_error", True),
            dry_run=data.get("dry_run", False),
            variables=dict(data.get("variables", {})),
        )


@dataclass
class CommandResult:
    """Result of executing a single command."""

    command: str
    exit_code: int = 0
    output: str = ""
    duration_seconds: float = 0.0
    skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "output": self.output,
            "duration_seconds": self.duration_seconds,
            "skipped": self.skipped,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommandResult:
        return cls(
            command=data["command"],
            exit_code=data.get("exit_code", 0),
            output=data.get("output", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
            skipped=data.get("skipped", False),
        )


@dataclass
class BatchResult:
    """Aggregate result of a batch execution."""

    results: list[CommandResult] = field(default_factory=list)
    total_commands: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "total_commands": self.total_commands,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchResult:
        return cls(
            results=[CommandResult.from_dict(r) for r in data.get("results", [])],
            total_commands=data.get("total_commands", 0),
            succeeded=data.get("succeeded", 0),
            failed=data.get("failed", 0),
            skipped=data.get("skipped", 0),
        )


def parse_script(content: str) -> list[str]:
    """Parse script content into a list of commands.

    One command per line. Blank lines and lines starting with # are skipped.
    Leading/trailing whitespace is stripped from each line.
    """
    commands: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        commands.append(stripped)
    return commands


def substitute_variables(command: str, variables: dict[str, str]) -> str:
    """Replace $VAR and ${VAR} patterns with values from variables dict.

    Built-in variables (always available, overridable):
      $DATE - current date as YYYY-MM-DD
      $TIMESTAMP - current UTC time in ISO format
    """
    now = datetime.now(timezone.utc)
    builtins = {
        "DATE": now.strftime("%Y-%m-%d"),
        "TIMESTAMP": now.isoformat(),
    }
    merged = {**builtins, **variables}

    def _replace(match: re.Match) -> str:
        # group(1) is from ${VAR}, group(2) is from $VAR
        var_name = match.group(1) or match.group(2)
        return merged.get(var_name, match.group(0))

    # Match ${VAR} first, then $VAR (word chars only)
    return re.sub(r"\$\{(\w+)\}|\$(\w+)", _replace, command)


def execute_batch(
    commands: list[str],
    executor: Callable[[str], tuple[int, str]],
    config: BatchConfig | None = None,
) -> BatchResult:
    """Execute a list of commands via the given executor callback.

    The executor receives a command string and returns (exit_code, output).
    If config.stop_on_error is True and a command fails (exit_code != 0),
    remaining commands are skipped.
    If config.dry_run is True, commands are not executed.
    Variables from config.variables are substituted before execution.
    """
    if config is None:
        config = BatchConfig()

    results: list[CommandResult] = []
    succeeded = 0
    failed = 0
    skipped = 0
    should_skip = False

    for cmd in commands:
        resolved = substitute_variables(cmd, config.variables)

        if should_skip:
            results.append(CommandResult(command=resolved, skipped=True))
            skipped += 1
            continue

        if config.dry_run:
            results.append(CommandResult(command=resolved, skipped=True))
            skipped += 1
            continue

        start = time.monotonic()
        exit_code, output = executor(resolved)
        elapsed = time.monotonic() - start

        result = CommandResult(
            command=resolved,
            exit_code=exit_code,
            output=output,
            duration_seconds=round(elapsed, 4),
        )
        results.append(result)

        if exit_code == 0:
            succeeded += 1
        else:
            failed += 1
            if config.stop_on_error:
                should_skip = True

    return BatchResult(
        results=results,
        total_commands=len(commands),
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
    )


def load_script_file(path: str) -> list[str]:
    """Read a script file and parse it into commands."""
    with open(path, encoding="utf-8") as f:
        return parse_script(f.read())


def format_batch_report(result: BatchResult) -> str:
    """Format a BatchResult as a human-readable report."""
    lines: list[str] = []
    lines.append("Batch Execution Report")
    lines.append("=" * 40)
    lines.append("")

    for i, r in enumerate(result.results, 1):
        if r.skipped:
            status = "SKIP"
        elif r.exit_code == 0:
            status = "OK"
        else:
            status = f"FAIL (exit {r.exit_code})"
        lines.append(f"  [{i}] {status}: {r.command}")
        if r.output:
            for out_line in r.output.splitlines():
                lines.append(f"       {out_line}")
        if r.duration_seconds > 0:
            lines.append(f"       ({r.duration_seconds:.3f}s)")

    lines.append("")
    lines.append(
        f"Total: {result.total_commands}  "
        f"Succeeded: {result.succeeded}  "
        f"Failed: {result.failed}  "
        f"Skipped: {result.skipped}"
    )
    return "\n".join(lines)
