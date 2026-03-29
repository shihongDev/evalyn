"""Machine-readable JSON output for scripting and CI pipeline integration.

Pure Python, no external dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


@dataclass
class JSONOutputConfig:
    """Configuration for JSON output formatting."""

    pretty: bool = False
    streaming: bool = False
    include_metadata: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pretty": self.pretty,
            "streaming": self.streaming,
            "include_metadata": self.include_metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JSONOutputConfig:
        return cls(
            pretty=data.get("pretty", False),
            streaming=data.get("streaming", False),
            include_metadata=data.get("include_metadata", True),
        )


@dataclass
class CommandResult:
    """Result of a CLI command in machine-readable format."""

    command: str
    status: str  # "success", "failure", "error"
    exit_code: int
    data: Any = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "command": self.command,
            "status": self.status,
            "exit_code": self.exit_code,
            "data": self.data,
        }
        if self.errors:
            d["errors"] = list(self.errors)
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CommandResult:
        return cls(
            command=data.get("command", ""),
            status=data.get("status", "error"),
            exit_code=data.get("exit_code", 2),
            data=data.get("data"),
            errors=data.get("errors", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProgressEvent:
    """A single progress event for streaming output."""

    event_type: str  # "start", "progress", "complete", "error"
    progress: float  # 0.0 to 1.0
    message: str
    timestamp: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "progress": self.progress,
            "message": self.message,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProgressEvent:
        return cls(
            event_type=data.get("event_type", ""),
            progress=data.get("progress", 0.0),
            message=data.get("message", ""),
            timestamp=data.get("timestamp", ""),
        )


def format_json_output(
    result: CommandResult, config: Optional[JSONOutputConfig] = None
) -> str:
    """Format CommandResult as JSON string."""
    if config is None:
        config = JSONOutputConfig()
    d = result.as_dict()
    if not config.include_metadata:
        d.pop("metadata", None)
    if config.pretty:
        return json.dumps(d, indent=2)
    return json.dumps(d, separators=(",", ":"))


def format_jsonl_stream(events: List[ProgressEvent]) -> str:
    """Format progress events as JSONL - one per line."""
    lines: List[str] = []
    for event in events:
        lines.append(json.dumps(event.as_dict(), separators=(",", ":")))
    return "\n".join(lines)


def format_progress_event(event: ProgressEvent) -> str:
    """Format a single progress event as JSON line."""
    return json.dumps(event.as_dict(), separators=(",", ":"))


def create_success_result(
    command: str, data: Any, metadata: Optional[Dict[str, Any]] = None
) -> CommandResult:
    """Create a success result with exit_code=0."""
    return CommandResult(
        command=command,
        status="success",
        exit_code=0,
        data=data,
        metadata=metadata or {},
    )


def create_failure_result(
    command: str, errors: List[str], data: Any = None
) -> CommandResult:
    """Create a failure result with exit_code=1."""
    return CommandResult(
        command=command,
        status="failure",
        exit_code=1,
        data=data,
        errors=errors,
    )


def create_error_result(command: str, error: str) -> CommandResult:
    """Create an error result with exit_code=2."""
    return CommandResult(
        command=command,
        status="error",
        exit_code=2,
        errors=[error],
    )


def determine_exit_code(result: CommandResult) -> int:
    """Determine exit code: 0=success, 1=failure, 2=error."""
    return result.exit_code


def wrap_command_output(
    command: str, fn: Callable[[], Any], json_mode: bool = False
) -> tuple:
    """Execute fn, catch exceptions, return (output_string, exit_code).

    If json_mode, format as JSON; otherwise return str(result).
    """
    try:
        data = fn()
        result = create_success_result(command, data)
    except Exception as exc:
        result = create_error_result(command, str(exc))

    if json_mode:
        output = format_json_output(result)
    else:
        if result.status == "success":
            output = str(result.data) if result.data is not None else ""
        else:
            output = "; ".join(result.errors)

    return (output, result.exit_code)
