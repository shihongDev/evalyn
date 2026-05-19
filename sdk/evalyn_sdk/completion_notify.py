"""
CLI notification on completion: formatting and configuration for
system notifications when long-running commands finish.

Pure Python, no external dependencies. Provides the notification
framework and message formatting - does not actually send notifications.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NotifyConfig:
    """Configuration for CLI completion notifications."""

    enabled: bool = True
    min_duration_seconds: float = 30.0
    sound: bool = False
    method: str = "auto"  # "auto", "terminal_bell", "osascript", "notify_send"

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_duration_seconds": self.min_duration_seconds,
            "sound": self.sound,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NotifyConfig:
        return cls(
            enabled=data.get("enabled", True),
            min_duration_seconds=data.get("min_duration_seconds", 30.0),
            sound=data.get("sound", False),
            method=data.get("method", "auto"),
        )


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------


@dataclass
class NotifyEvent:
    """Record of a completed command notification."""

    command: str
    duration_seconds: float
    exit_code: int
    message: str
    timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
            "message": self.message,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NotifyEvent:
        return cls(
            command=data["command"],
            duration_seconds=data["duration_seconds"],
            exit_code=data["exit_code"],
            message=data["message"],
            timestamp=data["timestamp"],
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def should_notify(duration: float, config: NotifyConfig) -> bool:
    """Return True if a notification should be sent based on duration and config."""
    if not config.enabled:
        return False
    return duration >= config.min_duration_seconds


def build_notification_message(command: str, duration: float, exit_code: int) -> str:
    """Build a human-readable notification message.

    Format: "evalyn <command> completed in <duration>s (success/failed)"
    """
    status = "success" if exit_code == 0 else "failed"
    return f"evalyn {command} completed in {duration:.1f}s ({status})"


def detect_notification_method() -> str:
    """Detect the best notification method for the current platform.

    Checks for osascript (macOS), notify-send (Linux), falls back to
    terminal_bell.
    """
    if shutil.which("osascript"):
        return "osascript"
    if shutil.which("notify-send"):
        return "notify_send"
    return "terminal_bell"


def format_terminal_bell() -> str:
    """Return the terminal bell character."""
    return "\a"


def format_osascript_command(message: str, title: str = "evalyn") -> list[str]:
    """Build an osascript command list for macOS notifications."""
    script = f'display notification "{message}" with title "{title}"'
    return ["osascript", "-e", script]


def format_notify_send_command(message: str, title: str = "evalyn") -> list[str]:
    """Build a notify-send command list for Linux notifications."""
    return ["notify-send", title, message]


def create_notify_event(
    command: str, duration: float, exit_code: int
) -> NotifyEvent:
    """Factory: create a NotifyEvent with auto-generated timestamp and message."""
    message = build_notification_message(command, duration, exit_code)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return NotifyEvent(
        command=command,
        duration_seconds=duration,
        exit_code=exit_code,
        message=message,
        timestamp=timestamp,
    )


def format_notify_summary(event: NotifyEvent) -> str:
    """Format a human-readable summary of a notification event."""
    status = "success" if event.exit_code == 0 else "failed"
    lines = [
        f"Command: {event.command}",
        f"Status: {status}",
        f"Duration: {event.duration_seconds:.1f}s",
        f"Time: {event.timestamp}",
    ]
    return "\n".join(lines)
