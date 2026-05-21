"""CLI quick rerun: rerun last command with modified flags."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DESTRUCTIVE_COMMANDS = frozenset({"gc", "delete", "reset"})


@dataclass
class RerunConfig:
    """Configuration for the quick-rerun feature."""

    history_path: str = ".evalyn/history.jsonl"
    allow_destructive: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "history_path": self.history_path,
            "allow_destructive": self.allow_destructive,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RerunConfig:
        return cls(
            history_path=str(data.get("history_path", ".evalyn/history.jsonl")),
            allow_destructive=bool(data.get("allow_destructive", False)),
        )


@dataclass
class RerunRequest:
    """A fully resolved rerun request."""

    original_command: str
    original_args: list[str]
    override_args: list[str]
    merged_command: str

    def as_dict(self) -> dict[str, object]:
        return {
            "original_command": self.original_command,
            "original_args": self.original_args,
            "override_args": self.override_args,
            "merged_command": self.merged_command,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RerunRequest:
        return cls(
            original_command=str(data.get("original_command", "")),
            original_args=list(data.get("original_args", [])),  # type: ignore[arg-type]
            override_args=list(data.get("override_args", [])),  # type: ignore[arg-type]
            merged_command=str(data.get("merged_command", "")),
        )


def get_last_command(history_path: str) -> tuple[str, list[str]] | None:
    """Read the last entry from a JSONL history file.

    Returns (command, args) or None if the file is empty or missing.
    """
    path = Path(history_path)
    if not path.exists():
        return None

    last_line: str | None = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_line = stripped

    if last_line is None:
        return None

    entry = json.loads(last_line)
    command = entry.get("command", "")
    args = entry.get("args", [])
    return (command, args)


def _parse_flag_name(token: str) -> str | None:
    """Extract the flag name from a token like --flag=value or --flag."""
    if not token.startswith("-"):
        return None
    if "=" in token:
        return token.split("=", 1)[0]
    return token


def merge_args(original_args: list[str], overrides: list[str]) -> list[str]:
    """Merge original args with overrides.

    Override rules:
    - If override is --flag=value, replace matching --flag or --flag=old in original.
    - If override is --flag followed by a value, replace matching --flag value in original.
    - New flags are appended.
    """
    # Build a list of (flag_name, tokens) from overrides
    override_entries: list[tuple[str | None, list[str]]] = []
    i = 0
    while i < len(overrides):
        token = overrides[i]
        flag = _parse_flag_name(token)
        if flag is not None and "=" in token:
            # --flag=value form
            override_entries.append((flag, [token]))
            i += 1
        elif flag is not None:
            # --flag or --flag value form
            if i + 1 < len(overrides) and not overrides[i + 1].startswith("-"):
                override_entries.append((flag, [token, overrides[i + 1]]))
                i += 2
            else:
                # Boolean flag (no value)
                override_entries.append((flag, [token]))
                i += 1
        else:
            # Positional arg
            override_entries.append((None, [token]))
            i += 1

    # Track which override flags have been used (by index)
    used_overrides: set[int] = set()

    # Process original args, replacing where overrides match
    result: list[str] = []
    j = 0
    while j < len(original_args):
        token = original_args[j]
        orig_flag = _parse_flag_name(token)

        if orig_flag is not None:
            # Check if any override matches this flag
            matched = False
            for idx, (oflag, otokens) in enumerate(override_entries):
                if oflag == orig_flag and idx not in used_overrides:
                    result.extend(otokens)
                    used_overrides.add(idx)
                    matched = True
                    break

            if matched:
                # Skip original tokens for this flag
                if "=" in token:
                    j += 1
                else:
                    j += 1
                    # Skip the value token too if present
                    if j < len(original_args) and not original_args[j].startswith("-"):
                        j += 1
            else:
                # Keep original
                result.append(token)
                j += 1
        else:
            result.append(token)
            j += 1

    # Append any overrides that were not used
    for idx, (_oflag, otokens) in enumerate(override_entries):
        if idx not in used_overrides:
            result.extend(otokens)

    return result


def build_rerun_request(history_path: str, overrides: list[str]) -> RerunRequest | None:
    """Build a rerun request from the last command and given overrides.

    Returns None if no previous command exists.
    """
    last = get_last_command(history_path)
    if last is None:
        return None

    command, original_args = last
    merged = merge_args(original_args, overrides)
    merged_str = " ".join(["evalyn", command] + merged)

    return RerunRequest(
        original_command=command,
        original_args=original_args,
        override_args=overrides,
        merged_command=merged_str,
    )


def format_rerun_preview(request: RerunRequest) -> str:
    """Show what will be executed."""
    return f"Rerunning: {request.merged_command}"


def is_destructive_command(command: str) -> bool:
    """Return True for destructive commands like gc, delete, reset."""
    return command.lower() in _DESTRUCTIVE_COMMANDS
