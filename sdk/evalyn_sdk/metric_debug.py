"""Metric debug mode: verbose logging of judge interactions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class DebugEntry:
    """Single debug log entry for a judge interaction."""

    item_id: str
    metric_id: str
    input_text: str
    output_text: str
    prompt_sent: str
    judge_response: str = ""
    score: float = 0.0
    reasoning: str = ""
    timestamp: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "metric_id": self.metric_id,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "prompt_sent": self.prompt_sent,
            "judge_response": self.judge_response,
            "score": self.score,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DebugEntry:
        return cls(
            item_id=data.get("item_id", ""),
            metric_id=data.get("metric_id", ""),
            input_text=data.get("input_text", ""),
            output_text=data.get("output_text", ""),
            prompt_sent=data.get("prompt_sent", ""),
            judge_response=data.get("judge_response", ""),
            score=float(data.get("score", 0.0)),
            reasoning=data.get("reasoning", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class DebugConfig:
    """Configuration for debug logging."""

    enabled: bool = False
    log_path: str = ".evalyn/debug.jsonl"
    include_prompts: bool = True
    include_responses: bool = True
    max_text_length: int = 500

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "log_path": self.log_path,
            "include_prompts": self.include_prompts,
            "include_responses": self.include_responses,
            "max_text_length": self.max_text_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DebugConfig:
        return cls(
            enabled=data.get("enabled", False),
            log_path=data.get("log_path", ".evalyn/debug.jsonl"),
            include_prompts=data.get("include_prompts", True),
            include_responses=data.get("include_responses", True),
            max_text_length=int(data.get("max_text_length", 500)),
        )


@dataclass
class DebugLog:
    """Collection of debug entries with configuration."""

    entries: list[DebugEntry] = field(default_factory=list)
    config: DebugConfig = field(default_factory=DebugConfig)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.as_dict() for e in self.entries],
            "config": self.config.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DebugLog:
        return cls(
            entries=[
                DebugEntry.from_dict(e) for e in data.get("entries", [])
            ],
            config=DebugConfig.from_dict(data.get("config", {})),
        )


# ---------------------------------------------------------------------------
# Pure Functions
# ---------------------------------------------------------------------------


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text with marker if it exceeds max_length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"


def create_debug_entry(
    item_id: str,
    metric_id: str,
    input_text: str,
    output_text: str,
    prompt: str,
    config: DebugConfig | None = None,
) -> DebugEntry:
    """Factory for DebugEntry with auto-timestamp and text truncation."""
    cfg = config or DebugConfig()
    max_len = cfg.max_text_length
    ts = datetime.now(timezone.utc).isoformat()
    return DebugEntry(
        item_id=item_id,
        metric_id=metric_id,
        input_text=truncate_text(input_text, max_len),
        output_text=truncate_text(output_text, max_len),
        prompt_sent=truncate_text(prompt, max_len) if cfg.include_prompts else "",
        timestamp=ts,
    )


def append_debug_entry(log_path: str, entry: DebugEntry) -> None:
    """Append a single debug entry as JSONL to the log file."""
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry.as_dict()) + "\n")


def read_debug_log(log_path: str, limit: int = 50) -> list[DebugEntry]:
    """Read the last N entries from a JSONL debug log."""
    if not os.path.exists(log_path):
        return []
    entries: list[DebugEntry] = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(DebugEntry.from_dict(json.loads(line)))
    # Return last N entries
    if limit > 0 and len(entries) > limit:
        entries = entries[-limit:]
    return entries


def filter_debug_entries(
    entries: list[DebugEntry],
    item_id: str | None = None,
    metric_id: str | None = None,
) -> list[DebugEntry]:
    """Filter debug entries by item_id and/or metric_id."""
    result = entries
    if item_id is not None:
        result = [e for e in result if e.item_id == item_id]
    if metric_id is not None:
        result = [e for e in result if e.metric_id == metric_id]
    return result


def format_debug_entry(entry: DebugEntry) -> str:
    """Format a single debug entry as a detailed human-readable view."""
    lines = [
        f"Item:      {entry.item_id}",
        f"Metric:    {entry.metric_id}",
        f"Score:     {entry.score}",
        f"Timestamp: {entry.timestamp}",
        f"Input:     {entry.input_text}",
        f"Output:    {entry.output_text}",
        f"Prompt:    {entry.prompt_sent}",
        f"Response:  {entry.judge_response}",
        f"Reasoning: {entry.reasoning}",
    ]
    return "\n".join(lines)


def format_debug_summary(entries: list[DebugEntry]) -> str:
    """Format entries as a summary table."""
    if not entries:
        return "No debug entries."
    header = f"{'Item':<20} {'Metric':<20} {'Score':<8} {'Reasoning'}"
    sep = "-" * 70
    lines = [header, sep]
    for e in entries:
        reasoning_preview = e.reasoning[:40] + "..." if len(e.reasoning) > 40 else e.reasoning
        lines.append(f"{e.item_id:<20} {e.metric_id:<20} {e.score:<8.2f} {reasoning_preview}")
    return "\n".join(lines)
