"""
Dataset changelog: automatic log of all build-dataset operations and parameters.

Tracks build, filter, merge, and other dataset transformations with
timestamps and parameter snapshots for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ChangelogEntry:
    """Single recorded operation on a dataset."""

    operation: str
    timestamp: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    items_affected: int = 0
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "timestamp": self.timestamp,
            "parameters": self.parameters,
            "items_affected": self.items_affected,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChangelogEntry:
        return cls(
            operation=data["operation"],
            timestamp=data["timestamp"],
            parameters=data.get("parameters", {}),
            items_affected=data.get("items_affected", 0),
            description=data.get("description", ""),
        )


@dataclass
class DatasetChangelog:
    """Ordered log of dataset operations."""

    entries: List[ChangelogEntry] = field(default_factory=list)
    dataset_name: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "entries": [e.as_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetChangelog:
        return cls(
            dataset_name=data.get("dataset_name", ""),
            entries=[ChangelogEntry.from_dict(e) for e in data.get("entries", [])],
        )

    def add_entry(
        self,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        items_affected: int = 0,
        description: str = "",
    ) -> None:
        """Add entry with auto-generated UTC timestamp."""
        entry = ChangelogEntry(
            operation=operation,
            timestamp=datetime.now(timezone.utc).isoformat(),
            parameters=parameters or {},
            items_affected=items_affected,
            description=description,
        )
        self.entries.append(entry)

    def get_entries_by_operation(self, operation: str) -> List[ChangelogEntry]:
        """Filter entries by operation name."""
        return [e for e in self.entries if e.operation == operation]

    @property
    def count(self) -> int:
        return len(self.entries)

    @property
    def latest(self) -> Optional[ChangelogEntry]:
        """Most recent entry, or None if empty."""
        if not self.entries:
            return None
        return self.entries[-1]

    def format_text(self) -> str:
        """Human-readable multi-line summary."""
        lines: List[str] = []
        lines.append(f"Dataset Changelog: {self.dataset_name or '(unnamed)'}")
        lines.append(f"Total entries: {self.count}")
        lines.append("")
        for i, entry in enumerate(self.entries, 1):
            lines.append(f"  {i}. [{entry.timestamp}] {entry.operation}")
            if entry.description:
                lines.append(f"     {entry.description}")
            if entry.items_affected:
                lines.append(f"     Items affected: {entry.items_affected}")
            if entry.parameters:
                lines.append(f"     Parameters: {entry.parameters}")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.as_dict())

    @classmethod
    def from_json(cls, data: str) -> DatasetChangelog:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(data))


# ---------------------------------------------------------------------------
# Factory / helper functions
# ---------------------------------------------------------------------------


def create_changelog(dataset_name: str = "") -> DatasetChangelog:
    """Create a new empty changelog."""
    return DatasetChangelog(dataset_name=dataset_name)


def record_build(
    changelog: DatasetChangelog,
    source: str,
    item_count: int,
    params: Optional[Dict[str, Any]] = None,
) -> DatasetChangelog:
    """Record a build operation."""
    changelog.add_entry(
        operation="build",
        parameters={"source": source, **(params or {})},
        items_affected=item_count,
        description=f"Built from {source}",
    )
    return changelog


def record_filter(
    changelog: DatasetChangelog,
    filter_desc: str,
    before_count: int,
    after_count: int,
) -> DatasetChangelog:
    """Record a filtering operation."""
    changelog.add_entry(
        operation="filter",
        parameters={
            "filter": filter_desc,
            "before_count": before_count,
            "after_count": after_count,
        },
        items_affected=before_count - after_count,
        description=f"Filter: {filter_desc} ({before_count} -> {after_count})",
    )
    return changelog


def record_merge(
    changelog: DatasetChangelog,
    sources: List[str],
    result_count: int,
) -> DatasetChangelog:
    """Record a merge operation."""
    changelog.add_entry(
        operation="merge",
        parameters={"sources": sources},
        items_affected=result_count,
        description=f"Merged {len(sources)} sources into {result_count} items",
    )
    return changelog


def render_changelog(changelog: DatasetChangelog, max_entries: int = 20) -> str:
    """Formatted text of the most recent entries."""
    lines: List[str] = []
    lines.append(f"Changelog: {changelog.dataset_name or '(unnamed)'}")
    recent = changelog.entries[-max_entries:] if changelog.entries else []
    for entry in recent:
        lines.append(f"  [{entry.timestamp}] {entry.operation}: {entry.description}")
    if not recent:
        lines.append("  (no entries)")
    return "\n".join(lines)
